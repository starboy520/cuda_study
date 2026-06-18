# 02 Warp 调度：scheduler、scoreboard 与延迟隐藏

## 0. 先建立大局观：GPU 怎么"同时"跑那么多线程

一个 SM 物理上只有几十个执行单元，却能"同时"驻留上千个线程。秘密在于 **warp scheduler**——
它不是真让所有线程同时跑，而是**飞快地在大量 warp 之间切换**，让执行单元永不空闲。本章讲清
这个调度机制，它是"为什么 GPU 需要大量线程"（卷一）的硬件答案。

```text
关键洞察：
  GPU 不靠"减少单个操作的延迟"取胜，而靠"用大量 warp 填满延迟"取胜
  某个 warp 等内存（几百拍）时，scheduler 切到其他就绪 warp 执行
  -> 执行单元一直在干活，延迟被"藏"起来了
```

## 0.1 术语速查表

| 术语 | 一句话定义 |
|---|---|
| **warp scheduler** | 每拍决定发射哪个 warp 指令的硬件 |
| **issue（发射）** | 把一条指令送进执行单元 |
| **scoreboard** | 追踪每个操作数是否就绪的记分牌 |
| **stall（停顿）** | warp 因等待无法发射指令 |
| **latency（延迟）** | 一个操作从发起到完成的时间 |
| **throughput（吞吐）** | 单位时间完成的操作数 |
| **延迟隐藏** | 用其他 warp 的计算填满等待的空隙 |

## 1. Warp 是调度的基本单位

回忆卷一：32 个线程组成一个 warp，它们**以 SIMT 方式执行同一条指令**。硬件层面，scheduler
不是调度单个线程，而是**以 warp 为单位**发射指令：

```text
一拍（cycle），warp scheduler：
  从"就绪的 warp"中挑一个 -> 发射它的下一条指令给执行单元 -> 32 条 lane 一起执行
```

这就是 warp=32 的硬件根源：scheduler 一次发射的指令，正好驱动 32 条 lane 并行。

一个 SM 通常有 **4 个 warp scheduler**（对应 4 个 sub-partition，卷九/01），每拍能发射多条
指令，并行推进多个 warp。

## 2. 为什么需要"大量 warp"：延迟隐藏

这是整个 GPU 设计哲学的核心（卷一讲过直觉，这里讲机制）。一个操作有**延迟**：

```text
寄存器运算：    ~几拍
shared 访问：   ~几十拍
global 访问：   ~几百拍（最致命）
```

如果一个 warp 发起 global load，要等几百拍才能拿到数据。**这期间如果只有这一个 warp，执行
单元就空转几百拍**——巨大浪费。

GPU 的解法：**让 SM 驻留很多 warp**。当 warp A 等内存时，scheduler 立刻切到就绪的 warp B、C、
D... 执行它们的指令：

```text
只有 1 个 warp：
  [warp A 发 load] [等待 400 拍，执行单元空转] [拿到数据继续]
                    ↑ 浪费

有很多 warp：
  [warp A 发 load][warp B 算][warp C 算][warp D 算]...[warp A 数据到了，继续]
                  ↑ A 等待时，BCD 在干活，执行单元不空转 -> 延迟被藏住
```

> 这就是"为什么 GPU 要大量线程"（卷一）和"为什么要足够 occupancy"（卷五/03）的硬件答案：
> **要有足够多的就绪 warp，才能填满内存延迟的空隙**。occupancy 不够 → warp 不够 → 延迟藏不住
> → 执行单元空转 → 慢。

## 3. Scoreboard：怎么知道一个 warp 能不能发射

scheduler 怎么知道哪个 warp"就绪"（能发射下一条指令）？靠 **scoreboard（记分牌）**——它追踪
每条指令的操作数是否已经准备好：

```text
warp 发了一条 global load -> 目标寄存器标记为"未就绪"（数据还在路上）
后续指令如果要用这个寄存器 -> scoreboard 发现它未就绪 -> 这个 warp 不能发射（stall）
数据到达 -> 寄存器标记"就绪" -> 这个 warp 重新可发射
```

所以每一拍，scheduler 看 scoreboard，从"所有操作数都就绪"的 warp 里挑一个发射。这是硬件
自动做的依赖追踪。

## 4. Stall：warp 为什么发不出指令

一个 warp 无法发射指令叫 **stall**。常见原因（卷五/03 的 stall reason 的硬件来源）：

```text
Long Scoreboard：等 global memory 返回（最常见，对应访存延迟）
Barrier：        卡在 __syncthreads() 等其他 warp
Short Scoreboard：等 shared memory / 较快的依赖
MIO Throttle：   访存/特殊单元的队列满了
Not Selected：   有就绪 warp 但这拍没轮到它（其实是好事，说明并发足够）
```

Nsight Compute 的 Warp State（卷五/05）显示的就是这些 stall 占比。理解它们的硬件来源，才能
对症下药：

```text
Long Scoreboard 高 -> 访存延迟没藏住 -> 提高并发(更多warp)/减少依赖/用shared复用
Barrier 高         -> 同步太多/负载不均 -> 减少 __syncthreads、均衡负载
Not Selected 高    -> 并发充足，通常别动
```

## 5. Latency vs Throughput：GPU 的设计取舍

理解 GPU 必须分清这两个词（卷一也讲过，这里到硬件层）：

```text
Latency（延迟）：  单个操作从发起到完成的时间 —— GPU 的单操作延迟其实不低
Throughput（吞吐）：单位时间完成的操作总数   —— GPU 靠海量并行做到极高吞吐

GPU 的哲学：不追求低延迟（单个操作可能慢），追求高吞吐（同时做超多操作）
          用大量 warp 隐藏延迟，把"慢但多"变成"总体极快"
```

对比 CPU：

```text
CPU：少量强核，大 cache，复杂乱序 -> 优化单任务延迟（让一件事尽快做完）
GPU：海量简单核，大量 warp 隐藏延迟 -> 优化总吞吐（同时做海量事）
```

## 6. 这对写代码的启示

把调度机制对应到优化实践：

```text
1. 要有足够 warp（occupancy）：否则延迟藏不住（卷五/03）
   但够用即可，不是越高越好（带宽可能才是墙）
2. 减少长延迟依赖链：连续依赖的指令无法并行，让 scheduler 有别的 warp 可切
3. 用 shared 复用减少 global 访问：global 延迟最长，最难藏
4. 提高 ILP（指令级并行）：一个 warp 内有独立指令，也能填空隙
   -> 有时低 occupancy + 高 ILP 也能跑满（卷五/03 的 Volkov 反例）
```

## 7. 实践

1. 用 `cudaGetDeviceProperties` 查每 SM 的最大驻留 warp 数（maxThreadsPerMultiProcessor/32）。
2. 写一个 occupancy 很低的 kernel（如每线程用超多寄存器），用 ncu 看 Long Scoreboard stall
   高、性能差，体会延迟没藏住。
3. 把它改成正常 occupancy，对比 stall 和性能变化。
4. 解释：为什么连续依赖的长指令链（每条都用上一条结果）对 GPU 不友好？

## 8. 面试题（附参考答案）

**Q1：GPU 怎么"同时"跑上千线程？**
不是真同时，而是 warp scheduler 飞快在大量 warp 间切换。某 warp 等内存时切到其他就绪 warp
执行，让执行单元不空转——这叫延迟隐藏。

**Q2：为什么 GPU 需要大量线程/warp？**
内存访问延迟长（global 几百拍），单个 warp 等待时执行单元会空转。大量 warp 让 scheduler 总有
就绪 warp 可切，用别的 warp 的计算填满等待空隙，藏住延迟。

**Q3：scoreboard 是干什么的？**
追踪每条指令的操作数是否就绪。global load 的目标寄存器在数据到达前标"未就绪"，依赖它的指令
就不能发射（stall）。scheduler 据此从操作数全就绪的 warp 里挑一个发射。

**Q4：latency 和 throughput 的区别？GPU 优化哪个？**
latency 是单操作完成时间，throughput 是单位时间完成总数。GPU 单操作延迟其实不低，但靠海量
并行 + 延迟隐藏做到极高吞吐。GPU 优化吞吐，不优化单操作延迟。

**Q5：occupancy 低为什么可能慢？**
occupancy 低 = 驻留 warp 少 = scheduler 可切换的就绪 warp 少 = 内存延迟藏不住 = 执行单元空转。
但够用即可，已达带宽墙或 ILP 充足时低 occupancy 也能快。

## 9. 资料映射

- NVIDIA 架构白皮书（SM 微架构、warp scheduler 部分）。
- CUDA Programming Guide：Hardware Implementation、Multiprocessor Level。
- 配套：[卷一 延迟隐藏](../volume01_gpu_basics/04_延迟隐藏与大量线程.md)、[卷五第 03 章 Occupancy 与延迟隐藏](../volume05_performance/03_Occupancy_分歧与延迟隐藏.md)。
