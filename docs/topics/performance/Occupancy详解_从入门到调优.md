# Occupancy 详解:从入门到调优

> 一份完整的 Occupancy 文档,从零基础讲到性能调优。分两部分:
> - **基础篇**:occupancy 是什么、为什么需要它(延迟隐藏)、定义、限制因素 —— 第一次接触看这里。
> - **进阶篇**:精确计算、系统化调优、与其他指标配合、profiler 实战 —— 概念熟悉后深入看这里。
>
> 硬件基准:Tesla T4(CC 7.5, sm_75, 40 SM)。其他卡数字不同,但方法通用。
> 配套实验:`week01_basics/mat_mul_naive/mat_mul.cu`。

---

## 目录

**基础篇**
1. [一句话是什么](#1-一句话是什么)
2. [为什么需要它:延迟隐藏(核心动机)](#2-为什么需要它延迟隐藏核心动机)
3. [先复习:SM、Warp、Block 怎么协作](#3-先复习smwarpblock-怎么协作)
4. [Occupancy 的定义](#4-occupancy-的定义)
5. [什么限制了 Occupancy](#5-什么限制了-occupancy)
6. [T4 上的关键数字](#6-t4-上的关键数字)
7. [手工估算例子(mat_mul)](#7-手工估算例子mat_mul)
8. [高 Occupancy ≠ 一定更快](#8-高-occupancy--一定更快)
9. [动手实验:改 block 看 GFLOPS](#9-动手实验改-block-看-gflops)
10. [基础篇自测](#10-基础篇自测)

**进阶篇**
11. [先纠正三个常见误解](#11-先纠正三个常见误解)
12. [Occupancy 的精确计算](#12-occupancy-的精确计算)
13. [四个限制因素深析](#13-四个限制因素深析)
14. [理论 vs 实际 occupancy](#14-理论-vs-实际-occupancy)
15. [Occupancy 与延迟隐藏的定量关系](#15-occupancy-与延迟隐藏的定量关系)
16. [什么时候 occupancy 重要、什么时候不重要](#16-什么时候-occupancy-重要什么时候不重要)
17. [系统化调优方法论](#17-系统化调优方法论)
18. [寄存器调优实战](#18-寄存器调优实战)
19. [Shared memory 调优实战](#19-shared-memory-调优实战)
20. [Block size 调优实战](#20-block-size-调优实战)
21. [用 Nsight Compute 看 occupancy](#21-用-nsight-compute-看-occupancy)
22. [完整调优案例](#22-完整调优案例)
23. [面试深度题](#23-面试深度题)
24. [速查表](#24-速查表)
25. [API 查询代码](#25-api-查询代码)
26. [阅读资料](#26-阅读资料)

---

# 基础篇

## 1. 一句话是什么

**Occupancy(占用率)= 一个 SM 上实际在跑的 warp 数,占该 SM 最多能同时驻留 warp 数的比例。**

Occupancy 高 → SM 里同时有更多 warp 可以切换 → **更有机会**在等内存时换别的 warp 干活(隐藏
延迟)。但 **Occupancy 只是「调度机会」,不是「算力本身」**。

> 如果你是第一次接触这个概念,先别急着记定义。下一节从「GPU 为什么慢」讲起,让你明白 occupancy
> 到底要解决什么问题——理解了动机,定义自然就懂了。

---

## 2. 为什么需要它:延迟隐藏(核心动机)

这一节是整篇的灵魂。**occupancy 存在的唯一理由,就是「延迟隐藏」。** 不理解延迟隐藏,occupancy
就只是一个背不下来的公式。

### 第一步:GPU 访问内存,非常慢

一个最关键的事实:GPU 上的线程要从 global memory(显存)读一个数,**要等几百个时钟周期**才能
拿到。而做一次加法只要几个周期。打个比方:

```text
做一次计算   ≈ 抬手拿起桌上的笔(1 秒)
读一次显存   ≈ 走到隔壁仓库取一个零件(400 秒)
```

所以"等数据"才是 GPU 的主要瓶颈,不是"算得慢"。

### 第二步:只有一个 warp 时,SM 在空等

假设一个 SM 上只有 1 个 warp。它执行到「读显存」那条指令时,必须停下来等数据回来——这几百个
周期里,**整个 SM 什么都没干,纯空转**:

```text
只有 1 个 warp:
Warp A:  算 ── 发出读显存 ──────等 400 周期────── 拿到数据 ── 继续算
SM 状态: 忙          【────────── 空转浪费 ──────────】        忙
```

这就像一个工人去仓库取零件,取的时候整个车间停摆。太浪费了。

### 第三步:养很多 warp,一个等、就切另一个

GPU 的解决办法非常聪明:**在一个 SM 上同时养很多 warp**。当 warp A 卡在等显存时,调度器立刻切
去执行 warp B;B 也等了,就切 warp C……等轮一圈回来,A 的数据早就到了。于是「等待」被「别人的
计算」填满,SM 几乎不空转:

```text
养了很多 warp:
Warp A:  算 ─ 读显存 ──────等──────  拿到 ─ 算
Warp B:        算 ─ 读显存 ──────等──────  拿到 ─ 算
Warp C:              算 ─ 读显存 ──────等──────  拿到
Warp D:                    算 ─ 读显存 ──────等
SM 状态: 忙   忙    忙     忙   忙 ...(一直有 warp 可算,不空转)
```

**这就是「延迟隐藏(latency hiding)」**:延迟没有消失(每个 warp 还是等了 400 周期),但 SM 利用
别的 warp 的计算把这段等待「盖住」了,整体吞吐大幅提升。

### 第四步:为什么 GPU 切 warp 不要钱(关键!)

你可能会问:CPU 切换线程很贵(要保存/恢复一堆寄存器状态),GPU 切来切去不亏吗?

**不亏,因为 GPU 的 warp 切换几乎是零开销的。** 原因是:

```text
CPU 切线程:线程的状态平时在内存里,切换要"搬进搬出"寄存器 → 贵
GPU 切 warp:所有驻留 warp 的寄存器状态【一直常驻在 SM 的寄存器文件里】
            → 切换只是「换个 warp 编号继续发指令」,不搬数据 → 几乎免费
```

正因为切换免费,GPU 才敢「多养 warp、疯狂轮换」来隐藏延迟。这也解释了下一个问题:为什么 warp
的数量(寄存器够不够养这么多)这么重要。

### 第五步:Occupancy 就是衡量「养了多少 warp 可供轮换」

现在回头看定义就通了:

```text
Occupancy = SM 上实际驻留的 warp 数 / SM 最多能驻留的 warp 数
          = 「你养了多少 warp」 / 「最多能养多少」
```

- occupancy 高 = 养的 warp 多 = 有充足的「备胎」可以在等待时轮换 = **更容易隐藏延迟**。
- occupancy 低 = 养的 warp 少 = 一个 warp 等待时可能没有别的可切 = SM 容易空转。

所以 occupancy 衡量的不是「算力」,而是「**隐藏延迟的能力有多强**」。

> 一句话记住:**occupancy 高 ≈ 备胎 warp 多 ≈ 等数据时有活干 ≈ 不空转。**
> 它是手段(多养 warp 来盖住延迟),不是目的(目的是高吞吐)。

---

## 3. 先复习:SM、Warp、Block 怎么协作

```text
Grid(一次 kernel launch)
  └── 很多 Block
        └── 很多 Thread(按 blockDim 排列)

硬件:
  GPU 有 40 个 SM(T4)
  Block 被调度到某个 SM 上执行
  Warp = 32 个线程,是 SM 调度的最小单位
  同一 warp 内线程执行同一条指令(SIMT)
```

和 `mat_mul` 的对应:

```cuda
dim3 block(16, 16);   // 256 线程 / block = 8 个 warp
dim3 grid(...);
matmul_gpu<<<grid, block>>>(...);
```

| 概念 | mat_mul 例子 |
|------|----------------|
| 一个 thread 做什么 | 算 C 的一个元素 |
| 一个 block 多少线程 | 16×16 = 256 = **8 warp** |
| grid | 盖住整个 M×N 输出矩阵 |

---

## 4. Occupancy 的定义

```text
Occupancy = 活跃 warp 数 / SM 最大 warp 容量
```

T4(Turing)每个 SM 最多同时驻留 **32 个 warp**(= 1024 线程)。例:某 SM 上同时有 16 个活跃 warp:

```text
Occupancy = 16 / 32 = 50%
```

**「活跃 warp」是什么意思**:不是「grid 里总共有多少 warp」,而是 **此刻在这个 SM 上已经被调度、
占着资源** 的 warp。一个 grid 可能有上万个 block,但 **每个 SM 同一时刻只执行其中一部分**;其余
block 在排队等 SM 空位。

---

## 5. 什么限制了 Occupancy

一个 kernel 能在 SM 上驻留多少 block(进而有多少 warp),由下面几条**资源约束共同决定,取最严格
(最小)的那条**:

| 限制因素 | T4 上限值 | 含义 |
|----------|-----------|------|
| **每 block 线程数** | ≤ **1024** | 单个 block 三维乘积上限 |
| **每 SM 线程数** | ≤ **1024** | 一个 SM 上所有驻留 block 的线程总和 |
| **每 SM warp 数** | ≤ **32** | = 1024/32,occupancy 的分母 |
| **每 SM block 数** | ≤ **16** | 一个 SM 同时容纳的 block 数上限 |
| **每 SM 寄存器数** | **65536** 个(32-bit) | 平分给 SM 上所有驻留线程 |
| **每 block 寄存器数** | ≤ **65536** | 单个 block 能用的寄存器上限 |
| **每 SM shared memory** | **64 KB** | 所有驻留 block 共享 |
| **每 block shared memory** | ≤ **48 KB**(默认) | 单个 block 能用的 shared 上限 |

> 这些值都是 **T4(Turing, sm_75)** 的硬件常量,换架构会变。**不要背,用 `device_query` /
> `cudaGetDeviceProperties` 查**(见下一节)。

可以记:**block 越「胖」(线程多 / 寄存器多 / shared 大),SM 能同时塞的 block 越少,Occupancy
可能越低。** 这就是「木桶效应」——occupancy 由最紧的那条资源决定。(精确算法见进阶篇第 12 节)

### 寄存器怎么限制 Occupancy(直觉版)

表格里「寄存器用量」最抽象,用 T4 真实数字算一遍就懂。前提:每个 SM 有 65536 个寄存器,最多
驻留 1024 线程。这些寄存器要**平分给所有驻留线程**,每个线程用多少由编译器定死:

```text
能驻留的线程数 ≤ 65536 / 每线程寄存器数

每线程 32 寄存器 → 65536/32 = 2048(但线程上限 1024)→ 满 occupancy ✅
每线程 64 寄存器 → 65536/64 = 1024 线程 = 32 warp → 刚好 100% ✅
每线程 128 寄存器 → 65536/128 = 512 线程 = 16 warp → 50% ⚠️
每线程 255 寄存器 → 65536/255 ≈ 256 线程 = 8 warp → 25% ❌
```

**kernel 越「吃寄存器」,一个 SM 能养的 warp 越少,occupancy 越低,隐藏延迟的备胎越少。** 这就是
为什么写复杂 kernel(很多局部变量)有时会莫名变慢。

> 但**不要为了凑 occupancy 盲目压寄存器**:强行减寄存器可能导致编译器把值 spill 到 local memory
> (在显存里,慢),反而更糟。查寄存器用量:`nvcc -Xptxas=-v -arch=sm_75 xxx.cu`,看 `Used N registers`。

---

## 6. T4 上的关键数字

来自 `device_query`:

| 属性 | T4 值 | `cudaDeviceProp` 字段 |
|------|-------|----------------------|
| SM 数量 | 40 | `multiProcessorCount` |
| warp size | 32 | `warpSize` |
| max threads / block | 1024 | `maxThreadsPerBlock` |
| **max threads / SM** | **1024** | `maxThreadsPerMultiProcessor` |
| **max warps / SM** | **32** | (= 上面 /32,occupancy 分母) |
| max blocks / SM | 16 | `maxBlocksPerMultiProcessor` |
| **registers / SM** | **65536** | `regsPerMultiprocessor` |
| registers / block | 65536 | `regsPerBlock` |
| shared mem / block | 48 KB | `sharedMemPerBlock` |
| shared mem / SM | 64 KB | `sharedMemPerMultiprocessor` |

> 别背,用 `cudaGetDeviceProperties` 查。换架构(如 A100 每 SM 2048 线程、64 warp)数字全变。

---

## 7. 手工估算例子(mat_mul)

kernel:`matmul_gpu`,block `(16,16)` = 256 线程 = **8 warp/block**。假设寄存器压力小,shared = 0。

```text
每 SM 最多 1024 线程 → 1024 / 256 = 4 个 block 同时驻留
→ 4 × 8 = 32 warp
→ Occupancy ≈ 32/32 = 100%(理论上限)
```

改 block 看变化:

```text
(32,32)=1024 线程=32 warp/block → 每 SM 只放 1 个 block → 32 warp → 仍可能 100%,但并行粒度变粗
(8,8)=64 线程=2 warp/block → 1024/64=16 个 block(也撞 maxBlocksPerSM=16)→ 16×2=32 warp → 也可到 100%
```

看起来都能 100%?那为什么 GFLOPS 还会变 —— 见下一节。

---

## 8. 高 Occupancy ≠ 一定更快

### 原因 1:Occupancy 只影响「隐藏延迟」的能力

- **Memory-bound** kernel(如 vec_add、transpose naive):等内存多 → 高 Occupancy **可能**有帮助
- **Compute-bound** kernel(如大矩阵 GEMM):SM 已经在拼命算 → 再塞更多 warp **不一定**更快

`mat_mul` 大矩阵偏 **compute-bound**,所以改 block,Occupancy 可能差不多但 GFLOPS 仍变化(线程
组织、缓存、launch 开销不同)。

### 原因 2:寄存器 spill 与指令效率

为强行提高 Occupancy 而减寄存器,有时反而让编译器 spill 到 local memory,**更慢**。

### 原因 3:block 太小 → 调度开销占比大

(8,8) 只有 64 线程,grid 里 block 数量爆炸,launch / 调度开销上升,GFLOPS 可能下降。

### 结论(写进笔记)

```text
1. Occupancy = SM 上活跃 warp / 最大 warp 容量。
2. 限制因素:threads/block、registers、shared memory(取最小)。
3. 高 Occupancy 有利于 memory-bound kernel 隐藏延迟。
4. 高 Occupancy ≠ 一定更快;要看实测。
5. 入门阶段只需观察 block 变化对性能的影响,不追求调到极致。
```

> 这个"不一定更快"的结论很重要,进阶篇第 11 节会把它展开成三个具体误解 + 量化解释。

---

## 9. 动手实验:改 block 看 GFLOPS

固定 1024³,只改 `blockDim`,记录 kernel GFLOPS。改 `mat_mul.cu` 的 `test_matmul_gpu`:

```cuda
dim3 block(16, 16);   // 改成 (8,8) 或 (32,32)
dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
```

记录表(填 `notes/week01.md`):

| blockDim | 线程/block | warp/block | Kernel (ms) | GFLOPS (1024³) | 备注 |
|----------|------------|------------|---------------|----------------|------|
| (8,8) | 64 | 2 | | | block 多,调度开销可能大 |
| (16,16) | 256 | 8 | 5.269 | 407.5 | 基线 |
| (32,32) | 1024 | 32 | | | 每 block 线程上限 |

```bash
cd week01_basics/mat_mul_naive
nvcc -O3 -arch=sm_75 -std=c++14 -o mat_mul mat_mul.cu
./mat_mul --m 1024 --n 1024 --k 1024
```

预期:三种 block 都 PASS;GFLOPS 有差异,不一定 (32,32) 最快;若 (32,32) 和 (16,16) 接近,说明
naive matmul 主要不是 Occupancy 瓶颈。

---

## 10. 基础篇自测

1. Occupancy 是什么?(一句话)
2. T4 一个 SM 最多多少个 warp?
3. `(16,16)` block 有几个 warp?
4. 列出 3 个限制 Occupancy 的因素。
5. 为什么 vec_add 和 mat_mul 对 Occupancy 的敏感度可能不同?
6. 高 Occupancy 一定更快吗?为什么?

<details>
<summary>参考答案</summary>

1. SM 上活跃 warp 数占该 SM 最大 warp 容量的比例。
2. 32 个 warp(1024 线程)。
3. 256 / 32 = 8 个 warp。
4. 每 block 线程数、寄存器用量、shared memory 用量(还有每 SM 最大 block 数)。
5. vec_add 偏 memory-bound,更吃带宽和延迟隐藏;mat_mul 大矩阵偏 compute-bound。
6. 不一定;compute-bound 时算力已饱和,且过小 block 有调度开销,寄存器 spill 也会变慢。

</details>

> 基础篇到此。如果以上都懂了,继续进阶篇——把"取最小"算精确、把"不一定更快"讲透、学会系统
> 化调优。

---

# 进阶篇

> 假设你已掌握基础篇:occupancy 是什么、为什么(延迟隐藏)、定义、限制因素。下面深入"怎么精确
> 算、被什么限制、怎么系统化调优、和其他指标怎么配合"。

## 11. 先纠正三个常见误解

进阶的第一步是丢掉错误直觉:

```text
误解一:"occupancy 越高越快"
真相:  occupancy 是"延迟隐藏的机会",不是性能本身。超过"够隐藏延迟"的临界点后,
       再高也没用;甚至为提 occupancy 牺牲每线程资源(寄存器)反而更慢。

误解二:"occupancy 100% 是目标"
真相:  很多高性能 kernel(尤其 GEMM)故意用低 occupancy + 高 ILP/寄存器复用,
       跑得比 100% occupancy 还快(经典的 Volkov 反例)。

误解三:"occupancy 低就是 bug,必须修"
真相:  先判断 kernel 是不是 latency-bound。如果已经 memory-bound 打满带宽,
       或 compute-bound 算力跑满,occupancy 低完全没问题。
```

> 贯穿进阶篇的核心观点:**occupancy 是手段不是目的。目的是"让执行单元/内存带宽不空闲"。**
> occupancy 只在"延迟没被隐藏住、执行单元在空等"时才需要提高。

---

## 12. Occupancy 的精确计算

### 12.1 公式

```text
occupancy = 实际驻留 warp 数 / SM 最大 warp 数
实际驻留 warp 数 = 每 SM 驻留 block 数 × 每 block warp 数
每 SM 驻留 block 数 = 取下面四个限制的最小值
```

### 12.2 四个限制取最小值

```text
限制1(线程/warp): SM最大warp数 / 每block的warp数
限制2(block数):    SM最大block数(硬件上限)
限制3(寄存器):     SM寄存器总数 / (每线程寄存器数 × 每block线程数)
限制4(shared):     SM shared总量 / 每block的shared用量

每SM驻留block数 = min(限制1, 限制2, 限制3, 限制4)
```

### 12.3 完整计算示例

kernel:`block=256 线程`,`每线程 40 寄存器`,`每 block 8KB shared`。

```text
每 block warp 数 = 256/32 = 8 warp
限制1(warp):    32 / 8 = 4 个 block
限制2(block):   16 个 block(硬件上限,不构成瓶颈)
限制3(寄存器):  65536 / (40 × 256) = 6.4 → 6 个 block
限制4(shared):  64KB / 8KB = 8 个 block

每 SM 驻留 = min(4,16,6,8) = 4 个 block ← 被 warp 数限制
驻留 warp = 4×8 = 32 → occupancy = 32/32 = 100%
```

寄存器提到 80:

```text
限制3: 65536 / (80×256) = 3.2 → 3 个 block
min(4,16,3,8) = 3 → 24 warp → occupancy = 75%   ← 寄存器开始拖累
```

> 这就是"寄存器压力降 occupancy"的精确机制:寄存器是固定物理资源,每线程用得多,限制3 就小,
> 驻留 block 变少。

---

## 13. 四个限制因素深析

### 13.1 线程/warp 数限制

block 太小(如 32 线程=1 warp),即使其他资源充足,也受 block 数上限制约:

```text
block=32(1 warp),T4 最多 16 block → 只有 16 warp → occupancy = 50%
→ block 太小浪费:warp 数上限用不满
```

### 13.2 寄存器限制(最常见瓶颈)

```bash
nvcc -Xptxas=-v -arch=sm_75 kernel.cu -o kernel
# ptxas info: Used 40 registers, ...
```

寄存器多的原因:复杂计算、大量局部变量、循环展开、内联。**register tiling**(卷六)故意用更多
寄存器换数据复用——这是"低 occupancy 高性能"的典型。

### 13.3 Shared memory 限制

```text
T4 每 SM 64KB shared
每 block 用 32KB → 只能驻留 2 block
每 block 用 16KB → 能驻留 4 block
```

tile 大小的权衡:大 tile 复用多但占 shared 多、压低 occupancy。T4 的 shared/L1 可配置(卷九/04)。

### 13.4 Block 数硬限制

SM 有"最多驻留几个 block"的硬上限。block 很小、其他资源充足时,这个上限成为瓶颈(见 13.1)。

### 13.5 限制因素诊断表

```text
occupancy 不到 100%,是谁限制的?
├─ block 很小(<128) → 可能是 block 数上限 → 增大 block
├─ 寄存器多(-Xptxas=-v)→ 寄存器限制 → 减寄存器或接受
├─ shared 用得多 → shared 限制 → 减 tile 或调 L1/shared 比例
└─ 都不明显 → ncu Occupancy section 看 Block Limit 哪个最小
```

---

## 14. 理论 vs 实际 occupancy

```text
理论 occupancy(theoretical):按资源算出的上限(第 12 节)。"最多能驻留多少 warp"。
实际 occupancy(achieved):  运行时真正达到的平均值(ncu achieved_occupancy)。"实际平均多少"。
```

**两者为什么会差**(理论高但实际低):

```text
1. 尾部效应(tail effect):grid block 数不是 SM 整数倍,最后一批让部分 SM 空闲。
   如 40 SM 但只有 50 个 block:第一轮 40 满载,第二轮只 10 个 → 30 个 SM 闲置。
2. 负载不均衡:有的 block 早结束,SM 上 warp 动态减少。
3. block 执行时间差异大。
4. 启动/退出阶段 warp 不满。
```

> 诊断:理论高但实际低 → 多半是 **grid 太小或尾部效应**。解法:增大 grid(block 数远多于 SM 数,
> 如几百个 block 喂 40 SM)、grid-stride loop、均衡每 block 工作量。

---

## 15. Occupancy 与延迟隐藏的定量关系

### 15.1 需要多少 warp 才能隐藏延迟

核心(Little's Law):

```text
需要的并发 = 延迟 × 吞吐
要隐藏内存延迟,需足够多"在途内存请求"填满延迟窗口。warp 越多,在途请求越多。
```

### 15.2 关键洞察:够了就行,不用满

```text
occupancy 0%→50%:延迟逐渐被隐藏,性能快速上升
occupancy 50%→100%:延迟可能已隐藏住,性能曲线变平(再加 warp 没有更多延迟可藏)

存在"临界 occupancy":到了它延迟就藏住了,超过它收益递减/为零。
临界点取决于 kernel 访存/计算比,通常远低于 100%。
```

这解释了误解一:**不是越高越快,而是"到临界点"就够了**。很多 kernel 50-60% 就跑满。

### 15.3 ILP 可以替代部分 occupancy

```text
低 occupancy + 高 ILP:每线程做多个独立的数(如 register tiling 算 4×4),
                       一个 warp 就有很多独立指令并行 → 也能隐藏延迟
高 occupancy + 低 ILP:每线程做一个数,靠 warp 数量堆并发

两条路都能隐藏延迟,可互相替代。这就是 Volkov "低占用高性能"的原理。
```

---

## 16. 什么时候 occupancy 重要、什么时候不重要

```text
occupancy 重要(该提高):
  ✓ kernel 是 latency-bound(执行单元空等,stall 多是 Long Scoreboard)
  ✓ occupancy 很低(<30%)且 ILP 也不高
  ✓ ncu 显示 issue slot 利用率低、warp 经常无可发射

occupancy 不重要(别瞎提):
  ✗ 已 memory-bound,DRAM 带宽打满(提 occupancy 不增带宽)
  ✗ 已 compute-bound,算力跑满
  ✗ 低 occupancy 但高 ILP,性能已很好
  ✗ 为提 occupancy 要牺牲寄存器/shared,可能得不偿失
```

> 决策流程:**先判 bound 类型(ncu Speed of Light)→ 只有 latency-bound 才考虑 occupancy →
> 再看是 occupancy 不足还是 ILP 不足**。不要一上来就盯 occupancy 数字。

---

## 17. 系统化调优方法论

```text
1. 正确性固定:优化前 kernel 必须正确(CPU reference + sanitizer)
2. 测可信基线:warmup + 多次 + 中位数 + Event 计时(卷五/01)
3. 判 bound 类型:ncu Speed of Light → memory / compute / latency
4. 对症下药:
   ├─ memory-bound → 合并访问/shared复用/减数据搬运(不是 occupancy!)
   ├─ compute-bound → 减指令/FMA/Tensor Core
   └─ latency-bound → 才看 occupancy:
        ├─ occupancy 低且 ILP 低 → 提 occupancy(调寄存器/shared/block)
        └─ occupancy 低但 ILP 高 → 可能已够,别动
5. 扫参数:block size、寄存器上限、tile 大小,each 测时间 + occupancy
6. 验证因果:改了 X,occupancy 和时间按预期变化吗?
7. 留最优 + 记录:不只留最快数字,记下为什么(卷十/04 基线)
```

> 关键纪律:**一次只改一个变量**。同时改 block size 和寄存器上限,无法归因。

---

## 18. 寄存器调优实战

### 18.1 查寄存器用量

```bash
nvcc -Xptxas=-v -arch=sm_75 kernel.cu -o kernel
# ptxas info: Used 40 registers, 8192 bytes smem, ...
#                  ^^ 每线程寄存器数;还要看 spill stores/loads(非0就是溢出,警告!)
```

### 18.2 控制寄存器数的手段

```text
方法1:__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)
  __global__ void __launch_bounds__(256, 4) myKernel(...) { }
  含义:每 block 最多 256 线程,目标每 SM 至少 4 个 block → 编译器据此限制寄存器

方法2:--maxrregcount=N(编译选项)全局限制,但太狠会 spill

方法3:精简代码(减局部变量、避免不必要展开、缩小局部数组)
```

### 18.3 权衡

```text
寄存器多:+ 复用多(register tiling)、ILP 高   - occupancy 低
寄存器少:+ occupancy 高                       - 可能 spill 或复用减少
最优点要实测:扫不同 --maxrregcount,测时间 + occupancy + 看 spill
```

> 警告:**强压寄存器导致 spill 通常比低 occupancy 更糟**。spill 把变量塞进 local memory(在显存,
> 几百拍),看到 `spill stores/loads` 非 0 要警惕。

---

## 19. Shared memory 调优实战

### 19.1 用量与 occupancy

```text
每 block shared = tile 元素数 × 元素大小 + padding
[32][33] float = 32×33×4 = 4224 字节/block → 64KB/4224 ≈ 15 block(不成瓶颈)
[64][65] float ≈ 16.6KB → 只能 3-4 block,压低 occupancy
```

### 19.2 调 L1/shared 划分

T4 的 shared 和 L1 共用物理 SRAM(卷九/04),可调比例:

```cpp
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);  // 尽量给 shared
```

```text
复用密集(GEMM/卷积):偏 shared       随机访问多:偏 L1
```

---

## 20. Block size 调优实战

### 20.1 经验起点

```text
- 一定是 32 的倍数(否则最后一个 warp 不满)
- 常见好值:128 / 256 / 512
- 太小(<64):block 数上限成瓶颈,occupancy 上不去
- 太大(1024):每 block 资源多,驻留 block 少;尾部效应明显
- 256 是很多 kernel 的甜点,但必须实测
```

### 20.2 用 API 求理论最优

```cpp
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);
// blockSize = 使理论 occupancy 最高的 block 大小
```

> API 给的是**理论最优**,不保证实际最快。是好起点,但仍要扫参数实测。

### 20.3 扫描

```text
对 {64,128,256,512,1024} 各跑一遍,记录:
  block size | 时间 | 理论 occupancy | 实际 occupancy(ncu) | 寄存器 | spill
选最快的,并解释为什么(不只留数字)
```

---

## 21. 用 Nsight Compute 看 occupancy

### 21.1 命令

```bash
nvcc -lineinfo -arch=sm_75 kernel.cu -o kernel
ncu --section Occupancy --kernel-name regex:myKernel ./kernel <args>
```

### 21.2 关键字段

```text
Theoretical Occupancy:    理论上限(资源算出)
Achieved Occupancy:       实际达到(运行时平均)
Block Limit Registers:    寄存器限制的 block 数(限制3)
Block Limit Shared Mem:   shared 限制的 block 数(限制4)
Block Limit Warps:        warp 数限制的 block 数(限制1)
Block Limit SM:           SM block 上限(限制2)
                          → 看哪个最小,就是 occupancy 的瓶颈!
```

### 21.3 判读流程

```text
1. Theoretical vs Achieved:
   理论低 → 资源限制(看 Block Limit 哪个最小)
   理论高但实际低 → 尾部效应/grid 太小/负载不均(第 14 节)
2. 看 Block Limit 四个值哪个最小 → 定位瓶颈资源
3. 结合 Speed of Light:occupancy 低但已 memory-bound → 别管 occupancy
4. 结合 Warp State:Long Scoreboard 高 + occupancy 低 → 提 occupancy 有望
```

---

## 22. 完整调优案例

**目标:优化一个 latency-bound 的 kernel。**

```text
第0步 现状:kernel 慢,理论 occupancy 50%
第1步 测基线:warmup+中位数,记录时间
第2步 判 bound:ncu Speed of Light → memory 40%, compute 30% → 都不高 → latency-bound
第3步 看 Warp State:Long Scoreboard stall 高 → 访存延迟没藏住 → occupancy 有望帮上
第4步 看 Occupancy section:
        Theoretical 50%, Achieved 45%(接近,不是尾部效应)
        Block Limit Registers = 3(最小!)→ 寄存器是瓶颈
第5步 查寄存器:-Xptxas=-v → 每线程 72 寄存器,无 spill
第6步 优化:加 __launch_bounds__(256, 4),促使编译器降到 64 寄存器
第7步 复测:理论 occupancy 升到 75%,Long Scoreboard 下降,时间下降 25%
        → 假设验证:寄存器降→occupancy升→延迟更好隐藏→变快 ✓
第8步 继续?:再压到 48 → 出现 spill → 反而变慢 → 回退,64 是甜点
第9步 记录:64 寄存器/256 block 为最优,附 occupancy 和时间数据
```

> 完整调优逻辑:**判 bound → 定位限制资源 → 对症调整 → 验证因果 → 找甜点(不是越高越好)**。

---

## 23. 面试深度题

**Q1:occupancy 100% 一定最快吗?为什么?**
不一定。occupancy 是延迟隐藏的机会,到临界点后收益递减。很多高性能 kernel(如 GEMM)用低
occupancy + 高 ILP/寄存器复用,比 100% 更快(Volkov 反例)。且为提 occupancy 牺牲寄存器可能 spill
反而变慢。

**Q2:理论 occupancy 高但实际低,什么原因?**
尾部效应(grid block 数不是 SM 整数倍)、grid 太小、负载不均衡、block 执行时间差异。解法:增大
grid、grid-stride loop、均衡负载。

**Q3:怎么定位是哪个资源限制了 occupancy?**
ncu Occupancy section 看四个 Block Limit,最小的就是瓶颈;或手算 min(限制1-4)。寄存器最常见,
`-Xptxas=-v` 查每线程寄存器数。

**Q4:ILP 和 occupancy 什么关系?**
都是隐藏延迟的手段,可互相替代。高 ILP(一个 warp 内多条独立指令,如 register tiling 每线程算多个
输出)能在低 occupancy 下隐藏延迟。"低 occupancy 高性能"靠的就是高 ILP。

**Q5:什么时候不该管 occupancy?**
已 memory-bound(带宽打满)或 compute-bound(算力跑满)时提 occupancy 无用。先用 ncu Speed of Light
判 bound,只有 latency-bound 且 occupancy/ILP 都不足时才提。

**Q6:`__launch_bounds__` 是干什么的?**
在 kernel 上标注最大线程数和目标每 SM block 数,提示编译器控制寄存器以保证目标 occupancy。用于
编译器默认寄存器太多、压低 occupancy 时(但太狠会 spill)。

**Q7:为什么 block size 不能太小?**
SM 有 block 数硬上限(T4 ~16)。block 太小(32=1warp),即使资源充足也最多 16 warp,occupancy 只
50%。要 block 够大(≥128)才能用满 warp 数上限。

**Q8:register spill 和低 occupancy 哪个更糟?**
通常 spill 更糟。spill 把变量塞进 local memory(物理在显存,几百拍),每次访问都慢;低 occupancy
只是并发少一点。强压寄存器导致 spill 往往得不偿失。

---

## 24. 速查表

### Occupancy 计算

```text
occupancy = 驻留warp / SM最大warp
驻留block = min(warp限制, block上限, 寄存器限制, shared限制)
T4: 32 warp/SM, ~16 block/SM, 64K 寄存器/SM, 64KB shared/SM
```

### 调优决策树

```text
慢?
└─ ncu Speed of Light 判 bound
   ├─ memory-bound → 合并/shared/减搬运(不是 occupancy)
   ├─ compute-bound → 减指令/Tensor Core
   └─ latency-bound(都低 + Long Scoreboard 高)
      ├─ occupancy 低 + ILP 低 → 提 occupancy
      │   └─ 看 Block Limit 哪个最小 → 调对应资源
      │       ├─ 寄存器 → __launch_bounds__ / --maxrregcount(防 spill)
      │       ├─ shared → 减 tile / 调 L1-shared 比例
      │       └─ block 太小 → 增大 block(≥128)
      └─ occupancy 低 + ILP 高 → 可能已够,别瞎动
```

### 关键工具命令

```text
nvcc -Xptxas=-v ...                          看寄存器/shared/spill
cudaOccupancyMaxPotentialBlockSize(...)      求理论最优 block size
cudaOccupancyMaxActiveBlocksPerMultiprocessor 算驻留 block 数
ncu --section Occupancy ...                  看理论/实际 occupancy + Block Limit
```

### 一句话原则

```text
occupancy 是手段不是目的:目的是执行单元/带宽不空闲。
先判 bound 类型,只有 latency-bound 才看 occupancy。
够隐藏延迟就行,不用追 100%;ILP 可替代 occupancy。
调寄存器防 spill,调 shared 看复用,调 block 别太小。
一次改一个变量,验证因果,找甜点而非最高值。
```

---

## 25. API 查询代码

不想手算时,用 Runtime API 查理论每 SM 最多能 active 几个 block:

```cuda
#include <cuda_runtime.h>
#include <cstdio>

__global__ void matmul_gpu(const float* A, const float* B, float* C,
                           int M, int N, int K);   // 已有声明

void print_occupancy_hint(int block_x, int block_y) {
  int block_size = block_x * block_y;
  int active_blocks = 0;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, matmul_gpu, block_size, 0);   // 最后参数 0 = 动态 shared 字节
  if (err != cudaSuccess) {
    printf("occupancy API error: %s\n", cudaGetErrorString(err));
    return;
  }
  const int warps_per_block = (block_size + 31) / 32;
  const int active_warps = active_blocks * warps_per_block;
  const int max_warps_per_sm = 32;                  // Turing T4
  printf("block(%d,%d) threads=%d → max active blocks/SM=%d, warps=%d/%d = %.1f%%\n",
         block_x, block_y, block_size, active_blocks, active_warps,
         max_warps_per_sm, 100.0f * active_warps / max_warps_per_sm);
}

int main() {
  print_occupancy_hint(8, 8);
  print_occupancy_hint(16, 16);
  print_occupancy_hint(32, 32);
  return 0;
}
```

| API | 作用 |
|-----|------|
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | 每 SM 最多几个 block |
| `cudaOccupancyMaxPotentialBlockSize` | 编译器帮你猜较优 block size |
| Nsight Compute `achieved_occupancy` | **实测** Occupancy |

> 这是硬件资源模型下的**理论上界**,实际运行还受 grid 大小、负载均衡等影响。

### 附:Occupancy 是 SM 内部指标,grid 大是另一回事(易混点)

Occupancy 是 **SM 内部** 指标;**grid 很大** 只保证有很多 block 可分给 40 个 SM,**不保证** 每个
SM 的 Occupancy 都高。

```text
1024×1024 输出,block(16,16):grid = 64×64 = 4096 个 block
  40 个 SM 平均分 → 每 SM 约 100 个 block 排队
  → block 供应充足,Occupancy 主要受 block「胖瘦」限制,而非 grid 不够大
```

小矩阵(如 256×256)grid 很小 → 部分 SM 空闲 → 整体利用率下降,这和单 SM Occupancy 相关但不同。

---

## 26. 阅读资料

| 资料 | 用途 |
|------|------|
| Programming Guide **2.3** Kernel Launch and Occupancy | 概念 + 限制因素 |
| CUDA C++ Best Practices Guide(Occupancy) | 为何高 occupancy 不保证性能 |
| Nsight Compute Occupancy section 文档 | profiler 判读 |
| CUDA Occupancy Calculator | 精确计算工具 |
| 本地 `week01_basics/device_query/` | T4 硬件参数 |

配套课程章节:
- [卷五/03 Occupancy、分歧与延迟隐藏](../cuda_deep_course/course/volume05_performance/03_Occupancy_分歧与延迟隐藏.md)
- [卷九/02 Warp 调度与延迟隐藏](../cuda_deep_course/course/volume09_hardware_architecture/02_Warp调度_scheduler_scoreboard与延迟隐藏.md) —— 硬件根源
- [卷九/04 存储层次硬件](../cuda_deep_course/course/volume09_hardware_architecture/04_存储层次硬件_寄存器_shared_L1L2_显存.md) —— 寄存器/shared 物理结构
- [卷五/05 Nsight Compute](../cuda_deep_course/course/volume05_performance/05_Nsight_Compute与Compute_Sanitizer.md) —— profiler 判读

---

**返回**:[CUDA学习路线图.md](CUDA学习路线图.md)
