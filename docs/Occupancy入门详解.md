# Occupancy 入门详解

对应学习路径：Week 1 **Step 11** / Day 7  
硬件环境：**Tesla T4（CC 7.5，sm_75，40 SM）**  
配套实验：`week01_basics/mat_mul_naive/mat_mul.cu`

---

## 目录

1. [一句话是什么](#一句话是什么)
2. [为什么需要它：延迟隐藏（核心动机，先看这个）](#为什么需要它延迟隐藏核心动机先看这个)
3. [先复习：SM、Warp、Block 怎么协作](#先复习smwarpblock-怎么协作)
4. [Occupancy 的定义](#occupancy-的定义)
5. [什么限制了 Occupancy](#什么限制了-occupancy)
6. [T4 上的关键数字](#t4-上的关键数字)
7. [手工估算例子（mat_mul）](#手工估算例子mat_mul)
8. [高 Occupancy ≠ 一定更快](#高-occupancy--一定更快)
9. [动手实验：改 block 看 GFLOPS](#动手实验改-block-看-gflops)
10. [API 查询：`cudaOccupancyMaxActiveBlocksPerMultiprocessor`](#api-查询cudaoccupancymaxactiveblockspermultiprocessor)
11. [和 Week 5 的关系](#和-week-5-的关系)
12. [自测题](#自测题)
13. [阅读资料](#阅读资料)

---

## 一句话是什么

**Occupancy（占用率）= 一个 SM 上实际在跑的 warp 数，占该 SM 最多能同时驻留 warp 数的比例。**

Occupancy 高 → SM 里同时有更多 warp 可以切换 → **更有机会**在等内存时换别的 warp 干活（隐藏延迟）。  
但 **Occupancy 只是「调度机会」，不是「算力本身」**。

> 如果你是第一次接触这个概念，先别急着记上面的定义。下一节会从「GPU 为什么慢」
> 讲起，让你明白 occupancy 到底要解决什么问题——理解了动机，定义自然就懂了。

---

## 为什么需要它：延迟隐藏（核心动机，先看这个）

这一节是整篇的灵魂。**occupancy 存在的唯一理由，就是「延迟隐藏」。** 不理解延迟
隐藏，occupancy 就只是一个背不下来的公式。

### 第一步：GPU 访问内存，非常慢

一个最关键的事实：GPU 上的线程要从 global memory（显存）读一个数，**要等几百个
时钟周期**才能拿到。而做一次加法只要几个周期。打个比方：

```text
做一次计算   ≈ 抬手拿起桌上的笔（1 秒）
读一次显存   ≈ 走到隔壁仓库取一个零件（400 秒）
```

所以"等数据"才是 GPU 的主要瓶颈，不是"算得慢"。

### 第二步：只有一个 warp 时，SM 在空等

假设一个 SM 上只有 1 个 warp 在跑。它执行到「读显存」那条指令时，必须停下来等
数据回来——这几百个周期里，**整个 SM 什么都没干，纯空转**：

```text
只有 1 个 warp：
Warp A:  算 ── 发出读显存 ──────等 400 周期────── 拿到数据 ── 继续算
SM 状态: 忙          【────────── 空转浪费 ──────────】        忙
```

这就像一个工人去仓库取零件，取的时候整个车间停摆。太浪费了。

### 第三步：养很多 warp，一个等、就切另一个

GPU 的解决办法非常聪明：**在一个 SM 上同时养很多 warp**。当 warp A 卡在等显存
时，调度器立刻切去执行 warp B；B 也等了，就切 warp C……等轮一圈回来，A 的数据
早就到了。于是「等待」被「别人的计算」填满，SM 几乎不空转：

```text
养了很多 warp：
Warp A:  算 ─ 读显存 ──────等──────  拿到 ─ 算
Warp B:        算 ─ 读显存 ──────等──────  拿到 ─ 算
Warp C:              算 ─ 读显存 ──────等──────  拿到
Warp D:                    算 ─ 读显存 ──────等
SM 状态: 忙   忙    忙     忙   忙 ...（一直有 warp 可算，不空转）
```

**这就是「延迟隐藏（latency hiding）」**：延迟没有消失（每个 warp 还是等了 400
周期），但 SM 利用别的 warp 的计算把这段等待「盖住」了，整体吞吐大幅提升。

### 第四步：为什么 GPU 切 warp 不要钱（关键！）

你可能会问：CPU 切换线程很贵（要保存/恢复一堆寄存器状态），GPU 切来切去不亏吗？

**不亏，因为 GPU 的 warp 切换几乎是零开销的。** 原因是：

```text
CPU 切线程：线程的状态平时在内存里，切换要"搬进搬出"寄存器 → 贵
GPU 切 warp：所有驻留 warp 的寄存器状态【一直常驻在 SM 的寄存器文件里】
            → 切换只是「换个 warp 编号继续发指令」，不搬数据 → 几乎免费
```

正因为切换免费，GPU 才敢「多养 warp、疯狂轮换」来隐藏延迟。这也解释了下一个问题：
为什么 warp 的数量（寄存器够不够养这么多）这么重要。

### 第五步：Occupancy 就是衡量「养了多少 warp 可供轮换」

现在回头看定义就通了：

```text
Occupancy = SM 上实际驻留的 warp 数 / SM 最多能驻留的 warp 数
          = 「你养了多少 warp」 / 「最多能养多少」
```

- occupancy 高 = 养的 warp 多 = 有充足的「备胎」可以在等待时轮换 = **更容易隐藏延迟**。
- occupancy 低 = 养的 warp 少 = 一个 warp 等待时可能没有别的可切 = SM 容易空转。

所以 occupancy 衡量的不是「算力」，而是「**隐藏延迟的能力有多强**」。这也是为什么
后面会反复强调：高 occupancy 对「等内存等得多」的 memory-bound kernel 帮助大，
对「本来就在拼命算」的 compute-bound kernel 帮助有限。

> 一句话记住：**occupancy 高 ≈ 备胎 warp 多 ≈ 等数据时有活干 ≈ 不空转。**
> 它是手段（多养 warp 来盖住延迟），不是目的（目的是高吞吐）。

---

## 先复习：SM、Warp、Block 怎么协作

```text
Grid（一次 kernel launch）
  └── 很多 Block
        └── 很多 Thread（按 blockDim 排列）

硬件：
  GPU 有 40 个 SM（T4）
  Block 被调度到某个 SM 上执行
  Warp = 32 个线程，是 SM 调度的最小单位
  同一 warp 内线程执行同一条指令（SIMT）
```

和 `mat_mul` 的对应：

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

## Occupancy 的定义

### 公式（Week 1 理解版）

```text
Occupancy = 活跃 warp 数 / SM 最大 warp 容量
```

T4（Turing）每个 SM 最多同时驻留 **32 个 warp**（= 1024 线程）。

例：某 SM 上同时有 16 个活跃 warp：

```text
Occupancy = 16 / 32 = 50%
```

### 「活跃 warp」是什么意思

不是「grid 里总共有多少 warp」，而是 **此刻在这个 SM 上已经被调度、占着资源** 的 warp。

一个 grid 可能有上万个 block，但 **每个 SM 同一时刻只执行其中一部分**；其余 block 在排队等 SM 空位。

---

## 什么限制了 Occupancy

一个 kernel 能在 SM 上驻留多少 block（进而有多少 warp），由下面几条**资源约束
共同决定，取最严格（最小）的那条**。每条都标了 T4 的具体数值：

| 限制因素 | T4 上限值 | 含义 | mat_mul naive |
|----------|-----------|------|----------------|
| **每 block 线程数** | ≤ **1024** | 单个 block 三维乘积上限；越大 → 每 SM 能放的 block 越少 | (16,16)=256 ✓ |
| **每 SM 线程数** | ≤ **1024** | 一个 SM 上所有驻留 block 的线程总和 | 4×256=1024 ✓ |
| **每 SM warp 数** | ≤ **32** | = 1024/32，occupancy 的分母就是它 | — |
| **每 SM block 数** | ≤ **16** | 一个 SM 同时容纳的 block 数上限 | 通常不是瓶颈 |
| **每 SM 寄存器数** | **65536** 个（32-bit） | 平分给 SM 上所有驻留线程；kernel 越吃寄存器 → 驻留线程越少 | naive 很少 ✓ |
| **每 block 寄存器数** | ≤ **65536** | 单个 block 能用的寄存器上限（= 整个 SM 的量） | naive 很少 ✓ |
| **每 SM shared memory** | **64 KB** | 所有驻留 block 共享；用太多 → block 数下降 | naive 没用 ✓ |
| **每 block shared memory** | ≤ **48 KB**（默认）<br/>最高可配到 ~64 KB | 单个 block 能用的 shared 上限 | naive 没用 ✓ |

> 这些值都是 **T4（Turing, sm_75）** 的硬件常量。换架构会变（如 A100 每 SM 最多
> 2048 线程、64 warp）。**不要背，用 `device_query` / `cudaGetDeviceProperties` 查**
> （见下面「T4 上的关键数字」一节）。

可以记：**block 越「胖」（线程多 / 寄存器多 / shared 大），SM 能同时塞的 block 越少，Occupancy 可能越低。**

### 怎么算"取最小"——一个完整例子

假设某 kernel：block = 256 线程（8 warp），每线程 40 寄存器，每 block 用 8 KB shared。
逐条算 T4 上「每 SM 最多几个 block」：

```text
按线程：    1024 / 256        = 4 个 block
按寄存器：  65536 / (256×40)  = 65536/10240 ≈ 6 个 block
按 shared： 64 KB / 8 KB      = 8 个 block
按 block 数上限：               16 个 block
按 warp 数： 32 / 8           = 4 个 block

取最小 → 4 个 block/SM → 4×8 = 32 warp → occupancy = 32/32 = 100% ✅
（这里"线程"和"warp"两条同时卡在 4，是瓶颈）
```

只要其中**任何一条**变紧（比如每线程寄存器涨到 80），结果就会被它拉低：

```text
寄存器变 80：65536 / (256×80) ≈ 3 个 block → 3×8 = 24 warp → occupancy = 75%
```

这就是「取最小」的实际含义——**occupancy 由最紧的那条资源决定，木桶效应。**

### 用数字看「寄存器」怎么限制 Occupancy

表格里「寄存器用量」最抽象，用 T4 的真实数字算一遍就懂了。前提（T4 硬件常量）：

```text
每个 SM：65536 个寄存器，最多驻留 1024 线程（= 32 warp）
```

这 65536 个寄存器要**平分给 SM 上同时驻留的所有线程**。而每个线程用多少寄存器，
是编译器在编译时定死的。于是有一个简单的除法：

```text
能驻留的线程数 ≤ 65536 / 每线程寄存器数

每线程 32 寄存器 → 65536/32 = 2048（但 T4 线程上限 1024）→ 线程上限先到，满 occupancy ✅
每线程 64 寄存器 → 65536/64 = 1024 线程 = 32 warp → 刚好满 occupancy 100% ✅
每线程 128 寄存器 → 65536/128 = 512 线程 = 16 warp → occupancy = 16/32 = 50% ⚠️
每线程 255 寄存器 → 65536/255 ≈ 256 线程 = 8 warp → occupancy = 8/32 = 25% ❌
```

看懂这张表，你就理解了一句话：**kernel 越「吃寄存器」，一个 SM 能养的 warp 越少，
occupancy 越低，隐藏延迟的备胎就越少。** 这也是为什么写复杂 kernel（很多局部变量）
有时会莫名变慢——寄存器涨上去，occupancy 掉下来了。

> 但注意（呼应后面「高 occupancy ≠ 一定更快」）：**不要为了凑 occupancy 盲目压寄存器**。
> 强行减寄存器可能导致编译器把值 spill 到 local memory（在显存里，慢），反而更糟。
> 查自己 kernel 的寄存器用量：`nvcc -Xptxas=-v -arch=sm_75 xxx.cu`，看 `Used N registers`。

---

## T4 上的关键数字

来自 `device_query`（你的机器实测）：

| 属性 | T4 值 | 和 Occupancy 的关系 |
|------|-------|---------------------|
| SM 数量 | 40 | grid 很大时，block 分布到 40 个 SM |
| warp size | 32 | 256 线程/block = 8 warp |
| max threads / block | 1024 | (32,32) 是上限 |
| **max threads / SM** | **1024** | 一个 SM 所有 block 的线程总和上限 |
| **max warps / SM** | **32** | = 1024/32，occupancy 的分母 |
| max blocks / SM | 16 | 一个 SM 最多 16 个 block |
| **registers / SM** | **65536**（32-bit） | 平分给所有驻留线程 |
| **registers / block** | **65536** | 单 block 寄存器上限 |
| shared mem / block | 48 KB（默认） | 用 shared 优化时要小心 |
| shared mem / SM | 64 KB | 所有 block 共享这 64KB |

对应的 `cudaDeviceProp` 字段（用 `cudaGetDeviceProperties` 查，别背）：

```text
multiProcessorCount             = 40     SM 数量
maxThreadsPerBlock              = 1024   每 block 线程上限
maxThreadsPerMultiProcessor     = 1024   每 SM 线程上限
maxBlocksPerMultiProcessor      = 16     每 SM block 上限
regsPerBlock                    = 65536  每 block 寄存器上限
regsPerMultiprocessor           = 65536  每 SM 寄存器总数
sharedMemPerBlock               = 49152  每 block shared (48 KB)
sharedMemPerMultiprocessor      = 65536  每 SM shared (64 KB)
warpSize                        = 32
```

Turing 每 SM **最多 32 warp、65536 个 32-bit 寄存器**（架构常量，查 PG 5.1）。

---

## 手工估算例子（mat_mul）

kernel：`matmul_gpu`，block `(16,16)` = 256 线程 = **8 warp/block**。  
假设寄存器压力很小，shared memory = 0。

**按线程数 / block 数限制：**

```text
每 SM 最多 1024 线程 → 1024 / 256 = 4 个 block 同时驻留
→ 4 × 8 = 32 warp
→ Occupancy ≈ 32/32 = 100%（理论上限，理想情况）
```

**若 block 改成 (32,32) = 1024 线程 = 32 warp/block：**

```text
每 SM 最多 1024 线程 → 只能放 1 个 block
→ 32 warp
→ Occupancy 仍可能是 100%，但只剩 1 个 block/SM，并行粒度变粗
```

**若 block 改成 (8,8) = 64 线程 = 2 warp/block：**

```text
1024 / 64 = 16 个 block（也受 maxBlocksPerSM=16 限制）
→ 16 × 2 = 32 warp
→ Occupancy 也可到 100%
```

看起来都能 100%？那为什么 GFLOPS 还会变 —— 见下一节。

---

## 高 Occupancy ≠ 一定更快

### 原因 1：Occupancy 只影响「隐藏延迟」的能力

- **Memory-bound** kernel（如 vec_add、transpose naive）：等内存多 → 高 Occupancy **可能**有帮助
- **Compute-bound** kernel（如大矩阵 GEMM）：SM 已经在拼命算 → 再塞更多 warp **不一定**更快

`mat_mul` 大矩阵偏 **compute-bound**，所以 block 从 (16,16) 改 (32,32)，Occupancy 可能差不多，但 GFLOPS 仍会变化（线程组织、缓存、launch 开销不同）。

### 原因 2：寄存器 spill 与指令效率

为了强行提高 Occupancy 而减少寄存器，有时反而让编译器 spill 到 local memory，**更慢**。

### 原因 3：block 太小 → 调度开销占比大

(8,8) 只有 64 线程，grid 里 block 数量爆炸，launch / 调度开销上升，GFLOPS 可能下降。

### Week 1 结论（写进笔记用）

```text
1. Occupancy = SM 上活跃 warp / 最大 warp 容量。
2. 限制因素：threads/block、registers、shared memory。
3. 高 Occupancy 有利于 memory-bound kernel 隐藏延迟。
4. 高 Occupancy ≠ 一定更快；mat_mul 要看实测 GFLOPS。
5. Week 1 只需观察 block 变化对性能的影响，不追求调到极致。
```

---

## 动手实验：改 block 看 GFLOPS

Step 11 推荐实验：固定 1024³，只改 `blockDim`，记录 kernel GFLOPS。

### 改哪里

`mat_mul.cu` 的 `test_matmul_gpu`：

```cuda
dim3 block(16, 16);   // 改成 (8,8) 或 (32,32)
dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
```

### 记录表（填 `notes/week01.md`）

| blockDim | 线程/block | warp/block | Kernel (ms) | GFLOPS (1024³) | 备注 |
|----------|------------|------------|---------------|----------------|------|
| (8,8) | 64 | 2 | | | block 多，调度开销可能大 |
| (16,16) | 256 | 8 | 5.269 | 407.5 | 当前基线 |
| (32,32) | 1024 | 32 | | | 每 block 线程上限 |

### 命令

```bash
cd week01_basics/mat_mul_naive
nvcc -O3 -arch=sm_75 -std=c++14 -o mat_mul mat_mul.cu
./mat_mul --m 1024 --n 1024 --k 1024
```

### 预期现象（不必完全一致）

- 三种 block **结果都应 PASS**
- GFLOPS **会有差异**，不一定 (32,32) 最快
- 若 (32,32) 和 (16,16) 接近：说明 naive matmul 主要不是 Occupancy 瓶颈

---

## API 查询：`cudaOccupancyMaxActiveBlocksPerMultiprocessor`

不想手算时，用 Runtime API 查 **理论** 每 SM 最多能 active 几个 block。

### 示例代码（可加到 mat_mul.cu 或单独小工具）

```cuda
#include <cuda_runtime.h>
#include <cstdio>

__global__ void matmul_gpu(const float* A, const float* B, float* C,
                           int M, int N, int K);  // 已有声明

void print_occupancy_hint(int block_x, int block_y) {
  int block_size = block_x * block_y;
  int min_grid_size = 0;
  int active_blocks = 0;

  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, matmul_gpu, block_size, 0);
  // 最后一个参数 0 = 动态 shared memory 字节数

  if (err != cudaSuccess) {
    printf("occupancy API error: %s\n", cudaGetErrorString(err));
    return;
  }

  const int warps_per_block = (block_size + 31) / 32;
  const int active_warps = active_blocks * warps_per_block;
  const int max_warps_per_sm = 32;  // Turing T4
  const float occ = 100.0f * active_warps / max_warps_per_sm;

  printf("block(%d,%d) threads=%d warps/block=%d\n",
         block_x, block_y, block_size, warps_per_block);
  printf("  max active blocks/SM = %d\n", active_blocks);
  printf("  active warps/SM (upper bound) = %d / %d = %.1f%%\n",
         active_warps, max_warps_per_sm, occ);
}

int main() {
  print_occupancy_hint(8, 8);
  print_occupancy_hint(16, 16);
  print_occupancy_hint(32, 32);
  return 0;
}
```

### 怎么读输出

```text
max active blocks/SM = 4
active warps/SM = 4 × 8 = 32 → 100%
```

这是 **硬件资源模型下的理论上界**，实际 kernel 运行时还受 grid 大小、负载均衡等影响。

### 相关 API（Week 5 再用）

| API | 作用 |
|-----|------|
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | 每 SM 最多几个 block |
| `cudaOccupancyMaxPotentialBlockSize` | 编译器帮你猜较优 block size |
| Nsight Compute `achieved_occupancy` | **实测** Occupancy（Week 6） |

---

## 和 Week 5 的关系

| 阶段 | 学到什么 |
|------|----------|
| **Week 1（现在）** | 知道 Occupancy 定义、限制因素、高 occupancy 不一定更快；做 block 对比实验 |
| **Week 5** | Roofline、寄存器/shared 调优、tile GEMM；Occupancy 作为调参手段之一 |
| **Week 6** | Nsight 看 **achieved occupancy** 实测曲线 |

Week 1 **不需要**：
- 背完整 Occupancy 公式推导
- 用 Excel / CUDA Occupancy Calculator 精确到小数点
- 为了 100% Occupancy 牺牲算法结构

---

## 自测题

1. Occupancy 是什么？（一句话）
2. T4 一个 SM 最多多少个 warp？
3. `(16,16)` block 有几个 warp？
4. 列出 3 个限制 Occupancy 的因素。
5. 为什么 vec_add 和 mat_mul 对 Occupancy 的敏感度可能不同？
6. 高 Occupancy 一定更快吗？为什么？

<details>
<summary>参考答案</summary>

1. SM 上活跃 warp 数占该 SM 最大 warp 容量的比例。
2. 32 个 warp（1024 线程）。
3. 256 / 32 = 8 个 warp。
4. 每 block 线程数、寄存器用量、shared memory 用量（还有每 SM 最大 block 数）。
5. vec_add 偏 memory-bound，更吃带宽和延迟隐藏；mat_mul 大矩阵偏 compute-bound。
6. 不一定；compute-bound 时算力已饱和，且过小 block 有调度开销，寄存器 spill 也会变慢。

</details>

---

## 阅读资料

| 资料 | 章节（v13.x） | Week 1 读什么 |
|------|---------------|---------------|
| Programming Guide | **2.3** Kernel Launch and Occupancy | 概念 + 限制因素 |
| [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | **Occupancy**（前半） | 为何高 occupancy 不保证性能 |
| Runtime API | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | 可选实验 |
| 本地 | `week01_basics/device_query/device_query.cu` | T4 硬件参数 |
| 本地 | `docs/T4实战指南.md` | SM 数量、性能预期 |
| 本地 | `notes/week01.md` | Occupancy 实验表 |

---

## 附录：和 mat_mul grid 的关系（易混点）

Occupancy 是 **SM 内部** 指标；**grid 很大** 只保证有很多 block 可以分给 40 个 SM，**不保证** 每个 SM 的 Occupancy 都高。

```text
1024×1024 输出，block (16,16)：
  grid = (64, 64) = 4096 个 block
  40 个 SM 平均分 → 每 SM 约 100 个 block 排队
  → block 供应充足，Occupancy 主要受 block「胖瘦」限制，而不是 grid 不够大
```

小矩阵（如 256×256）grid 很小 → 部分 SM 可能空闲 → **整体利用率**下降，这和单 SM Occupancy 是相关但不同的概念（Week 5 再分）。

---

**返回**：[Week1详细步骤.md](Week1详细步骤.md) Step 11 · [CUDA学习路线图.md](CUDA学习路线图.md)
