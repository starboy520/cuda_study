# 08 Hopper / Blackwell 深挖：TMA、WGMMA、Cluster、DSM

这一章回答一个面试里很常见的问题：

> Hopper、Blackwell 到底比 Ampere 多了什么？这些东西和写 CUDA kernel 有什么关系？

不要先背参数。先抓主线：

```text
Ampere:
  让 global -> shared 的搬运异步化，代表特性是 cp.async。

Hopper:
  进一步把大块多维搬运硬件化，代表特性是 TMA。
  让多个 block 可以作为 cluster 协同，代表特性是 Thread Block Cluster + DSM。
  让 warp group 级矩阵指令更强，代表特性是 WGMMA。

Blackwell:
  延续 CUDA 编程模型，同时增强 Tensor Core、低精度、内存层次和互连。
  数据中心 B200/GB200 属于 compute capability 10.0，B300/GB300 属于 10.3。
```

本章不追未公开细节，只讲官方公开的 CUDA 编程模型和调优含义。

## 1. 为什么 Hopper/Blackwell 对面试重要

CUDA 面试分两类：

```text
Kernel 优化面试：
  你能不能解释 cp.async、TMA、WGMMA、shared memory、occupancy、bank conflict？

AI infra 面试：
  你能不能解释 H100/B200 为什么适合训练/推理？NVLink、HBM、低精度、MIG、NCCL 怎么影响系统？
```

卷九前几章已经讲了 SM、warp、执行单元、内存层次。Hopper/Blackwell 是这些基础概念的现代化版本：

```text
不是重新发明 CUDA，
而是在同一套 CUDA 模型下，把数据搬运、矩阵计算、跨 block 协作做得更强。
```

## 2. Ampere 的 cp.async 是什么位置

理解 Hopper 的 TMA 前，先回忆 Ampere 的 `cp.async`。

传统 global 到 shared 的搬运：

```text
global memory -> register -> shared memory
```

问题：

```text
1. 中间占用寄存器。
2. 加载和计算容易串行。
3. 写双缓冲 pipeline 时，代码复杂。
```

Ampere 引入 `cp.async` 后：

```text
global memory -> shared memory
```

它的价值是：

```text
省寄存器
可异步
能和计算重叠
适合 GEMM / convolution / stencil 的 tiled load
```

但 `cp.async` 仍然是线程发起的搬运。你要组织每个 thread 搬什么、怎么 coalesce、怎么同步。

## 3. Hopper 的 TMA：把大块搬运交给硬件引擎

TMA 是 Tensor Memory Accelerator。

你可以先这样理解：

```text
cp.async:
  thread 自己发起许多小块 copy。

TMA:
  给硬件一个描述符，让硬件搬一个较大的多维 tile。
```

TMA 特别适合：

```text
矩阵 tile
多维 tensor tile
layout 转换或 swizzle 后的数据块
GEMM / attention / stencil 中的规律性大块搬运
```

它解决的问题是：

```text
以前为了把一个复杂 tile 搬进 shared，
你要写很多 thread-level indexing。

TMA 把“怎么搬一个多维 tile”这件事交给硬件，
kernel 更容易把精力放到计算 pipeline。
```

面试回答可以说：

```text
TMA 是 Hopper 引入的数据搬运能力，
把大块多维 global/shared transfer 硬件化，
比 thread 发起的 cp.async 更适合高性能 GEMM、attention 这类 tiled 算子。
```

## 4. TMA 不是什么

TMA 不是：

```text
不是让 global memory 变快。
不是自动帮你优化所有访存。
不是 T4/A100 都能用的通用功能。
```

它需要：

```text
Hopper 级别硬件支持
正确的数据布局
正确的同步和 pipeline 设计
足够大的 tile 才能体现价值
```

初学阶段你不需要手写 TMA，但要知道它处在优化阶梯中的位置：

```text
普通 tiled load
  -> cp.async 双缓冲
    -> TMA 大块异步搬运
```

## 5. Thread Block Cluster：多个 block 可以被组织成一个 cluster

传统 CUDA 模型里：

```text
thread 组成 block
block 组成 grid
```

普通情况下：

```text
block 内可以 __syncthreads()
block 间不能直接同步
block 间不能直接共享 shared memory
```

Hopper 引入 Thread Block Cluster 后，CUDA 模型多了一层：

```text
thread -> block -> cluster -> grid
```

Cluster 的意义：

```text
一个 cluster 中的多个 block 可以被保证协同调度。
cluster 内 block 可以进行 cluster 级同步。
cluster 内可以使用 Distributed Shared Memory。
```

这让一些过去必须拆成多个 kernel 或走 global memory 的算法，有机会在一个 kernel 内完成更多协作。

## 6. DSM：Distributed Shared Memory

DSM 是 Distributed Shared Memory。

先别把它理解成“一个巨大 shared memory”。更准确地说：

```text
每个 block 仍然有自己的 shared memory。
在 cluster 内，其他 block 可以访问这个 block 的 shared memory。
```

所以：

```text
block A 的 shared
block B 的 shared
block C 的 shared

在同一个 cluster 内可以形成一个分布式共享地址空间。
```

它适合的场景：

```text
block 间需要共享中间结果
tile 边界需要交换数据
histogram / stencil / certain reductions
某些大 tile 放不进单个 block shared，但可拆到 cluster
```

面试回答：

```text
DSM 让 cluster 内 block 可以互访 shared memory，
突破传统“block 间不能直接共享 shared”的限制，
但它仍然是受硬件和 cluster 调度约束的高级特性，不是普通全局同步的替代品。
```

## 7. WGMMA：Warp Group 级矩阵乘

早期 Tensor Core 编程常听到：

```text
WMMA
MMA
```

Hopper 之后你还会看到：

```text
WGMMA
```

可以先这样理解：

```text
MMA:
  warp 级或更细粒度的矩阵乘加指令。

WGMMA:
  warp group 级矩阵乘加。
  多个 warp 共同完成更大的矩阵运算。
```

为什么需要 warp group？

```text
现代 Tensor Core 吞吐极高。
单个 warp 组织的数据和指令规模不一定足够高效。
warp group 可以协调更多线程、更多数据，喂饱更强的 Tensor Core。
```

如果你读 CUTLASS、FlashAttention、Tensor Core GEMM，会看到这些层次：

```text
threadblock tile
warp group tile
warp tile
mma tile
```

WGMMA 就在这种层次化 GEMM 设计中发挥作用。

## 8. Hopper 的三个关键词如何连起来

Hopper 高性能 GEMM / attention 的简化故事：

```text
1. TMA 把大块 A/B tile 异步搬到 shared。
2. WGMMA 用 Tensor Core 做大吞吐矩阵乘加。
3. Cluster/DSM 让多个 block 在更大范围内协作。
```

这不是说每个 kernel 都必须用三个特性。

而是：

```text
TMA 解决数据怎么高效进 shared。
WGMMA 解决 Tensor Core 怎么高效计算。
Cluster/DSM 解决 block 之间怎么在受控范围内协作。
```

## 9. Blackwell：延续模型，增强 AI 和系统能力

Blackwell 不是让 CUDA 编程模型全部推倒重来。

官方调优指南强调：

```text
Blackwell 保留并扩展 Ampere/Hopper 的 CUDA 编程模型。
遵循前代最佳实践的程序通常能在 Blackwell 上获得加速。
```

对 CUDA 程序员最重要的公开点：

```text
1. 数据中心 Blackwell B200/GB200 是 compute capability 10.0。
2. Blackwell Ultra B300/GB300 是 compute capability 10.3。
3. RTX Blackwell 工作站/消费级属于 compute capability 12.x。
4. B200 支持更大的 shared/L1 carveout 选项和更强的内存系统。
5. Blackwell 继续强化 Tensor Core、低精度和 NVLink。
```

注意：

```text
不同 Blackwell 产品线的 compute capability 不一样。
不要只说“Blackwell 是 10.x”就结束。
面试里最好说：B200/GB200 是 10.0，B300/GB300 是 10.3，RTX Blackwell 是 12.x。
```

## 10. 面试如何回答“讲讲 Hopper 和 Blackwell”

推荐回答：

```text
Hopper 的核心不是只堆更多 SM，而是强化了 AI 计算和数据移动。
计算上有第四代 Tensor Core、FP8 Transformer Engine、WGMMA。
数据移动上有 TMA，把大块多维 global/shared transfer 硬件化。
编程模型上有 Thread Block Cluster 和 DSM，让 block 间可以在 cluster 内协作。

Blackwell 延续 CUDA 模型，继续强化低精度 Tensor Core、内存层次和 NVLink。
对 AI infra 来说，Blackwell 更像系统级平台升级，包括更强多卡互连和更低精度推理/训练能力。
```

如果想更工程化：

```text
写 kernel 时：
  关注 compute capability、是否能用 TMA/WGMMA/cluster。

做 AI infra 时：
  关注 HBM 容量/带宽、NVLink/NVSwitch、MIG、NCCL topology、功耗和可靠性。
```

## 11. 常见误区

### 误区一：新架构一定要手写新指令

不一定。很多情况下：

```text
cuBLAS / cuDNN / CUTLASS / TensorRT-LLM 已经帮你用上新特性。
```

你需要懂原理，但不一定每次都手写。

### 误区二：TMA 可以替代所有 coalescing 优化

不对。数据布局仍然重要。

```text
TMA 是高级搬运能力，不是坏 layout 的万能药。
```

### 误区三：Cluster 等于全局同步

不对。

```text
Cluster 只保证 cluster 内 block 的协作。
它不是整个 grid 的任意全局同步。
```

## 12. 实践

1. 查你的当前 GPU 的 compute capability，列出它是否支持 `cp.async`、TMA、cluster。
2. 阅读 CUDA Programming Guide 的 Thread Block Cluster 小节，画出 `thread -> block -> cluster -> grid`。
3. 找一段 CUTLASS Hopper GEMM 文档或代码，只标记 TMA/WGMMA 相关名词，不要求看懂所有模板。
4. 准备 2 分钟口述：“Hopper 相比 Ampere 的关键变化是什么？”

## 13. 面试题

**Q1：TMA 相比 cp.async 解决了什么？**

`cp.async` 是线程发起的 global 到 shared 异步拷贝，适合 Ampere 上的 tiled pipeline。TMA 是 Hopper
的硬件搬运引擎，面向更大的多维 tile transfer，减少线程级搬运指令和复杂索引，更适合高性能 GEMM
和 attention。

**Q2：Thread Block Cluster 和 DSM 是什么？**

Cluster 是 Hopper 引入的 block 之上的可选层级，同一 cluster 内的 block 可以协同调度和同步。
DSM 让 cluster 内 block 可以访问彼此的 shared memory，突破普通 CUDA 中 block 间 shared 不互通
的限制。

**Q3：WGMMA 和 Tensor Core 有什么关系？**

WGMMA 是面向 Tensor Core 的 warp group 级矩阵乘加指令/机制。它让多个 warp 共同组织更大的矩阵
运算，用来更好地喂饱 Hopper 及之后架构的 Tensor Core。

**Q4：Blackwell 面试要抓哪几个点？**

抓 CUDA 模型延续、compute capability 分化、低精度 Tensor Core、FP4/NVFP4、HBM/L2/shared
增强、NVLink/NVL72 系统能力。不要只背 SM 数量。

## 14. 资料映射

- CUDA Programming Guide：Thread Block Clusters、Distributed Shared Memory、异步拷贝、Compute Capabilities。
- NVIDIA Hopper 架构白皮书：Transformer Engine、TMA、cluster。
- NVIDIA Blackwell Tuning Guide：Blackwell SM、occupancy、cluster、memory system、NVLink。
- NVIDIA CUDA GPU Compute Capability 表：B200/GB200、B300/GB300、RTX Blackwell 的 CC。
