# 01 GPU 逻辑层次：die、GPC、TPC、SM

## 0. 先建立大局观：GPU 是一座"分层的工厂"

你写 CUDA 时面对的是 grid/block/thread（软件层次）。但芯片内部是另一套**硬件层次**——一块
GPU 芯片（die）里有很多层结构，最终落到执行你 kernel 的 **SM**。理解这个硬件层次，能解释
很多软件行为的根源（为什么 block 不能跨 SM、为什么要很多 block）。

用工厂类比：

```text
GPU die  ≈ 整座工厂
GPC      ≈ 厂区（几个车间的集合）
TPC      ≈ 车间（含一两条生产线）
SM       ≈ 生产线（真正干活的地方，执行你的 kernel）
CUDA Core ≈ 生产线上的工人（执行单条运算）
```

## 0.1 术语速查表

| 术语 | 全称 | 一句话定义 |
|---|---|---|
| **die** | — | 一整块 GPU 芯片 |
| **GPC** | Graphics Processing Cluster | 一组 TPC 的集合（厂区）|
| **TPC** | Texture Processing Cluster | 含 1-2 个 SM（车间）|
| **SM** | Streaming Multiprocessor | GPU 的核心执行单元（生产线）|
| **CUDA Core** | — | SM 内执行标量运算的单元 |
| **SM 分区** | sub-partition | SM 内部再分的 4 个处理块 |

## 1. 从软件到硬件：两套层次的对应

回忆卷一：你的软件层次和硬件层次是这样映射的：

```text
软件（你写的）          硬件（芯片里的）         映射关系
─────────────────────────────────────────────────────
Grid                    整个 GPU                 一个 kernel 用整个 GPU
Block                   一个 SM                  一个 block 调度到一个 SM 上
Warp（32线程）          SM 的一个调度单位         scheduler 按 warp 发射指令
Thread                  CUDA Core（运算时）       thread 的运算落到 core 上
```

**关键**：block ↔ SM 是核心映射。一个 block 被分配到某个 SM 后，**整体待在那里直到结束**——
这就是卷一/卷四说的"block 不能跨 SM"的硬件根源（它的寄存器、shared、barrier 状态都物理地
存在那个 SM 里）。

## 2. die → GPC → TPC → SM：自顶向下

一块现代 GPU 芯片的逻辑层次：

```text
GPU die（整块芯片）
├── GPC 0（图形处理簇）
│   ├── TPC 0（纹理处理簇）
│   │   ├── SM 0   <- 真正执行 kernel
│   │   └── SM 1
│   ├── TPC 1
│   │   └── SM 2, SM 3
│   └── ...
├── GPC 1
│   └── ...
└── ...
共享：L2 cache、显存控制器、互连
```

各层职责：

```text
die：  整块芯片，包含所有计算资源 + L2 + 显存接口
GPC：  一组 TPC，是芯片的大分区（也含光栅化等图形单元，计算用得少）
TPC：  含 1~2 个 SM（不同架构数量不同）
SM：   核心！真正执行你的 kernel 的地方（下一节细看）
```

> 对计算（CUDA）而言，**SM 是最重要的层**。GPC/TPC 主要是芯片的组织/制造分区，图形渲染时更
> 重要；做通用计算时，你主要关心"有多少 SM、每个 SM 多强"。

## 3. SM 内部有什么

SM 是干活的核心，里面有（不同架构略有差异，这是通用结构）：

```text
一个 SM 内部：
├── 多个 CUDA Core      执行 FP32/INT 标量运算（卷九/03）
├── Tensor Core         矩阵运算加速（新架构有，卷九/03）
├── SFU                 特殊函数单元（sin/cos/exp/sqrt）
├── Load/Store Unit     访存单元
├── Warp Scheduler      决定下一拍发射哪个 warp 的指令（卷九/02）
├── 寄存器文件          所有驻留线程的寄存器（很大，卷九/04）
├── Shared Memory / L1  片上快速存储（卷九/04）
└── 通常分成 4 个 sub-partition（处理块），各有自己的调度器
```

**SM 内部再分 4 个 sub-partition（处理块）**，每个处理块有自己的 warp scheduler 和一批执行
单元。一个 warp 被分到某个处理块上调度。这解释了为什么 SM 能同时推进很多 warp。

## 4. 为什么这个结构解释了软件行为

把硬件层次和你学过的软件优化对应起来：

```text
"block 不能跨 SM"
  -> block 的资源（寄存器/shared/barrier）物理上在一个 SM 里，跨 SM 无法共享

"一个 SM 能驻留多个 block"
  -> SM 的寄存器文件和 shared memory 够大时，可同时容纳多个 block 的资源

"为什么要很多 block / 大 grid"
  -> 要喂满所有 SM（一块 GPU 几十个 SM），block 太少有 SM 闲置

"occupancy 受寄存器/shared 限制"
  -> 一个 SM 的寄存器文件和 shared 是固定的，被驻留线程瓜分（卷五/03）

"warp = 32"
  -> SM 的调度器以 warp 为单位发射指令（卷九/02）
```

## 5. SM 数量：GPU 算力的核心指标

不同 GPU 的主要差异之一就是 **SM 数量**：

```text
T4（Turing）：     40 个 SM
A100（Ampere）：   108 个 SM
H100（Hopper）：   132 个 SM
（具体数字以官方规格为准）
```

SM 越多，能并行的 block/warp 越多，峰值算力越高。这就是为什么查一块卡先看 SM 数（用
`cudaGetDeviceProperties` 的 `multiProcessorCount`）。

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("SM 数量: %d\n", prop.multiProcessorCount);
```

## 6. 实践

1. 用 `cudaGetDeviceProperties` 打印你的 GPU 的 SM 数量、每 SM 最大线程/warp 数。
2. 画出你的 GPU 的逻辑层次图（die→GPC→TPC→SM），SM 数从规格表查。
3. 解释：为什么一个只启动 4 个 block 的 kernel 在 40-SM 的 T4 上利用率很低？
4. 把你学过的 5 条软件优化规则，各自对应到本章的某个硬件结构。

## 7. 面试题（附参考答案）

**Q1：GPU 的硬件层次是怎样的？**
die（整块芯片）→ GPC（图形处理簇）→ TPC（含 1-2 个 SM）→ SM（核心执行单元）→ CUDA Core
（执行单元）。对计算而言 SM 是最重要的层。

**Q2：为什么一个 block 不能跨多个 SM？**
block 被调度到某个 SM 后，它的寄存器、shared memory、barrier 状态都物理地存在那个 SM 上，
跨 SM 无法共享这些资源，所以整体待在一个 SM 直到结束。

**Q3：一个 SM 能同时驻留多个 block 吗？**
能。SM 的寄存器文件和 shared memory 够大时可同时容纳多个 block 的资源。能容纳几个取决于每个
block 的资源用量（影响 occupancy）。

**Q4：SM 内部有哪些主要部件？**
CUDA Core（标量运算）、Tensor Core（矩阵）、SFU（特殊函数）、Load/Store Unit（访存）、Warp
Scheduler（调度）、寄存器文件、Shared Memory/L1。通常再分 4 个 sub-partition。

**Q5：为什么查一块 GPU 先看 SM 数量？**
SM 是核心执行单元，数量直接决定能并行多少 block/warp 和峰值算力。用
`cudaGetDeviceProperties` 的 `multiProcessorCount` 查。

## 8. 资料映射

- NVIDIA 各架构白皮书（Turing/Ampere/Hopper）；GPU 架构图。
- CUDA Programming Guide：Hardware Implementation。
- 配套：[卷一 GPU 基础](../volume01_gpu_basics/README.md)、[卷五第 03 章 Occupancy](../volume05_performance/03_Occupancy_分歧与延迟隐藏.md)。
