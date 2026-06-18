# 07 读规格表、Compute Capability、白皮书与 SASS

## 0. 先建立大局观：学会"自己查"，而非死记

GPU 型号、架构特性年年更新，没人能记住所有参数。真正的能力是**会读官方资料**——给你一块没
见过的卡，你能从规格表、Compute Capability、白皮书读出它的关键能力，从 SASS 读出代码到底
怎么执行。本章教你这套"自己查"的方法，是整个硬件卷的收口。

```text
四种官方信息源，各回答不同问题：
  规格表        -> 这块卡有多强（SM 数、带宽、算力、TDP）
  Compute Capability -> 它支持哪些特性（能不能用某个新功能）
  白皮书        -> 架构细节和设计理念（深入理解）
  SASS          -> 代码到底编成了什么机器指令（极致优化）
```

## 0.1 术语速查表

| 信息源 | 回答什么 | 怎么获取 |
|---|---|---|
| **规格表** | 卡的硬件参数 | 官网产品页 / `deviceQuery` |
| **Compute Capability** | 支持哪些 CUDA 特性 | `cudaGetDeviceProperties` |
| **白皮书** | 架构设计细节 | NVIDIA 官网各架构 whitepaper |
| **SASS** | 真实机器指令 | `cuobjdump --dump-sass` |
| **deviceQuery** | 运行时查卡参数 | CUDA Samples / 自己写 |

## 1. 读规格表：抓关键几个数

面对一块卡的规格表，**先看这几个对计算最关键的数**：

```text
SM 数量：        并行能力的核心（卷九/01）—— multiProcessorCount
FP32 算力(TFLOPS)：峰值计算能力（CUDA Core × 频率 × 2）
显存容量(GB)：    能放多大数据（卷十/09 显存预算）
显存带宽(GB/s)：  访存上限（Roofline 用，卷五/02）
显存类型：        GDDR vs HBM（卷九/04）
Tensor 算力：     AI 性能（卷九/03）
TDP(W)：          功耗/持续性能（卷九/05）
Compute Capability：支持的特性（下一节）
```

例：T4 的关键数（记住量级，卷五/卷十用）：

```text
SM: 40   FP32: 8.1 TFLOPS   显存: 16GB GDDR6   带宽: 320 GB/s
TDP: 70W   CC: 7.5
```

> 读规格表的目的不是背数字，而是**快速判断这卡适合干什么、瓶颈在哪**。如带宽 320、算力 8.1
> → 拐点约 25（卷五/02）→ 大多数 kernel 会 memory-bound。

## 2. Compute Capability：能力的版本号

**Compute Capability (CC)** 是 GPU 能力的版本号（卷二/06、卷十/05 反复用）：

```text
CC = major.minor，如 7.5（Turing）、8.0（Ampere）、9.0（Hopper）
- 决定支持哪些特性（cp.async 要 8.0+、TMA 要 9.0）
- 决定编译目标（sm_75 等，卷二/06）
- 数字越大通常特性越多
```

运行时查（卷二/06 §5.1 强调不能用 `__CUDA_ARCH__`）：

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("%s: CC %d.%d, %d SMs, %.1f GB, %.0f GB/s\n",
       prop.name, prop.major, prop.minor,
       prop.multiProcessorCount,
       prop.totalGlobalMem / 1e9,
       2.0 * prop.memoryClockRate * (prop.memoryBusWidth/8) / 1e6);
```

CC 表告诉你某个特性要哪个版本——查"我想用的功能，这卡支不支持"就靠它。

## 3. deviceQuery：运行时把卡看个透

CUDA Samples 有个 `deviceQuery`，或自己用 `cudaGetDeviceProperties` 打印一切。关键字段：

```text
multiProcessorCount       SM 数量
maxThreadsPerMultiProcessor  每 SM 最大线程（/32 = 最大 warp，算 occupancy）
sharedMemPerBlock / PerMultiprocessor  shared 容量
regsPerBlock              寄存器数
warpSize                  32（基本不变）
memoryBusWidth / Clock    算显存带宽
l2CacheSize               L2 大小（卷九/04）
major / minor             Compute Capability
```

> 这是你做任何性能分析的起点——知道卡的资源上限，才能判断 occupancy、显存预算等（卷五/03、
> 卷十/09）。你卷一就跑过 deviceQuery，现在能看懂每个字段的硬件含义了。

## 4. 白皮书：深入架构细节

每代架构 NVIDIA 都出**白皮书（whitepaper）**，讲 SM 结构、新特性、设计理念。什么时候读：

```text
- 想深入理解某代架构（如 SM 内部到底怎么组织）
- 想知道新特性的设计动机（如 TMA 为什么这样设计）
- 准备架构相关的深入面试
```

读白皮书的方法：

```text
- 不用通读，按需查（想懂 Tensor Core 就读那一节）
- 关注"架构图"（SM 框图）和"新特性"章节
- 配合本卷的硬件心智模型读，事半功倍
```

> 提醒（卷九 README）：白皮书是官方权威，但它给的是**简化的教学框图**，不是完整硬件布局。
> 别把简化图当成精确的物理结构。

## 5. 读 SASS：代码到底怎么执行

最底层的信息源——**SASS（真实机器指令，卷二/06）**。什么时候要读到这一层：

```text
- 极致优化：想确认编译器到底生成了什么（有没有用 FMA、有没有 spill）
- 排查诡异性能：PTX 看着对但慢，可能 SASS 层有问题
- 验证假设：编译器有没有按你想的优化
```

怎么看（卷二/06、卷五/05）：

```bash
nvcc -arch=sm_75 -lineinfo kernel.cu -o kernel   # -lineinfo 关联源码行
cuobjdump --dump-sass kernel                      # dump SASS
nvcc -Xptxas=-v ...                               # 看寄存器/spill 报告（卷二/06 §7）
```

读 SASS 看什么：

```text
FADD/FMUL/FFMA：浮点运算（FFMA = FMA，卷八/01）
LDG/STG：       global load/store（合并好不好影响事务数）
LDS/STS：       shared load/store
BAR.SYNC：      __syncthreads()
寄存器 R0,R1... 用了多少（对应 occupancy）
```

> 读 SASS 是进阶技能，不必常用。但会读它意味着你能"看穿"代码到硬件的最后一跳，这是资深
> 工程师和初学者的区别之一。

## 6. 一个完整的"查一块新卡"流程

给你一块没见过的 GPU，按这个流程摸清它：

```text
1. deviceQuery / 规格表：SM 数、算力、显存、带宽、CC、TDP
2. 算 Roofline 拐点（算力/带宽）：判断你的 kernel 会 memory 还是 compute bound（卷五/02）
3. 查 CC 对应特性：能不能用 Tensor Core/cp.async/TMA（卷九/06）
4. 算资源上限：每 SM 寄存器/shared/最大 warp -> occupancy 分析（卷五/03）
5. 需要时读白皮书深入、读 SASS 验证
```

## 7. 实践

1. 写一个完整的 deviceQuery，打印第 3 节所有关键字段，在你的 T4 上跑。
2. 用查到的算力和带宽算 T4 的 Roofline 拐点，验证约等于 25（卷五/02）。
3. 对你的一个 kernel `cuobjdump --dump-sass`，找出 FFMA、LDG、BAR.SYNC 指令。
4. 用第 6 节流程，假设拿到一块 A100，列出你会查的项和预期结论。

## 8. 面试题（附参考答案）

**Q1：拿到一块没见过的 GPU，你怎么快速了解它？**
deviceQuery/规格表看 SM 数、算力、显存、带宽、CC、TDP；算 Roofline 拐点判断 bound 类型；查 CC
对应特性；算资源上限做 occupancy 分析；需要时读白皮书/SASS。

**Q2：Compute Capability 是什么，怎么查？**
GPU 能力的版本号（如 7.5），决定支持哪些特性和编译目标。用 `cudaGetDeviceProperties` 读
`major/minor`（不能用编译期的 `__CUDA_ARCH__`）。

**Q3：什么时候需要读 SASS？**
极致优化时确认编译器实际生成了什么（FMA？spill？）、排查 PTX 看着对但慢的问题、验证编译器
有没有按预期优化。用 `cuobjdump --dump-sass`。

**Q4：规格表上对计算最重要的几个数是什么？**
SM 数（并行能力）、FP32 算力、显存容量（数据预算）、显存带宽（访存上限/Roofline）、CC（特性）、
TDP（持续性能）。

**Q5：白皮书的架构图能当精确硬件布局吗？**
不能。白皮书给的是简化的教学框图，便于理解，不是完整物理布局。用它建立心智模型，别当精确
电路图。

## 9. 资料映射

- NVIDIA 各架构白皮书；GPU 规格表；CUDA Samples deviceQuery。
- CUDA Programming Guide：Compute Capabilities；cuobjdump 文档。
- 配套：[卷二第 06 章 NVCC/PTX/SASS](../volume02_programming_model/06_NVCC_PTX与编译流程.md)、[卷五第 02 章 Roofline](../volume05_performance/02_性能指标_Scaling与Roofline.md)、[卷一第 05 章 T4 设备观察](../volume01_gpu_basics/05_T4设备观察实验.md)。

---

> **卷九完成，整个教材十卷全部完成。** 从"GPU 为什么和 CPU 不同"到"看穿芯片内部结构、会读
> 一切官方资料"——你已经走完了从入门到精通的完整路径。软件优化和硬件结构在你脑中打通，剩下
> 的是不断实践、做项目、上面试。祝你成为优秀的 CUDA 工程师。
