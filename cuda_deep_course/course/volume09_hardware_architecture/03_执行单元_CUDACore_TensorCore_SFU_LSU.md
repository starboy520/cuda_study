# 03 执行单元：CUDA Core、Tensor Core、SFU、LSU

## 0. 先建立大局观：SM 里不止一种"工人"

卷九/01 说 SM 是生产线。但生产线上的工人**分工种**——有的算普通加减乘（CUDA Core），有的专
做矩阵运算（Tensor Core），有的算超越函数（SFU），有的负责搬数据（LSU）。理解这些单元的
分工，能解释"为什么某些运算特别快/慢"、"Tensor Core 凭什么快几十倍"。

```text
SM 里的执行单元（按工种）：
  CUDA Core    -> 普通 FP32/INT 标量运算（最通用）
  Tensor Core  -> 矩阵乘加（AI 算力的来源）
  SFU          -> 特殊函数：sin/cos/exp/sqrt/倒数
  Load/Store   -> 访存单元（读写内存）
```

## 0.1 术语速查表

| 单元 | 全称 | 干什么 |
|---|---|---|
| **CUDA Core** | — | FP32/INT 标量算术（a+b, a*b）|
| **Tensor Core** | — | 小矩阵乘加（D = A×B + C），一拍做一堆乘加 |
| **SFU** | Special Function Unit | sin/cos/exp/log/sqrt/rsqrt 等超越函数 |
| **LSU** | Load/Store Unit | 内存读写访问 |
| **FP64 单元** | — | 双精度运算（数量通常远少于 FP32）|

## 1. CUDA Core：最通用的算术单元

CUDA Core 是 SM 里数量最多的执行单元，做最常见的**标量运算**：

```text
FP32 加、减、乘、FMA（融合乘加，卷八/01）
INT32 整数运算
这是你大部分 kernel 代码落到的地方
```

一个 SM 有几十到上百个 CUDA Core。**FP32 峰值算力 ≈ CUDA Core 数 × 频率 × 2（FMA 算 2 FLOP）**。
这就是规格表上"X TFLOP/s FP32"的来源。

> FP64（双精度）单元通常**远少于** FP32（卷八/02）：消费级卡 FP64:FP32 可能 1:32 或更低。所以
> 大量用 double 在这些卡上很慢——物理上就没几个 FP64 单元。

## 2. Tensor Core：AI 算力的来源

Tensor Core 是为**矩阵乘加**专门设计的单元，从 Volta 架构引入。它一条指令做一个**小矩阵的
乘加**（如 4×4×4），相当于一拍完成几十个乘加：

```text
普通 CUDA Core：一拍做 1 个乘加（标量）
Tensor Core：   一拍做一个小矩阵乘加（D = A×B + C，几十个乘加）
-> 矩阵运算吞吐高出 CUDA Core 一个数量级
```

为什么 AI 这么依赖它：深度学习的核心是 GEMM（卷六），Tensor Core 把 GEMM 加速几十倍。这就是
"AI 算力"（如 H100 的几千 TFLOPS）主要来自 Tensor Core 而非 CUDA Core。

代价与限制：

```text
- 主要支持低精度（FP16/BF16/INT8/FP8），FP32 用得少（精度 vs 速度，卷八/02）
- 要用专门接口：WMMA API、或 cuBLAS/CUTLASS（它们底层用 Tensor Core，卷六/03）
- 数据要满足特定形状和对齐
- T4（Turing）有 Tensor Core，但你写普通 FP32 kernel 用不到它
```

> 实战：你手写的 FP32 GEMM 用的是 CUDA Core；要用 Tensor Core 得走 WMMA 或直接调 cuBLAS。
> 这是手写 GEMM 难超 cuBLAS 的原因之一（卷六/03、卷八/03）。

## 3. SFU：特殊函数单元

SFU 算**超越函数**——这些函数用普通加乘算很慢，硬件专门加速：

```text
SFU 处理：sin, cos, exp, log, sqrt, rsqrt（倒数平方根）, 倒数
特点：每 SM 的 SFU 数量少（远少于 CUDA Core）-> 吞吐低
```

启示：

```text
- 大量用 sin/cos/exp/sqrt 的 kernel 可能被 SFU 吞吐限制
- CUDA 提供快速近似版（__sinf, __expf, __frcp_rn 等），精度低但更快
  普通版 vs 快速版是精度/速度权衡（卷八/01）
- softmax 的 exp（卷六/04）就走 SFU，大量 exp 时要注意
```

## 4. Load/Store Unit：访存的执行单元

LSU 负责**内存读写**——你的 `data[i]`、`tile[...]` 这些访存指令由它执行：

```text
LSU 处理：global / shared / local memory 的 load 和 store
它和内存系统（L1/L2/显存，卷九/04）打交道
```

LSU 的吞吐和内存系统决定了访存性能。**合并访问（卷三/02）的硬件意义**就在这里：一个 warp 的
32 个访问如果连续，LSU + 内存系统能用最少的事务完成；不合并就要多次事务，LSU 和内存带宽都
被浪费。

```text
合并访问  -> LSU 一次事务服务整个 warp -> 高效
跨步访问  -> LSU 要发多次事务 -> 访存吞吐暴跌（卷三/02 的 1/8 利用率）
```

## 5. 执行单元配比决定"什么运算快"

不同执行单元的**数量配比**，决定了不同运算的相对吞吐：

```text
FP32 加乘：    CUDA Core 多 -> 快
矩阵运算：     Tensor Core -> 极快（有的话）
sin/exp/sqrt： SFU 少 -> 相对慢
FP64：         单元少 -> 慢（消费级卡）
访存：         看 LSU + 内存带宽
```

这解释了一个现象：**两个 FLOP 数相同的 kernel，性能可能差很多**——一个全是 FP32 FMA（CUDA
Core 吃得下），另一个全是 exp（SFU 少，成瓶颈）。所以"算力"不是单一数字，要看**用的是哪种
执行单元**。

> 用 ncu 的 pipe utilization（各执行管线利用率）能看出 kernel 被哪个单元限制（卷五/05）。

## 6. 实践

1. 查你的 GPU 规格：每 SM 的 CUDA Core 数、有没有 Tensor Core、FP64 比例。
2. 写两个 FLOP 数相近的 kernel：一个纯 FP32 FMA、一个纯 expf，对比性能，体会 SFU 瓶颈。
3. 对比 `expf` 和 `__expf`（快速版）的性能和精度差异。
4. 解释：为什么 H100 的"AI 算力"几千 TFLOPS 远高于它的 FP32 算力？

## 7. 面试题（附参考答案）

**Q1：SM 里有哪些执行单元，各干什么？**
CUDA Core（FP32/INT 标量运算）、Tensor Core（矩阵乘加，AI 算力来源）、SFU（sin/cos/exp/sqrt
等超越函数）、Load/Store Unit（内存访问）。还有数量较少的 FP64 单元。

**Q2：Tensor Core 为什么快？有什么限制？**
它一条指令做一个小矩阵乘加（几十个乘加），矩阵吞吐比 CUDA Core 高一个数量级。限制：主要支持
低精度（FP16/BF16/INT8）、要走 WMMA/cuBLAS、数据要特定形状。AI 算力主要来自它。

**Q3：为什么大量用 double 在很多 GPU 上很慢？**
消费级/部分卡的 FP64 执行单元数量远少于 FP32（如 1:32），物理上就没几个 FP64 单元，所以 double
吞吐低。常用混合精度规避（卷八/02）。

**Q4：为什么 FLOP 数相同的两个 kernel 性能可能差很多？**
取决于用哪种执行单元。全是 FP32 FMA 的 kernel（CUDA Core 多）快；全是 exp 的（SFU 少）慢。
"算力"不是单一数字，要看占用哪个管线。

**Q5：合并访问的硬件意义是什么？**
LSU + 内存系统对一个 warp 的连续访问能用最少事务完成；跨步访问要多次事务，浪费 LSU 吞吐和
内存带宽。这是卷三合并访问优化的硬件根源。

## 8. 资料映射

- NVIDIA 架构白皮书（SM 执行单元部分）。
- CUDA Programming Guide：Arithmetic Instructions、Tensor Cores。
- 配套：[卷六第 03 章 Tensor Core](../volume06_operators/03_向量化_双缓冲与Tensor_Core.md)、[卷三第 02 章 合并访问](../volume03_memory_system/02_合并访问_对齐_AoS与SoA.md)。
