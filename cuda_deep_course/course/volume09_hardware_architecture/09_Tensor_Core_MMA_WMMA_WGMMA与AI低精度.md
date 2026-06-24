# 09 Tensor Core、MMA、WMMA、WGMMA 与 AI 低精度

这一章把 AI 算力的硬件来源讲清楚。

> AI infra 面试里，不能只说“Tensor Core 很快”。你要能解释它为什么快、用什么数据类型、和
> CUDA Core 有什么区别、WMMA/MMA/WGMMA 这些词分别处在哪一层。

## 1. 先从普通 CUDA Core 开始

普通 CUDA Core 做的是标量运算：

```text
c = a * b + c
```

这是一条 FMA，通常算 2 FLOP：

```text
1 次乘法 + 1 次加法
```

如果你写普通 FP32 GEMM：

```cpp
sum += A[row * K + k] * B[k * N + col];
```

编译后主要落到 CUDA Core 的 FP32 FMA 管线。

## 2. Tensor Core 做什么

Tensor Core 做的是小矩阵乘加：

```text
D = A x B + C
```

不是一个数乘一个数，而是一小块矩阵乘一小块矩阵。

直觉：

```text
CUDA Core:
  一次做一个标量 FMA。

Tensor Core:
  一次做一个小矩阵 MMA，里面包含很多乘加。
```

所以深度学习里 GEMM、convolution、attention 投影这些大矩阵运算，能在 Tensor Core 上获得远高于
普通 FP32 CUDA Core 的吞吐。

## 3. 为什么 AI 算力数字那么夸张

规格表里你会看到：

```text
FP32 TFLOPS
Tensor TFLOPS
FP8 / FP4 Tensor TFLOPS
```

Tensor 算力通常远大于 FP32 算力，因为：

```text
1. Tensor Core 专门做矩阵乘加。
2. 使用低精度，单位面积能做更多运算。
3. 矩阵乘的规则性很强，硬件容易做成高吞吐。
```

面试回答：

```text
AI 算力主要来自 Tensor Core，而不是 CUDA Core。
低精度让吞吐继续提高，但需要数值范围、缩放和误差控制。
```

## 4. 低精度路线：FP32 -> TF32/BF16/FP16 -> FP8 -> FP4

训练和推理的硬件演进主线：

```text
FP32:
  范围和精度较好，但吞吐和带宽成本高。

TF32:
  Ampere 引入，面向 FP32 输入的 Tensor Core 加速路径。

FP16 / BF16:
  深度学习训练常用。BF16 指数范围接近 FP32，训练更稳。

FP8:
  Hopper Transformer Engine 的重点，适合大模型训练/推理。

FP4 / NVFP4:
  Blackwell 继续降低精度，主要面向更高吞吐和更大模型推理/训练。
```

注意：

```text
低精度不是随便把 float 砍小。
需要 scale、amax、量化策略、混合精度累加。
```

这就是 Transformer Engine 的价值：在低精度吞吐和模型精度之间做自动化管理。

## 5. WMMA、MMA、WGMMA 分别是什么

可以按抽象层级理解：

```text
cuBLAS / cuDNN / CUTLASS:
  库层。通常生产环境优先使用。

WMMA:
  CUDA C++ 提供的 warp-level matrix API。
  比较高层，使用 fragment 抽象。

MMA:
  更接近机器指令的矩阵乘加操作。
  PTX/SASS 层常见 mma.sync 等形式。

WGMMA:
  Hopper 之后的 warp group 级矩阵乘加。
  多个 warp 共同组织更大、更高吞吐的 Tensor Core 操作。
```

一句话：

```text
WMMA 是更高层的 API。
MMA 是更底层的矩阵指令。
WGMMA 是 warp group 级的现代 Tensor Core 路径。
```

## 6. Fragment 是什么

使用 WMMA 时常见：

```text
fragment
```

它表示：

```text
一个 warp 协同持有的一小块矩阵数据。
```

不是某一个 thread 拿完整矩阵。

更准确：

```text
一个 warp 中的多个 lane 各自持有 fragment 的一部分。
硬件和 API 定义了这些元素如何分布在 lane / register 中。
```

所以你不应该把 WMMA fragment 想成普通 C++ 二维数组。

## 7. Tensor Core 和普通 GEMM Tiling 的关系

普通 CUDA Core GEMM 优化：

```text
global -> shared -> register
thread/block 做 tiling
CUDA Core 做 FMA
```

Tensor Core GEMM 优化：

```text
global -> shared -> fragment/register
threadblock / warp / warp group 做 tiling
Tensor Core 做 MMA
```

共同点：

```text
都需要 tiling。
都需要数据复用。
都需要处理 global/shared/register 层次。
```

不同点：

```text
普通 GEMM 的内层是标量 FMA。
Tensor Core GEMM 的内层是 MMA/WGMMA。
```

## 8. 为什么 cuBLAS / CUTLASS 难超

高性能 Tensor Core GEMM 要同时做好：

```text
1. Threadblock tiling
2. Warp / warp group tiling
3. Shared memory layout 和 bank conflict
4. Global memory coalescing
5. cp.async / TMA pipeline
6. MMA/WGMMA 指令选择
7. epilogue：bias、activation、scale、quantization
8. 多架构 specialization
```

cuBLAS 和 CUTLASS 的优势就是：

```text
它们把这些层次都做了架构特化和大量调参。
```

所以工程上：

```text
标准 GEMM 用库。
学习和特殊融合算子才手写。
```

## 9. Tensor Core 和 AI Infra 的连接

AI infra 不是只写 kernel，还要理解系统吞吐。

Tensor Core 影响：

```text
训练吞吐
推理 token/s
batch size 和 latency tradeoff
低精度量化策略
显存带宽压力
通信/计算重叠空间
```

例如 LLM 推理：

```text
Prefill:
  大 GEMM 多，Tensor Core 利用率高。

Decode:
  batch 小时 GEMM 可能变瘦，memory / launch / KV cache 访问更关键。
```

因此面试里不要只说“Tensor Core 快”，还要说：

```text
能不能喂饱 Tensor Core，取决于 batch、shape、数据布局、低精度、内存和通信。
```

## 10. 怎么确认自己是否用上 Tensor Core

方法：

```text
1. 使用 cuBLAS/cuBLASLt/CUTLASS/TensorRT 时，看数据类型和 algo。
2. 用 Nsight Compute 看 Tensor Core pipe utilization。
3. 读 SASS/PTX，看是否出现 mma / wgmma 相关指令。
4. 看 GFLOPS 是否接近 Tensor Core 量级，而不是 FP32 CUDA Core 量级。
```

反例：

```text
你写了普通 FP32 for-loop GEMM，
即使 GPU 有 Tensor Core，也不会自动变成 Tensor Core GEMM。
```

## 11. 面试口述模板

可以这样回答：

```text
Tensor Core 是 NVIDIA GPU 中专门做小矩阵乘加的硬件单元，是 AI 算力的主要来源。
相比 CUDA Core 的标量 FMA，Tensor Core 一条 MMA 指令可以完成一小块矩阵乘加，
并且支持 FP16、BF16、TF32、FP8、FP4 等低精度格式。

软件上可以通过 cuBLAS、CUTLASS、WMMA 或更底层 MMA/WGMMA 使用它。
高性能 Tensor Core GEMM 仍然需要 tiling、shared memory、pipeline、layout 和低精度数值管理。
工程上标准 GEMM 一般用库，手写主要用于学习或特殊融合算子。
```

## 12. 常见面试题

**Q1：Tensor Core 和 CUDA Core 有什么区别？**

CUDA Core 做标量 FP32/INT 运算；Tensor Core 做小矩阵乘加，低精度吞吐极高，是 AI GEMM/卷积的
主要算力来源。

**Q2：WMMA 和 MMA 有什么区别？**

WMMA 是 CUDA C++ 层的 warp-level matrix API，使用 fragment 抽象；MMA 更接近 PTX/SASS 的矩阵
乘加指令。WGMMA 是 Hopper 之后的 warp group 级矩阵乘加路径。

**Q3：为什么低精度能提高 AI 性能？**

低精度数据更小，带宽和存储压力更低，硬件单位面积可以做更多乘加。但需要缩放、混合精度累加和
误差控制，否则会影响模型精度。

**Q4：怎么确认 kernel 用上 Tensor Core？**

用 Nsight Compute 看 Tensor Core pipe，或读 PTX/SASS 查 MMA/WGMMA 指令，也可以从性能是否达到
Tensor Core 量级判断。普通 FP32 for-loop GEMM 不会自动用 Tensor Core。

## 13. 实践

1. 在你的 T4 上查 Tensor Core 支持的数据类型。
2. 用 cuBLAS 跑 FP32、FP16 GEMM，对比吞吐差异。
3. 用 Nsight Compute 看 Tensor Core 利用率。
4. 对一个 CUTLASS GEMM 示例，只标记 threadblock tile、warp tile、MMA tile 三层。

## 14. 资料映射

- CUDA Programming Guide：Warp Matrix Functions、Tensor Cores、Compute Capabilities。
- NVIDIA Hopper / Blackwell 官方架构资料：Transformer Engine、FP8/FP4、WGMMA。
- CUTLASS 文档：threadblock / warp / instruction tile 层次。
- 配套：[卷六 Tensor Core](../volume06_operators/03_向量化_双缓冲与Tensor_Core.md)、[卷六 GEMM 外积视角](../volume06_operators/02B_GEMM_Register_Tiling外积视角.md)。
