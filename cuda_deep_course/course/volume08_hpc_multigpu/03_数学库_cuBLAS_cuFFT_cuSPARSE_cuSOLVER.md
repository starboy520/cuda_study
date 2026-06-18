# 03 数学库：cuBLAS / cuFFT / cuSPARSE / cuSOLVER

## 0. 先建立大局观：不要重复造工业轮子

前几卷你手写了 GEMM、reduction、卷积——**为了理解原理**。但工程中，标准数值运算几乎都有
NVIDIA 官方高度优化的库，它们由专家针对每代架构调到接近峰值。本章讲清这些库各管什么、何时
该用库而非手写。

```text
学习时：手写，为了懂原理、能定制、面试能讲
工程时：标准算子优先用库（接近峰值、省时间、少 bug），库没有的才手写
```

这正是卷六/06"库的边界"在 HPC 数值库上的展开。

## 0.1 四大数学库速查表

| 库 | 管什么 | 类比 |
|---|---|---|
| **cuBLAS** | 稠密线性代数（矩阵/向量运算，GEMM 是核心）| GPU 版 BLAS |
| **cuFFT** | 快速傅里叶变换（FFT）| GPU 版 FFTW |
| **cuSPARSE** | 稀疏矩阵运算（SpMV、稀疏 GEMM）| GPU 版稀疏 BLAS |
| **cuSOLVER** | 线性方程组、特征值、矩阵分解（LU/QR/SVD）| GPU 版 LAPACK |

> 记忆：**BLAS 管"算"（乘加），SOLVER 管"解"（方程/分解），FFT 管"变换"，SPARSE 管"稀疏"**。

## 1. cuBLAS：稠密线性代数

最常用的库，核心是 GEMM。三层 BLAS：

```text
Level 1：向量-向量（如 axpy: y = a*x + y、dot 点积）
Level 2：矩阵-向量（如 gemv: y = A*x）
Level 3：矩阵-矩阵（如 gemm: C = A*B）—— 计算密集，GPU 最擅长
```

基本用法（注意**列主序**，这是最大的坑）：

```cpp
cublasHandle_t handle;
cublasCreate(&handle);
float alpha = 1.0f, beta = 0.0f;
// C = alpha * A * B + beta * C
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
cublasDestroy(handle);
```

> **列主序陷阱**：cuBLAS 源自 Fortran，矩阵是**列主序**。而 C/C++ 数组是行主序。直接传行主序
> 数据会算出转置的结果。常见处理：把"行主序的 A×B"转化为"列主序的 Bᵀ×Aᵀ"，或交换参数顺序。
> 这是用 cuBLAS 第一个要踩的坑（卷十/Week4 也提醒过）。

为什么手写 GEMM 难超过 cuBLAS：它做了寄存器分块、向量化、双缓冲、按架构调参、甚至用 Tensor
Core——你手写版能到它 70-80% 已经很好（卷六）。

## 2. cuFFT：快速傅里叶变换

FFT 把信号在时域和频域间转换，是信号处理、图像、卷积加速的基础：

```text
用途：频谱分析、大 filter 卷积（频域相乘比时域卷积快）、求解 PDE
基本流程：
  cufftPlan1d/2d/3d(...)   创建变换计划（plan）
  cufftExecC2C(...)        执行变换（复数到复数）
  cufftDestroy(plan)       销毁
```

关键概念：**plan 复用**。创建 plan 有开销，同样尺寸的多次变换要复用 plan，别每次重建。

> 何时用：需要 FFT 时直接用 cuFFT，手写 FFT 极难调到它的性能。大 filter 卷积可以"FFT → 频域
> 相乘 → 逆 FFT"，比直接卷积快（卷四/05 的卷积在 filter 大时可考虑）。

## 3. cuSPARSE：稀疏矩阵

回忆卷四/05 的 SpMV——稀疏矩阵每行非零数不同、负载不均衡，手写难做好。cuSPARSE 内部按矩阵
特征选最优策略：

```text
用途：SpMV、稀疏矩阵乘、格式转换（COO/CSR/CSC 互转）
优势：内部针对不同稀疏模式选 thread-per-row / warp-per-row / load-balanced
何时用：有稀疏运算需求时优先，比手写省力且通常更快
```

> 卷四/05 手写 SpMV 是为了理解负载均衡难点；工程上稀疏运算用 cuSPARSE。

## 4. cuSOLVER：求解与分解

解决"解方程、分解矩阵"这类比单纯乘加更复杂的问题：

```text
用途：
  线性方程组 Ax=b（LU 分解）
  最小二乘（QR 分解）
  特征值 / 奇异值（SVD）
  Cholesky 分解（对称正定矩阵）
何时用：科学计算、优化、信号处理里需要解方程或分解时
```

这些算法数值上很微妙（涉及卷八/01 的稳定性、灾难性消减），自己实现极易出数值问题，**强烈
建议用库**。

## 5. 何时用库、何时手写（核心决策）

```text
用库：
  ✅ 标准算子（GEMM、FFT、解方程）—— 库接近峰值，自己很难超
  ✅ 数值敏感的算法（分解、特征值）—— 库处理了稳定性
  ✅ 想快速出原型 / 做正确性参考

手写：
  ✅ 库没有的算子（自定义融合、特殊形状）
  ✅ 需要和前后算子融合（库的边界是单算子，融合要自己写，卷六）
  ✅ 特殊精度 / 特殊硬件特性
  ✅ 学习理解原理
```

> 一句话（贯穿卷六/十）：**标准的用库，独特的手写**。面试时这是高频判断题——别显得"什么都
> 想自己写"，成熟工程师知道何时站在库的肩膀上。

## 6. 用库也要注意的事

```text
- handle/plan 要复用：创建有开销，别在循环里反复建销
- 数据布局：cuBLAS 列主序，传错算出转置
- 同步：库调用多是异步的，计时/取结果前要同步（卷二/05）
- 错误检查：库有自己的状态码（cublasStatus_t 等），也要检查
- 版本/精度：注意库对数据类型、对齐的要求
```

## 7. 实践

1. 用 cuBLAS `cublasSgemm` 跑 GEMM，先在小矩阵上解决列主序问题、验证结果对，再和你手写版
   对比性能（Week4 P2 项目）。
2. 故意把行主序数据直接传 cuBLAS，观察算出转置的错误结果，理解列主序陷阱。
3. 查 cuSPARSE 的 SpMV 接口，和你卷四/05 手写版对比。

## 8. 面试题（附参考答案）

**Q1：cuBLAS / cuFFT / cuSPARSE / cuSOLVER 各管什么？**
cuBLAS 稠密线性代数（GEMM 为核心）、cuFFT 快速傅里叶变换、cuSPARSE 稀疏矩阵运算、cuSOLVER
解方程和矩阵分解（LU/QR/SVD）。

**Q2：用 cuBLAS 最容易踩的坑是什么？**
列主序。cuBLAS 源自 Fortran 用列主序，C/C++ 行主序数据直接传会算出转置结果。要转置处理或
交换参数。

**Q3：什么时候该手写 kernel 而非用库？**
库没有的算子、需要算子融合（库边界是单算子）、特殊精度/形状、或学习理解时。标准算子（GEMM/
FFT/解方程）用库更快更稳。

**Q4：为什么手写 GEMM 很难超过 cuBLAS？**
cuBLAS 做了寄存器分块、向量化、双缓冲、按架构调参、Tensor Core，由专家针对每代硬件优化。手写
到它 70-80% 已很好。

**Q5：用库有哪些工程注意点？**
复用 handle/plan（创建有开销）、注意数据布局（列主序）、库调用异步要同步、检查库的状态码、
注意精度和对齐要求。

## 9. 资料映射

- cuBLAS / cuFFT / cuSPARSE / cuSOLVER 官方文档。
- 配套：[卷六第 06 章 库的边界与算子工程化](../volume06_operators/06_库的边界与算子工程化.md)、[卷四第 05 章 SpMV](../volume04_parallel_algorithms/05_Convolution_Stencil与SpMV.md)。
