# cuBLAS 函数速查

> cuBLAS 是完整的 BLAS（Basic Linear Algebra Subprograms）GPU 实现，有几百个函数。
> 不用背，理解命名规律 + 记住常用几个即可。

## 命名规律（看懂就不怕多）

```text
cublas [精度] [操作] [变体]
        ↓      ↓      ↓
        S/D/C/Z  gemm/gemv/dot...  _v2 / _64
```

### 精度前缀（第一个字母）
```text
S = Single（FP32）      cublasSgemm
D = Double（FP64）      cublasDgemm
C = Complex（复数 FP32） cublasCgemm
Z = Complex 双精度       cublasZgemm
→ Sgemm/Dgemm/Cgemm/Zgemm = 同一操作的 4 种精度
```

## BLAS 三个等级（按操作复杂度）

```text
Level 1（向量-向量）：
  dot(点积)、axpy(y=ax+y)、scal(缩放)、nrm2(范数)、
  Iamax(最大绝对值下标)、asum(绝对值和)
Level 2（矩阵-向量）：
  gemv(矩阵×向量)、ger(外积)、trmv(三角矩阵×向量)、symv(对称)
Level 3（矩阵-矩阵）：
  gemm(矩阵乘)、symm(对称)、trsm(三角求解)、syrk、trmm
```

## 你实际会用到的（记这几个）

| 函数 | 作用 | 备注 |
|------|------|------|
| `cublasSgemm` | FP32 矩阵乘 | 你已用，仅 FP32 |
| `cublasGemmEx` | 通用 GEMM，混合精度 | ⭐ FP16/BF16/INT8 输入 + FP32 累加，对照用这个 |
| `cublasSgemv` | 矩阵 × 向量 | decode 阶段 GEMV |
| `cublasSdot` | 点积 | |
| `cublasIsamax` | 最大绝对值下标 | 对照你的 argmax |
| `cublasSaxpy` | y = a·x + y | |

## GemmEx：混合精度的关键

```text
cublasSgemm   只能 FP32
cublasGemmEx  能指定输入/累加/输出类型：
  数据类型：
    CUDA_R_32F  = FP32
    CUDA_R_16F  = FP16
    CUDA_R_16BF = BF16
    CUDA_R_8I   = INT8
  计算类型（累加）：
    CUBLAS_COMPUTE_32F      = FP32 累加
    CUBLAS_COMPUTE_32F_FAST_TF32 = TF32 Tensor Core
→ 对照 FP16/BF16/INT8 都用 GemmEx，不是 Sgemm
```

## 后缀含义

```text
_v2 : 新 API（用句柄 cublasHandle_t、参数传指针）
      头文件的 #define cublasSgemm cublasSgemm_v2 自动映射，你不用管
_64 : 支持 64 位整数维度（超大矩阵，维度 > 21 亿）
      普通用 32 位版即可
```

## 为什么函数这么多

```text
BLAS 是几十年的标准（Fortran 时代），覆盖所有线性代数基础操作：
  向量运算 × 矩阵向量 × 矩阵矩阵，每个 × 4 种精度 × 各种变体
cuBLAS 是 BLAS 的 GPU 实现，继承了这一大堆函数。
但实际只用少数几个（gemm/gemv/dot/Iamax）。
```

## 基本用法（5 步）

```cpp
#include <cublas_v2.h>
cublasHandle_t handle;
cublasCreate(&handle);
float alpha = 1.0f, beta = 0.0f;
// 行优先 C=A*B 的技巧：交换 A/B，传 N,M,K
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
cublasDestroy(handle);
// 编译：nvcc x.cu -o x -lcublas
```

## 一句话

```text
cuBLAS = 完整 BLAS 库，函数 = 操作(gemm/gemv/dot) × 精度(S/D/C/Z) × 变体。
命名：cublas[精度][操作][变体]。
实际只用几个：Sgemm(FP32) / GemmEx(混合精度) / Sgemv / Sdot / Isamax。
混合精度对照用 cublasGemmEx（能指定类型），不是 Sgemm。
_v2/_64 是新API/64位维度，不用管。用时查文档，不背。
```
