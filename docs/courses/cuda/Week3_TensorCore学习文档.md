# Week 3：Tensor Core 与混合精度——自包含学习手册

> 本周目标：不依赖其他学习文档，完成 Tensor Core 原理、混合精度、WMMA FP16 GEMM、正确性验证、benchmark、Nsight Compute 分析、FP8 scaling 和 DeepGEMM 入门。
>
> 硬件边界：T4（SM 7.5）完成 FP16 WMMA；A100（SM 8.0）额外观察 TF32、BF16；FP8 以原理和工程设计为主，T4/A100 不强行实现原生 FP8 Tensor Core kernel。

---

## 0. 怎么使用这份文档

### 0.1 前置知识

默认你已经掌握：

- CUDA thread、block、grid、warp；
- global/shared/register memory；
- naive GEMM、shared-memory tiling、register tiling；
- CUDA Event 计时，知道 `ncu` 是 Nsight Compute 命令行工具。

### 0.2 本周必须产出

```text
wmma_fp16_gemm.cu       独立重写的 WMMA GEMM
mixed_precision_table.md 精度对照笔记（也可直接填写本文模板）
tensor_core_profile.md  benchmark + NCU 证据 + 结论
fp8_scaling_notes.md    FP8 scaling 与 DeepGEMM 口述笔记
```

如果你只想维护一个文件，也可以把实验结果直接填入本文末尾的模板。

### 0.3 Demo 不是最终作业

本文给出完整 Demo，是为了避免你把时间耗在头文件、错误检查和计时样板上。正确训练方式是：

1. **读懂**：能解释每一行和每个 leading dimension；
2. **运行并修改**：改变矩阵值、规模、布局，观察正确性和性能；
3. **关掉答案重写**：只看第 7 节骨架，独立完成核心 kernel；
4. **测量并解释**：与 FP32/cuBLAS 对照，用 NCU 和指令证据说明是否用了 Tensor Core。

只复制、编译、得到 `PASS`，不算完成 Week 3。

---

## 1. 从 CUDA Core 到 Tensor Core

### 1.1 普通 GEMM 的内层

普通 FP32 GEMM 最内层是标量 FMA：

```cpp
acc += a * b;
```

一次 FMA 通常按一次乘法加一次加法计算，即 2 FLOP。shared-memory tiling 和 register tiling 改善了数据复用，但最终执行的仍主要是 CUDA Core 上的标量 FMA。

### 1.2 Tensor Core 做什么

Tensor Core 面向矩阵乘加：

```text
D = A × B + C
```

软件不是让某一个线程独自提交一块矩阵，而是让一个 warp 协作描述和执行 MMA（Matrix Multiply-Accumulate）。以常用 FP16 WMMA 形状 `m16n16k16` 为例，一次 warp 级操作更新一个 `16×16` 输出 tile，并沿 K 维消费 16 个元素。

要注意：

> `wmma::mma_sync` 是 CUDA C++ 的 warp 级 API。把它说成“一条固定的底层指令完成整个 16×16×16”只是入门直觉，不是严格的机器指令描述。编译器会按 GPU 架构把它降低为相应的 PTX/SASS 矩阵指令序列。

### 1.3 为什么吞吐高

Tensor Core 快，不是“算法少算了”，而是硬件针对规则的小矩阵乘加提供了更高吞吐：

1. 矩阵结构固定，硬件能并行安排大量乘加；
2. FP16/BF16/TF32/INT8 等较低精度降低存储和计算成本；
3. 一个 warp 以集体方式驱动专用矩阵管线；
4. 深度学习的 GEMM/卷积/投影具有足够规则性，适合这种硬件。

但“有 Tensor Core”不等于“kernel 一定快”。高性能 GEMM 仍需要：

```text
global memory
    ↓ 合并访问、向量化
shared memory
    ↓ 合适布局、避免 bank conflict、流水
fragment / registers
    ↓ warp-level MMA
Tensor Core
    ↓ epilogue / global store
```

如果矩阵很小、shape 不合适、访存喂不饱、启动开销占主导，实际性能仍会很低。

### 1.4 与前两周 GEMM 的关系

| 阶段 | 数据复用 | 计算核心 |
|---|---|---|
| naive GEMM | 几乎没有 | 每线程标量 FMA |
| shared tiling | block 内复用 | 标量 FMA |
| register tiling | thread 内复用 | 标量 FMA |
| Tensor Core GEMM | block/warp/instruction 多级 tiling | warp-level MMA |

Tensor Core 不是替代 tiling，而是替换高性能 GEMM 最内层的计算原语。

---

## 2. 混合精度：先把格式分清楚

### 2.1 核心对照表

下表中的最大有限值是常见近似值；FP8 数值按 NVIDIA 常用 E4M3/E5M2 语义理解。

| 格式 | 存储位 | 指数位 | 尾数字段位 | 最大有限值（约） | 机器 epsilon（约） | T4 Tensor Core | A100 Tensor Core | 常见累加 | 主要风险 |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| FP32 | 32 | 8 | 23 | `3.4e38` | `1.19e-7` | 普通 FP32 不走 TC | 可经 TF32 路径 | FP32 | 带宽和吞吐成本高 |
| TF32 | 输入/存储仍为 32；乘法有效精度约 19 位 | 8 | 10 | 接近 FP32 | `9.77e-4` 量级 | 不支持 | 支持 | FP32 | 精度低于完整 FP32 |
| FP16 | 16 | 5 | 10 | `65504` | `9.77e-4` | 支持 | 支持 | 常用 FP32 | 范围小，易上溢/下溢 |
| BF16 | 16 | 8 | 7 | `3.39e38` | `7.81e-3` | 不支持 | 支持 | 常用 FP32 | 精度比 FP16 粗，仍会舍入/溢出 |
| FP8 E4M3 | 8 | 4 | 3 | `448` | 很粗 | 不支持原生 FP8 TC | 不支持原生 FP8 TC | 通常更高精度 | 范围窄，需要 scaling |
| FP8 E5M2 | 8 | 5 | 2 | `57344` | 更粗 | 不支持原生 FP8 TC | 不支持原生 FP8 TC | 通常更高精度 | 精度很低，需要 scaling |
| INT8 | 8 | — | — | `127` | 量化步长决定 | 支持 | 支持 | INT32 | 需要 scale/zero-point，饱和 |

### 2.2 FP16 与 BF16

两者都是 16 位：

- FP16 尾数更多，单位范围内更精细，但指数位少，动态范围小；
- BF16 指数位与 FP32 相同，动态范围接近 FP32，因此训练中通常比 FP16 更不容易因范围不足而溢出；
- BF16 **并非不会溢出**，也不代表精度问题“无所谓”。它只有 7 位尾数字段，舍入更粗。

### 2.3 TF32 到底是什么

TF32 是 Ampere 引入的 Tensor Core 计算路径：输入和输出接口仍可使用 FP32 存储，乘法部分采用大致“FP32 指数范围 + 10 位尾数字段”的有效精度，累加常为 FP32。

必须把这两句话分开：

- 支持 TF32 的 cuBLAS/框架在相应 math mode 下，可以让 FP32 GEMM 使用 Tensor Core；
- 你自己写的普通 FP32 `for` 循环 kernel **不会自动变成 Tensor Core kernel**。

### 2.4 为什么常用 FP32 累加

若 K 很大，需要把许多乘积相加。即使每个乘积来自 FP16，累计误差也会随加法次数增加。FP32 accumulator 提供更大的范围和更多有效位，通常是性能、范围和精度之间的好折中。

但 FP32 累加不能恢复输入转成 FP16 时已经丢掉的信息，所以：

```text
FP16 input × FP16 input + FP32 accumulate
```

通常比纯 FP16 累加稳定，却不等价于完整 FP32 GEMM。

---

## 3. WMMA、MMA、WGMMA：不要混成一个词

| 层次 | 含义 | 本周要求 |
|---|---|---|
| cuBLAS/cuBLASLt/CUTLASS | 库和模板层，生产环境通常优先 | 会调用、会对照 |
| WMMA | CUDA C++ warp-level matrix API，使用 fragment 抽象 | 必须手写 |
| MMA | 更接近 PTX/SASS 的矩阵乘加指令族，如 `mma.sync` | 能辨认、理解层次 |
| WGMMA | Hopper 的 warp-group 级矩阵乘加，多 warp 协作 | 只理解概念 |

本周选 WMMA，是因为它把底层 lane/register 映射隐藏起来，能用较少代码看清 Tensor Core 的数据流。工业级 kernel 往往使用 CUTLASS/CuTe 或更底层的 MMA/WGMMA，而不只停留在 WMMA。

---

## 4. WMMA 编程模型与硬约束

### 4.1 fragment 是什么

```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16,
               half, wmma::row_major> a_frag;
```

fragment 表示一个 warp 协同持有的矩阵片段：

- 每个 lane 只持有其中一部分元素；
- 元素如何分布到 lane 和寄存器，对程序员是不透明的；
- 不要把 fragment 当成普通二维数组，也不要依赖未规定的 `x[]` 元素映射；
- `load_matrix_sync`、`mma_sync`、`store_matrix_sync` 都是 warp 集体操作。

### 4.2 四类操作

```cpp
wmma::fill_fragment(c_frag, 0.0f);
wmma::load_matrix_sync(a_frag, A_ptr, lda);
wmma::load_matrix_sync(b_frag, B_ptr, ldb);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(C_ptr, c_frag, ldc, wmma::mem_row_major);
```

含义分别是清零 accumulator、加载 A/B、执行 `C=A×B+C`、把 accumulator 写回内存。

### 4.3 layout 与 leading dimension

layout 说明矩阵逻辑布局；leading dimension 说明相邻两行（row-major）或相邻两列（column-major）起点相隔多少个元素。

若 A 是 row-major `M×K`：

```text
A[row, k] = A[row * K + k]
lda = K
```

若 B 是 row-major `K×N`：

```text
B[k, col] = B[k * N + col]
ldb = N
```

leading dimension 不是 tile 宽度。即使每次只加载 `16×16` tile，A/B 仍嵌在原始大矩阵中，所以本例传 `K` 和 `N`。

### 4.4 必须遵守的约束

1. **全 warp 参与**：warp 中所有 lane 必须以相同参数到达 WMMA 集体操作；分歧可能导致未定义行为甚至挂起。
2. **地址与 stride 合法**：`load_matrix_sync`/`store_matrix_sync` 对基地址和 stride 有对齐要求。CUDA 分配的基地址足够对齐；FP16 的 leading dimension 通常至少满足 8 元素（16 字节）倍数。
3. **shape 匹配**：本周 Demo 只处理 M/N/K 都是 16 倍数的情况，避免边界 warp 读越界。
4. **layout 一致**：声明的 row/column major 必须与实际内存解释一致。
5. **一个 warp 一个 tile**：本例采用最直观映射，不代表这是最高性能映射。

---

## 5. Demo A：一个 warp 算一个 `16×16×16`

这个 Demo 只用于理解 API。A 和 B 全为 1，结果 C 的每个元素都应为 16。

```cpp
// wmma_one_tile.cu
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err));                                    \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

namespace wmma = nvcuda::wmma;

__global__ void one_tile(const half* A, const half* B, float* C) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16,
                 half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16,
                 half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);
  wmma::load_matrix_sync(a_frag, A, 16);
  wmma::load_matrix_sync(b_frag, B, 16);
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

int main() {
  constexpr int E = 16 * 16;
  std::vector<half> hA(E, __float2half(1.0f));
  std::vector<half> hB(E, __float2half(1.0f));
  std::vector<float> hC(E, 0.0f);
  half *dA = nullptr, *dB = nullptr;
  float* dC = nullptr;

  CHECK_CUDA(cudaMalloc(&dA, E * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dB, E * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dC, E * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dA, hA.data(), E * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), E * sizeof(half), cudaMemcpyHostToDevice));

  one_tile<<<1, 32>>>(dA, dB, dC);  // 恰好一个 warp
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(hC.data(), dC, E * sizeof(float), cudaMemcpyDeviceToHost));

  float max_error = 0.0f;
  for (float x : hC) max_error = std::fmax(max_error, std::fabs(x - 16.0f));
  std::printf("C[0]=%.1f, max_error=%g, %s\n", hC[0], max_error,
              max_error == 0.0f ? "PASS" : "FAIL");

  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));
  return max_error == 0.0f ? 0 : 1;
}
```

编译运行：

```bash
nvcc -O3 -std=c++17 -arch=sm_75 wmma_one_tile.cu -o wmma_one_tile
./wmma_one_tile
# 预期：C[0]=16.0, max_error=0, PASS
```

你必须能回答：

- 为什么启动 32 个线程而不是 1 个？
- 为什么 `lda/ldb/ldc` 都是 16？
- 为什么 `c_frag` 是 float？
- 如果只有 lane 0 调用 `mma_sync` 会怎样？

---

## 6. Demo B：完整的多-tile FP16 WMMA GEMM

### 6.1 映射

本例矩阵均为 row-major：

```text
A: M × K, FP16
B: K × N, FP16
C: M × N, FP32
```

一个 warp 负责 C 的一个 `16×16` tile。warp 沿 K 维循环：

```text
C_tile = 0
for k0 = 0, 16, 32, ... K-16:
    A_tile = A[tile_row*16 : +16, k0 : k0+16]
    B_tile = B[k0 : k0+16, tile_col*16 : +16]
    C_tile += A_tile × B_tile
store C_tile
```

这版代码故意不加 shared-memory staging、double buffering 和多 warp 协作。它是“正确、清楚、可测”的教学基线，不是 cuBLAS 竞速实现。

### 6.2 完整程序

复制标记之间的代码到 `wmma_fp16_gemm.cu`。

<!-- BEGIN_WMMA_GEMM_CU -->
```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err));                                    \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

namespace wmma = nvcuda::wmma;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARP_SIZE = 32;

__global__ void wmma_fp16_gemm(const half* A, const half* B, float* C,
                               int M, int N, int K) {
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int warp_id = global_thread / WARP_SIZE;
  const int tile_cols = N / WMMA_N;
  const int total_warps = (M / WMMA_M) * tile_cols;

  // 对同一 warp 的 32 个 lane 条件相同，因此这是 warp-uniform return。
  if (warp_id >= total_warps) return;

  const int tile_row = warp_id / tile_cols;
  const int tile_col = warp_id % tile_cols;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                 half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                 half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                 float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += WMMA_K) {
    const half* a_tile = A + tile_row * WMMA_M * K + k0;
    const half* b_tile = B + k0 * N + tile_col * WMMA_N;
    wmma::load_matrix_sync(a_frag, a_tile, K);
    wmma::load_matrix_sync(b_frag, b_tile, N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  float* c_tile = C + tile_row * WMMA_M * N + tile_col * WMMA_N;
  wmma::store_matrix_sync(c_tile, c_frag, N, wmma::mem_row_major);
}

void cpu_reference_samples(const std::vector<half>& A,
                           const std::vector<half>& B,
                           const std::vector<float>& C,
                           int M, int N, int K) {
  const int sample_rows[] = {0, M / 3, M / 2, M - 1};
  const int sample_cols[] = {0, N / 3, N / 2, N - 1};
  float max_abs = 0.0f;
  float max_rel = 0.0f;

  for (int row : sample_rows) {
    for (int col : sample_cols) {
      double ref = 0.0;
      for (int k = 0; k < K; ++k) {
        ref += static_cast<double>(__half2float(A[row * K + k])) *
               static_cast<double>(__half2float(B[k * N + col]));
      }
      const float abs_err = std::fabs(C[row * N + col] - static_cast<float>(ref));
      const float rel_err = abs_err / (std::fabs(static_cast<float>(ref)) + 1e-6f);
      max_abs = std::max(max_abs, abs_err);
      max_rel = std::max(max_rel, rel_err);
    }
  }
  std::printf("sample max_abs=%g, max_rel=%g\n", max_abs, max_rel);
  if (max_abs > 2e-2f && max_rel > 2e-2f) {
    std::fprintf(stderr, "FAIL: sampled result exceeds teaching-demo tolerance\n");
    std::exit(EXIT_FAILURE);
  }
  std::puts("correctness: PASS (16 sampled outputs)");
}

int main(int argc, char** argv) {
  const int size = argc > 1 ? std::atoi(argv[1]) : 256;
  const int M = size, N = size, K = size;
  if (M <= 0 || M % 16 != 0 || N % 16 != 0 || K % 16 != 0) {
    std::fprintf(stderr, "size must be a positive multiple of 16\n");
    return EXIT_FAILURE;
  }

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
  std::vector<half> hA(static_cast<size_t>(M) * K);
  std::vector<half> hB(static_cast<size_t>(K) * N);
  std::vector<float> hC(static_cast<size_t>(M) * N);
  for (half& x : hA) x = __float2half(dist(rng));
  for (half& x : hB) x = __float2half(dist(rng));

  half *dA = nullptr, *dB = nullptr;
  float* dC = nullptr;
  CHECK_CUDA(cudaMalloc(&dA, hA.size() * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dB, hB.size() * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dC, hC.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(half),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(half),
                        cudaMemcpyHostToDevice));

  const int total_warps = (M / WMMA_M) * (N / WMMA_N);
  const dim3 block(128);  // 4 warps
  const dim3 grid((total_warps * WARP_SIZE + block.x - 1) / block.x);

  for (int i = 0; i < 10; ++i)
    wmma_fp16_gemm<<<grid, block>>>(dA, dB, dC, M, N, K);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  constexpr int iterations = 100;
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i)
    wmma_fp16_gemm<<<grid, block>>>(dA, dB, dC, M, N, K);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaGetLastError());

  float total_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
  const double ms = total_ms / iterations;
  const double gflops = 2.0 * M * N * K / (ms * 1.0e6);

  CHECK_CUDA(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  cpu_reference_samples(hA, hB, hC, M, N, K);
  std::printf("M=N=K=%d, time=%.4f ms, %.2f GFLOPS\n", size, ms, gflops);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));
  return 0;
}
```
<!-- END_WMMA_GEMM_CU -->

编译：

```bash
# T4
nvcc -O3 -std=c++17 -arch=sm_75 wmma_fp16_gemm.cu -o wmma_fp16_gemm

# A100
nvcc -O3 -std=c++17 -arch=sm_80 wmma_fp16_gemm.cu -o wmma_fp16_gemm_a100
```

运行：

```bash
./wmma_fp16_gemm 256
./wmma_fp16_gemm 512
./wmma_fp16_gemm 1024
./wmma_fp16_gemm 2048
```

### 6.3 这份 Demo 为什么不会接近 cuBLAS

- 每个 warp 直接从 global memory 加载 tile，没有 shared-memory 复用；
- 相邻 warp 会重复读取 A 或 B；
- 没有异步预取、double buffering、warp specialization；
- 一个 warp 只算一个很小的输出 tile；
- 没有针对 T4/A100 分别调 tile、流水级数和 epilogue。

因此它的意义是证明“会正确驱动 Tensor Core”，不是证明“比 cuBLAS 快”。

---

## 7. 必做：关掉答案，独立重写

只看下面骨架完成 `wmma_fp16_gemm.cu` 的 kernel。方括号不是合法代码，是你要补的内容。

```cpp
__global__ void wmma_fp16_gemm(const half* A, const half* B, float* C,
                               int M, int N, int K) {
  int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = [计算全局 warp id];
  int tile_cols = [N 方向 tile 数];
  if ([整个 warp 都越界]) return;

  int tile_row = [warp 映射到的 C tile row];
  int tile_col = [warp 映射到的 C tile col];

  [声明 A fragment：FP16 row-major]
  [声明 B fragment：FP16 row-major]
  [声明 C fragment：FP32 accumulator]
  [把 C fragment 清零]

  for (int k0 = 0; k0 < K; k0 += 16) {
    const half* a_tile = [A tile 首地址];
    const half* b_tile = [B tile 首地址];
    [加载 A，lda 是什么？]
    [加载 B，ldb 是什么？]
    [执行 MMA]
  }

  float* c_tile = [C tile 首地址];
  [row-major 写回，ldc 是什么？]
}
```

阶段检查点：

1. 先用 A/B 全 1、M=N=K=16，C 是否全 16；
2. 再用 32，确认 4 个 C tile 都正确；
3. 改成随机小数，对 CPU/cuBLAS；
4. 故意把 B 的 `ldb` 从 N 改成 K：方阵可能掩盖错误，所以再测 `M=256,N=512,K=128`；
5. 把 block 改成非 32 倍数，解释为什么最后一个不完整 warp 是危险设计；
6. 最后再做 benchmark，不能用错误结果换性能。

建议独立版本支持不同的 M/N/K，而不只接受一个方阵 size。

---

## 8. 正确性：低精度结果应该怎么比

### 8.1 不要做 bitwise equality

WMMA 路径有 FP32→FP16 输入量化、不同的乘加顺序和融合行为。与 CPU double 或 FP32 reference 逐 bit 相等不合理。

常用误差：

```text
abs_err = |out - ref|
rel_err = |out - ref| / (|ref| + epsilon)
```

只看最大相对误差也有陷阱：当 reference 接近 0，分母很小，相对误差会爆炸。最好同时报告：

- max absolute error；
- max/mean relative error（忽略或单独统计接近 0 的 reference）；
- 是否出现 NaN/Inf；
- 输入范围、K 和 reference 精度。

Demo 使用 `[-0.25,0.25]` 输入和 2% 教学容差只是起点，不是通用行业阈值。K 更大、输入范围更大或分布不同，容差必须重新评估。

### 8.2 验证顺序

```text
全 1 小矩阵
→ 非方阵与非对称值（暴露 layout/stride 错误）
→ 随机小数
→ 特殊值：0、很小值、接近 FP16 上限的值
→ 大 K 数值稳定性
```

大规模 benchmark 不要每次做完整 `O(MNK)` CPU reference。可以先对小矩阵完整校验，大矩阵抽样，并用 cuBLAS 做完整 GPU reference。

---

## 9. cuBLAS 对照：公平比较才有意义

### 9.1 比较三条路径

| 路径 | 输入 | 累加/计算 | 用途 |
|---|---|---|---|
| FP32 CUDA Core GEMM | FP32 | FP32 | 普通核心基线 |
| 本文 WMMA | FP16 | FP32 | 教学 Tensor Core kernel |
| cuBLAS `GemmEx` | FP16 | FP32 | 工业库参考 |

不要把 FP16 Tensor Core 与 FP32 naive kernel 的速度差全部归因于“代码优化”；数据类型、带宽、硬件管线都不同。

### 9.2 row-major 调 cuBLAS 的常见技巧

cuBLAS 传统接口按 column-major 解释。若 A/B/C 在内存中都是 row-major，利用：

```text
C_row = A_row × B_row
C_row^T = B_row^T × A_row^T
```

可交换 A/B 和 M/N：

```cpp
float alpha = 1.0f, beta = 0.0f;
cublasGemmEx(handle,
             CUBLAS_OP_N, CUBLAS_OP_N,
             N, M, K,
             &alpha,
             dB, CUDA_R_16F, N,
             dA, CUDA_R_16F, K,
             &beta,
             dC_ref, CUDA_R_32F, N,
             CUBLAS_COMPUTE_32F,
             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

需要链接：

```bash
nvcc -O3 -std=c++17 -arch=sm_75 compare.cu -lcublas -o compare
```

如果比较 TF32，使用 FP32 输入，并明确记录 CUDA/cuBLAS 版本以及选择的 compute/math mode；不要只写“cuBLAS 默认”。

### 9.3 benchmark 最低规范

1. 同一 GPU、同一时钟/功耗环境；
2. 相同 M/N/K 和可比较的输入；
3. 预热至少 10 次，计时至少几十次；
4. CUDA Event 记录 GPU 时间，避免把分配和 H2D/D2H 混入 kernel 时间；
5. 每条路径先过正确性；
6. 报告输入、累加、输出类型；
7. GFLOPS 公式：

```text
GFLOPS = 2 × M × N × K / time_seconds / 1e9
       = 2 × M × N × K / time_ms / 1e6
```

结果表：

| GPU | CUDA | M,N,K | 实现 | 输入/累加/输出 | time ms | GFLOPS | max abs | max rel |
|---|---|---|---|---|---:|---:|---:|---:|
| T4 |  | 512 | FP32 baseline | FP32/FP32/FP32 |  |  |  |  |
| T4 |  | 512 | WMMA demo | FP16/FP32/FP32 |  |  |  |  |
| T4 |  | 512 | cuBLAS | FP16/FP32/FP32 |  |  |  |  |

---

## 10. Nsight Compute：证明真的用了 Tensor Core

### 10.1 先得到可复现实验

为避免 profile 100 次循环产生大量数据，可把 Demo 中 `iterations` 临时改小，或者单独增加只启动一次的 profile 模式。

```bash
ncu --set full --kernel-name regex:wmma.* ./wmma_fp16_gemm 1024
```

不同 NCU/GPU 版本的 metric 名称会变化，先查询本机：

```bash
ncu --query-metrics | rg -i "tensor|mma|hmma"
```

再查看编译后的指令：

```bash
cuobjdump --dump-sass ./wmma_fp16_gemm | rg "HMMA|MMA"
```

WMMA API、SASS 中的 HMMA/MMA 证据、Tensor pipe 活跃度和达到 Tensor Core 量级的吞吐应相互印证。不要只凭一个百分比下结论。

### 10.2 重点指标怎么读

| 观察 | 你想回答的问题 | 常见解释 |
|---|---|---|
| Tensor/MMA pipe activity | 专用矩阵管线是否工作 | 低可能是 tile 太少或喂不饱 |
| SM busy / achieved occupancy | SM 是否有足够工作 | occupancy 低不一定是根因 |
| DRAM/L2 throughput | 是否受数据供给限制 | 本文 global→fragment Demo 可能较明显 |
| registers/thread | fragment 占用多少寄存器 | 过高会限制常驻 warp |
| warp stall reasons | warp 在等什么 | memory dependency、not selected 等需结合上下文 |
| kernel duration | 是否太短 | 小矩阵时启动/测量噪声占比大 |

### 10.3 你应该形成的分析

不要只写“Tensor utilization 低”。应写成：

```text
在 T4、M=N=K=1024、FP16 input/FP32 accumulate 下，SASS 出现 HMMA，说明使用了
Tensor Core。kernel 的 Tensor pipe 有活动，但与 cuBLAS 仍差 X 倍。DRAM/L2 指标和代码
显示多个 warp 重复从 global 加载 A/B，且没有 shared-memory staging/pipeline，因此主要改进方向
是 block/warp 多级 tiling与数据复用，而不是继续增加单个 warp 的 MMA 次数。
```

---

## 11. 常见错误与排查

| 现象 | 高概率原因 | 检查方法 |
|---|---|---|
| 编译提示架构不支持 | `-arch` 太低或类型/shape 不支持 | T4 用 `sm_75`，A100 用 `sm_80` |
| 结果像转置或完全错误 | A/B layout、tile 地址或 lda/ldb 错 | 用非方阵和非对称数据 |
| 非法内存访问 | M/N/K 不是 16 倍数，边界 tile 越界 | 先限制 shape，再实现 padding/boundary path |
| kernel 挂起/行为异常 | warp 内分歧调用 WMMA 集体操作 | 检查条件是否对整个 warp 一致 |
| 小矩阵 WMMA 更慢 | 启动开销、并行度不足 | 测 512/1024/2048，增加预热 |
| GFLOPS 异常高 | 计时公式、单位、异步计时错误 | Event 同 stream，stop 后 synchronize |
| GFLOPS 异常低 | Debug 编译、频繁 global load、tile 太少 | `-O3`，看 NCU 和 SASS |
| 与 CPU 差异大 | FP16 量化、溢出、stride 错 | 先全 1，再缩小输入范围，再查 layout |
| 第一次运行特别慢 | CUDA context/JIT/缓存预热 | 计时前 warmup |
| cuBLAS 结果错 | row-major/column-major 参数混淆 | 用小非方阵手算并核对 M/N/lda/ldb |

---

## 12. T4 与 A100 分别练什么

### 12.1 T4（Turing，SM 7.5）

必做：

- FP16 input + FP32 accumulator 的 WMMA；
- 与 FP32 CUDA Core 和 FP16 cuBLAS 对比；
- 在 SASS 中找 HMMA；
- 解释为何教学 WMMA 与 cuBLAS 有差距。

T4 不支持 BF16/TF32/FP8 的原生 Tensor Core 路径。它支持 INT8/INT4 Tensor Core，但本周不展开量化 kernel。

### 12.2 A100（Ampere，SM 8.0）

在 T4 实验之外增加：

- FP16 WMMA 重跑并对比代际差异；
- cuBLAS FP32 完整精度路径与允许 TF32 的路径对比；
- BF16 Tensor Core GEMM（优先用 cuBLAS/CUTLASS 对照，不要求再写一套复杂 WMMA）；
- 记录精度、吞吐和输入范围差异。

不要写“A100 上 FP32 自动用 TF32”。准确说法是：支持的库/API 在启用相应计算模式时可选择 TF32 Tensor Core 路径。

---

## 13. FP8：为什么不能直接把 float 砍成 8 位

### 13.1 E4M3 与 E5M2

- E4M3：更多尾数字段，精度相对好，范围较小；
- E5M2：指数更多，范围更大，精度更粗；
- “前向一定 E4M3、反向一定 E5M2”只是常见 hybrid recipe，不是不可变规则；现代方案会按张量、算子和 scaling recipe 选择。

### 13.2 scaling 的最小数学模型

假设某 FP8 格式的最大有限正数为 `FP8_MAX`，张量最大绝对值为：

```text
amax = max_i |x_i|
```

一种最直观的 scale：

```text
scale = FP8_MAX / max(amax, epsilon)
q_i = FP8(clip(x_i × scale, -FP8_MAX, FP8_MAX))
x_hat_i = FP32(q_i) / scale
```

真实系统还要决定：

- scale 是否限制为 2 的幂；
- 用当前 amax 还是历史窗口（delayed scaling）；
- scale 是每 tensor、每 channel 还是每 block；
- NaN/Inf 和异常值怎么处理；
- scale/scale inverse 用什么精度保存；
- GEMM 的 scale 如何布局和重排，才能被硬件高效读取。

### 13.3 per-tensor 与 per-block

| 方法 | 优点 | 缺点 |
|---|---|---|
| per-tensor scale | 元数据少、实现简单 | 一个异常值会压缩整个张量有效精度 |
| per-channel/group scale | 适应不同通道范围 | 元数据和布局更复杂 |
| per-block scale | 局部范围适配好，适合细粒度 FP8 | scale 数量多，需要高效 layout/swizzle |

FP8 工程难点不只在转换指令，而在“统计范围 → 生成 scale → 转换 → GEMM → 反缩放/融合 epilogue”的整条数据流。

### 13.4 为什么本周不在 T4/A100 手写 FP8 TC

T4/A100 没有 Hopper 那样的原生高吞吐 FP8 Tensor Core 路径。可以软件模拟 FP8 量化，但那主要验证数值误差，不代表 FP8 Tensor Core 性能。因此本周把时间用于理解 scaling、布局和工程权衡更划算。

---

## 14. DeepGEMM：读懂它在解决什么

> 以下是截至 2026-06 的学习视角。项目会快速演进，具体支持范围以其当前 README 为准；本文内容足够完成本周主线。

DeepGEMM 是面向现代 LLM 计算原语的高性能 Tensor Core kernel 库。当前项目不仅涉及 FP8，也包含 FP4、BF16、MoE 等方向，并通过轻量 JIT 针对 shape/配置生成内核。其主要 kernel 面向 SM90/SM100 一类更新架构，不是给 T4/A100 直接运行的 Week 3 Demo。

### 14.1 五个关键词

**1. FP8/BF16/FP4 GEMM**

不同输入格式、累加和 epilogue 组合，需要不同 Tensor Core 指令、scale 处理和数据布局。

**2. Grouped GEMM**

MoE 中每个专家接收的 token 数不同，相当于一组 M 不同、N/K 相近的 GEMM。逐个 launch 浪费调度开销和并行度，grouped GEMM 把它们组织起来执行。

**3. JIT specialization**

LLM shape 和硬件架构已知时，可在运行时为具体 shape、tile、流水级数生成/编译 kernel，减少通用代码的分支与保守选择。

**4. Scale layout / swizzle**

细粒度 FP8/FP4 scaling 会产生许多 scale。逻辑上正确的二维 scale 排列不一定符合 TMA/Tensor Core 消费方式，因此常需要重排，使 scale 加载合并、对齐且能进入流水。

**5. TMA/WGMMA 与流水**

Hopper 以后可用 TMA 搬运 tile、用 WGMMA 让 warp group 驱动 Tensor Core，并通过 warp specialization 重叠加载与计算。它们是现代工业 kernel 与本文简单 WMMA 的代际差异。

### 14.2 你不需要做什么

本周不要求读完源码、复现 benchmark 或背模板。你只需能回答：

```text
DeepGEMM 为什么不仅是“调用一次 MMA”？
因为高性能低精度 GEMM 还要解决 shape 特化、多级 tiling、数据搬运流水、scale 布局、
epilogue、MoE grouped 调度以及架构专用指令。
```

---

## 15. Tensor Core 与 LLM 推理

Tensor Core 峰值很高，不代表所有 LLM 阶段都接近峰值：

- **Prefill**：一次处理较多 token，QKV/MLP 通常形成较大 GEMM，更容易利用 Tensor Core；
- **Decode**：每步 token 少，若 batch 小，矩阵可能很“瘦”，KV cache 访问、显存带宽、kernel launch 和调度开销更突出；
- **MoE**：总计算量大，但 token 路由后每个专家 shape 不均匀，还叠加通信和 grouped GEMM 问题。

所以面试中不要只说“换 FP8，算力翻倍”。还要问：

```text
shape 是否足够大？
Tensor Core 能否被喂饱？
瓶颈是 compute、memory、launch 还是 communication？
量化/scale/布局的额外成本是多少？
```

---

## 16. Day 1–7 执行安排

### Day 1：混合精度账本

- 学习：第 2 节；
- 动手：填写 FP32/TF32/FP16/BF16/FP8 表格；
- 实验：写一个小程序把一组边界值转 FP16 再转回 FP32；
- 产出：精度表 + 观察到的舍入/溢出；
- 闭卷题：FP16 与 BF16 各自把位数花在哪里，为什么？

### Day 2：Tensor Core 与 WMMA 心智模型

- 学习：第 1、3、4 节；
- 动手：画出一个 warp、A/B fragment、C tile 和 K 循环；
- 实验：运行 Demo A，修改 A/B 值并预测 C；
- 产出：一张数据流图 + 200 字口述；
- 闭卷题：fragment 为什么不能当普通二维数组？

### Day 3：跑通完整 Demo

- 学习：第 5、6 节；
- 动手：编译 Demo B，运行 256/512/1024；
- 测量：记录 time、GFLOPS、误差；
- 产出：第一版 benchmark 表；
- 闭卷题：A/B/C 的 lda/ldb/ldc 为什么分别是 K/N/N？

### Day 4：独立重写与正确性

- 学习：第 7、8、11 节；
- 动手：关掉 Demo B，只看骨架重写；
- 测量：全 1、随机数、非方阵、抽样 reference；
- 产出：你自己的 `wmma_fp16_gemm.cu`；
- 闭卷题：为什么边界判断必须 warp-uniform？

### Day 5：cuBLAS 与 NCU

- 学习：第 9、10 节；
- 动手：加入 cuBLAS `GemmEx` 对照；
- 测量：FP32 baseline、WMMA、cuBLAS；运行 NCU 并查 SASS；
- 产出：`tensor_core_profile.md`；
- 闭卷题：如何用三类证据证明使用了 Tensor Core？

### Day 6：A100 精度路径与 FP8

- 学习：第 12、13 节；
- 动手：A100 上重跑 FP16；用库对照完整 FP32、TF32、BF16；
- 测量：性能与误差，不只记录速度；
- 产出：FP8 scaling 推导 + A100 对照表；
- 闭卷题：TF32 为什么不是普通 FP32 kernel 的自动加速开关？

### Day 7：DeepGEMM 与面试复盘

- 学习：第 14、15 节；
- 动手：用自己的话解释 grouped GEMM、JIT、scale layout；
- 测量：复查本周数据能否支撑结论；
- 产出：一页复盘 + 3 分钟口述录音；
- 闭卷题：DeepGEMM 比简单 WMMA Demo 多解决了哪些工程问题？

---

## 17. 可直接填写的记录模板

### 17.1 环境

```text
GPU:
Compute Capability:
CUDA Toolkit:
Driver:
NVCC flags:
GPU clocks/power mode（若固定）:
```

### 17.2 benchmark

| M | N | K | 实现 | 输入/累加/输出 | block/grid | time ms | GFLOPS | max abs | max rel |
|---:|---:|---:|---|---|---|---:|---:|---:|---:|
| 256 | 256 | 256 | WMMA | FP16/FP32/FP32 |  |  |  |  |  |
| 512 | 512 | 512 | WMMA | FP16/FP32/FP32 |  |  |  |  |  |
| 1024 | 1024 | 1024 | WMMA | FP16/FP32/FP32 |  |  |  |  |  |

### 17.3 NCU

| 规模 | Tensor/MMA 证据 | SM busy | DRAM/L2 | occupancy | registers | 主要 stall | 结论 |
|---|---|---:|---:|---:|---:|---|---|
| 1024³ |  |  |  |  |  |  |  |

### 17.4 优化解释

```text
现象：
证据：
当前瓶颈判断：
为什么不是另一个瓶颈：
下一步改动：
如何验证改动有效：
```

---

## 18. 最终验收

### 18.1 代码与数据

- [ ] 不看完整 Demo，能写出 fragment/load/mma/store 主流程；
- [ ] 能处理至少三个规模并通过正确性；
- [ ] 能用非方阵暴露 layout/stride 错误；
- [ ] 有 FP32 baseline、WMMA、cuBLAS 的公平对比；
- [ ] 有 NCU 报告和 HMMA/MMA 指令证据；
- [ ] 能解释为什么教学 Demo 远慢于 cuBLAS。

### 18.2 闭卷口述

- [ ] Tensor Core 与 CUDA Core 的分工；
- [ ] WMMA、MMA、WGMMA 的抽象层级；
- [ ] fragment 是什么，为什么映射不透明；
- [ ] FP16、BF16、TF32 的范围和精度区别；
- [ ] 为什么常用 FP32 accumulator；
- [ ] 普通 FP32 kernel 为什么不会自动走 TF32；
- [ ] FP8 为什么需要 amax 和 scaling；
- [ ] per-tensor 与 per-block scale 的权衡；
- [ ] 怎么证明 kernel 使用了 Tensor Core；
- [ ] grouped GEMM 为什么适合 MoE；
- [ ] DeepGEMM 解决了哪些简单 WMMA 没解决的问题；
- [ ] 为什么 LLM decode 不一定吃满 Tensor Core。

### 18.3 三分钟参考口述

```text
Tensor Core 是 NVIDIA GPU 中面向矩阵乘加的专用计算单元。普通 CUDA Core GEMM 的内层是
标量 FMA，而 Tensor Core 通过 warp 或 warp group 协作执行 MMA，以 FP16、BF16、TF32、
FP8 等格式获得高吞吐。WMMA 是 CUDA C++ 的 warp-level API，fragment 表示由整个 warp
分布式持有的矩阵片段；更底层有 MMA，Hopper 还有 WGMMA。

高性能 Tensor Core GEMM 仍需要 global/shared/register 多级 tiling、合并访问、合适 layout、
流水和 epilogue。低精度也不是免费加速：FP16 范围小，BF16 精度粗，FP8 更需要 amax、scale、
溢出和误差管理。验证时我会同时做 reference 误差、CUDA Event benchmark、NCU Tensor pipe
指标和 SASS HMMA/MMA 指令检查。标准 GEMM 工程上优先使用 cuBLAS/CUTLASS，手写 WMMA
主要用于理解硬件，特殊融合或研究更底层 kernel。
```

---

## 19. 可选深挖资料（不读也能完成本周）

- [CUDA Programming Guide：Warp Matrix Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf)
- [NVIDIA Transformer Engine：FP8/FP4 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
- [DeepSeek DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)

资料的作用是查精确 API、架构支持和最新工程实现。学习主线、Demo、实验和验收已经全部包含在本文中。
