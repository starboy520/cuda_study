// gemm_tiled.cu
//
// 卷六实验：GEMM 优化阶梯实测 —— naive vs shared-memory tiling。
//
// ============================================================================
// 目的：把卷六第 01/02 章的"理论提升"变成"真实 GFLOPS 数字"。
//   - naive GEMM：每个输出独立读 A 一行、B 一列，重复读 + B 不合并 → 慢
//   - tiled GEMM：把 A、B 子块载入 shared memory 复用，提升算术强度 → 快
//
// 三个版本：
//   gemmNaive  : 基线（卷二第 09 章那种）
//   gemmTiled  : shared-memory tiling，TILE×TILE 子块复用
//   cuBLAS（可选）: 作为"天花板"对比，本实验默认不依赖，避免链接复杂度
//
// 编译：
//   nvcc -O3 -arch=sm_75 gemm_tiled.cu -o gemm_tiled
// 运行：
//   ./gemm_tiled            # 默认 1024×1024×1024
//   ./gemm_tiled 2048 2048 2048
//   ./gemm_tiled 1024 768 512   # 非方阵
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "../../common/cuda_check.cuh"

constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// 版本 1：naive。每个线程算一个 C[row][col]，独立读 A 一行、B 一列。
// 病根：① A 被重复读（C 一行 N 个输出都读同一行 A）② B[k*N+col] 跨步不合并。
// ---------------------------------------------------------------------------
__global__ void gemmNaive(const float* A, const float* B, float* C,
                          int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

// ---------------------------------------------------------------------------
// 版本 2：shared-memory tiling。一个 block 算一个 TILE×TILE 的 C 块。
// 沿 K 维分段：每段把 A、B 各一个 TILE×TILE 子块载入 shared，block 内复用。
// 收益：① A/B 元素从 global 读 1 次、在 shared 复用 TILE 次 → AI 提升约 TILE 倍
//       ② 加载阶段 A、B 都按行连续读 → 都合并（治好 naive 的病根②）
// ---------------------------------------------------------------------------
__global__ void gemmTiled(const float* A, const float* B, float* C,
                          int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float sum = 0.0f;

  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    int aCol = t * TILE + threadIdx.x;
    int bRow = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] =
        (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
    __syncthreads();                       // 等整块加载完

    for (int k = 0; k < TILE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();                       // 等算完再覆盖 shared
  }

  if (row < M && col < N) C[row * N + col] = sum;
}

using GemmKernel = void (*)(const float*, const float*, float*, int, int, int);

static float timeGemm(GemmKernel kernel, const float* dA, const float* dB,
                      float* dC, int M, int N, int K) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  kernel<<<grid, block>>>(dA, dB, dC, M, N, K);  // warmup
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

// CPU 参考（double 累加），仅用于较小尺寸验证正确性
static void cpuGemm(const std::vector<float>& A, const std::vector<float>& B,
                    std::vector<float>& C, int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double s = 0.0;
      for (int k = 0; k < K; ++k) s += (double)A[i * K + k] * B[k * N + j];
      C[i * N + j] = (float)s;
    }
}

static bool verify(const std::vector<float>& got, const std::vector<float>& ref) {
  for (size_t i = 0; i < ref.size(); ++i) {
    double rel = std::abs(got[i] - ref[i]) / (std::abs(ref[i]) + 1e-6);
    if (rel > 1e-3) {
      printf("  mismatch at %zu: got %.3f ref %.3f\n", i, got[i], ref[i]);
      return false;
    }
  }
  return true;
}

static double gflops(int M, int N, int K, float ms) {
  double flop = 2.0 * M * N * K;
  return flop / (ms / 1000.0) / 1e9;
}

int main(int argc, char** argv) {
  int M = 1024, N = 1024, K = 1024;
  if (argc >= 4) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
  }
  printf("GEMM M=%d N=%d K=%d  (2*M*N*K = %.2f GFLOP)\n", M, N, K,
         2.0 * M * N * K / 1e9);

  std::vector<float> hA(M * K), hB(K * N), hC(M * N), hRef(M * N);
  for (int i = 0; i < M * K; ++i) hA[i] = (float)((i % 13) - 6) * 0.1f;
  for (int i = 0; i < K * N; ++i) hB[i] = (float)((i % 7) - 3) * 0.2f;

  // 只在较小规模做 CPU 参考（CPU GEMM 是 O(N^3)，大矩阵太慢）
  bool doVerify = ((long)M * N * K <= 512L * 512 * 512);
  if (doVerify) cpuGemm(hA, hB, hRef, M, N, K);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  struct V { const char* name; GemmKernel k; };
  V versions[] = {{"naive", gemmNaive}, {"tiled", gemmTiled}};

  printf("\n%-7s %12s %14s %10s\n", "版本", "Kernel(ms)", "GFLOPS", "正确性");
  float baseMs = 0.0f;
  for (auto& v : versions) {
    float ms = timeGemm(v.k, dA, dB, dC, M, N, K);
    if (baseMs == 0.0f) baseMs = ms;
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    const char* ok = doVerify ? (verify(hC, hRef) ? "PASS" : "FAIL") : "skip";
    printf("%-7s %12.3f %14.1f %10s\n", v.name, ms, gflops(M, N, K, ms), ok);
  }

  printf("\nTILE=%d。tiled 应明显快于 naive（数据复用提升算术强度）。\n", TILE);
  printf("大矩阵跳过 CPU 验证（O(N^3) 太慢）；小矩阵(<=512^3)会验证正确性。\n");

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return 0;
}
