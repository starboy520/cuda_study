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
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err));                                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

namespace wmma = nvcuda::wmma;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARP_SIZE = 32;

// ============================================================
// WMMA FP16 GEMM —— 你来写
//   A: M×K (FP16, row-major)   B: K×N (FP16, row-major)   C: M×N (FP32)
//   一个 warp 算 C 的一个 16×16 tile，沿 K 循环累加。
//
// 步骤（对照手册第7节骨架）：
//   1. warp_id = 全局线程号 / 32
//   2. tile_cols = N / WMMA_N
//   3. 越界返回（必须 warp-uniform：整个 warp 条件相同）
//   4. tile_row = warp_id / tile_cols; tile_col = warp_id % tile_cols;
//   5. 声明 a_frag(matrix_a,half,row_major)、b_frag(matrix_b,half,row_major)、c_frag(accumulator,float)
//   6. fill_fragment(c_frag, 0)
//   7. for k0=0; k0<K; k0+=16:
//        a_tile = A + tile_row*16*K + k0;   (lda = K)
//        b_tile = B + k0*N + tile_col*16;   (ldb = N)
//        load_matrix_sync + mma_sync
//   8. c_tile = C + tile_row*16*N + tile_col*16; store_matrix_sync(ldc = N)
// ============================================================
__global__ void wmma_fp16_gemm(const half* A, const half* B, float* C,
                               int M, int N, int K) {
    // Wmm的版本，组织方式是一维的，不是二维block
    // 核心思路， 用c矩阵， 来判断第几块， 然后每块
    int g_thread_num = blockDim.x * blockIdx.x + threadIdx.x;
    // 当前线程属于第几个warp,  一个warp 32个线程协作。

    int warp_id = g_thread_num/ 32;

    int tile_size_col = N / WMMA_N; // 列方向 可以切成多少块， 
    int tile_size_row = M / WMMA_M; // 行方向， 可以且成多少块， 

    if (warp_id >= tile_size_col * tile_size_row) return;  // warp 大于 总数， 则不需要做了

    // 这两行怎么理解， warp_id 是一维， 判断  warp_id处理(tile_row, tile_col)块
    int tile_row = warp_id / tile_size_col;
    int tile_col = warp_id % tile_size_col;

    // a 处理 tile_row 这一tile（行）, b 处理   tile_col这一列（tile）

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_freg;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    for (int k = 0; k < K; k+=WMMA_K) {
        // tile_row *wmma_m 行， 每行K 个元素
        const half* a_tile = A + tile_row * WMMA_M * K + k;
        // 第k 行，每行N 个元素， tile_col * WMMA_N列
        const half* b_tile = B +  k * N + tile_col * WMMA_N;
        
        wmma::load_matrix_sync(a_frag, a_tile, K);
        wmma::load_matrix_sync(b_freg, b_tile, N);
        wmma::mma_sync(c_frag, a_frag, b_freg, c_frag);
    }

    float* c_tile = C + tile_row * WMMA_M * N + tile_col * WMMA_N;
    wmma::store_matrix_sync(c_tile, c_frag, N, wmma::mem_row_major);
}


// ---- 抽样 CPU 参考校验（大矩阵不做全量 O(MNK)）----
static void cpu_reference_samples(const std::vector<half>& A,
                                  const std::vector<half>& B,
                                  const std::vector<float>& C,
                                  int M, int N, int K) {
    const int sample_rows[] = {0, M / 3, M / 2, M - 1};
    const int sample_cols[] = {0, N / 3, N / 2, N - 1};
    float max_abs = 0.0f, max_rel = 0.0f;
    for (int row : sample_rows) {
        for (int col : sample_cols) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k)
                ref += (double)__half2float(A[row * K + k]) *
                       (double)__half2float(B[k * N + col]);
            float abs_err = std::fabs(C[row * N + col] - (float)ref);
            float rel_err = abs_err / (std::fabs((float)ref) + 1e-6f);
            max_abs = std::max(max_abs, abs_err);
            max_rel = std::max(max_rel, rel_err);
        }
    }
    std::printf("sample max_abs=%g, max_rel=%g -> %s\n", max_abs, max_rel,
                (max_abs < 2e-2f || max_rel < 2e-2f) ? "PASS" : "FAIL");
}

int main(int argc, char** argv) {
    const int size = argc > 1 ? std::atoi(argv[1]) : 256;
    const int M = size, N = size, K = size;
    if (M <= 0 || M % 16 || N % 16 || K % 16) {
        std::fprintf(stderr, "size 必须是正的 16 倍数\n");
        return EXIT_FAILURE;
    }

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    std::vector<half> hA((size_t)M * K), hB((size_t)K * N);
    std::vector<float> hC((size_t)M * N);
    for (half& x : hA) x = __float2half(dist(rng));
    for (half& x : hB) x = __float2half(dist(rng));

    half *dA, *dB; float* dC;
    CHECK_CUDA(cudaMalloc(&dA, hA.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dB, hB.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(half), cudaMemcpyHostToDevice));

    const int total_warps = (M / WMMA_M) * (N / WMMA_N);
    const dim3 block(128);  // 4 warps
    const dim3 grid((total_warps * WARP_SIZE + block.x - 1) / block.x);

    // warmup
    for (int i = 0; i < 10; ++i) wmma_fp16_gemm<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
    const int iters = 100;
    CHECK_CUDA(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) wmma_fp16_gemm<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float total_ms = 0.0f; CHECK_CUDA(cudaEventElapsedTime(&total_ms, s, e));
    double ms = total_ms / iters;
    double gflops = 2.0 * M * N * K / (ms * 1.0e6);

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cpu_reference_samples(hA, hB, hC, M, N, K);
    std::printf("M=N=K=%d, time=%.4f ms, %.2f GFLOPS\n", size, ms, gflops);

    CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
    return 0;
}
