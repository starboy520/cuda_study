#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

// ============================================================================
// 2D thread tiling GEMM —— 自己来写
//   C = A(MxK) * B(KxN)，row-major
//   思路：block 算 BM x BN 的 C tile，每个 thread 算 TM x TN 个元素，外积累加

// ============================================================================

#ifndef TM
#define TM 8
#endif

#ifndef TN
#define TN 8
#endif

#ifndef BM
#define BM 64
#endif

#ifndef BN
#define BN 64
#endif

#ifndef BK
#define BK 32
#endif  

namespace cg = cooperative_groups;
__device__ void load_tile_async(
        float* smem, int smem_ld,
        const float* src, int tileH, int tileW, int row_base, int col_base,
        int ld, int bound_row, int bound_col, int tid, int n_thread,
        cuda::pipeline<cuda::thread_scope_block>& pipe) {
    for (int i = tid; i < tileH * tileW; i+= n_thread) {
        int r = i / tileW;
        int c = i % tileW;
        int g_row = row_base + r;
        int g_col = col_base + c;
        if (g_row < bound_row && g_col < bound_col) {
            cuda::memcpy_async(&smem[r * smem_ld + c], &src[g_row * ld + g_col],
            sizeof(float), pipe);
        } else {
            // TODO
        }
    }
}

__global__ void gemm_2d_double_buffering_tiling(const float* a, const float* b,
        float* c, int M, int N, int K) {
    // TODO: 自己实现 2D thread tiling
    __shared__ float sa[2][BM][BK + 1];  // +1 避免 bank conflict
    __shared__ float sb[2][BK][BN + 1];  // +1 避免 bank conflict

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss;

    auto block  = cg::this_thread_block();
    auto pipe = cuda::make_pipeline(block, &pss);

    float reg_c[TM][TN] = {0.0};
    int tid = threadIdx.y *blockDim.x + threadIdx.x;
    int n_thread = blockDim.x * blockDim.y;

    int stage = 0;
    int step = 0;
    // 先预取第一阶段
    pipe.producer_acquire();
    load_tile_async(&sa[stage][0][0], BK+1, a, BM, BK, blockIdx.y * BM, step, K, M, K, tid, n_thread, pipe);
    load_tile_async(&sb[stage][0][0], BN+1, b, BK, BN, step, blockIdx.x * BN, N, K, N, tid, n_thread, pipe);
    /*
    for (int i = tid; i < BM * BK; i += n_thread) {
        int row = i / BK;
        int col = i % BK;
        int global_row = blockIdx.y * BM + row; // 一个block处理(bm*bn行), 这里逻辑不一样，相同
        int global_column = step + col;
        if (global_row < M && global_column < K) {
            cuda::memcpy_async(&sa[stage][row][col], a+global_row*K+global_column, sizeof(float), pipe);
        } else {
            sa[stage][row][col] = 0;
        }
    }

    for (int i = tid; i < BK * BN; i += n_thread) {
        int row = i / BN;
        int col = i % BN;
        int global_row = step + row;
        int global_col = blockIdx.x * BN + col;
        if (global_row < K && global_col < N) {
            cuda::memcpy_async(&sb[stage][row][col],
                b + global_row * N + global_col, sizeof(float), pipe);
        } else {
            sb[stage][row][col] = 0.0;
        }
    }
    */
    pipe.producer_commit();


    for (; step < K; step += BK) {
        /**
        线程是班运工,编号依次搬数据即可
        1. 把 block 内线程拍平成 1D：tid = threadIdx.y * blockDim.x + threadIdx.x
   总线程数 nthreads = blockDim.x * blockDim.y
2. sa 要搬 BM*BK 个元素，用 grid-stride loop：
   for (idx = tid; idx < BM*BK; idx += nthreads)
       int r = idx / BK, col = idx % BK;   // sa[r][col]
       int gr = blockIdx.y*BM + r, gc = step + col;
       sa[r][col] = (gr<M && gc<K) ? a[gr*K+gc] : 0;
3. sb 同理搬 BK*BN 个元素
        */
        stage = (stage + 1) % 2;

        if (step + BK < K) {
            pipe.producer_acquire();
            load_tile_async(&sa[stage][0][0], BK+1, a, BM, BK, blockIdx.y * BM, step+BK, K, M, K, tid, n_thread, pipe);
            load_tile_async(&sb[stage][0][0], BN+1, b, BK, BN, step+BK, blockIdx.x * BN, N, K, N, tid, n_thread, pipe);
            /*
            for (int i = tid; i < BM * BK; i += n_thread) {
                int row = i / BK;
                int col = i % BK;
                int global_row = blockIdx.y * BM + row; // 一个block处理(bm*bn行), 这里逻辑不一样，相同
                int global_column = step + BK + col;
                if (global_row < M && global_column < K) {
                    cuda::memcpy_async(&sa[stage][row][col], a+global_row*K+global_column, sizeof(float), pipe);
                } else {
                    sa[stage][row][col] = 0;
                }
            }
        
            for (int i = tid; i < BK * BN; i += n_thread) {
                int row = i / BN;
                int col = i % BN;
                int global_row = step + BK + row;
                int global_col = blockIdx.x * BN + col;
                if (global_row < K && global_col < N) {
                    cuda::memcpy_async(&sb[stage][row][col], 
                        b + global_row * N + global_col, sizeof(float), pipe);
                } else {
                    sb[stage][row][col] = 0.0;
                }
            }
            */
            //__syncthreads();
            pipe.producer_commit();
        }


        pipe.consumer_wait();
        int lastStage = (stage == 0) ? 1 : 0;
        for (int k = 0; k < BK; k++) {
            float reg_a[TM] = {0.0};

            for (int i = 0; i < TM; i++) {
                reg_a[i] = sa[lastStage][threadIdx.y * TM + i][k];
            }

            float reg_b[TN] = {0.0};
            for (int i = 0; i < TN; i++) {
                reg_b[i] = sb[lastStage][k][threadIdx.x * TN + i];
            }

            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        pipe.consumer_release();
    }

    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int row = blockIdx.y * BM + threadIdx.y * TM + i;
            int column = blockIdx.x * BN + threadIdx.x * TN + j;
            if (row < M && column < N) {
                c[row * N + column] = reg_c[i][j];
            }
        }
    }
}

void gemm_2d_thread_tiling_launcher(const float* a, const float* b,
        float* c, int M, int N, int K) {
    // TODO: 自己配置 block / grid，并启动 kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // TODO: gemm_2d_thread_tiling<<<grid, block>>>(a, b, c, M, N, K);
    dim3 block(BN/TN, BM/TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_2d_double_buffering_tiling<<<grid, block>>>(a, b, c,  M,  N, K);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float tm = 0;
    cudaEventElapsedTime(&tm, start, stop);

    double gflops = 2.0 * M * N * K / (tm / 1e3) / 1e9;
    printf("gemm_2d_double_buffering_tiling cost: %.3f ms, %.2f GFLOPS\n", tm, gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// 以下是测试框架，保持不动
// ============================================================================

// CPU 参考实现
static void gemm_cpu_ref(const float* a, const float* b, float* c,
        int M, int N, int K) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[row * K + k] * b[k * N + col];
            }
            c[row * N + col] = sum;
        }
    }
}

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err),    \
                   __FILE__, __LINE__);                                    \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

int main(int argc, char** argv) {
    int M = 512, N = 512, K = 512;
    if (argc == 2) {
        M = N = K = atoi(argv[1]);
    } else if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    printf("GEMM shape: M=%d N=%d K=%d\n", M, N, K);

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float* hA = (float*)malloc(bytesA);
    float* hB = (float*)malloc(bytesB);
    float* hC = (float*)malloc(bytesC);
    float* hRef = (float*)malloc(bytesC);

    for (size_t i = 0; i < (size_t)M * K; i++) hA[i] = (float)(rand() % 10) / 10.0f;
    for (size_t i = 0; i < (size_t)K * N; i++) hB[i] = (float)(rand() % 10) / 10.0f;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    // warmup
    gemm_2d_thread_tiling_launcher(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());

    // 正式计时
    gemm_2d_thread_tiling_launcher(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

    gemm_cpu_ref(hA, hB, hRef, M, N, K);
    double max_abs_err = 0.0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        double e = fabs((double)hC[i] - (double)hRef[i]);
        if (e > max_abs_err) max_abs_err = e;
    }
    printf("max abs error vs CPU: %e -> %s\n", max_abs_err,
           max_abs_err < 1e-2 ? "PASS" : "FAIL");

    free(hA); free(hB); free(hC); free(hRef);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
