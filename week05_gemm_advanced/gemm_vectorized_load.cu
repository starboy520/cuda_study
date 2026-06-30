#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

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


__global__ void gemm_2d_thread_tiling_vectorized_load(const float* a, const float* b,
        float* c, int M, int N, int K) {
    // TODO: 自己实现 2D thread tiling
    __shared__ float sa[BM][BK + 1];  // +1 避免 bank conflict（写 shared 是标量，不影响 float4 读 global）
    __shared__ float sb[BK][BN + 1];  // +1 避免 bank conflict

    float reg_c[TM][TN] = {0.0};
    
    for (int step = 0; step < K; step += BK) {
        /*  style1, 每个线程要算什么就搬什么数据
        // ---- 加载 sa[BM][BK]：每个线程 (x,y) 搬 TM * (BK/TM) 个元素 ----
        // row    = TM*threadIdx.y + i  ：i 从 0..TM-1，所以每个线程负责【连续的 TM 行】
        //                                （ty 0 -> 行 0..7，ty 1 -> 行 8..15 ...）
        // column = j*TM + threadIdx.x  ：j 从 0..BK/TM-1，步长是 blockDim.x(=TM=8)，
        //                                所以列是【跳着取】的（tx -> tx, tx+8, tx+16, tx+24）
        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < BK/TM; j++) {
                int column = j * TM + threadIdx.x;
                int row = TM * threadIdx.y + i;
                int globalRow = blockIdx.y * BM + threadIdx.y * TM + i;
                int globalColumn = step + column;
                if (globalRow < M && globalColumn < K) {
                    sa[row][column] = a[globalRow * K + globalColumn];
                } else {
                    sa[row][column] = 0.0f;
                }
            }
        }
        */


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

        int tid = threadIdx.y *blockDim.x + threadIdx.x;
        int n_thread = blockDim.x * blockDim.y;
        int load_size = BM * BK / 4; // 每个线程load float4
        for (int i = tid; i < load_size; i += n_thread) {
            int row = i* 4 / BK;
            int col = i * 4 % BK;
            int global_row = blockIdx.y * BM + row; // 一个block处理(bm*bn行), 这里逻辑不一样，相同
            int global_column = step + col;
            if (global_row < M && global_column < K) {
                const float4 v_a = reinterpret_cast<const float4 *>(a+global_row * K + global_column)[0]; 
                sa[row][col] = v_a.x;
                sa[row][col+1] = v_a.y;
                sa[row][col+2] = v_a.z;
                sa[row][col+3] = v_a.w;
            } else {
                sa[row][col] = 0.;
                sa[row][col+1] = 0.0;
                sa[row][col+2] = 0.0;
                sa[row][col+3] = 0.0;
            }
        }
    
        for (int i = tid; i < BK * BN / 4; i += n_thread) {
            int row = i * 4 / BN;
            int col = i * 4 % BN;
            int global_row = step + row;
            int global_col = blockIdx.x * BN + col;
            if (global_row < K && global_col < N) {
                const float4 v_b = reinterpret_cast<const float4 *>(b+global_row * N + global_col)[0]; 
                sb[row][col] = v_b.x;
                sb[row][col + 1] = v_b.y;
                sb[row][col + 2] = v_b.z;  
                sb[row][col + 3] = v_b.w;            

            } else {
                sb[row][col] = 0;
                sb[row][col + 1] = 0;
                sb[row][col + 2] = 0;  
                sb[row][col + 3] = 0;     
            }
        }

        /*

        // ---- 加载 sb[BK][BN]：模式和 sa 正好相反 ----
        // row    = i*TN + threadIdx.y  ：步长 blockDim.y(=TN=8)，所以行是【跳着取】的
        //                                （ty -> ty, ty+8, ty+16, ty+24）
        // column = TN*threadIdx.x + j  ：j 从 0..TN-1，所以每个线程负责【连续的 TN 列】
        //                                （tx 0 -> 列 0..7，tx 1 -> 列 8..15 ...）
        for (int i = 0; i <  BK / TN; i++) {
            for (int j = 0; j < TN; j++) {
                int row = i * TN + threadIdx.y;
                int column = TN * threadIdx.x + j;
                int globalRow = step + row;
                int globalColumn = blockIdx.x * BN + threadIdx.x * TN + j;
                if (globalRow < K && globalColumn < N) {
                    sb[row][column] = b[globalRow * N + globalColumn];
                } else {
                    sb[row][column] = 0.0;
                }
            }
        }
        */
        __syncthreads();


        for (int k = 0; k < BK; k++) {
            float reg_a[TM] = {0.0};

            for (int i = 0; i < TM; i++) {
                reg_a[i] = sa[threadIdx.y * TM + i][k];
            }

            float reg_b[TN] = {0.0};
            for (int i = 0; i < TN; i++) {
                reg_b[i] = sb[k][threadIdx.x * TN + i];
            }

            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
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
    gemm_2d_thread_tiling_vectorized_load<<<grid, block>>>(a, b, c,  M,  N, K);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float tm = 0;
    cudaEventElapsedTime(&tm, start, stop);

    double gflops = 2.0 * M * N * K / (tm / 1e3) / 1e9;
    printf("gemm_2d_thread_tiling cost: %.3f ms, %.2f GFLOPS\n", tm, gflops);

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
