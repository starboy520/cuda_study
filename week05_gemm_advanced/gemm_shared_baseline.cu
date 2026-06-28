#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

constexpr int TILE = 32;

// 这里每个block 大小是(tile*tile)
__global__ void gemm_shared_baseline(const float* a, const float* b, 
        float* c, int M, int N, int K) {
    // 共享内存，tile的大小是TILE*TILE, 但是为了避免bank conflict，第二维多开一个元素
    // 其实就是: 每个现成话对应一个c的元素， 每个block对应一个tile的c
    __shared__ float TILEA[TILE][TILE+1];
    __shared__ float TILEB[TILE][TILE+1];

    // c的行列， 比较好理解，每个thread对应一个c的元素;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;

    // c[y][x] = sum(a[y][i]*b[i][x]), tile在 k方向展开，
    // c[threadidx.y][threadidx.x];
    float sum = 0.0f;
    for (int i = 0; i < (K+TILE-1) / TILE; i++) {
        // rowa 不变
        int columnA = i * TILE + threadIdx.x;
        if (row < M && columnA < K) {
            TILEA[threadIdx.y][threadIdx.x] = a[row * K + i * TILE + threadIdx.x];
        } else {
            TILEA[threadIdx.y][threadIdx.x] = 0.0;
        }

        int rowB = i * TILE + threadIdx.y;
        if (rowB < K && column < N) {
            TILEB[threadIdx.y][threadIdx.x] = b[rowB * N + column];
        } else {
            TILEB[threadIdx.y][threadIdx.x] = 0;
        }


        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += TILEA[threadIdx.y][k] * TILEB[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < M && column < N) {
        c[row * N + column] = sum;
    }
}

void gemm_shared_baseline_launcher(const float* a, const float* b, 
        float* c, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gemm_shared_baseline<<<grid, block>>>(a, b, c, M, N, K);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float tm = 0;
    cudaEventElapsedTime(&tm, start, stop);

    // GFLOPS = 2*M*N*K / time_s / 1e9 = 2*M*N*K / time_ms / 1e6
    double gflops = 2.0 * M * N * K / (tm / 1e3) / 1e9;
    printf("gemm_shared_baseline cost: %.3f ms, %.2f GFLOPS\n", tm, gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// CPU 参考实现，用于校验正确性
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
    // 默认 M=N=K=512，可通过命令行覆盖：
    //   ./gemm_shared_baseline 1024            -> M=N=K=1024
    //   ./gemm_shared_baseline 1024 512 768    -> M=1024 N=512 K=768
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

    // host 内存
    float* hA = (float*)malloc(bytesA);
    float* hB = (float*)malloc(bytesB);
    float* hC = (float*)malloc(bytesC);       // GPU 结果
    float* hRef = (float*)malloc(bytesC);     // CPU 参考结果

    // 初始化输入
    for (size_t i = 0; i < (size_t)M * K; i++) hA[i] = (float)(rand() % 10) / 10.0f;
    for (size_t i = 0; i < (size_t)K * N; i++) hB[i] = (float)(rand() % 10) / 10.0f;

    // device 内存
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    // warmup（不计时），让首次 launch 开销不污染计时
    gemm_shared_baseline_launcher(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());

    // 正式计时
    gemm_shared_baseline_launcher(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

    // CPU 校验
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