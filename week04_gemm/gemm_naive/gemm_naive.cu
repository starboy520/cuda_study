#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include "../../common/cuda_check.cuh"

__global__ void gemm_naive(float* a, float* b, float* c, int M, int N, int K) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int column = threadIdx.x + blockDim.x * blockIdx.x;

    if (row < M && column < N) {
        //C[M][N] = a[M][K] * B[K][N]
        float sum = 0.0f;
        for (int i = 0; i < K ; i++) {
            //b[i][column]  b[i*N+column]
            sum += a[row * K + i] * b[i * N + column];
        }
        c[row * N + column] = sum;
    }
}


float test_matmul_gpu(const float* A, const float* B, float* C, int M, int N,
    int K) {
float* d_A = nullptr;
float* d_B = nullptr;
float* d_C = nullptr;

CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

dim3 block(16, 16);
dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

cudaEvent_t start, stop;
CUDA_CHECK(cudaEventCreate(&start));
CUDA_CHECK(cudaEventCreate(&stop));

CUDA_CHECK(cudaEventRecord(start, 0));
gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
CUDA_CHECK(cudaEventRecord(stop, 0));
CUDA_CHECK(cudaEventSynchronize(stop));

float kernel_ms = 0.0f;
CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
CUDA_CHECK(cudaGetLastError());

CUDA_CHECK(cudaEventDestroy(start));
CUDA_CHECK(cudaEventDestroy(stop));

CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

CUDA_CHECK(cudaFree(d_A));
CUDA_CHECK(cudaFree(d_B));
CUDA_CHECK(cudaFree(d_C));

return kernel_ms;
}

// ---- CPU 参考：三重循环 GEMM ----
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];   // A[i][k] * B[k][j]
            }
            C[i * N + j] = sum;
        }
    }
}

// ---- 校验：相对误差容差（float 累加有误差）----
bool check_result(const float* C_cpu, const float* C_gpu, int M, int N,
                  float rel_eps = 1e-3f) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            const float cpu = C_cpu[i * N + j];
            const float gpu = C_gpu[i * N + j];
            const float diff = std::fabs(cpu - gpu);
            const float denom = std::max({std::fabs(cpu), std::fabs(gpu), 1.0f});
            if (diff / denom > rel_eps) {
                std::cerr << "FAIL at (" << i << "," << j << "): cpu=" << cpu
                          << " gpu=" << gpu << " rel=" << diff / denom << "\n";
                return false;
            }
        }
    }
    std::cout << "PASS\n";
    return true;
}

int main(int argc, char** argv) {
    // 默认 512³；可命令行指定非方阵：./gemm_naive M N K
    int M = 512, N = 512, K = 512;
    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    printf("M=%d N=%d K=%d\n", M, N, K);

    std::vector<float> A(M * K), B(K * N), C_cpu(M * N, 0.0f), C_gpu(M * N, 0.0f);
    // 用小值填充，避免 float 累加溢出（值太大校验会假 FAIL）
    for (int i = 0; i < M * K; i++) A[i] = static_cast<float>((i % 13) - 6) * 0.1f;
    for (int i = 0; i < K * N; i++) B[i] = static_cast<float>((i % 7) - 3) * 0.1f;

    matmul_cpu(A.data(), B.data(), C_cpu.data(), M, N, K);
    const float ms = test_matmul_gpu(A.data(), B.data(), C_gpu.data(), M, N, K);

    const double flops  = 2.0 * M * N * K;            // FMA 算 2 FLOP
    const double gflops = flops / (ms / 1e3) / 1e9;   // GFLOPS = FLOP / 秒 / 1e9
    printf("Kernel time: %.3f ms\n", ms);
    printf("GFLOPS: %.3f\n", gflops);

    return check_result(C_cpu.data(), C_gpu.data(), M, N) ? 0 : 1;
}