#include <cublas_v2.h>
#include "../common/cuda_check.cuh"

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

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    float alpha = 1.0, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("cuBLAS: %.3f ms, %.2f GFLOPS\n", ms, 2.0*M*N*K/(ms/1e3)/1e9);
    cublasDestroy(handle);


}