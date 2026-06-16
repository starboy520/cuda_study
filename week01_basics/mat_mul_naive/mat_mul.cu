#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void matmul_gpu(const float* A, const float* B, float* C, int M,
                           int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

struct ParseArgs {
  int M = 512;
  int N = 512;
  int K = 512;

  bool parse(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
        M = std::atoi(argv[++i]);
      } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
        N = std::atoi(argv[++i]);
      } else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
        K = std::atoi(argv[++i]);
      } else {
        std::cerr << "Unknown or incomplete arg: " << argv[i] << "\n";
        std::cerr << "Usage: " << argv[0] << " [--m M] [--n N] [--k K]\n";
        return false;
      }
    }
    return true;
  }
};

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
  matmul_gpu<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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

bool check_result(const float* C_cpu, const float* C_gpu, int M, int N,
                  float rel_eps = 1e-4f) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      const float cpu = C_cpu[i * N + j];
      const float gpu = C_gpu[i * N + j];
      const float diff = std::fabs(cpu - gpu);
      const float denom = std::max({std::fabs(cpu), std::fabs(gpu), 1.0f});
      if (diff / denom > rel_eps) {
        std::cerr << "FAIL at (" << i << ", " << j << "): cpu=" << cpu
                  << " gpu=" << gpu << " rel_diff=" << diff / denom
                  << std::endl;
        return false;
      }
    }
  }
  std::cout << "PASS" << std::endl;
  return true;
}

int main(int argc, char** argv) {
  ParseArgs args;
  if (!args.parse(argc, argv)) {
    return 1;
  }

  std::cout << "M=" << args.M << " N=" << args.N << " K=" << args.K << "\n";

  std::vector<float> A(args.M * args.K);
  std::vector<float> B(args.K * args.N);
  std::vector<float> C_cpu(args.M * args.N, 0.0f);
  std::vector<float> C_gpu(args.M * args.N, 0.0f);

  for (int i = 0; i < args.M; i++) {
    for (int j = 0; j < args.K; j++) {
      A[i * args.K + j] = static_cast<float>(i * args.K + j);
    }
  }

  for (int i = 0; i < args.K; i++) {
    for (int j = 0; j < args.N; j++) {
      B[i * args.N + j] = static_cast<float>(i * args.N + j);
    }
  }

  matmul_cpu(A.data(), B.data(), C_cpu.data(), args.M, args.N, args.K);

  const float kernel_ms =
      test_matmul_gpu(A.data(), B.data(), C_gpu.data(), args.M, args.N, args.K);

  const double flops = 2.0 * args.M * args.N * args.K;
  const double gflops = flops / static_cast<double>(kernel_ms) / 1e6;
  printf("Kernel time: %.3f ms\n", kernel_ms);
  printf("GFLOPS: %.3f\n", gflops);

  if (!check_result(C_cpu.data(), C_gpu.data(), args.M, args.N)) {
    return 1;
  }

  return 0;
}
