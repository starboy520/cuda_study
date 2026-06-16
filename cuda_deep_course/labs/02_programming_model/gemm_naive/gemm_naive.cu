#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../../common/cuda_check.cuh"

__global__ void gemmNaive(const float* a, const float* b, float* c, int m,
                          int n, int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    float sum = 0.0F;
    for (int inner = 0; inner < k; ++inner) {
      sum += a[row * k + inner] * b[inner * n + col];
    }
    c[row * n + col] = sum;
  }
}

void gemmCpu(const std::vector<float>& a, const std::vector<float>& b,
             std::vector<float>& c, int m, int n, int k) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      double sum = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        sum += static_cast<double>(a[row * k + inner]) *
               static_cast<double>(b[inner * n + col]);
      }
      c[row * n + col] = static_cast<float>(sum);
    }
  }
}

int main(int argc, char** argv) {
  int m = 37;
  int n = 29;
  int k = 41;
  if (argc >= 4) {
    m = std::atoi(argv[1]);
    n = std::atoi(argv[2]);
    k = std::atoi(argv[3]);
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return 1;
  }

  std::vector<float> a(static_cast<std::size_t>(m) * k);
  std::vector<float> b(static_cast<std::size_t>(k) * n);
  std::vector<float> expected(static_cast<std::size_t>(m) * n);
  std::vector<float> actual(static_cast<std::size_t>(m) * n);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const int value = static_cast<int>(i % 11) - 5;
    a[i] = static_cast<float>(value) * 0.1F;
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    const int value = static_cast<int>(i % 7) - 3;
    b[i] = static_cast<float>(value) * 0.2F;
  }
  gemmCpu(a, b, expected, m, n, k);

  float *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceA, a.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&deviceB, b.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&deviceC, actual.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(deviceA, a.data(), a.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(deviceB, b.data(), b.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  const dim3 block(16, 16);
  const dim3 grid((n + block.x - 1) / block.x,
                  (m + block.y - 1) / block.y);

  gemmNaive<<<grid, block>>>(deviceA, deviceB, deviceC, m, n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  gemmNaive<<<grid, block>>>(deviceA, deviceB, deviceC, m, n, k);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  CUDA_CHECK(cudaMemcpy(actual.data(), deviceC, actual.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  bool passed = true;
  float maxError = 0.0F;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    maxError = std::max(maxError, std::fabs(actual[i] - expected[i]));
    if (std::fabs(actual[i] - expected[i]) > 1.0e-3F) {
      passed = false;
    }
  }

  const double operations =
      2.0 * static_cast<double>(m) * n * k;
  const double gflops = operations / (milliseconds * 1.0e6);
  std::cout << "A=" << m << 'x' << k << ", B=" << k << 'x' << n
            << ", C=" << m << 'x' << n << '\n';
  std::cout << std::fixed << std::setprecision(4)
            << "time=" << milliseconds << " ms, GFLOPS=" << gflops
            << ", max_error=" << maxError << ' '
            << (passed ? "PASS" : "FAIL") << '\n';

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(deviceA));
  CUDA_CHECK(cudaFree(deviceB));
  CUDA_CHECK(cudaFree(deviceC));
  return passed ? 0 : 1;
}
