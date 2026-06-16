#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../../common/cuda_check.cuh"

__global__ void vectorAdd(const float* a, const float* b, float* c, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    c[index] = a[index] + b[index];
  }
}

void vectorAddCpu(const float* a, const float* b, float* c, int count) {
  for (int index = 0; index < count; ++index) {
    c[index] = a[index] + b[index];
  }
}

bool verify(const float* expected, const float* actual, int count) {
  constexpr float tolerance = 1.0e-6F;
  for (int index = 0; index < count; ++index) {
    if (std::fabs(expected[index] - actual[index]) > tolerance) {
      std::cerr << "Mismatch at index " << index
                << ": expected=" << expected[index]
                << ", actual=" << actual[index] << '\n';
      return false;
    }
  }
  return true;
}

bool runCase(int count) {
  std::vector<float> hostA(count);
  std::vector<float> hostB(count);
  std::vector<float> expected(count);
  std::vector<float> actual(count);

  for (int index = 0; index < count; ++index) {
    hostA[index] = static_cast<float>(index % 101) * 0.25F;
    hostB[index] = static_cast<float>(index % 37) * -0.5F;
  }
  vectorAddCpu(hostA.data(), hostB.data(), expected.data(), count);

  const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
  float* deviceA = nullptr;
  float* deviceB = nullptr;
  float* deviceC = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceA, bytes));
  CUDA_CHECK(cudaMalloc(&deviceB, bytes));
  CUDA_CHECK(cudaMalloc(&deviceC, bytes));

  CUDA_CHECK(
      cudaMemcpy(deviceA, hostA.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(deviceB, hostB.data(), bytes, cudaMemcpyHostToDevice));

  constexpr int threads = 256;
  const int blocks = (count + threads - 1) / threads;

  vectorAdd<<<blocks, threads>>>(deviceA, deviceB, deviceC, count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  constexpr int measuredIterations = 20;
  CUDA_CHECK(cudaEventRecord(start));
  for (int iteration = 0; iteration < measuredIterations; ++iteration) {
    vectorAdd<<<blocks, threads>>>(deviceA, deviceB, deviceC, count);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsedMs = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

  CUDA_CHECK(
      cudaMemcpy(actual.data(), deviceC, bytes, cudaMemcpyDeviceToHost));
  const bool passed = verify(expected.data(), actual.data(), count);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(deviceA));
  CUDA_CHECK(cudaFree(deviceB));
  CUDA_CHECK(cudaFree(deviceC));

  std::cout << "count=" << std::setw(8) << count
            << " blocks=" << std::setw(5) << blocks
            << " threads=" << threads << " average_kernel_ms="
            << std::fixed << std::setprecision(6)
            << elapsedMs / measuredIterations << ' '
            << (passed ? "PASS" : "FAIL") << '\n';
  return passed;
}

int main() {
  const std::vector<int> testSizes = {
      1, 31, 32, 33, 255, 256, 257, 1'000'003,
  };

  bool allPassed = true;
  for (int count : testSizes) {
    allPassed = runCase(count) && allPassed;
  }

  return allPassed ? 0 : 1;
}

