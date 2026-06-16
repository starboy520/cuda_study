#include <chrono>
#include <iomanip>
#include <iostream>

#include "../../common/cuda_check.cuh"

__global__ void repeatAdd(float* data, int count, int repeats) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    float value = data[index];
    for (int i = 0; i < repeats; ++i) {
      value += 1.0F;
    }
    data[index] = value;
  }
}

int main() {
  constexpr int count = 1 << 20;
  constexpr int repeats = 256;
  constexpr int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);

  float* data = nullptr;
  CUDA_CHECK(cudaMalloc(&data, bytes));
  CUDA_CHECK(cudaMemset(data, 0, bytes));

  repeatAdd<<<blocks, threads>>>(data, count, repeats);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  const auto launchStart = std::chrono::steady_clock::now();
  repeatAdd<<<blocks, threads>>>(data, count, repeats);
  const auto launchStop = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaGetLastError());
  const double launchOnlyUs =
      std::chrono::duration<double, std::micro>(
          launchStop - launchStart).count();

  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  repeatAdd<<<blocks, threads>>>(data, count, repeats);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float eventMs = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&eventMs, start, stop));

  const auto endToEndStart = std::chrono::steady_clock::now();
  repeatAdd<<<blocks, threads>>>(data, count, repeats);
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto endToEndStop = std::chrono::steady_clock::now();
  const double synchronizedMs =
      std::chrono::duration<double, std::milli>(
          endToEndStop - endToEndStart).count();

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "CPU timer around launch only: " << launchOnlyUs << " us\n";
  std::cout << "CUDA Event kernel interval: " << eventMs << " ms\n";
  std::cout << "CPU timer with synchronize: " << synchronizedMs << " ms\n";
  std::cout << "Launch-only time is not kernel execution time.\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(data));
  return 0;
}

