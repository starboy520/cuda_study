#include <cmath>
#include <iostream>
#include <vector>

#include "../../common/cuda_check.cuh"

__global__ void scaleKernel(float* data, int count, float scale) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    data[index] *= scale;
  }
}

int main(int argc, char** argv) {
  int count = 1000;
  if (argc >= 2) {
    count = std::stoi(argv[1]);
  }
  if (count <= 0) {
    std::cerr << "count must be positive\n";
    return 1;
  }

  constexpr float scale = 2.5F;
  const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
  std::vector<float> host(count);
  for (int index = 0; index < count; ++index) {
    host[index] = static_cast<float>(index) * 0.25F;
  }

  float* device = nullptr;
  std::cout << "1. cudaMalloc " << bytes << " bytes\n";
  CUDA_CHECK(cudaMalloc(&device, bytes));

  std::cout << "2. cudaMemcpy Host -> Device\n";
  CUDA_CHECK(
      cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice));

  constexpr int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  std::cout << "3. launch scaleKernel<<<" << blocks << ", " << threads
            << ">>>\n";
  scaleKernel<<<blocks, threads>>>(device, count, scale);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::cout << "4. cudaMemcpy Device -> Host\n";
  CUDA_CHECK(
      cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));

  bool passed = true;
  for (int index = 0; index < count; ++index) {
    const float expected = static_cast<float>(index) * 0.25F * scale;
    if (std::fabs(host[index] - expected) > 1.0e-6F) {
      std::cerr << "Mismatch at " << index << '\n';
      passed = false;
      break;
    }
  }

  std::cout << "5. cudaFree\n";
  CUDA_CHECK(cudaFree(device));
  std::cout << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}

