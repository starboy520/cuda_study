#include <iostream>

#include "../../common/cuda_check.cuh"

__device__ __forceinline__ float fusedExpression(float x, float y) {
  return x * y + x;
}

__global__ void inspectKernel(const float* input, float* output, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = fusedExpression(input[index], 2.0F);
  }
}

int main() {
  constexpr int count = 32;
  float *input = nullptr, *output = nullptr;
  CUDA_CHECK(cudaMalloc(&input, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&output, count * sizeof(float)));
  inspectKernel<<<1, 32>>>(input, output, count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(input));
  CUDA_CHECK(cudaFree(output));
  std::cout << "PASS\n";
  return 0;
}

