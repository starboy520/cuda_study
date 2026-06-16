#include <iostream>
#include <vector>

#include "../../common/cuda_check.cuh"

__host__ __device__ float square(float value) {
  return value * value;
}

__device__ float addBias(float value, float bias) {
  return value + bias;
}

__global__ void transformKernel(const float* input, float* output, int count,
                                float bias) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = square(addBias(input[index], bias));
  }
}

int main() {
  constexpr int count = 8;
  constexpr float bias = 0.5F;
  std::vector<float> input(count);
  std::vector<float> output(count);

  for (int i = 0; i < count; ++i) {
    input[i] = static_cast<float>(i);
  }

  std::cout << "Host calls __host__ __device__ square(3) = "
            << square(3.0F) << '\n';

  float *deviceInput = nullptr, *deviceOutput = nullptr;
  const std::size_t bytes = count * sizeof(float);
  CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
  CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));
  CUDA_CHECK(cudaMemcpy(deviceInput, input.data(), bytes,
                        cudaMemcpyHostToDevice));

  transformKernel<<<1, 32>>>(deviceInput, deviceOutput, count, bias);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), deviceOutput, bytes,
                        cudaMemcpyDeviceToHost));

  bool passed = true;
  for (int i = 0; i < count; ++i) {
    const float expected = square(input[i] + bias);
    passed = passed && output[i] == expected;
    std::cout << "input=" << input[i] << " output=" << output[i]
              << " expected=" << expected << '\n';
  }

  CUDA_CHECK(cudaFree(deviceInput));
  CUDA_CHECK(cudaFree(deviceOutput));
  std::cout << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}

