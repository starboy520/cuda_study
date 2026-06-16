#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../../common/cuda_check.cuh"

constexpr int kThreads = 256;

__global__ void reduceShared(const float* input, float* output, int count) {
  __shared__ float values[kThreads];
  const int tid = threadIdx.x;
  const int start = blockIdx.x * blockDim.x * 2 + tid;

  float sum = 0.0F;
  if (start < count) {
    sum += input[start];
  }
  if (start + blockDim.x < count) {
    sum += input[start + blockDim.x];
  }
  values[tid] = sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
      values[tid] += values[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[blockIdx.x] = values[0];
  }
}

__global__ void reduceWarpShuffle(const float* input, float* output,
                                  int count) {
  __shared__ float values[kThreads];
  const int tid = threadIdx.x;
  const int start = blockIdx.x * blockDim.x * 2 + tid;

  float sum = 0.0F;
  if (start < count) {
    sum += input[start];
  }
  if (start + blockDim.x < count) {
    sum += input[start + blockDim.x];
  }
  values[tid] = sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset >= 32; offset /= 2) {
    if (tid < offset) {
      values[tid] += values[tid + offset];
    }
    __syncthreads();
  }

  if (tid < 32) {
    sum = values[tid];
    sum += __shfl_down_sync(0xffffffffU, sum, 16);
    sum += __shfl_down_sync(0xffffffffU, sum, 8);
    sum += __shfl_down_sync(0xffffffffU, sum, 4);
    sum += __shfl_down_sync(0xffffffffU, sum, 2);
    sum += __shfl_down_sync(0xffffffffU, sum, 1);
    if (tid == 0) {
      output[blockIdx.x] = sum;
    }
  }
}

using ReductionKernel = void (*)(const float*, float*, int);

float runReduction(ReductionKernel kernel, const float* input, float* bufferA,
                   float* bufferB, int count, float& result) {
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  const float* currentInput = input;
  float* currentOutput = bufferA;
  int currentCount = count;
  while (currentCount > 1) {
    const int blocks =
        (currentCount + kThreads * 2 - 1) / (kThreads * 2);
    kernel<<<blocks, kThreads>>>(currentInput, currentOutput, currentCount);
    CUDA_CHECK(cudaGetLastError());
    currentCount = blocks;
    currentInput = currentOutput;
    currentOutput = (currentOutput == bufferA) ? bufferB : bufferA;
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  CUDA_CHECK(cudaMemcpy(&result, currentInput, sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return milliseconds;
}

int main(int argc, char** argv) {
  int count = 1'000'003;
  if (argc >= 2) {
    count = std::atoi(argv[1]);
  }
  if (count <= 0) {
    std::cerr << "count must be positive\n";
    return 1;
  }

  std::vector<float> input(count);
  double expected = 0.0;
  for (int i = 0; i < count; ++i) {
    input[i] = static_cast<float>((i % 17) - 8) * 0.125F;
    expected += input[i];
  }

  float *deviceInput = nullptr, *bufferA = nullptr, *bufferB = nullptr;
  const std::size_t inputBytes =
      static_cast<std::size_t>(count) * sizeof(float);
  const int firstBlocks = (count + kThreads * 2 - 1) / (kThreads * 2);
  const std::size_t bufferBytes =
      static_cast<std::size_t>(std::max(firstBlocks, 1)) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&deviceInput, inputBytes));
  CUDA_CHECK(cudaMalloc(&bufferA, bufferBytes));
  CUDA_CHECK(cudaMalloc(&bufferB, bufferBytes));
  CUDA_CHECK(cudaMemcpy(deviceInput, input.data(), inputBytes,
                        cudaMemcpyHostToDevice));

  float sharedResult = 0.0F;
  float warpResult = 0.0F;
  const float sharedMs = runReduction(
      reduceShared, deviceInput, bufferA, bufferB, count, sharedResult);
  const float warpMs = runReduction(
      reduceWarpShuffle, deviceInput, bufferA, bufferB, count, warpResult);

  const double tolerance =
      std::max(1.0e-3, std::abs(expected) * 1.0e-5);
  const bool sharedOk = std::abs(sharedResult - expected) <= tolerance;
  const bool warpOk = std::abs(warpResult - expected) <= tolerance;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "count=" << count << ", expected=" << expected << '\n';
  std::cout << "reduceShared:      result=" << sharedResult
            << ", time=" << sharedMs << " ms, "
            << (sharedOk ? "PASS" : "FAIL") << '\n';
  std::cout << "reduceWarpShuffle: result=" << warpResult
            << ", time=" << warpMs << " ms, "
            << (warpOk ? "PASS" : "FAIL") << '\n';

  CUDA_CHECK(cudaFree(deviceInput));
  CUDA_CHECK(cudaFree(bufferA));
  CUDA_CHECK(cudaFree(bufferB));
  return sharedOk && warpOk ? 0 : 1;
}
