#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../../common/cuda_check.cuh"

__global__ void copyContiguous(const float* input, float* output, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = input[index];
  }
}

__global__ void copyStrided(const float* input, float* output, int count,
                            int stride) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const int source = static_cast<int>(
        (static_cast<long long>(index) * stride) % count);
    output[index] = input[source];
  }
}

float measureContiguous(const float* input, float* output, int count,
                        int iterations) {
  constexpr int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  copyContiguous<<<blocks, threads>>>(input, output, count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    copyContiguous<<<blocks, threads>>>(input, output, count);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float milliseconds = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return milliseconds / iterations;
}

float measureStrided(const float* input, float* output, int count, int stride,
                     int iterations) {
  constexpr int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  copyStrided<<<blocks, threads>>>(input, output, count, stride);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    copyStrided<<<blocks, threads>>>(input, output, count, stride);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float milliseconds = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return milliseconds / iterations;
}

bool verify(const std::vector<float>& input, const std::vector<float>& output,
            int stride) {
  const int count = static_cast<int>(input.size());
  for (int index = 0; index < count; ++index) {
    const int source = static_cast<int>(
        (static_cast<long long>(index) * stride) % count);
    if (output[index] != input[source]) {
      std::cerr << "Mismatch at " << index << '\n';
      return false;
    }
  }
  return true;
}

int main() {
  constexpr int count = 1 << 22;
  constexpr int iterations = 50;
  const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);

  std::vector<float> hostInput(count);
  std::vector<float> hostOutput(count);
  for (int index = 0; index < count; ++index) {
    hostInput[index] = static_cast<float>(index);
  }

  float *deviceInput = nullptr, *deviceOutput = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
  CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));
  CUDA_CHECK(cudaMemcpy(deviceInput, hostInput.data(), bytes,
                        cudaMemcpyHostToDevice));

  const float contiguousMs =
      measureContiguous(deviceInput, deviceOutput, count, iterations);
  CUDA_CHECK(cudaMemcpy(hostOutput.data(), deviceOutput, bytes,
                        cudaMemcpyDeviceToHost));
  const bool contiguousOk = verify(hostInput, hostOutput, 1);

  constexpr int stride = 32;
  const float stridedMs =
      measureStrided(deviceInput, deviceOutput, count, stride, iterations);
  CUDA_CHECK(cudaMemcpy(hostOutput.data(), deviceOutput, bytes,
                        cudaMemcpyDeviceToHost));
  const bool stridedOk = verify(hostInput, hostOutput, stride);

  const double movedBytes = 2.0 * static_cast<double>(bytes);
  const double contiguousGBs = movedBytes / (contiguousMs * 1.0e6);
  const double stridedLogicalGBs = movedBytes / (stridedMs * 1.0e6);

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "count=" << count << ", bytes/array=" << bytes << '\n';
  std::cout << "contiguous: " << contiguousMs << " ms, "
            << contiguousGBs << " logical GB/s, "
            << (contiguousOk ? "PASS" : "FAIL") << '\n';
  std::cout << "stride=" << stride << ": " << stridedMs << " ms, "
            << stridedLogicalGBs << " logical GB/s, "
            << (stridedOk ? "PASS" : "FAIL") << '\n';
  std::cout << "Logical GB/s counts requested float reads and writes; "
               "actual DRAM transactions may be larger.\n";

  CUDA_CHECK(cudaFree(deviceInput));
  CUDA_CHECK(cudaFree(deviceOutput));
  return contiguousOk && stridedOk ? 0 : 1;
}

