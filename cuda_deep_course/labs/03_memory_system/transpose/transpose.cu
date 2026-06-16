#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../../common/cuda_check.cuh"

constexpr int kTile = 32;
constexpr int kBlockRows = 8;

__global__ void transposeNaive(const float* input, float* output, int width,
                               int height) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < height && col < width) {
    output[col * height + row] = input[row * width + col];
  }
}

__global__ void transposeShared(const float* input, float* output, int width,
                                int height) {
  __shared__ float tile[kTile][kTile + 1];

  const int inputCol = blockIdx.x * kTile + threadIdx.x;
  const int inputRowBase = blockIdx.y * kTile + threadIdx.y;

  for (int offset = 0; offset < kTile; offset += kBlockRows) {
    const int inputRow = inputRowBase + offset;
    if (inputRow < height && inputCol < width) {
      tile[threadIdx.y + offset][threadIdx.x] =
          input[inputRow * width + inputCol];
    }
  }
  __syncthreads();

  const int outputCol = blockIdx.y * kTile + threadIdx.x;
  const int outputRowBase = blockIdx.x * kTile + threadIdx.y;

  for (int offset = 0; offset < kTile; offset += kBlockRows) {
    const int outputRow = outputRowBase + offset;
    if (outputRow < width && outputCol < height) {
      output[outputRow * height + outputCol] =
          tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}

void transposeCpu(const std::vector<float>& input, std::vector<float>& output,
                  int width, int height) {
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      output[col * height + row] = input[row * width + col];
    }
  }
}

bool verify(const std::vector<float>& expected,
            const std::vector<float>& actual) {
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (expected[index] != actual[index]) {
      std::cerr << "Mismatch at linear index " << index
                << ": expected=" << expected[index]
                << ", actual=" << actual[index] << '\n';
      return false;
    }
  }
  return true;
}

template <typename Launch>
float measure(Launch launch, int iterations) {
  launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return milliseconds / iterations;
}

int main(int argc, char** argv) {
  int width = 1003;
  int height = 769;
  if (argc >= 3) {
    width = std::atoi(argv[1]);
    height = std::atoi(argv[2]);
  }
  if (width <= 0 || height <= 0) {
    std::cerr << "Usage: ./transpose [width height], both positive\n";
    return 1;
  }

  const std::size_t count =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const std::size_t bytes = count * sizeof(float);
  std::vector<float> input(count);
  std::vector<float> expected(count);
  std::vector<float> actual(count);
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      input[row * width + col] = static_cast<float>(row * width + col);
    }
  }
  transposeCpu(input, expected, width, height);

  float *deviceInput = nullptr, *deviceOutput = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
  CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));
  CUDA_CHECK(cudaMemcpy(deviceInput, input.data(), bytes,
                        cudaMemcpyHostToDevice));

  constexpr int iterations = 50;
  const dim3 naiveBlock(32, 8);
  const dim3 naiveGrid((width + naiveBlock.x - 1) / naiveBlock.x,
                       (height + naiveBlock.y - 1) / naiveBlock.y);
  const float naiveMs = measure(
      [&] {
        transposeNaive<<<naiveGrid, naiveBlock>>>(
            deviceInput, deviceOutput, width, height);
      },
      iterations);
  CUDA_CHECK(cudaMemcpy(actual.data(), deviceOutput, bytes,
                        cudaMemcpyDeviceToHost));
  const bool naiveOk = verify(expected, actual);

  const dim3 sharedBlock(kTile, kBlockRows);
  const dim3 sharedGrid((width + kTile - 1) / kTile,
                        (height + kTile - 1) / kTile);
  const float sharedMs = measure(
      [&] {
        transposeShared<<<sharedGrid, sharedBlock>>>(
            deviceInput, deviceOutput, width, height);
      },
      iterations);
  CUDA_CHECK(cudaMemcpy(actual.data(), deviceOutput, bytes,
                        cudaMemcpyDeviceToHost));
  const bool sharedOk = verify(expected, actual);

  const double movedBytes = 2.0 * static_cast<double>(bytes);
  std::cout << "input=" << height << " rows x " << width
            << " cols, output=" << width << " rows x " << height
            << " cols\n";
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "transposeNaive:  " << naiveMs << " ms, "
            << movedBytes / (naiveMs * 1.0e6) << " GB/s, "
            << (naiveOk ? "PASS" : "FAIL") << '\n';
  std::cout << "transposeShared: " << sharedMs << " ms, "
            << movedBytes / (sharedMs * 1.0e6) << " GB/s, "
            << (sharedOk ? "PASS" : "FAIL") << '\n';

  CUDA_CHECK(cudaFree(deviceInput));
  CUDA_CHECK(cudaFree(deviceOutput));
  return naiveOk && sharedOk ? 0 : 1;
}

