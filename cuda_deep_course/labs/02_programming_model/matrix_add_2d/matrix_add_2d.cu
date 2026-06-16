#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../common/cuda_check.cuh"

__global__ void matrixAdd(const float* a, const float* b, float* c, int width,
                          int height) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < height && col < width) {
    const int index = row * width + col;
    c[index] = a[index] + b[index];
  }
}

int main(int argc, char** argv) {
  int width = 37;
  int height = 23;
  if (argc >= 3) {
    width = std::atoi(argv[1]);
    height = std::atoi(argv[2]);
  }
  if (width <= 0 || height <= 0) {
    return 1;
  }

  const int count = width * height;
  const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
  std::vector<float> a(count), b(count), c(count);
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const int index = row * width + col;
      a[index] = static_cast<float>(row);
      b[index] = static_cast<float>(col);
    }
  }

  float *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceA, bytes));
  CUDA_CHECK(cudaMalloc(&deviceB, bytes));
  CUDA_CHECK(cudaMalloc(&deviceC, bytes));
  CUDA_CHECK(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice));

  const dim3 block(16, 8);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  matrixAdd<<<grid, block>>>(deviceA, deviceB, deviceC, width, height);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(c.data(), deviceC, bytes, cudaMemcpyDeviceToHost));

  bool passed = true;
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const int index = row * width + col;
      if (std::fabs(c[index] - (a[index] + b[index])) > 1.0e-6F) {
        passed = false;
      }
    }
  }

  std::cout << "matrix=" << height << " rows x " << width
            << " cols, block=(" << block.x << ',' << block.y << "), grid=("
            << grid.x << ',' << grid.y << ") "
            << (passed ? "PASS" : "FAIL") << '\n';

  CUDA_CHECK(cudaFree(deviceA));
  CUDA_CHECK(cudaFree(deviceB));
  CUDA_CHECK(cudaFree(deviceC));
  return passed ? 0 : 1;
}

