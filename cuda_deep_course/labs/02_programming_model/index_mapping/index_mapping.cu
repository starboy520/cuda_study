#include <cstdio>

#include "../../common/cuda_check.cuh"

__global__ void print1DMapping() {
  const int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int linearThread = threadIdx.x;
  const int warp = linearThread / warpSize;
  const int lane = linearThread % warpSize;
  printf("1D block=%u thread=%u -> linear=%d warp=%d lane=%d global=%d\n",
         blockIdx.x, threadIdx.x, linearThread, warp, lane, globalIndex);
}

__global__ void print2DMapping(int width) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int linearIndex = row * width + col;
  const int linearThread = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp = linearThread / warpSize;
  const int lane = linearThread % warpSize;

  printf("2D block=(%u,%u) local=(row=%u,col=%u) -> "
         "thread_linear=%d warp=%d lane=%d "
         "global=(row=%d,col=%d) matrix_linear=%d\n",
         blockIdx.x, blockIdx.y, threadIdx.y, threadIdx.x, linearThread, warp,
         lane, row, col, linearIndex);
}

int main() {
  std::printf("1D mapping: grid=1 block, block=40 threads\n");
  print1DMapping<<<1, 40>>>();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::printf("\n2D mapping: grid=(1,1), block=(16 columns,3 rows)\n");
  constexpr int width = 16;
  const dim3 block(16, 3);
  const dim3 grid(1, 1);
  print2DMapping<<<grid, block>>>(width);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::printf(
      "\nNote: device printf order is not a thread synchronization guarantee.\n");
  return 0;
}
