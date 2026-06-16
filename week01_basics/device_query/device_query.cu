#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

void print_device(int device) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  printf("--- device %d ---\n", device);
  printf("GPU: %s\n", prop.name);
  printf("SM count: %d\n", prop.multiProcessorCount);
  printf("compute capability: %d.%d\n", prop.major, prop.minor);
  printf("warp size: %d\n", prop.warpSize);
  printf("max threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
  printf("max warps per SM: %d\n",
         prop.maxThreadsPerMultiProcessor / prop.warpSize);
  printf("registers per block: %d\n", prop.regsPerBlock);
  printf("registers per SM: %d\n", prop.regsPerMultiprocessor);
  printf("shared memory per block: %zu bytes (%.1f KB)\n", prop.sharedMemPerBlock,
         prop.sharedMemPerBlock / 1024.0);
  printf("shared memory per SM:    %zu bytes (%.1f KB)\n",
         prop.sharedMemPerMultiprocessor,
         prop.sharedMemPerMultiprocessor / 1024.0);
  printf("global memory: %zu bytes (%.1f GB)\n", prop.totalGlobalMem,
         prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
  printf("max threads dim: %d %d %d\n", prop.maxThreadsDim[0],
         prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("max blocks per multiprocessor: %d\n",
         prop.maxBlocksPerMultiProcessor);
  printf("max grid size: %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1],
         prop.maxGridSize[2]);
  printf("total constant memory: %zu bytes (%.1f KB)\n", prop.totalConstMem,
         prop.totalConstMem / 1024.0);
  printf("\n");
}

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  printf("CUDA device count: %d\n\n", device_count);

  for (int device = 0; device < device_count; ++device) {
    print_device(device);
  }

  return 0;
}
