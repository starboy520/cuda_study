
#include <cstdio>
#include <cmath>


#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)



constexpr int K =8;
CONSTEXPR int N = 1 << 20;

__global__ void example_local_memory(int* input_data, int* output_data, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > total) {
        return;
    }

    float acc = input_data[idx];
