#include <cstdio>


__global__ void raceKernel(int* in, int n) {
    __shared__ int value[256];
    if (threadIdx.x == 0) {
        value[threadIdx.x] = 42;
    }


    __syncthreads();
    if (threadIdx.x == 1) {
        in[0] = value[threadIdx.x - 1];
    }
}

#define CUDA_CHECK(call)    \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)


int main() {
    int* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(int)));

    raceKernel<<<1, 2>>>(d_in, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_in;
    CUDA_CHECK(cudaMemcpy(&h_in, d_in, sizeof(int), cudaMemcpyDeviceToHost));
    printf("h_in: %d\n", h_in);

    CUDA_CHECK(cudaFree(d_in));
    return 0;
}