#include <cstdio>
#include <cstdlib>
#include <ctime>
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

__global__ void example_sync(const int* input_data, int* output_data,
                             int total) {
  __shared__ int shared_data[256];

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < total) {
    shared_data[threadIdx.x] = input_data[idx];
  } else {
    shared_data[threadIdx.x] = 0;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int sum = 0;
    for (int i = 0; i < blockDim.x; ++i) {
      sum += shared_data[i];
    }
    output_data[blockIdx.x] = sum;
  }
}

void init_array(int* array, int size) {
  std::srand(static_cast<unsigned>(std::time(nullptr)));
  for (int i = 0; i < size; ++i) {
    array[i] = static_cast<int>(10.f * (std::rand() / static_cast<float>(RAND_MAX)));
  }
}

int cpu_block_sum(const int* input, int block_idx, int block_size, int total) {
  const int offset = block_idx * block_size;
  const int count = (offset + block_size <= total) ? block_size : (total - offset);
  int sum = 0;
  for (int j = 0; j < count; ++j) {
    sum += input[offset + j];
  }
  return sum;
}

int main(int argc, char** argv) {
  const int block_size = 256;
  int input_length = 2048;
  if (argc > 1) {
    input_length = std::atoi(argv[1]);
  }

  const int grid_size = (input_length + block_size - 1) / block_size;

  int* h_input = new int[input_length];
  int* h_output = new int[grid_size];
  init_array(h_input, input_length);

  int *d_input = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input_length * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_output, grid_size * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input, input_length * sizeof(int),
                        cudaMemcpyHostToDevice));

  example_sync<<<grid_size, block_size>>>(d_input, d_output, input_length);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_output, d_output, grid_size * sizeof(int),
                        cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < grid_size; ++i) {
    const int ref = cpu_block_sum(h_input, i, block_size, input_length);
    if (h_output[i] != ref) {
      printf("FAIL at block %d: gpu=%d cpu=%d\n", i, h_output[i], ref);
      ok = false;
    }
  }

  printf("input_length=%d grid_size=%d block_size=%d\n", input_length, grid_size,
         block_size);
  printf("%s\n", ok ? "PASS" : "FAIL");

  delete[] h_input;
  delete[] h_output;
  cudaFree(d_input);
  cudaFree(d_output);

  return ok ? 0 : 1;
}
