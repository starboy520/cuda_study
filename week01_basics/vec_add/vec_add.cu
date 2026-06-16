#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

void print_device_shared_memory() {
  int device = 0;
  cudaDeviceProp prop{};

  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  printf("GPU: %s\n", prop.name);
  printf("SM count: %d\n", prop.multiProcessorCount);
  printf("compute capability: %d.%d\n", prop.major, prop.minor);
  printf("warp size: %d\n", prop.warpSize);
  printf("max threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("shared memory per block: %zu bytes (%.1f KB)\n",
         prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0);
  printf("shared memory per SM:    %zu bytes (%.1f KB)\n",
         prop.sharedMemPerMultiprocessor,
         prop.sharedMemPerMultiprocessor / 1024.0);
  printf("global memory: %zu bytes (%.1f GB)\n",
         prop.totalGlobalMem, prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
}

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// 学习怎么使用shared_memory 
__global__ void reduce_sum(const float* input, float* blocksums, int n) {
  __shared__ float sdata[256];
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_id < n) {
    sdata[threadIdx.x] = input[global_id];
  } else {
    sdata[threadIdx.x] = 0;
  }

  __syncthreads();

  for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    blocksums[blockIdx.x] = sdata[0];
  }
}


void test_reduce_sum() {
  int n = 1 << 20;
  std::vector<float> h_a(n);
  for (int i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i);
  }
  

  float *d_a = nullptr;
  float *d_blocksums = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_blocksums, 4096 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  reduce_sum<<<4096, 256>>>(d_a, d_blocksums, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float *h_blocksums = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_blocksums, 4096 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(h_blocksums, d_blocksums, 4096 * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < 4096; ++i) {
    int start = i * 256;
    int end = start + 255;
    float expected = (start + end) * 256 / 2.0f;
    if (h_blocksums[i] != expected) {
      ok = false;
      printf("block %d sum=%f, expected=%f\n", i, h_blocksums[i], expected);
    }
  }

  printf("reduce_sum result=%s\n", ok ? "PASS" : "FAIL");

  cudaFree(d_a);
  cudaFree(d_blocksums);
  cudaFreeHost(h_blocksums);
}

int main(int argc, char** argv) {
  print_device_shared_memory();

  int n = 1 << 20;  // 1M elements
  if (argc > 1) {
    n = std::atoi(argv[1]);
  }

  std::vector<float> h_a(n), h_b(n), h_c(n);
  for (int i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(i * 2);
  }

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  size_t bytes = n * sizeof(float);


  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  CUDA_CHECK(cudaEventRecord(start, 0));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float h2d_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));
  printf("H2D time: %f ms\n", h2d_ms);


  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  CUDA_CHECK(cudaEventRecord(start, 0));
  vec_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
  printf("Kernel time: %f ms\n", kernel_ms);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start, 0));
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float d2h_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));
  printf("D2H time: %f ms\n", d2h_ms);

  bool ok = true;
  for (int i = 0; i < n; ++i) {
    if (h_c[i] != h_a[i] + h_b[i]) {
      ok = false;
      break;
    }
  }
  printf("n=%d, result=%s\n", n, ok ? "PASS" : "FAIL");
  printf("total timed: %f ms\n", h2d_ms + kernel_ms + d2h_ms);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  test_reduce_sum();

  return ok ? 0 : 1;
}
