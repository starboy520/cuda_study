#include <cstdio>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define  CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)


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
  double cpu_sum = 0.0;                 // 用 double 累加做参考值（避免精度损失）
  for (int i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i);
    cpu_sum += h_a[i];
  }

  float *d_a = nullptr;
  float *d_blocksums = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
  // 第一轮要写 ceil(1M/256)=4096 个 block sum，分配够大即可
  CUDA_CHECK(cudaMalloc(&d_blocksums, 4096 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  // 多阶段归约：每轮把数组缩短约 256 倍，直到只剩 1 个值。
  // in 指向当前输入，out 指向当前输出，两者每轮乒乓交换。
  float* in = d_a;
  float* out = d_blocksums;
  int cur = n;
  while (cur > 1) {
    int blocks = (cur + 255) / 256;
    reduce_sum<<<blocks, 256>>>(in, out, cur);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cur = blocks;                       // 这一阶段产出 blocks 个值 → 下一阶段输入量
    std::swap(in, out);                 // 本轮输出当作下一轮输入
  }
  // 循环结束后，最终的 1 个总和在 in[0]（最后一次 swap 把输出换到了 in）

  float gpu_sum = 0.0f;
  CUDA_CHECK(cudaMemcpy(&gpu_sum, in, sizeof(float), cudaMemcpyDeviceToHost));

  // float 累加 50 万亿量级必然有精度误差，用相对容差比较，不能用 ==
  double rel_err = std::abs(gpu_sum - cpu_sum) / cpu_sum;
  bool ok = rel_err < 1e-4;

  printf("n=%d\n", n);
  printf("CPU sum (double) = %.1f\n", cpu_sum);
  printf("GPU sum (float)  = %.1f\n", gpu_sum);
  printf("relative error   = %.6e\n", rel_err);
  printf("reduce_sum result = %s\n", ok ? "PASS" : "FAIL");

  cudaFree(d_a);
  cudaFree(d_blocksums);
}


int main() {
    test_reduce_sum();
    return 0;   
}