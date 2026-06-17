// softmax.cu
//
// 卷六实验：Softmax 数值稳定性 —— 朴素版(会溢出成 NaN) vs 稳定版(减最大值)。
//
// ============================================================================
// 目的：亲眼看到"为什么 softmax 必须减最大值"。
//   - softmaxNaive : 直接 exp(x_i)/Σexp(x_j)，大输入时 exp 溢出 → NaN
//   - softmaxStable : exp(x_i - max)/Σexp(x_j - max)，永不溢出，结果相同
//
// 用两组数据演示：
//   普通数据（小值）   → 两版都正常，结果一致
//   大值数据（含 1000）→ 朴素版 NaN，稳定版正常
//
// 编译：
//   nvcc -O3 -arch=sm_75 softmax.cu -o softmax
// 运行：
//   ./softmax
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "../../common/cuda_check.cuh"

constexpr int THREADS = 256;

// ---------------------------------------------------------------------------
// 朴素 softmax：一个 block 处理一行，直接 exp(x_i)。
// 大输入时 exp(x_i) 溢出（exp(89) 就超过 float 上限）→ inf/inf = NaN。
// ---------------------------------------------------------------------------
__global__ void softmaxNaive(const float* x, float* out, int rows, int cols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  __shared__ float red[THREADS];

  // 求 Σ exp(x_i)（没有减最大值）
  float localSum = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x)
    localSum += expf(x[row * cols + c]);     // 大值 → inf
  red[tid] = localSum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  float sum = red[0];
  __syncthreads();

  for (int c = tid; c < cols; c += blockDim.x)
    out[row * cols + c] = expf(x[row * cols + c]) / sum;  // inf/inf = NaN
}

// ---------------------------------------------------------------------------
// 稳定 softmax：先求每行最大值 m，再 exp(x_i - m)。
// 因 x_i - m <= 0 → exp <= 1，永不溢出。softmax 平移不变，结果与朴素版一致。
// ---------------------------------------------------------------------------
__global__ void softmaxStable(const float* x, float* out, int rows, int cols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  __shared__ float red[THREADS];

  // ① 求最大值（归约，算子是 fmaxf）
  float localMax = -INFINITY;
  for (int c = tid; c < cols; c += blockDim.x)
    localMax = fmaxf(localMax, x[row * cols + c]);
  red[tid] = localMax;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  float m = red[0];
  __syncthreads();

  // ② 求 Σ exp(x_i - m)（归约，求和）
  float localSum = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x)
    localSum += expf(x[row * cols + c] - m);
  red[tid] = localSum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  float sum = red[0];
  __syncthreads();

  // ③ 归一化
  for (int c = tid; c < cols; c += blockDim.x)
    out[row * cols + c] = expf(x[row * cols + c] - m) / sum;
}

using SoftmaxKernel = void (*)(const float*, float*, int, int);

static void run(const char* name, SoftmaxKernel kernel,
                const std::vector<float>& hx, int rows, int cols) {
  size_t n = (size_t)rows * cols;
  float *dx, *dout;
  CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  kernel<<<rows, THREADS>>>(dx, dout, rows, cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hout(n);
  CUDA_CHECK(cudaMemcpy(hout.data(), dout, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // 检查每行和是否 ≈ 1，以及是否有 NaN
  bool hasNaN = false;
  double rowSum = 0.0;
  for (int c = 0; c < cols; ++c) {
    if (std::isnan(hout[c])) hasNaN = true;
    rowSum += hout[c];
  }
  printf("  %-14s row0 sum=%.4f  %s\n", name, rowSum,
         hasNaN ? "出现 NaN ❌" : "正常 ✅");

  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dout));
}

int main() {
  const int rows = 4, cols = 1024;
  std::vector<float> small(rows * cols), big(rows * cols);

  // 普通数据：小值，两版都该正常
  for (int i = 0; i < rows * cols; ++i)
    small[i] = (float)((i % 21) - 10) * 0.1f;   // 约 [-1, 1]

  // 大值数据：含接近 1000 的值，朴素版 exp 必溢出
  for (int i = 0; i < rows * cols; ++i)
    big[i] = (float)(i % 1000) + 500.0f;        // 约 [500, 1500]

  printf("=== 普通数据（小值，约 [-1,1]）===\n");
  run("naive",  softmaxNaive,  small, rows, cols);
  run("stable", softmaxStable, small, rows, cols);

  printf("\n=== 大值数据（约 [500,1500]，exp 会溢出）===\n");
  run("naive",  softmaxNaive,  big, rows, cols);
  run("stable", softmaxStable, big, rows, cols);

  printf("\n结论：朴素 softmax 在大输入上 exp 溢出 → sum=inf、输出 NaN；\n");
  printf("      稳定版减最大值后 exp<=1 永不溢出，每行和正确归一到 1。\n");
  printf("      正确的 softmax 每行输出之和应 = 1。\n");
  return 0;
}
