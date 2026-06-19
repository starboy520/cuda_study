// unified_memory_demo.cu
//
// 统一内存(Unified Memory)实验：同一个任务用三种写法，实测性能差距。
//
// ============================================================================
// 要演示的事：
//   统一内存(cudaMallocManaged)很方便，但默认性能可能很差——尤其当 CPU/GPU
//   交替访问同一批页、触发反复的"页迁移(page migration)"时(ping-pong)。
//   用 cudaMemPrefetchAsync 提前把页迁到位，能把性能救回来，接近手动 cudaMemcpy。
//
// 任务：对一个大数组做 ITERS 轮处理，每轮 = GPU kernel 加工一遍。
//       为了制造"CPU 也要碰数据"的真实场景，每轮之间让 CPU 读一下校验值，
//       这正是统一内存最容易 ping-pong 的模式。
//
// 三个版本：
//   runManagedNaive   : cudaMallocManaged，每轮 CPU/GPU 交替访问 → 反复页迁移(慢)
//   runManagedPrefetch: 同上，但用 cudaMemPrefetchAsync 主动把页迁到位(快)
//   runManualCopy      : 传统 cudaMalloc + cudaMemcpy，一次拷入、循环全在 GPU(基准)
//
// 编译：
//   nvcc -O3 -arch=sm_75 unified_memory_demo.cu -o unified_memory_demo
// 运行：
//   ./unified_memory_demo            # 默认 16M 元素, 30 轮
//   ./unified_memory_demo 16777216 30
//
// 用 Nsight Systems 可直接看到 naive 版满屏的页迁移：
//   nsys profile --trace=cuda,um -o um_report ./unified_memory_demo
// ============================================================================

#include <cstdio>
#include <cstdlib>

#include "../../common/cuda_check.cuh"

__global__ void processKernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = data[idx] * 1.0001f + 0.5f;
}

static int divUp(int a, int b) { return (a + b - 1) / b; }

// CUDA 13 的 cudaMemPrefetchAsync 需要 cudaMemLocation。封装两个辅助构造。
static cudaMemLocation deviceLoc(int device) {
  cudaMemLocation loc{};
  loc.type = cudaMemLocationTypeDevice;
  loc.id = device;
  return loc;
}
static cudaMemLocation hostLoc() {
  cudaMemLocation loc{};
  loc.type = cudaMemLocationTypeHost;
  loc.id = 0;
  return loc;
}

// ---------------------------------------------------------------------------
// 版本 1：朴素 managed —— 每轮 CPU 都碰一下数据，触发 ping-pong 页迁移。
// 模式：GPU 加工(页迁到GPU) → CPU 读校验(页迁回CPU) → 下一轮再迁回GPU ...
// 同一批页每轮跨 PCIe 来回搬两次，迁移开销主导耗时。
// ---------------------------------------------------------------------------
float runManagedNaive(int n, int iters, int device) {
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  const int threads = 256;
  float* data;
  CUDA_CHECK(cudaMallocManaged(&data, bytes));
  for (int i = 0; i < n; ++i) data[i] = 1.0f;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  volatile float sink = 0.0f;
  for (int it = 0; it < iters; ++it) {
    processKernel<<<divUp(n, threads), threads>>>(data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    // CPU 读一个值 → 触发整页(及周边)迁回 CPU，制造 ping-pong
    sink = data[it % n];
  }
  (void)sink;

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaFree(data));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

// ---------------------------------------------------------------------------
// 版本 2：managed + prefetch —— 主动把页迁到位，消除运行时零散缺页。
// kernel 前把整块预取到 GPU；CPU 要读的那一刻只预取需要的一小段回 CPU。
// 把"大量零散 page fault"换成"少量批量、可重叠的迁移"。
// ---------------------------------------------------------------------------
float runManagedPrefetch(int n, int iters, int device) {
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  const int threads = 256;
  float* data;
  CUDA_CHECK(cudaMallocManaged(&data, bytes));
  for (int i = 0; i < n; ++i) data[i] = 1.0f;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // 一次性把整块预取到 GPU，之后循环里数据就驻留在 GPU，不再每轮整批迁移
  CUDA_CHECK(cudaMemPrefetchAsync(data, bytes, deviceLoc(device), 0));

  volatile float sink = 0.0f;
  for (int it = 0; it < iters; ++it) {
    processKernel<<<divUp(n, threads), threads>>>(data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 只把要读的那一个元素所在的小范围预取回 CPU，而不是让整批页 ping-pong
    int i = it % n;
    CUDA_CHECK(cudaMemPrefetchAsync(&data[i], sizeof(float), hostLoc(), 0));
    sink = data[i];
    // 读完再把这一小段送回 GPU，保持主体驻留在 GPU
    CUDA_CHECK(cudaMemPrefetchAsync(&data[i], sizeof(float), deviceLoc(device), 0));
  }
  (void)sink;

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaFree(data));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

// ---------------------------------------------------------------------------
// 版本 3：传统手动 —— cudaMalloc + 一次 cudaMemcpy 拷入，循环全程在 GPU。
// 这是"数据一次搬到 GPU、之后都在 GPU 算"的最优基准。
// CPU 校验只在最后做一次（模拟真实里少量回读）。
// ---------------------------------------------------------------------------
float runManualCopy(int n, int iters, int device) {
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  const int threads = 256;
  float* h = new float[n];
  for (int i = 0; i < n; ++i) h[i] = 1.0f;
  float* d;
  CUDA_CHECK(cudaMalloc(&d, bytes));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
  volatile float sink = 0.0f;
  for (int it = 0; it < iters; ++it) {
    processKernel<<<divUp(n, threads), threads>>>(d, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 模拟每轮少量回读：只拷回一个元素
    float one = 0.0f;
    CUDA_CHECK(cudaMemcpy(&one, &d[it % n], sizeof(float),
                          cudaMemcpyDeviceToHost));
    sink = one;
  }
  (void)sink;

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaFree(d));
  delete[] h;
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

int main(int argc, char** argv) {
  int n = 1 << 24;   // 16M 元素 ≈ 64 MB
  int iters = 30;
  if (argc >= 2) n = std::atoi(argv[1]);
  if (argc >= 3) iters = std::atoi(argv[2]);

  int device = 0;
  CUDA_CHECK(cudaSetDevice(device));

  printf("n=%d (%.1f MB), iters=%d\n", n,
         n * sizeof(float) / 1e6, iters);

  // warmup
  runManualCopy(n, 3, device);

  float msNaive = runManagedNaive(n, iters, device);
  float msPrefetch = runManagedPrefetch(n, iters, device);
  float msManual = runManualCopy(n, iters, device);

  printf("\n");
  printf("managed 朴素 (ping-pong 页迁移) : %8.2f ms   (基准)\n", msNaive);
  printf("managed + prefetch (主动迁页)   : %8.2f ms   比朴素快 %.2fx\n",
         msPrefetch, msNaive / msPrefetch);
  printf("手动 cudaMemcpy (一次拷入)       : %8.2f ms   比朴素快 %.2fx\n",
         msManual, msNaive / msManual);
  printf("\n结论：同样用统一内存，加不加 prefetch 性能可能差很多。\n");
  printf("      朴素 managed 因 CPU/GPU 交替访问反复迁页而变慢；\n");
  printf("      prefetch 主动把页迁到位，能逼近手动 cudaMemcpy。\n");

  return 0;
}
