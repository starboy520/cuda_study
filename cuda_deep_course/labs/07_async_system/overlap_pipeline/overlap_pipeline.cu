// overlap_pipeline.cu
//
// 卷七核心实验：传输与计算重叠（multi-stream pipeline）。
//
// ============================================================================
// 要演示的事：
//   一个端到端任务 = 把 N 个元素从 Host 拷到 Device(H2D) → kernel 计算 → 拷回 Host(D2H)。
//   朴素做法用单个 stream，三步严格串行：传的时候 SM 空着，算的时候 PCIe 空着。
//   把数据切成多块、用多个 stream，让"第 i+1 块在传"与"第 i 块在算"同时发生，
//   就能把 PCIe 和 SM 的空闲互相填上 → 端到端更快。
//
//   关键前提：必须用 pinned(页锁定) host 内存(cudaMallocHost)，
//             否则 cudaMemcpyAsync 会退化成同步，重叠不会发生。
//
// 三个版本对比：
//   runSerialPinned     : 单 stream，H2D→K→D2H 整体串行（pinned 内存）
//   runOverlapped       : 多 stream 分块流水线，重叠传输与计算（pinned 内存）
//   runSerialPageable   : 单 stream + pageable 内存，作为"为什么要 pinned"的反例
//
// 编译：
//   nvcc -O3 -arch=sm_75 overlap_pipeline.cu -o overlap_pipeline
// 运行：
//   ./overlap_pipeline                # 默认 64M 元素、16 块、4 流
//   ./overlap_pipeline 67108864 32 4  # 自定义 元素数 块数 流数
//
// 用 Nsight Systems 看时间线，直接肉眼确认重叠：
//   nsys profile --trace=cuda -o overlap_report ./overlap_pipeline
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "../../common/cuda_check.cuh"

// 让 kernel 有"可观但不夸张"的计算量，使 kernel 时间和拷贝时间量级接近，
// 这样重叠收益才明显（若 kernel 极快，瓶颈全在传输，重叠收益有限）。
__global__ void computeKernel(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float x = in[idx];
  // 一段纯算术，制造 compute 负载
  for (int i = 0; i < 200; ++i) {
    x = x * 0.9999f + 0.0001f;
    x = sqrtf(x * x + 1.0f) - 1.0f;
  }
  out[idx] = x;
}

static int divUp(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// 版本 1：单 stream，pinned 内存，整体串行（基准）。
// 一次性 H2D 全部 → 一次 kernel → 一次 D2H 全部。
// 时间线：[====H2D====][====K====][====D2H====]  三段首尾相接，互不重叠。
// ---------------------------------------------------------------------------
float runSerialPinned(const float* h_in, float* h_out, float* d_in,
                      float* d_out, int n) {
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  const int threads = 256;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice));
  computeKernel<<<divUp(n, threads), threads>>>(d_in, d_out, n);
  CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

// ---------------------------------------------------------------------------
// 版本 2：多 stream 分块流水线，重叠传输与计算（核心）。
// 把 N 切成 chunks 块，轮流丢进 nStreams 个 stream。每块在自己的 stream 里
// 按 H2D→K→D2H 顺序执行；不同 stream 之间可以并发，于是：
//   stream0: H2D0 ─ K0 ─ D2H0
//   stream1:      H2D1 ─ K1 ─ D2H1
//   stream2:           H2D2 ─ K2 ─ D2H2
// 一个 stream 在算时，另一个 stream 正好在传，PCIe 和 SM 同时忙。
// ---------------------------------------------------------------------------
float runOverlapped(const float* h_in, float* h_out, float* d_in, float* d_out,
                    int n, int chunks, int nStreams) {
  const int threads = 256;
  const int chunkSize = divUp(n, chunks);

  std::vector<cudaStream_t> streams(nStreams);
  for (auto& s : streams) CUDA_CHECK(cudaStreamCreate(&s));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  for (int c = 0; c < chunks; ++c) {
    const int offset = c * chunkSize;
    const int cur = (offset + chunkSize <= n) ? chunkSize : (n - offset);
    if (cur <= 0) break;
    const size_t curBytes = static_cast<size_t>(cur) * sizeof(float);
    cudaStream_t s = streams[c % nStreams];

    // 同一 stream 内 H2D→K→D2H 保持顺序；不同 stream 间并发重叠。
    CUDA_CHECK(cudaMemcpyAsync(d_in + offset, h_in + offset, curBytes,
                               cudaMemcpyHostToDevice, s));
    computeKernel<<<divUp(cur, threads), threads, 0, s>>>(d_in + offset,
                                                          d_out + offset, cur);
    CUDA_CHECK(cudaMemcpyAsync(h_out + offset, d_out + offset, curBytes,
                               cudaMemcpyDeviceToHost, s));
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  for (auto& s : streams) CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

// ---------------------------------------------------------------------------
// 版本 3：单 stream + pageable 内存（反例）。
// 用普通 new/malloc 的 host 内存做"分块多 stream"，看似和版本2一样，
// 但 pageable 内存上的 cudaMemcpyAsync 无法真正异步 → 退化成串行，
// 几乎拿不到重叠收益。用来证明 "pinned 是异步重叠的前提"。
// ---------------------------------------------------------------------------
float runOverlapPageable(const float* h_in, float* h_out, float* d_in,
                         float* d_out, int n, int chunks, int nStreams) {
  const int threads = 256;
  const int chunkSize = divUp(n, chunks);

  std::vector<cudaStream_t> streams(nStreams);
  for (auto& s : streams) CUDA_CHECK(cudaStreamCreate(&s));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  for (int c = 0; c < chunks; ++c) {
    const int offset = c * chunkSize;
    const int cur = (offset + chunkSize <= n) ? chunkSize : (n - offset);
    if (cur <= 0) break;
    const size_t curBytes = static_cast<size_t>(cur) * sizeof(float);
    cudaStream_t s = streams[c % nStreams];
    CUDA_CHECK(cudaMemcpyAsync(d_in + offset, h_in + offset, curBytes,
                               cudaMemcpyHostToDevice, s));
    computeKernel<<<divUp(cur, threads), threads, 0, s>>>(d_in + offset,
                                                          d_out + offset, cur);
    CUDA_CHECK(cudaMemcpyAsync(h_out + offset, d_out + offset, curBytes,
                               cudaMemcpyDeviceToHost, s));
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  for (auto& s : streams) CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

int main(int argc, char** argv) {
  int n = 1 << 26;     // 64M 元素 ≈ 256 MB，传输时间够大，便于观察
  int chunks = 16;     // 切成几块
  int nStreams = 4;    // 几个并发 stream
  if (argc >= 2) n = std::atoi(argv[1]);
  if (argc >= 3) chunks = std::atoi(argv[2]);
  if (argc >= 4) nStreams = std::atoi(argv[3]);

  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  printf("n=%d (%.1f MB), chunks=%d, streams=%d\n", n, bytes / 1e6, chunks,
         nStreams);

  // device buffers
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  // pinned host buffers
  float *h_in, *h_out;
  CUDA_CHECK(cudaMallocHost(&h_in, bytes));
  CUDA_CHECK(cudaMallocHost(&h_out, bytes));
  for (int i = 0; i < n; ++i) h_in[i] = static_cast<float>(i % 1000) * 0.001f;

  // pageable host buffers
  float* p_in = new float[n];
  float* p_out = new float[n];
  for (int i = 0; i < n; ++i) p_in[i] = h_in[i];

  // warmup（首次有 context/JIT 开销，先跑一次丢弃）
  runSerialPinned(h_in, h_out, d_in, d_out, n);

  float msSerial = runSerialPinned(h_in, h_out, d_in, d_out, n);
  float msOverlap = runOverlapped(h_in, h_out, d_in, d_out, n, chunks, nStreams);
  float msPageable =
      runOverlapPageable(p_in, p_out, d_in, d_out, n, chunks, nStreams);

  printf("\n");
  printf("串行 (pinned, 单 stream)         : %7.2f ms\n", msSerial);
  printf("重叠 (pinned, %d streams 分块)    : %7.2f ms   加速 %.2fx\n", nStreams,
         msOverlap, msSerial / msOverlap);
  printf("分块 (pageable, %d streams)       : %7.2f ms   加速 %.2fx (应≈1, 退化)\n",
         nStreams, msPageable, msSerial / msPageable);
  printf("\n结论：pinned + 多 stream 分块能重叠传输与计算；pageable 无法真异步，\n");
  printf("      即使写成多 stream 也几乎拿不到重叠收益。用 nsys 看时间线可证实。\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  delete[] p_in;
  delete[] p_out;
  return 0;
}
