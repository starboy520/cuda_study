// cuda_graph.cu
//
// 卷七实验：CUDA Graph —— 用"录制一次、重放多次"消除反复 launch 的 CPU 开销。
//
// ============================================================================
// 要演示的事：
//   很多程序每次迭代都执行"同一串固定的 kernel/拷贝"（推理、迭代求解器等）。
//   传统方式每次迭代都重新 launch 每个 kernel，每次 launch 都有 CPU 端开销
//   （几微秒）。当 kernel 本身很短、且序列很长时，这些 launch 开销累加起来
//   会成为瓶颈——GPU 在等 CPU 一个个发指令。
//
//   CUDA Graph 把整串操作"录制(capture)"成一张图，"实例化(instantiate)"一次，
//   之后每次迭代只需一次 cudaGraphLaunch 重放整张图，CPU 几乎不再逐个 launch。
//
// 本 demo 让每次迭代连续启动 CHAIN 个很短的 kernel，对比：
//   runStreamLaunch : 每次迭代逐个 launch CHAIN 个 kernel（传统方式）
//   runGraphLaunch  : 录制一次，之后每次迭代只 cudaGraphLaunch 重放
//
// 编译：
//   nvcc -O3 -arch=sm_75 cuda_graph.cu -o cuda_graph
// 运行：
//   ./cuda_graph                 # 默认 iters=1000, chain=50, n=100000
//   ./cuda_graph 2000 100 50000  # 自定义 iters chain n
// ============================================================================

#include <cstdio>
#include <cstdlib>

#include "../../common/cuda_check.cuh"

// 故意做成"很短"的 kernel：计算量小，使 launch 开销相对显著。
__global__ void tinyKernel(float* data, int n, float k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = data[idx] * 1.0001f + k;
}

static int divUp(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// 传统方式：每次迭代都在 stream 上逐个 launch CHAIN 个 kernel。
// 每个 <<<>>> 都有一次 CPU launch 开销。
// ---------------------------------------------------------------------------
float runStreamLaunch(float* d_data, int n, int chain, int iters,
                      cudaStream_t s) {
  const int threads = 256;
  const int blocks = divUp(n, threads);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, s));

  for (int it = 0; it < iters; ++it) {
    for (int c = 0; c < chain; ++c) {
      tinyKernel<<<blocks, threads, 0, s>>>(d_data, n, 0.0001f);
    }
  }

  CUDA_CHECK(cudaEventRecord(stop, s));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

// ---------------------------------------------------------------------------
// CUDA Graph：先用 stream capture 录制"一串 CHAIN 个 kernel"成图，
// 实例化一次，之后每次迭代只 cudaGraphLaunch 重放。
// ---------------------------------------------------------------------------
float runGraphLaunch(float* d_data, int n, int chain, int iters,
                     cudaStream_t s) {
  const int threads = 256;
  const int blocks = divUp(n, threads);

  // 1) 录制：把 CHAIN 个 kernel 的提交过程捕获成一张 graph
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int c = 0; c < chain; ++c) {
    tinyKernel<<<blocks, threads, 0, s>>>(d_data, n, 0.0001f);
  }
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));

  // 2) 实例化（只做一次）
  CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, s));

  // 3) 重放：每次迭代只一次 launch，整张图的 CHAIN 个 kernel 一起跑
  for (int it = 0; it < iters; ++it) {
    CUDA_CHECK(cudaGraphLaunch(graphExec, s));
  }

  CUDA_CHECK(cudaEventRecord(stop, s));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaGraphExecDestroy(graphExec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

int main(int argc, char** argv) {
  int iters = 1000;  // 迭代次数
  int chain = 50;    // 每次迭代连续启动多少个短 kernel
  int n = 100000;    // 每个 kernel 处理的元素数（故意不大，让 launch 开销显著）
  if (argc >= 2) iters = std::atoi(argv[1]);
  if (argc >= 3) chain = std::atoi(argv[2]);
  if (argc >= 4) n = std::atoi(argv[3]);

  printf("iters=%d, chain=%d, n=%d  (每次迭代 launch %d 个短 kernel)\n", iters,
         chain, n, chain);

  float* d_data;
  CUDA_CHECK(cudaMalloc(&d_data, static_cast<size_t>(n) * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_data, 0, static_cast<size_t>(n) * sizeof(float)));

  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  // warmup
  runStreamLaunch(d_data, n, chain, 10, s);

  float msStream = runStreamLaunch(d_data, n, chain, iters, s);
  float msGraph = runGraphLaunch(d_data, n, chain, iters, s);

  printf("\n");
  printf("逐个 launch (stream)   : %7.2f ms\n", msStream);
  printf("CUDA Graph 重放        : %7.2f ms   加速 %.2fx\n", msGraph,
         msStream / msGraph);
  printf("\n结论：当 kernel 短、序列固定且重复多次时，Graph 把每次迭代的\n");
  printf("      N 次 launch 压成 1 次重放，省下大量 CPU launch 开销。\n");

  CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaFree(d_data));
  return 0;
}
