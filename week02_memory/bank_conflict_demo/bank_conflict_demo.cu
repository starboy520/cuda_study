// bank_conflict_demo.cu
//
// 演示 shared memory 的 "bank conflict"（存储体冲突）以及 padding 如何消除它。
//
// ============================================================================
// 背景速记：
//   shared memory 有 32 个并行存取通道，叫 bank。把 shared 想成一排 4 字节格子，
//   格子按编号轮流分给 32 个 bank：bank = (字地址) % 32。
//
//   硬件规则：一个 bank 一次只能服务一个地址。
//     - 一个 warp 的 32 个线程落在 32 个不同 bank → 一拍并行完成   ✅
//     - 多个线程挤进"同一 bank 的不同地址"        → 串行，N 路冲突 ❌
//     - 多个线程读"同一 bank 的同一地址"          → 广播，不算冲突 ✅
//
//   最坏案例：32x32 的 shared 数组按"列"访问。
//     tile[row][col] 的字地址 = row*32 + col，bank = (row*32+col)%32 = col。
//     按列读时 32 个线程固定同一 col、变化 row → 全部落在同一个 bank(=col)
//     → 32 路冲突 → 慢约 32 倍。
//
//   修复：把数组开成 [32][33]，多一列 padding(从不存数据，只用来错位)。
//     字地址 = row*33 + col，bank = (row*33+col)%32 = (row+col)%32。
//     按列读时 bank=(row+col)%32 随 row 变化 → 32 个线程落在 32 个不同 bank
//     → 冲突消失。
// ============================================================================
//
// 本 demo 两个 kernel 做同一件事：把一个 tile 按列求和写回。
//   - colSumConflict : __shared__ float tile[32][32]  → 按列读触发 32 路冲突
//   - colSumPadded   : __shared__ float tile[32][33]  → padding 消除冲突
// 为了放大差距，每个线程把这件事重复 REPEAT 次，让 shared 访问成为主要开销。
//
// 编译：
//   nvcc -O3 -arch=sm_75 bank_conflict_demo.cu -o bank_conflict_demo
//
// 运行（打印两个 kernel 的耗时与加速比）：
//   ./bank_conflict_demo
//
// 用 Nsight Compute 亲眼看 bank conflict 计数（需要 GPU 计数器权限）：
//   ncu --metrics \
//     l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
//     --kernel-name regex:colSum ./bank_conflict_demo
//   冲突版该指标会很大，padding 版应接近 0。

#include <cstdio>

#include "../../common/cuda_check.cuh"

constexpr int TILE = 32;        // 32x32 tile，正好一个 warp 一列
constexpr int REPEAT = 2000;    // 重复多次放大 shared 访问开销
constexpr int BLOCKS = 4096;    // 多个 block 让 GPU 跑满

// ----------------------------------------------------------------------------
// 冲突版：tile[32][32]。
// block = (32,32) = 1024 线程。线性 id = threadIdx.y*32 + threadIdx.x，
// 所以一个 warp 是连续 32 个线性 id，即 "ty 固定、tx = 0..31"。
//
// 关键访问是 tile[tx][ty]：
//   字地址 = tx*32 + ty，bank = (tx*32 + ty) % 32 = ty。
//   一个 warp 内 ty 固定、tx 变化 → 32 个线程的 bank 都 = ty（同一个 bank），
//   但 tx 不同 → 访问的是该 bank 里 32 个不同地址 → 教科书式的 32 路冲突。
//   硬件被迫把这 32 个请求串行成 32 拍，慢约 32 倍。
// ----------------------------------------------------------------------------
__global__ void colSumConflict(const float* in, float* out) {
  __shared__ float tile[TILE][TILE];

  int tx = threadIdx.x;  // 0..31
  int ty = threadIdx.y;  // 0..31

  // 载入 shared（这一步是合并访问，不是观察重点）
  tile[ty][tx] = in[ty * TILE + tx];
  __syncthreads();

  // tile[row][ty]：warp 内 ty 固定、不同线程 tx 不同。
  // 每次迭代换一行 row=(tx+r)%TILE，既防止编译器把读取提到循环外，
  // 又保持 bank=(row*32+ty)%32=ty 不变 → warp 内 32 线程恒压同一 bank=ty
  // → 32 路冲突。重复 REPEAT 次放大耗时。
  float sum = 0.0f;
  for (int r = 0; r < REPEAT; ++r) {
    int row = (tx + r) % TILE;
    sum += tile[row][ty] * 1.0001f;
  }

  out[blockIdx.x * TILE * TILE + ty * TILE + tx] = sum;
}

// ----------------------------------------------------------------------------
// padding 版：tile[32][33]，其余完全相同。
// 访问 tile[row][ty]：字地址 = row*33 + ty，bank = (row*33+ty)%32 = (row+ty)%32。
// 每次迭代 row=(tx+r)%TILE，warp 内 tx=0..31 → row 取 32 个不同值
// → bank=(row+ty)%32 取 32 个不同值 → 落 32 个不同 bank → 冲突消除。
// ----------------------------------------------------------------------------
__global__ void colSumPadded(const float* in, float* out) {
  __shared__ float tile[TILE][TILE + 1];  // +1 padding

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  tile[ty][tx] = in[ty * TILE + tx];
  __syncthreads();

  float sum = 0.0f;
  for (int r = 0; r < REPEAT; ++r) {
    int row = (tx + r) % TILE;
    sum += tile[row][ty] * 1.0001f;
  }

  out[blockIdx.x * TILE * TILE + ty * TILE + tx] = sum;
}

static float timeKernel(void (*kernel)(const float*, float*),
                        const float* d_in, float* d_out) {
  dim3 block(TILE, TILE);   // 1024 线程/block
  dim3 grid(BLOCKS);

  // warmup
  kernel<<<grid, block>>>(d_in, d_out);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  kernel<<<grid, block>>>(d_in, d_out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms;
}

int main() {
  const int inElems = TILE * TILE;
  const size_t inBytes = inElems * sizeof(float);
  const size_t outBytes = static_cast<size_t>(BLOCKS) * TILE * TILE * sizeof(float);

  float* h_in = new float[inElems];
  for (int i = 0; i < inElems; ++i) h_in[i] = static_cast<float>(i % 7) * 0.5f;

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, inBytes));
  CUDA_CHECK(cudaMalloc(&d_out, outBytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, inBytes, cudaMemcpyHostToDevice));

  float msConflict = timeKernel(colSumConflict, d_in, d_out);
  float msPadded = timeKernel(colSumPadded, d_in, d_out);

  printf("colSumConflict (tile[32][32], 32路冲突): %.3f ms\n", msConflict);
  printf("colSumPadded   (tile[32][33], padding) : %.3f ms\n", msPadded);
  printf("加速比 (conflict / padded)             : %.2fx\n", msConflict / msPadded);
  printf("\n说明：padding 版应明显更快。差距大小取决于 kernel 是否真的被\n");
  printf("shared 访问主导；用 Nsight Compute 的 bank_conflicts 指标可直接证实。\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  delete[] h_in;
  return 0;
}
