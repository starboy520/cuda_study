#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// ---- 常量定义（避免 magic number）----
constexpr int      WARP_SIZE   = 32;            // 一个 warp 的 lane 数（硬件定死）
constexpr int      MAX_THREADS = 1024;          // 一个 block 的线程数上限
constexpr int      MAX_WARPS   = MAX_THREADS / WARP_SIZE;  // 一个 block 最多几个 warp = 32
constexpr unsigned FULL_MASK   = 0xffffffffU;   // __shfl_*_sync 的全 warp 掩码


/**
Global atomic baseline（v1）——亲手感受"正确但慢"。
Shared tree（v2）——把竞争关进 block。
每 thread 两元素（v3）——砍掉最浪费的第一轮。
Warp shuffle 收尾（v4）——后 32 个值免 barrier。
Grid-stride 多元素累加——让一个 block 吃远多于 512 个元素，进一步减少 block 数与阶段数。
与 CUB DeviceReduce 对比——看工业库还领先在哪（向量化加载、动态调参等）。

*/
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)


__global__ void reduction_sum_race_global(
        unsigned long long * input, int size,
        unsigned long long* total) {
    int idx = threadIdx.x + blockDim.x  * blockIdx.x;
    if (idx < size) {
        atomicAdd(total, input[idx]);
    }
}

__global__ void shared_atomic_add(unsigned long long * in, int n, unsigned long long* sum) {
    __shared__ unsigned long long value;
    if (threadIdx.x == 0) {
        value = 0;
    }

    __syncthreads();

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&value, static_cast<unsigned long long>(in[idx]));
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(sum, value);
    }
}


__global__ void reduction_v3(unsigned long long* input,
    int size, unsigned long long* sum) {

    extern __shared__ unsigned long long s[];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // 每线程把自己负责的 global 元素搬进 shared（越界填 0）
    s[tid] = (idx < size) ? input[idx] : 0ULL;
    __syncthreads();

    // 在 shared 上做树形归约，不破坏 input
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            s[tid] += s[tid + offset];
        }
        __syncthreads();
    }

    // 每个 block 的部分和写回
    if (tid == 0) {
        sum[blockIdx.x] = s[0];
    }
}


__global__ void reduction_stride(unsigned long long* input, int size, unsigned long long* sum) {

    __shared__ unsigned long long value[MAX_WARPS];   // 每 warp 一个部分和

    unsigned long long local = 0;
    int stride = blockDim.x * gridDim.x;
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < size; idx += stride) {
        local += input[idx];
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local += __shfl_down_sync(FULL_MASK, local, offset);
    }

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        value[wid] = local;
    }
    __syncthreads();
    if (wid == 0) {
        local = (lane < blockDim.x / WARP_SIZE) ? value[lane] : 0;

        for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
            local += __shfl_down_sync(FULL_MASK, local, o);
        if (lane == 0) sum[blockIdx.x] = local;
    }
}


__global__ void reduction_my(unsigned long long* input, int size, unsigned long long *sum) {
    // value[] 存"每个 warp 一个部分和"，大小 = block 最多几个 warp = MAX_WARPS(=1024/32=32)
    // 注意：WARP_SIZE(=32) 是"warp 有 32 lane"，MAX_WARPS(=32) 是"block 最多 32 warp"——含义不同
    __shared__ unsigned long long  value[MAX_WARPS];

    // ① grid-stride：每线程在寄存器里攒任意多个元素
    //    步长=总线程数，每圈 warp 内地址仍连续 → 保持合并访问
    unsigned long long local = 0;
    int stride = gridDim.x * blockDim.x;
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += stride) {
        local += input[i];
    }

    // ② 第一级归约：warp 内 shuffle，log2(WARP_SIZE) 步把每个 warp 收成 1 个值(落在该 warp 的 lane0)
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local += __shfl_down_sync(FULL_MASK, local, offset);
    }

    int lane = threadIdx.x % WARP_SIZE;   // 在 warp 内的编号 0~31
    int warp = threadIdx.x / WARP_SIZE;   // 第几个 warp
    if (lane == 0) {
        value[warp] = local;       // 每个 warp 的部分和写进 shared
    }

    __syncthreads();               // 等所有 warp 都写完，第一个 warp 才能读

    // ③ 第二级归约：让第一个 warp 把"各 warp 的部分和"再 shuffle 归约成 1 个
    if (warp == 0) {
        // lane < 实际 warp 数才有有效值，越界填 0
        local = (lane < blockDim.x / WARP_SIZE) ? value[lane] : 0;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            local += __shfl_down_sync(FULL_MASK, local, offset);
        }

        if (lane == 0) {
            sum[blockIdx.x] = local;   // 每个 block 输出 1 个部分和（跨 block 需第二趟归约）
        }
    }

}

__global__ void reduction_shuffle_only(unsigned long long* input, int size, 
    unsigned long long* sum) {
    unsigned long long local = 0;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    local = (idx < size) ? input[idx] : 0;

    for (int offset = WARP_SIZE / 2; offset > 0; offset /=2) {
        local += __shfl_down_sync(FULL_MASK, local, offset);
    }

    int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) {
        atomicAdd(sum, local);
    }

}
// ---- 把"每 block 一个部分和"的数组拷回 host 再加起来（用于验证 v3 / stride）----
static unsigned long long cpuSumPartials(unsigned long long* d_partial, int n) {
    unsigned long long* h = (unsigned long long*)malloc(n * sizeof(unsigned long long));
    CUDA_CHECK(cudaMemcpy(h, d_partial, n * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    unsigned long long s = 0;
    for (int i = 0; i < n; ++i) s += h[i];
    free(h);
    return s;
}

int main() {
    const int N     = 1 << 24;                 // 16M 个元素
    const int block = 256;                      // 每 block 线程数
    const int gridStride = 1024;                // stride 版的 grid：写小，靠循环覆盖全部
    const size_t bytes = (size_t)N * sizeof(unsigned long long);

    // 全填 1，期望和 = N（避免 idx 求和溢出，结果一目了然）
    unsigned long long* h_in = (unsigned long long*)malloc(bytes);
    for (int i = 0; i < N; ++i) h_in[i] = 1ULL;
    const unsigned long long expected = (unsigned long long)N;

    unsigned long long *d_in, *d_total, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(unsigned long long)));

    const int gridV3 = (N + block - 1) / block;          // v3：一 block 管 block 个元素
    CUDA_CHECK(cudaMalloc(&d_partial, (size_t)gridV3 * sizeof(unsigned long long)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    printf("N = %d, expected sum = %llu\n\n", N, expected);

    // ---- v1: 全局 atomic 基线 ----
    {
        int grid = (N + block - 1) / block;
        CUDA_CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));
        reduction_sum_race_global<<<grid, block>>>(d_in, N, d_total);  // warmup
        CUDA_CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaEventRecord(start));
        reduction_sum_race_global<<<grid, block>>>(d_in, N, d_total);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        unsigned long long got;
        CUDA_CHECK(cudaMemcpy(&got, d_total, sizeof(got), cudaMemcpyDeviceToHost));
        printf("v1 global atomic      : %7.3f ms  sum=%llu  %s\n",
               ms, got, got == expected ? "PASS" : "FAIL");
    }

    // ---- v2: shared atomic 分层聚合 ----
    {
        int grid = (N + block - 1) / block;
        CUDA_CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));
        shared_atomic_add<<<grid, block>>>(d_in, N, d_total);          // warmup
        CUDA_CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaEventRecord(start));
        shared_atomic_add<<<grid, block>>>(d_in, N, d_total);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        unsigned long long got;
        CUDA_CHECK(cudaMemcpy(&got, d_total, sizeof(got), cudaMemcpyDeviceToHost));
        printf("v2 shared atomic      : %7.3f ms  sum=%llu  %s\n",
               ms, got, got == expected ? "PASS" : "FAIL");
    }

    // ---- v3: shared 树形归约（每 block 一个部分和，需二次归约）----
    {
        size_t smem = block * sizeof(unsigned long long);            // 动态 shared
        reduction_v3<<<gridV3, block, smem>>>(d_in, N, d_partial);   // warmup
        CUDA_CHECK(cudaEventRecord(start));
        reduction_v3<<<gridV3, block, smem>>>(d_in, N, d_partial);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        unsigned long long got = cpuSumPartials(d_partial, gridV3);  // 二次归约
        printf("v3 shared tree        : %7.3f ms  sum=%llu  %s  (kernel 时间，不含二次归约)\n",
               ms, got, got == expected ? "PASS" : "FAIL");
    }

    // ---- stride: grid-stride + warp shuffle 满血版 ----
    {
        reduction_stride<<<gridStride, block>>>(d_in, N, d_partial);  // warmup
        CUDA_CHECK(cudaEventRecord(start));
        reduction_stride<<<gridStride, block>>>(d_in, N, d_partial);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        unsigned long long got = cpuSumPartials(d_partial, gridStride);
        printf("stride grid+shuffle   : %7.3f ms  sum=%llu  %s  (kernel 时间，不含二次归约)\n",
               ms, got, got == expected ? "PASS" : "FAIL");
    }

    // ---- stride 两趟全 GPU 归约（第二趟也在 GPU 上，得到最终单值）----
    // 同一个 kernel 复用两次：
    //   趟1: N 个 → gridStride 个部分和(d_partial)
    //   趟2: gridStride 个部分和 → 1 个最终值(d_total)，用 1 个 block 收尾
    {
        // warmup（两趟都跑一遍）
        reduction_stride<<<gridStride, block>>>(d_in, N, d_partial);
        reduction_stride<<<1, block>>>(d_partial, gridStride, d_total);

        CUDA_CHECK(cudaEventRecord(start));
        reduction_stride<<<gridStride, block>>>(d_in, N, d_partial);          // 趟1
        reduction_stride<<<1, block>>>(d_partial, gridStride, d_total);        // 趟2
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        unsigned long long got;
        CUDA_CHECK(cudaMemcpy(&got, d_total, sizeof(got), cudaMemcpyDeviceToHost));
        printf("stride two-pass (GPU) : %7.3f ms  sum=%llu  %s  (含两趟，全程不回 CPU)\n",
               ms, got, got == expected ? "PASS" : "FAIL");
    }

    // ---- reduction_my：自己手写的两阶段归约，两趟全 GPU ----
    {
        reduction_my<<<gridStride, block>>>(d_in, N, d_partial);          // warmup
        reduction_my<<<1, block>>>(d_partial, gridStride, d_total);

        CUDA_CHECK(cudaEventRecord(start));
        reduction_my<<<gridStride, block>>>(d_in, N, d_partial);          // 趟1
        reduction_my<<<1, block>>>(d_partial, gridStride, d_total);        // 趟2
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        unsigned long long got;
        CUDA_CHECK(cudaMemcpy(&got, d_total, sizeof(got), cudaMemcpyDeviceToHost));
        printf("my two-pass (GPU)     : %7.3f ms  sum=%llu  %s  (自己手写，应与 stride 一致)\n",
               ms, got, got == expected ? "PASS" : "FAIL");
    }

    // ---- CUB DeviceReduce::Sum——工业库标杆 ----
    // 两段式 API：第一次传 d_temp=nullptr 探测临时显存大小，分配后第二次真算。
    {
        void*  d_temp = nullptr;
        size_t temp_bytes = 0;
        // 探测所需临时空间（不计算，只填 temp_bytes）
        cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_total, N);
        CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

        cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_total, N);  // warmup
        CUDA_CHECK(cudaEventRecord(start));
        cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_total, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        unsigned long long got;
        CUDA_CHECK(cudaMemcpy(&got, d_total, sizeof(got), cudaMemcpyDeviceToHost));
        printf("cub DeviceReduce::Sum : %7.3f ms  sum=%llu  %s  (工业库标杆)\n",
               ms, got, got == expected ? "PASS" : "FAIL");
        CUDA_CHECK(cudaFree(d_temp));
    }

    // ---- reduction_shuffle_only：纯 shuffle 版（一线程一元素 + warp shuffle + lane0 atomicAdd）----
    // 没有 grid-stride 循环、没有 shared：grid 必须开满(一线程一元素)，atomic 一趟出最终值。
    {
        int grid = (N + block - 1) / block;                          // 65536，覆盖全部元素
        CUDA_CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));
        reduction_shuffle_only<<<grid, block>>>(d_in, N, d_total);   // warmup
        CUDA_CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaEventRecord(start));
        reduction_shuffle_only<<<grid, block>>>(d_in, N, d_total);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        unsigned long long got;
        CUDA_CHECK(cudaMemcpy(&got, d_total, sizeof(got), cudaMemcpyDeviceToHost));
        printf("shuffle_only (1线程1元素): %7.3f ms  sum=%llu  %s  (纯shuffle+atomic，无stride无shared)\n",
               ms, got, got == expected ? "PASS" : "FAIL");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_total));
    CUDA_CHECK(cudaFree(d_partial));
    free(h_in);
    return 0;
}

