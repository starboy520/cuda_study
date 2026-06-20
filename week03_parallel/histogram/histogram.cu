#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../../common/cuda_check.cuh"

constexpr int BINS = 256;        // bin 数量（值域 0~255，shared 放得下）

// ============================================================================
// 版本1：histGlobal —— 每个线程直接 atomicAdd 到 global 的 bin（全员怼一个数组）
// TODO: 你来写
//   提示：int i = 全局索引；if(i<n) atomicAdd(&hist[in[i]], 1);
// ============================================================================
__global__ void histGlobal(const int* in, int* hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int value = in[idx];
        if (value >= 0 && value < BINS) {
            atomicAdd(&hist[value], 1);
        }
    }
}

// ============================================================================
// 版本2：histShared —— privatization：每 block 先在 shared 建私有直方图，最后合并
// 三步：① 清零私有直方图 ② 在 shared 上 atomic ③ 合并回 global
// TODO: 你来写
//   提示：extern __shared__ int local[];  启动时第三参数给 BINS*sizeof(int)
//   清零/合并都要用 grid-stride 风格遍历 bins（线程数可能 < bins）：
//     for (int b = threadIdx.x; b < BINS; b += blockDim.x) ...
// ============================================================================
__global__ void histShared(const int* in, int* hist, int n) {
    // TODO
    __shared__ int local[BINS];
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) local[b] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int value = in[idx];
        if (value >= 0 && value < BINS) {
            atomicAdd(&local[value], 1);
        }
    }
    __syncthreads();

    
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) {
        atomicAdd(&hist[b], local[b]);
    }
}

// ---- CPU 参考：串行统计直方图 ----
static void cpu_hist(const int* in, int* hist, int n) {
    for (int b = 0; b < BINS; b++) hist[b] = 0;
    for (int i = 0; i < n; i++) hist[in[i]]++;
}

// ---- 校验两个直方图是否一致 ----
static bool check_hist(const int* ref, const int* got) {
    for (int b = 0; b < BINS; b++) {
        if (ref[b] != got[b]) {
            printf("  mismatch at bin %d: ref=%d got=%d\n", b, ref[b], got[b]);
            return false;
        }
    }
    return true;
}

// ---- 跑一个 kernel：计时 + 校验 ----
// useShared=true 时用 histShared（需要动态 shared），否则用 histGlobal
static void run_and_check(const char* name, bool useShared,
                          const int* d_in, int* d_hist, int n,
                          const int* h_ref, int* h_got) {
    const int block = 256;
    const int grid  = (n + block - 1) / block;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup
    CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(int)));
    if (useShared) histShared<<<grid, block, BINS * sizeof(int)>>>(d_in, d_hist, n);
    else           histGlobal<<<grid, block>>>(d_in, d_hist, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(int)));
    CUDA_CHECK(cudaEventRecord(start));
    if (useShared) histShared<<<grid, block, BINS * sizeof(int)>>>(d_in, d_hist, n);
    else           histGlobal<<<grid, block>>>(d_in, d_hist, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_got, d_hist, BINS * sizeof(int), cudaMemcpyDeviceToHost));

    printf("  %-14s %7.3f ms  %s\n", name, ms,
           check_hist(h_ref, h_got) ? "PASS" : "FAIL");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    const int N = 1 << 24;   // 16M 个数据

    int* h_in  = (int*)malloc(N * sizeof(int));
    int  h_ref[BINS];
    int  h_got[BINS];

    int *d_in, *d_hist;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist, BINS * sizeof(int)));

    // ===== 场景A：均匀分布（值随机散在 0~255）=====
    for (int i = 0; i < N; i++) h_in[i] = rand() % BINS;
    cpu_hist(h_in, h_ref, N);
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    printf("=== 场景A：均匀分布（值散在 0~255）===\n");
    run_and_check("global atomic", false, d_in, d_hist, N, h_ref, h_got);
    run_and_check("shared privat", true,  d_in, d_hist, N, h_ref, h_got);

    // ===== 场景B：90% 集中在 bin 0（暴露竞争）=====
    for (int i = 0; i < N; i++) h_in[i] = (rand() % 10 == 0) ? (rand() % BINS) : 0;
    cpu_hist(h_in, h_ref, N);
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    printf("=== 场景B：90%% 集中在 bin 0（竞争激烈）===\n");
    run_and_check("global atomic", false, d_in, d_hist, N, h_ref, h_got);
    run_and_check("shared privat", true,  d_in, d_hist, N, h_ref, h_got);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_hist));
    free(h_in);
    return 0;
}
