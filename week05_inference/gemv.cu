// Week5 Day2：GEMV（矩阵×向量）—— y = W·x + b
//
// 背景：decode 单请求时，线性层 X 只有一行 token → GEMM 塌成 GEMV（M=1）。
//       GEMV 是 memory-bound：权重 W 每个元素只用一次，瓶颈在读 W 的带宽。
//
// 形状：W[N,K]（行主序）, x[K], b[N], y[N]
//   每个输出 y[row] = b[row] + Σ_k W[row,k]·x[k]   （行之间独立）
//
// 分工：
//   - CPU reference + 测试框架（对拍/计时/GB-s/GFLOPS）已写好，无需改动
//   - 两个 kernel（gemv_thread / gemv_warp）由你实现（TODO）
//
// 编译：
//   nvcc -O3 -lineinfo -std=c++17 -arch=sm_80 gemv.cu -o gemv
//   nvcc -O3 -lineinfo -arch=sm_80 -Xptxas=-v gemv.cu -o gemv   # 看寄存器/spill

#include <cuda_runtime.h>

// ===========================================================================
// 【你来写】v0：一个线程负责一行
//   row = blockIdx.x*blockDim.x + threadIdx.x
//   缺点：同一时刻相邻线程在不同 row、同一个 k，地址 stride = K → 大 K 时
//        warp 访问不连续（未合并）。先写对，Day3 再优化。
// ===========================================================================
__global__ void gemv_thread(const float* W, const float* x,
                            const float* b, float* y, int N, int K) {
    // TODO 1：算出本线程负责的 row，越界 return。
    // TODO 2：sum = b ? b[row] : 0；for k in [0,K): sum = fmaf(W[row*K+k], x[k], sum)。
    // TODO 3：y[row] = sum。
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < N) {
        
        const float* row_W = W + K * row;
        float sum = b ? b[row] : 0.0f;
        for (int i = 0; i < K; i++) {
            sum = fmaf(x[i], row_W[i], sum);
        }
        y[row] = sum;
    }
}

// ===========================================================================
// 【你来写】v1：一个 warp 负责一行（lane-stride + warp reduction）
//   一个 block 有 blockDim.x/32 个 warp，各处理一行。
//   同一 warp 的 32 个 lane 在同一轮访问连续的 k → W 合并访问（比 v0 好）。
// ===========================================================================
__global__ void gemv_warp(const float* W, const float* x,
                          const float* b, float* y, int N, int K) {

    // TODO 1：lane-stride 遍历 for (k = lane; k < K; k += 32) 累加 W[row*K+k]*x[k]。
    // TODO 2：warp reduction（偏移 16,8,4,2,1，用 __shfl_down_sync(0xffffffff, sum, off)）。
    // TODO 3：lane 0 加 bias 并写 y[row] = sum + (b ? b[row] : 0)。

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warp_num_per_block = blockDim.x / 32;
    int row = warp_num_per_block * blockIdx.x + warp_id;
    // 每个warp 负责一行， 
    if (row >= N) return;
    const float* w_row = W + row * K;
    float sum = 0.0f;
    for (int i = lane_id; i < K; i += 32) {
        sum += w_row[i] * x[i];
    }
    //
    for (int offset = 16; offset; offset /=2) {
        sum = sum + __shfl_down_sync(0xffffffffu, sum, offset); 
    }
    if (lane_id == 0) {
        y[row] = sum + (b ? b[row] : 0.0);
    }
}

// ===========================================================================
// CPU 参考 + 测试框架（已写好，无需改动）
// ===========================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(call) do {                                                          \
    cudaError_t e = (call);                                                     \
    if (e != cudaSuccess) {                                                     \
        std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,           \
                     cudaGetErrorString(e));                                    \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
} while (0)

// ---- CPU 参考：double 累加，作为正确性基准 ----
static void gemv_cpu(const float* W, const float* x, const float* b,
                     float* y, int N, int K) {
    for (int row = 0; row < N; ++row) {
        double sum = b ? (double)b[row] : 0.0;   // bias 可选（这里恒有）
        for (int k = 0; k < K; ++k)
            sum += (double)W[row * K + k] * (double)x[k];
        y[row] = (float)sum;
    }
}

// ---- 一个 kernel 变体的“对拍 + 计时”封装 ----
//   launch：一个 lambda，接收 (dW,dx,db,dy,N,K) 并配好 grid/block 后启动 kernel。
//   把重复的 warmup / CUDA Event 计时 / 误差统计 / 带宽吞吐都收在这里。
template <typename LaunchFn>
static void bench(const char* name, LaunchFn launch,
                  const float* dW, const float* dx, const float* db, float* dy,
                  int N, int K, const std::vector<float>& ref) {
    // 一次功能运行 + 拷回结果做正确性检查
    launch(dW, dx, db, dy, N, K);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<float> got(N);
    CK(cudaMemcpy(got.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 误差：绝对 + 相对（相对分母加 1e-6 防止除 0）
    double max_abs = 0.0, max_rel = 0.0;
    bool finite = true;
    for (int i = 0; i < N; ++i) {
        finite = finite && std::isfinite(got[i]);
        double diff = std::fabs((double)got[i] - (double)ref[i]);
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, diff / (std::fabs((double)ref[i]) + 1e-6));
    }
    bool ok = finite && max_rel < 1e-3;   // fp32 累加 over 大 K 会有 ~K*eps 误差，容差放宽到 1e-3

    // 计时：warmup 预热（触发 JIT/缓存），再多次取平均，避免单次噪声
    const int warmup = 5, iters = 50;
    for (int i = 0; i < warmup; ++i) launch(dW, dx, db, dy, N, K);
    CK(cudaDeviceSynchronize());

    cudaEvent_t beg, end;
    CK(cudaEventCreate(&beg)); CK(cudaEventCreate(&end));
    CK(cudaEventRecord(beg));
    for (int i = 0; i < iters; ++i) launch(dW, dx, db, dy, N, K);
    CK(cudaEventRecord(end));
    CK(cudaEventSynchronize(end));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, beg, end));
    ms /= iters;
    CK(cudaEventDestroy(beg)); CK(cudaEventDestroy(end));

    // logical bytes：算法层面的访存账本（W 主导 + x/b/y）。
    //   注意这只是“理论应读写字节”，不等于 DRAM 实际 transaction（那要 ncu 看）。
    double logical_bytes = ((double)N * K + K + N + N) * sizeof(float);
    double sec  = ms / 1e3;
    double gbps = logical_bytes / sec / 1e9;      // 有效带宽（GB/s）
    double gflops = 2.0 * N * K / sec / 1e9;       // GEMV 计算量 ≈ 2NK

    std::printf("  %-12s  %-4s  max_abs=%.3e max_rel=%.3e  %.4f ms  "
                "%.1f GB/s  %.1f GFLOPS\n",
                name, ok ? "PASS" : "FAIL", max_abs, max_rel, ms, gbps, gflops);
}

static void run_case(int N, int K) {
    std::printf("N=%-5d K=%-5d\n", N, K);

    // 固定可复现的输入（避免每次随机导致对拍抖动）
    std::vector<float> hW((size_t)N * K), hx(K), hb(N), ref(N);
    for (size_t i = 0; i < hW.size(); ++i)
        hW[i] = (float)((int)((i * 17) % 23) - 11) / 11.0f;   // 先转 int 再减，避免无符号下溢
    for (int i = 0; i < K; ++i) hx[i] = (float)((i * 13) % 19 - 9) / 9.0f;
    for (int i = 0; i < N; ++i) hb[i] = (float)((i * 7)  % 29 - 14) / 14.0f;

    gemv_cpu(hW.data(), hx.data(), hb.data(), ref.data(), N, K);

    float *dW, *dx, *db, *dy;
    CK(cudaMalloc(&dW, (size_t)N * K * sizeof(float)));
    CK(cudaMalloc(&dx, K * sizeof(float)));
    CK(cudaMalloc(&db, N * sizeof(float)));
    CK(cudaMalloc(&dy, N * sizeof(float)));
    CK(cudaMemcpy(dW, hW.data(), (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, hx.data(), K * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // v0：一线程一行。grid = ceil(N/256)，block = 256
    bench("thread/row",
          [](const float* W, const float* x, const float* b, float* y, int N, int K) {
              int threads = 256;
              int blocks  = (N + threads - 1) / threads;
              gemv_thread<<<blocks, threads>>>(W, x, b, y, N, K);
          },
          dW, dx, db, dy, N, K, ref);

    // v1：一 warp 一行。block = 256（=8 warp），grid = ceil(N/8)
    bench("warp/row",
          [](const float* W, const float* x, const float* b, float* y, int N, int K) {
              int threads = 256;
              int warps_per_block = threads / 32;
              int blocks  = (N + warps_per_block - 1) / warps_per_block;
              gemv_warp<<<blocks, threads>>>(W, x, b, y, N, K);
          },
          dW, dx, db, dy, N, K, ref);

    cudaFree(dW); cudaFree(dx); cudaFree(db); cudaFree(dy);
}

int main() {
    run_case(3,    4);      // 对齐手算（§9）
    run_case(37,   24);     // 非 warp/向量整除，测边界
    run_case(4096, 4096);   // 典型大权重，看带宽
    run_case(1,    4097);   // 极少输出行 + 非整除 K
    return 0;
}
