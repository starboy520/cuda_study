// Day 2 Online Softmax（CUDA）—— 你来写 kernel 主体
//
// 目标：对每一行做 softmax，但用 online 方式维护 (m, l) 一对状态，
//       而不是"先扫一遍求全局 max，再扫一遍求和"。
//       正确性证明见 docs/Online_Softmax正确性证明.md
//
// 与昨天 row_softmax 的区别：
//   昨天：blockReduceMaxF 求 max，再 blockReduceSumF 求和 —— 两次独立归约
//   今天：归约的是"一对状态 Pair{m,l}"，合并算子是 combine（不是单独 max/sum）
//
// 布局：一个 block 处理一行；grid.x = 行数；block 一维，线程 grid-stride 扫。
//
// 编译：
//   nvcc -O3 -std=c++17 -arch=sm_80 online_softmax.cu -o online_softmax

#include <cuda_runtime.h>
#include <math_constants.h>   // CUDART_INF_F

constexpr int WARP_SIZE = 32;

// ===========================================================================
// 归约"零件"（已提供，直接用；类比昨天 common.cuh 的 blockReduceMaxF/SumF）
// ===========================================================================

// 一对 online softmax 状态：m = running max，l = 相对 m 的指数和
struct Pair {
    float m;
    float l;
};

// 合并两个状态（定理 1 / combine）：换到共同基准 max，再相加
//   m = max(a.m, b.m); l = a.l*exp(a.m-m) + b.l*exp(b.m-m)
//   空状态约定 (-INF, 0)，注意别算出 -inf-(-inf)=NaN
__device__ __forceinline__ Pair combine(Pair a, Pair b) {
    // TODO: 你来写
    float new_m = fmax(a.m, b.m);
    if (isinf(new_m) && new_m < 0.0f) {
        return {-INFINITY, 0.0};
    }
    float new_l = a.l * expf(a.m - new_m) + b.l * expf(b.m - new_m);
    return {new_m, new_l};
}

// warp 内归约一对状态：每轮 __shfl_down_sync 同时交换 m 和 l，再 combine
//   结果落在 lane 0
__device__ __forceinline__ Pair warpReducePair(Pair p) {
    // TODO: 你来写
    for (int offset = 16; offset > 0; offset /= 2) {
        float m = __shfl_down_sync(0xffffffffu, p.m, offset);
        float l = __shfl_down_sync(0xffffffffu, p.l, offset);
        p = combine(p, Pair{m, l});
    }
    return p;
}

// block 内归约一对状态：各 warp 先 warpReduce → leader 写 shared → warp0 再合并
//   结果落在 thread 0
__device__ __forceinline__ Pair blockReducePair(Pair p) {
    // TODO: 你来写
    __shared__ Pair pair[1024];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    p = warpReducePair(p);
    if (lane_id == 0) {
        pair[warp_id] = p;
    }
    __syncthreads();

    int num_warp = blockDim.x /32;
        if (lane_id < num_warp) {
            p = pair[lane_id];
        } else {
            p = {-INFINITY, 0.0f};
        }
    if (warp_id == 0) {


        p = warpReducePair(p);
    }

    return p;

}

// ===========================================================================
// 【必须手写】online softmax kernel
//   in  : [rows, n]，每行做一次 softmax
//   out : [rows, n]，写归一化概率
//
// 步骤：
//   1. 定位本行：int row = blockIdx.x; const float* xin = in + row*n; ...
//   2. 【online 局部扫描】每个线程用 grid-stride 扫自己那几个元素，
//      维护线程局部 Pair state（初值 {-INF, 0}），每读一个元素 x：
//         state = combine(state, Pair{x, 1.0f});
//      （合并单元素集合 {x} 的状态就是 {x, 1}，见证明第 5 节）
//   3. Pair final = blockReducePair(state);
//   4. 用 __shared__ 把 final.m / final.l 广播给所有线程（+ __syncthreads()）
//   5. 【写回】每个线程 grid-stride 写 out[row*n+j] = expf(xin[j]-m) / l
//
// 提示：
//   - 第 2 步就是 online 的精髓：不预先求全局 max，边读边 combine
//   - blockReducePair 结果只有 thread 0 正确 → 必须 shared 广播（同昨天）
//   - 大值输入（如 1000+）也不会溢出，因为 combine 里始终减当前 max
// ===========================================================================
__global__ void online_softmax(const float* in, float* out, int n) {

    __shared__ Pair f_p;

    // TODO: 你来写（步骤 1~5）
    // m : curMax
    // l : sum(e(m-in[i]);
    const float* row = in + blockIdx.x * n;
    int stride = blockDim.x;
    Pair pair{-INFINITY, 0.0f};
    for (int i = threadIdx.x; i < n; i += stride) {
        pair = combine(pair, {row[i], 1}); // 知道这里为什么1
    }
    pair = blockReducePair(pair);
    if (threadIdx.x == 0) {
        f_p = pair;
    }

    __syncthreads();
    float* o_row = out + n * blockIdx.x;
    for (int i = threadIdx.x; i < n; i += stride) {
        o_row[i] = expf(row[i] - f_p.m) / f_p.l;
    }

}

// ===========================================================================
// 测试框架（CPU stable softmax 参考 + main）
// ===========================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(call) do {                                                          \
    cudaError_t e = (call);                                                     \
    if (e != cudaSuccess) {                                                      \
        std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,           \
                     cudaGetErrorString(e));                                    \
        std::exit(EXIT_FAILURE);                                                \
    }                                                                           \
} while (0)

static void softmax_cpu(const float* x, float* y, int n) {
    float m = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < n; ++j) m = std::max(m, x[j]);
    float l = 0.0f;
    for (int j = 0; j < n; ++j) l += std::exp(x[j] - m);
    for (int j = 0; j < n; ++j) y[j] = std::exp(x[j] - m) / l;
}

static bool run_case(int rows, int n, float lo, float hi, const char* name) {
    std::vector<float> hin(rows*n), hgot(rows*n), href(rows*n);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < n; ++j) {
            float t = (float)((r*131 + j*17) % 1000) / 999.0f;  // 0..1
            hin[r*n+j] = lo + t * (hi - lo);
        }
    for (int r = 0; r < rows; ++r) softmax_cpu(hin.data()+r*n, href.data()+r*n, n);

    float *din, *dout;
    CK(cudaMalloc(&din, rows*n*sizeof(float)));
    CK(cudaMalloc(&dout, rows*n*sizeof(float)));
    CK(cudaMemcpy(din, hin.data(), rows*n*sizeof(float), cudaMemcpyHostToDevice));

    online_softmax<<<rows, 256>>>(din, dout, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hgot.data(), dout, rows*n*sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f; bool finite = true; float worst_rowsum_err = 0.0f;
    for (int r = 0; r < rows; ++r) {
        float s = 0.0f;
        for (int j = 0; j < n; ++j) {
            finite = finite && std::isfinite(hgot[r*n+j]);
            max_abs = std::max(max_abs, std::fabs(hgot[r*n+j] - href[r*n+j]));
            s += hgot[r*n+j];
        }
        worst_rowsum_err = std::max(worst_rowsum_err, std::fabs(s - 1.0f));
    }
    bool ok = finite && max_abs < 1e-4f && worst_rowsum_err < 1e-4f;
    std::printf("%-22s rows=%d n=%-5d range=[%.0f,%.0f] max_abs=%.2e rowsum_err=%.2e %s\n",
                name, rows, n, lo, hi, max_abs, worst_rowsum_err, ok ? "PASS" : "FAIL");

    cudaFree(din); cudaFree(dout);
    return ok;
}

int main() {
    bool ok = true;
    ok = run_case(4,   1000, -5,   5,    "normal")        && ok;
    ok = run_case(4,   1000,  500, 1499, "large-values")  && ok;  // 测不溢出
    ok = run_case(3,   1031, -3,   3,    "non-divisible") && ok;  // 非整除
    ok = run_case(4,   1000, -9,  -3,    "all-negative")  && ok;  // 全负
    ok = run_case(8,   64,   -2,   2,    "small-n")       && ok;
    std::printf("%s\n", ok ? "ALL PASS" : "SOME FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
