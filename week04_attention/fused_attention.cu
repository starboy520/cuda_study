// Day 4 融合版 Attention —— 把三个 kernel 合成一个（你来写 kernel）
//
// 核心思想（对比 naive_attention.cu 的三 kernel）：
//   三 kernel：scores[N,N] 两次落 HBM、两次读回
//   融合版  ：一个 block 负责一条 query i，scores 只放 shared[N]，
//            QK → softmax → PV 一条龙走完，中间量不落 HBM
//   → 省显存（不用 N×N 全局 buffer）+ 省带宽（不来回搬）
//
// 布局：
//   grid.x = N        每个 block 负责 query i = blockIdx.x
//   block  = 256       1 维线程，stride 扫
//   dynamic shared: s_scores[n]   （launch 时给 n*sizeof(float)）
//
// 编译：
//   nvcc -O3 -std=c++17 -arch=sm_80 fused_attention.cu -o fused_attention

#include <cuda_runtime.h>
#include <math_constants.h>
#include "../common/common.cuh"   // blockReduceMaxF / blockReduceSumF

// ---------------------------------------------------------------------------
// 融合 kernel：一个 block 一条 query
//   阶段1：s_scores[j] = (Q[i]·K[j]) * rsqrtf(d)，causal 时 j>i 写 -INFINITY
//          for (int j = tid; j < n; j += blockDim.x) { ... }
//          __syncthreads();
//   阶段2：对 s_scores[0..n-1] 做 stable softmax（原地改成权重）
//          - 局部 max → blockReduceMaxF → shared 广播
//          - 局部 Σexp(s-max) → blockReduceSumF → shared 广播
//          - s_scores[j] = exp(s_scores[j] - row_max)   （先只算分子，最后再除）
//          __syncthreads();
//   阶段3：out[i*d + x] = (Σ_j s_scores[j] * v[j*d + x]) / denom
//          for (int x = tid; x < d; x += blockDim.x) { ... }
//
// 提示：
//   - causal 每行至少 j=i 可见 → denom 一定 > 0，不用担心整行被 mask
//   - 广播 row_max / denom 要用 __shared__ + __syncthreads()（和 row_softmax 一样）
//   - scale 用 rsqrtf((float)d)
// ---------------------------------------------------------------------------
__global__ void fused_attention(const float* q, const float* k, const float* v,
                                float* out, int n, int d, bool causal) {
    extern __shared__ float s_scores[];   // 大小 = n（launch 时指定）
    __shared__ float b_max;
    __shared__ float b_denom;

    int i   = blockIdx.x;     // 这个 block 负责 query i
    int tid = threadIdx.x;

    // TODO: 阶段1 —— 算 s_scores[j]（含 causal），记得 __syncthreads()

    // TODO: 阶段2 —— 对 s_scores 做 softmax（原地变权重分子），记得广播 + __syncthreads()

    // TODO: 阶段3 —— out[i,x] = Σ_j s_scores[j]*V[j,x] / denom
}

// ===========================================================================
// 测试框架（CPU 参考 + main）—— 直接复用，验证融合 kernel 正确性
// ===========================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define CK(call) do {                                                          \
    cudaError_t e = (call);                                                     \
    if (e != cudaSuccess) {                                                      \
        std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,           \
                     cudaGetErrorString(e));                                    \
        std::exit(EXIT_FAILURE);                                                \
    }                                                                           \
} while (0)

static void attention_cpu(const float* q, const float* k, const float* v,
                          float* out, int n, int d, bool causal) {
    const float scale = 1.0f / std::sqrt((float)d);
    std::vector<float> s(n);
    for (int i = 0; i < n; ++i) {
        float m = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < n; ++j) {
            if (causal && j > i) { s[j] = -std::numeric_limits<float>::infinity(); continue; }
            float dot = 0.0f;
            for (int x = 0; x < d; ++x) dot += q[i*d+x] * k[j*d+x];
            s[j] = dot * scale;
            m = std::max(m, s[j]);
        }
        float denom = 0.0f;
        for (int j = 0; j < n; ++j) {
            if (causal && j > i) { s[j] = 0.0f; }
            else { s[j] = std::exp(s[j] - m); denom += s[j]; }
        }
        for (int x = 0; x < d; ++x) {
            float acc = 0.0f;
            for (int j = 0; j < n; ++j) acc += (s[j] / denom) * v[j*d+x];
            out[i*d+x] = acc;
        }
    }
}

static bool run_case(int n, int d, bool causal) {
    std::vector<float> hq(n*d), hk(n*d), hv(n*d), href(n*d), hgot(n*d);
    for (int idx = 0; idx < n*d; ++idx) {
        hq[idx] = (float)((idx*17) % 23 - 11) / 11.0f;
        hk[idx] = (float)((idx*13) % 19 - 9) / 9.0f;
        hv[idx] = (float)((idx*7)  % 29 - 14) / 14.0f;
    }
    attention_cpu(hq.data(), hk.data(), hv.data(), href.data(), n, d, causal);

    float *dq, *dk, *dv, *dout;
    CK(cudaMalloc(&dq, n*d*sizeof(float)));
    CK(cudaMalloc(&dk, n*d*sizeof(float)));
    CK(cudaMalloc(&dv, n*d*sizeof(float)));
    CK(cudaMalloc(&dout, n*d*sizeof(float)));
    CK(cudaMemcpy(dq, hq.data(), n*d*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk, hk.data(), n*d*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, hv.data(), n*d*sizeof(float), cudaMemcpyHostToDevice));

    // 一个 block 一条 query；dynamic shared = n 个 float 放 scores
    fused_attention<<<n, 256, n*sizeof(float)>>>(dq, dk, dv, dout, n, d, causal);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hgot.data(), dout, n*d*sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f; bool finite = true;
    for (int idx = 0; idx < n*d; ++idx) {
        finite = finite && std::isfinite(hgot[idx]);
        max_abs = std::max(max_abs, std::fabs(hgot[idx] - href[idx]));
    }
    bool ok = finite && max_abs < 2e-4f;
    std::printf("N=%-4d D=%-3d causal=%d  max_abs=%.3e  %s\n",
                n, d, causal, max_abs, ok ? "PASS" : "FAIL");

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
    return ok;
}

int main() {
    bool ok = true;
    ok = run_case(3,   2,  false) && ok;
    ok = run_case(8,   8,  false) && ok;
    ok = run_case(8,   8,  true ) && ok;
    ok = run_case(37,  24, false) && ok;
    ok = run_case(37,  24, true ) && ok;
    ok = run_case(128, 64, false) && ok;
    ok = run_case(128, 64, true ) && ok;
    std::printf("%s\n", ok ? "ALL PASS" : "SOME FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
