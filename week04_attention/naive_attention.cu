// Day 1 朴素 CUDA Attention（三阶段）——你来手写三个 kernel
//
// 数据流（单 batch、单 head，FP32）：
//   Q,K,V: [N, D]
//   1) qk_scores : scores[N,N] = Q K^T / sqrt(D)   （可选 causal mask）
//   2) row_softmax: 对 scores 每一行做 stable softmax（原地）
//   3) pv_output : out[N,D] = probs[N,N] @ V[N,D]
//
// 注意：这一版故意把完整的 [N,N] scores/probs materialize 出来，
// 方便你看清标准实现的问题（长序列时 N^2 中间量落 HBM）。
//
// 编译（等你写完 kernel、再自己加 main 之后）：
//   nvcc -O3 -std=c++17 -arch=sm_80 naive_attention.cu -o naive_attention

#include <cuda_runtime.h>
#include <math_constants.h>  // CUDART_INF_F

// ---------------------------------------------------------------------------
// Kernel 1: QK^T / sqrt(D)
//   推荐映射：二维 grid，每个线程算一个 scores[i, j]
//     int j = blockIdx.x * blockDim.x + threadIdx.x;  // key   (列)
//     int i = blockIdx.y * blockDim.y + threadIdx.y;  // query (行)
//   核心式：scores[i*n + j] = (Σ_x q[i*d + x] * k[j*d + x]) / sqrtf(d)
//   causal：若 causal 且 j > i，则写 -INFINITY
//   K 数学上是 K^T，但内存里不用真转置：读 k[j*d + x] 就是取 K 的第 j 行。
// ---------------------------------------------------------------------------
__global__ void qk_scores(const float* q, const float* k, float* scores,
                          int n, int d, bool causal) {
    // TODO: 你来写
    //score[i][j] = Σ query[i, d] * [k, j]
    float c = rsqrtf(d);
    int j = blockDim.x * blockIdx.x + threadIdx.x; // key
    int i = blockDim.y * blockIdx.y + threadIdx.y; // query
    if (i < n && j < n) {
        if (causal && i  < j) {
            scores[i *n + j] = -INFINITY;
        } else {
            float score = 0.0;
            for (int f = 0; f < d; f++) {
                score += q[i * d + f] * k[j * d + f];
            }
            scores[i * n + j] = score * c;
        }

    }

}

#include "../common/common.cuh"

// ---------------------------------------------------------------------------
// Kernel 2: row softmax（原地改写 scores）
//   一个 block 处理一行：blockIdx.x = 行号 i
//   三遍：
//     row_max = max_j scores[i, j]
//     denom   = Σ_j exp(scores[i, j] - row_max)
//     scores[i, j] = exp(scores[i, j] - row_max) / denom
//   可复用你 operator_practice/softmax 里的 block reduction 思路。
//   注意 -INFINITY 的项（causal mask）：exp(-inf)=0，天然不贡献。
// ---------------------------------------------------------------------------
__global__ void row_softmax(float* scores, int n) {
    __shared__ float b_max;
    __shared__ float b_rd;
    // TODO: 你来写
    int i = blockIdx.x;
    int tid = threadIdx.x;
    float* row = scores + i * n;


    float local_max = -INFINITY;
    for (int j = tid; j < n; j += blockDim.x) {
        local_max = fmax(local_max, row[j]);
    }
    float row_max = blockReduceMaxF(local_max);
    if (threadIdx.x == 0) {
        b_max = row_max;
    }

    __syncthreads();

    float local_sum = 0.0;
    for (int j = tid; j < n; j+=blockDim.x) {
        local_sum += expf(row[j] - b_max);
    }
    float denom = blockReduceSumF(local_sum);
    if (threadIdx.x == 0) {
        b_rd = denom;
    }
    __syncthreads();
    for (int j = tid; j < n; j+=blockDim.x) {
        row[j] = expf(row[j] - b_max) / b_rd;
    }

}

// ---------------------------------------------------------------------------
// Kernel 3: PV
//   推荐映射：二维 grid，每个线程算一个 out[i, x]
//     int x = blockIdx.x * blockDim.x + threadIdx.x;  // feature 维
//     int i = blockIdx.y * blockDim.y + threadIdx.y;  // query 行
//   核心式：out[i*d + x] = Σ_j probs[i*n + j] * v[j*d + x]
//   这就是普通的 [N,N] × [N,D] GEMM。
// ---------------------------------------------------------------------------
__global__ void pv_output(const float* probs, const float* v, float* out,
                          int n, int d) {
    // TODO: 你来写
    int j = blockDim.x * blockIdx.x + threadIdx.x; // key
    int i = blockDim.y * blockIdx.y + threadIdx.y; // query
    float sum = 0.0f;
    if (i < n && j < d) {
        for (int idx_col = 0; idx_col < n; idx_col++) {
            sum += probs[i * n + idx_col] * v[idx_col * d + j];
        }
        out[i *d + j] = sum;
    }

}

// ===========================================================================
// 测试框架（CPU 参考 + main）—— 验证上面三个 kernel 的正确性
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define CK(call) do {                                                          \
    cudaError_t e = (call);                                                     \
    if (e != cudaSuccess) {                                                      \
        std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,            \
                     cudaGetErrorString(e));                                     \
        std::exit(EXIT_FAILURE);                                                 \
    }                                                                            \
} while (0)

// 权威参考：单 head、FP32、可选 causal，直接融合算完（不模拟三 kernel 数据流）
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

    float *dq, *dk, *dv, *dscores, *dout;
    CK(cudaMalloc(&dq, n*d*sizeof(float)));
    CK(cudaMalloc(&dk, n*d*sizeof(float)));
    CK(cudaMalloc(&dv, n*d*sizeof(float)));
    CK(cudaMalloc(&dscores, (size_t)n*n*sizeof(float)));
    CK(cudaMalloc(&dout, n*d*sizeof(float)));
    CK(cudaMemcpy(dq, hq.data(), n*d*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk, hk.data(), n*d*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, hv.data(), n*d*sizeof(float), cudaMemcpyHostToDevice));

    // 1) QK^T
    dim3 b1(16, 16), g1((n+15)/16, (n+15)/16);
    qk_scores<<<g1, b1>>>(dq, dk, dscores, n, d, causal);
    // 2) row softmax（一 block 一行）
    row_softmax<<<n, 256>>>(dscores, n);
    // 3) PV
    dim3 b3(16, 16), g3((d+15)/16, (n+15)/16);
    pv_output<<<g3, b3>>>(dscores, dv, dout, n, d);
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

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dscores); cudaFree(dout);
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
