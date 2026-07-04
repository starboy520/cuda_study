// Day 4 教学版 Tiled Attention（FlashAttention 数据流）—— 你来填 7 个空位
//
// 目标：把 Day1 的融合 attention + Day2 的 online softmax 拼起来，
//       但 K/V 也分块（Bc=16 一批），全程只维护 m/l/acc 三个 running 状态，
//       永不存完整 [N,N] scores → 这就是 FlashAttention 的数据流。
//
// 实现边界（教学版，先求正确不求快）：
//   FP32 输入/累加、单 batch、单 head、一个 block 处理一条 query、
//   K/V 按 Bc=16 分块、D≤128、支持任意 N、可选 causal。
//
// 三个 running 状态（每条 query 一份）：
//   m   = 目前见过的最大 score（防溢出基准）
//   l   = 相对 m 的指数和（softmax 分母）
//   acc = 未归一化输出 O_acc（相对同一 m 的分子，[D] 向量）
//   新 tile 出现更大 max 时：l 和 acc 都要乘 alpha=exp(m_old-m_new)（证明定理4）
//
// 编译：
//   nvcc -O3 -std=c++17 -arch=sm_80 tiled_attention.cu -o tiled_attention

#include <cuda_runtime.h>
#include <math_constants.h>   // CUDART_INF_F

constexpr int BC    = 16;     // 每个 tile 处理 16 个 K/V
constexpr int MAX_D = 128;    // head_dim 上限

// ===========================================================================
// 【必须手写】tiled attention kernel
//   一个 block 负责 query i = blockIdx.x
//   Q/K/V/out: [N, D]，行主序
// ===========================================================================
__global__ void tiled_attention(const float* q, const float* k, const float* v,
                                float* out, int n, int d, bool causal) {
    __shared__ float q_s[MAX_D];      // 当前 query 的 D 维
    __shared__ float k_s[BC][MAX_D];  // 当前 K tile
    __shared__ float v_s[BC][MAX_D];  // 当前 V tile
    __shared__ float scores[BC];      // 当前 tile 的局部 score（随后原地改成权重）
    __shared__ float acc[MAX_D];      // 未归一化 O_acc
    __shared__ float m;               // running max
    __shared__ float l;               // running 分母
    __shared__ float alpha;           // 本 tile 的重缩放因子

    const int tid   = threadIdx.x;
    const int query = blockIdx.x;
    if (query >= n) return;

    const float scale = rsqrtf((float)d);

    // TODO 0: 初始化 —— 加载 Q 到 q_s，acc/m/l 清零
    //   for (x = tid; x < d; x += blockDim.x): q_s[x]=q[query*d+x]; acc[x]=0;
    //   if (tid==0): m=-INF; l=0;
    //   记得 __syncthreads()
    //  一个block 负责  一个query， 所以一个query的 D维 需要协同加载；
    const float* query_row = q + blockIdx.x * d;
    float* out_row = out + blockIdx.x *d;
    for (int i = threadIdx.x; i< d; i += blockDim.x) {
        q_s[i] = query_row[i];
        acc[i] = 0.0f;
    }
    if (tid == 0) { m = -CUDART_INF_F; l = 0.0f; }
    __syncthreads();

    // ---- 遍历 K/V tile ----
    for (int k0 = 0; k0 < n; k0 += BC) {
        const int valid = min(BC, n - k0);   // 最后一块可能不足 BC

        // TODO 1: 协作加载当前 K/V tile 到 k_s/v_s（越界不加载）
        //   线程用 linear = tid; linear < valid*d; linear += blockDim.x 展平搬
        //   k_s[j][x] = k[(k0+j)*d + x];  v_s 同理
        //   搬运的是d* valid;
        int total = d * valid;
        const float* key_begin = k + k0 * d;
        const float* v_begin = v + k0 *d;
        for (int liner = threadIdx.x; liner < total; liner += blockDim.x) {
            int j = liner/d;
            int x = liner%d;
            k_s[j][x] = key_begin[liner];
            v_s[j][x] = v_begin[liner];
        }
        __syncthreads();

        //   记得 __syncthreads()

        // TODO 2: 前 valid 个线程各算一个 key 的局部 score
        //   if (tid < valid):
        //     key = k0 + tid;
        //     若 causal && key > query → scores[tid] = -INF
        //     否则 dot = Σ_x q_s[x]*k_s[tid][x]; scores[tid] = dot * scale;

        if (threadIdx.x < valid) {
            int key_id = k0 + tid;
            // query 要>= key （仅在 causal 时才 mask）
            if (causal && key_id > query) {
                scores[threadIdx.x] = -INFINITY;
            } else {
                float dot = 0.0;
                for (int i = 0; i < d; i++) {
                    dot += q_s[i] * k_s[threadIdx.x][i];  
                }
                scores[threadIdx.x] = dot * scale;
            }
        }
        __syncthreads();
        //   记得 __syncthreads()

        // TODO 3~5: 线程0 求本 tile 的 max，更新 m/l，把 scores 原地改成 exp(score-m_new)
        //   if (tid == 0):
        //     m_block = max over valid 个 scores
        //     （若整块被 mask：m_block=-INF → 本 tile 无贡献，alpha=1，scores 全置0，跳过）
        //     m_new = max(m, m_block)
        //     alpha = isinf(m)? 0 : exp(m - m_new)   // 旧状态重缩放因子
        //     tile_l = 0
        //     for j<valid: w = exp(scores[j]-m_new); scores[j]=w; tile_l += w
        //     l = alpha*l + tile_l;  m = m_new;

        //   // 4. m_new=max(m_old,m_block)，计算 alpha，更新 l
        // 5. acc[d] *= alpha
        // 6. acc[d] += Σ_j exp(score_j-m_new)*V_j[d]
        if (threadIdx.x == 0) {
            float cur_max = -INFINITY;
            for (int i = 0; i < valid; i++) {
                cur_max = fmax(cur_max, scores[i]);
            }
            if (isinf(cur_max) && cur_max < 0.0f) {
                alpha = 1.0f;
                for (int i = 0; i < valid; i++) {
                    scores[i] = 0.0f;
                }
            } else {
                float m_new = fmax(m, cur_max);
                alpha = isinf(m) ? 0 : expf(m - m_new);
                float tile_l = 0;
                for (int i = 0; i < valid; i++) {
                    float w = expf(scores[i] - m_new); // 
                    scores[i] = w;
                    tile_l += w;
                }
                l = alpha*l + tile_l;
                m = m_new;
            }
        }
        __syncthreads();
        //   记得 __syncthreads()

        // TODO 6: 每个线程负责若干输出 feature d，更新 acc
        //   for (int x = tid; x < d; x += blockDim.x):
        //     add = Σ_{j<valid} scores[j] * v_s[j][x]
        //     acc[x] = alpha*acc[x] + add;    // 旧 acc 也要乘 alpha！
        //   记得 __syncthreads()（acc 用完 v_s 后才能覆盖下一 tile）

        // 这里核心记住， 就是每个scorej 都要去乘以v_s[j][x]
        for (int x = threadIdx.x; x < d; x += blockDim.x) {
            float add = 0.0f;
            for (int j = 0; j < valid; j++) {
                add += scores[j] * v_s[j][x];
            }
            acc[x] = acc[x] * alpha + add;
        }
        __syncthreads();   // v_s 用完，下一 tile 才能覆盖
    }

    // TODO 7: 最终归一化写回 out[query*d + x] = acc[x] / l
    //   for (int x = tid; x < d; x += blockDim.x) ...
    for (int i = threadIdx.x; i < d; i+=blockDim.x) {
        out_row[i] = acc[i] / l;
    }
}

// ===========================================================================
// 测试框架（CPU 参考 + main）
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
        float mm = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < n; ++j) {
            if (causal && j > i) { s[j] = -std::numeric_limits<float>::infinity(); continue; }
            float dot = 0.0f;
            for (int x = 0; x < d; ++x) dot += q[i*d+x] * k[j*d+x];
            s[j] = dot * scale;
            mm = std::max(mm, s[j]);
        }
        float denom = 0.0f;
        for (int j = 0; j < n; ++j) {
            if (causal && j > i) { s[j] = 0.0f; }
            else { s[j] = std::exp(s[j] - mm); denom += s[j]; }
        }
        for (int x = 0; x < d; ++x) {
            float a = 0.0f;
            for (int j = 0; j < n; ++j) a += s[j] * v[j*d+x];
            out[i*d+x] = a / denom;
        }
    }
}

static bool run_case(int n, int d, bool causal) {
    if (d > MAX_D) { std::printf("D=%d > MAX_D\n", d); return false; }
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

    tiled_attention<<<n, 128>>>(dq, dk, dv, dout, n, d, causal);
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
    ok = run_case(37,  24, false) && ok;   // 非整除 N（Bc 不整除）
    ok = run_case(37,  24, true ) && ok;
    ok = run_case(128, 64, false) && ok;
    ok = run_case(128, 64, true ) && ok;
    std::printf("%s\n", ok ? "ALL PASS" : "SOME FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
