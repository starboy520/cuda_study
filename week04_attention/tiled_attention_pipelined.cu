// Day 5 教学版 Tiled Attention + cp.async 双缓冲流水 —— 你来填 kernel
//
// 目标：在 Day4 的 tiled_attention 基础上，用 cp.async 把「加载下一个 K/V tile」
//       和「计算当前 tile」重叠起来（software pipeline / double buffering），
//       藏掉 HBM 访存延迟。数学（m/l/acc + online softmax）和 Day4 完全一样，
//       唯一新增的是「异步预取 + 双缓冲」这套数据流。
//
// 实现边界（和 Day4 保持一致，方便对拍）：
//   FP32 输入/累加、单 batch、单 head、一个 block 处理一条 query、
//   K/V 按 Bc=16 分块、D≤128、支持任意 N、可选 causal。
//
// 你要做的（kernel 里的 TODO）：
//   - 用 __pipeline_memcpy_async 异步搬 K/V tile 到 shared 的「某一个 buffer」
//   - 双缓冲：STAGES=2，buf = tile 序号 & 1，一边算 buf、一边预取 1-buf
//   - 用 __pipeline_commit() / __pipeline_wait_prior(0) 控制「等哪一批搬完」
//   - 其余 online softmax 逻辑照搬你 Day4 的 tiled_attention.cu
//
// cp.async 三件套（sm_80+，头文件 <cuda_pipeline.h>）：
//   __pipeline_memcpy_async(dst_shared_ptr, src_global_ptr, bytes); // 发起一次异步拷贝
//   __pipeline_commit();          // 把「刚才发起的这一批」打包成一个 group
//   __pipeline_wait_prior(N);     // 等到「只剩最近 N 个 group 未完成」为止
//   注意：拷贝字节数只能是 4 / 8 / 16（对应 float / float2 / float4）。
//
// main + CPU 参考 + 正确性/计时验证已由脚手架写好，你只改 kernel。
//
// 编译：
//   nvcc -O3 -std=c++17 -arch=sm_80 tiled_attention_pipelined.cu -o tiled_attention_pipelined

#include <cuda_runtime.h>
#include <cuda_pipeline.h>    // __pipeline_memcpy_async / commit / wait_prior
#include <math_constants.h>   // CUDART_INF_F
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

constexpr int BC     = 16;    // 每个 tile 处理 16 个 K/V
constexpr int MAX_D  = 128;   // head_dim 上限
constexpr int STAGES = 2;     // 双缓冲：同时最多 2 个 tile 在 shared 里

// ===========================================================================
// 【你来手写】cp.async 双缓冲版 tiled attention kernel
//   一个 block 负责 query i = blockIdx.x
//   Q/K/V/out: [N, D]，行主序
//
//   推荐骨架（伪代码）：
//     load q_s; acc=0; m=-INF; l=0
//     prefetch tile0 到 buf0（cp.async + commit）           // 序幕
//     for (t = 0; t < num_tiles; ++t):
//        buf = t & 1
//        if (t+1 < num_tiles): prefetch tile(t+1) 到 buf(1-buf)（cp.async + commit）
//        __pipeline_wait_prior(  (t+1<num_tiles) ? 1 : 0 );  // 等当前 tile 搬完
//        __syncthreads();
//        用 k_s[buf]/v_s[buf] 算 score → online softmax 更新 m/l/acc（照搬 Day4）
//        __syncthreads();
//     out = acc / l
//
//   关键点：预取「下一块」和计算「当前块」之间不能有依赖；wait_prior 的参数决定
//           你允许多少个 group 还在飞（in-flight）。双缓冲典型是保留 1 个在飞。
// ===========================================================================
namespace cg = cooperative_groups;

__device__ void load_title_async(float* s_mem, int smem_ld,
    const float* k, int tileH, int tileW, 
    int row_base, int column_base, 
    int ld, int bound_row, int bound_col, int tid, int n_thread,
    cuda::pipeline<cuda::thread_scope_block>& pipe) {
    for (int i = tid; i < tileH* tileW; i+= n_thread) {
        int r = i / tileW;
        int c = i % tileW;
        int global_row = row_base + r;
        int global_col = column_base + c;
        if (global_row < bound_row && global_col < bound_col) {
            cuda::memcpy_async(&s_mem[r*smem_ld + c], 
                &k[global_row * ld + global_col], 
                sizeof(float), pipe);
        }
    }
}

__global__ void tiled_attention_pipelined(const float* q, const float* k,
                                           const float* v, float* out,
                                           int n, int d, bool causal) {
    __shared__ float q_s[MAX_D];               // 当前 query 的 D 维
    __shared__ float k_s[STAGES][BC][MAX_D];   // 双缓冲 K tile
    __shared__ float v_s[STAGES][BC][MAX_D];   // 双缓冲 V tile
    __shared__ float scores[BC];               // 当前 tile 的局部 score（随后原地改成权重）
    __shared__ float acc[MAX_D];               // 未归一化 O_acc
    __shared__ float m;                        // running max
    __shared__ float l;                        // running 分母
    __shared__ float alpha;                    // 本 tile 的重缩放因子

    // TODO: 全部由你实现。
    //   提示：
    //   - 每个 tile 的有效行数 valid = min(BC, n - k0)，最后一块可能不足 BC。
    //   - cp.async 搬运时按 float 逐个搬也行（__pipeline_memcpy_async(&k_s[buf][j][x],
    //     &k[(k0+j)*d + x], sizeof(float))），进阶可改 float4 一次搬 16B。
    //   - 越界的行（j >= valid）不要发起 async 拷贝。
    //   - online softmax（求 max / alpha / 更新 l / 累加 acc）和 Day4 一模一样，
    //     直接把你 tiled_attention.cu 的 TODO2~7 逻辑搬过来，只是 K/V 改成读 k_s[buf]。

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss;
    auto block = cg::this_thread_block();
    auto pipe = cuda::make_pipeline(block, &pss);
    float scale = rsqrtf(float(d));

    int stage = 0;
    int tid = threadIdx.x;
    int query = blockIdx.x;

    const float* query_row = q + blockIdx.x * d;
    float* out_row = out + blockIdx.x * d;
    // cooperate load current query;
    for (int i = tid; i < d; i += blockDim.x) {
        q_s[i] = query_row[i];
        acc[i] = 0.0f;
    }
    if (tid == 0) {
        m = -INFINITY;
        l = 0.0f;
    }
    __syncthreads();

    int tile_step = 0;
    int valid = min(BC, n - tile_step);

    pipe.producer_acquire();
    load_title_async(&k_s[stage][0][0], MAX_D, 
        k, valid, d, 
        tile_step, 0, d, 
        n, d, 
        tid, blockDim.x, pipe);
    load_title_async(&v_s[stage][0][0], MAX_D, 
        v, valid, d, 
        tile_step, 0, d,
        n, d, 
        tid, blockDim.x, pipe);

    pipe.producer_commit();

    for (; tile_step < n; tile_step += valid) {
        int last_stage = stage;
        stage = (stage + 1) % 2;

        valid = min(BC, n - tile_step);
        int next_start = tile_step + valid;
        int next = min(BC, n - next_start);
        if (next > 0) {
            pipe.producer_acquire();
            load_title_async(&k_s[stage][0][0], MAX_D, 
                k, next, d, 
                next_start, 0, d,
                n, d, tid, blockDim.x, pipe);
            load_title_async(&v_s[stage][0][0], MAX_D, 
                v, next, d, 
                next_start, 0, d, 
                n, d, tid, blockDim.x, pipe);
            pipe.producer_commit();
        }

        pipe.consumer_wait();
        if (tid < valid) {
            int key = tile_step + tid;
            if (causal && key > query) {
                scores[tid] = -INFINITY;
            } else {
                float dot = 0.0;
                for (int i = 0; i < d; i++) {
                    dot += q_s[i] * k_s[last_stage][tid][i];
                }
                scores[tid] = dot * scale;
            }
        }
        __syncthreads();

        if (tid == 0) {
            float block_m = -INFINITY;
            for (int i = 0; i < valid; i++) {
                block_m = fmax(block_m, scores[i]);
            }
            if (isinf(block_m) && block_m < 0.0f) {
                alpha = 1.0f;
                for (int i = 0; i < valid; i++) {
                    scores[i] = 0.0f;
                }
            } else {
                float m_new = fmax(block_m, m);
                alpha = isinf(m) ? 0 : expf(m - m_new);
                float tile_l = 0;
                for (int i = 0; i < valid; i++) {
                    float w = expf(scores[i] - m_new);
                    scores[i] = w;
                    tile_l += w;
                }
                l = l * alpha + tile_l;
                m = m_new;
            }
        }
        __syncthreads();

        for (int x = tid; x < d; x += blockDim.x) {
            float add = 0.0;
            for (int i = 0; i < valid; i++) {
                add += scores[i] * v_s[last_stage][i][x];
            }
            acc[x] = acc[x] * alpha + add;
        }
        pipe.consumer_release();
    }
    for (int i = tid; i < d; i += blockDim.x) {
        out_row[i] = acc[i] / l;
    }

}

// ===========================================================================
// 测试框架（CPU 参考 + 正确性 + 计时）—— 脚手架已写好，无需改动
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

static bool run_case(int n, int d, bool causal, bool timed) {
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

    tiled_attention_pipelined<<<n, 128>>>(dq, dk, dv, dout, n, d, causal);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hgot.data(), dout, n*d*sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f; bool finite = true;
    for (int idx = 0; idx < n*d; ++idx) {
        finite = finite && std::isfinite(hgot[idx]);
        max_abs = std::max(max_abs, std::fabs(hgot[idx] - href[idx]));
    }
    bool ok = finite && max_abs < 2e-4f;

    // 可选计时：多次运行取平均（用 CUDA event）
    float ms = 0.0f;
    if (timed) {
        cudaEvent_t beg, end;
        CK(cudaEventCreate(&beg)); CK(cudaEventCreate(&end));
        const int warmup = 5, iters = 50;
        for (int i = 0; i < warmup; ++i)
            tiled_attention_pipelined<<<n, 128>>>(dq, dk, dv, dout, n, d, causal);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(beg));
        for (int i = 0; i < iters; ++i)
            tiled_attention_pipelined<<<n, 128>>>(dq, dk, dv, dout, n, d, causal);
        CK(cudaEventRecord(end));
        CK(cudaEventSynchronize(end));
        CK(cudaEventElapsedTime(&ms, beg, end));
        ms /= iters;
        CK(cudaEventDestroy(beg)); CK(cudaEventDestroy(end));
    }

    if (timed)
        std::printf("N=%-4d D=%-3d causal=%d  max_abs=%.3e  %-4s  %.4f ms/iter\n",
                    n, d, causal, max_abs, ok ? "PASS" : "FAIL", ms);
    else
        std::printf("N=%-4d D=%-3d causal=%d  max_abs=%.3e  %s\n",
                    n, d, causal, max_abs, ok ? "PASS" : "FAIL");

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
    return ok;
}

int main() {
    bool ok = true;
    // 正确性：覆盖非整除 N、causal/非 causal
    ok = run_case(3,    2,  false, false) && ok;
    ok = run_case(8,    8,  false, false) && ok;
    ok = run_case(8,    8,  true , false) && ok;
    ok = run_case(37,   24, false, false) && ok;   // 非整除 N（Bc 不整除）
    ok = run_case(37,   24, true , false) && ok;
    ok = run_case(128,  64, false, false) && ok;
    ok = run_case(128,  64, true , false) && ok;
    // 性能：大一点的规模看流水收益
    std::printf("--- timing ---\n");
    ok = run_case(2048, 64, false, true) && ok;
    ok = run_case(2048, 64, true , true) && ok;
    std::printf("%s\n", ok ? "ALL PASS" : "SOME FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
