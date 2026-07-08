// Week5 Day6：INT8 weight-only 量化 GEMV —— y = (Q·scale)·x
//
// 背景：decode 是 memory-bound，瓶颈在读权重 W 的带宽。把 W 从 FP32(4B) 压成
//   INT8(1B)，读带宽降到 1/4。代价：需要 scale + 反量化，且字节减半 ≠ 时间减半。
//
// 关键概念（weight-only）：
//   只有权重 W 变 int8；输入 x、累加、输出 y 全程是 float。
//   量化(quantize)是【离线一次】的预处理(这里在 CPU 造数据时做)；
//   反量化+GEMV(dequant)是【在线每步】在 GPU 跑 —— 就是你要写的 kernel。
//
// 数学（对称 per-channel，每行一个 scale）：
//   量化:   scale[r] = max(|W[r,:]|)/127;  Q[r,k] = clamp(round(W[r,k]/scale[r]), -127,127)
//   反量化+点积: y[r] = Σ_k  (Q[r,k] * scale[r]) * x[k]
//                        └────── 当场把 int8 还原回近似 float ──────┘
//
// 形状：Q[N,K](int8,行主序)、scales[N](float)、x[K](float)、y[N](float)
//
// 分工：
//   - CPU 量化 + CPU 反量化参考 + FP 原始参考 + 测试框架 已写好，无需改动
//   - GPU kernel dequant_gemv_warp（warp 一行）由你实现（TODO）
//
// 编译：
//   nvcc -O3 -lineinfo -std=c++17 -arch=sm_80 dequant_gemv.cu -o dequant_gemv
//   compute-sanitizer --tool memcheck ./dequant_gemv

#include <cuda_runtime.h>
#include "../common/common.cuh"   // warpReduceSumF（warp 内 32→1 归约到 lane 0）

// ===========================================================================
// 【你来写】INT8 反量化 GEMV：一个 warp 负责一行
//   Q[N,K] int8 行主序；scales[N]；x[K] float；y[N] float
// ===========================================================================
__global__ void dequant_gemv_warp(const int8_t* Q, const float* scales,
                                   const float* x, float* y, int N, int K) {
    // 建议步骤：
    // 1. 定位：lane = threadIdx.x&31; 本 warp 的全局编号 = row
    //      int warps_per_block = blockDim.x/32;
    //      int row = blockIdx.x*warps_per_block + threadIdx.x/32;
    //      if (row >= N) return;
    // 2. 读本行 scale：float scale = scales[row];
    // 3. lane-stride 点积：for (k=lane; k<K; k+=32)
    //      int8 q = Q[row*K + k];                 // 有符号，别当 uint8
    //      sum = fmaf((float)q * scale, x[k], sum);  // 反量化 q*scale 再乘 x
    // 4. warp 归约：sum = warpReduceSumF(sum);  // 结果落 lane 0
    // 5. lane 0 写回：if (lane==0) y[row] = sum;
    //
    // 提示：
    //   - Q[row*K+k] 是 int8_t（-128~127），转 float 用 (float)q，注意 signedness。
    //   - 同一 warp 的 lane 在同轮访问连续 k → 合并访问（int8 更要注意 sector）。
    //   - scale 每行相同，各 lane 直接读 scales[row]（靠 cache 广播）即可。

    // 一个warp 负责一行  

    int num_thread_per_warp = 32;
    int num_warps_per_block = blockDim.x/num_thread_per_warp;
    int row = num_warps_per_block * blockIdx.x + threadIdx.x / 32;
    if (row >= N) return;

    const int8_t* q_row = Q + row * K;
    const float* x_row = x;
    float total = 0.0f;
    float scale = scales[row];
    for (int i = threadIdx.x%32; i < K; i+=32) {
        int8_t cur = q_row[i];
        total = fmaf(x_row[i], scale * q_row[i], total);
    }
    total = warpReduceSumF(total);
    if (threadIdx.x % 32 == 0) {
        y[row] = total;
    }
}

__global__ void dequant_gemv_warp_vec(const int8_t* Q, const float* scales,
                                   const float* x, float* y, int N, int K) {

    int num_thread_per_warp = 32;
    int num_warps_per_block = blockDim.x/num_thread_per_warp;
    int row = num_warps_per_block * blockIdx.x + threadIdx.x / 32;
    if (row >= N) return;

    const int8_t* q_row = Q + row * K;
    const float* x_row = x;
    
    float total = 0.0f;
    float scale = scales[row];
    const char4* vec_q_row = reinterpret_cast<const char4*>(q_row);
    const float4* vec_x_row = reinterpret_cast<const float4*>(x_row);
    for (int i = threadIdx.x%32; i < K/4; i += 32) {
        total = fmaf(vec_x_row[i].x, vec_q_row[i].x * scale, total);
        total = fmaf(vec_x_row[i].y, vec_q_row[i].y * scale, total);
        total = fmaf(vec_x_row[i].z, vec_q_row[i].z * scale, total);
        total = fmaf(vec_x_row[i].w, vec_q_row[i].w * scale, total);
    }

    if (threadIdx.x%32 == 0 && K % 4 != 0) {
        for (int i = K/4 * 4; i < K; i++) {
            total = fmaf(x_row[i], q_row[i] * scale, total);
        }
    }
    total = warpReduceSumF(total);
    if (threadIdx.x % 32 == 0) {
        y[row] = total;
    }
}

// ===========================================================================
// CPU 量化 + 参考 + 测试框架（已写好，无需改动）
// ===========================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
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

// ---- 离线量化：float W[N,K] → int8 Q[N,K] + float scales[N]（每行一个 scale）----
static void quantize_per_row(const float* W, int8_t* Q, float* scales,
                             int N, int K) {
    for (int r = 0; r < N; ++r) {
        float amax = 0.0f;
        for (int k = 0; k < K; ++k) amax = std::fmax(amax, std::fabs(W[r*K+k]));
        float s = (amax == 0.0f) ? 1.0f : amax / 127.0f;   // 全零行兜底
        scales[r] = s;
        for (int k = 0; k < K; ++k) {
            int q = (int)std::lrintf(W[r*K+k] / s);
            Q[r*K+k] = (int8_t)std::max(-127, std::min(127, q));  // clamp
        }
    }
}

// ---- 参考 A：反量化 GEMV（golden，验证你的 kernel 对不对）----
//   和 kernel 用同一份 Q/scale，所以误差应极小（只差浮点累加顺序）。
static void dequant_gemv_cpu(const int8_t* Q, const float* scales,
                             const float* x, float* y, int N, int K) {
    for (int r = 0; r < N; ++r) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k)
            sum += (double)Q[r*K+k] * scales[r] * x[k];
        y[r] = (float)sum;
    }
}

// ---- 参考 B：原始 FP GEMV（用未量化的 W，衡量“量化损失”，仅报告不判 PASS）----
static void gemv_fp_cpu(const float* W, const float* x, float* y, int N, int K) {
    for (int r = 0; r < N; ++r) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) sum += (double)W[r*K+k] * x[k];
        y[r] = (float)sum;
    }
}

// 造权重 W（data_mode 控制数值分布，测数值稳定性）
static void gen_W(std::vector<float>& W, int N, int K, int data_mode) {
    for (int r = 0; r < N; ++r) {
        for (int k = 0; k < K; ++k) {
            size_t i = (size_t)r*K + k;
            float v;
            switch (data_mode) {
                case 1:  // 每 3 行有一整行全零（测 amax==0 兜底）
                    v = (r % 3 == 0) ? 0.0f
                                     : (float)((int)((i*13)%17)-8) / 8.0f;
                    break;
                case 2:  // 极端范围：行内混大值与小值（测 per-row scale 的意义）
                    v = ((k & 1) ? 1000.0f : 0.01f) * (((i*7)%5)-2);
                    break;
                default: // 固定随机
                    v = (float)((int)((i*17)%23)-11) / 11.0f;
            }
            W[i] = v;
        }
    }
}

static bool run_case(int N, int K, int data_mode) {
    std::vector<float> hW((size_t)N*K), hx(K);
    std::vector<int8_t> hQ((size_t)N*K);
    std::vector<float> hScales(N);
    std::vector<float> ref_dequant(N), ref_fp(N), got(N);

    gen_W(hW, N, K, data_mode);
    for (int k = 0; k < K; ++k) hx[k] = (float)((int)((k*13)%19)-9) / 9.0f;

    // 离线量化（CPU）
    quantize_per_row(hW.data(), hQ.data(), hScales.data(), N, K);
    // 两个参考
    dequant_gemv_cpu(hQ.data(), hScales.data(), hx.data(), ref_dequant.data(), N, K);
    gemv_fp_cpu(hW.data(), hx.data(), ref_fp.data(), N, K);

    // 上 GPU（只传 int8 的 Q + float 的 scale/x）
    int8_t *dQ; float *dScales, *dx, *dy;
    CK(cudaMalloc(&dQ, hQ.size()*sizeof(int8_t)));
    CK(cudaMalloc(&dScales, N*sizeof(float)));
    CK(cudaMalloc(&dx, K*sizeof(float)));
    CK(cudaMalloc(&dy, N*sizeof(float)));
    CK(cudaMemcpy(dQ, hQ.data(), hQ.size()*sizeof(int8_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dScales, hScales.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, hx.data(), K*sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;                       // 8 warps/block
    int warps_per_block = threads / 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;

    // 算相对误差（对 ref_dequant），非有限值返回 inf
    auto rel_err = [&](const std::vector<float>& g) -> double {
        double r = 0; bool fin = true;
        for (int i = 0; i < N; ++i) {
            fin = fin && std::isfinite(g[i]);
            double d = std::fabs((double)g[i] - ref_dequant[i]);
            r = std::max(r, d / (std::fabs((double)ref_dequant[i]) + 1e-6));
        }
        return fin ? r : INFINITY;
    };

    // --- 标量版 ---
    dequant_gemv_warp<<<blocks, threads>>>(dQ, dScales, dx, dy, N, K);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), dy, N*sizeof(float), cudaMemcpyDeviceToHost));
    double k_rel = rel_err(got);

    // --- 向量化版（char4 要求行首 4 字节对齐 → 只在 K%4==0 跑）---
    double v_rel = -1.0;   // <0 表示跳过
    if (K % 4 == 0) {
        std::vector<float> got2(N);
        CK(cudaMemset(dy, 0, N*sizeof(float)));
        dequant_gemv_warp_vec<<<blocks, threads>>>(dQ, dScales, dx, dy, N, K);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got2.data(), dy, N*sizeof(float), cudaMemcpyDeviceToHost));
        v_rel = rel_err(got2);
    }

    // 量化损失（dequant 参考 vs 原始 FP）—— 只报告，不判 PASS
    double q_rel = 0;
    for (int r = 0; r < N; ++r) {
        double d = std::fabs((double)ref_dequant[r] - ref_fp[r]);
        q_rel = std::max(q_rel, d / (std::fabs((double)ref_fp[r]) + 1e-6));
    }

    bool ok = std::isfinite(k_rel) && k_rel < 1e-3
              && (v_rel < 0 || (std::isfinite(v_rel) && v_rel < 1e-3));
    const char* mode_name[] = {"rand", "zero", "extreme"};
    if (v_rel < 0)
        std::printf("N=%-5d K=%-5d [%-7s] 标量=%.3e  向量化=跳过(K%%4!=0)  量化损失=%.3e  %s\n",
                    N, K, mode_name[data_mode], k_rel, q_rel, ok ? "PASS" : "FAIL");
    else
        std::printf("N=%-5d K=%-5d [%-7s] 标量=%.3e  向量化=%.3e  量化损失=%.3e  %s\n",
                    N, K, mode_name[data_mode], k_rel, v_rel, q_rel, ok ? "PASS" : "FAIL");

    cudaFree(dQ); cudaFree(dScales); cudaFree(dx); cudaFree(dy);
    return ok;
}

int main() {
    bool ok = true;
    ok = run_case(3,    4,    0) && ok;   // 对齐手算
    ok = run_case(37,   29,   0) && ok;   // 非整除
    ok = run_case(4096, 4096, 0) && ok;   // 大规模
    ok = run_case(64,   512,  1) && ok;   // 全零行（amax==0 兜底）
    ok = run_case(64,   512,  2) && ok;   // 极端范围（per-row scale 的价值）
    std::printf("%s\n", ok ? "ALL PASS" : "SOME FAIL");
    std::printf("说明：kernel_rel 判对错（应 <1e-3）；量化损失_rel 只报告"
                "（int8 精度天生的误差，extreme 会明显偏大）。\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
