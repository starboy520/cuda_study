// Week5 Day4：融合算子 —— Residual + RMSNorm + SiLU 一个 kernel 搞定
//
// 背景：decode 里这些"小而频繁"的逐元素/归一化算子，FLOP 少但读写 activation 多，
//   易受 HBM/launch 影响。分开写是 3 个 kernel，中间结果 z、n 反复落 HBM；
//   融合成 1 个 kernel，中间结果留寄存器，只读 x/r/gamma、只写 out。
//
// 数学（一行 D 个元素）：
//   z_i   = x_i + r_i                                  # residual
//   rms   = sqrt( mean(z^2) + eps )                    # RMSNorm 的分母
//   n_i   = z_i / rms * gamma_i                        # 归一化 + 缩放
//   out_i = SiLU(n_i) = n_i * sigmoid(n_i) = n_i / (1 + exp(-n_i))
//
// HBM 账本（FP32，解释融合为什么值得）：
//   不融合(3 kernel)：residual 读x,r写z；rmsnorm 读z,gamma写n；silu 读n写out
//                    ≈ 28 B/元素（中间 z、n 反复往返 HBM）
//   融合(1 kernel)：读 x,r,gamma，写 out ≈ 16 B/元素
//
// 布局：一个 block 处理一行。x/r/out: [rows, D] 行主序；gamma: [D]（所有行共享）。
//
// 分工：CPU reference + 测试框架已写好；kernel 由你实现（TODO）。
//
// 编译：
//   nvcc -O3 -lineinfo -std=c++17 -arch=sm_80 fused_rmsnorm.cu -o fused_rmsnorm
//   compute-sanitizer --tool memcheck ./fused_rmsnorm   # 再 initcheck/racecheck/synccheck

#include <cuda_runtime.h>
#include "../common/common.cuh"   // blockReduceSumF 等（若路径不同请调整）

// ===========================================================================
// 【你来写】融合 kernel：一个 block 一行
//   x, r, out: [rows, D] 行主序；gamma: [D]
// ===========================================================================
__global__ void fused_residual_rms_silu(const float* x, const float* r,
                                         const float* gamma, float* out,
                                         int rows, int D, float eps) {
    // 建议步骤：
    // 1. row = blockIdx.x; 越界 return；定位 x_row/r_row/out_row = ...+row*D
    // 2. 第一遍：每个线程 grid-stride 累加 z^2 的局部和：
    //      float local = 0;
    //      for (d = tid; d < D; d += blockDim.x):
    //          z = x_row[d] + r_row[d];
    //          local += z * z;
    // 3. block reduction 求 Σz^2（复用 common.cuh 的 blockReduceSumF）
    // 4. 算 inv_rms 并广播给全 block：
    //      线程 0 写 __shared__ float inv = rsqrtf(sumsq/D + eps);
    //      __syncthreads(); 之后所有线程读 inv
    // 5. 第二遍：再读 x/r（整行 z 存不下寄存器，教学版重读），算 norm+SiLU 写回：
    //      for (d = tid; d < D; d += blockDim.x):
    //          z = x_row[d] + r_row[d];
    //          n = z * inv * gamma[d];
    //          out_row[d] = n / (1.0f + expf(-n));      // SiLU
    //
    // 提示：__shared__ float inv; 广播 inv_rms。注意累加建议 FP32。
    // 数学（一行 D 个元素）：
    //   z_i   = x_i + r_i                                  # residual
    //   rms   = sqrt( mean(z^2) + eps )                    # RMSNorm 的分母
    //   n_i   = z_i / rms * gamma_i                        # 归一化 + 缩放
    //   out_i = SiLU(n_i) = n_i * sigmoid(n_i) = n_i / (1 + exp(-n_i))
    // 一个block一行

    __shared__ float inv_rms;
    int row = blockIdx.x;
    const float* x_row = x + D * blockIdx.x;
    const float* r_row = r + D * blockIdx.x;
    const float* g_row = gamma + D * blockIdx.x;
    float* o_row = out + D * blockIdx.x;
    float sum = 0.0;
    for (int i = threadIdx.x ; i < D; i += blockDim.x) {
        sum += (x_row[i] + r_row[i]) * (x_row[i] + r_row[i]);
    }
    sum = blockReduceSumF(sum);
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sum/D + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < D; i+= blockDim.x) {
        float n = (x_row[i] + r_row[i]) * inv_rms * gamma[i];
        o_row[i] = n / (1.0f + expf(-n));
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

// ---- CPU 参考：double 累加平方和，作为正确性基准 ----
static void fused_cpu(const float* x, const float* r, const float* gamma,
                      float* out, int rows, int D, float eps) {
    for (int row = 0; row < rows; ++row) {
        const float* xr = x + (size_t)row * D;
        const float* rr = r + (size_t)row * D;
        float* orow = out + (size_t)row * D;
        double ss = 0.0;
        for (int d = 0; d < D; ++d) {
            double z = (double)xr[d] + rr[d];
            ss += z * z;
        }
        float inv = rsqrtf((float)(ss / D) + eps);
        for (int d = 0; d < D; ++d) {
            float z = xr[d] + rr[d];
            float n = z * inv * gamma[d];
            orow[d] = n / (1.0f + std::exp(-n));   // SiLU
        }
    }
}

static bool run_case(int rows, int D, int data_mode) {
    // data_mode: 0=固定随机, 1=全零, 2=大值, 3=正负混合极端
    std::vector<float> hx((size_t)rows*D), hr((size_t)rows*D), hg(D);
    std::vector<float> hgot((size_t)rows*D), ref((size_t)rows*D);
    for (size_t i = 0; i < hx.size(); ++i) {
        switch (data_mode) {
            case 1: hx[i] = 0.0f; hr[i] = 0.0f; break;
            case 2: hx[i] = (float)((i%7)-3) * 1000.0f; hr[i] = (float)((i%5)-2) * 800.0f; break;
            case 3: hx[i] = (i&1)? 50.0f : -50.0f; hr[i] = (i&1)? -30.0f : 30.0f; break;
            default:
                hx[i] = (float)((int)((i*17)%23)-11)/11.0f;
                hr[i] = (float)((int)((i*13)%19)-9)/9.0f;
        }
    }
    for (int d = 0; d < D; ++d) hg[d] = (float)((int)((d*7)%13)-6)/6.0f + 1.0f; // gamma≈[0.17,1.83]

    const float eps = 1e-5f;
    fused_cpu(hx.data(), hr.data(), hg.data(), ref.data(), rows, D, eps);

    float *dx, *dr, *dg, *dout;
    CK(cudaMalloc(&dx, hx.size()*sizeof(float)));
    CK(cudaMalloc(&dr, hr.size()*sizeof(float)));
    CK(cudaMalloc(&dg, D*sizeof(float)));
    CK(cudaMalloc(&dout, hx.size()*sizeof(float)));
    CK(cudaMemcpy(dx, hx.data(), hx.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dr, hr.data(), hr.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dg, hg.data(), D*sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    fused_residual_rms_silu<<<rows, threads>>>(dx, dr, dg, dout, rows, D, eps);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hgot.data(), dout, hx.size()*sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs = 0.0, max_rel = 0.0; bool finite = true;
    for (size_t i = 0; i < hgot.size(); ++i) {
        finite = finite && std::isfinite(hgot[i]);
        double diff = std::fabs((double)hgot[i] - ref[i]);
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, diff / (std::fabs((double)ref[i]) + 1e-6));
    }
    bool ok = finite && max_rel < 1e-3;
    const char* mode_name[] = {"rand", "zero", "large", "mixed"};
    std::printf("rows=%-4d D=%-5d [%-5s]  max_abs=%.3e max_rel=%.3e  %s\n",
                rows, D, mode_name[data_mode], max_abs, max_rel, ok ? "PASS" : "FAIL");

    cudaFree(dx); cudaFree(dr); cudaFree(dg); cudaFree(dout);
    return ok;
}

int main() {
    bool ok = true;
    ok = run_case(1,   4,    0) && ok;   // 对齐手算
    ok = run_case(3,   37,   0) && ok;   // 非整除
    ok = run_case(8,   4096, 0) && ok;   // 常见维度
    ok = run_case(128, 4096, 0) && ok;   // 大规模
    ok = run_case(4,   1024, 1) && ok;   // 全零（rms→只剩 eps，别 NaN/Inf）
    ok = run_case(4,   1024, 2) && ok;   // 大值（数值稳定）
    ok = run_case(4,   1024, 3) && ok;   // 正负混合
    std::printf("%s\n", ok ? "ALL PASS" : "SOME FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
