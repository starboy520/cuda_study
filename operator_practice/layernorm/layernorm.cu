#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#include "../../common/common.cuh"

// ============================================================
// LayerNorm kernel —— 布局：一个 block 处理一行
//   x      : [rows, D]  输入
//   gamma  : [D]        缩放参数（所有行共享）
//   beta   : [D]        平移参数（所有行共享）
//   y      : [rows, D]  输出
//   D      : 每行长度
//   eps    : 防止除0
//

// μ   = (1/D) * Σ x[i]                 # 均值
// σ²  = (1/D) * Σ (x[i] - μ)²          # 方差
// y[i] = (x[i] - μ) / sqrt(σ² + eps) * gamma[i] + beta[i]

// 实现步骤（你来填）：
//   1. row = blockIdx.x; 定位本行起始 const float* xrow = x + row * D;
//   2. 第一遍归约：块内 reduce 求 Σx  → μ = sum / D
//   3. 第二遍归约：块内 reduce 求 Σ(x-μ)² → σ² = sum2 / D
//   4. float inv = rsqrtf(σ² + eps);
//   5. 每个线程 grid-stride 写回：y[i] = (x[i]-μ)*inv*gamma[i] + beta[i]
//
// 提示：μ 和 σ² 算完后要让全 block 都能读到，可用 __shared__ float 广播。
// ============================================================
__global__ void layernorm(const float* x, const float* gamma,
                          const float* beta, float* y,
                          int D, float eps) {
    __shared__ float  miu;
    __shared__ float sigma;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    int row = blockIdx.x;
    
    const float* xrow = x + row * D;    
    float* y_row = y + row * D;

    int stride = gridDim.x * blockDim.x;
    float localSum = 0;
    int idx = threadIdx.x;
    for (int i = idx ; i < D; i += blockDim.x) {
        localSum += xrow[i];
    }
    localSum = blockReduceSumF(localSum);

    if (warp == 0 && lane == 0) {
        miu = localSum / D;
    }  
    __syncthreads();
    float sum2 = 0;
    for (int i = idx; i < D; i+= blockDim.x) {
        sum2 += (xrow[i] - miu) * (xrow[i] - miu); 
    }
    sum2 = blockReduceSumF(sum2);
    if (warp == 0 && lane == 0) {
        sigma = sum2 / D;
    }
    __syncthreads();
    float inv = rsqrtf(sigma + eps);
    for (int i = idx; i < D; i += blockDim.x) {
        y_row[i] = (xrow[i] - miu) * inv  *gamma[i] + beta[i];
    }

}


// ============================================================
// RMSNorm kernel —— 一 block 一行；比 LayerNorm 少一遍归约、不减均值、无 beta
//   公式：rms = sqrt(mean(x^2) + eps); y[i] = x[i] / rms * gamma[i]
//   x      : [rows, D]  输入
//   gamma  : [D]        缩放参数（所有行共享）
//   y      : [rows, D]  输出
//
// 实现步骤（你来填）：
//   1. row = blockIdx.x; xrow = x + row*D; y_row = y + row*D;
//   2. 一遍归约：blockReduceSumF 求 Σx² → meansq = sumsq / D
//   3. inv = rsqrtf(meansq + eps);
//   4. grid-stride 写回：y_row[i] = xrow[i] * inv * gamma[i]
// ============================================================
__global__ void rmsnorm(const float* x, const float* gamma,
                        float* y, int D, float eps) {
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    __shared__ float inv;

    const float* xrow = x + blockIdx.x * D;
    float* yrow = y + blockIdx.x * D;
    float sum = 0.0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += xrow[i] * xrow[i];
    }
    sum = blockReduceSumF(sum);
    if (warp == 0 && lane == 0) {
        inv = rsqrtf(sum/D + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        yrow[i] = xrow[i] * inv * gamma[i];
    }
}


// ---- 与 layernorm_cpu 平行，校验 RMSNorm ----
static void rmsnorm_cpu(const float* x, const float* gamma,
                        float* y, int rows, int D, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * D;
        float* yr = y + r * D;
        double sumsq = 0.0;
        for (int i = 0; i < D; ++i) sumsq += (double)xr[i] * xr[i];
        double inv = 1.0 / std::sqrt(sumsq / D + eps);
        for (int i = 0; i < D; ++i)
            yr[i] = (float)(xr[i] * inv) * gamma[i];
    }
}


// ---- CPU 参考实现，用于校验 ----
static void layernorm_cpu(const float* x, const float* gamma,
                          const float* beta, float* y,
                          int rows, int D, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * D;
        float* yr = y + r * D;
        double mean = 0.0;
        for (int i = 0; i < D; ++i) mean += xr[i];
        mean /= D;
        double var = 0.0;
        for (int i = 0; i < D; ++i) var += (xr[i] - mean) * (xr[i] - mean);
        var /= D;
        double inv = 1.0 / std::sqrt(var + eps);
        for (int i = 0; i < D; ++i)
            yr[i] = (float)((xr[i] - mean) * inv) * gamma[i] + beta[i];
    }
}


int main() {
    const int   rows = 4;
    const int   D    = 1024;
    const float eps  = 1e-5f;
    const int   n    = rows * D;

    // ---- host 数据 ----
    float* h_x     = new float[n];
    float* h_gamma = new float[D];
    float* h_beta  = new float[D];
    float* h_y     = new float[n];      // GPU 结果
    float* h_ref   = new float[n];      // CPU 参考

    for (int i = 0; i < n; ++i) h_x[i] = (float)((i % 100) - 50) * 0.1f;  // 随便造点数据
    for (int i = 0; i < D; ++i) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }   // 先用 γ=1, β=0

    layernorm_cpu(h_x, h_gamma, h_beta, h_ref, rows, D, eps);

    // ---- device 数据 ----
    float *d_x, *d_gamma, *d_beta, *d_y;
    cudaMalloc(&d_x,     sizeof(float) * n);
    cudaMalloc(&d_gamma, sizeof(float) * D);
    cudaMalloc(&d_beta,  sizeof(float) * D);
    cudaMalloc(&d_y,     sizeof(float) * n);

    cudaMemcpy(d_x,     h_x,     sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, sizeof(float) * D, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta,  h_beta,  sizeof(float) * D, cudaMemcpyHostToDevice);

    // ---- launch：一个 block 处理一行，每 block 用 256 线程 ----
    int threads = 256;
    layernorm<<<rows, threads>>>(d_x, d_gamma, d_beta, d_y, D, eps);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, sizeof(float) * n, cudaMemcpyDeviceToHost);

    // ---- 校验：逐元素相对误差 ----
    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = std::fabs(h_y[i] - h_ref[i]);
        double denom = std::fabs(h_ref[i]) + 1e-6;
        max_err = std::fmax(max_err, diff / denom);
    }
    printf("max relative error = %.3e  %s\n",
           max_err, max_err < 1e-4 ? "PASS" : "FAIL");

    // ---- RMSNorm 校验 ----
    rmsnorm_cpu(h_x, h_gamma, h_ref, rows, D, eps);
    rmsnorm<<<rows, threads>>>(d_x, d_gamma, d_y, D, eps);
    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, sizeof(float) * n, cudaMemcpyDeviceToHost);
    double max_err_r = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = std::fabs(h_y[i] - h_ref[i]);
        double denom = std::fabs(h_ref[i]) + 1e-6;
        max_err_r = std::fmax(max_err_r, diff / denom);
    }
    printf("rmsnorm max relative error = %.3e  %s\n",
           max_err_r, max_err_r < 1e-4 ? "PASS" : "FAIL");

    cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_y);
    delete[] h_x; delete[] h_gamma; delete[] h_beta; delete[] h_y; delete[] h_ref;
    return 0;
}
