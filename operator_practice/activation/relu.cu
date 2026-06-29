#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// ReLU —— 逐元素算子，整个数组平铺（不分行）
//   y[i] = max(0, x[i])
// 步骤（你来填）：grid-stride 遍历 n 个元素，写 y[i] = fmaxf(0, x[i])
// 进阶：用 float4 一次 4 个元素，处理尾部
// ============================================================
__global__ void relu(const float* x, float* y, int n) {
    // TODO: 你来实现
    int stride = blockDim.x * gridDim.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    for (int i = idx; i < n / 4; i += stride) {
        float4 v = x4[i];
        y4[i] = make_float4(fmaxf(v.x, 0.0f), fmaxf(v.y, 0.0f), fmaxf(v.z, 0.0f), fmaxf(v.w, 0.0f));
    }

    if (idx == 0) {
        for (int i = n / 4 * 4; i < n; i++) {
            y[i] = fmaxf(x[i], 0.0f);
        }
    }
}

constexpr int N = 1 << 20;

int main() {
    float* h_x = new float[N];
    float* h_y = new float[N];
    for (int i = 0; i < N; i++) h_x[i] = (float)((i % 200) - 100) * 0.1f;  // 含正负

    float *d_x, *d_y;
    cudaMalloc(&d_x, sizeof(float) * N);
    cudaMalloc(&d_y, sizeof(float) * N);
    cudaMemcpy(d_x, h_x, sizeof(float) * N, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu<<<blocks, threads>>>(d_x, d_y, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, sizeof(float) * N, cudaMemcpyDeviceToHost);

    int fail = 0;
    for (int i = 0; i < N; i++) {
        float ref = h_x[i] > 0 ? h_x[i] : 0;
        if (fabs(h_y[i] - ref) > 1e-6) { fail++; }
    }
    printf("relu %s (fail=%d)\n", fail == 0 ? "PASS" : "FAIL", fail);

    cudaFree(d_x); cudaFree(d_y);
    delete[] h_x; delete[] h_y;
    return 0;
}
