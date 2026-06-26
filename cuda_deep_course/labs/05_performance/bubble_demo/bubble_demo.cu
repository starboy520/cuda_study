// bubble_demo.cu —— 故意制造 GPU 空洞(bubble),用 nsys 观察
// 思路:每轮跑一个很快的 kernel,然后 cudaDeviceSynchronize + CPU sleep
//       → CPU 干别的事时 GPU 空闲 → 时间线上出现空洞
#include <cstdio>
#include <unistd.h>   // usleep
#include <cuda_runtime.h>

__global__ void quickKernel(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = a[i] * 1.0001f + 1.0f;   // 很轻,kernel 很快
}

int main() {
    const int n = 1 << 20;                      // 1M
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemset(d, 0, n * sizeof(float));

    const int ITERS = 6;
    for (int it = 0; it < ITERS; ++it) {
        quickKernel<<<(n + 255) / 256, 256>>>(d, n);  // GPU 干活(很短)
        cudaDeviceSynchronize();                      // 等 GPU 跑完
        usleep(10000);                                // ★CPU 睡 10ms → GPU 空闲=空洞
    }

    cudaFree(d);
    printf("done: %d iters, each = quickKernel + 10ms CPU idle\n", ITERS);
    return 0;
}
