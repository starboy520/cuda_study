#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>

#include "../../common/cuda_check.cuh"

/**
softmax(x_i) = exp(x_i) / Σ exp(x_j)
             = exp(x_i - c) / Σ exp(x_j - c)   （分子分母同乘 exp(-c)，约掉）

算法：
1. 先求出m = max(array)
2. 求sum(exp(x-m))
3. 对每个idx 算e(x)/sum(exp(x-m))

注意：块内归约只在单个 block 内有效 → 必须用 <<<1, blockSize>>> 启动，
      且 blockSize 是 32 的倍数、≤1024。
             */
__global__ void softmaxRow(const float* input,
    int length, float* output) {
        __shared__ float record[1024];
        int stride = blockDim.x * gridDim.x;
        int idx = blockDim.x * blockIdx.x + threadIdx.x;

        float localMax = -INFINITY;
        for (int i = idx ; i < length; i += stride) {
            localMax = fmax(localMax, input[i]);
        }
        __syncthreads();

        // 树形规约
        /*
        for (int s = blockDim.x/2; s > 0; s /= 2) {
            if (threadIdx.x < s) {
                record[threadIdx.x] = fmax(record[threadIdx.x], record[threadIdx.x + s]);
            }
            __syncthreads();
        }
            */
        // 用warp来做规约，
        int lane = threadIdx.x % 32;
        int warpId = threadIdx.x / 32;
        for (int offset = 16; offset > 0; offset /= 2) {
            float cur = __shfl_down_sync(0xffffffffu, localMax, offset);
            if (lane < offset) {
                localMax = fmax(localMax, cur);
            }
        }
        if (lane == 0) {
            record[warpId] = localMax;

        }
        __syncthreads();
        localMax = (lane < blockDim.x/32) ? record[lane] : -INFINITY;
        __syncthreads();

        if (warpId == 0) {
            for (int offset = 16; offset > 0; offset /= 2) {
                float cur = __shfl_down_sync(0xffffffffu, localMax, offset);
                if (lane < offset) {
                    localMax = fmax(localMax, cur);
                }
            }
            if (lane == 0) {
                record[0] = localMax;
            }
        }

        __syncthreads();
        float m = record[0];
        __syncthreads();

        float localSum = 0.0;
        for (int i = idx; i < length; i+= stride) {
            localSum += expf(input[i] - m);
        }
        //record[threadIdx.x] = localSum;
        //__syncthreads();

        /*
        for (int s = blockDim.x/2; s > 0; s /= 2) {
            if (threadIdx.x < s) {
                record[threadIdx.x] += record[threadIdx.x + s];
            }
            __syncthreads();
        }
        float sum = record[0];
        __syncthreads();
*/
        for (int offset = 16; offset > 0; offset /= 2) {
            float cur = __shfl_down_sync(0xffffffffu, localSum, offset);
            if (lane < offset) {
                localSum += cur;
            }
        }
        if (lane == 0) {
            record[warpId] = localSum;
        }
        __syncthreads();
        localSum = (lane < blockDim.x/32) ? record[lane] : 0.0f;
        __syncthreads();
        if (warpId == 0) {
            for (int offset = 16; offset > 0; offset /= 2) {
                float cur = __shfl_down_sync(0xffffffffu, localSum, offset);
                if (lane < offset) {
                    localSum += cur;
                }
            }
            if (lane == 0) {
                record[0] = localSum;
            }
        }
        __syncthreads();
        float sum = record[0];

        for (int i = idx; i < length; i += stride) {
            output[i] = expf(input[i] - m)/ sum;
        }
}

// ---- CPU 参考：数值稳定版 softmax（减 max）----
void softmax_cpu(const float* x, float* out, int n) {
    float m = -INFINITY;
    for (int i = 0; i < n; ++i) m = std::fmax(m, x[i]);
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += std::exp(x[i] - m);
    for (int i = 0; i < n; ++i) out[i] = std::exp(x[i] - m) / s;
}

// ---- 校验：每个 ∈(0,1)、行和≈1、与 CPU 逐元素比 ----
bool check(const float* cpu, const float* gpu, int n, const char* tag) {
    double row_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        if (std::isnan(gpu[i])) {
            std::cerr << tag << " FAIL: gpu[" << i << "] = NaN\n";
            return false;
        }
        row_sum += gpu[i];
        const float diff = std::fabs(cpu[i] - gpu[i]);
        const float denom = std::max({std::fabs(cpu[i]), std::fabs(gpu[i]), 1e-6f});
        if (diff / denom > 1e-4f) {
            std::cerr << tag << " FAIL at " << i << ": cpu=" << cpu[i]
                      << " gpu=" << gpu[i] << "\n";
            return false;
        }
    }
    if (std::fabs(row_sum - 1.0) > 1e-3) {
        std::cerr << tag << " FAIL: 行和 = " << row_sum << " (应≈1)\n";
        return false;
    }
    std::cout << tag << " PASS (行和=" << row_sum << ")\n";
    return true;
}

void run(const std::vector<float>& h_in, const char* tag) {
    const int n = static_cast<int>(h_in.size());
    std::vector<float> h_cpu(n), h_gpu(n);
    softmax_cpu(h_in.data(), h_cpu.data(), n);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    softmaxRow<<<1, 256>>>(d_in, n, d_out);   // 单 block,256 线程(32 的倍数)
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    check(h_cpu.data(), h_gpu.data(), n, tag);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    // ① 普通小值：验证正确性 + 平移不变
    std::vector<float> normal(1000);
    for (int i = 0; i < 1000; ++i) normal[i] = static_cast<float>((i % 13) - 6) * 0.5f;
    run(normal, "普通值 ");

    // ② 大值：验证数值稳定(朴素版会 NaN,稳定版应正常)
    std::vector<float> big(1000);
    for (int i = 0; i < 1000; ++i) big[i] = 500.0f + static_cast<float>(i % 1000);
    run(big, "大值   ");

    return 0;
}
