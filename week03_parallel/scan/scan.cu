
#include <cstdio>

__global__ void scanHillisSteele(float* data, int n) {
    extern __shared__ float tmp[];
    int t = threadIdx.x;
    tmp[t] = (t<n) ? data[t] : 0.0f;

    __syncthreads();

    for (int offset = 1; offset <n; offset = offset * 2) {
        float add = (t >= offset) ? tmp[t-offset] : 0.0f;
        __syncthreads();
        tmp[t] += add;
        __syncthreads();
    }

    if (t < n) data[t] = tmp[t];
}

__global__ void scanHillsSteelsUsingWarpAndLane(float* data, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int value = (idx < n) ? data[idx] : 0;
    int lane = threadIdx.x & 31;
    for (int offset = 1; offset < 32; offset *=2) {
        float l = __shfl_up_sync(0xffffffffu, value, offset);
        if(lane >= offset) {
         value += l;
        }

    }
    data[idx] = value;
}


__global__ void scanHillisSteeleExclusive(float* data, int n) {
    extern __shared__ float tmp[];
    int t = threadIdx.x;
    tmp[t] = (t<n) ? data[t] : 0.0f;

    __syncthreads();

    for (int offset = 1; offset <n; offset = offset * 2) {
        float add = (t >= offset) ? tmp[t-offset] : 0.0f;
        __syncthreads();
        tmp[t] += add;
        __syncthreads();
    }

    if (t < n) {
        data[t] = (t == 0) ? 0.0f : tmp[t-1];
    }
}

void cpu_scan(const float* input, float* output, int n) {
    if (n > 0) {
        output[0] = input[0];
        for (int i = 1; i < n; i++) {
            output[i] = output[i-1] + input[i];
        }
    }
}

// exclusive: 每个位置 = 左边所有（不含自己），最左补 0
void cpu_scan_exclusive(const float* input, float* output, int n) {
    if (n > 0) {
        output[0] = 0.0f;
        for (int i = 1; i < n; i++) {
            output[i] = output[i-1] + input[i-1];  // 注意是 input[i-1]
        }
    }
}

bool check_result(const float* cpu_result, const float* gpu_result, int n) {
    for (int i = 0; i < n; i++) {
        if (abs(cpu_result[i] - gpu_result[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}


int main() {
    constexpr int N = 128;
    float h_input[N];
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i);  // 填 0,1,2...，inclusive 期望 = 三角数 i*(i+1)/2
    }

    float h_ref_inc[N];   // CPU inclusive 参考
    float h_ref_exc[N];   // CPU exclusive 参考
    float h_out[N];       // GPU 结果拷回
    cpu_scan(h_input, h_ref_inc, N);
    cpu_scan_exclusive(h_input, h_ref_exc, N);

    float* d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(float));

    // ---- 测 inclusive ----
    cudaMemcpy(d_data, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    scanHillisSteele<<<1, N, N * sizeof(float)>>>(d_data, N);
    cudaMemcpy(h_out, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("inclusive: %s  (末值=%.0f, 期望 8128)\n",
           check_result(h_ref_inc, h_out, N) ? "PASS" : "FAIL", h_out[N-1]);

    // ---- 测 exclusive（注意每次重新上传原始数据，kernel 会改写 d_data）----
    cudaMemcpy(d_data, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    scanHillisSteeleExclusive<<<1, N, N * sizeof(float)>>>(d_data, N);
    cudaMemcpy(h_out, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("exclusive: %s  (末值=%.0f, 期望 8001)\n",
           check_result(h_ref_exc, h_out, N) ? "PASS" : "FAIL", h_out[N-1]);

    // 测scanHillsSteelsUsingWarpAndLane
    cudaMemcpy(d_data, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    scanHillsSteelsUsingWarpAndLane<<<1, 32>>>(d_data, 32);   // 只跑一个 warp(32 个元素)
    cudaMemcpy(h_out, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("using warp and lane(单warp,前32): %s  (末值=%.0f, 期望 496)\n",
           check_result(h_ref_inc, h_out, 32) ? "PASS" : "FAIL", h_out[31]);



    cudaFree(d_data);
}