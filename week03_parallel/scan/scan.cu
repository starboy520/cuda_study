
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

// ============================================================================
// block 级两阶段 inclusive scan：warp shuffle scan + shared 存 warp 偏移
// 目标：一个 block 内 scan 最多 1024 个元素（32 warp × 32 lane），突破单 warp 的 32 限制
// 用 <<<1, blockDim>>> 启动，blockDim 必须是 32 的倍数
constexpr int LANES_PER_WARP = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int MAX_WARPS_PER_BLOCK = MAX_THREADS_PER_BLOCK / LANES_PER_WARP;

// ============================================================================
__global__ void scanBlockTwoStage(float* data, int n) {
    __shared__ float warpSum[MAX_WARPS_PER_BLOCK];   // 每个 warp 的总和（最多 32 个 warp）

    int t    = threadIdx.x;
    int lane = t & 31;              // warp 内编号 0~31
    int wid  = t >> 5;              // 第几个 warp（t / 32）

    float val = (t < n) ? data[t] : 0.0f;

    // ── 第①步：warp 内 inclusive scan（用 __shfl_up_sync，和你上面那个 kernel 一样）──
    // TODO: 5 步 shuffle，循环后 val = 本 warp 内从头到 lane 的 inclusive scan
    for (int offset = 1; offset < 32; offset *= 2) {
        // TODO
        float cur = __shfl_up_sync(0xffffffffu, val, offset);
        if (lane >= offset) {
            val += cur;
        }
    }

    // data[idx] = val;  // 写回（先写回，后续还会改这个位置）
    if (lane == 31) {
        warpSum[wid] = val;  // 每个 warp 的最后一个 lane 的 val 就是这个 warp 的总和
    }

    // ── 第②步：每个 warp 的总和（= 该 warp 最后一个 lane 的 val）存进 shared ──
    // 提示：lane == 31 的 val 就是这个 warp 的总和
    // TODO: if (lane == 31) warpSum[wid] = val;
    __syncthreads();

    // ── 第③步：让第一个 warp 对 warpSum[] 做 exclusive scan ──
    //   得到每个 warp 的"起始偏移"= 它前面所有 warp 的总和
    //   提示：可以先做 inclusive scan 再转 exclusive，或直接 exclusive
    //   num_warps = blockDim.x / 32
    // TODO: 只让 wid == 0 的那个 warp 干活，对 warpSum 的前 num_warps 个做 exclusive scan
    if (wid == 0) {
        // TODO
        int num_warps = blockDim.x / 32;
        float s = (lane < num_warps) ? warpSum[lane] : 0.0f;
        for (int offset = 1; offset < 32; offset *= 2) {
            float cur = __shfl_up_sync(0xffffffffu, s, offset);
            if (lane >= offset) {
                s += cur;
            }
        }
        if (lane < num_warps) {
            warpSum[lane] = s;
        }
    }

    __syncthreads();
    val += (wid > 0) ? warpSum[wid - 1] : 0.0f;  // 每个 warp 的起始偏移 = 前面所有 warp 的总和

    // ── 第④步：每个元素 += 它所在 warp 的偏移（warpSum[wid]）──
    // TODO: val += warpSum[wid];

    // 写回
    if (t < n) data[t] = val;
}

__global__ void scanThreeStage(float* data, int n, float* blockSum, int blocks) {
    __shared__ float warpSum[MAX_WARPS_PER_BLOCK];
    int t = threadIdx.x;
    int lan = t % 32;
    int wid = t / 32;

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float val = (idx < n) ? data[idx] : 0.0f;

    for (int offset = 1; offset < 32; offset *= 2) {
        float cur = __shfl_up_sync(0xffffffffu, val, offset);
        if (lan >= offset) {
            val += cur;
        }
    }

    if (lan == 31) {
        warpSum[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        int numWarp = blockDim.x /32;
        float s = (lan < numWarp) ? warpSum[lan] : 0.0f;
        for (int offset = 1; offset < 32; offset *= 2) {
            float cur =__shfl_up_sync(0xffffffffu, s, offset);
            if (lan >= offset) {
                s += cur;
            }
        }
        if (lan < numWarp) {
            warpSum[lan] = s;
        }
    }

    __syncthreads();
    val += (wid > 0) ? warpSum[wid-1] : 0;

    // 写回 block 内 scan 结果（关键！否则趟3 加偏移加在原始值上）
    if (idx < n) {
        data[idx] = val;
    }

    // 输出本 block 的总和（blockSum 为 nullptr 时不写，供趟2 复用本 kernel 时避免自我覆盖）
    if (blockSum && threadIdx.x == blockDim.x - 1) {
        blockSum[blockIdx.x] = val;
    }
}

__global__ void addBlockSum(float* input, int n, float* blockSum, int blocks) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int blkIdx = blockIdx.x;
    if (idx < n) {
        input[idx] += (blkIdx > 0) ? blockSum[blkIdx - 1] : 0;
    }
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

    // ---- 测 scanBlockTwoStage（block 级两阶段，128 个元素 = 4 个 warp，跨 warp 要接得上）----
    cudaMemcpy(d_data, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    scanBlockTwoStage<<<1, N>>>(d_data, N);   // 注意：用 __shared__ 固定数组，无需第三参数
    cudaMemcpy(h_out, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("block two-stage(128,跨4warp): %s  (末值=%.0f, 期望 8128)\n",
           check_result(h_ref_inc, h_out, N) ? "PASS" : "FAIL", h_out[N-1]);

    cudaFree(d_data);

    // ========================================================================
    // 三趟 grid 级 scan 测试（N 跨多个 block）
    // ========================================================================
    {
        const int M     = 2048;          // 元素数（跨多个 block）
        const int block = 256;           // 每 block 线程数
        const int grid  = (M + block - 1) / block;   // = 8 个 block

        float* h_in  = (float*)malloc(M * sizeof(float));
        float* h_ref = (float*)malloc(M * sizeof(float));
        float* h_res = (float*)malloc(M * sizeof(float));
        for (int i = 0; i < M; i++) h_in[i] = static_cast<float>(i % 7);  // 0..6 循环，强测试
        cpu_scan(h_in, h_ref, M);        // CPU inclusive 参考

        float *d_in, *d_blockSum;
        cudaMalloc(&d_in, M * sizeof(float));
        cudaMalloc(&d_blockSum, grid * sizeof(float));   // 每 block 一个总和
        cudaMemcpy(d_in, h_in, M * sizeof(float), cudaMemcpyHostToDevice);

        // 趟1：每个 block scan 自己的 tile + 输出 block 总和
        scanThreeStage<<<grid, block>>>(d_in, M, d_blockSum, grid);
        // 趟2：对 blockSum（grid 个）再 inclusive scan（1 个 block 搞定，要求 grid ≤ 1024）
        //      blockSum 传 nullptr：趟2 只想 scan d_blockSum，不要再写 block 总和（否则自我覆盖）
        scanThreeStage<<<1, grid>>>(d_blockSum, grid, nullptr, 1);
        // 趟3：每个元素 += 前面所有 block 的总和（blockSum[blkIdx-1]）
        addBlockSum<<<grid, block>>>(d_in, M, d_blockSum, grid);

        cudaMemcpy(h_res, d_in, M * sizeof(float), cudaMemcpyDeviceToHost);
        printf("three-stage grid scan(M=%d, %d blocks): %s  (末值=%.0f, 期望 %.0f)\n",
               M, grid, check_result(h_ref, h_res, M) ? "PASS" : "FAIL",
               h_res[M-1], h_ref[M-1]);

        cudaFree(d_in);
        cudaFree(d_blockSum);
        free(h_in); free(h_ref); free(h_res);
    }

    // ========================================================================
    // 1M 元素计时测试（三趟版上限：grid=1024 × block=1024 = 100万）
    // ========================================================================
    {
        const int M     = 1 << 20;       // 1,048,576 个元素（正好 1M）
        const int block = 1024;          // 每 block 1024 线程
        const int grid  = (M + block - 1) / block;   // = 1024 个 block（正好到上限）

        float* h_in  = (float*)malloc(M * sizeof(float));
        float* h_ref = (float*)malloc(M * sizeof(float));
        float* h_res = (float*)malloc(M * sizeof(float));
        for (int i = 0; i < M; i++) h_in[i] = static_cast<float>(i % 7);
        cpu_scan(h_in, h_ref, M);

        float *d_in, *d_blockSum;
        cudaMalloc(&d_in, M * sizeof(float));
        cudaMalloc(&d_blockSum, grid * sizeof(float));
        cudaMemcpy(d_in, h_in, M * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // warmup（首次含 JIT/context 开销，不计时）
        scanThreeStage<<<grid, block>>>(d_in, M, d_blockSum, grid);
        scanThreeStage<<<1, grid>>>(d_blockSum, grid, nullptr, 1);
        addBlockSum<<<grid, block>>>(d_in, M, d_blockSum, grid);

        // 重新上传（warmup 改了 d_in），正式计时三趟
        cudaMemcpy(d_in, h_in, M * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(start);
        scanThreeStage<<<grid, block>>>(d_in, M, d_blockSum, grid);
        scanThreeStage<<<1, grid>>>(d_blockSum, grid, nullptr, 1);
        addBlockSum<<<grid, block>>>(d_in, M, d_blockSum, grid);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        cudaMemcpy(h_res, d_in, M * sizeof(float), cudaMemcpyDeviceToHost);

        printf("three-stage 1M(grid=%d): %s  %.3f ms  (末值=%.0f, 期望 %.0f)\n",
               grid, check_result(h_ref, h_res, M) ? "PASS" : "FAIL",
               ms, h_res[M-1], h_ref[M-1]);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_in);
        cudaFree(d_blockSum);
        free(h_in); free(h_ref); free(h_res);
    }
}