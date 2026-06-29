#include <cstdio>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 1024;
constexpr int MAX_WARP_NUM = MAX_BLOCK_SIZE / WARP_SIZE;

// 归约单元：值 + 下标
struct ValIdx {
    float val;
    int   idx;
};

// 取大（平局保留 a，即更小下标）。你可在此基础上做 warp/block 归约。
__device__ __forceinline__ ValIdx argmaxOp(ValIdx a, ValIdx b) {
    return (b.val > a.val) ? b : a;
}

// ============================================================
// argmax kernel —— 找最大值的下标
//   a    : 输入数组
//   size : 元素个数
//   out  : 每个 block 输出一个 ValIdx（需第二趟合并）
//
// 步骤（你来填）：
//   1. grid-stride：每线程维护本地 best{val,idx}
//   2. warp shuffle：val 和 idx 各 shuffle 一次，用 argmaxOp 合并
//   3. 两级归约：warp0 收各 warp 部分结果（shared ValIdx[]）
//   4. thread0 写 out[blockIdx.x]
// ============================================================
__global__ void argmax(const float* a, int size, ValIdx* out) {
    // TODO: 你来实现
    int stride = blockDim.x * gridDim.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    ValIdx local_best;
    local_best.val = -1e20f; // 初始化为一个很小的值
    local_best.idx = -1; // 初始化为无效下标
    for (int i = idx; i < size; i += stride) {
        ValIdx current;
        current.val = a[i];
        current.idx = i;
        local_best = argmaxOp(local_best, current);
    }
    // TODO: warp shuffle + shared memory 归约
    for (int offset = 16; offset > 0; offset /= 2) {
        ValIdx shuffled;
        shuffled.val = __shfl_down_sync(0xffffffffu, local_best.val, offset);
        shuffled.idx = __shfl_down_sync(0xffffffffu, local_best.idx, offset);
        local_best = argmaxOp(local_best, shuffled);
    }
    __shared__ ValIdx s[MAX_WARP_NUM];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        s[warp_id] = local_best;
    }
    __syncthreads();
    if (warp_id == 0) {
        local_best = (lane < (blockDim.x / WARP_SIZE)) ? s[lane] : ValIdx{-1e20f, -1};
        for (int offset = 16; offset > 0; offset /= 2) {
            ValIdx shuffled;
            shuffled.val = __shfl_down_sync(0xffffffffu, local_best.val, offset);
            shuffled.idx = __shfl_down_sync(0xffffffffu, local_best.idx, offset);
            local_best = argmaxOp(local_best, shuffled);
        }
        if (lane == 0) {
            out[blockIdx.x] = local_best;
        }
    }
}


// 第二趟：合并第一趟产出的 ValIdx 数组（输入已是 ValIdx，逻辑同 argmax）
__global__ void mergeArgmax(const ValIdx* in, int n, ValIdx* out) {
    int stride = blockDim.x * gridDim.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    ValIdx local_best{-1e20f, -1};
    for (int i = idx; i < n; i += stride) {
        local_best = argmaxOp(local_best, in[i]);   // 直接合并，idx 已在里面
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        ValIdx sh;
        sh.val = __shfl_down_sync(0xffffffffu, local_best.val, offset);
        sh.idx = __shfl_down_sync(0xffffffffu, local_best.idx, offset);
        local_best = argmaxOp(local_best, sh);
    }
    __shared__ ValIdx s[MAX_WARP_NUM];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) s[warp_id] = local_best;
    __syncthreads();
    if (warp_id == 0) {
        ValIdx invalid{-1e20f, -1};
        local_best = (lane < blockDim.x / WARP_SIZE) ? s[lane] : invalid;
        for (int offset = 16; offset > 0; offset /= 2) {
            ValIdx sh;
            sh.val = __shfl_down_sync(0xffffffffu, local_best.val, offset);
            sh.idx = __shfl_down_sync(0xffffffffu, local_best.idx, offset);
            local_best = argmaxOp(local_best, sh);
        }
        if (lane == 0) out[blockIdx.x] = local_best;
    }
}


constexpr int N = 1 << 20;

int main() {
    float* h = new float[N];
    for (int i = 0; i < N; i++) h[i] = (float)((i * 1103515245 + 12345) % 10007);  // 伪随机
    // CPU 参考
    float best = h[0]; int bi = 0;
    for (int i = 1; i < N; i++) if (h[i] > best) { best = h[i]; bi = i; }

    float* d_a = nullptr;
    cudaMalloc(&d_a, sizeof(float) * N);
    cudaMemcpy(d_a, h, sizeof(float) * N, cudaMemcpyHostToDevice);

    int block_num = (N + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    ValIdx *d_out, *d_total;
    cudaMalloc(&d_out, sizeof(ValIdx) * block_num);
    cudaMalloc(&d_total, sizeof(ValIdx) * 1);

    argmax<<<block_num, MAX_BLOCK_SIZE>>>(d_a, N, d_out);
    mergeArgmax<<<1, MAX_BLOCK_SIZE>>>(d_out, block_num, d_total);
    cudaDeviceSynchronize();

    ValIdx r;
    cudaMemcpy(&r, d_total, sizeof(ValIdx), cudaMemcpyDeviceToHost);
    printf("gpu idx=%d val=%.1f, cpu idx=%d val=%.1f  %s\n",
           r.idx, r.val, bi, best, (r.idx == bi) ? "PASS" : "FAIL");

    cudaFree(d_a); cudaFree(d_out); cudaFree(d_total);
    delete[] h;
    return 0;
}
