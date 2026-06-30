
#pragma once
#include <cuda_runtime.h>
// warp 内规约，把一个warp32个数据规约成一个 (落在lane 0)
constexpr int WARP_SIZE = 32;
constexpr int MAX_THEAD_NUM_PER_BLOCK = 1024;
constexpr int MAX_WARP_NUM = MAX_THEAD_NUM_PER_BLOCK / WARP_SIZE;

__device__ __forceinline__ long long  warpReduceSum(long long value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}


__device__ __forceinline__ long long blockReduceSum(long long value) {
    __shared__ long long s[MAX_WARP_NUM];
    int lane = threadIdx.x % WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;

    value = warpReduceSum(value);
    if (lane == 0) {
        s[warp] = value;
    }
    __syncthreads();
    
    int num_warp = blockDim.x / WARP_SIZE;
    value = (lane < num_warp) ? s[lane] : 0;
    if (warp == 0) {
        value = warpReduceSum(value);
    }
    return value;
}

// ---- float 版（用于 LayerNorm 等浮点归约，避免被截断成整数）----
__device__ __forceinline__ float warpReduceSumF(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float blockReduceSumF(float value) {
    __shared__ float s[MAX_WARP_NUM];
    int lane = threadIdx.x % WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;

    value = warpReduceSumF(value);
    if (lane == 0) {
        s[warp] = value;
    }
    __syncthreads();

    int num_warp = blockDim.x / WARP_SIZE;
    value = (lane < num_warp) ? s[lane] : 0.0f;
    if (warp == 0) {
        value = warpReduceSumF(value);
    }
    return value;
}

__device__ __forceinline__ float warpReduceMaxF(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmax(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}
__device__ __forceinline__ float blockReduceMaxF(float value) {
    __shared__ float s[MAX_WARP_NUM];
    int lane = threadIdx.x % WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    value = warpReduceMaxF(value);

    if (lane == 0) {
        s[warp] = value;
    }
    __syncthreads();

    int num_warp = blockDim.x / WARP_SIZE;
    value = -INFINITY;
    if (lane < num_warp) {
        value = s[lane];
    }
    if (warp == 0) {
        value = warpReduceMaxF(value);
    }
    return value;
}
