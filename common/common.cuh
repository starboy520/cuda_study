
#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

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

__device__ __forceinline__ float warpReduceSumF_cg(float value) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    return cg::reduce(warp, value, cg::plus<float>());
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

__device__ __forceinline__ float blockReduceSumF_cg(float value) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    value = cg::reduce(warp, value, cg::plus<float>());
    __shared__ float s[MAX_WARP_NUM];
    if (warp.thread_rank() == 0) {
        s[warp.meta_group_rank()] = value;
    }
    block.sync();

    int num_warps = warp.meta_group_size();
    if (warp.meta_group_rank() == 0) {
        float bv = (warp.thread_rank() < num_warps) ? s[warp.thread_rank()] : 0.0f;
        value = cg::reduce(warp, bv, cg::plus<float>());
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
