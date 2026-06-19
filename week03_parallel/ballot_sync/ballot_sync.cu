/**
题目:给定一个长度为 32 的整数数组(正好一个 warp),里面有正数、负数、零。用 __ballot_sync + __popc,让 lane 0 统计出数组里有多少个正数(> 0),并打印。


unsigned bits = __ballot_sync(0xffffffffU, x > 0);
// bits 的第 i 位 = 1 表示 lane i 的 x>0
int count = __popc(bits);   // __popc 数 1 的个数 = 满足条件的 lane 数
*/


#include <cstdio>
#include <cuda_runtime.h>
__global__ void ballot_sync_kernel(const int* input, int length) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int value = 0;
    if (idx < length) {
        value = input[idx];
    }

    int lane = threadIdx.x & 31;
    unsigned int bits = __ballot_sync(0xffffffffU, value > 0);
    int count = __popc(bits);
    if (lane == 0) {
        printf("positive count: %d\n", count);
    }

}


int main() {
    int data[32] = { 3, -1, 5, 0, 2, -4, 7, 1,    // 正数: 3,5,2,7,1 → 5个
        -2, 8, 0, -3, 6, 9, -1, 4,    // 正数: 8,6,9,4 → 4个
        1, 1, 1, -5, -6, 2, 3, 0,     // 正数: 1,1,1,2,3 → 5个
        -1, -2, 7, 8, 0, 0, 5, -9 };  // 正数: 7,8,5 → 3个
        
    int* d_data = nullptr;
    cudaMallocManaged(&d_data, 32* sizeof(int));

    cudaMemcpy(d_data, data, sizeof(int)*32, cudaMemcpyHostToDevice);
    ballot_sync_kernel<<<1,32>>>(d_data, 32);

    cudaDeviceSynchronize();
    return 0;
}