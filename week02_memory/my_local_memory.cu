
#include <cstdio>
#include <cmath>

#include "../common/cuda_check.cuh"



constexpr int K =8;
CONSTEXPR int N = 1 << 20;

__global__ void example_local_memory(int* input_data, int* output_data, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > total) {
        return;
    }

    float acc = input_data[idx];
