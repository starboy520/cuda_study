
#include <cstdio>
#include <ctime>
#include <cuda_runtime.h>
#include "../../common/cuda_check.cuh"


void cpu_add(int* in, int n, unsigned long long* sum) {
    for (int i = 0; i < n; i++) {
        *sum += static_cast<unsigned long long>(in[i]);
    }
}


__global__ void global_atomic_add(const int* in, int n, unsigned long long* sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(sum, static_cast<unsigned long long>(in[idx]));
    }
}


__global__ void shared_atomic_add(const int* in, int n, unsigned long long* sum) {
    __shared__ unsigned long long value;
    if (threadIdx.x == 0) {
        value = 0;
    }

    __syncthreads();

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&value, static_cast<unsigned long long>(in[idx]));
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(sum, value);
    }

}

// 方法3：完整三层聚合（寄存器 → shared → global）+ grid-stride
// 关键改进：每个线程先用 grid-stride 把多个元素攒进【寄存器 local】，
// 再每线程只发 1 次 shared atomic —— 把 shared atomic 次数从"每元素一次"降到"每线程一次"。
__global__ void register_shared_atomic_add(const int* in, int n, unsigned long long* sum) {
    __shared__ unsigned long long value;
    if (threadIdx.x == 0) {
        value = 0;
    }
    __syncthreads();

    // ① 寄存器层：grid-stride 遍历，多个元素先在寄存器累加，0 次 atomic
    unsigned long long local = 0;
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        local += static_cast<unsigned long long>(in[idx]);
    }

    // ② shared 层：每线程只发 1 次 shared atomic（而非每元素一次）
    atomicAdd(&value, local);
    __syncthreads();

    // ③ global 层：每 block 只发 1 次 global atomic
    if (threadIdx.x == 0) {
        atomicAdd(sum, value);
    }
}

using Kernel = void (*)(const int*, int, unsigned long long*);

float timing(Kernel kernel, const int* in, int total, unsigned long long* sum, int grid, int block) {
    CUDA_CHECK(cudaMemset(sum, 0, sizeof(unsigned long long)));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    kernel<<<grid, block>>>(in, total, sum);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));


    float time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&time,  start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return time;
}

int main() {
    constexpr int N = 1 << 24;
    constexpr int BLOCK_SIZE = 256;
    constexpr int GRIDSIZE = (N+BLOCK_SIZE-1)/BLOCK_SIZE;

    int* h_data = new int[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = i;
    }
    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

    unsigned long long* sum1 = nullptr;
    unsigned long long * sum2 = nullptr;
    unsigned long long * sum3 = nullptr;
    CUDA_CHECK(cudaMallocManaged(&sum1, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMallocManaged(&sum2, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMallocManaged(&sum3, sizeof(unsigned long long)));

    *sum1 = 0;
    *sum2 = 0;
    *sum3 = 0;

    // 方法3 用 grid-stride，每线程处理多个元素，所以 grid 可以小很多（这里取 1/8）
    constexpr int GRIDSIZE_STRIDE = GRIDSIZE / 8;

    float time1 = timing(global_atomic_add, d_data, N, sum1, GRIDSIZE, BLOCK_SIZE);
    float time2 = timing(shared_atomic_add, d_data, N, sum2, GRIDSIZE, BLOCK_SIZE);
    float time3 = timing(register_shared_atomic_add, d_data, N, sum3, GRIDSIZE_STRIDE, BLOCK_SIZE);


    unsigned long long cpu_result = 0;
    cpu_add(h_data, N, &cpu_result);
    if (cpu_result != *sum1 || cpu_result != *sum2 || cpu_result != *sum3) {
        printf("result not correct, cpu result: %llu, global: %llu, shared: %llu, reg+shared: %llu\n",
            cpu_result, *sum1, *sum2, *sum3);
    } else {
        printf("PASS: sum = %llu\n", cpu_result);
    }
    printf("Timing: global atomicAdd: %f ms, shared atomicAdd: %f ms, reg+shared(grid-stride): %f ms\n",
           time1, time2, time3);

    CUDA_CHECK(cudaFree(d_data));
    delete [] h_data;
    CUDA_CHECK(cudaFree(sum1));
    CUDA_CHECK(cudaFree(sum2));
    CUDA_CHECK(cudaFree(sum3));

}