#include <cstdio>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 1024;
constexpr int MAX_WARP_NUM = MAX_BLOCK_SIZE / WARP_SIZE; 

__global__ void reduce(const long long * a, int size, long long* sum) {
    __shared__ long long s[MAX_WARP_NUM];

    int stride = blockDim.x * gridDim.x;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    long long local_sum = 0;
    //for (int i = idx; i < size; i += stride) {
    //    local_sum += a[i];
    //}
    // 向量化主体：每次读 2 个 long long（16B）；尾元素（size 为奇数）在循环内判断处理
    for (int i = idx * 2; i < size; i += stride * 2) {
        if (i + 1 < size) {
            longlong2 v = *reinterpret_cast<const longlong2*>(&a[i]);
            local_sum += v.x + v.y;
        } else {
            local_sum += a[i];
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffffu, local_sum, offset);
    }

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) {
        s[warp_id] = local_sum;
    }

    __syncthreads();

    // 第二次规约
    if (warp_id == 0) {
        local_sum = (lane < blockDim.x/WARP_SIZE) ? s[lane] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffffu, local_sum, offset);
        }
        if (lane == 0) {
            sum[blockIdx.x] = local_sum;
        }
    }
}


constexpr int N = 1000000;

int main() {
    long long * a = new long long[N];
    for (int i = 0; i < N; i++) a[i] = i;
    
    long long * d_a = nullptr;
    cudaMalloc(&d_a, sizeof(long long) *N);
    cudaMemcpy(d_a, a, sizeof(long long) *N, cudaMemcpyHostToDevice);

    int block_num = (N + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;

    long long* out = nullptr;
    long long* total = nullptr;
    cudaMalloc(&out, sizeof(long long ) *block_num);
    cudaMalloc(&total, sizeof(long long)* 1);
    reduce<<<block_num, MAX_BLOCK_SIZE>>>(d_a, N, out);
    reduce<<<1, MAX_BLOCK_SIZE>>>(out, block_num, total);

    cudaDeviceSynchronize();

    long long* h_out = new long long;
    cudaMemcpy(h_out, total, sizeof(long long), cudaMemcpyDeviceToHost);
    printf("out %lld\n", *h_out);

    long long expected = 0;
    for (int i = 0; i < N; i++) { a[i] = i; expected += a[i]; }
// 末尾：
    printf("gpu=%lld expected=%lld %s\n",
       *h_out, expected, *h_out == expected ? "PASS" : "FAIL");
    delete h_out;
    cudaFree(d_a);
    cudaFree(out);
    cudaFree(total);
    delete[] a;
}