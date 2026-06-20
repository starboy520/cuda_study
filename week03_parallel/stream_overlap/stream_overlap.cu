/**

1.写一个 H2D → kernel → D2H 的任务，先单 stream 串行版测耗时。
2.改成 pinned memory + 多 stream 分块版，测重叠后耗时。
3.（可选）故意用 pageable 内存，验证重叠失效。
*/

/**
cudaMallocHost(&ptr, bytes)      // pinned 内存(真异步前提)
cudaStreamCreate(&stream)        // 建非默认 stream
cudaMemcpyAsync(dst, src, bytes, kind, stream)   // 异步拷贝
kernel<<<grid, block, 0, stream>>>(...)          // 在指定 stream 启动
cudaStreamSynchronize(stream)    // 等某个 stream
cudaStreamDestroy(stream)
*/

#include <cstdio>
#include "../../common/cuda_check.cuh"

__global__ void mykernel(float * input, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
          for (int rep = 0; rep < 50; rep++) input[i] = input[i] * 1.001f + 0.5f;
    }
}

constexpr int TRANSFER_SIZE = 256 * 1024 * 1024;// 256M;
constexpr int STREAM_NUMBER = 4;

void sequence_kernel() {
    //cudaStream_t stream;
    cudaEvent_t start, stop;
    //CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float* h_data = nullptr;
    float* d_data = nullptr;
    float* o_data = nullptr;

    int array_size = TRANSFER_SIZE/sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_data, TRANSFER_SIZE));   // ✓ 直接用 cudaMalloc 分配 device 内存
    h_data = (float*) std::malloc(TRANSFER_SIZE);
    o_data = (float*) std::malloc(TRANSFER_SIZE);

    for (int i = 0; i < array_size; i++) {
        h_data[i] = i;
    }
    CUDA_CHECK(cudaMalloc(&d_data, TRANSFER_SIZE));

    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, TRANSFER_SIZE,  cudaMemcpyHostToDevice));

    mykernel<<<256, 256>>>(d_data, array_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(o_data, d_data, TRANSFER_SIZE, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    float time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    printf("time: %lf\n", time);

    CUDA_CHECK(cudaFree(d_data));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_data);
    free(o_data);
}

void async_overlap_kernel() {
    cudaStream_t streams[STREAM_NUMBER];
    float* h_data = nullptr;
    float* d_data = nullptr;
    float* o_data = nullptr;

    CUDA_CHECK(cudaMallocHost(&h_data, TRANSFER_SIZE));
    CUDA_CHECK(cudaMallocHost(&o_data, TRANSFER_SIZE));

    int array_size = TRANSFER_SIZE/sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_data, TRANSFER_SIZE));

    for (int i = 0; i < array_size; i++) {
        h_data[i] = i;
    }
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0 ; i < STREAM_NUMBER; i++) {
        cudaStreamCreate(&streams[i]);
        int offset = i *  TRANSFER_SIZE/STREAM_NUMBER/sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset, TRANSFER_SIZE/STREAM_NUMBER, cudaMemcpyHostToDevice, streams[i]));
        mykernel<<<256, 256, 0, streams[i]>>>(d_data + offset, array_size/STREAM_NUMBER);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(o_data + offset, d_data + offset, TRANSFER_SIZE/STREAM_NUMBER, cudaMemcpyDeviceToHost, streams[i]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    printf("time: %lf\n", time);


    for (int i = 0; i < STREAM_NUMBER; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    CUDA_CHECK(cudaFree(d_data));

    CUDA_CHECK(cudaFreeHost(h_data));
    CUDA_CHECK(cudaFreeHost(o_data));
}

int main() {
    sequence_kernel();
    async_overlap_kernel();
    return 0;
}