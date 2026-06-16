#include <cstdio>
#include <cuda_runtime.h>



int main(int argc, char** argv) {
    cudaEvent_t event;
    cudaStream_t stream1;
    cudaStream_t stream2;


    size_t size = 1 << 20;
    float* d_data = nullptr;
    float* h_data = nullptr;

    cudaMalloc(&d_data, size * sizeof(float));
    cudaMallocHost(&h_data, size * sizeof(float));

    bool copStarted = false;

    cudaEventCreate(&event);

    return 0;
}

