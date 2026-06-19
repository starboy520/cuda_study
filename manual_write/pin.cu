#include <iostream>
using namespace std;


#define CUDA_CHECK(call)    \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)


__global__ void pinKernel(float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    printf("%f\n", static_cast<float>(in[idx]));
}


void test_pin_overlap() {
    const int CHUNKS = 4;
    const size_t bytes = 1024 * sizeof(float);
    
    float* h_pinned = nullptr;
    float* d_data = nullptr;

    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));

    for (int i = 0; i < 1024; i++) {
        h_pinned[i] = i;
    }

    const size_t chunk_bytes = bytes / CHUNKS;


    cudaStream_t streams[CHUNKS];
    for (int i = 0; i < CHUNKS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    for (int i = 0; i < CHUNKS; i++) {
        size_t off = i * (1024 / CHUNKS);
        CUDA_CHECK(cudaMemcpyAsync(d_data + off, h_pinned + off, chunk_bytes, cudaMemcpyHostToDevice, streams[i]));
        pinKernel<<<1, 256, 0, streams[i]>>>(d_data + off, 1024 / CHUNKS);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < CHUNKS; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeHost(h_pinned));
}

int main() {
    const size_t bytes = 1024* sizeof(float);
    float* h_pinned = nullptr;

    CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
    for (int i = 0; i < 1024; i++) {
        h_pinned[i] = i;
    }

    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpyAsync(d_data, h_pinned, bytes, cudaMemcpyHostToDevice, stream));

    pinKernel<<<4, 256, 0, stream>>>(d_data, 1024);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaStreamDestroy(stream));

    test_pin_overlap();

    return 0;
}