// my_transpose.cu —— 矩阵转置三版对比：naive / shared / padding
//
// 编译： nvcc -O3 -arch=sm_75 my_transpose.cu -o my_transpose
// 运行： ./my_transpose            # 默认 1024x1024
//        ./my_transpose 4096 4096
//        ./my_transpose 1003 769   # 非方阵 + 非整除，验证边界

#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int TILE = 32;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__,    \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)


// 版本 1：naive —— 写回跨步、不合并（慢）
__global__ void my_transpose_naive(const float* input, float* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        output[col * height + row] = input[row * width + col];
    }
}

// 版本 2：shared —— tile[32][32]，按列读 tile → 32 路 bank conflict
__global__ void my_transpose_shared(const float* input,
float* output, int width, int height) {
    __shared__ float tile[TILE ][TILE];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        tile[threadIdx.y][threadIdx.x] = input[row*width+col];
    }

    __syncthreads();

    const int row_out = blockIdx.x * blockDim.x + threadIdx.y;
    const int col_out = blockIdx.y * blockDim.y + threadIdx.x;

    if (row_out < width && col_out < height) {
        output[row_out*height+col_out] = tile[threadIdx.x][threadIdx.y];
    }
}


// 版本 3：padding —— tile[32][33]，bank=(x+y)%32 错位 → 消除 bank conflict（最快）
__global__ void my_transpose_padding(const float* input,
float* output, int width, int height) {
    __shared__ float tile[TILE ][TILE+1];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        tile[threadIdx.y][threadIdx.x] = input[row*width+col];
    }

    __syncthreads();

    const int row_out = blockIdx.x * blockDim.x + threadIdx.y;
    const int col_out = blockIdx.y * blockDim.y + threadIdx.x;

    if (row_out < width && col_out < height) {
        output[row_out*height+col_out] = tile[threadIdx.x][threadIdx.y];
    }
}

using TransposeKernel = void (*)(const float*, float*, int, int);

// 跑一个 kernel，返回耗时(ms)
static float timeKernel(TransposeKernel kernel, const float* d_in, float* d_out,
                        int width, int height) {
    dim3 block(TILE, TILE);
    dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);

    kernel<<<grid, block>>>(d_in, d_out, width, height);  // warmup
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    kernel<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

static void cpuTranspose(const std::vector<float>& in, std::vector<float>& out,
                         int width, int height) {
    for (int r = 0; r < height; ++r)
        for (int c = 0; c < width; ++c)
            out[c * height + r] = in[r * width + c];
}

static bool verify(const std::vector<float>& got, const std::vector<float>& ref) {
    for (size_t i = 0; i < ref.size(); ++i)
        if (got[i] != ref[i]) return false;
    return true;
}

// 有效带宽：读 N + 写 N 个元素，单位 GB/s
static double bandwidthGBs(int width, int height, float ms) {
    double bytes = 2.0 * width * height * sizeof(float);
    return bytes / (ms / 1000.0) / 1e9;
}

int main(int argc, char** argv) {
    int width = 1024, height = 1024;
    if (argc >= 3) {
        width = std::atoi(argv[1]);
        height = std::atoi(argv[2]);
    }
    printf("transpose %d x %d (width x height)\n", width, height);

    const size_t n = static_cast<size_t>(width) * height;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_in(n), h_out(n), h_ref(n);
    for (size_t i = 0; i < n; ++i) h_in[i] = static_cast<float>(i % 1000) * 0.5f;
    cpuTranspose(h_in, h_ref, width, height);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    struct Version { const char* name; TransposeKernel kernel; };
    Version versions[] = {
        {"naive ", my_transpose_naive},
        {"shared", my_transpose_shared},
        {"padded", my_transpose_padding},
    };

    printf("\n%-8s %12s %14s %8s\n", "版本", "Kernel(ms)", "带宽(GB/s)", "正确性");
    for (auto& v : versions) {
        float ms = timeKernel(v.kernel, d_in, d_out, width, height);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        bool ok = verify(h_out, h_ref);
        printf("%-8s %12.3f %14.1f %8s\n", v.name, ms,
               bandwidthGBs(width, height, ms), ok ? "PASS" : "FAIL");
    }

    printf("\n预期：naive < shared < padded（带宽递增）。\n");
    printf("padded 比 shared 快，正是消除了 32 路 bank conflict。\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}