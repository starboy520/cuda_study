#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int TILE = 32;

// ============================================================
// 矩阵转置：输入 input[height][width]，输出 output[width][height]
// 布局：block = 32×32，一个 block 处理一个 32×32 tile，一线程一元素
//
// 三版你来写：
//   v1 naive  : output[col*H+row] = input[row*W+col]（写回跨步不合并）
//   v2 shared : tile[32][32]，合并读进 shared，写回时交换 x/y 下标转置
//               （写回按列读 shared → 32 路 bank conflict）
//   v3 padded : tile[32][33]，padding 消 bank conflict
//
// 提示（shared/padded 版）：
//   读入： tile[threadIdx.y][threadIdx.x] = input[row*W+col];
//   __syncthreads();
//   输出坐标要交换 block：
//     row_out = blockIdx.x*32 + threadIdx.y;
//     col_out = blockIdx.y*32 + threadIdx.x;
//   写回： output[row_out*H+col_out] = tile[threadIdx.x][threadIdx.y];  // x/y 交换=转置
// ============================================================

__global__ void transpose_naive(const float* input, float* output, int W, int H) {
    // TODO  
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    if (row < H && col < W) output[col*H + row] = input[row*W + col];
}

__global__ void transpose_shared(const float* input, float* output, int W, int H) {
    // TODO: tile[TILE][TILE]
    __shared__ float tile[TILE][TILE];
    int global_col = blockDim.x * blockIdx.x + threadIdx.x;
    int global_row = blockDim.y * blockIdx.y + threadIdx.y;
    if (global_row < H && global_col < W) {
        tile[threadIdx.y][threadIdx.x] = input[global_row * W + global_col];
    }
    __syncthreads();
    int out_row = blockDim.x * blockIdx.x + threadIdx.y;
    int out_col = blockDim.y * blockIdx.y + threadIdx.x;
    if (out_row < W && out_col < H) {
        output[out_row * H + out_col] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_padded(const float* input, float* output, int W, int H) {
    // TODO: tile[TILE][TILE+1]
    __shared__ float tile[TILE][TILE+1];
    int global_col = blockDim.x * blockIdx.x + threadIdx.x;
    int global_row = blockDim.y * blockIdx.y + threadIdx.y;
    if (global_row < H && global_col < W) {
        tile[threadIdx.y][threadIdx.x] = input[global_row * W + global_col];
    }
    __syncthreads();
    int out_row = blockDim.x * blockIdx.x + threadIdx.y;
    int out_col = blockDim.y * blockIdx.y + threadIdx.x;
    if (out_row < W && out_col < H) {
        output[out_row * H + out_col] = tile[threadIdx.x][threadIdx.y];
    }
}


// ---- 框架：保持不动 ----
using Kernel = void (*)(const float*, float*, int, int);

static float timeKernel(Kernel k, const float* d_in, float* d_out, int W, int H) {
    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);
    k<<<grid, block>>>(d_in, d_out, W, H);  // warmup
    cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    k<<<grid, block>>>(d_in, d_out, W, H);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0.0f; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

static void cpuTranspose(const std::vector<float>& in, std::vector<float>& out, int W, int H) {
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            out[c * H + r] = in[r * W + c];
}

static bool verify(const std::vector<float>& got, const std::vector<float>& ref) {
    for (size_t i = 0; i < ref.size(); ++i)
        if (got[i] != ref[i]) return false;
    return true;
}

static double bandwidthGBs(int W, int H, float ms) {
    return 2.0 * W * H * sizeof(float) / (ms / 1000.0) / 1e9;
}

int main(int argc, char** argv) {
    int W = 1024, H = 1024;
    if (argc >= 3) { W = atoi(argv[1]); H = atoi(argv[2]); }
    printf("transpose %d x %d (W x H)\n", W, H);

    const size_t n = (size_t)W * H, bytes = n * sizeof(float);
    std::vector<float> h_in(n), h_out(n), h_ref(n);
    for (size_t i = 0; i < n; ++i) h_in[i] = (float)(i % 1000) * 0.5f;
    cpuTranspose(h_in, h_ref, W, H);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    struct V { const char* name; Kernel k; };
    V versions[] = {
        {"naive ", transpose_naive},
        {"shared", transpose_shared},
        {"padded", transpose_padded},
    };
    printf("\n%-8s %12s %14s %8s\n", "版本", "Kernel(ms)", "带宽(GB/s)", "正确性");
    for (auto& v : versions) {
        float ms = timeKernel(v.k, d_in, d_out, W, H);
        cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
        bool ok = verify(h_out, h_ref);
        printf("%-8s %12.3f %14.1f %8s\n", v.name, ms, bandwidthGBs(W, H, ms),
               ok ? "PASS" : "FAIL");
    }
    printf("\n预期：带宽 naive < shared < padded（padded 消了 bank conflict）\n");

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
