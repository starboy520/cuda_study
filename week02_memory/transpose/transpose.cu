#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// 统一约定（与 mat_mul / vec_add 一致）
//
//   threadIdx.x / blockIdx.x  →  列 col（横向）
//   threadIdx.y / blockIdx.y  →  行 row（纵向）
//
//   行主序索引：matrix[row * width + col]
//
//   转置：out[col][row] = in[row][col]
//         out[row_out * height + col_out] = in[row_in * width + col_in]
// ---------------------------------------------------------------------------

constexpr int TILE = 32;

// 版本 1：只用 global memory（读合并、写不合并，作对比基线）
__global__ void transpose_naive(const float* in, float* out, int width,
                                int height) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;  // x → 列
  const int row = blockIdx.y * blockDim.y + threadIdx.y;  // y → 行

  if (row < height && col < width) {
    const int in_idx = row * width + col;
    const int out_idx = col * height + row;  // out 尺寸 height×width → 宽=height
    out[out_idx] = in[in_idx];
  }
}

// 版本 2a：shared memory，同一线程读/写（最易理解，正确性优先）
__global__ void transpose_shared_simple(const float* in, float* out, int width,
                                        int height) {
  __shared__ float tile[TILE][TILE + 1];

  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  // 阶段 1：读 in[row][col] → shared
  if (row < height && col < width) {
    tile[threadIdx.y][threadIdx.x] = in[row * width + col];
  }
  __syncthreads();

  // 阶段 2：同一 (row,col) 写到 out[col][row]
  if (row < height && col < width) {
    out[col * height + row] = tile[threadIdx.y][threadIdx.x];
  }
}

// 版本 2b：shared memory + block 映射交换（写回也合并，Week 2 优化版）
__global__ void transpose_shared(const float* in, float* out, int width,
                               int height) {
  __shared__ float tile[TILE][TILE + 1];

  const int col_in = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_in = blockIdx.y * blockDim.y + threadIdx.y;

  if (row_in < height && col_in < width) {
    tile[threadIdx.y][threadIdx.x] = in[row_in * width + col_in];
  }
  __syncthreads();

  // 写回时交换 block 内 x/y 映射，使 global write 合并
  const int col_out = blockIdx.y * blockDim.y + threadIdx.x;
  const int row_out = blockIdx.x * blockDim.x + threadIdx.y;

  if (row_out < width && col_out < height) {
    out[row_out * height + col_out] = tile[threadIdx.x][threadIdx.y];
  }
}

// ---------------------------------------------------------------------------
// Host 工具
// ---------------------------------------------------------------------------

void transpose_cpu(const float* in, float* out, int width, int height) {
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      out[col * height + row] = in[row * width + col];
    }
  }
}

bool verify(const std::vector<float>& ref, const std::vector<float>& got) {
  if (ref.size() != got.size()) {
    return false;
  }
  for (size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) {
      return false;
    }
  }
  return true;
}

void run_kernel(void (*kernel)(const float*, float*, int, int), const char* name,
                int width, int height, const std::vector<float>& h_in,
                const std::vector<float>& h_ref) {
  const size_t in_bytes = h_in.size() * sizeof(float);
  const size_t out_bytes = h_ref.size() * sizeof(float);

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, in_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), in_bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE, TILE);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  kernel<<<grid, block>>>(d_in, d_out, width, height);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(h_ref.size());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

  const bool ok = verify(h_ref, h_out);
  printf("%s: %s (matrix %dx%d -> %dx%d)\n", name, ok ? "PASS" : "FAIL", width,
         height, height, width);

  cudaFree(d_in);
  cudaFree(d_out);
}

int main(int argc, char** argv) {
  int width = 1024;
  int height = 1024;
  if (argc > 1) {
    width = std::atoi(argv[1]);
    height = width;
  }
  if (argc > 2) {
    height = std::atoi(argv[2]);
  }

  printf("Unified convention: x=col, y=row, row-major index row*width+col\n\n");

  std::vector<float> h_in(width * height);
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      h_in[row * width + col] = static_cast<float>(row * 1000 + col);
    }
  }

  std::vector<float> h_ref(width * height);
  transpose_cpu(h_in.data(), h_ref.data(), width, height);

  run_kernel(transpose_naive, "transpose_naive", width, height, h_in, h_ref);
  run_kernel(transpose_shared_simple, "transpose_shared_simple", width, height,
             h_in, h_ref);
  run_kernel(transpose_shared, "transpose_shared", width, height, h_in, h_ref);

  return 0;
}
