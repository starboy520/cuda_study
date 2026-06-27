#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>

#include "../../common/cuda_check.cuh"

// ============================ Register Tiling GEMM ============================
// 设计契约（启动时必须满足）：
//   dim3 block(8, 8);                                  // 一个 block 64 个线程
//   dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);               // x 管列(N)、y 管行(M)
// 分块关系：
//   一个 block 算 C 的 BM×BN = 64×64 块；一个线程算 TM×TN = 8×8 个输出
//   线程数 = (BM/TM)×(BN/TN) = 8×8 = 64
// 数据流三级：global(慢) → shared(快, block 共用) → register(最快, thread 私有)
// 黄金律：协作加载时 shared[局部行][局部列] = global[基址行+局部行][基址列+局部列]
//        “局部行/列”表达式在 shared 下标和 global 下标里必须一字不差地相同。
// 注意：BK=128 时 shared=(64*128+128*64)*4=64KB > T4 默认 48KB 上限 → 启动失败。
//      这里先用 BK=16（sM[64][16]+sN[16][64]=8KB）让 T4 跑得动；
//      调大需用 cudaFuncSetAttribute 提升 shared 上限。
constexpr int TM = 8;   // 每个线程算 C 的 8 行
constexpr int TN = 8;   // 每个线程算 C 的 8 列
constexpr int BM = 64;  // 一个 block 算 C 的 64 行
constexpr int BN = 64;  // 一个 block 算 C 的 64 列
constexpr int BK = 16;  // K 维一次切多厚搬进 shared（须是 8 的倍数）

__global__ void gemm_register_tiled(const float* a, const float* b, float* c,
                                    int M, int N, int K) {
    // 当前 K 段的 A/B 子块（block 内所有线程共用）
    __shared__ float sM[BM][BK];   // A 子块：64 行 × BK 列
    __shared__ float sN[BK][BN];   // B 子块：BK 行 × 64 列

    // ---- 我的推导笔记 ----
    // 搬运数据，  sM: 每个线程搬运 8*(128/8) 数据
    // 每个block搬运 64*128, 每个block (8*8) 个线程，
    // 每个线程搬运(8*16)的数据 -- 重要
    // 但是每个线程需要搬完所有的K, 所以， 我们要沿着K 来切分，每次128(BK)个数
    //  a[i(0...7)][j(0...15)] 对应的   global:
    //  row = blockIdx.y * blockDim.y + threadIdx.y *TM + i;
    //  column = j * 8+ theadIdx.x  + k_block

    // 64 个累加器，全程住寄存器，跨所有 k_block 持续累加（永不清零）
    float c_reg[TM][TN] = {0.0f};

    // ===== 沿 K 维一大段一大段处理（宏观分块）=====
    for (int k_block = 0; k_block < K; k_block += BK) {

        // ---- ① 协作加载 A 子块到 sM ----
        // 每个线程搬 TM × (BK/8) 个 A 元素，铺满 sM[64][BK]
        for (int i = 0; i < TM; i++) {                  // 这个线程负责的 8 行
            int row = blockIdx.y * BM + threadIdx.y * TM + i;   // A 的全局行
            for (int j = 0; j < BK / 8; j++) {          // 沿 K 方向分 BK/8 段搬
                int column = k_block + j * 8 + threadIdx.x;     // A 的全局列(K 维)
                // 局部行 = ty*8+i，局部列 = j*8+tx（与 global 偏移一致）
                if (row < M && column < K) {
                    sM[threadIdx.y * 8 + i][j * 8 + threadIdx.x] = a[row * K + column];
                } else {
                    sM[threadIdx.y * 8 + i][j * 8 + threadIdx.x] = 0.0f;  // 越界填 0
                }
            }
        }

        // ---- ② 协作加载 B 子块到 sN ----
        // 注意：B 是 K×N，K 在行方向 → 加载索引和 A 镜像
        for (int i = 0; i < BK / 8; i++) {              // 沿 K(行)方向分段
            for (int j = 0; j < TN; j++) {              // 这个线程负责的 8 列
                int row = k_block + i * 8 + threadIdx.y;            // B 的全局行(K 维)
                int column = blockIdx.x * BN + threadIdx.x * 8 + j; // B 的全局列
                // 局部行 = i*8+ty，局部列 = tx*8+j（与 global 偏移一致）
                if (row < K && column < N) {
                    sN[i * 8 + threadIdx.y][threadIdx.x * 8 + j] = b[row * N + column];
                } else {
                    sN[i * 8 + threadIdx.y][threadIdx.x * 8 + j] = 0.0f;  // 越界填 0
                }
            }
        }

        __syncthreads();   // 等所有线程都搬完，shared 才完整可用

        // ---- ③ 在寄存器里做 8×8 外积（这一段 K 的部分和）----
        for (int k = 0; k < BK; k++) {
            // 从 shared 取这个线程要用的 8 个 A（自己的 8 行，第 k 列）
            float a_reg[TM];
            for (int i = 0; i < TM; i++) {
                a_reg[i] = sM[threadIdx.y * TM + i][k];
            }
            // 从 shared 取这个线程要用的 8 个 B（第 k 行，自己的 8 列）
            float b_reg[TN];
            for (int i = 0; i < TN; i++) {
                b_reg[i] = sN[k][threadIdx.x * TN + i];
            }
            // 外积：8 个 A × 8 个 B = 64 次乘加，累加到寄存器
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    c_reg[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();   // 等所有线程算完，才能覆盖 shared 装下一段 K
    }

    // ===== ④ 所有 K 段跑完，c_reg 才是最终结果，一次性写回 =====
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int row = blockIdx.y * BM + threadIdx.y * TM + i;
            int column = blockIdx.x * BN + threadIdx.x * TN + j;
            if (row < M && column < N) {
                c[row * N + column] = c_reg[i][j];
            }
            // 越界（尾块）直接丢弃，不写
        }
    }
}


constexpr int TILE_SIZE = 16;
__global__ void gemm_tiled(const float* a, const float*b, float*c, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    int row = threadIdx.y + blockDim.y * blockIdx.y;  // c 的行
    int column = threadIdx.x + blockDim.x * blockIdx.x;  // C  的列

    float sum = 0.0;
    for (int t = 0; t < (K + TILE_SIZE-1)/TILE_SIZE; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;  // a的行不变， 列一直在变， 
        int bRow = t * TILE_SIZE + threadIdx.y;  // b的列不变，行一直在变

        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K)? a[row * K + aCol] : 0; 


        tileB[threadIdx.y][threadIdx.x] = (bRow < K && column < N) ? b[bRow * N + column] : 0;

        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();

    }
    if (row < M && column < N) c[row * N + column] = sum;

}



// 选择要测哪个 kernel
enum KernelKind { KERNEL_TILED, KERNEL_REG };

float test_matmul_gpu(const float* A, const float* B, float* C, int M, int N,
    int K, KernelKind kind) {
float* d_A = nullptr;
float* d_B = nullptr;
float* d_C = nullptr;

CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

cudaEvent_t start, stop;
CUDA_CHECK(cudaEventCreate(&start));
CUDA_CHECK(cudaEventCreate(&stop));

CUDA_CHECK(cudaEventRecord(start, 0));
if (kind == KERNEL_TILED) {
    // 基础 tiled：block 16×16，一个线程算 1 个输出
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
} else {
    // register tiled：block 8×8，一个线程算 TM×TN 个输出
    dim3 block(BN / TN, BM / TM);                    // (8, 8)
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM); // x 管列(N)、y 管行(M)
    gemm_register_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}
CUDA_CHECK(cudaEventRecord(stop, 0));
CUDA_CHECK(cudaEventSynchronize(stop));

float kernel_ms = 0.0f;
CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
CUDA_CHECK(cudaGetLastError());

CUDA_CHECK(cudaEventDestroy(start));
CUDA_CHECK(cudaEventDestroy(stop));

CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

CUDA_CHECK(cudaFree(d_A));
CUDA_CHECK(cudaFree(d_B));
CUDA_CHECK(cudaFree(d_C));

return kernel_ms;
}

// ---- CPU 参考：三重循环 GEMM ----
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];   // A[i][k] * B[k][j]
            }
            C[i * N + j] = sum;
        }
    }
}

// ---- 校验：相对误差容差（float 累加有误差）----
bool check_result(const float* C_cpu, const float* C_gpu, int M, int N,
                  float rel_eps = 1e-3f) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            const float cpu = C_cpu[i * N + j];
            const float gpu = C_gpu[i * N + j];
            const float diff = std::fabs(cpu - gpu);
            const float denom = std::max({std::fabs(cpu), std::fabs(gpu), 1.0f});
            if (diff / denom > rel_eps) {
                std::cerr << "FAIL at (" << i << "," << j << "): cpu=" << cpu
                          << " gpu=" << gpu << " rel=" << diff / denom << "\n";
                return false;
            }
        }
    }
    std::cout << "PASS\n";
    return true;
}

int main(int argc, char** argv) {
    // 默认 512³；可命令行指定非方阵：./gemm_naive M N K
    int M = 512, N = 512, K = 512;
    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    printf("M=%d N=%d K=%d\n", M, N, K);

    std::vector<float> A(M * K), B(K * N), C_cpu(M * N, 0.0f), C_gpu(M * N, 0.0f);
    // 用小值填充，避免 float 累加溢出（值太大校验会假 FAIL）
    for (int i = 0; i < M * K; i++) A[i] = static_cast<float>((i % 13) - 6) * 0.1f;
    for (int i = 0; i < K * N; i++) B[i] = static_cast<float>((i % 7) - 3) * 0.1f;

    matmul_cpu(A.data(), B.data(), C_cpu.data(), M, N, K);

    // ---- 测 tiled 版 ----
    std::vector<float> C_tiled(M * N, 0.0f);
    const float ms_tiled = test_matmul_gpu(A.data(), B.data(), C_tiled.data(), M, N, K, KERNEL_TILED);
    const double gflops_tiled = 2.0 * M * N * K / (ms_tiled / 1e3) / 1e9;
    printf("\n[gemm_tiled]          %.3f ms, %.1f GFLOPS\n", ms_tiled, gflops_tiled);
    std::cout << "  "; check_result(C_cpu.data(), C_tiled.data(), M, N);

    // ---- 测 register tiled 版 ----
    std::vector<float> C_reg(M * N, 0.0f);
    const float ms_reg = test_matmul_gpu(A.data(), B.data(), C_reg.data(), M, N, K, KERNEL_REG);
    const double gflops_reg = 2.0 * M * N * K / (ms_reg / 1e3) / 1e9;
    printf("[gemm_register_tiled] %.3f ms, %.1f GFLOPS\n", ms_reg, gflops_reg);
    std::cout << "  "; bool ok = check_result(C_cpu.data(), C_reg.data(), M, N);

    printf("\n加速比(register / tiled): %.2fx\n", ms_tiled / ms_reg);
    return ok ? 0 : 1;
}