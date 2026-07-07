// Online Softmax 算子练习 —— 一遍(single-pass)算出 m 和 l
//
// 背景：普通 stable softmax 要 3 遍：① 求 max ② 求 Σexp(x-max) ③ 归一化。
//   online softmax 把 ①② 合并成 **一遍**：流式维护 running (m, l)，
//   出现更大 max 时把已累计的 l 乘 exp(m_old-m_new) 重缩放。
//   这是 FlashAttention 的数学核心，单独抽出来练。
//
// 数学（一个 row 的 D 个元素）：
//   目标：out[i] = exp(x[i] - M) / L，其中 M=max(x)，L=Σ_j exp(x[j]-M)
//   online 单遍：遍历元素维护 (m, l)：
//     新元素 x：m_new = max(m, x)
//               l     = l*exp(m - m_new) + exp(x - m_new)
//               m     = m_new
//   合并两个部分状态 (m1,l1) 和 (m2,l2)：
//     m = max(m1, m2)
//     l = l1*exp(m1 - m) + l2*exp(m2 - m)
//   （这个 combine 就是 online softmax 的精髓，也是你要写的重点）
//
// 布局：一个 block 处理一行。x/out: [rows, D] 行主序。
//
// 分工：CPU reference + 测试框架已写好；kernel 由你实现（TODO）。
//
// 编译：
//   nvcc -O3 -std=c++17 -arch=sm_80 online_softmax.cu -o online_softmax

#include <cuda_runtime.h>
#include <math_constants.h>   // CUDART_INF_F

// online softmax 的 running 状态：m=running max, l=相对 m 的指数和
struct MLState {
    float m;   // running max
    float l;   // running sum of exp(x - m)
};

// 合并两个 (m,l) 状态（online softmax 的核心算子）
__device__ __forceinline__ MLState combine(MLState a, MLState b) {
    // TODO（可先写这个 helper）：
    //   m = max(a.m, b.m)
    //   l = a.l*exp(a.m - m) + b.l*exp(b.m - m)
    //   处理空状态：m=-INF 时该项贡献 0（exp(-INF - m)=0），别算出 NaN
    MLState r;
    r.m = fmax(a.m, b.m);
    if (isinf(r.m)) {
        r.l = 0.0f;
    } else {
        r.l = a.l * expf(a.m - r.m) + b.l * expf(b.m - r.m);
    }

    return r;
}

// ===========================================================================
// 【你来写】online softmax kernel —— 一个 block 一行，单遍求 (m,l)
//   x, out: [rows, D] 行主序
// ===========================================================================
__global__ void online_softmax(const float* x, float* out, int rows, int D) {
    // 建议步骤：
    // 1. row = blockIdx.x; 定位 x_row = x + row*D, out_row = out + row*D
    // 2. 每个线程流式处理自己 stride 的元素，维护 local (m,l)：
    //      MLState st{-INF, 0};
    //      for (i = tid; i < D; i += blockDim.x):
    //          x_i = x_row[i];
    //          m_new = max(st.m, x_i);
    //          st.l  = st.l*expf(st.m - m_new) + expf(x_i - m_new);
    //          st.m  = m_new;
    // 3. block 内把各线程的 (m,l) 合并成全行的 (M,L)：
    //      - 简单版：各线程把 st 写进 __shared__ MLState buf[blockDim]，
    //        __syncthreads()，让 tid==0 串行 combine 全部，再广播；
    //      - 进阶版：warp 内用 shfl 对 (m,l) 做 pair-reduction（combine），
    //        warp leader 写 shared，再 combine 各 warp。
    //      注意 combine 不是简单相加！要用上面的 combine()。
    // 4. 广播最终 (M,L) 给全 block，每个线程写：
    //      out_row[i] = expf(x_row[i] - M) / L;   for i = tid; i<D; i+=blockDim.x
    //
    // 提示：__shared__ MLState 存部分状态；__shared__ float 广播最终 M/L。

    // 每个block 处理一行;
    __shared__ MLState st[32]; // 一个block 最多32个warp?
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* cur_row = x + row * D;
    float* out_row = out + row * D;

    MLState state{-INFINITY, 0};

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        state = combine(state, {cur_row[i], 1});
    }
    for (int offset = 16; offset > 0; offset /=2) {
        float m = __shfl_down_sync(0xffffffffU, state.m, offset);
        float l = __shfl_down_sync(0xffffffffU, state.l, offset);
        state = combine(state, {m, l});
    }
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane_id == 0) {
        st[warp_id] = state;
    }

    __syncthreads();
    
    int num_warp = blockDim.x / 32;
    MLState no{-INFINITY, 0.0F};
    state = warp_id < num_warp ? st[lane_id] : no;

    if (warp_id == 0) {
        for (int offset = 16; offset > 0; offset /=2) {
            float m = __shfl_down_sync(0xffffffffU, state.m, offset);
            float l = __shfl_down_sync(0xffffffffU, state.l, offset);
            state = combine(state, {m, l});
        }
        if (lane_id == 0) {
            st[lane_id] = state;
        }
    }
    __syncthreads();

    state = st[0];
    
    for (int i = threadIdx.x ; i < D; i+=blockDim.x) {
        out_row[i] = expf(cur_row[i]-state.m)/state.l;
    
    }
}

// ===========================================================================
// CPU 参考 + 测试框架（已写好，无需改动）
// ===========================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(call) do {                                                          \
    cudaError_t e = (call);                                                     \
    if (e != cudaSuccess) {                                                     \
        std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,           \
                     cudaGetErrorString(e));                                    \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
} while (0)

// ---- CPU 参考：标准 3 遍 stable softmax，作为正确性基准 ----
static void softmax_cpu(const float* x, float* out, int rows, int D) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t)r * D;
        float* orow = out + (size_t)r * D;
        // 1. max
        float m = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < D; ++i) m = std::max(m, xr[i]);
        // 2. sum exp
        double s = 0.0;
        for (int i = 0; i < D; ++i) s += std::exp((double)xr[i] - m);
        // 3. normalize
        for (int i = 0; i < D; ++i) orow[i] = (float)(std::exp((double)xr[i] - m) / s);
    }
}

static bool run_case(int rows, int D) {
    std::vector<float> hx((size_t)rows * D), hgot((size_t)rows * D), ref((size_t)rows * D);
    // 造点数据：含较大值，检验数值稳定性（online softmax 的卖点）
    for (size_t i = 0; i < hx.size(); ++i)
        hx[i] = (float)((int)((i * 17) % 197) - 98) / 7.0f;   // 约 [-14, 14]

    softmax_cpu(hx.data(), ref.data(), rows, D);

    float *dx, *dout;
    CK(cudaMalloc(&dx, hx.size() * sizeof(float)));
    CK(cudaMalloc(&dout, hx.size() * sizeof(float)));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    online_softmax<<<rows, threads>>>(dx, dout, rows, D);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hgot.data(), dout, hx.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // 误差 + 每行和是否≈1
    double max_abs = 0.0; bool finite = true; double worst_rowsum_err = 0.0;
    for (int r = 0; r < rows; ++r) {
        double rowsum = 0.0;
        for (int i = 0; i < D; ++i) {
            float g = hgot[(size_t)r * D + i];
            finite = finite && std::isfinite(g);
            max_abs = std::max(max_abs, std::fabs((double)g - ref[(size_t)r * D + i]));
            rowsum += g;
        }
        worst_rowsum_err = std::max(worst_rowsum_err, std::fabs(rowsum - 1.0));
    }
    bool ok = finite && max_abs < 2e-5 && worst_rowsum_err < 1e-3;

    std::printf("rows=%-4d D=%-5d  max_abs=%.3e  rowsum_err=%.3e  %s\n",
                rows, D, max_abs, worst_rowsum_err, ok ? "PASS" : "FAIL");

    cudaFree(dx); cudaFree(dout);
    return ok;
}

int main() {
    bool ok = true;
    ok = run_case(1,   4)     && ok;   // 最小
    ok = run_case(3,   37)    && ok;   // 非整除
    ok = run_case(8,   1024)  && ok;   // 常见维度
    ok = run_case(128, 4096)  && ok;   // 大规模
    ok = run_case(1,   100000)&& ok;   // 超长行（D 远大于线程数）
    std::printf("%s\n", ok ? "ALL PASS" : "SOME FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
