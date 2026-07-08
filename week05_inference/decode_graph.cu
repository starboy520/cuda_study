// Week5 Day5：CUDA Graph —— 用"录制一次、重放多次"砍掉 launch 开销
//
// 背景：decode 每生成一个 token，要把整个模型跑一遍 = 几百个"短" kernel。
//   kernel 越短，CPU 每次 launch 的提交开销占比越大 → 时间线上全是 gap：
//       CPU: launch A | launch B | launch C ...   （CPU 忙着提交）
//       GPU:    A    gap   B    gap   C           （GPU 频繁空等）
//   CUDA Graph 把"一串固定的 kernel 序列"录成一张图，之后每步只发 1 次
//   cudaGraphLaunch，几百次 CPU 提交 → 1 次，gap 消失。
//
// 这是"伪 decode graph"：不跑真 Transformer/Attention/KV，只用一串代表性的
//   短 kernel 模拟"decode 一步"，重复 STEPS 步，对比：
//       plain loop（每步逐个 launch） vs graph replay（录一次、重放 STEPS 次）
//   证明 graph 砍掉的是 CPU 提交开销（不改单个 kernel 的 FLOP/bytes）。
//
// 分工：
//   - kernel 序列 + plain baseline + CPU 对拍 + 计时框架 已写好，无需改动
//   - graph 的 capture / instantiate / replay 由你实现（run_graph 里的 TODO）
//
// 编译：
//   nvcc -O3 -lineinfo -std=c++17 -arch=sm_80 decode_graph.cu -o decode_graph
//   nsys profile --trace=cuda,nvtx -o /tmp/decode_graph ./decode_graph  # 看 gap

#include <cuda_runtime.h>
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

// ===========================================================================
// 代表性的"短 kernel"序列（已写好）
//   每个 kernel 都是逐元素仿射 x = a*x + b，且 |a|<1（收缩映射）：
//   数值有界、不会爆炸/NaN，CPU/GPU 按相同顺序做相同运算 → 容易对拍。
//   目的不是算得有意义，而是模拟 decode 一步里"很多短 kernel 串起来"。
// ===========================================================================
__global__ void k_affine(float* x, int n, float a, float b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = a * x[i] + b;   // 逐元素，非常短
}

// 一步 = 固定 6 个短 kernel 的链（有数据依赖：都在同一 buffer 上顺序改）。
// ★ 关键：这个序列被 plain 和 graph capture 复用，保证两者做完全相同的工作。
//   CPU 参考 cpu_step() 必须和这里的 (a,b) 序列逐个一致！
static void launch_one_step(float* d_buf, int n, cudaStream_t s) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    k_affine<<<blocks, threads, 0, s>>>(d_buf, n, 0.5f, 0.25f);
    k_affine<<<blocks, threads, 0, s>>>(d_buf, n, 0.5f, 0.10f);
    k_affine<<<blocks, threads, 0, s>>>(d_buf, n, 0.9f, 0.01f);
    k_affine<<<blocks, threads, 0, s>>>(d_buf, n, 0.8f, 0.05f);
    k_affine<<<blocks, threads, 0, s>>>(d_buf, n, 0.7f, 0.02f);
    k_affine<<<blocks, threads, 0, s>>>(d_buf, n, 0.6f, 0.03f);
}

// CPU 参考：对每个元素，按相同顺序做相同的仿射链，重复 steps 步。
static void cpu_reference(std::vector<float>& v, int steps) {
    const float a[6] = {0.5f, 0.5f, 0.9f, 0.8f, 0.7f, 0.6f};
    const float b[6] = {0.25f, 0.10f, 0.01f, 0.05f, 0.02f, 0.03f};
    for (int step = 0; step < steps; ++step)
        for (float& x : v)
            for (int k = 0; k < 6; ++k)
                x = a[k] * x + b[k];
}

// ===========================================================================
// 计时小工具（已写好）
// ===========================================================================
struct Timer {
    cudaEvent_t beg, end;
    Timer()  { CK(cudaEventCreate(&beg)); CK(cudaEventCreate(&end)); }
    ~Timer() { cudaEventDestroy(beg); cudaEventDestroy(end); }
    void start(cudaStream_t s) { CK(cudaEventRecord(beg, s)); }
    float stop_ms(cudaStream_t s) {
        CK(cudaEventRecord(end, s));
        CK(cudaEventSynchronize(end));
        float ms = 0; CK(cudaEventElapsedTime(&ms, beg, end));
        return ms;
    }
};

// 把设备 buffer 重置成同一份初始数据（每种方法跑前调用，保证同起点）
static void reset_device(float* d_buf, const std::vector<float>& init) {
    CK(cudaMemcpy(d_buf, init.data(), init.size() * sizeof(float),
                  cudaMemcpyHostToDevice));
}

// ===========================================================================
// 方法 A：plain loop —— 每步逐个 launch（已写好，作为 baseline）
//   返回稳态总耗时（ms），结果写回 out。
// ===========================================================================
static float run_plain(float* d_buf, int n, int steps, cudaStream_t s,
                       const std::vector<float>& init, std::vector<float>& out) {
    reset_device(d_buf, init);
    // warmup：让 context/JIT/cache 就绪，不计入计时
    for (int i = 0; i < 3; ++i) launch_one_step(d_buf, n, s);
    CK(cudaStreamSynchronize(s));
    reset_device(d_buf, init);   // warmup 改了数据，重置回初始

    Timer t; t.start(s);
    for (int step = 0; step < steps; ++step)
        launch_one_step(d_buf, n, s);      // 每步 6 次 launch，共 steps*6 次提交
    float ms = t.stop_ms(s);

    out.resize(n);
    CK(cudaMemcpy(out.data(), d_buf, n * sizeof(float), cudaMemcpyDeviceToHost));
    return ms;
}

// ===========================================================================
// 【你来写】方法 B：CUDA Graph —— 录一次、重放 STEPS 次
//   返回稳态重放总耗时（ms），结果写回 out。若未实现则返回 -1（main 会跳过对比）。
//
//   生命周期：BeginCapture → 提交"一步"的 kernel 序列 → EndCapture 得 cudaGraph_t
//            → Instantiate 得 cudaGraphExec_t（只做一次）
//            → 循环 cudaGraphLaunch 重放 STEPS 次 → 用完 destroy
// ===========================================================================
static float run_graph(float* d_buf, int n, int steps, cudaStream_t stream,
                       const std::vector<float>& init, std::vector<float>& out) {
    reset_device(d_buf, init);

    cudaGraph_t     graph = nullptr;
    cudaGraphExec_t graphExec  = nullptr;

    // ---- 录制 + 实例化（只做一次，不计入稳态重放计时）----
    // TODO 1：cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal)。
    // TODO 2：调用 launch_one_step(d_buf, n, s) 提交"一步"的 6 个 kernel 到 s。
    //         注意：capture 期间只是"录制"，kernel 并不真正执行。
    // TODO 3：cudaStreamEndCapture(s, &graph) 得到 graph。
    // TODO 4：cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) 得到 exec。

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launch_one_step(d_buf, n, stream);
    cudaStreamEndCapture(stream, &graph);

    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    if (graphExec == nullptr) return -1.0f;   // 还没实现，main 会跳过 graph 对比

    // warmup 重放几次
    for (int i = 0; i < 3; ++i) {
        // TODO 5：cudaGraphLaunch(exec, s)。
        cudaGraphLaunch(graphExec, stream);
    }
    CK(cudaStreamSynchronize(stream));
    reset_device(d_buf, init);   // warmup 改了数据，重置回初始

    // ---- 稳态计时：重放 steps 次（每次只 1 次提交）----
    Timer t; t.start(stream);
    for (int step = 0; step < steps; ++step) {
        // TODO 6：cudaGraphLaunch(exec, s)。
        cudaGraphLaunch(graphExec, stream);
    }
    float ms = t.stop_ms(stream);

    out.resize(n);
    CK(cudaMemcpy(out.data(), d_buf, n * sizeof(float), cudaMemcpyDeviceToHost));

    // TODO 7：按逆生命周期销毁：cudaGraphExecDestroy(exec)、cudaGraphDestroy(graph)。
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    return ms;
}

// ===========================================================================
// 对拍工具（已写好）
// ===========================================================================
static bool compare(const char* tag, const std::vector<float>& got,
                    const std::vector<float>& ref) {
    double max_abs = 0, max_rel = 0; bool finite = true;
    for (size_t i = 0; i < got.size(); ++i) {
        finite = finite && std::isfinite(got[i]);
        double d = std::fabs((double)got[i] - ref[i]);
        max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, d / (std::fabs((double)ref[i]) + 1e-6));
    }
    bool ok = finite && max_rel < 1e-3;
    std::printf("  [%-12s] max_abs=%.3e max_rel=%.3e  %s\n",
                tag, max_abs, max_rel, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    const int n     = 4096;    // 每个 kernel 的元素数（小 → kernel 短 → launch 占比大）
    const int steps = 2000;    // 重复步数（模拟生成 2000 个 token 的循环）

    // 固定初始数据
    std::vector<float> init(n);
    for (int i = 0; i < n; ++i) init[i] = (float)((i % 17) - 8) / 8.0f;

    // CPU 参考
    std::vector<float> ref = init;
    cpu_reference(ref, steps);

    cudaStream_t s; CK(cudaStreamCreate(&s));
    float* d_buf; CK(cudaMalloc(&d_buf, n * sizeof(float)));

    std::vector<float> got_plain, got_graph;
    float ms_plain = run_plain(d_buf, n, steps, s, init, got_plain);
    float ms_graph = run_graph(d_buf, n, steps, s, init, got_graph);

    std::printf("n=%d steps=%d  (每步 6 个短 kernel)\n", n, steps);
    std::printf("plain loop : %8.3f ms  (%.3f us/step)\n",
                ms_plain, ms_plain * 1e3 / steps);

    bool ok = compare("plain vs ref", got_plain, ref);

    if (ms_graph < 0) {
        std::printf("graph      : 未实现（run_graph 的 TODO 还没填）\n");
    } else {
        std::printf("graph replay: %8.3f ms  (%.3f us/step)   speedup=%.2fx\n",
                    ms_graph, ms_graph * 1e3 / steps, ms_plain / ms_graph);
        ok = compare("graph vs ref", got_graph, ref) && ok;
    }

    CK(cudaFree(d_buf));
    CK(cudaStreamDestroy(s));
    std::printf("%s\n", ok ? "ALL PASS" : "SOME FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
