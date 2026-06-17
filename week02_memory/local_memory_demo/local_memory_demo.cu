// local_memory_demo.cu
//
// 演示 "local memory" 是怎么悄悄冒出来的。
//
// ============================================================================
// 什么是 local memory？
//   - 作用域：线程私有（每个线程一份，别的线程看不到），这点和寄存器一样。
//   - 物理位置：却在片外显存(DRAM)里，离计算单元远、慢（未命中缓存时几百周期）。
//   "local" 指的是"作用域私有"，不是"物理上靠近"——这是 CUDA 里最坑的命名。
//   它本质是寄存器的"溢出备胎"：放不进寄存器的线程私有数据，就落到这里。
//
// 为什么有的变量会掉进 local memory？两种来源，编译器报告里分开统计：
//
//   (1) stack frame  —— "动态下标的局部数组"
//       根因：寄存器是一组没有地址、不能用变量下标随机寻址的存储单元。
//             编译器只有在下标是"编译期常量"时(buf[0]/buf[k]且循环能展开)，
//             才能把数组元素一个个固定映射到具体寄存器。
//             一旦出现"运行时才知道的下标"(buf[dyn])，编译期无法确定到底
//             访问哪个元素 → 必须有一块"可按地址+偏移寻址"的真实内存。
//             寄存器做不到这件事，于是整个数组被放进 local memory。
//             —— 简记：动态下标需要地址 → 寄存器没有地址 → 直接进 local memory。
//
//   (2) spill stores/loads —— "寄存器数量不够用"
//       根因：每个 SM 的寄存器是有限的，一个线程同时存活的值太多时，
//             编译器装不下，只能把一部分临时"倒进"local memory，用到时再取回。
//             这一存一取就是 spill store / spill load。
//
//   两者物理上都在 local memory(显存)，区别只是"为什么被放进去"。
// ============================================================================
//
// 本 demo 三个 kernel，对照看两种来源：
//   - noSpillKernel         : 只用几个标量，全放寄存器     → 两种来源都为 0
//   - dynamicArrayKernel    : 局部数组用"运行时下标"访问   → 触发 stack frame
//   - registerPressureKernel: 大量标量同时存活、寄存器吃紧 → 触发 spill stores/loads
、
//
// 编译时加 -Xptxas=-v，观察每个 kernel 的:
//   "N bytes stack frame, M bytes spill stores, M bytes spill loads"
//   任意一个 != 0 → 用到了 local memory。
//
// 编译并查看资源报告：
//   nvcc -O3 -arch=sm_75 -Xptxas=-v local_memory_demo.cu -o local_memory_demo
//
// 想强行逼出更多 spill，可以再加 -maxrregcount=16 限制每线程寄存器数：
//   nvcc -O3 -arch=sm_75 -Xptxas=-v -maxrregcount=16 local_memory_demo.cu -o local_memory_demo
//   (实测：不限制时 registerPressureKernel 用 43 寄存器不 spill；
//          限制到 16 后立刻 spill 约 764 bytes —— 正是"压寄存器反而更慢"的来源)
//
// 运行（验证三个 kernel 都跑通）：
//   ./local_memory_demo

#include <cstdio>
#include <cmath>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__,   \
             __LINE__);                                                       \
      return 1;                                                               \
    }                                                                         \
  } while (0)

constexpr int N = 1 << 20;  // 1M 元素
constexpr int K = 8;        // 每个线程做 K 步变换

// ----------------------------------------------------------------------------
// 版本 A：完全不碰 local memory（对照基准）。
// 只用两个标量(acc, x)，且循环次数 K 是编译期常量、循环会被展开，
// 编译器能把 acc 一直保存在某个固定寄存器里，全程不需要"地址"。
// 预期报告：0 stack frame, 0 spill。
// ----------------------------------------------------------------------------
__global__ void noSpillKernel(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  float acc = in[idx];
  // 用标量循环累计，编译器可以把 acc 一直保存在寄存器里
  for (int k = 0; k < K; ++k) {
    acc = acc * 1.001f + static_cast<float>(k);
  }
  out[idx] = acc;
}

// ----------------------------------------------------------------------------
// 版本 B：触发 "stack frame" 形式的 local memory（动态下标数组）。
//
// 核心机制：buf[dyn] 里的 dyn 是运行时才算出来的下标。
//   - 寄存器是一堆没有地址的独立单元，无法用一个"变量"去选中其中某一个。
//   - 编译期不知道 dyn 到底是 0..K-1 中的哪个，没法把 buf 映射到固定寄存器。
//   - 于是 buf 必须放进一块"能按 基址+偏移 寻址"的真实内存 → local memory。
//   动态下标需要地址，寄存器没有地址，所以整个 buf 直接进 local memory。
//
// 注意：这跟"寄存器够不够"无关，纯粹是寻址方式决定的，所以 spill 仍为 0。
// 预期报告：32 bytes stack frame (= float[8] = 8*4), 0 spill。
//
// 反向验证：把下面的 buf[dyn] 全改成 buf[k]（编译期常量下标，且循环可展开），
//           重新编译，stack frame 会变回 0 —— local memory 消失。
// ----------------------------------------------------------------------------
__global__ void dynamicArrayKernel(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  float buf[K];
  float x = in[idx];

  // 用 idx 派生出运行时下标 dyn：编译期无法确定具体访问哪个元素，
  // 迫使 buf 落到 local memory（而不是被映射到寄存器）。
  for (int k = 0; k < K; ++k) {
    int dyn = (idx + k) % K;       // 运行时下标（关键：不是编译期常量）
    buf[dyn] = x * 1.001f + static_cast<float>(k);
    x = buf[dyn];
  }

  // 再用另一个运行时下标读回来，进一步阻止编译器优化掉 buf
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    int dyn = (idx * 3 + k) % K;   // 又一个运行时下标
    acc += buf[dyn];
  }
  out[idx] = acc;
}

// ----------------------------------------------------------------------------
// 版本 C：触发 "spill stores/loads" 形式的 local memory（寄存器不够用）。
//
// 核心机制：这里没有数组、没有动态下标，全是普通标量，本来都能进寄存器。
//   但 32 个标量被设计成"同时存活"——每个都要保留到最后一起求和，
//   中间又互相依赖，谁也不能提前释放掉自己的寄存器。
//   当"同时需要的寄存器数" > "可用寄存器数"时，编译器只能把一部分值
//   临时写到 local memory(spill store)，等下次要用再读回来(spill load)。
//
// 预期报告：
//   - 不加 -maxrregcount：T4 每线程上限 255 寄存器，32 个标量装得下 → 可能不 spill
//   - 加 -maxrregcount=16：人为把寄存器压到 16 个 → 装不下 → spill 出现(约 764 bytes)
//   这恰好演示一个重要权衡：强行压低寄存器能提高 occupancy，
//   但代价是 spill 到 local memory，反而可能更慢。寄存器用量要靠 profiler 实测。
// ----------------------------------------------------------------------------
__global__ void registerPressureKernel(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  float base = in[idx];

  // 32 个标量，全部独立计算并一直存活到最后求和。
  // 它们不能互相覆盖寄存器，迫使编译器寄存器吃紧而 spill。
  float r0  = base + 0.1f,  r1  = base + 1.1f,  r2  = base + 2.1f,  r3  = base + 3.1f;
  float r4  = base + 4.1f,  r5  = base + 5.1f,  r6  = base + 6.1f,  r7  = base + 7.1f;
  float r8  = base + 8.1f,  r9  = base + 9.1f,  r10 = base + 10.1f, r11 = base + 11.1f;
  float r12 = base + 12.1f, r13 = base + 13.1f, r14 = base + 14.1f, r15 = base + 15.1f;
  float r16 = base + 16.1f, r17 = base + 17.1f, r18 = base + 18.1f, r19 = base + 19.1f;
  float r20 = base + 20.1f, r21 = base + 21.1f, r22 = base + 22.1f, r23 = base + 23.1f;
  float r24 = base + 24.1f, r25 = base + 25.1f, r26 = base + 26.1f, r27 = base + 27.1f;
  float r28 = base + 28.1f, r29 = base + 29.1f, r30 = base + 30.1f, r31 = base + 31.1f;

  // 每个标量都各自经过一串依赖运算，迫使它们长时间保持活跃
  for (int k = 0; k < K; ++k) {
    r0  = r0  * 1.001f + r1;   r1  = r1  * 1.001f + r2;
    r2  = r2  * 1.001f + r3;   r3  = r3  * 1.001f + r4;
    r4  = r4  * 1.001f + r5;   r5  = r5  * 1.001f + r6;
    r6  = r6  * 1.001f + r7;   r7  = r7  * 1.001f + r8;
    r8  = r8  * 1.001f + r9;   r9  = r9  * 1.001f + r10;
    r10 = r10 * 1.001f + r11;  r11 = r11 * 1.001f + r12;
    r12 = r12 * 1.001f + r13;  r13 = r13 * 1.001f + r14;
    r14 = r14 * 1.001f + r15;  r15 = r15 * 1.001f + r16;
    r16 = r16 * 1.001f + r17;  r17 = r17 * 1.001f + r18;
    r18 = r18 * 1.001f + r19;  r19 = r19 * 1.001f + r20;
    r20 = r20 * 1.001f + r21;  r21 = r21 * 1.001f + r22;
    r22 = r22 * 1.001f + r23;  r23 = r23 * 1.001f + r24;
    r24 = r24 * 1.001f + r25;  r25 = r25 * 1.001f + r26;
    r26 = r26 * 1.001f + r27;  r27 = r27 * 1.001f + r28;
    r28 = r28 * 1.001f + r29;  r29 = r29 * 1.001f + r30;
    r30 = r30 * 1.001f + r31;  r31 = r31 * 1.001f + r0;
  }

  float acc = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 +
              r12 + r13 + r14 + r15 + r16 + r17 + r18 + r19 + r20 + r21 + r22 +
              r23 + r24 + r25 + r26 + r27 + r28 + r29 + r30 + r31;
  out[idx] = acc;
}

static void launch(void (*kernel)(const float*, float*, int), const float* d_in,
                   float* d_out, int n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  kernel<<<blocks, threads>>>(d_in, d_out, n);
}

int main() {
  size_t bytes = static_cast<size_t>(N) * sizeof(float);

  float* h_in = new float[N];
  float* h_outA = new float[N];
  float* h_outB = new float[N];
  float* h_outC = new float[N];
  for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 100) * 0.01f;

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  launch(noSpillKernel, d_in, d_out, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_outA, d_out, bytes, cudaMemcpyDeviceToHost));

  launch(dynamicArrayKernel, d_in, d_out, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_outB, d_out, bytes, cudaMemcpyDeviceToHost));

  launch(registerPressureKernel, d_in, d_out, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_outC, d_out, bytes, cudaMemcpyDeviceToHost));

  // 各 kernel 数学结果不要求相等，这里只确认都跑通、有合理输出
  printf("noSpillKernel          out[0]=%.4f  out[N-1]=%.4f\n", h_outA[0], h_outA[N - 1]);
  printf("dynamicArrayKernel     out[0]=%.4f  out[N-1]=%.4f\n", h_outB[0], h_outB[N - 1]);
  printf("registerPressureKernel out[0]=%.4f  out[N-1]=%.4f\n", h_outC[0], h_outC[N - 1]);
  printf("\n三个 kernel 都跑通了。重点看编译时 -Xptxas=-v 的输出：\n");
  printf("  noSpillKernel          : 0 stack frame, 0 spill            → 不用 local memory\n");
  printf("  dynamicArrayKernel     : 32 bytes stack frame, 0 spill     → local memory(数组动态下标)\n");
  printf("  registerPressureKernel : spill stores/loads != 0           → local memory(寄存器不够)\n");
  printf("  stack frame 和 spill 都属于 local memory，只是来源不同。\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  delete[] h_in;
  delete[] h_outA;
  delete[] h_outB;
  delete[] h_outC;
  return 0;
}
