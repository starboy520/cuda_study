# 08 面试题：概念、性能分析与手写 kernel

## 0. 怎么用这一章

这是整个教材的**面试收口**——把前九卷的知识整理成面试官真正会问的形式。分三类：

```text
A. 概念题   —— 考你懂不懂原理（口头答，要简洁准确）
B. 性能分析题 —— 给场景让你定位瓶颈、提优化（考思维过程）
C. 手写 kernel —— 白板/共享屏写代码（考能不能真写出来）
```

每题给"一句话答案 + 关键补充"。**面试技巧：先给一句话结论，再按追问展开**——别一上来背一
大段。每题都标了对应卷，答不上来就回去复习。

## A. 概念题（30 题速答）

### 执行模型

**A1. warp 是什么？多少线程？**
GPU 的基本执行单位，32 个线程，同一 warp 的 lane 以 SIMT 方式同步执行同一条指令。（卷一）

**A2. SM 和 block 是硬件还是逻辑概念？**
SM 是硬件（物理执行单元）；block 是逻辑（程序员定义的线程协作单元），由调度器映射到 SM。（卷一）

**A3. 一个 block 能跨多个 SM 吗？**
不能。一个 block 整体驻留在单个 SM（它的寄存器/shared/barrier 状态都在该 SM）。（卷一）

**A4. 为什么 GPU 需要大量线程？**
延迟隐藏：某些 warp 等内存时，调度器切到其他就绪 warp 执行，让执行单元不空转。（卷一）

**A5. thread 等于 CUDA Core 吗？**
不等于。thread 是带状态的逻辑执行实例，CUDA Core 是物理 ALU；驻留 thread 远多于 core，靠
时分复用。（卷一）

**A6. SIMT 和 SIMD 的区别？**
SIMD 一条指令处理一个向量；SIMT 是多个独立线程执行同一指令，每个线程有自己的寄存器和分支
能力（可 divergence）。（卷一）

### 内存

**A7. CUDA 有哪些内存空间？**
register（thread 私有）、local（私有但在 global，慢）、shared（block 共享、片上）、global
（全局、大而慢带宽高）、constant（只读、同址广播）、texture（只读专用 cache）。（卷三/01）

**A8. local memory 在片上吗？**
不在。名字带 local 但物理在 global，因 register spill、动态索引数组、过大私有对象产生，是性能
警告。（卷三/01）

**A9. 什么是合并访问？**
一个 warp 的 32 条 lane 访问连续对齐地址，硬件用最少事务一次搬完；跨步访问则浪费大量带宽。（卷三/02）

**A10. 什么是 bank conflict？怎么解决？**
shared memory 分 32 个 bank，多条 lane 访问同一 bank 不同地址被迫串行。常用 padding（如
`[32][33]`）错开 bank 解决。（卷三/03）

**A11. AoS 和 SoA 怎么选？**
只用部分字段时用 SoA（访问连续、合并）；整体使用结构时 AoS 也可。GPU 默认优先 SoA。（卷三/02）

**A12. pinned memory 为什么快？**
页锁定不被换出，DMA 可直接搬，省掉驱动内部的暂存拷贝；且是 cudaMemcpyAsync 真异步重叠的
前提。（卷三/04）

**A13. constant memory 什么时候快？**
一个 warp 的所有 lane 读同一地址时一次广播最快；读散地址退化。适合小型只读、众线程共用的参数。（卷三/01）

### 同步与正确性

**A14. `__syncthreads` 和 `__threadfence` 区别？**
syncthreads 是 barrier：全员到齐 + 可见，会等待；threadfence 是 fence：只保证本线程写入按序
可见，不让任何线程等待。（卷四/01）

**A15. atomic 等于 barrier 吗？**
不。atomic 只保护单地址读改写不可分割，不让线程等待、不保证大家到齐。两者解决不同问题。（卷四/01,02）

**A16. 不同 block 怎么全局同步？**
普通 kernel 没有全 grid barrier。最常用是拆成多个 kernel（kernel 边界=同步墙）；或 atomic、
cooperative launch。（卷四/01）

**A17. 为什么不能依赖 warp 天然锁步？**
Volta+ 独立线程调度让 lane 可能不锁步，依赖它会在新架构间歇出错。要显式 `__syncwarp`/`_sync`
原语。（卷四/01,02）

**A18. data race 的构成条件？**
多线程访问同一地址 + 至少一个写 + 无同步，三者同时满足。危险在"时对时错"。（卷四/01）

### 编译与部署

**A19. PTX 和 SASS 区别？**
PTX 是跨架构稳定的虚拟中间码（无限虚拟寄存器）；SASS 是某代真实 GPU 的机器码（物理寄存器、
绑定架构）。PTX 经 ptxas/JIT 编成 SASS。（卷二/06）

**A20. compute_75 和 sm_75 区别？**
compute_ 是虚拟架构（决定 PTX）；sm_ 是真实架构（决定 SASS）。`-arch=sm_75` 隐含先生成
compute_75 PTX 再编 SASS。（卷二/06）

**A21. 为什么发布程序同时含 cubin 和 PTX？**
cubin 在目标卡启动快但只认那代；PTX 能在更新的卡 JIT 前向兼容。两者打进 fat binary 兼顾速度
和兼容。（卷二/06）

**A22. `no kernel image available` 怎么回事？**
编译架构和运行 GPU 不匹配，没有可用 cubin 也没有可 JIT 的 PTX。补 `-gencode` 或留 PTX。（卷十/05）

### 性能

**A23. 怎么判断 memory-bound 还是 compute-bound？**
ncu 看 Speed of Light，memory 和 compute 哪个更接近 100% 就被哪个限制；或用 Roofline 看 AI 在
拐点哪侧。（卷五/02,05）

**A24. occupancy 高一定快吗？**
不一定。occupancy 只是驻留并发 warp 比例，为隐藏延迟；若已达带宽墙或 ILP 充足，低 occupancy
也能快。（卷五/03）

**A25. arithmetic intensity 是什么？**
每搬 1 字节做多少 FLOP（FLOP/byte），是 Roofline 横轴。低则 memory-bound。（卷五/02）

**A26. warp divergence 发生在哪个范围，代价？**
warp 内部（32 lane 走不同路径）。各路径串行执行、被关 lane 空转，最坏 32 路慢约 32 倍。warp
之间不同路径无代价。（卷五/03）

**A27. register pressure 怎么影响性能？**
寄存器用多了每 SM 驻留 warp 变少（occupancy 降）；强压又会 spill 到 local 反而慢。要权衡，
`-Xptxas=-v` 看 spill。（卷五/03）

**A28. 为什么测性能要 warmup？**
首次含 context 初始化、JIT、首次内存触碰等一次性开销，会污染稳态测量。（卷五/01）

**A29. 为什么浮点 GPU 和 CPU 结果不同不一定是 bug？**
浮点不满足结合律，并行求和顺序不同导致末位差异，正常。要用容差比较而非 ==。（卷四/06）

**A30. Nsight Systems 和 Compute 区别？**
Systems 看系统级时间线（找哪段慢、是否重叠）；Compute 深挖单 kernel 的微架构指标。先系统、
后 kernel。（卷五/04,05）

## B. 性能分析题（考思维过程）

### B1. "一个 transpose kernel 很慢，怎么排查？"

```text
答题框架（展示思维过程，不要直接说答案）：
1. 先定性：transpose 计算量≈0，必然 memory-bound
2. 看访存模式：naive transpose 读连续但写跨步 -> global store 不合并（假设）
3. 上工具验证：ncu 看 gst_efficiency，若很低就证实假设
4. 优化：shared tile 把跨步写换成合并写
5. 再验证：gst_efficiency 应升到接近 100%，时间下降
6. 注意 bank conflict：tile 要加 padding
```
（卷五/06、卷三/03）

### B2. "kernel 计时很快，但整个程序很慢，下一步？"

```text
1. 这是经典的 Amdahl 信号：瓶颈不在 kernel
2. 用 nsys 看系统时间线：H2D/D2H 传输占比？GPU 有空洞吗？过度同步？
3. 常见原因：传输没和计算重叠、频繁 cudaDeviceSynchronize、launch 太碎
4. 优化：pinned + 多 stream 重叠；减少同步；考虑 CUDA Graph
```
（卷五/04、卷七）

### B3. "reduction 怎么从 naive 优化到最优？讲每步动机。"

```text
v1 global atomic：正确但全员争一个地址，串行化
v2 shared 树归约：把竞争关进 block 内
v3 每线程加 2 个：砍掉最浪费的第一轮（半数线程空闲）
v4 warp shuffle 收尾：最后 32 个值用寄存器交换，免 shared 免 barrier
跨 block：多阶段 kernel（kernel 边界做全局同步）
每步都有单一假设 + 可量化收益
```
（卷四/03、卷五/06）

## C. 手写 kernel 题（必须能默写）

### C1. Vector Add（热身）

```cpp
__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];           // 别忘边界判断
}
// launch: vecAdd<<<(n+255)/256, 256>>>(a,b,c,n);
```

### C2. Reduction（最高频，要能默写到 warp-shuffle）

```cpp
__global__ void reduce(const float* in, float* out, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;     // 每线程先加 2 个
    float sum = 0;
    if (i < n)              sum += in[i];
    if (i + blockDim.x < n) sum += in[i + blockDim.x];
    s[tid] = sum;
    __syncthreads();
    // shared 树归约，降到 32
    for (int off = blockDim.x/2; off >= 32; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }
    // warp shuffle 收尾（最后一个 warp）
    if (tid < 32) {
        float v = s[tid];
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) out[blockIdx.x] = v;
    }
}
```

### C3. Tiled Transpose（考 shared + bank conflict）

```cpp
__global__ void transpose(float* out, const float* in, int w, int h) {
    __shared__ float tile[32][33];                  // +1 消 bank conflict
    int x = blockIdx.x*32 + threadIdx.x;
    int y = blockIdx.y*32 + threadIdx.y;
    if (x < w && y < h) tile[threadIdx.y][threadIdx.x] = in[y*w + x];  // 合并读
    __syncthreads();
    int tx = blockIdx.y*32 + threadIdx.x;
    int ty = blockIdx.x*32 + threadIdx.y;
    if (tx < h && ty < w) out[ty*h + tx] = tile[threadIdx.x][threadIdx.y];  // 合并写
}
```

### C4. Tiled GEMM（手写题天花板）

```cpp
#define T 16
__global__ void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[T][T], Bs[T][T];
    int row = blockIdx.y*T + threadIdx.y;
    int col = blockIdx.x*T + threadIdx.x;
    float acc = 0;
    for (int k0 = 0; k0 < K; k0 += T) {
        // 协作加载一个 tile（含边界判断）
        As[threadIdx.y][threadIdx.x] =
            (row < M && k0+threadIdx.x < K) ? A[row*K + k0+threadIdx.x] : 0;
        Bs[threadIdx.y][threadIdx.x] =
            (k0+threadIdx.y < K && col < N) ? B[(k0+threadIdx.y)*N + col] : 0;
        __syncthreads();
        for (int k = 0; k < T; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();              // 算完再加载下一个 tile
    }
    if (row < M && col < N) C[row*N + col] = acc;
}
```

> 手写题通用提醒：**边界判断、`__syncthreads()` 位置、合并访问**是评分点。写完口头过一遍
> 这三项，比写得快更重要。

## 实践

1. C1-C4 不看答案默写，每个写完自查"边界/同步/合并"三项。
2. A 类 30 题盖住答案口头过，答错的回对应卷。
3. 找人模拟面试：随机抽 B 类题，逼自己讲出"思维过程"而非背答案。

## 资料映射

- 全教材卷一~卷七（本章是它们的面试收口）。
- 配套：各卷的"面试题"小节。
