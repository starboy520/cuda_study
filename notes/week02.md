# Week 2 学习笔记

## 内存层次表

| 内存 | 作用域 | 声明/分配 | T4 备注 |
|------|--------|-----------|---------|
| Register | 线程 | 自动 | 最快 |
| Shared | Block | `__shared__` | 48 KB/block |
| Global | Grid | `cudaMalloc` | 16 GB |

---

## Coalescing 笔记

- 一句话：一个wrap 32个线程访问的global memory 是连续且对其的， 硬件就能把他们合并成尽量少的内存事务一次搬完
- transpose naive 写回不合并原因：读是连续的， 写是跨步的

---

## transpose 带宽对比

测试条件：T4，`nvcc -O3 -arch=sm_75`，仅 kernel 时间（warmup 后单次）。
代码：`week02_memory/transpose/my_transpose.cu`。T4 理论峰值约 320 GB/s。

| 版本 | 规模 | Kernel (ms) | 带宽 GB/s | 占峰值 |
|------|------|-------------|-----------|--------|
| naive | 1024² | 0.108 | 77.5 | ~24% |
| shared_simple（无 padding） | 1024² | 0.081 | 103.9 | ~32% |
| shared（padding [32][33]） | 1024² | 0.041 | 206.4 | ~65% |
| shared（padding [32][33]） | 4096² | 0.535 | 251.1 | ~78% |

要点：
- naive → shared：合并了 global 访问（写回不再跨步），带宽 77→104。
- shared → padding：`[32][32]`→`[32][33]` 消除 32 路 bank conflict，带宽 104→206，几乎翻倍。
- 规模越大（4096²）效率越高（251 GB/s），因为固定开销被摊薄。

---

## reduction 性能

代码：`week02_memory/reduction/reduction.cu`（**自己手写**），n=1,048,576（1<<20）。

| 版本 | n | 做法 | 结果 |
|------|---|------|------|
| shared 树形 + 多阶段 | 1M | shared 树形归约 + Host 循环多阶段（1M→4096→16→1） | ✅ PASS，相对误差 0 |

验证：CPU 用 `double` 累加做参考（549,755,289,600），GPU `float` 多阶段归约，
相对误差 `< 1e-4` 判 PASS。本例输入是规整整数，误差恰好为 0。

要点（自己写时踩通的关键）：
- shared 树形：`for(stride=blockDim/2; stride>0; stride/=2)`，每轮活跃线程减半。
- sequential addressing：`if(tid < stride)`，活跃线程连续 → 避免 warp 分歧。
- 边界：`global_id < n` 时取值，否则填 0（加法单位元，不影响结果）。
- **多阶段**：block 间不能 `__syncthreads()`，靠"kernel 启动边界"当同步墙，
  Host 循环反复启动 kernel，每轮缩短约 256 倍，3 次归约到 1。
- 乒乓 buffer：`std::swap(in, out)`，上轮输出当下轮输入。

> 进阶（warp shuffle 收尾、atomic 基线对照）留到卷四，week2 不要求。

---

## 合并访问实测结论（5 句）

1. 分析访存的单位是 warp（32 线程），要把 32 个地址一起看，不能只看单线程。
2. 一个 warp 的地址连续且对齐时，硬件用最少的 32 字节事务一次搬完（合并）。
3. 地址跨步/分散时，每个线程独占一个 sector，搬运大量废数据，有效带宽暴跌。
4. transpose naive 读连续、写跨步，所以慢；shared 中转让读写 global 都连续。
5. shared 内按列访问会撞 bank conflict，`[32][33]` padding 用一列错位即可消除。

---

## 每日记录

### Day 1

**目标**：搞懂内存层次（register/shared/global）与 coalescing。

**实验**：写 transpose naive 版，理解写回为什么不合并。

**问题**：（待补：遇到的坑）

### Day 2

**目标**：用 shared memory 优化 transpose，理解两层转置。

**实验**：shared 版（tile 中转），读写 global 都合并。

**问题**：（待补）

### Day 3

**目标**：理解并消除 bank conflict。

**实验**：`[32][32]`→`[32][33]` padding，带宽从 104→206 GB/s。

**问题**：（待补）

---

## 本周总结

- **掌握**：
  - 内存层次：register（线程）/ shared（block）/ global（grid）的作用域与速度差异。
  - coalescing：warp 32 地址连续对齐才合并，跨步浪费带宽。
  - transpose 三版演进：naive → shared → padding，带宽 77→104→206 GB/s。
  - bank conflict：32 路冲突原理（bank=字地址%32）与 padding 修复。
  - 用 CUDA event 测有效带宽（2×矩阵字节 ÷ 秒）评价 memory-bound kernel。
- **薄弱**：（待补，例如：两层转置 row_out/col_out 推导、Nsight Compute 看 bank 指标）
- **下周重点**：Week 3 scan、stream 重叠
