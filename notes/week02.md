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

代码：`cuda_deep_course/labs/04_parallel_algorithms/reduction`，n=1,000,000，均 PASS。

| 版本 | n | Kernel (ms) | PASS |
|------|---|-------------|------|
| shared 树形 | 1M | 0.113 | ✅ |
| warp shuffle 收尾 | 1M | 0.060 | ✅ |
| atomic（可选） | 1M | 未实现（全局 atomic 串行化，仅作对照概念） | — |

要点：warp shuffle 收尾把最后 32 个值的归约从 shared+barrier 换成寄存器交换，
比纯 shared 树形快约 2 倍。

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
