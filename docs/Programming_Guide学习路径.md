# 以 Programming Guide 为主线的 CUDA 学习路径

> **适用人群**：有 C++ 基础、想**只用官方 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)** 作为主教材的人。
> **硬件**：Tesla T4 · SM 7.5 (Turing) · CUDA 13.x。
> **配套**：本文是「读什么 + 怎么读 + 必须自己补什么」的总纲；逐日步骤见 [Week1详细步骤.md](Week1详细步骤.md)、[Week2详细步骤.md](Week2详细步骤.md)。

---

## 〇、先认清 Guide 的定位（最重要）

| Guide 是什么 | Guide 不是什么 |
|--------------|----------------|
| **参考手册**：把概念、API、语义定义得很准 | ❌ 不是循序渐进的**教程** |
| 可以按需精确查阅某个特性 | ❌ **没有练习题**，不写代码学不会 |
| 语言无关的编程模型讲得清楚（Part 1） | ❌ **不教 profiling 工具操作**（在 Nsight 文档里） |
| 5.x 附录是权威 API 字典 | ❌ 优化"方法论"不集中（在 Best Practices Guide） |

**三条铁律**：
1. **不要从头到尾通读**。Guide 几百页，线性读必弃坑。按下文路径"点读"。
2. **每读一节，必须自己写一个对应 kernel**。Guide 没有习题，代码是你唯一的练习。
3. **看到 5.4 这种纯 API 章节，当字典查，不要"读"**。

---

## 一、能不能"只看 Guide"？——诚实评估

**能覆盖的（Guide 自给自足）**：编程模型、线程层次、内存层次、SIMT、kernel 写法、同步/原子/warp 原语、异步执行、Unified Memory、Tensor Core/WMMA、编译模型。

**Guide 覆盖不了、必须外部补的 3 个硬缺口**：

| 缺口 | 为什么 Guide 给不了 | 必备补充 |
|------|---------------------|----------|
| **Profiling 工具操作** | Guide 只讲"什么是 occupancy"，不讲"怎么用 ncu 看" | [Nsight Compute](https://docs.nvidia.com/nsight-compute/) + [Nsight Systems](https://docs.nvidia.com/nsight-systems/) 文档（只读"快速上手 + 关键指标"） |
| **优化方法论** | Guide 是定义式的，不成体系讲"怎么系统优化" | [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)（仍是官方，可视为 Guide 的姊妹篇） |
| **经典算法优化故事** | Guide 给原语，不讲 reduction/scan"一步步变快"的推导 | Mark Harris《Optimizing Parallel Reduction in CUDA》（1 篇 PDF，免费） |

> 结论：**Guide + Best Practices + Nsight 文档 + 1 篇 Harris reduction**，就是一套完全免费、几乎纯官方的闭环。其余（书、视频）都是可选加速器，不是必需。

**唯一建议的非官方补充（可选）**：一本系统教材帮你"串起来"。只挑一本：
- 《Programming Massively Parallel Processors》(PMPP, 4th, 2022) —— 当"讲解版"配合 Guide 读，不是必需。

---

## 二、总路径地图（Guide 章节 → 学习阶段）

> 标注：📖=Guide 精读 · 👀=Guide 扫读/查阅 · ✍️=必须自己写代码 · ➕=需要外部补充

```
阶段 1  入门与编程模型    Part 1 全 + 2.1 + 2.7 + 5.1
阶段 2  SIMT 与内存层次   2.3 + 5.4.4(同步) + ➕Best Practices(内存)
阶段 3  并行模式与同步    5.4.5(原子) + 5.4.6(warp) + 2.5(异步) + ➕Harris reduction
阶段 4  异步与现代内存    4.1 + 4.3 + 4.4 + 2.6
阶段 5  优化与 Tile/张量核 2.4 + 5.4.11(WMMA) + 4.10/4.11 + ➕Best Practices(全)
阶段 6  Profiling         ➕Nsight 文档 + 3.2(选读)
阶段 7  进阶专题(按需)    4.2 Graphs / 3.4 多GPU / 4.18 动态并行
```

---

## 三、分阶段详细路径

### 阶段 1：入门与编程模型（对应 Week 1）

**Guide 阅读**

| 节 | 方式 | 重点 |
|----|------|------|
| **1.1 Introduction** | 📖 | CUDA 是什么、为什么 GPU 快 |
| **1.2 Programming Model** | 📖 | **语言无关**的 Grid/Block/Thread、warp 概念——最该精读的一节 |
| **1.3 The CUDA platform** | 👀 | 平台/生态全景 |
| **2.1 Intro to CUDA C++** | 📖 | `__global__`、`<<<grid,block>>>`、`cudaMalloc/Memcpy`、线程索引公式 |
| **2.7 NVCC** | 📖 | 编译流程、`-arch=sm_75` 的由来 |
| **5.1 Compute Capabilities** | 👀 | 查 T4=7.5 的限制（每 block 线程、shared mem 等） |

**✍️ 必写代码**
1. `vec_add`：1M 元素，CPU 校验，cudaEvent 测 H2D/Kernel/D2H 三段
2. `device_query`：用 `cudaGetDeviceProperties` 打印 SM 数、warp size、shared mem
3. `mat_mul_naive`：512³/1024³ FP32，记录 GFLOPS（**这是后续优化的基线**）

**Week 1 整章跳过**：2.2 (Python)、2.4 (Tile)、2.5/2.6（后面再读）、Part 3/4

> 逐步清单已就绪：[Week1详细步骤.md](Week1详细步骤.md)（12 Step）

---

### 阶段 2：SIMT 执行与内存层次（对应 Week 2）

**Guide 阅读**

| 节 | 方式 | 重点 |
|----|------|------|
| **2.3 Writing SIMT Kernels** | 📖 | 本阶段主战场：SIMT、warp divergence、global/shared memory、**合并访问**、转置示例 |
| **5.4.4 Synchronization Primitives** | 📖 | `__syncthreads`、`__syncwarp`、memory fence 的精确语义 |
| **5.7 CUDA C++ Memory Model** | 👀 | 弱内存模型概念（先建立印象，阶段 3 回头看） |

**➕ 必备补充**
- [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) → **Memory Optimizations** 一章：合并访问、对齐、shared memory bank conflict 讲得比 Guide 成体系。

**✍️ 必写代码**
1. `transpose`：naive → coalesced → shared+padding 三版，对比带宽 GB/s
2. `reduction`：1M→1 求和，至少做到 shared memory 树形归约
3. 故意制造非合并访问，**用 ncu 看 `dram__throughput` 差异**（profiling 在阶段 6 系统学，这里先会跑一条命令）

> 逐步清单：[Week2详细步骤.md](Week2详细步骤.md)

---

### 阶段 3：并行模式、原子与 warp 级编程（对应 Week 3）

**Guide 阅读**

| 节 | 方式 | 重点 |
|----|------|------|
| **5.4.5 Atomic Functions** | 📖 | `atomicAdd` 等；适用场景与性能代价 |
| **5.4.6 Warp Functions** | 📖 | `__shfl_down_sync`、vote、reduce——warp 级 reduction 的基础 |
| **2.5 Asynchronous Execution** | 📖 | Stream / Event、`cudaMemcpyAsync`、拷贝与计算重叠 |
| **5.7 CUDA C++ Memory Model** | 📖 | 这次精读：原子的 memory order、thread scope |

**➕ 必备补充**
- Mark Harris《Optimizing Parallel Reduction in CUDA》全文——Guide 不会教你 reduction 怎么一步步从慢变快。

**✍️ 必写代码**
1. `scan`（prefix sum）：inclusive/exclusive，CPU 校验
2. `warp_reduce`：用 `__shfl_down_sync` 实现，对比 shared memory 版
3. `stream_overlap`：H2D+Kernel+D2H 流水线，比串行快 ≥15%

---

### 阶段 4：异步与现代内存模型（对应 Week 3–4 之间）

**Guide 阅读**

| 节 | 方式 | 重点 |
|----|------|------|
| **4.1 Unified Memory** | 📖 | `cudaMallocManaged`、page fault、T4 上的代价（了解，生产慎用） |
| **4.3 Stream-Ordered Memory Allocator** | 📖 | `cudaMallocAsync`——现代显存管理 |
| **4.4 Cooperative Groups** | 📖 | 现代 warp/block 同步抽象，替代裸 `__shfl`/`__syncthreads` 的推荐写法 |
| **2.6 Unified and System Memory** | 👀 | 入门版概念（和 4.1 互补） |

**✍️ 必写代码**
- 把阶段 3 的 `warp_reduce` 用 Cooperative Groups 重写一版，对比可读性与性能

---

### 阶段 5：性能优化、Tile 与 Tensor Core（对应 Week 5，就业核心）

**Guide 阅读**

| 节 | 方式 | 重点 |
|----|------|------|
| **2.3 Kernel Launch and Occupancy** | 📖 | 回头精读 occupancy、寄存器压力、block size 调参 |
| **2.4 Writing Tile Kernels** | 📖 | 新版 tile 编程模型（`__tile__` / `cuda::tiles`） |
| **5.4.11 Warp Matrix Functions** | 📖 | WMMA / Tensor Core（T4 支持 FP16） |
| **4.11 Asynchronous Data Copies** | 📖 | `cp.async` / `memcpy_async`——现代分块 GEMM 双缓冲基础 |
| **4.10 Pipelines** | 👀 | 流水化异步拷贝 |

**➕ 必备补充**
- [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) **全文精读**——这是优化方法论的核心，Guide 替代不了。
- Roofline 模型（Williams et al. 论文，理解算术强度 vs 带宽/算力上限）。

**✍️ 必写代码**
1. `gemm_tiled`：shared memory 分块，目标 ≥ naive 的 5×
2. `occupancy_sweep`：block 32–512 扫描，画 GFLOPS 曲线
3. 一条 WMMA 样例：FP16 Tensor Core 路径，对比 FP32

> T4 现实预期：手写 GEMM 追不上 cuBLAS，目标是**理解差距来源**，不是 beat cuBLAS。

---

### 阶段 6：Profiling（对应 Week 6）

> **这是 Guide 的最大盲区**——Guide 讲指标定义，不讲工具操作，必须看 Nsight 文档。

**➕ 必备补充（主线）**

| 文档 | 只读什么 |
|------|----------|
| [Nsight Systems](https://docs.nvidia.com/nsight-systems/) | Quick Start + 怎么看 timeline 找 CPU-GPU 重叠 |
| [Nsight Compute](https://docs.nvidia.com/nsight-compute/) | Quick Start + 关键指标：`sm__throughput`、`dram__throughput`、`achieved_occupancy` |

**Guide 选读**
- **3.2 Advanced Kernel Programming** 👀：SIMT 调度、active mask 等底层细节

**✍️ 必做**
1. 对阶段 5 的 GEMM 出 profiling 结论：memory-bound 还是 compute-bound？
2. 建一个 benchmark harness：warmup + 多次取中位数

---

### 阶段 7：进阶专题（按目标岗位选，Week 7–8）

Guide Part 4 有 20 节，**别全读**，按方向挑：

| 方向 | 读 Guide 哪节 |
|------|---------------|
| 推理 / 降低 launch 开销 | **4.2 CUDA Graphs** |
| 多卡 / HPC | **3.4 Programming Systems with Multiple GPUs** |
| 不规则并行 | **4.18 CUDA Dynamic Parallelism** |
| 缓存调优 | **4.13 L2 Cache Control** |

其余（4.6 Green Contexts、4.15 IPC、4.16 Virtual Memory 等）属于"用到再查",不在学习路径里。

---

## 四、5.x 附录怎么用（当字典，不要读）

| 附录 | 什么时候查 |
|------|-----------|
| **5.1 Compute Capabilities** | 想知道 T4 支持什么、限制多少 |
| **5.3 C++ Language Support** | 写到某个 C++ 特性不确定 device 端能不能用 |
| **5.4 C/C++ Language Extensions** | 查 `__syncthreads`/原子/shuffle/WMMA 的精确签名（5.4.4/5.4.5/5.4.6/5.4.11） |
| **5.5 Floating-Point Computation** | 遇到浮点精度/结果对不上 CPU |
| **5.7 / 5.8 Memory & Execution Model** | 深究同步语义、数据竞争 |

---

## 五、一句话总结

- **主线**：Programming Guide，按上面 7 阶段**点读**，不通读。
- **必补 3 样**：Best Practices Guide（优化）、Nsight 文档（profiling）、Harris reduction（算法故事）——前两样仍是官方。
- **每节都要写代码**，Guide 没习题，代码是唯一练习。
- **可选**：PMPP 第四版当讲解版串讲。

> 你现在在**阶段 1（Week 1）**，照 [Week1详细步骤.md](Week1详细步骤.md) 走即可。

---

**返回**：[CUDA学习路线图.md](CUDA学习路线图.md) · [学习资料索引.md](学习资料索引.md)
