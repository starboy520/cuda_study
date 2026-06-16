# Week 2 详细步骤清单

> **主题**：内存层次、合并访问（Coalescing）、Shared Memory、Bank Conflict  
> **预计用时**：5–7 天（每天 3–4 小时业余 / 2–3 天全职）  
> **本周交付**：`week02_memory/`（transpose 三版 + reduction 至少 2 版）+ `notes/week02.md`（带宽/GFLOPS 对比表 + 合并访问结论）

**使用方式**：按 Step 01 → 12 顺序做；每步有「阅读」「动手」「完成标志」；做完一步再勾 `[ ]`。

> **前置**：Week 1 完成 `vec_add`、`device_query`、`mat_mul_naive`（1024³ GFLOPS 基线已记录）。  
> **总纲**：本周对应 [Programming_Guide学习路径.md](Programming_Guide学习路径.md) 的 **阶段 2：SIMT 与内存层次**。  
> **官方文档**： [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)  
> **Week 1 对照**：[Week1详细步骤.md](Week1详细步骤.md)

> **阅读标注**（贯穿全文）：📖 精读 · 👀 扫读/查阅 · ✍️ 必须自己写代码 · ➕ 需外部补充（Guide 覆盖不了）。

---

## Week 2 学习路径（对齐官方最新结构）

**原则**：Week 2 主攻 **2.3** 里 Global/Shared Memory、Coalescing、Transpose；同步原语读 **5.4.4**；弱内存模型概念扫读 **5.7**（建立印象，Week 3 回头深读）；**不读 2.4 Tile Kernels**（Week 5）。

> **为什么本周靠 Guide 不够，要配 2 个补充**（出自总纲的“Guide 是参考手册不是教程”）：
> - ➕ **[Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) → 第 10 章 Memory Optimizations**（重点 **10.2.1** Coalesced Access、**10.2.3** Shared Memory）：合并访问/对齐/bank conflict 的**优化方法论**，Guide 只给定义不成体系。**10.2.3.3（C=AAT）就是 transpose 的官方优化版**（naive 12.8 → shared 140 → padding 199 GB/s）。
> - ➕ **Mark Harris《Optimizing Parallel Reduction》**：reduction “一步步变快”的推导，Guide 只给 `atomicAdd` 原语。
> - ✍️ **Guide 没有习题**：本周每个概念都要落到 transpose / reduction 代码上，代码是唯一练习。

### 7 天安排（业余节奏）

> **每日可勾选清单**：[Week2_Day1-Day7学习清单.md](Week2_Day1-Day7学习清单.md)（含每天学什么、看什么、动手命令、完成标准）

| Day | Step | 官方文档 | 代码/笔记 |
|-----|------|------------------|-----------|
| 1 | 01–02 | 📖 **2.3** Global Memory + Memory Coalescing；➕ Best Practices **10.2.1** Coalesced Access | 内存层次笔记、合并访问概念 |
| 2 | 03 | 📖 **2.3** Shared Memory + **5.4.4** Synchronization（`__syncthreads`）；👀 **5.7** Memory Model | 读懂 `vec_add` 里 `reduce_sum` |
| 3 | 04–05 | 📖 **2.3** Matrix Transpose 示例 | `transpose` naive + shared_simple PASS |
| 4 | 06 | 📖 **2.3** Bank Conflict；本地 [转置优化详解.md](../week02_memory/transpose/转置优化详解.md) | `transpose_shared` + GB/s 三版对比 |
| 5 | 07–08 | ➕ Harris «Optimizing Parallel Reduction» 前半 | `reduction` atomic 基线 + shared 树形归约 |
| 6 | 09 | 同上 + 📖 **2.3** 边界/正确性 | 多 block reduction 汇总 PASS |
| 7 | 10–12 | ➕ Best Practices 第 10 章（尤其 10.2.3.3 C=AAT）复盘 | 性能表、`notes/week02.md` 定稿 |

### 2.3 在 Week 2 读什么 / 不读什么

| Week 2 要读 | Week 2 跳过（以后读） |
|-------------|----------------------|
| Global / Shared / Local Memory 概念 | **2.4 Writing Tile Kernels**（Week 5） |
| Memory Coalescing | Occupancy 深入调优（Week 5） |
| Matrix Transpose（Global + Shared 版） | **5.4.5 Atomics** 系统学习（Week 3） |
| Bank Conflict、padding | **5.4.6 Warp shuffle**（Week 3） |
| **5.4.4** `__syncthreads` / barrier | Unified Memory（Week 3） |
| **5.7** 弱内存模型（👀 扫读建印象） | **5.7** memory order / fence 深入（Week 3） |
| Reduction 思想（Harris 文章） | Nsight Compute 系统使用（Week 6） |

### 你已有的进度（可跳过/快速验收）

| 项目 | 状态 | 对应 Step |
|------|------|-----------|
| `week02_memory/transpose/transpose.cu` 三版 | ✅ 已 PASS | Step 04–06 验收 |
| `my_transpose.cu` 手写 | ✅ 逻辑正确 | Step 06 加分 |
| [转置优化详解.md](../week02_memory/transpose/转置优化详解.md) | ✅ 已有 | Step 06 阅读 |
| `vec_add.cu` 里 `reduce_sum` | 🔄 预习版 | Step 03 对照，Step 08 迁到独立目录 |

---

## 总览

| Step | 主题 | 预计时间 | 产出 |
|------|------|----------|------|
| 01 | 内存层次总览 | 2 h | 6 种内存对比表 |
| 02 | 合并访问 Coalescing | 2.5 h | 能口述 warp 合并条件 |
| 03 | Shared Memory 入门 | 3 h | 读懂 `__shared__` + `__syncthreads` |
| 04 | transpose naive | 2 h | 带宽基线 GB/s |
| 05 | transpose shared_simple | 2 h | 理解 shared 中转 |
| 06 | transpose_shared 优化 | 3 h | 三版带宽对比 + padding |
| 07 | reduction 基线 | 2 h | atomic / 单线程对比 |
| 08 | shared 树形归约 | 3 h | 1 block 内 reduction PASS |
| 09 | 多 block reduction | 3 h | 1M 元素 sum PASS |
| 10 | 性能测试与记录 | 2 h | 对比表填满 |
| 11 | 分析工具初探（选做） | 2 h | nsys 或 sanitizer 截图 |
| 12 | 周复盘与交付 | 2 h | `notes/week02.md` 定稿 |

---

## Step 01：内存层次总览

### 阅读

| 资料 | 章节（v13.x） | 重点 |
|------|---------------|------|
| Programming Guide | **2.3** 页内搜 `Global Memory` / `Shared Memory` | 六种内存名字 |
| [Programming_Model详解.md](Programming_Model详解.md) | Memory 相关小节 | 与 Week 1 衔接 |
| [GPU架构图资源.md](GPU架构图资源.md) | Memory Hierarchy | 可视化 |
| [T4实战指南.md](T4实战指南.md) | 硬件速览 | 48KB shared/block |

### 核心概念（读完能填表）

| 内存类型 | 作用域 | 速度 | T4 容量级 | 声明/分配 |
|----------|--------|------|-----------|-----------|
| **Register** | 线程私有 | 最快 | 有限 | 编译器自动 |
| **Local** | 线程私有 | 慢（常 spill） | — | 局部变量过多 |
| **Shared** | Block 内共享 | 很快 | 48 KB/block | `__shared__` |
| **Global** | 全 Grid | 慢、带宽大 | 16 GB | `cudaMalloc` |
| **Constant** | 全 Grid 只读 | 有 cache | 64 KB | `__constant__` |
| **Texture** | 特殊只读 | 有 cache | — | Week 7 选读 |

### 动手

在 `notes/week02.md` 新建文件，画一张简图：

```text
Thread → Register / Local
Block  → Shared Memory（SM 上）
Grid   → Global Memory（显存）
```

### 完成标志

- [ ] 能区分 Global vs Shared vs Register
- [ ] 知道 T4：`sharedMemPerBlock = 48 KB`
- [ ] 理解：Week 2 优化主要动 **Global 访问模式** 和 **Shared**

### 交付物

`notes/week02.md` 第一节「内存层次表」

---

## Step 02：合并访问（Memory Coalescing）

### 阅读

| 资料 | 章节 | 重点 |
|------|------|------|
| 📖 Programming Guide | **2.3** 页内搜 `coalesced` / `Coalescing` | 连续线程 → 连续地址 |
| ➕ Best Practices Guide | **10.2.1 Coalesced Access to Global Memory**（含 10.2.1.1–10.2.1.4） | Simple / Misaligned / Strided 访问模式，看 transpose naive 为何不合并 |

### 核心理解

```text
一个 warp = 32 线程（T4）

理想：thread 0,1,2,...,31 访问地址 n, n+4, n+8, ...（float 连续）
     → GPU 合并成少量内存事务 → 高带宽

糟糕：32 线程访问地址间隔很大（如 stride = height）
     → 32 次独立事务 → 低带宽
```

与 Week 1 的联系：

| Kernel | 读 | 写 | Week 2 要理解的 |
|--------|----|----|-----------------|
| `mat_mul_naive` | 部分合并 | 部分不合并 | 为何 naive GEMM 慢 |
| `transpose_naive` | 合并 | **不合并** | 转置经典反例 |

### 动手

**不动代码**，对照 `week02_memory/transpose/transpose.cu` 里 `transpose_naive` 回答：

1. 读 `in` 时，同一 warp 地址连续吗？
2. 写 `out` 时，同一 warp 地址间隔多少？

答案写入 `notes/week02.md`。

### 完成标志

- [ ] 用一句话解释 Coalescing
- [ ] 能指出 `transpose_naive` 写回为何不合并

### 交付物

笔记 3–5 句 + 示意图（可手绘拍照或 ASCII）

---

## Step 03：Shared Memory 与 `__syncthreads`

### 阅读

| 资料 | 章节 | 重点 |
|------|------|------|
| 📖 Programming Guide | **2.3** Shared Memory | tile 中转思想 |
| 📖 Programming Guide | **5.4.4** Synchronization | `__syncthreads()` 语义 |
| 👀 Programming Guide | **5.7** CUDA C++ Memory Model | 弱内存模型概念（先建立印象，Week 3 原子部分回头精读） |
| 本地 | `week01_basics/vec_add/vec_add.cu` → `reduce_sum` | 已有预习代码 |

### 核心理解

```cuda
__shared__ float tile[TILE][TILE + 1];  // block 内所有线程共享

tile[ty][tx] = in[...];   // 各线程写入不同位置
__syncthreads();            // 必须：等全员完成再读
out[...] = tile[...];
```

**`__syncthreads()` 规则（Week 2 必记）**：

- 只同步 **同一个 block** 内线程
- block 内所有线程都必须到达，否则 **死锁**
- 不能放在 `if (threadIdx.x < n)` 分支里导致 warp 不一致（进阶坑，Step 08 再遇）

### 动手

1. 阅读 `vec_add.cu` 中 `reduce_sum`（约 44–65 行）
2. 回答：`stride /= 2` 在做什么？为何每轮要 `__syncthreads()`？

### 完成标志

- [ ] 能解释 `__shared__` 与 Global 的区别
- [ ] 能口述树形归约一轮在做什么

### 交付物

`notes/week02.md`「Shared Memory + syncthreads」小节

---

## Step 04：transpose — naive 版与带宽基线

### 阅读

| 资料 | 章节 | 重点 |
|------|------|------|
| Programming Guide | **2.3** Matrix Transpose（Global 版） | 2D 映射 |
| 本地 | [transpose/README.md](../week02_memory/transpose/README.md) | 统一 x→col, y→row |

### 动手

```bash
cd week02_memory/transpose
make && ./transpose 512
make && ./transpose 1024
make && ./transpose 4096
```

**带宽公式**（转置是 memory-bound）：

```text
有效带宽 GB/s = 2 × width × height × sizeof(float) / kernel_time_s / 1e9
                ↑ 读 + 写 各一遍
```

在 `transpose.cu` 的 `run_kernel` 中加 `cudaEvent` **只计 kernel**，或单独写计时（参考 `mat_mul.cu`）。

### 完成标志

- [ ] `transpose_naive` 512² / 1024² PASS
- [ ] 记录 naive 版 1024² 或 4096² kernel 带宽（GB/s）

### 交付物

`notes/week02.md` transpose 表第一行（naive）

---

## Step 05：transpose — shared_simple

### 阅读

| 资料 | 重点 |
|------|------|
| [转置优化详解.md](../week02_memory/transpose/转置优化详解.md) § 四 | shared_simple 版本 |
| 本地 `transpose.cu` 42–60 行 | 同线程读/写 |

### 动手

运行并确认 PASS：

```bash
./transpose 1024 768   # 非方阵
```

对比 naive vs shared_simple 带宽（shared_simple 写回仍不合并，可能差不多或略差）。

### 完成标志

- [ ] 理解 shared 作为「中转站」
- [ ] 能解释为何 shared_simple **读合并、写仍不合并**

### 交付物

笔记：shared_simple 与 naive 的差异（正确性相同，性能差异）

---

## Step 06：transpose_shared + Bank Conflict

### 阅读

| 资料 | 重点 |
|------|------|
| [转置优化详解.md](../week02_memory/transpose/转置优化详解.md) § 五–§ 六 | 写回 x/y 对调、`tile[TILE+1]` |
| 📖 Programming Guide | **2.3** Bank Conflict | 32 banks, 4B width |
| ➕ Best Practices Guide | **10.2.3.3 Shared Memory in Matrix Multiplication (C=AAT)** | **transpose 官方版**：naive 12.8 → shared 140 → padding `[TILE+1]` 199 GB/s，与你三版数字直接对比 |

### 核心理解

```text
Bank Conflict：同一 warp 内多线程访问 shared 同一 bank → 串行化

transpose 写阶段读 tile[tx][ty]：
  若 stride=32 → 32 路 conflict
  修复：tile[TILE][TILE+1]  padding +1
```

### 动手

1. 验收 `./transpose` 三版全 PASS
2. 填三版带宽表（1024² 或 4096²）：

| 版本 | Kernel (ms) | 带宽 GB/s | 读合并 | 写合并 |
|------|-------------|-----------|--------|--------|
| naive | | | ✓ | ✗ |
| shared_simple | | | ✓ | ✗ |
| shared | | | ✓ | ✓ |

3. （可选）对照 `my_transpose.cu` 与参考版差异

### 完成标志

- [ ] 三版 PASS + 带宽表有数据
- [ ] 能解释 `col_out = blockIdx.y * 32 + threadIdx.x`（不要求背，能讲清意图）
- [ ] 知道 `[TILE+1]` 是为了避免 bank conflict

### 交付物

更新 `transpose/README.md` 或 `notes/week02.md` 性能表

---

## Step 07：reduction — 基线与 atomic 对比

### 阅读

| 资料 | 重点 |
|------|------|
| Mark Harris «Optimizing Parallel Reduction in CUDA» | Kernel 1–2（interleaved / sequential addressing 先浏览） |
| Programming Guide | **5.4.5** `atomicAdd`（仅作对比，不深究） |

### 动手

新建 `week02_memory/reduction/`：

```
week02_memory/reduction/
├── reduction.cu
├── Makefile
└── README.md
```

**版本 A**：CPU sum 参考  
**版本 B**（慢速基线）：单线程 kernel 或 atomic 逐元素加（演示即可，不必优化）

```cuda
// 慢速对比示例：atomicAdd 到 global
atomicAdd(&output[0], input[i]);
```

### 完成标志

- [ ] 目录创建，能编译
- [ ] 1M 元素，CPU 参考值正确
- [ ] 知道 atomic 为何慢（串行化、争用）

### 交付物

`reduction.cu` 骨架 + CPU 参考

---

## Step 08：shared memory 树形归约

### 阅读

| 资料 | 重点 |
|------|------|
| Harris 文章 | Kernel 3–4（shared memory reduction） |
| `vec_add.cu` → `reduce_sum` | 迁移并独立测试 |

### 动手

实现 **单 block** 树形归约（`n <= 256` 或 `blockDim=256`）：

```cuda
__shared__ float sdata[256];
// load → __syncthreads → stride /= 2 循环 → thread 0 写结果
```

验收：

```bash
./reduction 256
./reduction 1024   # 若单 block，需 n <= 256；多 block 留 Step 09
```

### 完成标志

- [ ] 单 block 256 元素 PASS
- [ ] 能画出 256→128→64→…→1 的树形图

### 交付物

`reduce_shared` kernel PASS

---

## Step 09：多 block reduction 完整版

### 阅读

| 资料 | 重点 |
|------|------|
| Harris 文章 | 两阶段 reduction：block 内归约 → block 结果再归约 |
| `vec_add.cu` → `test_reduce_sum` | 第二段 CPU 求和思路 |

### 动手

**版本 C**：与 `vec_add` 相同思路

```text
Phase 1：每个 block 归约 → blocksums[blockIdx.x]
Phase 2：CPU 对 blocksums 求和（Week 2 够用）
         或 GPU 再 launch 一次（加分）
```

验收：

```bash
./reduction 1048576    # 1M
./reduction 16777216   # 16M（可选）
```

误差：`fabs(gpu_sum - cpu_sum) / cpu_sum < 1e-4`

### 完成标志

- [ ] 1M 元素 PASS
- [ ] 记录 shared 版 kernel 时间
- [ ] 与 atomic 版对比（若有）记录倍数差距

### 交付物

`reduction` 完整可运行 + Makefile

---

## Step 10：性能测试与对比表

### 阅读

| 资料 | 重点 |
|------|------|
| [T4实战指南.md](T4实战指南.md) | 转置 ~30 GB/s naive → ~200+ GB/s 优化 |
| [项目清单.md](项目清单.md) | P03、P04 验收 |

### 动手

统一用 `cudaEvent` **只计 kernel**，每项跑 3 次取中位数。

**transpose（4096² 或 1024²）**

| 版本 | Kernel (ms) | 带宽 GB/s |
|------|-------------|-----------|
| naive | | |
| shared_simple | | |
| shared | | |

**reduction（1M float）**

| 版本 | Kernel (ms) | 备注 |
|------|-------------|------|
| atomic（若有） | | 慢 |
| shared 树形 | | 主交付 |

### 完成标志

- [ ] 两张表填满
- [ ] 写 5 句「T4 上合并访问实测结论」

### 交付物

`notes/week02.md` 性能章节

---

## Step 11：分析工具初探（选做）

### 阅读

| 资料 | 重点 |
|------|------|
| [compute-sanitizer](https://docs.nvidia.com/compute-sanitizer/) | memcheck 入门 |
| Nsight Systems 文档 | 仅浏览 timeline 概念（Week 6 深入） |

### 动手

```bash
compute-sanitizer --tool memcheck ./transpose 512
# 或
nsys profile -o week2_transpose ./transpose 1024
```

看是否报 illegal access；若有，回到边界检查 `if (row < height && col < width)`。

### 完成标志

- [ ] 至少运行一次 sanitizer 无报错
- [ ] （可选）保存 nsys 截图

---

## Step 12：周复盘与正式交付

### 阅读

- 回顾 [CUDA学习路线图.md](CUDA学习路线图.md) Week 2 验收项
- [项目清单.md](项目清单.md) P03、P04

### 目录自检

```text
week02_memory/
├── transpose/          ✅ naive + shared_simple + shared
│   ├── transpose.cu
│   ├── 转置优化详解.md
│   └── Makefile
└── reduction/          ✅ 1M sum PASS
    ├── reduction.cu
    └── Makefile
notes/
└── week02.md           ✅ 定稿
```

### 自测题（闭卷能答再进入 Week 3）

1. Global Memory 和 Shared Memory 在作用域、速度、声明上有什么不同？
2. 什么是 Memory Coalescing？transpose naive 写回为何不合并？
3. `__syncthreads()` 同步范围是什么？用错会怎样？
4. 什么是 Bank Conflict？transpose 如何用 padding 解决？
5. 树形归约为什么每轮 `stride /= 2`？每轮为何要 sync？
6. 两阶段 reduction 第二阶段为什么可以只在 CPU 做（Week 2）？
7. （Week 3 预热）为什么 GPU 是「弱内存模型」？`__syncthreads()` 除了同步还隐含什么内存可见性保证？

### 完成标志

- [ ] 两个目录均可 `make` 通过
- [ ] `notes/week02.md` 含：内存层次表、transpose 带宽表、reduction 时间表、本周总结
- [ ] 6 道自测题能答对 ≥ 4
- [ ] 知道 Week 3 主题：scan、histogram、stream 重叠

### 交付物

`notes/week02.md` 定稿 + Git 提交（可选）

---

## 附录 A：`notes/week02.md` 建议模板

```markdown
# Week 2 学习笔记

## 内存层次表
（Step 01）

## Coalescing 笔记
（Step 02）

## transpose 带宽对比
| 版本 | 规模 | Kernel ms | GB/s |
（Step 06/10）

## reduction 性能
| 版本 | n | Kernel ms | PASS |
（Step 09/10）

## 合并访问实测结论（5 句）
1. ...
（Step 10）

## 本周总结
- 掌握：
- 薄弱：
- 下周重点：scan、stream
```

---

## 附录 B：常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| transpose shared FAIL | 写回索引对调错误 | 对照 [转置优化详解.md](../week02_memory/transpose/转置优化详解.md) |
| reduction 结果偏小 | 未初始化 shared / 越界线程未填 0 | `global_id >= n` 时 `sdata[i]=0` |
| reduction 死锁 | `__syncthreads` 在分支内不一致 | 保证 block 内所有线程都到达 |
| 带宽计算很小 | 计时包了 H2D/D2H | 只计 kernel；公式用 `/1e9` 且 time 用秒 |
| shared 版反而更慢 | 小矩阵 tile 开销大于收益 | 测 4096²；小矩阵仅验证正确性 |

---

## 附录 C：与 Week 1 代码的衔接

| Week 1 | Week 2 延伸 |
|--------|-------------|
| `vec_add` Global Memory | Coalescing 实验 |
| `vec_add` → `reduce_sum` | 独立 `reduction/` |
| `mat_mul_naive` | 理解 GEMM 访存（Week 5 优化） |
| `mat_mul` 2D grid/block | transpose 同一套 x→col, y→row |
| `cudaEvent` 计时 | transpose/reduction 带宽与耗时 |

---

**返回**：[CUDA学习路线图.md](CUDA学习路线图.md) · **总纲**：[Programming_Guide学习路径.md](Programming_Guide学习路径.md)（阶段 2）· **上一步**：[Week1详细步骤.md](Week1详细步骤.md)
