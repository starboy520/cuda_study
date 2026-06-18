# Week 3 每日详细安排（Day1–Day7）

> **主题**：并行模式与同步 —— Atomic、Warp 级原语（shuffle/vote）、Scan、Histogram、Stream/Event 重叠
> **硬件**：Tesla T4（sm_75）/ 也可在 A100（sm_80）跑，注意 occupancy 分母不同（T4=32 warp/SM，A100=64）
> **前置**：Week 2 已完成 transpose 三版 + reduction 多阶段（都已 PASS）
> **本周交付**：`week03_parallel/` 下 4 个程序（warp_reduce / scan / histogram / stream_overlap）+ `notes/week03.md`

---

## 使用方式

- 按 Day1 → Day7 顺序做；每天有「学什么 / 看什么 / 动手 / 完成标准」。
- 每个概念都要落到**自己手写的代码**上——这是唯一的练习方式（沿用 week2 的方法）。
- 每天在 `notes/week03.md` 记三件事：**目标 / 实验结果 / 遇到的问题**。
- 阅读标注：📖 精读 · 👀 扫读 · ✍️ 必须自己写 · ⏭️ 本周跳过。

---

## 本周总览（一张表看完）

| Day | 主题 | 动手产出 | 关键概念 |
|-----|------|----------|----------|
| 1 | Atomic 与竞争 | `atomic_sum` + 分层聚合 | race / atomic / 竞争串行化 |
| 2 | Warp 级原语 | `warp_reduce`（shuffle 版） | warp / lane / `__shfl_down_sync` / mask |
| 3 | Reduction 进阶 | 给 week2 reduction 加 shuffle 收尾 | 寄存器交换免 barrier |
| 4 | Scan（前缀和）上 | `scan` 单 block 版 | Hillis-Steele / inclusive vs exclusive |
| 5 | Scan 下 + Histogram | `scan` 多 block + `histogram` | 多阶段 scan / privatization |
| 6 | Stream / Event 重叠 | `stream_overlap` | stream 语义 / pinned / 重叠 |
| 7 | 复盘 + 性能表 | `notes/week03.md` 定稿 | 何时用 atomic/shuffle/shared |

---

## 本周用得上的现成资料（你工作区里已有）

> 这些是已经写好、深度扩写过的教材正文和可运行实验，**配合官方文档一起看**，比啃英文原文高效：

| 主题 | 教材正文（读） | 可运行实验（对照） |
|------|---------------|-------------------|
| Atomic / Warp 原语 | `cuda_deep_course/.../volume04_parallel_algorithms/02_Atomic与Warp级原语.md` | — |
| Reduction 进阶 | `.../volume04_.../03_Reduction从错误到优化.md` | `cuda_deep_course/labs/04_parallel_algorithms/reduction/` |
| Scan / Histogram | `.../volume04_.../04_Scan与Histogram.md` | — |
| Race / 同步 | `.../volume04_.../01_Race_同步与内存可见性.md` | — |
| Stream / 重叠 | `.../volume07_async_system/02_传输与计算重叠.md` | `cuda_deep_course/labs/07_async_system/overlap_pipeline/` |

官方文档（v13.x）：
- 📖 **5.4.5 Atomic Functions**、**5.4.6 Warp Functions**（shuffle/vote）、**5.4.4 Synchronization**
- 📖 **2.5 Asynchronous Execution**（Stream/Event）
- 👀 **4.1 Unified Memory**（你已有专门文档 `notes/CUDA统一内存详解.md`，扫读官方即可）
- 👀 Harris《Optimizing Parallel Reduction》（scan/reduction 优化思想）

⏭️ **本周跳过**（留到后面）：Cooperative Groups 深入、`cudaMallocAsync`、CUDA Graph、Blelloch work-efficient scan 的完整实现（先用 Hillis-Steele 建立直觉）。

---

## Day 1：Atomic 与竞争

**学什么**
- 什么是 data race（多线程并发读改写同一地址）。
- `atomicAdd` 如何保证 read-modify-write 不可分割。
- 为什么"百万线程 atomic 同一地址"会串行化、变慢。
- 分层聚合（privatization）：局部攒 → block 内聚合 → 少量 global atomic。

**看什么**
- 📖 教材 `volume04_.../02_Atomic与Warp级原语.md` 第 1–3 节（我深化过，有竞争量化的数字）。
- 📖 官方 5.4.5 Atomic Functions（`atomicAdd/atomicMax/atomicCAS` 列表）。
- 👀 教材 `volume04_.../01_Race_同步与内存可见性.md`（race 的本质）。

**动手** ✍️ 新建 `week03_parallel/atomic_sum/atomic_sum.cu`
1. 版本 A：所有线程 `atomicAdd(&result, input[i])` → 正确但慢，测耗时。
2. 版本 B：每 block 先在 shared 里 `atomicAdd` 聚合，block 末尾只发 1 次 global atomic。
3. 对比两版耗时，验证"分层聚合降竞争"。

**完成标准**
- [ ] 两版结果都正确（和 CPU 对比）。
- [ ] 能说出版本 A 慢的原因（全员争一个地址 → 串行化）。
- [ ] 笔记记录两版耗时差距。

---

## Day 2：Warp 级原语（shuffle / vote）

**学什么**
- warp = 32 线程，lane = warp 内编号（0–31）。
- `__shfl_down_sync(mask, val, delta)`：lane 之间**直接交换寄存器**，不经过 shared。
- 为什么要 `_sync` 后缀和 mask（Volta+ 独立线程调度，必须显式声明参与的 lane）。
- vote：`__ballot_sync` / `__any_sync` / `__all_sync`（把 predicate 收成 mask）。

**看什么**
- 📖 教材 `volume04_.../02_Atomic与Warp级原语.md` 第 4–7 节（shuffle/vote/mask）。
- 📖 官方 5.4.6 Warp Functions。

**动手** ✍️ 新建 `week03_parallel/warp_reduce/warp_reduce.cu`
1. 写一个**单 warp（32 线程）归约**：用 5 次 `__shfl_down_sync`（偏移 16/8/4/2/1）。
2. 手工推演 8 个 lane 的 shuffle-down，确认每步谁加谁。
3. 验证 lane 0 拿到 32 个数之和。

**完成标准**
- [ ] warp_reduce 结果正确。
- [ ] 能解释为什么偏移是 16/8/4/2/1（= log2(32) 轮树形归约）。
- [ ] 能解释 mask=0xffffffff 的含义。

---

## Day 3：Reduction 进阶（把 shuffle 用进去）

**学什么**
- 树形归约后期（剩 ≤32 个值）落在同一个 warp 内时，`__syncthreads()` 是浪费。
- 用 warp shuffle 收尾：最后 32 个值用寄存器交换，免 shared、免 block barrier。

**看什么**
- 📖 教材 `volume04_.../03_Reduction从错误到优化.md` 第 5 节（warp shuffle 收尾，我深度扩写过）。
- 对照实验 `labs/04_parallel_algorithms/reduction/reduction.cu`（有 `reduceShared` 和 `reduceWarpShuffle` 两版）。

**动手** ✍️ 改你 week2 的 `week02_memory/reduction/reduction.cu`（或复制到 week03）
1. 在你的 shared 树形归约里，当 `stride < 32` 时切换到 Day2 的 shuffle 收尾。
2. 对比"纯 shared 树形" vs "shared + shuffle 收尾"的耗时。
3. 用大输入（如 1<<24）测，小输入差距会被噪声淹没。

**完成标准**
- [ ] shuffle 版结果仍 PASS。
- [ ] 记录两版耗时（参考：lab 里 shuffle 版约比纯 shared 快近 2 倍）。
- [ ] 能说出 shuffle 收尾省了什么（shared 往返 + block barrier）。

---

## Day 4：Scan（前缀和）上 —— 单 block

**学什么**
- Scan 定义：inclusive `[a, a+b, a+b+c, ...]` vs exclusive `[0, a, a+b, ...]`。
- Scan 是 compact、排序、资源分配的基础，用途极广。
- Hillis-Steele 算法：每轮距离翻倍（offset=1,2,4,...），深度 O(log n)，工作量 O(n log n)。

**看什么**
- 📖 教材 `volume04_.../04_Scan与Histogram.md` 第 1–2 节（我补过 Hillis-Steele 逐轮图解 + 工作量数字）。
- 👀 官方/GPU Gems 3 Ch.39（前缀和，选读）。

**动手** ✍️ 新建 `week03_parallel/scan/scan.cu`
1. 先实现**单 block** inclusive scan（Hillis-Steele），block 内用 shared。
2. 注意每轮要 `__syncthreads()`（读上一轮别的线程的结果）。
3. 与 CPU 串行 scan 逐元素对比。

**完成标准**
- [ ] 单 block scan 正确（n ≤ 1024，如 256/512/1024）。
- [ ] 能手工推演 8 元素的 Hillis-Steele 每一轮。
- [ ] 理解 inclusive 和 exclusive 的差别（能互相转换）。

---

## Day 5：Scan 下（多 block）+ Histogram

**学什么**
- 多 block scan：① 每 block 内部 scan ② 收集每 block 总和 ③ scan 这些总和 ④ 把 offset 加回各 block（典型分层并行）。
- Histogram：每个输入值更新一个 bin（`atomicAdd(&hist[v], 1)`）。
- Histogram privatization：每 block 先在 shared 建私有直方图，最后合并到 global（减少 global atomic 竞争）——和 Day1 的分层聚合同一思路。

**看什么**
- 📖 教材 `volume04_.../04_Scan与Histogram.md` 第 4–7 节（多 block scan + histogram privatization）。

**动手** ✍️
1. 把 Day4 的 scan 扩展到**多 block**，支持 1M 元素，与 CPU 验证。
2. 新建 `week03_parallel/histogram/histogram.cu`：先 global atomic 版，再 shared privatization 版。
3. 用两种数据测 histogram：均匀分布 vs 90% 集中在一个 bin，观察竞争差异。

**完成标准**
- [ ] 多 block scan 在 1M 元素 PASS。
- [ ] histogram 两版都正确。
- [ ] 能解释为什么"集中分布"时 global atomic 版特别慢，privatization 能救。

---

## Day 6：Stream / Event 与传输计算重叠

**学什么**
- Stream 核心语义：**同一 stream 内顺序，跨 stream 间并发**。
- 默认 stream 的坑（会破坏并发），要显式建非默认 stream。
- pinned memory（`cudaMallocHost`）是 `cudaMemcpyAsync` 真异步的前提。
- 分块流水线：数据切块 + 多 stream，让"传 + 算"重叠。

**看什么**
- 📖 教材 `volume07_async_system/01_Stream与异步执行模型.md` + `02_传输与计算重叠.md`（我写的，有实测 2.86x）。
- 对照实验 `labs/07_async_system/overlap_pipeline/overlap_pipeline.cu`。
- 📖 官方 2.5 Asynchronous Execution。

**动手** ✍️ 新建 `week03_parallel/stream_overlap/stream_overlap.cu`
1. 写一个 `H2D → kernel → D2H` 的任务，先单 stream 串行版测耗时。
2. 改成 pinned memory + 多 stream 分块版，测重叠后耗时。
3. （可选）故意用 pageable 内存，验证重叠失效。

**完成标准**
- [ ] 重叠版比串行版快（目标 ≥15%，参考 lab 能到 2x+）。
- [ ] 能画出重叠前后的 timeline 示意图（手绘即可）。
- [ ] 能解释为什么 pinned 是重叠的前提。

---

## Day 7：复盘 + 性能表定稿

**动手**
1. 整理 `notes/week03.md`：四个程序的耗时/对比表、关键结论。
2. 写一段"何时用 atomic vs shuffle vs shared reduction"的口述总结。
3. 回顾本周遇到的坑（写进笔记，方便以后查）。

**本周核心问题自测**（能口头答出就算掌握）
- [ ] atomic 为什么保证正确但可能慢？怎么缓解（分层聚合）？
- [ ] warp shuffle 相比 shared 归约省了什么？什么时候用？
- [ ] Hillis-Steele scan 的工作量为什么是 O(n log n)？
- [ ] 多 block scan 为什么要分三步（块内 scan / 块和 scan / 加回 offset）？
- [ ] histogram privatization 解决什么问题？
- [ ] stream 的核心语义是什么？pinned memory 为什么是重叠前提？

---

## 本周交付清单

```text
week03_parallel/
├── atomic_sum/atomic_sum.cu          (Day1)
├── warp_reduce/warp_reduce.cu        (Day2)
├── reduction_shuffle/...             (Day3，可复用 week2 reduction)
├── scan/scan.cu                      (Day4-5)
├── histogram/histogram.cu            (Day5)
└── stream_overlap/stream_overlap.cu  (Day6)

notes/week03.md  —— 每日记录 + 4 张对比表 + 自测题答案
```

---

## 进度可视化

```mermaid
flowchart LR
  D1["Day1 Atomic<br/>竞争+分层聚合"] --> D2["Day2 Warp原语<br/>shuffle/vote"]
  D2 --> D3["Day3 Reduction<br/>shuffle收尾"]
  D3 --> D4["Day4 Scan上<br/>单block"]
  D4 --> D5["Day5 Scan下+Histogram<br/>多阶段/privatization"]
  D5 --> D6["Day6 Stream<br/>传输计算重叠"]
  D6 --> D7["Day7 复盘<br/>性能表"]
```

---

## 给你的提醒（沿用 week2 的节奏）

- **遇到"要不要深入/优化"，默认答案：记下来，留到卷四/卷五。** Week 3 目标是"会写、理解原理"，不是调到极致。
- **Blelloch work-efficient scan 本周不强求**：先用 Hillis-Steele 把 scan 跑通、理解概念，work-efficient 版留到后面。
- **每个概念都要落到代码**：看懂 ≠ 学会，亲手写出来 + 跑 PASS 才算。
- **大输入测性能**：小输入（n<10000）的耗时差异会被 launch 开销和噪声淹没，测性能用 1<<20 以上。
