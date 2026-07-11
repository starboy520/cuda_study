# PagedAttention 详解：从 KV cache 碎片到分页管理

> 面向：已懂 attention / FlashAttention / KV cache 字节账，想搞清 vLLM 核心机制。
> 硬件视角：A100，纯系统 + CUDA，不讨论 ML 训练。
> 一句话：**PagedAttention 用「操作系统虚拟内存分页」的思想管理 KV cache——逻辑连续、物理分散、用 block table 映射——解决连续存储的碎片与浪费，但不改变 attention 的 FLOP。**

---

## 1. 先看清问题：连续 KV cache 为什么不够用

回顾 KV cache 账（见 `kv_cache_accounting.md`）：每个请求、每层都要缓存历史 K/V，随生成不断变长。朴素做法是**给每个请求分配一整块连续显存**。这在「多请求并发 + 变长 + 动态到达/结束」的线上场景会同时踩三个坑：

### 坑 1：预留浪费（over-reservation）
你不知道一个请求最终会生成多长。若按**最大长度**预留（比如 2048 token 的空间），而请求只生成了 50 个 token，剩下 1998 个 token 的空间就**白占**。并发几百个请求时，浪费巨大。

### 坑 2：碎片（fragmentation）
- **外部碎片**：请求 A 结束后释放一块连续空间，但新请求 B 需要更大的连续块，A 留下的洞装不下 → 显存总量够、却没有**连续**的够大块。
- **内部碎片**：按块预留时，最后一块没填满的部分浪费。

### 坑 3：搬迁（migration）
若按当前长度紧凑分配、生成变长后就地扩容，后面可能没有连续空间，被迫**把整个 KV 搬到新位置**——昂贵，且打断流水。

```text
时间0: [AAAA____][BBBB____]          A、B 各预留一大块
时间1: A 生成变长，需要扩容，但右边是 B → 只能搬迁或预留过量
时间2: B 结束 → 留下空洞，C 来了却拼不出连续大块（外部碎片）
```

**结论**：连续分配让「预留 / 搬迁 / 碎片」三者无法同时避免。这正是 PagedAttention 出现的背景。

---

## 2. 核心思想：借用操作系统的虚拟内存分页

操作系统早就解决过一模一样的问题：进程要「看起来连续」的地址空间，但物理内存是分散的、有碎片的。解法是**分页**：

| 操作系统 | PagedAttention |
|---------|---------------|
| 虚拟地址空间（进程视角连续） | 逻辑 KV 序列（请求视角连续） |
| 物理内存页（4KB，分散） | 物理 KV block（固定 N 个 token，分散） |
| 页表（虚拟页→物理页） | **block table**（逻辑块→物理块） |
| 按需分配页 | 按需分配 block（生成到哪，分配到哪） |
| 页共享 / copy-on-write | prefix 共享 / beam search 共享 |

核心一句：**请求以为自己的 KV 是一段连续序列，实际物理上被切成固定大小的块、散落在显存各处，靠一张 block table 把「逻辑第几个块」翻译成「物理第几个块」。**

---

## 3. 三个概念：物理块、逻辑块、block table

### 物理块（physical block）
显存里预先切好的**固定大小**的 KV 存储单元。每个块存 `block_size` 个 token 的 K 和 V（比如 block_size=16）。所有物理块组成一个「块池」（block pool），编号 `P0, P1, P2, ...`。

一个物理块的字节大小（单层单请求，实际实现按 layer/head 组织）：
```text
block_bytes = 2(K,V) × block_size × Hkv × Dh × dtype_bytes
```

### 逻辑块（logical block）
从**请求视角**看，它的 KV 序列被按 `block_size` 切成逻辑块 `L0, L1, L2, ...`：
```text
token 0..15   → 逻辑块 L0
token 16..31  → 逻辑块 L1
token 32..47  → 逻辑块 L2
...
```

### block table
每个请求一张表，记录**它的每个逻辑块对应哪个物理块**：
```text
block_table[逻辑块号] = 物理块号
```

逻辑块**连续编号**（L0,L1,L2…），但它们指向的物理块**可以任意分散**（P2,P0,P3…）。

---

## 4. Block table 手推（核心动手）

设 `block_size = 2`（每块 2 个 token，方便手算），物理块池有 `P0..P3`。

请求 A 有 token 0..4（共 5 个 token），需要 3 个逻辑块（L0,L1,L2），调度器给它分配了物理块：

```text
block_table_A = [P2, P0, P3]
                 ↑L0 ↑L1 ↑L2
```

**给定 token t，怎么找到它的物理位置？**

```text
logical_block = t / block_size      // 它在第几个逻辑块
offset        = t % block_size      // 在块内第几个位置
physical_block = block_table_A[logical_block]
```

逐 token 推导：

| token t | logical_block = t/2 | offset = t%2 | physical_block |
|--------:|--------------------:|-------------:|----------------|
| 0 | 0 | 0 | P2 |
| 1 | 0 | 1 | P2 |
| 2 | 1 | 0 | P0 |
| 3 | 1 | 1 | P0 |
| 4 | 2 | 0 | P3 |

**读懂这张表**：token 0、1 在物理块 P2 的第 0、1 位；token 2、3 跳到 P0；token 4 又跳到 P3。物理上完全不连续，但请求只需按 t=0,1,2,3,4 顺序访问，block table 负责翻译。

---

## 5. 从「逻辑 token」到「物理地址」的完整计算

上面的 physical_block 只是「第几个块」。真正取 KV 元素时，地址还要叠加 layer / KV head / head dim / dtype 的 stride：

```text
最终地址 ≈ block_pool_base
         + physical_block × block_stride        // 定位到物理块
         + offset        × token_stride          // 块内第几个 token
         + layer         × layer_stride          // 第几层（每层独立 KV）
         + head          × head_stride            // 第几个 KV head
         + d             × dtype_bytes            // head 内第几维
```

关键点：**block table 只解决「逻辑块→物理块」这一层映射，不减少每个有效 KV 元素本身**。数据还是那么多，只是**存放位置的组织方式**变了。

---

## 6. Attention kernel 要怎么改

标准 FlashAttention 假设 KV 在**连续内存**里，直接 `K + row*D` 就能取。PagedAttention 下，KV **物理分散**，kernel 必须**先查 block table 再 gather**：

```text
标准（连续）：
  for j in 0..seq_len:
      k = K_cache[j]           // 连续，直接索引

Paged（分块分散）：
  for j in 0..seq_len:
      lb  = j / block_size
      off = j % block_size
      pb  = block_table[lb]                    // 查表
      k   = block_pool[pb][off]                // 到分散的物理块取
```

实现要点：
- **一个 query 仍和整个 KV 序列做 attention**，只是 K/V 分块存在不同物理块，逐块 gather。
- online softmax 的分块天然契合：**Flash 本来就是按 KV 块遍历的**，PagedAttention 只是让「每块的物理地址」来自 block table，而不是连续偏移。所以 **PagedAttention 常和 FlashAttention 融合实现**（vLLM 的 kernel 就是 paged + flash）。
- block table 通常放 global 或 constant memory，kernel 里先读它拿到物理块号。
- 边界：最后一个逻辑块可能没填满（只有部分 token 有效），要按实际长度 mask。

---

## 7. Block size 的权衡

`block_size` 是核心调参：

```text
太大（如 128）：
  - 最后一块内部浪费大（内部碎片）
  - 调度/分配粒度粗，不够灵活
  + block table 更短，查表开销小
  + 每块内更连续，访存效率好

太小（如 4）：
  - block table 更长，metadata 多、查表频繁
  - 块间跳转多，连续访问和 kernel 效率可能变差
  + 内部碎片小，显存利用率高
```

实践常用 16 左右，是「碎片 vs 访存效率 vs metadata」的折中。**没有唯一最优，取决于模型和负载**。

---

## 8. PagedAttention 的额外威力：内存共享

分页最漂亮的副产品是**多个请求/序列共享物理块**（像 OS 的共享页 + copy-on-write）：

### prefix 共享（prompt 复用）
多个请求有**相同的前缀**（比如同一个 system prompt、few-shot 示例）：它们的前缀 KV 完全一样 → **让它们的 block table 指向同一批物理块**，前缀 KV 只存一份。

```text
请求 A: block_table = [P0, P1, P2(A独有)]
请求 B: block_table = [P0, P1, P3(B独有)]
                       ↑共享前缀 P0,P1（只存一份）
```

### beam search / 并行采样共享
同一个 prompt 生成多个候选（beam），它们共享 prompt 部分的 KV，只有各自新生成的部分独立。**分叉时用 copy-on-write**：要改写共享块才复制。

这能**大幅省显存**，是连续分配根本做不到的（连续存储没法让两个请求指向同一段而各自独立扩展）。

---

## 9. PagedAttention 不做什么（重要，防误解）

| 误解 | 事实 |
|------|------|
| 把 attention 从 O(N²) 变 O(N) | ❌ 不改变 FLOP。计算量还是那么多，只改**存储管理** |
| 减少 KV cache 的总数据量 | ❌ 有效 KV 元素一个不少，只是**存放方式**变了（甚至加了 block table 的 metadata 开销） |
| 让 attention 变快 | ⚠️ 主要收益是**显存利用率 / 吞吐**（能同时塞更多请求），不是单请求 attention 更快；分散访问甚至可能略慢一点 |
| 替代 GQA/MLA | ❌ 正交。GQA/MLA 减少**每 token 的 KV 大小**；Paged 优化**KV 的存储管理**。两者可叠加 |

**一句话**：PagedAttention 优化的是**显存怎么管**（利用率、碎片、共享），不是**算得多快**或**数据多少**。它让你在同样显存下**跑更多并发请求**，从而提升整体**吞吐**。

---

## 10. 和 continuous batching、vLLM 的关系

PagedAttention 是 **vLLM** 的核心创新，它和 **continuous batching** 配合：

- **continuous batching**：每一步调度器重新挑选 active 请求，新到的插入、完成的移出，把多个请求的「当前 step」打包（M 从 1 → active_batch）。
- **PagedAttention**：让每个请求的 KV 灵活分块存储，**新请求随到随分配块、完成就回收块**，没有搬迁、碎片可控 → 才能支撑 continuous batching 的动态进出。

两者合起来 = vLLM 高吞吐的基础：**显存像内存池一样按块动态借还，请求自由进出，前缀还能共享。**

```text
调度器（continuous batching）：决定这一步跑哪些请求
        ↓
块管理器（PagedAttention）：给活跃请求按需分配/回收物理块，维护 block table
        ↓
paged flash attention kernel：按 block table gather KV，做 attention
```

---

## 11. 性能与工程注意

- **分散访问的代价**：物理块不连续，attention gather KV 时的访存局部性不如连续存储，可能有轻微带宽损失。好的 kernel 用合理 block_size + 块内连续 + 预取来缓解。
- **block table 开销**：查表、存表都有成本，block_size 太小会放大。
- **块池管理**：分配/回收物理块的簿记（空闲链表、引用计数用于共享块的 copy-on-write）要高效，否则调度成瓶颈。
- **和量化叠加**：KV 也能量化（INT8/FP8）再分页存，进一步省显存。

---

## 12. 闭卷自测

1. 连续 KV cache 有哪三个坑？（预留浪费 / 碎片 / 搬迁）
2. block table 映射的是什么？（逻辑块 → 物理块）
3. 给 token t、block_size B，怎么算它的物理块和块内偏移？（`t/B` 查表得物理块，`t%B` 是偏移）
4. block_size 太大 / 太小各有什么代价？
5. PagedAttention 改变 attention 的 FLOP 吗？（不改，只改存储管理）
6. 它靠什么省显存 / 提吞吐？（消除碎片和预留浪费 + 前缀/beam 共享物理块）
7. 它和 continuous batching 怎么配合？（分页支撑请求动态进出 + 块动态借还）
8. 它和 GQA/MLA 是替代还是正交？（正交：一个管存储、一个减每 token KV 大小）

## 闭卷口述

> 连续存 KV cache 会同时踩预留浪费、碎片、搬迁三个坑。PagedAttention 借操作系统分页思想：把每个请求的 KV 按固定 block_size 切成逻辑块，物理上分散存在块池里，用 block table 把逻辑块映射到物理块——`token t` 的物理块是 `block_table[t/block_size]`、块内偏移是 `t%block_size`。attention kernel 先查表再 gather KV，和 FlashAttention 的分块遍历天然契合。它不改变 attention 的 FLOP、不减少有效 KV 数据量，而是优化显存管理：消除碎片和预留浪费、支持前缀/beam 的物理块共享，从而在同样显存下跑更多并发、提升吞吐。它和 continuous batching 配合支撑请求动态进出，是 vLLM 的核心；和 GQA/MLA 正交可叠加。

---

## 附：与本仓库其他文档的关系

- KV cache 字节账：[kv_cache_accounting.md](kv_cache_accounting.md)
- decode 数据流与瓶颈：[decode_step_dataflow.md](decode_step_dataflow.md)
- KV cache 系统指南（含 PagedAttention 背景）：[大模型KVCache系统学习指南.md](大模型KVCache系统学习指南.md) §12
- Week5 Day7 block table 手推：[Week5增强版_LLM推理优化与decode.md](../../courses/inference/Week5增强版_LLM推理优化与decode.md) §46-48
- 官方：[vLLM 文档](https://docs.vllm.ai/)、[PagedAttention 论文](https://arxiv.org/abs/2309.06180)
