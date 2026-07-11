# CUDA 性能工程复习资料（知识体系版）

> **归档状态**：历史资料，不是当前执行入口。保留用于检索旧课程设计和技术解释；当前导航见相应 README。

> 定位：系统化、连贯讲解的 CUDA/GPU 性能工程复习资料，覆盖从硬件模型到深水区。
> 配套：题目练习见 [CUDA面试八股全集.md](../../interview/CUDA面试核心题库.md)；本篇是知识主线的系统梳理。
> 用法：通读建立体系，每章末的「关键结论」「高频易错」是背诵重点。
> 贯穿原则：回答架构问题**先结论、再条件边界、最后证据**；不用"永远/一定/默认都这样"这类跨架构绝对话术。

---

# 一、GPU 硬件与执行模型

## 1.1 CPU vs GPU 的设计哲学

CPU 和 GPU 的根本差异是**优化目标不同**：

- **CPU**：少量复杂核心，追求**单线程低延迟**。靠大 cache、乱序执行、分支预测、深流水，让**每一条指令尽快完成**。
- **GPU**：海量简单执行单元，追求**吞吐**。单次操作（如一次 global load）本身**并不快**，但靠**大量并行 warp**，某个 warp 等待时调度别的 ready warp，把延迟**藏起来**。

核心概念——**延迟隐藏（latency hiding）≠ 降低延迟（reduce latency）**：
- 降低延迟 = 让单次操作更快（CPU 路子）
- 延迟隐藏 = 单次操作还是慢，但用并行把等待"盖住"（GPU 路子）

**适合 GPU 的问题**：大量独立工作、规则数据访问、较高算术强度。**不适合**：强串行依赖、复杂分支、工作量很小（会被 launch 开销主导）。

## 1.2 硬件层级：GPC / TPC / SM / warp scheduler / CUDA Core

从大到小：**GPC → TPC → SM → (warp scheduler + 执行管线 + register file + shared/L1)**。

- **SM（Streaming Multiprocessor）**：block 实际驻留和执行的主要单元。SM 内有多个 warp scheduler、寄存器文件、shared/L1、多类执行管线（CUDA Core 算 FP32/INT、SFU/MUFU 算超越函数、Tensor Core 算矩阵、LSU 访存）。
- **CUDA Core**：标量算术执行资源之一，**不是**能独立调度线程的完整 CPU 核。
- **warp scheduler**：每拍从 eligible warp 里挑指令，发射到合适的执行管线。

**关键澄清**：
- 层级**不是**"上层每次只调一个下层"。一个 SM 可**同时驻留多个 block/warp**。
- **block 一旦分配到某 SM,不会迁移**——所以 block 内能用 shared memory / `__syncthreads`。
- scheduler 数、每拍 issue 能力随架构变，**别背"每 SM 4 个 scheduler、每个每拍一条"当定律**。

## 1.3 SIMT vs SIMD

- **SIMD**：暴露向量指令和向量 lane，程序员写向量操作。
- **SIMT**：程序员写**标量线程**，硬件按 **warp（通常 32 线程）** 把线程指令成组执行，同时保留每线程的寄存器和执行状态。

**warp divergence（分歧）**：同一 warp 的线程走不同控制流路径时，硬件用 **active mask 分阶段执行不同路径**，降低有效 lane 利用率。

**Volta 后的 independent thread scheduling（ITS）**：每线程有独立 PC 和栈，分歧线程能各自推进、交错执行，还支持分歧线程间更细的同步。**但**：① 分歧仍不免费（路径仍分阶段执行）；② 不能再依赖旧的"隐式 warp 同步"——老代码假设"32 线程锁步"省掉 `__syncwarp()`，ITS 后可能出错，必须显式同步。

## 1.4 warp divergence 的优化

**为什么慢**：分歧路径按 mask 分别执行，有效并行度下降。

**优化方向**（目标是让**分支边界和 warp 工作划分对齐**，而非机械删 `if`）：
- 按数据分组，让整个 warp 处理同类任务
- 把边界检查集中到少数 warp
- 减少路径间工作量不均

**判据（重要）**：分支会不会 divergence,看**分支条件在一个 warp 的 32 lane 里是否一致**：
- `threadIdx.x/32`（warp_id）、`blockIdx.x` → warp 内一致 → **不 divergence**
- `threadIdx.x%32`（lane_id）、`threadIdx.x<10`（线程级边界）、`data[tid]>0`（数据依赖）→ 可能 divergence

**predication**：把两个分支都算、再用条件选，避免跳转但两边都执行。分支短、工作小时划算；分支长、工作大时反而慢。**要实测,不一定更快。**

> **本章关键结论**：GPU 靠延迟隐藏(大量 warp 切换)而非降低延迟取胜。SM 是驻留/执行单元,block 不迁移。SIMT 写标量、按 warp 执行。divergence 判据 = 分支条件在 warp 内是否一致。
> **高频易错**：① "GPU 核多所以都更快"（错,串行/小任务不适合）② 把 CUDA Core 当独立 CPU 核 ③ 背固定 scheduler 数 ④ 以为 ITS 让分歧免费。

---

# 二、内存层次与访问

## 2.1 各级内存

| 内存 | 位置 | 作用域 | 特点 |
|------|------|--------|------|
| **register** | 片上 | 每线程 | 最快,数量有限 |
| **local** | **片外**(经 cache) | 每线程私有 | "local"不是片上!spill/大数组/动态索引会用它 |
| **shared** | 片上 | block 内 | 快,显式管理,和 L1 共享容量 |
| **L1/L2** | 片上 | L1 每 SM / L2 全局 | 硬件缓存 |
| **global** | 片外(HBM) | 全局 | 大,慢,带宽是常见瓶颈 |

**关键澄清**：
- **local memory 不是"靠近线程的快内存"**——它物理在片外 global 空间,经 cache。用它的原因：寄存器 spill、大局部数组、**动态索引的局部数组**（编译器无法确定下标）。
- **L1/shared 容量可配置且随架构变**,别给所有 GPU 套固定 KB 数字（Volta 96KB、A100 可到 164KB/SM 在 L1/shared 间划分）。

## 2.2 Global Memory Coalescing（合并访问）

**定义**：一个 warp 的内存请求按地址分布,合并成**尽可能少、有效字节利用率尽可能高**的内存事务。

**核心不是"用了连续数组",而是**：地址连续 + 对齐 + 有效字节利用率高。

- 相邻线程访问相邻元素通常有利
- 具体事务由架构、访问宽度、对齐、cache sector（每 32 字节一个 sector）决定
- **跨步访问**会触及更多 segment/sector → 每个事务里大量无效字节 → 浪费带宽

**两个经典场景**：
- **AoS（结构体数组）浪费事务**：当 warp 只访问结构体的**部分字段**时。字段交错存放,取单字段=跨步访问,每个事务搬回一堆用不上的字段。解法：改 **SoA**（每字段各自连续）。整个结构体都用时 AoS 影响小。
- **转置的读写冲突**：`out[j][i]=in[i][j]`,按行读 in（合并）则按列写 out（跨步），反之亦然,**读写合并方向天生冲突**。解法：**shared memory 中转**（合并读进 shared,再合并写出）。

**GEMV 实证（你的项目）**：v0 一线程一行,相邻线程地址 stride=K 不合并 → 72 GB/s；v1 一 warp 一行,32 lane 读连续 k 合并 → 1360 GB/s（19×,达峰值 ~70%）。**合并访问是 memory-bound kernel 的头号优化。**

## 2.3 Shared Memory 与 Bank Conflict

**为什么快**：片上,低延迟高带宽（比 global 快一两个数量级）。

**为什么可能成瓶颈**（两个机制,都不是"回退 global"）：
1. **占用率压力**：shared 是每 SM 有限资源,被所有驻留 block 共享。一个 block 用太多 → 能驻留的 block 变少 → **occupancy 下降** → 延迟藏不住。**注意：shared 不会 spill 到 global（超上限是启动失败）；会 spill 的是寄存器。**
2. **bank conflict**：shared 分 32 个 bank。一个 warp 的一次 shared 指令中,多个线程访问**同一 bank 的不同地址** → 分多波服务（串行化）。**同 bank 同地址是广播,不算冲突。**

**bank conflict 优化**：
- 经典转置用 `shared[32][33]`（padding 一列）改变行 stride,让按列访问时各 lane 落到不同 bank。
- **padding 不是万能**：增加 shared 占用,也可能对另一种访问模式无效。
- **证明冲突要用 ncu 指标**（如 bank conflict way 数/wavefront）,不能只推导。你 week05 从 4.8-way 降到 ~1.5-way 就是实证。

## 2.4 向量化访存（float4）

**为什么可能更快**：`float4`（128-bit）编译成一条 `LDG.E.128`,减少 load/store 指令数和地址计算。

**前提（缺一不可）**：
1. **16 字节对齐**（地址 %16==0；cudaMalloc 基址够,但加偏移 `row*K` 要 K%4==0 才每行对齐）
2. **元素个数是 4 的倍数**（否则单独处理尾部,防越界）
3. **内存连续**（那 4 个 float 物理相邻,transpose/AoS 不行）
4. **不越界**

**什么时候收益小/变慢**：
- 向量化只优化**指令和事务表达**,**不提高算法 AI**。若 kernel 已被计算/shared/latency 限制,收益很小。你 GEMV float4 只 +4%,因为 warp/row 已 memory-bound（DRAM 72%）——瓶颈是带宽不是指令。
- 强行向量化导致未对齐、复杂尾部、寄存器压力,反而变慢。
- **必须用 SASS 验证**真落成 `LDG.E.128`,源码写 `float4` 不保证（对齐不满足编译器会退回 4 条 32-bit）。宽 load 仍可能跨多个 sector（一条指令≠一个 sector）。

> **本章关键结论**：coalescing 看"warp 地址是否连续对齐",是 memory-bound 头号优化。shared 快但吃 occupancy + bank conflict。local 在片外。float4 减指令但 memory-bound 时收益小,且要 SASS 验证。
> **高频易错**：① 把 local 当片上 ② "地址递增就是一次事务" ③ "两线程同 bank 就冲突"（同地址是广播）④ 把 float4 类型等同一条机器指令。

---

# 三、同步、原子与一致性

## 3.1 三种同步原语

| 原语 | 作用 | 范围 |
|------|------|------|
| `__syncthreads()` | block 内 barrier,所有线程到齐 + 建立内存顺序 | 整个 block（CTA collective） |
| `__syncwarp(mask)` | warp 内指定 lane 同步 | mask 标明的 lane |
| `__threadfence()` | 约束**调用线程自己**的内存操作顺序/可见性 | 不等待其他线程 |

**关键区别**：
- **barrier（syncthreads）**：等参与线程到齐,并建立内存顺序。若只有部分线程到达且条件不一致 → **死锁或未定义**。
- **fence（threadfence）**：只保证**自己**的内存操作顺序（如 data 的写先于 flag 的写可见）,**不通知消费者、不等消费者**。

**producer-consumer 协议**：`写 data → __threadfence() → 写 flag`,消费者必须**主动轮询 flag**（用 volatile/atomic 正确读,防编译器缓存）,看到 flag 置位再读 data。fence 只管顺序,不管"通知"。

**`__syncwarp(mask)` 的 mask 必须准确**：标明哪些 lane 参与。分歧后做 shuffle/syncwarp,mask 必须精确反映活跃 lane,否则未定义/挂起。

## 3.2 原子操作

**保证**：指定地址的 read-modify-write 不被并发破坏。

**不保证**：整个多地址算法自动正确、不消除争用、不是全局 barrier。

**争用优化（分层聚合）**：大量线程竞争少数地址会串行化。典型路径：
```
寄存器局部聚合 → warp/block 聚合（shared atomic）→ 少量 global atomic 提交
```

- **shared atomic vs global atomic**：shared 在片上,延迟低,适合 block 内先聚合；global 在片外,延迟高、争用重。
- **warp-aggregated atomic**：warp 内多 lane 对**同一地址** atomicAdd 时,先用 shuffle/ballot 把增量合并成一个总和,只让 leader 发**一次** global atomic。32 次压成 1 次。

## 3.3 grid 级同步与 cooperative groups

**普通 kernel 没有任意 grid barrier**：block 可按任意顺序调度。若已运行的 block 自旋等待"还没调度上来的 block",而运行中的 block 占满 SM → **死锁**。不能假设所有 block 并发存在。

**全局阶段边界的可靠做法**：**用多个 kernel**。kernel 之间有隐式全局同步（kernel2 一定在 kernel1 全部 block 完成后开始）。三阶段 scan 就靠 kernel 边界做全局同步。

**cooperative groups**：
- 提供 grid_group（`grid.sync()`）,但需 **cooperative launch**（`cudaLaunchCooperativeKernel`）,且受"所有参与 block 能否同时驻留"约束——grid block 数不能超过能同时驻留的上限。
- 也提供 thread block、tiled partition 等结构化 collective,不只 grid sync。
- 能拆成多 kernel 时,kernel 边界通常更简单可靠。

> **本章关键结论**：barrier 等人+建顺序,fence 只管自己的顺序不通知。原子只保证单地址 RMW,靠分层聚合降争用。grid 同步靠 kernel 边界或 cooperative launch(有驻留约束)。
> **高频易错**：① "fence 后别人自动看到并消费" ② "用了 atomic 就没 data race" ③ "include cooperative_groups 就能让所有 block sync"。

---

# 四、Occupancy、寄存器与延迟隐藏

## 4.1 Occupancy

**定义**：驻留的 active warps 与架构最大 warps 的比例。它提供隐藏延迟的潜在 **TLP**,但**不是性能目标**,没有通用的"达到 X% 就够"。

**限制来源**：registers/thread、shared/block、threads/block、架构上限。四者取最紧的。

**理论 vs achieved**：
- **理论 occupancy**：资源算出的上限。
- **achieved occupancy**：运行时实际平均驻留比例,受尾部效应、调度不均影响,通常低于理论。

**为什么高 occupancy 不等于快**：
- 低 occupancy + 高 ILP/数据复用仍可能快（register tiling 就是）。
- 高 occupancy 也可能**所有 warp 都在等同一依赖**（如一起等 global 访存）→ 某拍没 eligible warp → issue 不足。
- 要同时看 **eligible/issued warps、管线利用、最终时间**。

## 4.2 寄存器、spill、ILP、TLP 的权衡

- **更多寄存器**：能存 accumulator、提高复用和 **ILP**,但压低驻留 warp（降 occupancy）；压太低会 **spill 到 local memory**（片外,慢）。
- **ILP（指令级并行）**：同一 warp 内多条独立指令（多个独立 load / 多个 accumulator）,一个 warp 自己就有活干。
- **TLP（线程级并行）**：更多 ready warp,靠数量切换藏延迟。
- 两者都能藏延迟,可组合。**最优点由复用、ILP、TLP、片外流量共同决定。**

**工具**：
- `-Xptxas=-v` 看 registers/spill
- SASS 看 local load/store 路径（区分动态索引局部数组 vs spill）
- ncu 看 occupancy、local traffic、eligible warp、吞吐
- `__launch_bounds__`/`maxrregcount` 只是影响编译器策略,**不是免费提高 occupancy 的开关**。

**为什么减寄存器可能降性能**：提了 occupancy,但可能装不下 accumulator（更多访存/重算）或 spill。register tiling 里寄存器存复用数据的价值常大于 occupancy 收益。

> **本章关键结论**：occupancy 是"能藏延迟的潜力"而非目标。高 occupancy 可能所有 warp 等同一依赖→issue 不足。寄存器多→复用/ILP↑但 occupancy↓,过低会 spill。找甜点靠扫描 + ncu。
> **高频易错**：背"60/70% 一般够"；只看 registers/thread 不看 accumulator 复用价值。

---

# 五、Stream、Event 与 CUDA Graph

## 5.1 Stream 并发

**规则**：同一 stream 内**顺序**执行；不同 stream 只是**允许**并发。真正重叠还取决于：依赖、硬件 copy engine、资源占用、内存类型、任务粒度。

**H2D/D2H 与 kernel 重叠的条件**：pinned（锁页）host 内存 + 异步 API + 不同 stream + 分块 + 正确 event 依赖。

**关键点**：
- **pageable memory 妨碍异步**：`cudaMemcpyAsync` 对 pageable 内存会退化成同步（先拷到内部 pinned 缓冲）。要用 `cudaMallocHost`/`cudaHostAlloc`。
- **default stream（null stream）隐式同步**：会和其他 stream 串行化。把工作放 default stream 破坏并发。
- 两个都占满 SM 的 kernel,即使在不同 stream 也未必并发。

## 5.2 CUDA Event

**能测**：GPU stream 时间线上的设备操作区间。**不能测**：CPU wall time、未建立依赖的其他 stream 工作。

**正确计时**：
- event 记在正确的 stream,等 stop event 完成再读。
- 短 kernel 要 **warmup + 多次重复**,避免首次 JIT/初始化、计时分辨率影响。
- **不能每次 kernel 后 `cudaDeviceSynchronize()` 再谈流水性能**——那会破坏异步/重叠,测到的是串行。
- **多 stream makespan** 用 host 时钟：起点 → 提交所有 stream → `cudaDeviceSynchronize()` → 终点。单 event 测不了整体 makespan。

## 5.3 CUDA Graph

**解决什么**：把重复的 launch/依赖序列实例化并重放,主要降低 **CPU launch 和调度开销**。**不改变**单个 kernel 的 FLOP/bytes（kernel 本身不会更快）。

**三阶段付费**：
- **capture**：记录序列成图（一次性）
- **instantiate**：编译成可执行 graph（一次性,较贵）
- **launch**：重放（每次便宜）
→ benchmark 要**分开报"首次(含 instantiate)"和"稳态(只 launch)"**。

**适合**：结构稳定、重复次数多、kernel 较短的流程（如 decode 循环的一堆小 kernel）。

**限制**：
- **graph update 只能改节点参数,不能改拓扑**（增删节点/改依赖）。
- 动态 shape、KV block table 每步变、条件控制会增加复杂度（要预分配/bucket/多 graph）。
- 大 kernel 主导时 launch 节省比例小。
- **不能消除数据依赖造成的 GPU wait**。

> **本章关键结论**：异 stream 只是"允许"并发,真重叠要 pinned+异步+依赖正确。event 测设备区间,别拿它当 wall time。Graph 省 launch 开销,不改单 kernel 速度,分首次/稳态测。
> **高频易错**：① "放两个 stream 就并行" ② 每 kernel 后 sync 谈流水 ③ "用了 Graph kernel 就更快"。

---

# 六、并行算法模式

## 6.1 Reduction（归约）

**优化链**：寄存器局部聚合 → warp shuffle 归约 → warp 结果经 shared 合并 → 跨 block 用第二 kernel 或少量 atomic。

**细节**：
- **grid-stride** 让每线程处理多个元素（提高每线程工作量,减 block 数）。
- **warp shuffle**（`__shfl_down_sync`,偏移 16→8→4→2→1）完成 32 lane 归约,不用 shared。
- **纯 shuffle 不能跨 warp**——shuffle 只在一个 warp 内。跨 warp 要：warp leader 写 shared → 第一个 warp 读回再 shuffle。shared 是 warp 间的桥。
- **partial warp 的 mask 要正确**,不能无条件 `0xffffffff`。
- **浮点 reduction 顺序影响结果**：浮点加法不满足结合律,不同归约顺序舍入路径不同 → 结果有微小差异 → 对拍用相对误差容差。

**实证**：week03 完整 reduction 追平 CUB ~91% 带宽峰值（但不能据此说所有 shape 都追平）。

## 6.2 Scan（前缀和）

**为什么比 reduction 难**：reduction 只要一个总结果,scan 要**保留每个前缀结果**；多 block 时必须处理 block sums、扫描 block sums、再把偏移加回。

**结构**：
- warp 内用 `__shfl_up_sync`
- block 内先扫各 warp,再扫 warp sums
- grid 级常用**三阶段 kernel**（kernel1 算 block 内 + block sums,kernel2 扫 block sums,kernel3 加回）——靠 **kernel 边界做全局同步**
- 数据更大时 block sums 还需递归

**算法复杂度**：
- **Hillis-Steele**：O(log n) 步但总工作 O(n log n)（work-inefficient）,简单
- **Blelloch（work-efficient）**：up-sweep + down-sweep,总工作 O(n),复杂

**应用——stream compaction**：对"是否保留"的 0/1 标志做 **exclusive scan** → 前缀和就是每个保留元素的目标下标 → 散射写出。

**易错**：inclusive/exclusive 的偏移、非整除边界。

## 6.3 Histogram（直方图）

**降 atomic 争用**：block/warp 私有化 bins（shared privatization）→ 近层级聚合 → 少量合并到 global。

**收益取决于数据分布**：
- **集中分布**：少数 bin 热点严重 → global atomic 争用大 → shared privatization 收益大（week03 实测 ~12×）。
- **均匀分布**：争用本就分散 → privatization 的初始化/合并开销可能抵消收益。

**bin 超过 shared 容量**：分段处理（多趟,每趟一部分 bin）、部分 bin 走 global、或排序聚合。

> **本章关键结论**：三个模式都是"先片上聚合,再逐层合并,最后少量 global"。reduction 靠 shuffle+shared,scan 多一步 block sums 处理,histogram 靠 privatization 且收益依赖分布。跨 block 全局同步靠 kernel 边界。
> **高频易错**：无条件 `0xffffffff` mask；以为一个 kernel 能同步任意多 block；以为 shared atomic 永远比 global 快。

---

# 七、GEMM 优化阶梯

GEMM 是性能工程的天花板载体。优化阶梯（每级解决上一级暴露的新瓶颈）：

## 7.1 Naive → Shared Tiling

**Naive 为什么慢**：每个输出元素重复从 global 读整行 A + 整列 B,**算术强度（AI = FLOP/bytes）极低** → memory-bound。

**Shared Tiling 解决什么**：把 `BM×BN×BK` 的 A/B tile 协作搬到 shared,让 block 内多个输出**复用**。global bytes 从 O(MNK) 降到约 `2MNK/BK`,AI 提高约 BK 倍。

**代价/新瓶颈**：shared 带宽、bank conflict、barrier、边界、布局。**global 瓶颈消失后,可能撞上 shared 带宽或指令吞吐。**

## 7.2 Register Tiling（1D/2D）

**2D register tiling**：每线程算 `TM×TN` 个输出,用 `regM[TM] × regN[TN]` **外积**更新多个 accumulator。
- **1D**：单向复用（A 或 B 之一）
- **2D**：A 和 B **都复用**——一次读 TM 个 A + TN 个 B,产生 TM×TN 个乘加,复用度最高,减少每 FLOP 对 shared 的读取,提高 ILP。

**代价**：accumulator 数增大 → 寄存器压力 → occupancy 下降。**最佳 tile 不是越大越好**,要结合 shared 布局、线程数、grid 规模、寄存器限制**扫描**。

**为什么 A100 上降 TN 可能更快**：TN 大→寄存器压力大→occupancy 低/spill；降 TN 减压,若此时 occupancy/latency 受限就更快；但降太多丢复用（AI 降）→ 有甜点（week05 找到 8×4）。

## 7.3 更高级（通往 cuBLAS/CUTLASS）

shared tiled 仍**远慢于 cuBLAS**,还差：向量化访存、**double buffering（cp.async 藏延迟）**、**Tensor Core**、shared swizzle 避 bank conflict、warp tiling、epilogue 优化。

## 7.4 系统评价一个 GEMM 版本

顺序：**验证数值 → benchmark → FLOP/bytes/AI 账本 → ptxas/SASS/ncu 看瓶颈和资源代价 → 与 cuBLAS 同 shape/精度对比**。

记录：GPU、CUDA 版本、M/N/K、layout、dtype、warmup/repeat、时间、GFLOPS、误差、register/shared、occupancy、关键 pipe。

**注意**：
- 大方阵单点成绩不代表非方阵/小矩阵/边界。
- ncu replay 时间不代表真实性能。
- **Roofline 解释不了 shared bottleneck**：Roofline 只建模 DRAM 带宽 vs 计算峰值。tiled 后 DRAM 不是墙,但卡在 shared 带宽/bank conflict/指令/延迟——这些是片上瓶颈,Roofline 图看不出,要 ncu 的 SOL/pipe。
- **算法 FLOP（2MNK）≠ Tensor Core 实际指令吞吐**（padding、非满 tile、精度转换让实际指令多）。

> **本章关键结论**：GEMM 阶梯 naive→shared tiling(提 AI)→register tiling(A/B 双复用+ILP,但压 occupancy,找甜点)→向量化/双缓冲/Tensor Core。评价要"数值+benchmark+账本+ncu+cuBLAS 对标",Roofline 看不出片上瓶颈。
> **高频易错**：① "用了 shared 就 compute-bound" ② 只用 occupancy 下降否定 register tiling ③ 用不同精度的 cuBLAS 峰值当对照。

---

# 八、Tensor Core 与底层 MMA

## 8.1 概念层级

- **Tensor Core**：矩阵乘加**硬件**。
- **WMMA**：CUDA C++ 的 **warp matrix API**（高层封装）。
- **`mma.sync`**：**PTX warp 级指令**语义（更底层,工业内核用它 + `ldmatrix`）。
- **HMMA**：某些架构反汇编里的半精度矩阵**机器指令**。

**编译链非一一对应**：一次 WMMA 调用可能 lower 成多条 PTX/SASS。**证明走 Tensor Core** 要组合四层证据：源码 API + SASS 矩阵指令(HMMA/MMA) + Tensor pipe 指标 + 性能量级。

**精度**：FP16 输入 + FP32 累加。**误差来自 FP16 输入本身的表示精度**（10 位尾数）,不是累加——累加用 FP32 正是为了减误差。

**为什么 Tensor Core kernel 仍可能 memory-bound**：算得快但**喂数据跟不上**（global→shared→fragment 搬运带宽/bank conflict/ldmatrix 供数不足）。week06 见 Tensor pipe 只 ~5.7% active → 瓶颈在供数,下一步先解决"怎么喂"而非堆 MMA。

## 8.2 ldmatrix 与 fragment

- **fragment**：warp **分布式持有**的矩阵 operand。**一个 lane 不持有完整行**,元素映射依赖具体 shape/type/layout。
- **`ldmatrix`**：协作地把 **shared tile 加载成 fragment**（shared→寄存器,**不是** global→shared）。`.m8n8.x1/x2/x4` 描述同时加载的矩阵数,`.trans` 用于转置形式。

**关键**：不同 MMA shape 有**不同的 lane→寄存器→元素映射**,不能把一个 shape 的 lane map 套到另一个 shape（读到错元素）。shared 地址、对齐、swizzle、所有 lane 一致参与都必须正确。

## 8.3 CUTLASS/CuTe 三级 tile

- **threadblock tile**：决定 CTA 数据复用
- **warp tile**：分配 CTA 内工作
- **instruction tile**：对应 MMA/FMA 基本操作

多层布局把 global/shared/register/指令形状连接起来。工业 GEMM 还需 **mainloop pipeline、shared swizzle、epilogue、边界、架构特化**。

- **swizzle 要同时满足 bank（避冲突）和 ldmatrix（特定访问 pattern）** 两个约束。
- **epilogue 也可能成瓶颈**（accumulator 写回 + bias/激活/scale/转精度 是大量 global store + 计算）。

> **本章关键结论**：Tensor Core 是硬件,WMMA/mma.sync/HMMA 是不同抽象层。fragment 是 warp 分布式持有,ldmatrix 做 shared→寄存器。证明走 TC 要四层证据。TC kernel 常 memory-bound(喂数瓶颈)。
> **高频易错**：① "调 WMMA 就近峰值" ② 把 fragment 当连续 C 数组 ③ ldmatrix 当 global→shared ④ 套错 MMA shape 的 lane map。

---

# 九、cp.async 与多级流水

## 9.1 cp.async

Ampere 起支持 **global→shared 异步复制**,避免显式经通用寄存器中转,让搬运和计算重叠。

**三件套**：
- `cp.async`（发起异步拷贝,`.ca/.cg` 是缓存提示）
- `commit_group`（把刚发起的一批打包成 group）
- `wait_group N`（等到**只剩最近 N 个 group** 未完成）

**`wait_group 1` 的直觉**：保留 1 个 group 在飞、其余等完——双缓冲：算当前 tile 前确保它搬完,同时下一 tile 预取继续在飞。`wait_group 0` = 全等完。

**等待职责分离**：`cp.async` 的 completion（wait_group）和 CTA 消费同步（barrier/syncthreads）是**不同职责**。普通 `__syncthreads()` **不能替代** async copy 的专用 completion。

**尾部 tile zero fill**：K 非整除时,越界部分不能读 global,要在 shared 填 0（跳过越界拷贝并写 0）,保证乘法时贡献 0。

## 9.2 多级流水（2-stage / 3-stage）

**环形 slot 生命周期**：`FREE → COPYING → READY → CONSUMING → FREE`。producer 覆盖一个 slot 前,必须等它变回 FREE（消费者读完）——靠 barrier/wait_group 保证不覆盖仍在读的 slot。

**3-stage 不一定比 2-stage 快**：更深 stage 能藏更长延迟,但增加 shared/CTA 占用、同步、prologue/epilogue 成本,甚至降 occupancy。**若原本不是 latency-bound,加 stage 无效。**

**怎么证明收益来自 overlap**（不能只看时间）：
- ncu 看 **eligible warp/issue 效率** 是否提升
- **long scoreboard stall 是否下降**（访存延迟被藏）
- 对比 sync/2/3-stage 的 GFLOPS + shared + occupancy + eligible warp

**你的实证**：week04 attention pipelined,long_scoreboard 从每指令 6.4 cycle 降到 0.03（-99.5%）,short_scoreboard 不变（0.60→0.59）——"一降一稳"精确证明 cp.async 藏的是 global 延迟。

> **本章关键结论**：cp.async 做 global→shared 异步,commit/wait_group 管完成(与消费同步是两回事)。多级流水靠环形 slot,3-stage 不一定更快(非 latency-bound 时无效)。证明 overlap 要看 long_scoreboard/eligible warp。
> **高频易错**：① "发完 cp.async 算几条指令数据就好了" ② 用 syncthreads 替代 wait_group ③ 只报时间不报资源代价。

---

# 十、PTX、SASS 与编译链

## 10.1 编译链

`CUDA C++ → NVVM/PTX → ptxas → cubin`；fatbin 可携带多个 cubin 和 PTX。

- **PTX**：**虚拟 ISA**,由 ptxas 或 driver JIT 针对目标架构生成机器码。PTX 虚拟寄存器 ≠ 物理寄存器。
- **SASS**：特定 GPU 的**机器码反汇编**,最接近执行事实。
- **`compute_80`**（虚拟架构,编译到 PTX）vs **`sm_80`**（真实架构,编译到 SASS/cubin）。
- **为什么同时放 cubin + PTX**：cubin 直接跑、启动快但只对特定架构；PTX 能被 driver JIT 到未来架构（向前兼容）。已知架构用 cubin,未知 fallback PTX。

**工具**：
- `nvcc -ptx` 看语义
- `-Xptxas=-v` 看资源（registers/spill）
- `cuobjdump --dump-sass` 看机器指令

## 10.2 用汇编证明优化

**先写预期指令变化,再对比同编译条件下的 SASS/ptxas/ncu**。不能只凭源码有 `float4`/WMMA/unroll 就下结论。

| 优化 | SASS 里看什么 |
|------|--------------|
| 向量化 | 实际 load 宽度和数量（`LDG.E.128`） |
| WMMA/Tensor Core | HMMA/MMA 指令 |
| register tiling | 独立 FFMA 和寄存器数 |
| spill | ptxas 报告 + local load/store（`LDL/STL`）+ runtime local traffic |

**关键**：机器码证明"生成了什么",ncu/时间证明"是否有价值"。

**两个易错**：
- **SASS 不能单独证明 bank conflict**——bank conflict 是运行时行为（取决于各 lane 实际地址）,SASS 只是静态指令,要 ncu 指标。
- **`-G`（device debug）构建不能做性能汇编分析**——它关优化、插调试信息,SASS 和真实优化版完全不同。用 `-O3 -lineinfo`。

> **本章关键结论**：PTX 是虚拟 ISA(可 JIT),SASS 是真机器码。compute_80→PTX,sm_80→cubin。证明优化要 SASS+ncu 双证据(机器码证"生成什么",ncu 证"有没有用"),别 grep 到一条就下结论。
> **高频易错**：① "GPU 逐条执行 .ptx 文本" ② SASS 单独证 bank conflict ③ 用 -G 做性能分析。

---

# 十一、Profiling、Roofline 与 Stall

## 11.1 三个工具分工

- **nsys**：系统时间线,看**并发、大时间块、launch gap、CPU/GPU 时间线、同步**（多 kernel/整个循环）。
- **ncu**：单 kernel 微观硬件行为,看 **SOL、memory workload、occupancy、scheduler、source counters**。
- **Roofline**：用 AI 和峰值给出**带宽/计算上限**,但**不建模所有片上瓶颈**。

**诊断顺序**：可靠 benchmark 确认问题 → nsys 定位大时间块/launch/同步 → ncu 看 SOL/memory/occupancy/scheduler → 综合判断。

**注意**：
- **ncu replay 时间不能代替 benchmark**——ncu 多次重放+插桩,严重拖慢（GEMV 0.05ms→14ms）。ncu 看指标,不看绝对时间。
- **AI 高不必然 compute-bound**——前提是其他资源没先成墙。可能卡在 shared/bank/指令/延迟/occupancy。
- **Roofline 显示 DRAM 不是墙,仍需看** shared、指令、latency、divergence、occupancy。

## 11.2 Warp 状态与 Scoreboard

- **active warp**：已驻留
- **eligible warp**：下一条指令依赖已就绪（能发射）
- **issued warp**：实际发射
- **Scoreboard**：跟踪数据依赖,防依赖未就绪时发射

**active 多但 eligible 少** = 很多 warp 同时等待。可用更多 TLP、增加 ILP、预取、减延迟,但**先定位依赖**。

**为什么 occupancy 高仍 issue 不足**：驻留 warp 多,但全在 stall（如一起等 global）→ 某拍无 eligible warp → 无指令可发。

## 11.3 Stall 分析

- **Long Scoreboard 高**：warp 等 **L1TEX 路径**依赖（global/local/texture/surface）,**不是"DRAM 慢"的同义词**。先看 source/SASS 对应 load,再查 cache/DRAM、local spill、load-use 距离、eligible warp。
- **Short Scoreboard**：常与 **shared memory** 访问依赖（及 MUFU）有关。
- **Not Selected 高** 反而可能说明 eligible warp 充足。

**形成可证伪的 stall 假设**：定位哪条 load（source counters/SASS 行号）→ 提假设（"这条未合并 load 导致 long scoreboard"）→ 给可验证预测（"改合并后 long scoreboard 应降、sector 数应减"）→ 改完 ncu 复测证实/证伪。

> **本章关键结论**：nsys 看系统时间线,ncu 看单 kernel 微观,Roofline 给上限但看不出片上瓶颈。ncu replay 时间≠真实性能。stall 分析要定位到具体 load + 可证伪假设。long scoreboard=等 L1TEX(不只 DRAM)。
> **高频易错**：① DRAM throughput 低就说访存没问题 ② AI 高就 compute-bound ③ long scoreboard 高就直接搬 shared ④ 把 active warp 数当每拍可发射数。

---

# 十二、调试与工程化

## 12.1 Compute Sanitizer 四工具

| 工具 | 查什么 |
|------|--------|
| **memcheck** | 非法/越界访问 |
| **initcheck** | 未初始化的 device global 读取 |
| **racecheck** | **shared** data hazard（缺同步的读写冲突） |
| **synccheck** | 同步原语使用错误 |

**顺序**：先 memcheck（越界会制造噪声）,再 initcheck/racecheck/synccheck。

- **racecheck 主要关注 shared**：shared 的 race 有明确"缺 syncthreads"模式,可检测；global race 更难静态界定,靠算法协议。
- **工具没报错 ≠ 算法正确**：错误 async stage、浮点误差、遗漏元素仍需 **CPU/reference 对拍 + 边界测试 + 重复压力测试**。
- **cuda-gdb**：sanitizer 报"有错"但定位不到逻辑原因时,或需要单步/看变量/下断点追复杂 bug。sanitizer 查"有没有错",cuda-gdb 查"为什么、在哪一步"。

## 12.2 可信的 benchmark 与回归测试

**性能**：固定环境和输入,warmup + 多次重复,正确时间域,记录编译/硬件/shape,性能阈值与噪声区分。记 median/percentile,避免首次 JIT/动态频率/其他负载/profiler replay。

**正确性覆盖**：小尺寸、非整除、极值、随机种子、NaN/Inf、容差。

- **浮点不能只用绝对误差**：值大时正常舍入的绝对误差也大（误判 FAIL）,值近 0 时相对可能很大。用**相对误差**（或结合）,阈值按 fp32/fp16 累加特性设（GEMV K=4096 放宽到 1e-3）。
- **避免编译器优化掉 microbenchmark**：结果没被用会 dead code elimination。要把结果写 global/volatile/校验,输入运行时才知道。

## 12.3 错误检查与 RAII

- CUDA API 和 kernel launch 都可能**异步失败**。launch 后 `cudaGetLastError()` 查配置/启动错误；需确认执行完成时在合适边界同步再检查。
- **`cudaGetLastError` 清除错误状态；`cudaPeekAtLastError` 只看不清**。
- device buffer/stream/event 用 **move-only RAII** 封装。**析构函数不抛异常**——里面的 CUDA 错误记录日志/吞掉,不 throw。
- **不要每个 launch 后全局同步**（破坏异步性能）。

> **本章关键结论**：sanitizer 四工具(memcheck/init/race/sync),racecheck 主查 shared,没报错≠正确(仍要 CPU 对拍)。benchmark 要 warmup+重复+相对误差+防 DCE。RAII 封装资源,析构不抛异常,getLastError 会清状态。
> **高频易错**：① 只跑 memcheck 就说并发正确性全验证 ② 浮点只用绝对误差 ③ 每 launch 后全局同步。

---

# 十三、多 GPU 基础

## 13.1 互连与通信

- **PCIe/NVLink**：互连**路径**。
- **P2P**：允许 GPU 直接访问/复制对端显存。
- **NCCL**：在拓扑上实现**集合通信**（AllReduce/AllGather/...）。

**带宽和路径取决于拓扑,不由 API 名保证**：
- 先用拓扑工具/`cudaDeviceCanAccessPeer` 确认 peer access 和链路。
- **P2P 不可用时绕道 host**：GPU0 → host → GPU1（staging）,慢。
- **标称 NVLink 带宽 ≠ 有效带宽**：受协议开销、消息大小、拓扑（是否经 NVSwitch）、争用影响。用 nccl-tests 的 busbw 实测。

## 13.2 集合通信

**AllReduce = ReduceScatter + AllGather**（适合带宽高效的环形实现）：
- **ReduceScatter**：各 rank 得到归约结果的一段
- **AllGather**：交换各段得到完整结果

**算法由拓扑/rank 数/消息大小/协议决定**,不能说 NCCL 永远只用 ring。通信量要区分：每 rank 注入字节、链路流量、总系统流量。

- **AllGather vs All-to-All**：AllGather 大家拿到全部段的拼接（结果相同）；All-to-All 每 rank 给每个 rank 发不同段（结果不同,类似转置）。
- **小消息 latency-bound**：固定开销（启动延迟、握手、同步）占主导,增带宽无用,靠合并小消息/减通信次数。

计算通信重叠要：分块 + stream/event 依赖 + 足够独立工作。

> **本章关键结论**：PCIe/NVLink 是路径,P2P 是直连能力,NCCL 是集合通信。带宽看拓扑不看 API 名。AllReduce=ReduceScatter+AllGather。小消息 latency-bound。
> **高频易错**：① "多卡调 NCCL 就线性扩展" ② 把数学 collective 与固定拓扑算法绑定 ③ 标称带宽当有效带宽。

---

# 十四、Hopper（架构演进认知）

> A100 是 sm_80,跑不了 Hopper 特性。以下是**架构理解**,能讲清"解决什么问题"即可,不声称实测。

## 14.1 A100 → Hopper 的对应

| A100（Ampere） | Hopper | 变化/解决什么 |
|---------------|--------|--------------|
| `cp.async`（多线程给地址） | **TMA** | descriptor+坐标发 bulk 多维搬运,卸载 per-thread 地址生成给专用引擎 |
| warp `mma.sync` | **WGMMA** | 4-warp group 异步协作 MMA |
| CTA | **Thread Block Cluster** | 一组 CTA 在同一 GPC 协同调度 |
| 本 CTA shared | **DSM（分布式 shared）** | cluster 内 CTA 访问彼此 shared 分区 |
| 普通阶段同步 | **mbarrier/transaction** | 把异步事务完成纳入 stage 状态 |

## 14.2 关键点

- **TMA 适合 warp specialization**：少量线程用 descriptor 就能发整块搬运,可让少数 producer 专职发 TMA、consumer 专职 WGMMA。
- **expected transaction bytes**：mbarrier 支持事务计数,TMA 搬完的字节累加到 barrier,达预期字节才放行——精确知道"数据到齐",适配 bulk 异步。
- **WGMMA** 有自己的 fence/commit/wait 和 operand 规则,**不能套 Ampere lane map**。
- **DSM 远端访问 ≠ 本地 shared 延迟**；被远端访问 shared 的 CTA **不能提前退出**（否则读到无效内存,cluster sync 协调生命周期）。
- **producer/consumer warp-group 分工**：producer 用 TMA 搬 global→shared 发 mbarrier；consumer 等 mbarrier 用 WGMMA 算,算完通知 producer 覆盖。这是 Hopper GEMM/FlashAttention-3 核心范式。

## 14.3 Persistent Kernel 与 Warp Specialization

- **Persistent kernel**：有限 CTA 常驻,从任务队列取工作,减 launch、支持动态负载均衡。常驻 CTA 数选"恰好占满所有 SM 的可驻留上限"。动态队列避免热点 atomic：批量领取/分片队列/warp-aggregated。
- **Warp specialization**：不同 warp 固定负责搬运/计算/epilogue,形成更深流水。Hopper TMA/WGMMA 让角色分工更自然,但协议更复杂。代价：调度公平、队列同步、资源静态占用、尾部负载、调试复杂度。

> **本章关键结论**：Hopper 用 TMA(bulk 异步搬运)+WGMMA(warp-group MMA)+Cluster/DSM(跨 CTA 协作)+mbarrier(事务同步)催生 producer/consumer warp specialization。A100 只能读源码理解,不实测。
> **高频易错**：① "TMA 只是更宽的 cp.async" ② 把 Ampere lane map 套 WGMMA ③ "Hopper 特性全开就最快"。

---

# 十五、复习方法与答题框架

## 15.1 答题通用套路

> **先给结论（15 秒）→ 再给条件和边界（什么情况成立/不成立）→ 最后用代码/profiler 证据支撑。**

## 15.2 必须避免的话术（会扣分）

- "永远/一定/默认都这样" 回答架构问题（scheduler 数、occupancy 阈值随架构变）
- 只报时间不报资源代价（reg/shared/occupancy）
- 只测一个大方阵就当通用性能
- grep 到一条 float4/HMMA 就说 kernel 优化好了
- 把数学 collective 与固定拓扑算法绑定

## 15.3 性能岗核心链条（背熟）

> 给个慢 kernel → benchmark 确认问题 → Roofline 定位 compute/memory → nsys 看大时间块/launch → ncu 看 SOL/stall/occupancy → 提**可证伪假设** → 改 → 复测。

## 15.4 你的项目证据库（面试用真实数据说话）

| 项目 | 可引用的数据 |
|------|-------------|
| GEMV | 合并访问 72→1360 GB/s（19×）；float4 只 +4% 因已 memory-bound（DRAM 72%/Compute 10%） |
| attention pipelined | long_scoreboard 6.4→0.03（-99.5%）、short 不变 → 证明 cp.async 藏 global 延迟 |
| GEMM | bank conflict 4.8→1.5 way；register tiling 8×4 甜点 |
| reduction | 追平 CUB ~91% 带宽峰值 |

## 15.5 核心公式速查

- **算术强度** AI = FLOP / bytes
- **GEMM**：FLOP ≈ 2MNK
- **GEMV**：FLOP ≈ 2NK,主要 bytes ≈ NK×dtype（W 主导）
- **GB/s** = bytes / 秒 / 1e9；**GFLOPS** = FLOP / 秒 / 1e9
- **KV cache**（LLM）：2×L×B×N×Hkv×Dh×dtype_bytes

## 15.6 自检清单

- [ ] 每个大主题能"先结论、再条件、后证据"讲清
- [ ] 闭卷写出 reduction / transpose / tiled GEMM
- [ ] 讲完 A100 GEMM 的"假设→profile→修复→复测"
- [ ] 会读 SASS 认出 LDG.E.128 / HMMA / spill(LDL/STL)
- [ ] 能用自己项目的真实数据支撑每个结论
- [ ] 不用固定 scheduler 数 / occupancy 阈值等跨架构绝对话术

---

> 这份是知识主线的系统梳理；题目练习和"追问速查"见 [八股全集](../../interview/CUDA面试核心题库.md) 与 [追问答案](../../interview/CUDA面试核心题库.md)。
> 复习路径：先通读本篇建立体系 → 用八股题目自测 → 卡住处回本篇对应章节精读 → 用你的项目数据串成"证据链"。
