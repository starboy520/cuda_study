# CUDA 面试八股 · 追问答案速查（配套）

> 配套：[CUDA面试八股全集.md](../../interview/CUDA面试核心题库.md)
> 用途：主答案(15秒/1分钟)原文已有；这份专门答**每题下面的"面试官继续追问"**——复习时最容易卡的地方。
> 读法：先自己答，再对照。每条尽量 1-3 句抓要害。
> 本篇覆盖第 1-7 章（硬件/内存/同步/occupancy/stream/并行算法/GEMM）。第 8-15 章见续篇。

---

# 第 1 章：GPU 硬件与执行模型

## Q1 CPU vs GPU 的追问

**1. latency hiding 和降低 latency 有什么区别？**
- **降低 latency**：让单次操作本身更快（CPU 的路子：大 cache、乱序、分支预测）。
- **latency hiding（隐藏延迟）**：单次操作还是慢，但某个 warp 等待时**调度别的 ready warp** 上来算，让执行单元不空转（GPU 的路子）。GPU 不是让 global load 变快，而是用大量并行 warp 把等待"盖住"。

**2. 为什么小 kernel 可能被 launch 开销主导？**
- 每次 kernel launch 有固定的 CPU 提交 + GPU 调度开销（微秒级）。如果 kernel 本身只跑几微秒，launch 开销就占了大头，GPU 计算再快也没用。这也是 CUDA Graph（合并 launch）和 kernel fusion 的动机。

## Q2 GPC/SM/warp scheduler 的追问

**1. block 在生命周期中会迁移 SM 吗？**
- **不会**。block 一旦被分配到某个 SM，就在那儿驻留到执行完，不会中途迁移。所以 block 内可以用 shared memory / `__syncthreads()`（它们是 SM 本地的）。

**2. 为什么同一条 warp 指令可能占用不同执行管线？**
- SM 内有多类执行单元（CUDA Core 算 FP32/INT、SFU 算超越函数、Tensor Core 算矩阵、LSU 访存）。不同指令类型（FFMA vs MUFU vs LDG vs HMMA）被派发到不同管线，scheduler 每拍从 eligible warp 里挑指令发到对应端口。

## Q3 SIMT vs SIMD 的追问

**1. independent thread scheduling 改变了什么？**
- Volta 后每个线程有独立 PC 和调用栈，warp 内分歧的线程能**各自推进、交错执行**，还允许分歧线程间做更细的同步。但**不代表分歧免费**——分歧路径仍要分阶段执行，也不能再依赖旧的"隐式 warp 同步"。

**2. 为什么旧式"隐式 warp 同步"代码可能出错？**
- 老代码假设"warp 内 32 线程锁步执行"，省掉了 `__syncwarp()`。Volta 独立调度后这个假设不成立，线程可能不再锁步，共享数据时不加 `__syncwarp` 会读到旧值。必须显式同步。

## Q4 warp divergence 的追问

**1. 分支条件按 `threadIdx.x/32` 划分会 divergence 吗？**
- **不会**。`threadIdx.x/32`（warp_id）在一个 warp 的 32 lane 里值完全相同，分支对整个 warp 一致 → 不分裂。判据：**分支条件在 warp 内是否一致**（warp_id/blockIdx 一致→不 divergence；lane_id/线程级边界/数据依赖→可能 divergence）。

**2. predication 一定更快吗？**
- 不一定。predication 把两个分支都算、再用条件选，**避免了分支跳转但两边都执行**。分支短、工作量小时划算；分支长、工作量大时，把两边都算反而更慢。要实测。

---

# 第 2 章：内存层次与访问

## Q5 各级内存的追问

**1. local memory 一定来自 spill 吗？**
- 不一定。除了寄存器溢出（spill），**动态索引的局部数组**（编译器无法确定下标）、**大的局部数组**也会放到 local memory。local 物理上在片外（global 空间），经 cache，不是片上寄存器。

**2. shared 为什么既快又可能成为瓶颈？**
- **快**：片上内存，低延迟高带宽。**瓶颈**：① 它是每 SM 有限资源，一个 block 用太多 → 驻留 block 变少 → **occupancy 下降**（不是回退 global，是吃 occupancy）；② **bank conflict**——warp 内访问同一 bank 不同地址会串行化。注意 shared 不会 spill，会 spill 的是寄存器。

## Q6 coalescing 的追问

**1. Array-of-Structs 什么时候会浪费事务？**
- 当 **warp 各线程只访问结构体的部分字段**时。字段交错存放，取单个字段变成跨步访问（stride=sizeof(struct)）→ 不合并 → 每个事务搬回一堆用不上的其他字段。解法：改 SoA（每字段各自连续）。整个结构体都用时 AoS 影响小。

**2. 转置为何常出现合并读、跨步写的冲突？**
- 转置要 `out[j][i] = in[i][j]`。若按行读 `in`（连续，合并），那写 `out` 就是按列（跨步，不合并）；反之亦然。**读和写的合并方向天生冲突**。解法：用 shared memory 中转（合并读进 shared，再合并写出），配 `[32][33]` padding 避免 bank conflict。

## Q7 bank conflict 的追问

**1. 为什么同 bank 同地址可以广播？**
- 硬件对 shared 有广播机制：一个 warp 里多个 lane 读**完全相同的地址**时，一次访问把值广播给所有请求的 lane，不算冲突。冲突只发生在"同 bank **不同**地址"（要分多波服务）。

**2. 如何用 ncu 证明冲突而不是只靠推导？**
- 看 ncu 的 shared memory 相关指标，如 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` 或 bank conflict 的 way 数 / wavefront 数。你 week05 就是从 4.8-way 降到约 1.5-way 证明的。光推导不够，要指标背书。

## Q8 float4 的追问

**1. `reinterpret_cast<float4*>` 需要哪些前提？**
- ① **16 字节对齐**（地址 %16==0；cudaMalloc 基址够，但加偏移 `row*K` 要 K%4==0 才保持）② **元素个数 4 的倍数**（否则单独处理尾部，防越界）③ **内存连续**（那 4 个 float 物理相邻，transpose/AoS 不行）④ **不越界**。

**2. 为什么宽 load 仍可能产生多个 memory sector？**
- 一条 `LDG.E.128` 是"每个线程读 16 字节"，但一个 warp 32 线程 × 16 字节 = 512 字节，跨多个 cache sector（每 sector 32 字节）。宽 load 减少的是**指令数**，但实际 DRAM sector 数由 warp 总访问范围和对齐决定，不是"一条指令一个 sector"。

---

# 第 3 章：同步、原子与一致性

## Q9 syncthreads/syncwarp/fence 的追问

**1. 为什么"写 data→threadfence→写 flag"仍需要消费者正确读取 flag？**
- `__threadfence()` 只保证**生产者自己**的内存操作顺序（data 的写先于 flag 的写对其他线程可见），它**不通知消费者、也不等消费者**。消费者必须**主动轮询 flag**（且用 volatile/atomic 正确读，防编译器缓存），看到 flag 置位后再读 data。fence 只管顺序，不管"通知"。

**2. `__syncwarp(mask)` 的 mask 为什么必须准确？**
- mask 标明"哪些 lane 参与这次同步/collective"。如果有 lane 已经分支走了却仍在 mask 里，或该参与的没在 mask 里，会导致未定义行为/挂起。分歧后做 shuffle/syncwarp，mask 必须精确反映当前活跃的 lane。

## Q10 atomic 的追问

**1. shared atomic 与 global atomic 的差异？**
- **shared atomic**：在片上 shared 上做，延迟低；适合 block 内先聚合。**global atomic**：在片外 global 上，延迟高、争用重。典型优化：寄存器局部聚合 → shared atomic 聚合 → 少量 global atomic 提交，逐层减少争用。

**2. warp-aggregated atomic 如何减少请求？**
- 一个 warp 里多个 lane 要对**同一地址** atomicAdd 时，先在 warp 内用 shuffle/ballot 把这些增量**合并成一个总和**，只让 leader lane 发**一次** global atomic。把 32 次原子请求压成 1 次，大幅减少争用。

## Q11 cooperative groups 的追问

**1. 为什么普通 kernel 中用 global counter 自旋等所有 block 可能死锁？**
- 普通 kernel 不保证所有 block **同时驻留**。如果已运行的 block 自旋等待"还没被调度上来的 block"，而这些运行中的 block 又占满了 SM，新 block 永远上不来 → **死锁**。所以不能假设 block 并发存在。

**2. 多 kernel scan 为什么自然获得全局阶段边界？**
- **kernel 之间有隐式全局同步**：第二个 kernel 一定在第一个 kernel 全部 block 执行完后才开始。所以用"kernel 边界"分阶段（kernel1 算 block 内 + block sums，kernel2 扫 block sums，kernel3 加回），天然获得可靠的全局同步，不需要 grid sync。

---

# 第 4 章：Occupancy、寄存器与延迟隐藏

## Q12 occupancy 的追问

**1. 理论 occupancy 与 achieved occupancy 区别？**
- **理论 occupancy**：由资源（registers/thread、shared/block、threads/block）算出的**上限**（能驻留多少 warp）。**achieved occupancy**：运行时**实际**平均驻留的 warp 比例，受尾部效应、block 调度不均、执行时长差异影响，通常低于理论值。

**2. 为什么减寄存器可能降低性能？**
- 减寄存器能提高 occupancy（更多 warp 驻留），但可能：① 装不下 accumulator/复用数据 → 更多访存或重算；② 寄存器不够 → **spill 到 local memory**（片外，慢）。register tiling 里寄存器存 accumulator 的复用价值往往大于 occupancy 收益。

## Q13 寄存器/spill/ILP/TLP 的追问

**1. 动态索引局部数组与 register spill 怎样区分？**
- 两者都用 local memory，但成因不同：**动态索引局部数组**是编译器无法把下标不定的数组放寄存器，主动放 local；**spill** 是寄存器用超了被迫溢出。区分：看 SASS 的 local load/store 对应源码位置 + `-Xptxas=-v` 的 spill 报告。

**2. 为什么 occupancy 上升后性能可能下降？**
- 高 occupancy 可能是靠"减寄存器/减 shared"换来的，代价是丢了数据复用（ILP 下降、访存增多）。如果原本瓶颈不是"warp 不够隐藏延迟"，堆 occupancy 无益反而有害。occupancy 是手段不是目标。

---

# 第 5 章：Stream、Event 与 CUDA Graph

## Q14 stream 并发的追问

**1. pageable memory 为什么妨碍真正异步传输？**
- 异步 H2D/D2H 要求 host 内存是**锁页（pinned）**的，DMA 引擎才能直接搬。pageable 内存可能被 OS 换出，所以 `cudaMemcpyAsync` 对 pageable 内存会**退化成同步**（先拷到内部 pinned 缓冲），无法和 kernel 真正重叠。要用 `cudaMallocHost` / `cudaHostAlloc`。

**2. default stream 语义如何影响并发？**
- 传统 default stream（null stream）是**隐式同步**的：它会和其他 stream 串行化（default stream 的操作等所有 stream 完成，反之亦然）。所以把工作放 default stream 会破坏并发。用非默认 stream，或编译时开 per-thread default stream。

## Q15 CUDA Event 的追问

**1. 为什么不能每次 kernel 后 `cudaDeviceSynchronize()` 再谈流水性能？**
- 每次 sync 会强制 CPU 等 GPU 全部做完，**破坏了异步和 stream 重叠**——你测到的是"串行执行"，掩盖了本可重叠的部分。测流水性能应只在最后同步一次，用 event 记区间。

**2. 如何测多个 stream 的 makespan？**
- makespan = 从第一个操作开始到所有 stream 全部完成的墙钟时间。用 host 时钟：起点前记时间 → 提交所有 stream 工作 → `cudaDeviceSynchronize()`（等全部完成）→ 记结束时间。单个 event 只测它所在 stream 的区间，测不了整体 makespan。

## Q16 CUDA Graph 的追问

**1. capture、instantiate、launch 分别在哪个阶段付费？**
- **capture**：记录 stream 上的操作序列成图（一次性，构建开销）。**instantiate**：把图编译成可执行 graph（一次性，较贵，做地址/依赖解析）。**launch**：重放（每次很便宜，省的就是这块）。所以 benchmark 要分开报"首次(含 instantiate)"和"稳态(只 launch)"。

**2. graph update 有哪些约束？**
- 可以更新节点参数（kernel 参数、memcpy 地址等），但**不能改图的拓扑结构**（增删节点、改依赖）。参数更新也有类型/形状约束。动态 shape、条件分支会让 graph 复用变复杂，可能要多个 graph 或重新 instantiate。

---

# 第 6 章：并行算法模式

## Q17 reduction 的追问

**1. 为什么纯 shuffle 不能直接跨 warp？**
- `__shfl_*_sync` 只能在**一个 warp 内的 32 lane**间交换数据，warp 之间没有 shuffle 通道。所以跨 warp 归约要：每个 warp 先 shuffle 归约 → warp leader 把结果写 shared → 第一个 warp 再从 shared 读回来 shuffle 归约。shared 是 warp 间的桥。

**2. 浮点 reduction 为什么不同顺序结果不同？**
- 浮点加法**不满足结合律**（`(a+b)+c ≠ a+(b+c)`，因为每步都有舍入）。不同的归约顺序（串行 vs 树形 vs 不同线程划分）舍入误差累积路径不同 → 结果有微小差异。所以对拍要用相对误差容差，不能要求 bit 一致。

## Q18 scan 的追问

**1. Hillis-Steele 与 Blelloch 的 work complexity？**
- **Hillis-Steele**：step 少（O(log n) 步），但总工作量 O(n log n)（work-inefficient），实现简单。**Blelloch（work-efficient）**：up-sweep + down-sweep，总工作量 O(n)（work-efficient），但步数和实现更复杂。小数据 Hillis-Steele 够，大数据 Blelloch 省工作量。

**2. Scan 如何用于 stream compaction？**
- stream compaction（把满足条件的元素紧凑排列）：先对"是否保留"的 0/1 标志做 **exclusive scan** → 每个保留元素的前缀和就是它在输出数组里的**目标下标** → 按这个下标散射写出。scan 提供了"我前面有几个保留的"这个偏移。

## Q19 histogram 的追问

**1. bin 超过 shared 容量怎么办？**
- shared 装不下所有 bin 时：① 分段处理（一次只统计一部分 bin 范围，多趟）；② 部分 bin 放 shared、其余走 global atomic；③ 用排序聚合替代原子。取舍看 bin 数和分布。

**2. 为什么均匀分布与集中分布结果不同？**
- **集中分布**：大量元素落进少数 bin → global atomic 热点严重 → shared privatization 收益大（week03 实测 ~12×）。**均匀分布**：原子争用本来就分散 → privatization 的初始化/合并开销可能抵消收益。优化收益强依赖数据分布。

---

# 第 7 章：GEMM 优化阶梯

## Q20 naive vs tiled GEMM 的追问

**1. 如何计算 naive 与 tiled 的 AI？**
- **AI = FLOP / bytes**。GEMM FLOP ≈ 2MNK。**naive**：每个输出元素重复从 global 读整行 A + 整列 B，bytes 巨大 → AI 低。**tiled**：一个 tile 的 A/B 读进 shared 后被 tile 内所有输出复用，global bytes 降到约 `2MNK/BK`（BK 是 tile 的 K 维），AI 提高约 BK 倍。

**2. 为什么 shared tiled 仍远慢于 cuBLAS？**
- shared tiling 解决了 global 访存，但还差：register tiling（减 shared 访问、提 ILP）、向量化、double buffering（cp.async 藏延迟）、Tensor Core、精细的 shared swizzle 避 bank conflict、warp tiling。cuBLAS 这些全做了。shared tiled 只是阶梯第二级。

## Q21 2D register tiling 的追问

**1. 1D 与 2D tiling 分别复用什么？**
- **1D tiling**（每线程 TM 个输出）：复用 **A 的一个片段**（一列 A 配多个 B）或反之，单向复用。**2D tiling**（每线程 TM×TN 个输出）：用 `regM[TM] × regN[TN]` 外积，**A 和 B 都复用**——一次读入 TM 个 A + TN 个 B，产生 TM×TN 个乘加，复用度最高。

**2. 为什么 A100 上降低 TN 可能更快？**
- TN 大 → 每线程 accumulator 多 → 寄存器压力大 → occupancy 下降 / spill。降 TN 减轻寄存器压力、提高 occupancy，若此时是 occupancy/latency 受限，就更快。但降太多又丢复用（AI 下降）→ 有个甜点（week05 找到 8×4）。

## Q22 评价 GEMM 版本的追问

**1. Roofline 为什么解释不了 shared bottleneck？**
- Roofline 只建模 **DRAM 带宽 vs 计算峰值** 两条线。如果 DRAM 不是瓶颈（tiled 后 global 流量已降），但卡在 **shared 带宽 / bank conflict / 指令吞吐 / 延迟**，这些都是**片上**瓶颈，Roofline 图上看不出来——它会显示"离 DRAM 屋顶还远"，但你已经撞了别的墙。要靠 ncu 的 SOL/pipe 指标。

**2. 如何区分算法 FLOP 与 Tensor Core 实际指令吞吐？**
- **算法 FLOP** = 2MNK（数学上的乘加数）。**Tensor Core 指令吞吐** = 实际发射的 HMMA/MMA 指令数 × 每指令算力。两者可能不等：padding、非满 tile、精度转换都让实际指令多于理论。要用 ncu 的 Tensor pipe active % 和 HMMA 指令计数看真实吞吐，不能只算 2MNK。

---

> 第 1-7 章追问答案完。**第 8-15 章**（Tensor Core/ldmatrix、cp.async 多级流水、PTX/SASS、profiling/stall、调试 sanitizer、多 GPU/NCCL、Hopper TMA/WGMMA、手写题评分点、模拟面试）见续篇——确认这个形式合用，我就接着补。

## 复习用法建议
- **B 必会**（Q1-7、9-10、12、14-15、17、20）：合上答案，能 15 秒说出无硬伤结论。
- **A 主线**（Q8、11、13、16、18-19、21-22）：能 1 分钟给证据链 + 一个反例。
- 每章追问先自己答，卡住再看这份——卡住的地方就是你的复习重点。
