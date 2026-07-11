# CUDA 面试核心题库：从基础正确性到 Kernel 性能深水区

> 适用：CUDA/HPC 开发、GPU Kernel、CUDA 性能工程岗位。纯 CUDA、硬件、性能、调试与工程视角；Attention、KV Cache、推理框架等内容见 [AI Infra 面试八股全集](AI_Infra面试核心题库.md)。

## 0. 使用方法

难度标签：

- **B 必会**：常规 CUDA/HPC 岗位基础；
- **A 主线**：Kernel/性能岗位核心；
- **C 加分**：工业内核、库或编译器方向上限。

回答顺序：先说结论，再给条件和边界，最后用代码或 profiler 证据支撑。不要用“永远、一定、默认都这样”回答架构问题。

## 目录

1. GPU 硬件与执行模型
2. 内存层次与访问
3. 同步、原子与一致性
4. Occupancy、寄存器与延迟隐藏
5. Stream、Event 与 CUDA Graph
6. 并行算法模式
7. GEMM 优化阶梯
8. Tensor Core 与底层 MMA
9. `cp.async` 与多级流水
10. PTX、SASS 与编译链
11. Profiling、Roofline 与 Stall
12. 调试与工程化
13. 多 GPU 基础
14. Hopper
15. 项目深挖、手写题与模拟面试

---

# 1. GPU 硬件与执行模型

### Q1：CPU 与 GPU 的根本区别是什么？

**难度：B 必会**

**15 秒回答**

CPU 用较少的复杂核心追求单线程低延迟；GPU 用大量并行执行资源追求吞吐，并通过大量 ready warp 隐藏延迟。

**1 分钟展开**

GPU 不是让一次 global load 本身变得没有延迟，而是在某个 warp 等待时调度其他 eligible warp。GPU 适合有大量独立工作、规则数据访问和较高算术强度的问题；串行依赖、强分支和工作量很小的任务可能更适合 CPU。

**面试官继续追问**

1. latency hiding 和降低 latency 有什么区别？
2. 为什么小 kernel 可能被 launch 开销主导？

**常见误区**

- “GPU 核心更多，所以所有程序都更快。”

**项目证据**

- [Week 1 计时记录](../../notes/week01.md)展示了 kernel、H2D、D2H 和固定开销的区别。

### Q2：GPC、TPC、SM、warp scheduler、CUDA Core 是什么关系？

**难度：B 必会**

**15 秒回答**

SM 是 CUDA block 实际驻留和执行的主要计算单元；SM 内有 scheduler、寄存器文件、shared/L1 和多类执行管线。GPC/TPC 是更高层物理组织，CUDA Core 是标量算术执行资源之一。

**1 分钟展开**

不能把层级简单理解成“上层每次只调用一个下层”。一个 SM 可同时驻留多个 block/warp，scheduler 从 eligible warp 中选择指令发射到适合的执行管线。scheduler 数量、每周期 issue 能力和执行端口随架构与指令类型变化，不能把“每个 SM 固定四个 scheduler、每个每周期一条”当通用定律。

**面试官继续追问**

1. block 在生命周期中会迁移 SM 吗？
2. 为什么同一条 warp 指令可能占用不同执行管线？

**常见误区**

- 把 CUDA Core 当成能独立调度 CUDA thread 的完整 CPU core。

**项目证据**

- [GPU 硬件架构卷](../../cuda_deep_course/course/volume09_hardware_architecture/README.md)。

### Q3：SIMT 与 SIMD 有什么区别？

**难度：B 必会**

**15 秒回答**

SIMD 暴露向量指令和向量 lane；SIMT 让程序员写标量线程，硬件按 warp 将线程指令成组执行，同时保留每线程寄存器和执行状态。

**1 分钟展开**

CUDA 中一个 warp 通常有 32 个线程。线程可以有不同地址和控制流，但当同一 warp 分歧时，硬件要用 active mask 分阶段执行不同路径。Volta 之后有 independent thread scheduling，但不代表 warp 分歧免费，也不代表 warp 内通信可以省略正确同步。

**面试官继续追问**

1. independent thread scheduling 改变了什么？
2. 为什么旧式“隐式 warp 同步”代码可能出错？

**常见误区**

- “每个线程有独立 PC，所以分支完全并行。”

**项目证据**

- [Programming Model 详解](../courses/cuda/Programming_Model详解.md)。

### Q4：warp divergence 为什么慢，应该怎样优化？

**难度：B 必会**

**15 秒回答**

同一 warp 的线程走不同路径时，各路径通常按 active mask 分别执行，降低有效 lane 利用率；优化目标是让分支边界与 warp 工作划分一致，而不是机械删除所有 `if`。

**1 分钟展开**

短分支可能被 predication，复杂分支可能产生控制流。优化方式包括按数据分组、让整 warp 处理同类任务、减少路径工作量不均和把边界检查移到少数 warp。把分支改成同时计算两边再选择，可能增加更多指令，应实测。

**面试官继续追问**

1. 分支条件按 `threadIdx.x/32` 划分会 divergence 吗？
2. predication 一定更快吗？

**常见误区**

- 把 block 间走不同分支也叫 warp divergence。

**项目证据**

- [性能卷：Occupancy、分歧与延迟隐藏](../../cuda_deep_course/course/volume05_performance/03_Occupancy_分歧与延迟隐藏.md)。

---

# 2. 内存层次与访问

### Q5：register、local、shared、L1/L2、global 的区别是什么？

**难度：B 必会**

**15 秒回答**

register 是每线程片上状态；local 是每线程私有地址空间但通常位于片外并经缓存；shared 是 CTA 内显式管理的片上存储；L1/L2 是缓存；global 是设备内存地址空间。

**1 分钟展开**

“local”不表示物理上靠近线程。大局部数组、动态索引或寄存器压力可能使用 local。shared 容量和 block 分配共同限制 occupancy。L1/shared 的物理组织和容量配置随架构变化，因此不能给所有 GPU 套一个固定 KB 数字。

**面试官继续追问**

1. local memory 一定来自 spill 吗？
2. shared 为什么既快又可能成为瓶颈？

**常见误区**

- 把 local memory 当片上寄存器的同义词。

**项目证据**

- [CUDA 内存模型详解](../../notes/CUDA内存模型详解.md)。

### Q6：什么是 global memory coalescing？

**难度：B 必会**

**15 秒回答**

coalescing 是 warp 的内存请求按地址分布合并成尽可能少、利用率尽可能高的内存事务；核心是地址连续、对齐和有效字节利用率，而不只是“用了连续数组”。

**1 分钟展开**

相邻线程访问相邻元素通常有利，但事务具体由架构、访问宽度、对齐和 cache sector 决定。跨步访问会触及更多 segment/sector；即使 DRAM 带宽没满，也可能因请求碎片、指令吞吐或 latency 受限。

**面试官继续追问**

1. Array-of-Structs 什么时候会浪费事务？
2. 转置为何常出现合并读、跨步写的冲突？

**常见误区**

- “只要地址递增就是一次事务。”

**项目证据**

- [矩阵转置实验](../../week02_memory/transpose/transpose.cu)。

### Q7：shared memory bank conflict 是什么？

**难度：B 必会**

**15 秒回答**

同一 warp 的一次 shared 指令中，多个线程访问同一 bank 的不同地址会增加服务波次；同地址广播通常不构成普通冲突。

**1 分钟展开**

bank 映射与地址、访问宽度和架构有关。经典转置用 `[32][33]` 改变行 stride，避免按列访问时所有 lane 落到相同 bank。padding 不是万能答案：它增加 shared 占用，也可能对另一种访问模式无效。

**面试官继续追问**

1. 为什么同 bank 同地址可以广播？
2. 如何用 ncu 证明冲突而不是只靠推导？

**常见误区**

- “只要两个线程访问同一 bank 就冲突。”

**项目证据**

- [A100 GEMM bank conflict 定位](../../week05_gemm_advanced/ncu_notes.md)：4.8-way 降到约 1.5-way，性能同步改善。

### Q8：`float4` 为什么可能更快，什么时候不会？

**难度：A 主线**

**15 秒回答**

`float4` 可减少 load/store 指令和地址计算，但要求 16-byte 对齐、布局和尾部正确；源码类型不保证最终一定是一条 128-bit SASS load。

**1 分钟展开**

向量化主要优化指令和事务表达，不会自动提高算法 AI。若 kernel 已被计算、shared 或 latency 限制，收益可能很小；若强行向量化导致未对齐、复杂尾部或寄存器压力，反而可能变慢。最终用 ptxas、SASS 和 ncu 验证。

**面试官继续追问**

1. `reinterpret_cast<float4*>` 需要哪些前提？
2. 为什么宽 load 仍可能产生多个 memory sector？

**常见误区**

- 把 C++ 的 16-byte 类型直接等同于一条固定机器指令。

**项目证据**

- [float4 GEMM](../../week05_gemm_advanced/gemm_vectorized_load.cu)。

---

# 3. 同步、原子与一致性

### Q9：`__syncthreads()`、`__syncwarp()` 和 memory fence 有什么区别？

**难度：B 必会**

**15 秒回答**

barrier 等参与线程到齐并建立相应内存顺序；fence 约束调用线程的内存操作顺序/可见范围，但不等待其他线程；warp sync 只覆盖指定 warp lanes。

**1 分钟展开**

`__syncthreads()` 是 CTA collective，若只有部分线程到达且条件不一致可能死锁或未定义。`__threadfence()` 不会通知消费者“数据好了”，也不会等待消费者；常与原子 flag、正确 cache/volatile/atomic 语义组合成 producer-consumer 协议。

**面试官继续追问**

1. 为什么“写 data→threadfence→写 flag”仍需要消费者正确读取 flag？
2. `__syncwarp(mask)` 的 mask 为什么必须准确？

**常见误区**

- “调用 `__threadfence()` 后其他线程自动看到并开始消费。”

**项目证据**

- [fence/flag 实验](../../week02_memory/fence_flag.cu)。

### Q10：Atomic 保证了什么，又没有保证什么？

**难度：B 必会**

**15 秒回答**

原子操作保证指定地址上的 read-modify-write 不被并发破坏；它不保证整个多地址算法自动正确，也不消除争用和顺序协议设计。

**1 分钟展开**

大量线程竞争少数地址会串行化。常见优化是寄存器局部聚合→warp/block 聚合→少量 global atomic。原子的 scope 和 memory order 在现代 CUDA 中也重要；不能把一个 atomic 当全局 barrier。

**面试官继续追问**

1. shared atomic 与 global atomic 的差异？
2. warp-aggregated atomic 如何减少请求？

**常见误区**

- “用了 atomic 就没有任何 data race。”

**项目证据**

- [Atomic 分层实验](../../week03_parallel/atomic_sum/atomic_sum.cu)。

### Q11：Cooperative Groups 能直接实现任意 grid 同步吗？

**难度：A 主线**

**15 秒回答**

不能泛化。grid-wide sync 需要 cooperative launch，并受所有参与 block 能否同时驻留等约束；普通 kernel 的 block 不能假设并发存在。

**1 分钟展开**

Cooperative Groups 也提供 thread block、tiled partition 等结构化 collective，不只 grid sync。若算法可以拆成多个 kernel，kernel 边界通常是更简单可靠的全局同步。persistent/cooperative kernel 要显式检查设备能力和 occupancy 上限。

**面试官继续追问**

1. 为什么普通 kernel 中用 global counter 自旋等所有 block 可能死锁？
2. 多 kernel scan 为什么自然获得全局阶段边界？

**常见误区**

- “include cooperative_groups 后就能让所有 block `sync()`。”

**项目证据**

- [三阶段 Scan](../../week03_parallel/scan/scan.cu)使用 kernel 边界完成全局同步。

---

# 4. Occupancy、寄存器与延迟隐藏

### Q12：Occupancy 是什么？越高越好吗？

**难度：B 必会**

**15 秒回答**

Occupancy 是驻留 active warps 与架构最大 warps 的比例；它提供隐藏延迟的潜在 TLP，但不是性能目标，没有通用的“达到某百分比就够”。

**1 分钟展开**

限制来自 registers/thread、shared/block、threads/block 和架构上限。低 occupancy 配合高 ILP、数据复用仍可能快；高 occupancy 也可能所有 warp 都在等相同依赖。应同时看 eligible/issued warps、管线利用和最终时间。

**面试官继续追问**

1. 理论 occupancy 与 achieved occupancy 区别？
2. 为什么减寄存器可能降低性能？

**常见误区**

- 背“60% 或 70% 一般够”作为跨 kernel 结论。

**项目证据**

- [Occupancy 参数实验](../../notes/deepseek/week01_day04.md)。

### Q13：寄存器、spill、ILP 和 TLP 怎样权衡？

**难度：A 主线**

**15 秒回答**

更多寄存器可保存 accumulator、提高复用和 ILP，但会压低驻留 warp；压得过低可能 spill 到 local。最优点由复用、ILP、TLP 和片外流量共同决定。

**1 分钟展开**

用 `-Xptxas=-v` 看 registers/spill，用 SASS 看 local 路径，用 ncu 看 occupancy、local traffic、eligible warp 和吞吐。`__launch_bounds__`/`maxrregcount` 只是影响编译器策略，不是免费提高 occupancy 的开关。

**面试官继续追问**

1. 动态索引局部数组与 register spill 怎样区分？
2. 为什么 occupancy 上升后性能可能下降？

**常见误区**

- 只看 registers/thread，不看 accumulator 复用价值。

**项目证据**

- A100 上 8×4 线程 tile 在 occupancy 与复用间取得甜点，详见 [ncu_notes](../../week05_gemm_advanced/ncu_notes.md)。

---

# 5. Stream、Event 与 CUDA Graph

### Q14：Stream 并发需要哪些条件？

**难度：B 必会**

**15 秒回答**

同一 stream 有顺序，异 stream 只是“允许”并发；真正重叠还取决于依赖、硬件 copy engine、资源占用、内存类型和任务粒度。

**1 分钟展开**

H2D/D2H 与 kernel 重叠通常需要 pinned host memory、异步 API、不同 stream、分块和正确 event 依赖。两个占满 SM 的 kernel 即便在不同 stream 也未必并发。计时要区分单 stream event 时间与全流程 wall time。

**面试官继续追问**

1. pageable memory 为什么妨碍真正异步传输？
2. default stream 语义如何影响并发？

**常见误区**

- “放到两个 stream 就一定并行。”

**项目证据**

- [Stream overlap](../../week03_parallel/stream_overlap/stream_overlap.cu)实测约 2.2×流程加速。

### Q15：CUDA Event 能测什么，不能测什么？

**难度：B 必会**

**15 秒回答**

Event 记录 GPU stream 时间线，适合测设备操作区间；它不直接等于 CPU wall time，也不会自动包含未建立依赖的其他 stream 工作。

**1 分钟展开**

Event 必须记录在正确 stream，并等待 stop event 完成。短 kernel 应 warmup、多次重复，避免首次初始化/JIT 和计时分辨率影响。端到端延迟用 host 时钟并同步整个目标流程。

**面试官继续追问**

1. 为什么不能每次 kernel 后 `cudaDeviceSynchronize()` 再谈流水性能？
2. 如何测多个 stream 的 makespan？

**常见误区**

- 把 kernel event 时间和 H2D+kernel+D2H 总耗时混用。

**项目证据**

- [Week 1 GFLOPS 计时修正](../../notes/week01.md)。

### Q16：CUDA Graph 解决什么，什么时候不值得？

**难度：A 主线**

**15 秒回答**

Graph 把重复的 launch/依赖序列实例化并重放，主要降低 CPU launch 与调度开销；它不会让单个 kernel 的算术自动更快。

**1 分钟展开**

适合结构稳定、重复次数多、kernel 较短的流程。可用 stream capture 或显式建图；动态 shape、节点参数更新、条件控制和资源生命周期会增加复杂度。若单 kernel 已运行数毫秒，launch 节省可能不显著。

**面试官继续追问**

1. capture、instantiate、launch 分别在哪个阶段付费？
2. graph update 有哪些约束？

**常见误区**

- “用了 Graph，GPU kernel 本身就更快。”

**项目证据**

- [CUDA Graph 课程实验](../../cuda_deep_course/labs/07_async_system/cuda_graph/cuda_graph.cu)；当前项目只有课程样例，回答时不要冒充完整性能专项。

---

# 6. 并行算法模式

### Q17：Reduction 从 naive 到高性能的优化链是什么？

**难度：B 必会**

**15 秒回答**

核心是先在线程寄存器聚合，再做 warp shuffle，warp 结果经 shared/第二阶段合并，减少同步、shared 流量和 global atomic。

**1 分钟展开**

grid-stride 让每线程处理多个元素；warp shuffle 完成 32 lane 归约；warp leader 写 shared；第一个 warp 合并。跨 block 可以第二个 kernel 或少量 atomic。要正确处理 partial warp mask、溢出和浮点非结合性。

**面试官继续追问**

1. 为什么纯 shuffle 不能直接跨 warp？
2. 浮点 reduction 为什么不同顺序结果不同？

**常见误区**

- 无条件使用 `0xffffffff` mask 处理不完整 warp。

**项目证据**

- [完整 Reduction](../../week03_parallel/reduction_sum_full/reduction_sum_full.cu)的旧记录曾写“与 CUB 对比达到约 91% 带宽峰值”，但旧记录字节口径未验证，不作为 canonical 证据。

### Q18：Scan 为什么比 Reduction 难？

**难度：A 主线**

**15 秒回答**

Reduction 只需一个总结果，Scan 要保留每个前缀结果；多 block 时必须处理 block sums、扫描 block sums，再把偏移加回。

**1 分钟展开**

warp 内可用 `__shfl_up_sync`，block 内先扫各 warp，再扫 warp sums。grid 级常用三阶段 kernel；数据更大时 block sums 还需递归。inclusive/exclusive 的偏移和非整除边界是高频错误。

**面试官继续追问**

1. Hillis-Steele 与 Blelloch 的 work complexity？
2. Scan 如何用于 stream compaction？

**常见误区**

- 以为一次普通 kernel 内可以无条件同步任意多 block。

**项目证据**

- [Scan 实现](../../week03_parallel/scan/scan.cu)。

### Q19：Histogram 怎样降低 atomic 争用？

**难度：A 主线**

**15 秒回答**

先按 block/warp 私有化 bins，在更近层级聚合，再少量合并到 global；收益取决于 bin 数、数据分布和 shared 容量。

**1 分钟展开**

高度集中分布下 global atomic 热点严重，shared privatization 常明显受益；bin 很多或分布稀疏时，初始化/合并和 shared 占用可能抵消收益。还可做 warp-private、分段或排序聚合。

**面试官继续追问**

1. bin 超过 shared 容量怎么办？
2. 为什么均匀分布与集中分布结果不同？

**常见误区**

- 认为 shared atomic 永远比 global atomic 快。

**项目证据**

- [Histogram](../../week03_parallel/histogram/histogram.cu)在集中分布下实测约 12× 改善。

---

# 7. GEMM 优化阶梯

### Q20：Naive GEMM 为什么慢，shared tiling 解决什么？

**难度：B 必会**

**15 秒回答**

naive GEMM 让每个输出重复从 global 读取 A/B，算术强度低；shared tiling 把 tile 搬到片上，让 block 内多个输出复用。

**1 分钟展开**

一个 `BM×BN×BK` tile 的 A/B 被 CTA 协作加载，计算 `BK` 段后继续下一 tile。收益来自减少 global bytes；代价是 shared、barrier、边界和布局。global 瓶颈消失后，shared 带宽、bank conflict、地址计算或计算管线可能成为新墙。

**面试官继续追问**

1. 如何计算 naive 与 tiled 的 AI？
2. 为什么 shared tiled 仍远慢于 cuBLAS？

**常见误区**

- “用了 shared 就一定 compute-bound。”

**项目证据**

- [GEMM benchmark](../../week05_gemm_advanced/benchmark.md)。

### Q21：2D register tiling 为什么有效？

**难度：A 主线**

**15 秒回答**

每线程计算 `TM×TN` 输出，用 `regM[TM]×regN[TN]` 外积更新多个 accumulator，减少每个 FLOP 对 shared 的读取并提高 ILP。

**1 分钟展开**

每个 K 步把 A/B 小片段读入寄存器并复用。代价是 accumulator 数增大、register pressure 上升和 occupancy 下降。最佳 tile 不是越大越好，要结合 shared 布局、线程数、grid 规模和寄存器限制扫描。

**面试官继续追问**

1. 1D 与 2D tiling 分别复用什么？
2. 为什么 A100 上降低 TN 可能更快？

**常见误区**

- 只用 occupancy 下降否定 register tiling。

**项目证据**

- [2D register tiling](../../week05_gemm_advanced/gemm_2d_thread_tiling.cu)和 [参数扫描](../../notes/deepseek/week01_day04.md)。

### Q22：怎样系统评价一个 GEMM 优化版本？

**难度：A 主线**

**15 秒回答**

先验证数值和 benchmark，再做 FLOP/bytes/AI 账本，最后结合 ptxas、SASS 和 ncu 判断瓶颈及资源代价，并与 cuBLAS 同 shape/精度比较。

**1 分钟展开**

记录 GPU、CUDA、M/N/K、layout、dtype、warmup/repeat、时间、GFLOPS、误差、register/shared、occupancy 和关键 pipe。方阵大尺寸的单点成绩不能代表非方阵、小矩阵或边界；ncu replay 时间也不能代替非 profiling 的真实性能。

**面试官继续追问**

1. Roofline 为什么解释不了 shared bottleneck？
2. 如何区分算法 FLOP 与 Tensor Core 实际指令吞吐？

**常见误区**

- 用不同精度的 cuBLAS 峰值作为不加说明的对照。

**项目证据**

- [Roofline 记录](../../week05_gemm_advanced/roofline.md)与 [ncu_notes](../../week05_gemm_advanced/ncu_notes.md)。

---

# 8. Tensor Core 与底层 MMA

### Q23：Tensor Core、WMMA、MMA、HMMA 分别是什么？

**难度：B 必会**

**15 秒回答**

Tensor Core 是矩阵乘加硬件；WMMA 是 CUDA C++ warp matrix API；`mma.sync` 是 PTX warp 级指令语义；HMMA 是某些架构反汇编中的半精度矩阵机器指令。

**1 分钟展开**

编译链不是一一对应：一次 WMMA 调用可能 lower 成多条 PTX/SASS。要证明走 Tensor Core，可组合源码 API、SASS 矩阵指令、Tensor pipe 指标和性能量级。输入、累加类型和 layout 决定精度与支持形状。

**面试官继续追问**

1. FP16 input + FP32 accumulate 的误差来自哪里？
2. 为什么 Tensor Core kernel 仍可能 memory-bound？

**常见误区**

- “调用 WMMA 就自动接近峰值。”

**项目证据**

- [WMMA GEMM](../../week06_tensorcore/wmma_fp16_gemm.cu)与 [三重证据](../../week06_tensorcore/tensor_core_profile.md)。

### Q24：`ldmatrix` 和 fragment 是什么？

**难度：A 主线**

**15 秒回答**

fragment 是 warp 分布式持有的矩阵 operand；`ldmatrix` 协作地把 shared tile 加载成符合特定 MMA lane/register 布局的 fragment。

**1 分钟展开**

一个 lane 不持有完整行，元素映射依赖具体 shape/type/layout。`ldmatrix.m8n8.x1/x2/x4` 描述同时加载的矩阵数量，`.trans` 用于相应转置形式。shared 地址、对齐、swizzle 和所有 lane 一致参与都必须正确。

**面试官继续追问**

1. `ldmatrix` 是 global→shared 吗？
2. 为什么不能把一个 MMA shape 的 lane map 套到另一个 shape？

**常见误区**

- 把 fragment 当普通连续 C 数组。

**项目证据**

- [CUDA 深水区教材 Part II](../topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)。

### Q25：CUTLASS/CuTe 的三级 tile 为什么存在？

**难度：C 加分**

**15 秒回答**

threadblock tile 决定 CTA 数据复用，warp tile 分配 CTA 内工作，instruction tile 对应 MMA/FMA 基本操作；多层布局把 global、shared、register 和指令形状连接起来。

**1 分钟展开**

工业 GEMM 还需 mainloop pipeline、swizzle、epilogue、边界和架构特化。CuTe 用 layout algebra 表达逻辑坐标到线程/值/地址映射。会读的关键不是背模板，而是还原“哪个线程在何层持有什么数据”。

**面试官继续追问**

1. 为什么 shared swizzle 要同时考虑 bank 与 `ldmatrix`？
2. epilogue 为什么也可能成为瓶颈？

**常见误区**

- 声称读过 CUTLASS README 就等于能实现 production mainloop。

**项目证据**

- 当前项目主要有教学 GEMM；此题属于源码阅读能力，不冒充完整 CUTLASS 实战。

---

# 9. `cp.async` 与多级流水

### Q26：`cp.async` 解决什么，怎样正确等待？

**难度：A 主线**

**15 秒回答**

Ampere `cp.async` 支持 global→shared 异步复制，避免显式中间通用寄存器，并用 `commit_group/wait_group` 管理完成；等待 copy 与 CTA 消费同步是不同职责。

**1 分钟展开**

`.ca/.cg` 是缓存提示。发出 copies 后 commit 成 group，consumer 在读 shared 前等待相应旧 group 完成，并根据线程协作协议使用 barrier。普通 `__syncthreads()` 不能替代 async copy 的专用 completion。

**面试官继续追问**

1. `wait_group 1` 的直觉是什么？
2. 尾部 tile 怎样 zero fill？

**常见误区**

- “发完 `cp.async` 后算几条指令，数据自然就好了。”

**项目证据**

- [两级 pipeline GEMM](../../week05_gemm_advanced/gem_double_buffering.cu)。

### Q27：为什么 3-stage 不一定比 2-stage 快？

**难度：A 主线**

**15 秒回答**

更深 stage 可能隐藏更长 latency，但会增加 shared/CTA、同步和 prologue/epilogue 成本，甚至降低 active CTA/occupancy。

**1 分钟展开**

正确实现要管理 `FREE→COPYING→READY→CONSUMING` 的环形 slot 生命周期。性能上比较 sync/2/3 stage 的 GFLOPS、shared、register、occupancy、eligible warp 和 stall；若原本不是 latency-bound，增加 stage 不会有效。

**面试官继续追问**

1. producer 何时能覆盖旧 slot？
2. 如何证明收益来自 overlap？

**常见误区**

- 只展示时间变化，不展示资源代价和调度指标。

**项目证据**

- [CUDA 深水区教材 Part III](../topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)。

---

# 10. PTX、SASS 与编译链

### Q28：PTX 与 SASS 有什么区别？

**难度：A 主线**

**15 秒回答**

PTX 是虚拟 ISA，由 ptxas 或 driver JIT 针对目标架构生成机器码；SASS 是特定 GPU 机器码的反汇编，更接近最终执行事实。

**1 分钟展开**

CUDA C++→NVVM/PTX→ptxas→cubin；fatbin 可携带多个 cubin 和 PTX。PTX 虚拟寄存器不等于物理寄存器。用 `nvcc -ptx` 看语义、`-Xptxas=-v` 看资源、`cuobjdump --dump-sass` 看机器指令。

**面试官继续追问**

1. 为什么发布包同时放 cubin 和 PTX？
2. `compute_80` 与 `sm_80` 区别？

**常见误区**

- “GPU 逐条执行 `.ptx` 文件文本。”

**项目证据**

- [PTX/SASS 两周教材](../topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)。

### Q29：怎样用汇编证明优化真的生效？

**难度：A 主线**

**15 秒回答**

先写预期指令变化，再对比同编译条件下的 SASS、ptxas 资源和 ncu；不能只凭源码出现 `float4`、WMMA 或 unroll 就下结论。

**1 分钟展开**

例如 vector load 看实际 load 宽度和数量；WMMA 看 HMMA/MMA；register tiling 看独立 FFMA 与寄存器；spill 看 ptxas、local load/store 和 runtime local traffic。机器码证明“生成了什么”，ncu/时间证明“是否有价值”。

**面试官继续追问**

1. 为什么 SASS 不能单独证明 bank conflict？
2. 为什么 `-G` 构建不能用于性能汇编分析？

**常见误区**

- grep 到一条目标指令就宣称整个 kernel 已优化。

**项目证据**

- [Tensor Core SASS 证据](../../week06_tensorcore/tensor_core_profile.md)。

---

# 11. Profiling、Roofline 与 Stall

### Q30：nsys、ncu、Roofline 分别回答什么？

**难度：B 必会**

**15 秒回答**

nsys 看系统时间线和并发；ncu 看单 kernel 微观硬件行为；Roofline 用 AI 和峰值给出带宽/计算上限，但不建模所有片上瓶颈。

**1 分钟展开**

先用可靠 benchmark 确认问题，再用 nsys 定位大时间块/launch/同步，用 ncu 看 SOL、memory、occupancy、scheduler 和 source counters。Roofline 若显示 DRAM 不是墙，仍需看 shared、指令、latency、divergence 和 occupancy。

**面试官继续追问**

1. ncu replay 时间为何不能代替正常 benchmark？
2. AI 高就必然 compute-bound 吗？

**常见误区**

- 看到 DRAM throughput 低就说访存没问题。

**项目证据**

- [A100 Roofline 与 ncu 联合分析](../../notes/deepseek/week01_day05.md)。

### Q31：Active、Eligible、Issued warp 和 Scoreboard 是什么？

**难度：A 主线**

**15 秒回答**

active warp 已驻留；eligible warp 的下一条指令依赖已就绪；issued 是实际发射。Scoreboard 跟踪数据依赖，防止依赖结果未就绪时发射。

**1 分钟展开**

active 多但 eligible 少说明很多 warp 同时等待；可用更多 TLP、增加 ILP、预取或减少延迟，但要先定位依赖。Scoreboard 不是缓存，stall 是等待症状，根因可能是内存、shared conflict、MUFU、依赖链等。

**面试官继续追问**

1. ILP 与 TLP 怎样分别隐藏延迟？
2. 为什么 occupancy 高仍可能 issue 不足？

**常见误区**

- 把 active warp 数等同于每周期可发射 warp 数。

**项目证据**

- [Scheduler/Scoreboard 教材](../topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)。

### Q32：Long Scoreboard 高意味着什么？

**难度：A 主线**

**15 秒回答**

它表示 warp 等待 L1TEX 路径相关依赖，可能涉及 global/local/texture/surface；不是“DRAM 慢”的同义词。

**1 分钟展开**

先看 source/SASS 对应 load，再查 cache/DRAM、local spill、load-use 距离和 eligible warp。只有 scheduler issue 不足且该 stall 真正主导时才优先优化。Not Selected 高反而可能说明 eligible warp 充足。

**面试官继续追问**

1. Short Scoreboard 常与哪些路径有关？
2. 怎样形成可证伪的 stall 假设？

**常见误区**

- “Long Scoreboard 高，直接把数据搬 shared。”

**项目证据**

- [ncu 指标详解](../topics/performance/Nsight_Compute_ncu详解.md)。

---

# 12. 调试与工程化

### Q33：Compute Sanitizer 四个常用工具分别做什么？

**难度：B 必会**

**15 秒回答**

memcheck 查非法/越界访问，initcheck 查未初始化 device global 读取，racecheck 查 shared data hazard，synccheck 查同步原语使用错误。

**1 分钟展开**

通常先 memcheck，再 initcheck/racecheck/synccheck，避免越界制造噪声。工具没有报告不等于算法正确：错误 async stage、浮点误差、遗漏元素仍需 CPU/reference、边界测试和重复压力测试。

**面试官继续追问**

1. racecheck 为什么主要关注 shared hazard？
2. 什么错误需要 cuda-gdb？

**常见误区**

- 只跑 memcheck 就称“并发正确性全部验证”。

**项目证据**

- 当前项目主要有 memcheck 记录；其他工具应如实说“理解用途，正在系统补实验”。

### Q34：怎样设计可信的 CUDA benchmark 和回归测试？

**难度：A 主线**

**15 秒回答**

固定环境和输入，warmup、多次重复、用正确时间域、验证结果，记录编译/硬件/shape，并把性能阈值与噪声区分。

**1 分钟展开**

正确性覆盖小尺寸、非整除、极值、随机种子、NaN/Inf 和容差。性能记录 median/percentile，避免首次 JIT、动态频率、其他负载和 profiler replay。回归要比较相同 GPU/driver/toolkit 或明确归一化边界。

**面试官继续追问**

1. 浮点 kernel 为什么不能只用绝对误差？
2. 如何避免编译器把 microbenchmark 优化掉？

**常见误区**

- 只测一次最大方阵并当作通用性能。

**项目证据**

- [实验完成标准](../../cuda_deep_course/course/实验方法与完成标准.md)。

### Q35：CUDA 工程中为什么要做错误检查和 RAII？

**难度：B 必会**

**15 秒回答**

CUDA API 和 kernel launch 都可能异步失败；统一错误检查、明确同步点和 RAII 资源生命周期能避免错误被延迟、资源泄漏和异常路径清理失败。

**1 分钟展开**

launch 后 `cudaGetLastError()` 查配置/启动错误；需要确认执行完成时在合适边界同步并检查。device buffer、stream、event 可封装 move-only RAII。不要每个 launch 后全局同步破坏异步性能。

**面试官继续追问**

1. `cudaPeekAtLastError` 与 `cudaGetLastError` 的状态差异？
2. 析构函数里如何处理 CUDA 错误？

**常见误区**

- 用宏错误检查替代资源所有权设计。

**项目证据**

- [公共错误检查头](../../common/cuda_check.cuh)。

---

# 13. 多 GPU 基础

### Q36：PCIe、NVLink、P2P 和 NCCL 的关系是什么？

**难度：A 主线**

**15 秒回答**

PCIe/NVLink 是互连路径，P2P 允许 GPU 直接访问/复制对端显存，NCCL 在拓扑上实现集合通信；带宽和路径取决于具体拓扑，不由 API 名保证。

**1 分钟展开**

先用拓扑工具/设备属性确认 peer access 和链路。跨 NUMA/PCIe switch、NVLink/NVSwitch 路径不同。NCCL 根据消息大小和拓扑选择 ring/tree 等协议；collective 仍需要正确 stream 和数据依赖。

**面试官继续追问**

1. P2P 不可用时可能走什么路径？
2. 为什么标称 NVLink 带宽不等于应用有效带宽？

**常见误区**

- “多卡调用 NCCL 就一定线性扩展。”

**项目证据**

- [多 GPU 与拓扑教材](../../cuda_deep_course/course/volume08_hpc_multigpu/README.md)；当前项目缺少真实多卡 benchmark，应如实说明。

### Q37：All-Reduce 为什么常分解为 Reduce-Scatter + All-Gather？

**难度：A 主线**

**15 秒回答**

Reduce-Scatter 让各 rank 得到归约结果的一段，All-Gather 再交换各段得到完整结果；这种分解适合带宽高效的环形实现。

**1 分钟展开**

具体算法由拓扑、rank 数、消息大小和协议决定，不能说 NCCL 永远只用 ring。通信量分析要区分每 rank 注入字节、链路流量和总系统流量。计算通信重叠还要求分块、stream/event 依赖和足够独立工作。

**面试官继续追问**

1. All-Gather 与 All-to-All 数据分布差异？
2. 小消息为何可能 latency-bound？

**常见误区**

- 把数学 collective 与某个固定拓扑算法绑定。

**项目证据**

- 模型并行场景见 [AI Infra 面试八股](AI_Infra面试核心题库.md)。

---

# 14. Hopper

### Q38：TMA 相比 `cp.async` 改变了什么？

**难度：C 加分**

**15 秒回答**

TMA 使用 tensor descriptor 和坐标发起 bulk 多维搬运，把大量 per-thread 地址生成和 copy 指令卸载给专用引擎，并与 transaction-aware barrier 协作。

**1 分钟展开**

`cp.async` 通常由多个线程提供具体 global/shared 地址；TMA 描述 shape、stride、box、swizzle 等，由少量线程发起。它仍需正确 descriptor、对齐、stage 和 mbarrier 协议，不是自动完成整个 pipeline。

**面试官继续追问**

1. 为什么 TMA 适合 warp specialization？
2. expected transaction bytes 有什么作用？

**常见误区**

- “TMA 只是单条更宽的 `cp.async`。”

**项目证据**

- [Hopper 迁移章节](../topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)。无 H100 时只能称架构理解，不能声称实测性能。

### Q39：WGMMA、Cluster 和 DSM 分别解决什么？

**难度：C 加分**

**15 秒回答**

WGMMA 让 4 个 warp 的 warp-group 异步协作 MMA；Cluster 保证一组 CTA 在同一 GPC 协同调度；DSM 让 cluster 内 CTA 访问彼此 shared 分区。

**1 分钟展开**

WGMMA 有自己的 fence/commit/wait 和 operand 规则，不能套 Ampere lane map。Cluster 提供 cluster sync 和生命周期保证；DSM 不是全 GPU shared，远端访问也不等同本地 shared 延迟。只有需要跨 CTA 复用/通信时才值得承担复杂度。

**面试官继续追问**

1. 远端 shared 所属 CTA 为什么不能提前退出？
2. producer/consumer warp-group 怎样分工？

**常见误区**

- “Hopper 新特性全部打开就自动最快。”

**项目证据**

- [Hopper 官方迁移式学习资料](../topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)。

### Q40：Persistent kernel 与 warp specialization 的价值是什么？

**难度：C 加分**

**15 秒回答**

Persistent kernel 让有限 CTA 常驻并从任务队列取工作，减少 launch、支持动态负载均衡；warp specialization 让不同 warp 固定负责搬运、计算或 epilogue，以形成更深流水。

**1 分钟展开**

代价是调度公平、队列同步、资源静态占用、尾部负载和调试复杂度。persistent 不适合所有大规则 grid；specialization 也会牺牲部分 warp 的计算占比。Hopper TMA/WGMMA 让角色分工更自然，但协议更复杂。

**面试官继续追问**

1. 常驻 CTA 数如何选择？
2. 动态任务队列怎样避免热点 atomic？

**常见误区**

- 把 grid-stride loop 直接等同于完整 persistent scheduler。

**项目证据**

- 当前项目尚无 production persistent kernel，本题属于 C 层设计理解。

---

# 15. 项目深挖、手写题与模拟面试

## 15.1 项目深挖题

### A100 GEMM

1. 为什么 T4 与 A100 上同一 kernel 的瓶颈画像不同？
2. 你怎样从 168 registers/thread 推出 occupancy 受限？
3. 为什么把 TN 从 8 降到 4 变快，但继续降复用又变慢？
4. bank conflict 的最初假设为何错误，ncu 怎样推翻它？
5. 如果下一步做 `mma.sync+cp.async`，你怎样设置基线和验收？

### Reduction / Scan / Stream

1. 旧记录曾称手写 reduction 追平 CUB，但字节口径未验证，不作为 canonical 证据；应如何重新建立可信比较？
2. 三阶段 scan 的全局同步在哪里？
3. stream overlap 的 2.2× 为什么不是 kernel 本身快了 2.2×？

### WMMA

1. 你如何用源码、SASS、性能和 ncu 四层证明 Tensor Core 路径？
2. Tensor pipe 只有约 5.7% 时，下一步为什么先看供数而不是继续堆 MMA？

## 15.2 高频手写题与评分点

| 题目 | 必须写对 | 加分项 |
|---|---|---|
| vector add | grid-stride、边界、错误检查 | float4 尾部与对齐 |
| reduction | 局部累加、shuffle mask、跨 warp | 两阶段、数值误差 |
| scan | inclusive/exclusive、warp sums | 多 block 递归 |
| transpose | coalesced、shared、padding | 非整除、GB/s |
| histogram | bin 边界、atomic | shared/warp privatization |
| tiled GEMM | 索引、tile load、barrier、K 尾部 | register tiling、AI |
| WMMA 骨架 | fragment、layout、leading dimension | SASS HMMA 证明 |

## 15.3 三档模拟面试

### B：常规 CUDA 开发（45 分钟）

1. SIMT、warp divergence、coalescing。
2. `__syncthreads` 与 fence。
3. 手写 block reduction。
4. stream overlap 条件。
5. memcheck/racecheck 使用。

### A：Kernel 性能（60 分钟）

1. 给出一个慢 GEMM，建立 Roofline 与 ncu 诊断顺序。
2. register tiling 与 occupancy 权衡。
3. SASS 验证 float4/WMMA。
4. `cp.async` 3-stage 状态机。
5. Long Scoreboard 的可证伪假设。

### C：专家加分（45 分钟）

1. `ldmatrix`、MMA fragment 与 shared layout。
2. CUTLASS mainloop 的层级 tile。
3. TMA/WGMMA/mbarrier producer-consumer。
4. persistent kernel 调度设计。

## 15.4 复习清单

- [ ] B 题能在 15 秒给出无硬伤结论。
- [ ] A 题能在 1 分钟内给出证据链和反例。
- [ ] C 题明确哪些是阅读理解、哪些有实测。
- [ ] 至少闭卷写出 reduction、transpose、tiled GEMM。
- [ ] 能讲完 A100 GEMM 的“假设→profile→修复→复测”。
- [ ] 不使用固定 scheduler 数、固定 occupancy 阈值等跨架构绝对话术。

## 官方资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)


---

# 附录 A：Q1-Q22 追问答案（原文并入）

> 来源：[CUDA 面试八股追问答案 Q1-Q22 归档原文](../archive/interview/CUDA面试八股_追问答案_Q1-Q22.md)。除重复一级标题外，正文按归档原文完整并入。


> 配套：[CUDA面试八股全集.md](CUDA面试核心题库.md)
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


---

# 附录 B：Q23-Q40 追问答案（原文并入）

> 来源：[CUDA 面试八股追问答案 Q23-Q40 归档原文](../archive/interview/CUDA面试八股_追问答案_Q23-Q40.md)。除重复一级标题外，正文按归档原文完整并入。


> 配套：[CUDA面试八股全集.md](CUDA面试核心题库.md)、[追问答案第1-7章](CUDA面试核心题库.md)
> 本篇覆盖第 8-15 章：Tensor Core / cp.async / PTX-SASS / profiling / 调试 / 多GPU / Hopper / 手写题与模拟面试。

---

# 第 8 章：Tensor Core 与底层 MMA

## Q23 TensorCore/WMMA/MMA/HMMA 的追问

**1. FP16 input + FP32 accumulate 的误差来自哪里？**
- 误差主要来自 **FP16 输入本身的表示精度**（10 位尾数，动态范围有限，大值/小值易舍入）。而**累加用 FP32** 恰恰是为了减少误差——长链累加若也用 FP16 会迅速丢精度。所以现代 Tensor Core 都是"低精度乘 + 高精度累加"。输入量化通常是主要误差源之一，FP32 累加显著降低但不能消除舍入误差。

**2. 为什么 Tensor Core kernel 仍可能 memory-bound？**
- Tensor Core 把"算"做得极快，但如果**喂数据跟不上**（global→shared→fragment 的搬运带宽不够、或 shared bank conflict、或 `ldmatrix` 供数不足），算力单元就闲着等数据。你 week06 看到 Tensor pipe 只 ~5.7% active，就是"算得快但喂不饱"→ 瓶颈在供数不在算。

## Q24 ldmatrix/fragment 的追问

**1. `ldmatrix` 是 global→shared 吗？**
- **不是**。`ldmatrix` 是 **shared→寄存器(fragment)**，把 shared 里的一块 tile 协作加载成符合 MMA lane/register 布局的 fragment。global→shared 是 `cp.async`/普通 load 的事。两者是流水的不同环节。

**2. 为什么不能把一个 MMA shape 的 lane map 套到另一个 shape？**
- 不同 MMA shape（如 m16n8k8 vs m16n8k16）规定了**不同的 lane→寄存器→矩阵元素映射**。哪个 lane 持有矩阵哪几个元素是硬件固定死的，且随 shape/dtype/layout 变。套错映射 → 读到错误元素 → 结果全错。必须查对应 shape 的官方 fragment 布局。

## Q25 CUTLASS 三级 tile 的追问

**1. 为什么 shared swizzle 要同时考虑 bank 与 `ldmatrix`？**
- swizzle（打乱 shared 存储布局）要同时满足两个约束：① 避免 warp 访问 shared 时的 **bank conflict**；② 满足 `ldmatrix` 要求的**特定 lane 访问 pattern**。两个约束互相牵制——不能只为 bank 优化而破坏 ldmatrix 需要的布局，得设计一个同时满足两者的 swizzle。

**2. epilogue 为什么也可能成为瓶颈？**
- epilogue（写回阶段：accumulator → 加 bias/激活/scale → 转精度 → 写 global）也有访存和计算。大 tile 的 accumulator 写回是大量 global store，若不做好合并/向量化，或融合的激活很重，epilogue 会占可观时间。工业 GEMM 会把 epilogue 也流水化/融合。

---

# 第 9 章：cp.async 与多级流水

## Q26 cp.async 的追问

**1. `wait_group 1` 的直觉是什么？**
- `cp.async.wait_group N` = "等到**最多只剩 N 个** group 还没完成"。`wait_group 1` 就是"保留 1 个 group 在飞、其余都等完"——用于双缓冲：当前 tile 算之前，确保它已搬完（wait 到只剩下一个正在预取的），同时让下一 tile 的预取继续在飞。`wait_group 0` = 全部等完。

**2. 尾部 tile 怎样 zero fill？**
- 当 K 不是 tile 的整数倍，最后一块不足。越界的部分不能读 global（会越界），要在 shared 里**填 0**：对越界位置跳过 cp.async 拷贝并显式写 0（或用 cp.async 的越界处理），保证矩阵乘时那些位置贡献 0，不污染结果。

## Q27 3-stage vs 2-stage 的追问

**1. producer 何时能覆盖旧 slot？**
- 环形 slot 生命周期 `FREE→COPYING→READY→CONSUMING→FREE`。producer 要覆盖一个 slot（重新往里搬新 tile），必须等它变回 **FREE**——即消费者已经**读完**那个 slot 的数据（CONSUMING 结束）。靠 barrier/mbarrier 或 wait_group 保证"不覆盖仍在被读的 slot"。

**2. 如何证明收益来自 overlap？**
- 不能只看时间变快。要用 ncu 看：① **eligible warp / issue 效率**是否提升（流水让调度器有活干）；② **long scoreboard stall 是否下降**（访存延迟被藏）；③ 对比 sync/2/3-stage 的 GFLOPS + shared + occupancy。你 week04 attention 就是用 long_scoreboard 6.4→0.03 证明 overlap 生效的。

---

# 第 10 章：PTX、SASS 与编译链

## Q28 PTX vs SASS 的追问

**1. 为什么发布包同时放 cubin 和 PTX？**
- **cubin**：针对特定架构（sm_80）编译好的机器码，能直接跑、启动快，但只对那个架构有效。**PTX**：虚拟 ISA，能被 driver **JIT** 编译到未来的新架构。同时放 = 已知架构用 cubin（快），未知/更新架构 fallback 到 PTX JIT（向前兼容）。这就是 fatbin。

**2. `compute_80` 与 `sm_80` 区别？**
- `compute_80` = **虚拟架构**，编译目标是 **PTX**（compute capability 8.0 的 PTX）。`sm_80` = **真实架构**，编译目标是 **SASS/cubin**（Ampere A100 机器码）。`-arch=compute_80 -code=sm_80,compute_80` 就是"生成 sm_80 cubin + 保留 PTX 备 JIT"。

## Q29 用汇编证明优化的追问

**1. 为什么 SASS 不能单独证明 bank conflict？**
- bank conflict 是**运行时**行为——取决于 warp 里 32 个 lane 的**实际地址**落在哪些 bank。SASS 只是静态指令（`LDS`/`STS`），看不出运行时各 lane 的地址分布。必须用 **ncu 的 bank conflict 指标**（运行时测量）才能证明，SASS 只能看有没有 shared 访问指令。

**2. 为什么 `-G` 构建不能用于性能汇编分析？**
- `-G`（device debug）会**关闭优化**、插入调试信息，生成的 SASS 和真实优化版完全不同（更多指令、无 unroll、寄存器分配不同）。拿它分析性能会得出错误结论。性能分析要用 `-O3 -lineinfo`（保留行号但不关优化）。

---

# 第 11 章：Profiling、Roofline 与 Stall

## Q30 nsys/ncu/Roofline 的追问

**1. ncu replay 时间为何不能代替正常 benchmark？**
- ncu 为了采集硬件计数器会**多次重放(replay)** kernel、插桩，严重拖慢执行（你见过 GEMV 从 0.05ms→14ms）。ncu 报的时间是插桩后的，**不是真实性能**。性能数字要用没插桩的正常运行（CUDA Event）测；ncu 只用来看**指标/瓶颈**，不看绝对时间。

**2. AI 高就必然 compute-bound 吗？**
- 不必然。AI 高只说明"每字节做的计算多"，落在 Roofline 计算受限区**的前提是其他资源没先成为墙**。可能 AI 高但卡在 **shared 带宽、bank conflict、指令吞吐、延迟、occupancy 不足**——这些 Roofline 不建模。要 ncu 看 SOL 各项才能定论。

## Q31 warp 状态/scoreboard 的追问

**1. ILP 与 TLP 怎样分别隐藏延迟？**
- **TLP（线程级并行）**：靠**更多 ready warp**——一个 warp 等待时调度别的 warp，用数量盖延迟。**ILP（指令级并行）**：靠**同一 warp 内多条独立指令**——发起多个独立 load / 多个 accumulator，让一个 warp 自己就有活干，不必切换。两者都能藏延迟，可组合。

**2. 为什么 occupancy 高仍可能 issue 不足？**
- occupancy 高 = 驻留 warp 多，但如果这些 warp **全都在 stall**（比如一起等同一类 global 访存），某一拍**没有 eligible warp**，调度器无指令可发 → issue 不足。occupancy 只保证"warp 多"，不保证"此刻有 warp 能发指令"。

## Q32 long scoreboard 的追问

**1. Short Scoreboard 常与哪些路径有关？**
- **Short scoreboard** 主要关联 **shared memory** 访问依赖（以及 MUFU/特殊函数单元等固定延迟操作）。区别于 **long scoreboard**（关联 L1TEX 路径：global/local/texture/surface 的较长延迟访存）。short = 等 shared/片上，long = 等 global/片外。

**2. 怎样形成可证伪的 stall 假设？**
- 别只说"访存慢"。要：① 定位是哪条 load（source counters / SASS 行号）；② 提出假设（如"这条 global load 未合并导致 long scoreboard"）；③ 给可验证的预测（"改合并后 long scoreboard 应下降、sector/请求数应减少"）；④ 改完用 ncu 复测证实或证伪。假设要能被数据推翻。

---

# 第 12 章：调试与工程化

## Q33 compute-sanitizer 四工具的追问

**1. racecheck 为什么主要关注 shared hazard？**
- racecheck 检测的是**同 block 内线程通过 shared memory 的数据竞争**（缺同步的读写冲突）。global 上的竞争更难静态界定（跨 block、跨 kernel），且通常靠算法协议保证；shared 的 race 有明确的"缺 `__syncthreads`"模式，是最常见也最可检测的并发 bug。

**2. 什么错误需要 cuda-gdb？**
- sanitizer 报"有非法访问/竞争"但**定位不到具体逻辑原因**时，或需要**单步、看变量、看线程状态、下断点**追踪控制流/数据流的复杂 bug（如特定线程走错分支、索引算错）。sanitizer 查"有没有错"，cuda-gdb 查"为什么错、在哪一步"。

## Q34 benchmark/回归测试的追问

**1. 浮点 kernel 为什么不能只用绝对误差？**
- 结果数值范围差异大：值很大时，正常的相对舍入误差换算成绝对误差也很大（会误判 FAIL）；值接近 0 时，绝对误差小但相对可能很大。应主要用**相对误差**（或绝对+相对结合），阈值按 fp32/fp16 累加特性设（你 GEMV K=4096 就放宽到 1e-3）。

**2. 如何避免编译器把 microbenchmark 优化掉？**
- 如果结果没被使用，编译器会把整个计算**删掉**（dead code elimination）。要：① 把结果写到 global / 用 volatile / `__syncthreads` 依赖它；② 让输入运行时才知道（不能编译期常量折叠）；③ 打印或校验部分结果。确保被测代码真的执行了。

## Q35 错误检查/RAII 的追问

**1. `cudaPeekAtLastError` 与 `cudaGetLastError` 的状态差异？**
- 两者都返回最近的错误，区别：**`cudaGetLastError`** 会**清除**错误状态（读完复位为 success）；**`cudaPeekAtLastError`** **只看不清**（保留错误状态）。kernel launch 后常用 `cudaGetLastError` 检查并清除，避免错误"粘"到下一次检查。

**2. 析构函数里如何处理 CUDA 错误？**
- 析构函数**不应抛异常**（C++ 规则，抛了可能 terminate）。所以 RAII 封装的析构里调 `cudaFree`/`cudaStreamDestroy` 时，错误应**记录日志/吞掉**，不 throw。资源释放尽力而为，错误在构造/使用阶段用返回码或异常处理。

---

# 第 13 章：多 GPU 基础

## Q36 PCIe/NVLink/P2P/NCCL 的追问

**1. P2P 不可用时可能走什么路径？**
- P2P（GPU 直接访问对端显存）不可用时，数据要**绕道 host**：GPU0 → host 内存 → GPU1（staging copy）。跨 PCIe switch、跨 NUMA、或硬件/驱动不支持 P2P 时就会这样，带宽和延迟都变差。要用设备属性 / `cudaDeviceCanAccessPeer` 先查。

**2. 为什么标称 NVLink 带宽不等于应用有效带宽？**
- 标称是**物理链路峰值**（双向、理想）。实际有效带宽受：协议开销、消息大小（小消息 latency-bound）、拓扑（是否经 NVSwitch）、争用、单双向、对齐等影响。应用能拿到的通常是峰值的一个折扣（用 nccl-tests 的 busbw 实测）。

## Q37 All-Reduce 分解的追问

**1. All-Gather 与 All-to-All 数据分布差异？**
- **All-Gather**：每个 rank 出一段，最后**所有 rank 都拿到全部段的拼接**（大家结果相同）。**All-to-All**：每个 rank 给**每个其他 rank 发不同的一段**（类似矩阵转置，各 rank 结果不同）。All-Gather 是"广播式收集"，All-to-All 是"个性化重分布"。

**2. 小消息为何可能 latency-bound？**
- 小消息时，数据传输本身很快，但每次通信的**固定开销**（启动延迟、协议握手、同步）占主导。这时增大带宽没用，瓶颈是延迟和消息数。优化靠**合并小消息**、减少通信次数，而非提高带宽。

---

# 第 14 章：Hopper

## Q38 TMA 的追问

**1. 为什么 TMA 适合 warp specialization？**
- TMA 由**少量线程**（甚至一个）用 descriptor + 坐标就能发起整块多维搬运，不需要每个线程算地址。这样可以让**少数 producer 线程/warp 专职发 TMA**，其余 consumer warp 专职算（WGMMA）。分工清晰，正是 warp specialization 的生产者-消费者模式。

**2. expected transaction bytes 有什么作用？**
- Hopper 的 mbarrier 支持"事务计数"：告诉 barrier **预期会到达多少字节**的异步拷贝。TMA 搬完的字节数累加到 barrier，等达到 expected bytes 时 barrier 才放行。这让消费者能**精确知道"数据全到齐了"**，而不是靠固定线程数计数，适配 TMA 的 bulk 异步语义。

## Q39 WGMMA/Cluster/DSM 的追问

**1. 远端 shared 所属 CTA 为什么不能提前退出？**
- DSM（分布式 shared）让 cluster 内一个 CTA 能访问**另一个 CTA 的 shared**。如果被访问的那个 CTA 提前退出、释放了 shared，访问方就读到无效内存。所以 cluster 内有生命周期保证：持有被远端访问 shared 的 CTA 必须**等到没人再访问**才能退出（cluster sync 协调）。

**2. producer/consumer warp-group 怎样分工？**
- **producer warp-group**：用 TMA 把 global 数据搬进 shared，发 mbarrier 通知。**consumer warp-group**：等 mbarrier，用 WGMMA 从 shared 算矩阵乘，算完通知 producer 可覆盖。两组通过 mbarrier + stage 环形缓冲协作，形成深流水——这是 Hopper GEMM/FlashAttention-3 的核心范式。

## Q40 persistent kernel/warp specialization 的追问

**1. 常驻 CTA 数如何选择？**
- 选"**恰好占满所有 SM 的可驻留上限**"——即 `SM 数 × 每 SM 能驻留的 CTA 数`（受 register/shared/threads 限制）。太多会超出无法全驻留（退化）；太少填不满 GPU。要用 occupancy API 算出每 SM 能驻留几个，再乘 SM 数。

**2. 动态任务队列怎样避免热点 atomic？**
- 全局一个 atomic 计数器发任务会成热点。缓解：① 每个 CTA 用 atomicAdd **批量领取**一批任务（减少 atomic 次数）；② 分片队列（每个 SM/CTA 一个本地队列 + work stealing）；③ warp-aggregated 领取。核心是降低对单一 atomic 地址的争用频率。

---

# 第 15 章：手写题与模拟面试（关键点速记）

## 15.2 高频手写题的"必须写对"要点

| 题目 | 一定不能错的点 | 加分 |
|------|--------------|------|
| **vector add** | grid-stride 循环、`if(i<n)` 边界、launch 后 `cudaGetLastError` | float4 + K%4 尾部 + 16B 对齐 |
| **reduction** | 寄存器局部累加 → warp shuffle(偏移16→1) → 跨 warp 走 shared → 精确 mask | 两阶段(第二 kernel/atomic)、说清浮点非结合 |
| **scan** | 分清 inclusive/exclusive、warp 内 shfl_up、warp sums 再扫、偏移加回 | 多 block 递归、非整除边界 |
| **transpose** | shared 中转(合并读+合并写)、`[32][33]` padding 避 bank conflict | 非整除 tile、报 GB/s |
| **histogram** | bin 边界、shared privatize + `__syncthreads` + 合并到 global | warp-private、分布相关性 |
| **tiled GEMM** | 行列索引、tile 协作 load、`__syncthreads`、K 尾部处理 | register tiling、算 AI |
| **WMMA 骨架** | fragment 声明、layout、leading dimension、`fill_fragment` 清零 | SASS 看 HMMA 证明 |

## 15.3 模拟面试答题框架

**通用套路（B/A/C 都适用）**：
> 先给**结论**（15 秒）→ 再给**条件和边界**（什么情况成立/不成立）→ 最后用**代码/profiler 证据**支撑。

**避免的话术**（会被扣分）：
- "永远/一定/默认都这样" 回答架构问题（scheduler 数、occupancy 阈值都随架构变）
- 只报时间不报资源代价（reg/shared/occupancy）
- 只测一个大方阵就说通用性能
- grep 到一条 float4/HMMA 就说 kernel 优化好了

**A 档(性能岗)高频链条**：
> 给个慢 kernel → 先 benchmark 确认问题 → Roofline 定位 compute/memory → nsys 看大时间块/launch → ncu 看 SOL/stall/occupancy → 提**可证伪假设** → 改 → 复测。这套"假设→profile→修复→复测"能讲顺，A 档就稳。

**你的项目证据库（面试直接引用）**：
- GEMV：合并访问 72→1360 GB/s（19×）、float4 只 +4% 因已 memory-bound（DRAM 72%）
- attention pipelined：long_scoreboard 6.4→0.03 证明 cp.async 藏延迟
- GEMM：bank conflict 4.8→1.5 way、register tiling 8×4 甜点
- reduction：旧记录曾写“追平 CUB ~91% 带宽”，但旧记录字节口径未验证，不作为 canonical 证据。

## 15.4 复习清单（自检）
- [ ] B 题 15 秒无硬伤结论
- [ ] A 题 1 分钟证据链 + 一个反例
- [ ] C 题分清"读过源码/架构理解" vs "有实测"
- [ ] 闭卷写出 reduction / transpose / tiled GEMM
- [ ] 讲完 A100 GEMM 的"假设→profile→修复→复测"
- [ ] 不用固定 scheduler 数 / occupancy 阈值等跨架构绝对话术

---

> 全 40 题追问 + 手写题 + 模拟面试要点整理完毕（分第1-7章、第8-15章两篇）。
> 复习建议：先合上答案自己答，卡住的地方标记出来重点记——卡住处就是你的薄弱点。
