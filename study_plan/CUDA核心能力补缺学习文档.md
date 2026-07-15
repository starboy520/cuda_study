# CUDA 核心能力补缺学习文档

> **定位**：这是一份知识地图、缺口清单和验收手册，不是第二条并行执行计划。每日任务仍以[四周聚焦计划](四周聚焦计划_AIInfra与CUDA深水区.md)及其当前周执行清单为准。
>
> **适用方向**：CUDA Kernel、推理性能、AI Infra / Model Serving。
>
> **学习原则**：不重复已经掌握的基础；核心 kernel 和实验代码由学习者亲手完成；每个主题都以“能解释、能实验、能取证、能说明边界”为完成标准。

## 文档导航

| 行范围 | 内容 |
| ---: | --- |
| 31～65 | 文档目标与完整证据链 |
| 66～101 | 七层能力总览与当前缺口 |
| 102～116 | 推荐学习顺序 |
| 117～181 | Kernel 编程维护项 |
| 182～252 | Kernel 优化与 Tensor Core |
| 253～352 | 编译链、PTX/SASS、资源与调度 |
| 353～435 | Stream、Event、异步与并发 |
| 436～531 | CUDA Graph |
| 532～612 | Pinned、Unified Memory 与 memory pool |
| 613～700 | `ncu`、`nsys` 与 Compute Sanitizer |
| 701～750 | 七层能力综合实验 |
| 751～790 | 总验收矩阵 |
| 791～807 | 本轮冻结主题 |
| 808～839 | 配套资料入口 |
| 840～842 | 最后执行口令 |

---

## 1. 为什么整理这份文档

当前已经具备较强的单 kernel 编程和优化能力，但 CUDA 不只包含“怎样把一个 kernel 写快”。完整的 CUDA 性能工程能力还包括：

```text
GPU 内部：线程怎样执行、数据怎样流动、指令怎样调度
GPU 外部：CPU 怎样提交任务、stream 怎样表达顺序与并发
运行时：内存怎样分配、迁移和复用
工程工具：怎样证明正确性、定位瓶颈、验证优化
```

因此，接下来的目标不是继续横向收集更多算子，而是把现有能力补成一条完整证据链：

```text
CUDA C++ 源码
    ↓
正确性与边界
    ↓
可信 benchmark
    ↓
PTX / SASS / ptxas 资源
    ↓
Nsight Compute 单 kernel 诊断
    ↓
Stream / Event / CUDA Graph 调度
    ↓
Nsight Systems 端到端时间线
```

### 一句话记忆

> 已经会“把一个 kernel 跑快”，接下来补“证明它为什么快，以及让一串 kernel 高效地跑起来”。

---

## 2. 七层能力总览

### 2.1 状态定义

| 状态 | 含义 |
| --- | --- |
| 🟢 已形成能力 | 有本人代码、正确性结果和性能/底层证据，不需要重新系统学习 |
| 🟡 部分掌握 | 用过或写过，但缺少完整语义、实验结果或可复述证据链 |
| ⚪ 尚未闭环 | 只有概念或教材，缺少本人实验和验收 |

### 2.2 当前能力盘点

| 层次 | 当前状态 | 已有基础 | 主要缺口 | 优先级 |
| --- | --- | --- | --- | --- |
| Kernel 编程 | 🟢 | thread/block/warp、shared memory、同步、atomic、reduction、scan、GEMM、Attention、GEMV | memory ordering、错误模型和同步边界随实验复习 | 维护 |
| Kernel 优化 | 🟢/🟡 | coalescing、bank conflict、occupancy、register tiling、vectorized load、double buffering | Tensor Core 指令路径、`ldmatrix`、`mma.sync`、`cp.async` 证据 | P0 |
| 编译与指令 | 🟡 | 已生成 PTX、cubin、SASS 和 ptxas 资源报告 | PTX/SASS 数据流、register/spill/local、scheduler/scoreboard、HMMA | P0 |
| 执行调度 | 🟡 | stream、pinned memory、异步拷贝、多 stream 实验 | Event 依赖、默认 stream、隐式同步、并发条件、`nsys` 验证 | P1 |
| CUDA Graph | 🟡 | 已写 capture、instantiate、replay 核心流程 | 错误检查、正式 benchmark、`nsys` timeline、动态更新边界 | P1 |
| 内存管理 | 🟡 | `cudaMalloc`、pinned memory、Unified Memory、prefetch | `cudaMallocAsync`、`cudaFreeAsync`、memory pool、同步与尾延迟 | P2 |
| 性能工具 | 🟢/🟡 | `ncu`、`memcheck`、CUDA Event 已实战 | `nsys`、`racecheck`、`synccheck`、`initcheck` 的系统闭环 | 贯穿 |

### 2.3 真正需要补的主干

```text
主线 A：PTX/SASS → register/spill → scheduler/scoreboard → Tensor Core 指令
主线 B：Stream/Event → 异步与并发 → Nsight Systems
主线 C：CUDA Graph → replay → launch overhead → Nsight Systems
主线 D：cudaMallocAsync → memory pool → allocator 开销
工具线：ncu + nsys + Compute Sanitizer 全程取证
```

Kernel 基础、coalescing、bank conflict、普通 register tiling 和基础 `ncu` 不再单独开课。

---

## 3. 推荐学习顺序

| 阶段 | 主题 | 建议投入 | 结束标志 |
| --- | --- | ---: | --- |
| 1 | 编译链、PTX/SASS、资源与调度诊断 | 5～7 天 | 能独立完成一个 kernel 的源码→指令→指标证据链 |
| 2 | Tensor Core 指令路径 | 2～3 天 | 能讲清 WMMA→PTX MMA→SASS HMMA，并有本人证据 |
| 3 | Stream、Event、异步与并发 | 2～3 天 | 能用 `nsys` 证明顺序、依赖与重叠 |
| 4 | CUDA Graph | 1～2 天 | plain 与 graph 正确性一致，有时间和 timeline 对比 |
| 5 | 异步内存分配与 memory pool | 1～2 天 | 能解释 stream-ordered allocation，并完成三种分配策略对比 |
| 6 | Sanitizer 工具闭环 | 1 天并贯穿前五阶段 | 能给不同错误选择正确工具并解释报告 |

这不是要求连续新增 14 天任务。当前首先完成[Week 2 PTX/SASS 执行清单](current/Week2_PTX_SASS与性能诊断.md)，其余模块在现有四周主线允许时插入，不能同时开启多个 WIP。

---

## 4. 模块一：Kernel 编程

### 4.1 当前结论

本模块已经达到“以项目持续维护”为主的阶段，不需要从 vector add 重新学习。

已经覆盖：

- 一维、二维 thread/block/grid 映射；
- shared memory、`__syncthreads()` 和 tile 协作；
- atomic、warp shuffle、ballot；
- reduction、scan、histogram；
- GEMM、softmax、Attention、GEMV；
- CPU reference、误差检查、CUDA Event 计时；
- 非整齐 shape 和基础 sanitizer。

### 4.2 只保留四个补强点

#### 同步范围

需要准确区分：

| 机制 | 保证什么 | 不保证什么 |
| --- | --- | --- |
| warp `_sync` primitive | 指定 mask 内参与线程的 warp 协作 | 不替代 block 级同步 |
| `__syncthreads()` | block 内线程到达 barrier，并建立相关 shared/global memory 可见性 | 不能同步不同 block |
| memory fence | 约束当前线程内存操作的可见顺序 | 本身不让其他线程等待 |
| stream ordering | 同一 stream 中设备任务按提交顺序执行 | 不等价于 Host 已等待任务完成 |

#### 错误模型

必须区分：

```text
launch/configuration error：通常由 cudaGetLastError() 检查
execution error：kernel 真正执行后才产生，常在后续同步 API 暴露
```

#### Grid 级假设

普通 kernel 不能假设所有 block 同时驻留，也不能用一个普通 global flag 随意模拟全 grid barrier。涉及跨 block 协作时要重新设计为多 kernel、atomic 协议或 cooperative launch。

#### 数值正确性

浮点结果不能只看一个绝对误差阈值。至少同时记录：

- `max_abs`；
- `max_rel`；
- NaN/Inf；
- 最差位置；
- dtype、累加顺序和 reference 精度。

### 4.3 维护性验收

后续每个新 kernel 都应满足：

- [ ] 小尺寸、非整齐尺寸和边界尺寸通过；
- [ ] 有 CPU 或库 reference；
- [ ] launch error 与 execution error 都被检查；
- [ ] `memcheck` 通过；
- [ ] 涉及 shared memory 或 warp 同步时，按需运行 `racecheck`/`synccheck`；
- [ ] 能口述每个 barrier 前后的生产者和消费者。

---

## 5. 模块二：Kernel 优化

### 5.1 已经形成的能力

现有 GEMM、Attention 和 GEMV 实验已经覆盖：

- global memory coalescing；
- shared-memory tiling；
- bank conflict 与 padding；
- register tiling 和数据复用；
- occupancy 与 register pressure；
- `float4` vectorized load；
- double buffering；
- memory-bound / compute-bound 判断；
- 用 `ncu` 验证假设而不是凭感觉优化。

其中 [GEMM ncu 记录](../week05_gemm_advanced/ncu_notes.md) 已经包含 bank conflict、occupancy、寄存器和不同 GPU 瓶颈变化的完整案例。

### 5.2 待补：Tensor Core 与异步流水

需要建立以下层级关系：

```text
CUDA C++ WMMA API
        ↓ 编译
PTX wmma.mma / mma.sync
        ↓ ptxas
SASS HMMA
        ↓ 硬件
Tensor Core pipeline
```

必须理解：

- WMMA 是 CUDA C++ API，不是硬件指令名称；
- PTX `mma.sync` 是虚拟 ISA 指令；
- A100 上实际执行的是 SASS HMMA 序列；
- fragment 是 warp 协作持有的逻辑矩阵片段；
- lane 与 fragment 元素的物理映射不能从普通数组直觉推断；
- `ldmatrix` 用于 warp 协作地从 shared memory 加载矩阵片段；
- `cp.async` 优化 global→shared 的搬运路径，不直接让 MMA 计算更快；
- double buffering 的价值是让 next tile 搬运与 current tile 计算重叠；
- 使用了异步指令不等于一定加速，仍需满足流水覆盖、资源和并行度条件。

### 5.3 实验对象

使用已有 [WMMA FP16 GEMM](../week06_tensorcore/wmma_fp16_gemm.cu)，不重新收集新的 Tensor Core 示例。

实验步骤：

1. 先完成 correctness，确认 FP16 输入、FP32 accumulate 和误差边界；
2. 生成 PTX、cubin、SASS 和 ptxas log；
3. 在 PTX 中定位 WMMA/MMA 指令；
4. 在 SASS 中定位 HMMA；
5. 检查当前实现的数据是否直接从 global memory 进入 fragment；
6. 解释为什么当前版本可能看不到 `LDSM`；
7. 用 `ncu` 确认 Tensor Core pipeline 活跃；
8. 与 cuBLAS 比较时只把 cuBLAS 当标尺，不以教学版超过 cuBLAS 为目标。

### 5.4 验收标准

- [ ] 能画出 WMMA→PTX→SASS 三层图；
- [ ] 能指出一条 PTX MMA 和对应的 SASS HMMA 区域；
- [ ] 能解释 fragment、lane 和矩阵 tile 的关系；
- [ ] 能解释 `ldmatrix` 解决什么数据搬运问题；
- [ ] 能解释 `cp.async` 的 prologue、steady state、epilogue；
- [ ] 能说明“指令出现”和“墙钟变快”为什么是两件事；
- [ ] 有 correctness、ptxas、SASS、`ncu` 和正常墙钟五类证据。

---

## 6. 模块三：编译与指令

### 6.1 本模块是当前最高优先级

当前已经完成编译链的第一步，但 PTX/SASS 数据流、资源、scheduler 和 scoreboard 还没有闭环。执行以[当前 Week 2 清单](current/Week2_PTX_SASS与性能诊断.md)为唯一入口。

### 6.2 编译链心智模型

```text
CUDA C++
   ↓ nvcc 前端 / NVVM
PTX：虚拟 ISA，表达 CUDA 线程、地址空间和指令语义
   ↓ ptxas 离线编译，或 Driver JIT
cubin：针对具体 sm_xx 的机器码容器
   ↓ cuobjdump / nvdisasm 反汇编
SASS：当前 GPU 实际机器指令的可读表示
```

必须能解释：

- `compute_80` 定义允许使用的 PTX 虚拟架构能力；
- `sm_80` 表示为 A100 生成具体机器码；
- PTX 不是 A100 直接执行的最终机器码；
- fatbin 可以同时携带一个或多个 cubin 以及 PTX；
- 有兼容 cubin 时通常直接加载，无合适 cubin 但有兼容 PTX 时可能 Driver JIT；
- `-lineinfo` 保留优化并提供源码关联信息；
- `-G` 面向 device debugging，可能显著改变优化和指令，不用于正式性能结论。

### 6.3 PTX/SASS 阅读顺序

不要逐行翻译，固定按数据流阅读：

```text
1. kernel 参数怎样加载
2. thread/block index 怎样计算
3. row/column 或 tile 坐标怎样形成
4. global 地址怎样计算
5. 数据怎样进入 register/shared
6. 主计算指令是什么
7. loop、predicate、barrier 在哪里
8. accumulator 怎样写回 global memory
```

每次分析必须区分三类结论：

| 证据 | 能证明 | 不能单独证明 |
| --- | --- | --- |
| PTX | 编译器虚拟 ISA 层表达的数据流和地址空间 | 最终机器指令、真实周期、实际瓶颈 |
| SASS | 当前源码、编译参数、架构下的真实指令形态 | 运行时吞吐、cache 命中和墙钟收益 |
| ptxas log | registers/thread、shared memory、spill 等静态资源 | 运行时 occupancy、eligible warp 和 stall |
| `ncu` | 被 profile launch 的运行时硬件指标 | 未 profile shape、端到端 CPU/GPU 调度成本 |
| 正常 benchmark | 当前条件下的实际时间 | 单凭时间无法确定底层原因 |

### 6.4 Register、spill 与 local memory

必须掌握：

- PTX 虚拟寄存器数量不等于 ptxas 最终物理 registers/thread；
- register tiling 增加复用，同时提高 register pressure；
- register pressure 可能降低 occupancy；
- 编译器无法放入寄存器的数据可能 spill 到 local address space；
- local 是线程私有地址空间，但物理存储可能位于 device memory 层次并经过 cache；
- local load/store 或 spill 出现不一定自动证明它是主瓶颈；
- 强行使用 `--maxrregcount` 可能用 occupancy 换来更多 spill，必须实测。

### 6.5 Scheduler 与 scoreboard

核心概念：

| 概念 | 含义 |
| --- | --- |
| Active warp | 已驻留在 SM 上、尚未完成的 warp |
| Eligible warp | 当前周期依赖和资源条件允许发射的 warp |
| Issued warp | 当前周期真正被 scheduler 选择并发射的 warp |
| Scoreboard | 跟踪指令数据依赖与结果就绪状态的硬件机制 |
| Latency hiding | 当前 warp 等待时发射其他 eligible warp 的工作方式 |

重要边界：

- occupancy 高只代表可能驻留的 warp 多，不代表 eligible warp 一定多；
- long scoreboard 表示依赖等待较长，但不能直接等同于 DRAM 带宽打满；
- short scoreboard 也必须结合 shared/L1、依赖链和指令路径解释；
- stall 百分比只能作为候选机制，最终要结合 SASS、吞吐、资源和墙钟；
- profiler replay 时间不能当正常 wall-clock 性能。

### 6.6 本模块最终验收

- [ ] 闭卷画出 CUDA C++→PTX→SASS/JIT；
- [ ] 独立生成 PTX、ptxas log、cubin 和 SASS；
- [ ] 对 naive GEMM 追踪 index→load→FMA→store；
- [ ] 对五版 GEMM 写出机器数据流差异；
- [ ] 证明 `float4` 是否形成真实宽加载；
- [ ] 证明异步版本是否出现目标搬运指令；
- [ ] 完成一个 register/ILP 单变量实验；
- [ ] 区分 active、eligible、issued warp；
- [ ] 正确解释 scoreboard 和 stall 的证据边界；
- [ ] 完成 WMMA→PTX MMA→SASS HMMA 观察。

---

## 7. 模块四：执行调度

### 7.1 为什么需要补

单 kernel 优化回答“GPU 内部怎样跑得更快”；执行调度回答“CPU、copy engine 和 GPU 怎样同时推进”。这是 CUDA Graph、推理 decode 和端到端性能分析的前置知识。

### 7.2 Stream 的核心语义

> 同一 stream 内有顺序保证；不同 stream 之间没有天然顺序保证，并且只在依赖与硬件资源允许时才可能并发。

“可以并发”不等于“一定并发”。并发还受以下条件限制：

- kernel 是否占满 SM 资源；
- GPU 是否支持所需 copy/compute overlap；
- Host memory 是否 pinned；
- 是否混入 legacy default stream；
- 是否存在 `cudaDeviceSynchronize()`、同步 memcpy、分配释放等隐式/显式同步；
- 任务粒度是否足够大，能覆盖调度开销；
- H2D、D2H 和 kernel 的依赖关系是否允许重叠。

### 7.3 Event 的两个角色

#### 计时

CUDA Event 记录的是它所在 stream 的进度点。开始和结束 Event 必须放在被测工作具有正确顺序关系的 stream 中，否则时间范围可能错误。

#### 跨 stream 依赖

推荐用：

```text
producer stream: work → cudaEventRecord(event)
consumer stream: cudaStreamWaitEvent(event) → dependent work
```

这样依赖留在设备侧，不需要 Host 通过 `cudaDeviceSynchronize()` 阻塞并破坏并发。

### 7.4 必做实验

使用现有 [stream overlap 实验](../week03_parallel/stream_overlap/stream_overlap.cu)作为对象，先 review 正确性和计时，再由学习者亲手修正或重写关键调度部分。

对比四种版本：

| 版本 | Host memory | Stream | 预期用途 |
| --- | --- | --- | --- |
| A | pageable | 单 stream / 同步 API | 串行 baseline |
| B | pinned | 单 stream | 排除 pageable staging 影响 |
| C | pinned | 多 stream 分块 | 尝试 H2D→kernel→D2H pipeline |
| D | pinned | 多 stream + Event 依赖 | 显式表达跨 stream producer/consumer |

每个版本必须：

- CPU 对拍；
- 正常 wall-clock 或正确 Event 计时；
- 用 `nsys` 观察 H2D、kernel、D2H；
- 记录是否真正 overlap；
- 若没有 overlap，解释是资源、依赖、Host memory、默认 stream 还是粒度问题。

### 7.5 Nsight Systems 验收问题

打开 timeline 后必须能指出：

1. 哪个 Host thread 调用了哪个 CUDA API；
2. CUDA API 调用时长与 GPU kernel 时长为什么不是一回事；
3. 每个操作属于哪个 stream；
4. kernel 之间是否有 launch gap；
5. H2D、kernel、D2H 是否重叠；
6. GPU idle 是 CPU 提交慢、同步、依赖还是工作不足；
7. `cudaDeviceSynchronize()` 在哪里阻断 Host；
8. 多 stream 没有加速时，时间线给出了什么反证。

### 7.6 本模块验收

- [ ] 能解释 legacy default stream 与 per-thread default stream；
- [ ] 能解释同 stream 顺序和跨 stream 无天然顺序；
- [ ] 能用 Event 建立跨 stream 依赖；
- [ ] 能解释 pageable memory 为什么妨碍可靠异步传输；
- [ ] 能列出常见隐式同步来源；
- [ ] 能用 `nsys` 证明是否 overlap，而不是只比较总时间；
- [ ] 能说明 kernel concurrency 为什么受 SM 资源限制。

---

## 8. 模块五：CUDA Graph

### 8.1 CUDA Graph 解决什么

普通循环每轮逐个提交多个短 kernel：

```text
CPU：launch A → launch B → launch C → ...
GPU：   A   gap   B   gap   C
```

CUDA Graph 把固定任务 DAG 录制并实例化，稳态每轮只需一次 graph replay：

```text
Capture → Instantiate → Replay → Replay → Replay
```

它主要减少：

- CPU 重复提交工作；
- 多个短 kernel 之间的 launch gap；
- 固定任务序列的重复准备成本。

它不会直接减少：

- 单个 kernel 的 FLOP；
- 单个 kernel 的显存读写；
- 单个 kernel 内部的 bank conflict；
- 单个 kernel 的指令延迟。

### 一句话记忆

> CUDA Graph 不让单个 kernel 算得更快，而是让一串固定 kernel 被更低开销地反复提交。

### 8.2 生命周期

```text
cudaStreamBeginCapture
    ↓
向 stream 提交固定操作序列
    ↓
cudaStreamEndCapture → cudaGraph_t
    ↓
cudaGraphInstantiate → cudaGraphExec_t
    ↓
cudaGraphLaunch 重放多次
    ↓
cudaGraphExecDestroy / cudaGraphDestroy
```

要区分：

- `cudaGraph_t` 是图的定义；
- `cudaGraphExec_t` 是实例化后的可执行图；
- capture 和 instantiate 属于准备成本；
- replay 才是稳态路径；
- benchmark 不能把一次性准备成本混入稳态 replay，除非明确测端到端首轮延迟。

### 8.3 必做实验

使用已有 [decode graph 实验](../week05_inference/decode_graph.cu)。核心 capture/replay 已经写出，但仍需完成工程验收：

1. 给所有 Graph API 加统一 CUDA 错误检查；
2. plain loop 与 graph replay 使用完全相同的一步操作；
3. 两者从相同初始数据开始；
4. CPU reference、plain、graph 三方对拍；
5. warmup 后分别测稳态总时间和每 step 时间；
6. 记录 plain/graph speedup；
7. 用 `nsys` 对比 kernel gap 与 CUDA API 提交；
8. 改变每步 kernel 数量，观察 Graph 收益趋势；
9. 增大单 kernel 工作量，观察 Graph 收益是否缩小。

### 8.4 动态性边界

第一阶段只要求知道：

- 数据内容变化通常不要求改变 graph 拓扑；
- kernel 参数、指针或节点参数变化可以研究 node parameter update 或 `cudaGraphExecUpdate()`；
- 拓扑和控制流频繁变化可能使 capture/update 成本抵消收益；
- capture 期间某些同步、分配或跨线程行为受到限制；
- 具体支持范围随 CUDA 版本变化，工程使用时查对应 Runtime API 文档。

暂不深入 conditional node、child graph、graph memory node 和复杂手工 node 管理。

### 8.5 本模块验收

- [ ] 能解释 capture、instantiate、replay 各自作用；
- [ ] 能区分 `cudaGraph_t` 与 `cudaGraphExec_t`；
- [ ] plain 和 graph 结果一致；
- [ ] benchmark 排除了 capture/instantiate 的稳态外成本；
- [ ] `nsys` 能看到 replay 后 CPU 提交和 kernel gap 的变化；
- [ ] 能解释 kernel 变长后 Graph 收益为什么下降；
- [ ] 能说明 Graph 适合固定重复 DAG，不适合高度动态且不重复的序列。

---

## 9. 模块六：内存管理

### 9.1 已有基础

现有实验已经涉及：

- `cudaMalloc()` / `cudaFree()`；
- pageable 与 pinned Host memory；
- `cudaMemcpy()` / `cudaMemcpyAsync()`；
- Unified Memory；
- page migration；
- `cudaMemPrefetchAsync()`。

Unified Memory 可使用[现有实验](../week02_memory/unified_memory_demo/unified_memory_demo.cu)复习，不再重新写一套相同 demo。

### 9.2 Pinned memory

必须理解：

- pageable Host memory 不能被 DMA engine 直接稳定地异步访问，Runtime 可能需要 staging；
- pinned memory 为异步 H2D/D2H 和 copy/compute overlap 提供前提；
- pinned memory 是有限系统资源，过量使用会影响 OS；
- pinned memory 只是必要条件之一，不保证一定 overlap；
- 零拷贝、mapped memory 与普通 pinned staging 是不同概念。

### 9.3 Unified Memory

必须理解：

- `cudaMallocManaged()` 提供统一虚拟地址，不代表数据同时物理存在于 CPU 和 GPU；
- 首次访问可能触发 page fault 和 migration；
- oversubscription、访问模式和 CPU/GPU 交替访问可能造成抖动；
- prefetch 和 memory advice 是提示/控制迁移行为的工具；
- Unified Memory 主要改善编程与数据管理，不自动保证最佳性能。

### 9.4 Stream-ordered allocator

传统分配释放可能带来较高 Runtime/Driver 成本和同步影响。需要补：

- `cudaMallocAsync()`；
- `cudaFreeAsync()`；
- 分配和释放被排序到指定 stream；
- 默认 memory pool；
- pool 中已保留内存的复用；
- release threshold；
- reserved memory 与 used memory；
- 跨 stream 复用为什么需要正确依赖；
- allocator 对高频请求和推理服务尾延迟的影响。

### 9.5 必做实验

在同一个进程内重复执行固定大小与混合大小分配，比较：

| 方案 | 分配策略 | 要观察什么 |
| --- | --- | --- |
| A | 每轮 `cudaMalloc()` + `cudaFree()` | API 开销与潜在同步 |
| B | `cudaMallocAsync()` + `cudaFreeAsync()` | stream ordering 与 pool 稳态复用 |
| C | 启动时预分配，循环内复用 | 最小稳态 allocator 开销 |

实验要求：

- 初始化 CUDA context 后再开始正式测量；
- 区分首轮、稳态和释放阶段；
- 不在 sanitizer 插桩下比较性能；
- 用 `nsys` 看 CUDA API 时间和同步；
- 查询并记录 pool 的 reserved/used 指标；
- 分配得到的内存必须被实际使用，避免实验被简化成无意义 API 循环；
- 说明预分配最快并不意味着它适合所有动态 workload。

### 9.6 本模块验收

- [ ] 能解释 pageable 与 pinned memory；
- [ ] 能解释 Unified Memory page migration；
- [ ] 能解释 prefetch 为什么可能改善首次 GPU 访问；
- [ ] 能解释 `cudaMallocAsync()` 的 stream-ordered 语义；
- [ ] 能区分 pool reserved 与 used memory；
- [ ] 能说明跨 stream allocator 复用需要依赖关系；
- [ ] 有 traditional/async/preallocated 三种策略的实测对比。

---

## 10. 模块七：性能与正确性工具

### 10.1 工具分工

| 工具 | 回答的问题 | 不应该拿它做什么 |
| --- | --- | --- |
| CUDA Event | 指定 stream 上一段 GPU 工作经过多久 | 解释 CPU launch gap 或整个系统时间线 |
| 正常 wall-clock | 用户或 Host 看到的端到端时间是多少 | 单独推断 kernel 内部瓶颈 |
| Nsight Compute | 单个 kernel 内部为什么慢 | 分析整个进程的 CPU/GPU 调度 |
| Nsight Systems | 整个程序哪里有空洞、同步和重叠 | 取代 `ncu` 的深层 kernel 指标 |
| Compute Sanitizer | 是否有内存、竞态、同步或初始化错误 | 比较真实性能 |
| PTX/SASS 工具 | 编译器生成了什么数据流和指令 | 单凭指令存在宣称实际加速 |

### 10.2 Nsight Compute 最低流程

现有能力继续按以下顺序固化：

```text
1. 正常运行确认正确性和墙钟
2. 选择准确 kernel 与 launch
3. 看 Speed of Light 高层画像
4. 看 Memory Workload / Compute Workload
5. 看 Launch Statistics 与 Occupancy
6. 看 Warp State / Scheduler / Stall
7. 回到 Source / SASS 建立机制解释
8. 提出一个单变量假设
9. 正常运行复测墙钟
```

必须避免：

- 把 `ncu` replay 后的 duration 当正式性能；
- 只看 achieved occupancy；
- 只看最高 stall 百分比；
- profile 错误的 warmup/shape/版本；
- 一次修改多个变量后强行归因。

### 10.3 Nsight Systems 最低流程

重点观察：

- CUDA API；
- OS Runtime / Host thread；
- stream 与 device timeline；
- H2D/D2H；
- kernel launch gap；
- Host synchronization；
- CUDA Graph launch；
- NVTX 标注范围。

建议顺序：

```text
先用 nsys 找：时间花在哪里、GPU 为什么空闲
再用 ncu 查：确定的热点 kernel 内部为什么慢
```

### 10.4 Compute Sanitizer 四种模式

| 模式 | 重点问题 | 推荐实验对象 |
| --- | --- | --- |
| `memcheck` | 越界、非法地址、泄漏等 | 所有新 kernel |
| `racecheck` | shared memory 数据竞态 | shared tiled GEMM、reduction |
| `synccheck` | barrier、warp 同步使用错误 | reduction、warp primitive、分支内 barrier |
| `initcheck` | 未初始化 device memory 读取 | 故意漏初始化的最小 demo |

可以使用 [race 示例](../week03_parallel/race.cu)和现有 shared-memory kernel 构造最小错误，不要在大型程序里盲目翻报告。

### 10.5 工具使用纪律

- sanitizer 只判断正确性，不记录性能；
- profiler 只采集必要 launch，避免报告淹没重点；
- benchmark 与 profiler 分开运行；
- 每份报告记录 GPU、CUDA、编译参数、shape、kernel 名称；
- profiler 指标必须与正常墙钟互相印证；
- 报告中的每个结论都写清“能证明什么、不能证明什么”。

### 10.6 本模块验收

- [ ] 能根据问题选择 Event、wall-clock、`ncu`、`nsys` 或 sanitizer；
- [ ] 能用 `ncu` 完成一次从 SOL 到 SASS 的诊断；
- [ ] 能用 `nsys` 找到 launch gap、同步和 overlap；
- [ ] 能分别制造并识别 memory、race、sync、uninitialized-read 错误；
- [ ] 不在 sanitizer 或 profiler 插桩下汇报正式性能；
- [ ] 每个性能结论都有正常墙钟和至少一种机制证据。

---

## 11. 综合实验：把七层能力串起来

最终不再新造一个大型项目，而是选择已有 decode graph 或 Attention 流水作为综合对象。

### 11.1 推荐数据流

```text
Host 准备 pinned input
    ↓
多 stream H2D
    ↓ Event dependency
短 kernel 序列
    ↓
CUDA Graph capture / replay
    ↓
D2H 或后续消费
```

### 11.2 综合取证顺序

1. CPU/reference 确认结果；
2. `memcheck`，按共享与同步情况增加 `racecheck`/`synccheck`；
3. 正常 wall-clock 和 CUDA Event 分层计时；
4. `nsys` 找 CPU 提交、stream、同步和 idle；
5. 对热点 kernel 使用 `ncu`；
6. 对关键 kernel 导出 PTX/SASS；
7. 只修改一个变量；
8. 重新跑 correctness、wall-clock 和对应 profiler；
9. 记录成功或失败的优化；
10. 写清硬件、shape、dtype、版本和适用边界。

### 11.3 综合验收口述

最终应能在 10 分钟内完整回答：

```text
这个 workload 做什么？
正确性怎样保证？
端到端时间花在哪里？
热点 kernel 内部瓶颈是什么？
PTX/SASS 提供了什么证据？
stream/event 如何表达依赖？
CUDA Graph 减少了什么开销？
memory allocator 是否进入稳态路径？
优化前后数据是什么？
结论在哪些条件下成立？
```

---

## 12. 总验收矩阵

每项按 0～2 分评价：

```text
0：不会或只有名词印象
1：看提示能解释/完成
2：闭卷能解释，有本人实验和证据
```

| 能力 | 当前估计 | 通过要求 | 本人复测 |
| --- | ---: | ---: | ---: |
| thread/block/warp 与边界 | 2 | 2 |  |
| shared/sync/atomic | 2 | 2 |  |
| coalescing/bank conflict | 2 | 2 |  |
| occupancy/register tiling | 2 | 2 |  |
| Tensor Core 指令路径 | 1 | 2 |  |
| CUDA 编译链 | 1 | 2 |  |
| PTX/SASS 数据流 | 1 | 2 |  |
| register/spill/local | 1 | 2 |  |
| scheduler/scoreboard | 0～1 | 2 |  |
| stream/default stream | 1 | 2 |  |
| Event 跨 stream 依赖 | 1 | 2 |  |
| memcpy/kernel overlap | 1 | 2 |  |
| CUDA Graph | 1 | 2 |  |
| pinned/Unified Memory | 1～2 | 2 |  |
| async allocator/memory pool | 0 | 2 |  |
| Nsight Compute | 2 | 2 |  |
| Nsight Systems | 0～1 | 2 |  |
| Compute Sanitizer 四模式 | 1 | 2 |  |

通过规则：

- 所有 P0/P1 项达到 2；
- P2 项至少达到 1，并有明确补做实验；
- 不能只靠阅读或口述给 2 分；
- 代码、数据、工具报告和证据边界至少具备三类才算闭环。

---

## 13. 明确不纳入本轮的内容

为了防止再次发散，本轮不把以下主题加入主线：

- 多 GPU、NCCL、NVSHMEM；
- GPUDirect RDMA；
- MPS、MIG、CUDA IPC；
- Driver API 深入和自定义 JIT 系统；
- Dynamic Parallelism；
- 图形互操作、texture/surface 专题；
- Hopper TMA/WGMMA 实现；
- 生产级 CUTLASS、FlashAttention 或 vLLM 源码复刻。

这些并非不重要，而是与当前“单 GPU CUDA 性能证据链 + 推理调度基础”的主目标相比优先级更低。

---

## 14. 资料入口

### 当前主线

- [Week 2 PTX/SASS 与性能诊断](current/Week2_PTX_SASS与性能诊断.md)
- [PTX/SASS Worklog](../notes/week02_ptx_sass.md)
- [PTX/SASS 深水区教材](../docs/topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)
- [Nsight Compute 详解](../docs/topics/performance/Nsight_Compute_ncu详解.md)

### 执行调度与 Graph

- [Stream 与异步执行模型](../cuda_deep_course/course/volume07_async_system/01_Stream与异步执行模型.md)
- [Event、并发 Kernel 与依赖](../cuda_deep_course/course/volume07_async_system/03_Event_并发Kernel与依赖.md)
- [CUDA Graph](../cuda_deep_course/course/volume07_async_system/04_CUDA_Graph.md)
- [Stream overlap 实验](../week03_parallel/stream_overlap/stream_overlap.cu)
- [Decode Graph 实验](../week05_inference/decode_graph.cu)

### Tensor Core 与内存

- [WMMA FP16 GEMM](../week06_tensorcore/wmma_fp16_gemm.cu)
- [Tensor Core 学习记录](../week06_tensorcore/README.md)
- [Unified Memory 实验](../week02_memory/unified_memory_demo/unified_memory_demo.cu)
- [CUDA 内存模型笔记](../notes/CUDA内存模型详解.md)

### 工具

- [Nsight Compute 与 Compute Sanitizer](../cuda_deep_course/course/volume05_performance/05_Nsight_Compute与Compute_Sanitizer.md)
- [GEMM ncu 实战记录](../week05_gemm_advanced/ncu_notes.md)
- [Attention ncu 流水记录](../week04_attention/ncu_pipeline_notes.md)

---

## 15. 最后执行口令

> 先完成当前 PTX/SASS 主线；然后依次补 Stream/Event、CUDA Graph 和 memory pool。每学一项都必须留下本人代码、正确性、正常墙钟、工具证据和适用边界，不再用“看过”代替“掌握”。
