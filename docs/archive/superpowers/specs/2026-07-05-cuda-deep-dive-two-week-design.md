# CUDA 深水区两周教材设计

## 目标

编写一份可在约两周内高强度学完的纯 CUDA 教材，帮助已经完成中级 CUDA 性能训练的学习者，从 CUDA C++ 算法与存储优化继续下钻到编译器、机器指令、warp 级 Tensor Core 指令、异步流水和 GPU 调度，并建立从 A100 到 Hopper 的架构迁移视角。

教材不讨论模型、Attention、推理或其他 ML 内容。

## 读者起点

读者已经完成并有实验记录的能力包括：

- grid/block/warp 执行模型；
- coalescing、shared memory、bank conflict 与 padding；
- atomic、shuffle、ballot、reduction、scan、histogram；
- naive、shared tiled、2D register tiled、float4、double-buffered GEMM；
- A100 上的 occupancy、寄存器压力、Roofline 与 Nsight Compute 调优；
- WMMA FP16 GEMM，以及用 HMMA SASS 和 Tensor pipe 指标验证 Tensor Core 路径。

因此教材不重复 CUDA 入门，而是解释这些现有成果的下一层硬件和指令原因。

## 交付形式

- 一份自包含 Markdown 文档；
- 建议路径：`docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md`；
- 按 14 天组织，但正文同时按知识模块保持连续；
- 目标是每天约 2–4 小时的阅读、编码、反汇编与 profiling；
- 每个模块都连接到项目里已有的 GEMM/WMMA 源码和实验记录。

## 教学原则

每个主题按同一闭环展开：

```text
问题与现象
→ 硬件/编译器模型
→ 最小例子
→ 工具命令
→ 现有项目对照
→ 学习者手写核心空位
→ PTX/SASS/ncu 证据
→ 常见错误
→ 验收题和闭卷口述
```

教材提供可编译的外围框架、参考结果、测试方法和分级提示；`mma.sync`、`ldmatrix`、原生 `cp.async` group 与 stage 调度等核心部分由学习者实现，不在主线中直接给出完整答案。

## 模块一：PTX、SASS 与编译器证据链（Day 1–3）

### 学习目标

- 理解 CUDA C++、NVVM、PTX、ptxas、cubin/fatbin、SASS 和 driver JIT 的关系；
- 分清 PTX 虚拟 ISA 与 SASS 真实机器指令；
- 会生成、反汇编、筛选和关联源码；
- 能识别算术、global/shared/local 访存、控制流、谓词、同步和 Tensor Core 指令；
- 能用 `ptxas -v`、SASS 和 ncu 判断寄存器分配、spill、向量化是否生效；
- 能用指令级证据解释两版 kernel 的性能差异。

### 实验主线

依次对比项目中的 naive GEMM、shared GEMM、2D register tiled GEMM、float4 GEMM 和 WMMA GEMM：

- global load/store 数量与宽度；
- FFMA 与地址计算；
- LDS/STS 与 shared memory 数据流；
- 寄存器数量和 local spill；
- WMMA 到 HMMA/MMA 的 lowering；
- 源码优化是否真正落成预期指令。

### 边界

要求会读、会查、会写少量 inline PTX，不要求纯 PTX 编写完整 kernel，也不要求掌握 SASS 二进制编码。

## 模块二：`ldmatrix`、`mma.sync` 与 warp fragment（Day 4–6）

### 学习目标

- 理解 MMA 的 `M×N×K` 指令形状和数据类型；
- 理解一个 warp 如何协作持有 A、B、C/D fragment；
- 理解逻辑矩阵坐标、shared memory layout、lane 与寄存器编号之间的映射；
- 理解 `ldmatrix.sync.aligned` 为何适合 shared-to-register 矩阵加载及转置变体；
- 能阅读 `mma.sync.aligned` 和 `ldmatrix` 的 PTX operand 约束；
- 从最小 microkernel 逐步接到 shared tiled GEMM。

### 实验主线

1. 用 WMMA 输出和 SASS 建立已知基线；
2. 纸上推演 lane/fragment 映射；
3. 编写 shared tile 初始化与可观测的 `ldmatrix` 加载实验；
4. 留出 inline PTX operand 绑定空位，由学习者完成最小 MMA microkernel；
5. 与 CPU/cuBLAS 参考结果对齐；
6. 用 SASS 验证预期矩阵指令；
7. 分析正确但不快的 microkernel 缺少哪些数据复用和流水。

## 模块三：A100 原生 `cp.async` 多级流水（Day 7–9）

### 学习目标

- 理解 synchronous global→register→shared 与 `cp.async` global→shared 的差别；
- 区分 `cp.async.ca`、`cp.async.cg`、对齐、粒度和缓存路径；
- 理解 `commit_group`、`wait_group N` 和 async group 的生命周期；
- 掌握 prologue、steady state、epilogue 三段式 pipeline；
- 从现有 `cuda::pipeline` 两 stage 实现下钻到原生指令语义；
- 实现并比较 2-stage、3-stage，必要时探索 4-stage；
- 量化 stage 数增加带来的 latency hiding、shared memory 占用和 occupancy 权衡。

### 实验主线

以项目中的 double-buffered GEMM 为起点：

- 先画清当前两 buffer 的所有权和同步点；
- 建立同步加载基线；
- 留出原生 `cp.async`、commit 与 wait 空位；
- 扩展为环形 stage；
- 用正确性检查与 compute-sanitizer 排除越界和同步错误；
- 用 ncu 比较 stall、occupancy、shared memory 和吞吐变化。

### 架构边界

本模块以 A100/sm_80 为准，不把 Hopper TMA 的 bulk copy 或 cluster barrier 语义混入 Ampere 主线。

## 模块四：Warp Scheduler、Scoreboard 与 Stall（Day 10–11）

### 学习目标

- 理解 SM subpartition、warp scheduler、eligible/active/issued warp 的关系；
- 理解 scoreboard 如何跟踪数据依赖；
- 区分 latency、throughput、dependency、occupancy 和 ILP/TLP；
- 理解 long scoreboard、short scoreboard、wait、barrier、not selected、math/dispatch throttle 等 stall 类别的含义和局限；
- 避免把单个 stall 百分比直接当成根因；
- 能从 ncu 现象回到指令和源码提出可证伪假设。

### 实验主线

使用四类 microbenchmark 和现有 GEMM：

- 独立 FMA 链与依赖 FMA 链：观察 ILP；
- coalesced global load 后立即消费与穿插独立计算：观察 scoreboard；
- 不同 block/register 配置：区分 occupancy 与 issue 能力；
- 两版 GEMM：建立“指标 → 指令 → 源码 → 修改 → 复测”的诊断表。

## 模块五：Hopper 架构迁移（Day 12–13）

### 学习目标

从 Ampere 已掌握机制出发理解 Hopper 为什么改变高性能 kernel 的组织方式，而不是孤立背诵新名词。

### 内容

- `cp.async` 与 TMA：线程协作搬运和描述符驱动 bulk tensor copy 的差异；
- warp MMA 与 WGMMA：warp 到 warp-group、同步到异步矩阵计算的变化；
- Thread Block Cluster：超越单 CTA 的协作范围；
- Distributed Shared Memory：cluster 内 CTA 地址空间与访问语义；
- transaction barrier / mbarrier 在 TMA pipeline 中的作用；
- producer/consumer warp specialization；
- 为什么 Hopper kernel 常出现更深流水、更明确的角色分工和更复杂的资源编排；
- 如何阅读 CUTLASS/CuTe 或其他 Hopper kernel 的 mainloop，而不要求在 A100 上运行。

### 边界

提供架构图、Ampere/Hopper 对照表、伪代码、术语翻译和源码阅读导航；不要求编写或离线编译完整 sm_90 kernel，也不假装没有 H100 就能完成真实性能结论。

## Day 14：综合证据链

选择项目中一个 GEMM kernel，形成完整分析报告：

```text
源码数据流
→ PTX 关键段
→ SASS 关键段
→ ptxas 资源报告
→ ncu 指标
→ 瓶颈假设
→ 一项修改
→ 正确性和性能复测
→ 如果迁移到 Hopper，哪些部分会被 TMA/WGMMA 重构
```

## 可视化设计

正文至少包含以下 Mermaid 或 ASCII 图：

1. CUDA C++ → PTX → SASS → 执行的编译链；
2. 一个 warp 中 lane、fragment 与寄存器的关系；
3. `ldmatrix` 的 shared→register 数据路径；
4. 2-stage 与 3-stage pipeline 时间线；
5. scheduler/scoreboard 的依赖等待图；
6. Ampere `cp.async+mma` 与 Hopper `TMA+WGMMA` 主循环对照；
7. cluster 与 DSM 的空间关系。

可视化服务仅用于设计阶段，最终 Markdown 文档不依赖浏览器服务即可阅读。

## 练习与答案策略

- 核心实现使用 `TODO 1..N` 标记；
- 每个练习提供三级提示：数据依赖提示、operand/layout 提示、接近代码的伪代码提示；
- 主文档不直接给完整核心答案；
- 提供编译命令、运行参数、参考输出形态和失败排查表；
- 如果后续需要完整答案，单独生成参考实现，避免学习时直接看到。

## 资料与准确性

涉及 PTX 指令签名、架构支持范围和 Nsight 指标时，以 NVIDIA 官方 PTX ISA、CUDA C++ Programming Guide、Ampere/Hopper Tuning Guide、Nsight Compute 文档和 Hopper Architecture Whitepaper 为主要依据。正文应区分：

- PTX 语义保证；
- 某个编译器版本的实际 lowering；
- A100 实测结论；
- Hopper 架构层面的推断或官方描述。

## 验收标准

- 文档为一份连续、自包含、纯 CUDA 的两周教材；
- 不重复基础 CUDA，也不默认读者已经会 PTX/SASS；
- 1–4 模块都有概念、实验、工具、手写空位、验证和自测；
- Hopper 能从 A100 已学机制自然迁移，不是名词列表；
- 所有命令与项目现有路径对应；
- 数值、指令名、架构支持和工具参数经官方资料核对；
- Markdown 标题、围栏、Mermaid 和本地链接有效；
- 不修改用户已有 CUDA 源码和未提交工作；本轮只新增这一份教材，练习骨架以内嵌代码块提供，不额外创建源码文件。
