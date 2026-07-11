# CUDA 深水区两周教材 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一份可在约两周内学完的纯 CUDA 深度教材，覆盖 PTX/SASS、`ldmatrix+mma.sync`、A100 原生 `cp.async` 多级流水、scheduler/scoreboard/stall 与 Hopper 架构迁移。

**Architecture:** 单一 Markdown 文档按五个知识模块和 14 天节奏组织。每章复用项目现有 GEMM/WMMA 作为观察对象，采用“概念→最小例子→命令→手写核心空位→证据→排错→验收”的固定闭环；只新增教材，不修改现有源码。

**Tech Stack:** Markdown、CUDA C++、PTX ISA、SASS、inline PTX、Nsight Compute、A100/sm_80、Hopper/sm_90 架构资料

---

### Task 1: 建立资料与项目证据基线

**Files:**
- Create: `docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md`
- Reference: `week05_gemm_advanced/*.cu`
- Reference: `week06_tensorcore/*.cu`
- Reference: `week05_gemm_advanced/*.md`
- Reference: `week06_tensorcore/*.md`

- [ ] **Step 1: 记录目标文档不存在的 RED 基线**

运行：

```bash
test ! -e 'docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md'
```

预期：退出码 0。

- [ ] **Step 2: 核对现有源码与实验路径**

确认 naive/shared/register/float4/double-buffered GEMM、WMMA GEMM、benchmark 和 ncu 笔记的实际文件路径；只有存在的路径才能写入教材链接。

- [ ] **Step 3: 核对官方资料**

只使用 NVIDIA 官方资料核对易变或精确技术事实：

- PTX ISA：`mma.sync`、`ldmatrix`、`cp.async`、commit/wait group、mbarrier；
- CUDA Programming Guide：编译模型、inline PTX、memory hierarchy；
- Ampere Tuning Guide：A100 异步拷贝与架构限制；
- Nsight Compute Profiling Guide：scheduler、scoreboard、warp stall 指标；
- Hopper Tuning Guide / Hopper Whitepaper：TMA、WGMMA、cluster、DSM。

把每项事实分成 PTX 语义、编译器实际 lowering、A100 实测、Hopper 官方描述四类，避免互相替代。

- [ ] **Step 4: 创建文档骨架**

建立标题、适用读者、两周总览、五个 Part、Day 14 综合实验、附录和参考资料章节。此时不填正文。

### Task 2: 编写 Part I——PTX/SASS 证据链（Day 1–3）

**Files:**
- Modify: `docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md`

- [ ] **Step 1: 写编译链与层级边界**

详细解释 CUDA C++、front end/NVVM、PTX、ptxas、cubin、fatbin、SASS、driver JIT；加入 Mermaid 编译链，并解释为什么 PTX 不是 GPU 最终执行指令。

- [ ] **Step 2: 写工具实验**

提供针对现有项目的可复制命令：

```bash
nvcc -O3 -lineinfo -arch=sm_80 -ptx ...
nvcc -O3 -lineinfo -arch=sm_80 -Xptxas=-v ...
cuobjdump --dump-ptx ...
cuobjdump --dump-sass ...
nvdisasm ...
```

解释各输出回答什么问题，并避免假设所有编译产物已经存在。

- [ ] **Step 3: 写常见 PTX/SASS 阅读词典**

覆盖寄存器、谓词、地址空间、load/store、FMA、分支、barrier、shared/local 访问、矩阵指令；指令拼写以官方资料和实际 sm_80 反汇编为准，不把某个版本的 SASS 助记符当跨代 ABI。

- [ ] **Step 4: 写五版 GEMM 对比实验**

给出 naive→shared→register tiled→float4→WMMA 的观察表、问题模板和预期方向；明确“预期”必须由实际反汇编验证。

- [ ] **Step 5: 写 spill 与向量化实验**

提供由学习者完成的最小源码片段和 TODO，展示寄存器压力、local memory、对齐与宽 load；同时给出 `ptxas`、SASS 和 ncu 三层验证。

- [ ] **Step 6: 写 Day 1–3 自测**

包含术语辨析、命令选择、指令取证、错误判断和闭卷口述。

### Task 3: 编写 Part II——`ldmatrix + mma.sync`（Day 4–6）

**Files:**
- Modify: `docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md`

- [ ] **Step 1: 从 WMMA 下钻到 warp MMA**

用现有 WMMA 代码和 HMMA 记录建立已知基线，区分 WMMA C++ API、PTX MMA 与 SASS HMMA/MMA。

- [ ] **Step 2: 解释 MMA shape 与 operand**

从 `D=A×B+C`、`M×N×K`、输入/累加类型、layout 和寄存器 operand 开始，逐步解释 sm_80 可实践的一个具体 MMA 形状。

- [ ] **Step 3: 解释 fragment 分布**

使用表格和 Mermaid/ASCII 图说明 lane、寄存器、逻辑矩阵元素的关系；明确 fragment 映射依赖具体指令形状和数据类型，不能泛化成一张万能表。

- [ ] **Step 4: 解释 `ldmatrix`**

说明 shared 地址由哪些 lane 提供、一次加载多少矩阵、`.trans` 的作用和布局/对齐要求；加入 shared→register 数据路径图。

- [ ] **Step 5: 添加学习者手写 microkernel**

在文档内嵌可编译外围框架，核心 inline PTX operand、`ldmatrix` 和 `mma.sync` 使用 `TODO 1..N`。提供三级提示、CPU/cuBLAS 对齐方法、编译命令和 SASS 验证方法，不提供完整核心答案。

- [ ] **Step 6: 从 microkernel 回到 GEMM**

解释为什么单条 MMA 正确不代表 GEMM 高性能，以及 warp/block/instruction tile、shared layout、复用、pipeline 和边界处理如何连接。

- [ ] **Step 7: 写 Day 4–6 自测**

要求能画出数据路径、解释 operand 和读懂一段最小 PTX。

### Task 4: 编写 Part III——A100 原生 `cp.async` 多级流水（Day 7–9）

**Files:**
- Modify: `docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md`

- [ ] **Step 1: 建立同步搬运基线**

解释 global→register→shared 的普通路径和 global→shared async copy 的目标；连接项目现有 `cuda::pipeline` 两级实现。

- [ ] **Step 2: 解释原生 PTX 语义**

准确解释 `cp.async.ca/cg.shared.global`、合法 copy size、对齐、zero fill、`commit_group`、`wait_group` 和 visibility；区分 async group completion 与 block-level consumer synchronization。

- [ ] **Step 3: 画 2-stage/3-stage 时间线**

展示 prologue、steady state、epilogue，以及 tile 编号、stage slot、producer/consumer 所有权和 wait 深度。

- [ ] **Step 4: 添加学习者手写 3-stage 骨架**

文档内嵌环形 stage 外围框架，留出原生 `cp.async`、commit、wait 和同步 TODO；提供三级提示和边界 tile 处理策略。

- [ ] **Step 5: 写性能验证方案**

比较同步、2-stage、3-stage，记录正确性、GFLOPS、register、shared、occupancy、eligible warp、stall 与 memory throughput；强调 stage 更多不保证更快。

- [ ] **Step 6: 写 Day 7–9 自测与排错表**

覆盖读到未完成 tile、覆盖仍在消费的 slot、错误 wait depth、缺 block barrier、非对齐和尾部处理。

### Task 5: 编写 Part IV——Scheduler、Scoreboard 与 Stall（Day 10–11）

**Files:**
- Modify: `docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md`

- [ ] **Step 1: 写 warp 调度硬件模型**

解释 active、eligible、selected/issued warp，scheduler/subpartition、latency hiding、ILP/TLP，并注明具体数量依架构而异。

- [ ] **Step 2: 写 scoreboard 数据依赖模型**

用依赖链图解释 load-use、FMA dependency chain、barrier 等等待，不把 scoreboard 描述成缓存或队列。

- [ ] **Step 3: 写 ncu stall 指标阅读方法**

以当前 NVIDIA 文档中的定义为准解释常见类别，并给出“指标只是症状”的诊断流程：先确认吞吐和资源，再关联源码/SASS，形成单一假设，修改后复测。

- [ ] **Step 4: 添加四组 microbenchmark**

内嵌依赖 FMA vs 多 accumulator、load-use vs 穿插计算、occupancy sweep、现有 GEMM 对比框架；核心实验参数由学习者填写。

- [ ] **Step 5: 写 Day 10–11 自测**

给出四组 ncu 症状，要求选择下一步证据，避免“见到 long scoreboard 就直接加 shared”一类机械结论。

### Task 6: 编写 Part V——Hopper 迁移（Day 12–13）

**Files:**
- Modify: `docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md`

- [ ] **Step 1: 写 Ampere→Hopper 总图**

用一张对照图串联 `cp.async→TMA`、warp MMA→WGMMA、CTA→cluster、单 CTA shared→DSM，并说明它们共同改变 mainloop 组织方式。

- [ ] **Step 2: 详细解释 TMA**

覆盖 tensor map/descriptor、bulk multidimensional transfer、坐标与边界、异步完成和 barrier 协作；不把 TMA 说成“更宽的 cp.async”。

- [ ] **Step 3: 详细解释 WGMMA**

覆盖 warp-group、异步 MMA、operand 来源、fence/commit/wait 思想和与 warp-level MMA 的差异；指令细节只采用官方资料支持的表述。

- [ ] **Step 4: 解释 Cluster 与 DSM**

说明 cluster 调度保证、rank、map shared address、远程 shared 访问、同步和适用场景；加入空间关系图。

- [ ] **Step 5: 解释 mbarrier 与 warp specialization**

把 producer/consumer 角色、TMA arrival、WGMMA consumer 和多 stage buffer 连成概念性主循环；明确这是 Hopper 阅读模型，不是 A100 可运行实验。

- [ ] **Step 6: 写源码阅读路线和 Day 12–13 自测**

提供阅读 CUTLASS/CuTe Hopper mainloop 时的术语映射、观察顺序和边界，不要求 clone、编译或运行外部项目。

### Task 7: 编写 Day 14、附录与两周执行表

**Files:**
- Modify: `docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md`

- [ ] **Step 1: 写 Day 14 综合分析模板**

要求选择一个现有 GEMM，交付源码数据流、PTX、SASS、资源、ncu、假设、修改、复测和 Hopper 迁移判断。

- [ ] **Step 2: 写每日清单**

每一天列出阅读、动手、预期产物、验收标准和时间建议；总量按两周高强度而非入门慢速课程设计。

- [ ] **Step 3: 写命令速查与术语表**

汇总 nvcc、ptxas、cuobjdump、nvdisasm、ncu、compute-sanitizer 命令，以及 PTX/SASS、lane/fragment、stage/group、scheduler/scoreboard、TMA/WGMMA/DSM 等术语。

- [ ] **Step 4: 写官方参考资料**

只列实际使用过的官方页面，并在正文关键结论附近放链接。

### Task 8: 全文验证与提交

**Files:**
- Test: `docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md`

- [ ] **Step 1: 验证结构和范围**

运行脚本确认五个 Part、Day 1–14、`TODO 1`、三级提示、A100、Hopper、PTX、SASS、`ldmatrix`、`mma.sync`、`cp.async`、scoreboard、TMA、WGMMA、cluster、DSM 全部存在。

- [ ] **Step 2: 验证 Markdown**

检查代码围栏成对、标题层级合理、Mermaid 围栏成对、本地链接指向存在文件、没有 `TBD` 或无意的占位文本。

- [ ] **Step 3: 验证技术片段**

逐个核对 shell 命令、PTX 指令拼写、architecture guard、inline PTX operand 和 ncu metric/section；无法在当前环境执行的 sm_90 内容必须标为架构阅读而非实测。

- [ ] **Step 4: 验证修改范围**

```bash
git diff --check
git diff --name-only -- 'docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md'
```

确认只新增目标教材，不纳入用户已有的暂存 rename 和其他未提交文件。

- [ ] **Step 5: 提交**

```bash
git add -- 'docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md'
git commit -m 'docs: add two-week CUDA deep-dive guide'
```
