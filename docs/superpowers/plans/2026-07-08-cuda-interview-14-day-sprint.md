# CUDA 工程师面试 14 天突击计划 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 编写一份基于本仓库真实学习记录、每天约 8 小时、可直接执行并可逐日验收的 CUDA 工程师面试 14 天突击文档。

**Architecture:** 最终只新增 `docs/CUDA工程师面试_14天突击计划.md`。文档先建立能力基线和使用规则，再按“7 天深水区、4 天独立实现与项目收口、3 天面试转换”展开，最后提供岗位匹配、项目讲解、追问和补救附录；现有教材、源码与笔记仅通过相对链接复用，不做修改。

**Tech Stack:** Markdown、CUDA C++ 仓库资料、A100/sm_80、PTX/SASS 工具链、Nsight Compute、Compute Sanitizer、Git。

---

## File Structure

- Create: `docs/CUDA工程师面试_14天突击计划.md`
  - 单一学习入口，包含能力画像、14 天日程、每日验收、旗舰项目、模拟面试与投递检查表。
- Reference only: `docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md`
  - Day 1～7 主教材。
- Reference only: `docs/CUDA复习资料_知识体系.md`、`docs/CUDA面试八股全集.md`、追问答案和卷十面试资料
  - Day 12～14 概念、追问、性能诊断和系统题来源。
- Reference only: `week03_parallel/`、`week04_attention/`、`week04_gemm/`、`week05_gemm_advanced/`、`week05_inference/`、`week06_tensorcore/`
  - 手写训练、性能证据和旗舰项目来源。

### Task 1: 建立文档骨架、能力基线与执行规则

**Files:**
- Create: `docs/CUDA工程师面试_14天突击计划.md`
- Reference: `docs/superpowers/specs/2026-07-08-cuda-interview-14-day-sprint-design.md`
- Reference: `notes/week03.md`
- Reference: `week05_gemm_advanced/benchmark.md`
- Reference: `week05_gemm_advanced/ncu_notes.md`
- Reference: `week06_tensorcore/tensor_core_profile.md`
- Reference: `week04_attention/ncu_pipeline_notes.md`

- [ ] **Step 1: 写标题、定位和使用说明**

  明确这是每天约 8 小时、最多 14 天的面试闭环计划；说明“阅读完成不等于验收完成”，每天必须留下概念、代码、证据和面试输出。

- [ ] **Step 2: 写基于仓库证据的能力画像**

  用表格列出执行模型、内存、并行算法、GEMM、Tensor Core、Attention、profiling、系统工程、底层指令和面试表达的当前水平。引用仓库中 reduction 91% 带宽、GEMM 12681 GFLOPS、WMMA 16484 GFLOPS、`cp.async` stall 下降和 GEMV 19 倍提升等证据，不把文档存在等同于个人掌握。

- [ ] **Step 3: 写优先级、每日时间盒和补考规则**

  定义“必须完成、尽量完成、时间不足可跳过”三个等级；每天按 2 小时概念、3 小时代码、1.5 小时证据、1.5 小时闭卷训练组织；未通过时次日最多补考 90 分钟，再失败则记入 Day 12 薄弱项清单。

- [ ] **Step 4: 写突击前准备清单**

  列出 A100 环境确认、`nvcc`、`ncu`、`cuobjdump`、`compute-sanitizer`、计时与输出目录约定。命令只写读环境或生成 `/tmp/cuda_interview_sprint/` 产物的安全操作，不要求整理当前工作区。

- [ ] **Step 5: 检查骨架结构**

  Run: `rg -n '^#|能力画像|必须完成|补考|突击前准备' docs/CUDA工程师面试_14天突击计划.md`

  Expected: 标题、能力画像、执行规则和准备清单均能被检索到。

- [ ] **Step 6: 提交第一部分**

  ```bash
  git add -- docs/CUDA工程师面试_14天突击计划.md
  git commit -m "docs: start CUDA interview sprint guide"
  ```

### Task 2: 编写 Day 1～7 深水区路线

**Files:**
- Modify: `docs/CUDA工程师面试_14天突击计划.md`
- Reference: `docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md`
- Reference: `week04_gemm/gemm_naive/gemm_naive.cu`
- Reference: `week04_gemm/gemm_tiled/gemm_tiled.cu`
- Reference: `week05_gemm_advanced/gemm_2d_thread_tiling.cu`
- Reference: `week05_gemm_advanced/gemm_vectorized_load.cu`
- Reference: `week05_gemm_advanced/gem_double_buffering.cu`
- Reference: `week06_tensorcore/wmma_fp16_gemm.cu`

- [ ] **Step 1: 写 Day 1 编译链与第一份反汇编**

  覆盖 CUDA C++→PTX→ptxas→cubin/SASS、`compute_80` 与 `sm_80`，要求为 naive GEMM 生成 PTX、ptxas resource log 和 SASS，并完成闭卷口述。

- [ ] **Step 2: 写 Day 2 PTX/SASS 数据流阅读**

  安排 naive/shared/register/float4/WMMA 五版对照；要求只摘录关键 load、shared、FFMA/HMMA、控制流和 store，不逐行翻译汇编。

- [ ] **Step 3: 写 Day 3 寄存器、spill、向量化取证**

  要求使用 `-Xptxas=-v`、SASS 和 ncu local traffic 互证 spill；验证 `float4` 是否落成宽 load，并区分“源码类型”和“最终指令”。

- [ ] **Step 4: 写 Day 4 scheduler、scoreboard 与 stall**

  从 active/eligible/issued warp、long/short scoreboard、ILP/TLP 和 occupancy 建立诊断顺序；要求完成一份“指标→假设→单一修改→复测”短报告。

- [ ] **Step 5: 写 Day 5 fragment、ldmatrix 与 mma.sync**

  先画逻辑矩阵到 lane/register 的映射，再完成最小 `ldmatrix`/`mma.sync` 观察实验；明确只要求一个确定 shape/type，不将映射错误推广到所有 MMA 变体。

- [ ] **Step 6: 写 Day 6 cp.async 与多 stage 流水**

  覆盖 copy、commit、wait、consume 状态机以及 wait 与 CTA barrier 的区别；要求比较同步、2-stage、3-stage 的正确性、资源、stall 和墙钟时间，允许 3-stage 更慢但必须解释。

- [ ] **Step 7: 写 Day 7 Hopper 对照与综合报告**

  对照 Ampere `cp.async + mma.sync` 与 Hopper `TMA + WGMMA`，补充 Cluster/DSM 和 warp specialization；当天下午以现有 GEMM 或 Attention 完成一份源码、资源、SASS、ncu、修改与复测的综合证据链。

- [ ] **Step 8: 核对七天的固定字段**

  Run: `for d in {1..7}; do rg -q "Day $d" docs/CUDA工程师面试_14天突击计划.md || exit 1; done`

  Expected: exit code 0。每一天人工确认包含时间表、必读、必做、输出物、验收题和降级策略。

- [ ] **Step 9: 提交深水区部分**

  ```bash
  git add -- docs/CUDA工程师面试_14天突击计划.md
  git commit -m "docs: add seven-day CUDA deep-dive sprint"
  ```

### Task 3: 编写 Day 8～11 独立手写与项目收口

**Files:**
- Modify: `docs/CUDA工程师面试_14天突击计划.md`
- Reference: `week03_parallel/reduction_sum_full/reduction_sum_full.cu`
- Reference: `operator_practice/transpose/transpose.cu`
- Reference: `week04_gemm/softmax/softmax.cu`
- Reference: `week04_gemm/gemm_tiled/gemm_tiled.cu`
- Reference: `week05_inference/gemv.cu`
- Reference: `week05_inference/fused_rmsnorm.cu`
- Reference: `week05_inference/dequant_gemv.cu`
- Reference: `week05_inference/decode_graph.cu`

- [ ] **Step 1: 写 Day 8 reduction、transpose 限时手写**

  要求先写正确版本，再优化；规定时间、边界输入、CPU reference、重复运行和 sanitizer 验收；加入复盘问题：线程映射、同步位置、bank conflict 和吞吐上限。

- [ ] **Step 2: 写 Day 9 softmax、GEMV、RMSNorm 限时手写**

  覆盖数值稳定、warp reduction、非规则维度和融合读写；明确 GEMV 的 thread-per-row 与 warp-per-row 对照，以及 RMSNorm 的平方和归约和 epsilon。

- [ ] **Step 3: 写 Day 10 tiled GEMM 与推理专项补齐**

  tiled GEMM 要求闭卷完成核心索引、协作加载、同步、边界补零和写回；剩余时间按 CUDA Graph→Fused RMSNorm→Dequant GEMV 顺序完成仓库练习。

- [ ] **Step 4: 写 Day 11 A100 旗舰项目收口**

  默认推荐 GEMM，允许用户改选 Attention。要求形成 baseline、版本阶梯、正确性矩阵、可靠计时、Roofline、ncu、PTX/SASS、失败优化、库/理论上限对比和下一步方案。

- [ ] **Step 5: 写旗舰项目 README/报告模板**

  在计划文档中提供可复制的 Markdown 模板，字段包括硬件软件版本、shape/dtype、命令、结果表、瓶颈证据、失败实验和 5/10 分钟口述稿；不新建或覆盖实际项目 README。

- [ ] **Step 6: 核对 Day 8～11 和核心 kernel 覆盖**

  Run: `for d in {8..11}; do rg -q "Day $d" docs/CUDA工程师面试_14天突击计划.md || exit 1; done && rg -n 'reduction|transpose|softmax|GEMV|RMSNorm|tiled GEMM|CUDA Graph|Dequant' docs/CUDA工程师面试_14天突击计划.md`

  Expected: exit code 0，并列出全部核心练习。

- [ ] **Step 7: 提交独立实现部分**

  ```bash
  git add -- docs/CUDA工程师面试_14天突击计划.md
  git commit -m "docs: add kernel and portfolio interview sprint"
  ```

### Task 4: 编写 Day 12～14 面试转换

**Files:**
- Modify: `docs/CUDA工程师面试_14天突击计划.md`
- Reference: `docs/CUDA复习资料_知识体系.md`
- Reference: `docs/CUDA面试八股全集.md`
- Reference: `docs/CUDA面试八股_追问答案.md`
- Reference: `docs/CUDA面试八股_追问答案_续.md`
- Reference: `cuda_deep_course/course/volume10_engineering_interview/08_面试题_概念_性能分析_手写kernel.md`
- Reference: `cuda_deep_course/course/volume10_engineering_interview/09_系统设计题.md`

- [ ] **Step 1: 写 Day 12 知识体系闭卷压缩**

  按执行模型、内存、同步、性能、异步、Tensor Core、编译部署七组组织题目；先闭卷作答，再对照资料修正；保留 90 分钟回补前 11 天薄弱项。

- [ ] **Step 2: 写 Day 13 限时编码与性能诊断模拟**

  安排两道 kernel 手写题和三道 ncu 症状诊断题；要求回答采用“现象→候选原因→补充证据→单一实验”，禁止看到一个 stall 名就直接下结论。

- [ ] **Step 3: 写 Day 14 两轮模拟面试与投递判断**

  第一轮覆盖概念、手写和项目，第二轮覆盖性能追问、架构取舍和系统题；给出 45 分钟时间轴、评分表、硬失败项和“通过后当天开始投递”的判断规则。

- [ ] **Step 4: 写项目口述与追问模板**

  提供 30 秒自我定位、2 分钟项目摘要、5 分钟优化链和 10 分钟深挖结构，并列出关于正确性、计时、瓶颈、失败优化、可移植性和库对比的追问。

- [ ] **Step 5: 核对 Day 12～14**

  Run: `for d in {12..14}; do rg -q "Day $d" docs/CUDA工程师面试_14天突击计划.md || exit 1; done && rg -n '45 分钟|模拟面试|限时|性能诊断|项目口述' docs/CUDA工程师面试_14天突击计划.md`

  Expected: exit code 0，并显示模拟面试和输出模板位置。

- [ ] **Step 6: 提交面试转换部分**

  ```bash
  git add -- docs/CUDA工程师面试_14天突击计划.md
  git commit -m "docs: complete CUDA interview conversion sprint"
  ```

### Task 5: 完成附录、链接验证和最终自检

**Files:**
- Modify: `docs/CUDA工程师面试_14天突击计划.md`

- [ ] **Step 1: 写岗位匹配和内容分级附录**

  区分初级 CUDA、性能优化、LLM 推理算子、CUTLASS/高性能算子和多 GPU/HPC 岗位；标明当前 14 天计划能覆盖到的范围以及仍需生产经验的部分。

- [ ] **Step 2: 写投递前检查表和压缩方案**

  检查表覆盖独立编码、sanitizer、可靠计时、ncu、SASS、项目口述和模拟面试。补充落后 1～2 天、3～4 天时的压缩顺序：先削减 Hopper 细节和扩展 kernel，不删除核心手写、旗舰案例与模拟面试。

- [ ] **Step 3: 写第 14 天后的补救路径**

  分别为 PTX/SASS、独立手写、性能诊断、C++ 工程化、PyTorch/LLM 和多 GPU 薄弱项给出下一步，不把这些扩展塞回当前 14 天必做范围。

- [ ] **Step 4: 验证相对 Markdown 链接**

  提取文档中的本地 `.md`/`.cu` 链接，以 `docs/` 为基准解析 `../` 路径，并逐一执行 `test -e`。如有不存在的路径，修正链接，不创建假文件。

- [ ] **Step 5: 检查结构、占位符和 Markdown 空白**

  Run:

  ```bash
  rg -n '^## Day ([1-9]|1[0-4])' docs/CUDA工程师面试_14天突击计划.md
  rg -n 'TBD|以后再写|待补充|PLACEHOLDER' docs/CUDA工程师面试_14天突击计划.md
  git diff --check -- docs/CUDA工程师面试_14天突击计划.md
  ```

  Expected: Day 1～14 各出现一次；占位符搜索无输出；`git diff --check` 无输出。

- [ ] **Step 6: 对照设计说明做最终覆盖检查**

  人工逐项确认：14 天均有时间表/任务/输出/验收；A100 实测与 Hopper 概念边界明确；四类每日输出齐全；包含旗舰案例、岗位匹配、投递检查、追问和补救路径；未修改任何现有源码。

- [ ] **Step 7: 查看最终变更范围**

  Run: `git status --short && git diff --stat HEAD -- docs/CUDA工程师面试_14天突击计划.md`

  Expected: 本任务只涉及最终学习文档；用户原有未提交删除和新文件保持原样。

- [ ] **Step 8: 提交最终校订**

  ```bash
  git add -- docs/CUDA工程师面试_14天突击计划.md
  git commit -m "docs: finalize 14-day CUDA interview sprint"
  ```
