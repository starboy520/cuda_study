# Week5 LLM 推理优化自包含教材 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Week5 任务清单完整重写为适合 LLM 推理小白、但可在 7 天高强度完成的自包含 CUDA 教材。

**Architecture:** 单一 Markdown 文档围绕一次 token 的生成流程组织，每天执行“概念→shape/数字→CPU/reference→CUDA 映射→TODO 骨架→提示→验证→profiling→自测”闭环。保留原文所有实践主题，同时修正过度绝对化的性能结论。

**Tech Stack:** Markdown、CUDA C++、A100/sm_80、Nsight Compute、Nsight Systems、Compute Sanitizer、NVIDIA 官方 CUDA/Hopper 文档、推理框架官方文档

---

### Task 1: 建立原文迁移与技术事实基线

**Files:**
- Modify: `docs/Week5增强版_LLM推理优化与decode.md`

- [ ] **Step 1: 保存原文任务清单**

记录必须迁移的主题和产出：KV accounting、GEMV、occupancy、ncu/nsys、融合、sanitizer、Graph、grid sync、量化 GEMV、Paged Attention、框架、Hopper、最终 decode 图。

- [ ] **Step 2: 运行 RED 教学结构检查**

断言原文缺少：Day 0/推理基础、完整 CPU GEMV、可编译外围框架、三级提示、RMSNorm 数字例子、Graph 生命周期代码、量化手算、block-table 手推和条件性性能判断。预期失败。

- [ ] **Step 3: 核对本地依赖路径**

确认 Week4 Attention、AI Infra 八股、KV Cache 指南、Occupancy/ncu、async/Graph、CUDA 深水区、Compute Sanitizer 资料均存在；正文只链接实际路径。

- [ ] **Step 4: 核对官方资料**

以官方一手来源核对 CUDA Graph、cooperative launch、occupancy API、Compute Sanitizer、vLLM/Paged Attention、TensorRT-LLM 调度与 KV、SGLang、Hopper TMA/WGMMA/cluster。

- [ ] **Step 5: 重建文档骨架**

写入适用读者、学习方法、7 天总览、统一符号、实验目录建议、Day 1–7、最终验收、命令速查和官方资料章节。

### Task 2: 编写 Day 1——推理全景与 KV Cache

**Files:**
- Modify: `docs/Week5增强版_LLM推理优化与decode.md`

- [ ] **Step 1: 解释 token 到 logits 的完整最小流程**

定义 token/id、embedding、hidden state、Transformer block、logits、sampling；用 3-token、4-hidden 的微型 shape 贯穿。

- [ ] **Step 2: 解释 Transformer 层中当天相关组件**

只讲到推理所需：RMSNorm、Attention、线性层、残差；明确后续每天对应位置。

- [ ] **Step 3: 推导 prefill/decode 时间线**

写出首次 prompt 和后续三个 decode step 的输入 shape、输出和缓存变化；解释 effective batch，不把 decode 永远写成 M=1。

- [ ] **Step 4: 从重复计算推导 KV Cache**

先展示没有 cache 会重复做什么，再推导 MHA/GQA 的字节公式，完整计算至少两组配置。

- [ ] **Step 5: 写性能指标和条件判断**

解释 TTFT、TPOT、throughput、latency；用 shape/AI 而不是阶段名称判断 bound。

- [ ] **Step 6: 添加练习、自测和口述**

提供 KV accounting 表格、三组配置练习、答案检查公式和闭卷问题。

### Task 3: 编写 Day 2–3——GEMV 从零到性能分析

**Files:**
- Modify: `docs/Week5增强版_LLM推理优化与decode.md`

- [ ] **Step 1: 写 GEMV 数学和 3×4 手算**

从线性层 `y=Wx+b` 到 `W[N,K]、x[K]、y[N]`，逐元素算完，并与 GEMM 的 M 维建立联系。

- [ ] **Step 2: 写完整 CPU reference 和测试器**

内嵌可编译 host 代码：随机初始化、CPU GEMV、误差检查、CUDA Event、有效 bytes/GB/s/GFLOPS。

- [ ] **Step 3: 写 CUDA v0/v1 外围框架**

v0 一线程一行完整演示；v1 一 warp 一行保留点积、mask、shuffle 核心 TODO，并给三级提示。

- [ ] **Step 4: 解释 coalescing 与映射权衡**

明确同一 warp 沿一行读取的合并访问、跨行映射、x 复用和小/大 K 方案。

- [ ] **Step 5: 写 Day 3 向量化与缓存实验**

加入 float4/half2 前提、shared/constant/cache 对 x 的适用条件、多行/block 方案；不承诺源码向量类型必然生成宽 SASS。

- [ ] **Step 6: 写 occupancy/ILP/TLP 与工具实验**

提供 A100 资源手算例子、ptxas 命令、ncu sections、nsys 时间线命令、Roofline/GEMV bytes 表；删除固定 70% 带宽标准。

- [ ] **Step 7: 写多 shape 验收**

至少 `N/K=3/4`、非整除、典型 hidden、大 batch/多 vectors；完成 baseline→假设→修改→复测模板。

### Task 4: 编写 Day 4——RMSNorm、Residual 与融合

**Files:**
- Modify: `docs/Week5增强版_LLM推理优化与decode.md`

- [ ] **Step 1: 写数学和最小数字例子**

分别解释 residual、RMSNorm、SiLU；手算一条 4 元素向量，解释 epsilon 和 gamma。

- [ ] **Step 2: 写 CPU reference**

内嵌 residual→RMSNorm→SiLU reference，明确 FP32 累计和输出 dtype。

- [ ] **Step 3: 写未融合 HBM 账本**

列每个 kernel 的 read/write bytes，再算 fused 理想 logical bytes；区分 logical 与实际 transaction。

- [ ] **Step 4: 写 CUDA 外围框架和 TODO**

提供未融合版本接口与 fused kernel 框架，留 block/warp reduction、inverse RMS 和写回核心 TODO，提供三级提示。

- [ ] **Step 5: 写融合代价与 profiler**

解释 register、occupancy、并行度、数值和维护成本；给 ncu 对比表。

- [ ] **Step 6: 写 sanitizer 验证**

加入四工具用途、运行顺序和边界测试，不把 memcheck 无错误等同算法正确。

### Task 5: 编写 Day 5——CUDA Graph 与 Cooperative Grid Sync

**Files:**
- Modify: `docs/Week5增强版_LLM推理优化与decode.md`

- [ ] **Step 1: 写 launch 时间线和 Graph 生命周期**

用 Mermaid/ASCII 对比逐 launch 与 replay；解释 capture、end、instantiate、launch、update、destroy 和错误生命周期。

- [ ] **Step 2: 写伪 decode Graph 外围程序**

提供若干短 kernel、普通循环和 Graph 循环框架；capture/instantiate/replay 为学习者 TODO，给三级提示。

- [ ] **Step 3: 写正确 benchmark 和 nsys 实验**

分开首轮 instantiate 成本、稳态 replay、kernel 时间和 CPU wall time；说明动态 shape/address/batch 限制。

- [ ] **Step 4: 从多 kernel reduction 引出 grid sync**

解释普通 kernel 无 grid barrier、global counter 自旋死锁风险、cooperative launch 能力与同时驻留约束。

- [ ] **Step 5: 写 grid reduction 框架**

提供 CPU reference、两阶段版本和 cooperative kernel 外围；学习者完成 `grid.sync()` 前后核心状态。

- [ ] **Step 6: 写 API/occupancy 检查和验收**

检查 cooperative launch 支持、最大可驻留 blocks、launch 失败与正确性；不宣称单 kernel 一定更快。

### Task 6: 编写 Day 6——INT8 Weight-only GEMV

**Files:**
- Modify: `docs/Week5增强版_LLM推理优化与decode.md`

- [ ] **Step 1: 写 INT8/scale 的数字手算**

从浮点权重计算 scale、round/clamp、int8、dequant 和 dot product；展示量化误差。

- [ ] **Step 2: 区分量化方案**

解释 symmetric/asymmetric、per-tensor/channel/group、weight-only/W8A8/KV quant 的状态和 kernel 差异。

- [ ] **Step 3: 写 CPU quant/dequant reference**

提供 per-channel quantizer、dequant GEMV、误差统计和 baseline bytes。

- [ ] **Step 4: 写 CUDA TODO 框架**

提供 INT8 storage、scale array 和输出；学习者完成向量加载、scale、累加与 warp reduction。

- [ ] **Step 5: 写性能/正确性判定**

比较 FP baseline 与 quantized storage 的时间、DRAM bytes、计算/转换指令和误差；明确 bytes 减半不保证等比加速。

### Task 7: 编写 Day 7——Paged Attention、框架与 Hopper

**Files:**
- Modify: `docs/Week5增强版_LLM推理优化与decode.md`

- [ ] **Step 1: 从连续 KV 分配问题引出分页**

用请求到达/增长/结束时间线说明预留、内部/外部浪费和动态容量问题。

- [ ] **Step 2: 手推 block table**

使用两个请求、固定 block size、四个物理块，逐 token 计算逻辑块、offset、物理块和地址；解释 metadata/间接寻址成本。

- [ ] **Step 3: 写 continuous batching**

说明每步 active requests 变化、KV capacity、prefill/decode 混合、吞吐、TTFT/TPOT/P99 权衡。

- [ ] **Step 4: 写框架地图**

基于官方当前文档描述 vLLM、TensorRT-LLM、SGLang 的职责与选择维度，不做永久功能排行榜。

- [ ] **Step 5: 写 Hopper 迁移课**

自包含解释 A100 `cp.async/mma` 到 TMA/WGMMA/cluster/DSM/mbarrier/warp specialization；链接深度教材，不要求 H100 实测。

- [ ] **Step 6: 写最终 decode 数据流和口述**

把七天主题放回一次请求，标注每步 state、FLOP、bytes、launch/communication 和可能瓶颈。

### Task 8: 全文迁移、技术和 Markdown 验证

**Files:**
- Test: `docs/Week5增强版_LLM推理优化与decode.md`

- [ ] **Step 1: 原任务迁移检查**

确认原文八个补充点和全部产出有对应章节，没有在重写中丢失。

- [ ] **Step 2: 固定结构检查**

每个 Day 均有位置、前置、数字/shape、reference/练习、CUDA/TODO、三级提示、命令、测试、profiling、错误、自测/口述；系统概念日允许用算账/映射替代 kernel。

- [ ] **Step 3: 风险句检查**

确认不存在无条件的 prefill/decode bound、decode=M1、GEMV 无复用、带宽>70%、Graph 完美适配、INT8 必然加速等旧表述。

- [ ] **Step 4: 代码与命令检查**

提取完整命名代码块做语法编译；TODO 骨架检查接口、类型和依赖一致；验证 nvcc/ncu/nsys/sanitizer 命令在当前环境存在或明确标注需目标工具/硬件。

- [ ] **Step 5: 链接与结构检查**

检查代码围栏、Mermaid、公式定界符、本地链接、官方链接、标题和 `git diff --check`。

- [ ] **Step 6: 修改范围检查**

只修改目标文档，不纳入现有源码或其他未提交文件。

- [ ] **Step 7: 提交**

```bash
git add -- 'docs/Week5增强版_LLM推理优化与decode.md'
git commit -m 'docs: turn Week5 inference plan into self-contained course'
```
