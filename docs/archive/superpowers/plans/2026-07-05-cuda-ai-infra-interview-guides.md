# CUDA 与 AI Infra 面试题库 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 CUDA 八股重写为纯 CUDA/GPU 性能分层题库，并新增独立的 AI Infra 面试题库。

**Architecture:** 两份 Markdown 文档共用统一题目模板，通过本地链接交叉引用。CUDA 文档覆盖 B 基础、A 性能、C 专家三层；AI Infra 文档接收原有 ML/LLM 内容并扩展到推理、通信、并行与系统设计。先做迁移和事实基线，再分文档编写，最后统一术语、去重和验证。

**Tech Stack:** Markdown、CUDA C++、NVIDIA 官方文档、PyTorch/推理框架官方资料、Git、文档结构检查脚本

---

### Task 1: 建立迁移与准确性基线

**Files:**
- Modify: `docs/CUDA面试八股全集.md`
- Create: `docs/AI_Infra面试八股全集.md`

- [ ] **Step 1: 保存原文主题清单**

提取原文 15 章标题和关键主题，建立以下迁移表：

```text
CUDA 保留：GPU 架构、执行、内存、访问、同步、occupancy、profiling、并行算法、GEMM、Tensor Core、异步
CUDA 扩展：PTX/SASS、MMA、cp.async、scheduler/stall、调试、Graph、Hopper、工程化
AI Infra 迁移：Attention、Softmax、KV cache、推理、量化、框架、并行与 MoE 场景
共享但分层：NCCL、Tensor Core、CUDA Graph、GEMM/GEMV、精度格式
```

- [ ] **Step 2: 运行修改前 RED 检查**

断言应失败：

- CUDA 文档仍包含 `Attention / Softmax / LLM` 与 `LLM 推理系统`；
- CUDA 文档缺少统一的“15 秒回答”“面试官继续追问”“项目证据”和 B/A/C 标签；
- AI Infra 文档尚不存在；
- CUDA 文档仍含高风险表述，如固定 scheduler 数、`__threadfence()` 自动可见、经验 occupancy 阈值。

- [ ] **Step 3: 核对项目证据路径**

确认以下类别的真实源码/记录路径：

- Week 1 计时/device query/GEMM；
- Week 2 memory/transpose/stream；
- Week 3 reduction/scan/histogram；
- Week 5 GEMM、benchmark、ncu；
- Week 6 WMMA/Tensor Core；
- Week 4 Attention；
- CUDA 深水区两周教材；
- cuda_deep_course profiling/async/engineering/hardware 章节。

- [ ] **Step 4: 核对官方资料**

只使用官方一手来源核对：

- CUDA/PTX/Hopper/Nsight/NCCL：NVIDIA 官方文档；
- PyTorch 语义：PyTorch 官方文档；
- vLLM、TensorRT-LLM、SGLang、DeepGEMM、FlashMLA、DeepEP：官方文档或官方仓库；
- 不稳定的规格、框架特性和默认行为必须带条件或版本边界。

### Task 2: 重写 CUDA 文档基础层（B 必会）

**Files:**
- Modify: `docs/CUDA面试八股全集.md`

- [ ] **Step 1: 重建文档骨架**

写入定位、使用方法、B/A/C 标签、统一答题模板、目录和复习路径；删除 ML/LLM 专属章节正文，但保留迁移内容直到 AI 文档确认接收。

- [ ] **Step 2: 编写硬件与执行模型题组**

覆盖 CPU/GPU、GPC/TPC/SM、warp scheduler、SIMT/SIMD、warp divergence、block/grid、grid-stride、independent thread scheduling。修正架构数量不能泛化的问题。

- [ ] **Step 3: 编写内存与访问题组**

覆盖 register/local/shared/L1/L2/global/constant/texture、coalescing、transaction/sector、对齐、AoS/SoA、vector load、bank conflict、padding、cache。明确 `float4` 需要 SASS 验证。

- [ ] **Step 4: 编写同步与一致性题组**

覆盖 barrier/fence、`__syncthreads`、`__syncwarp`、atomics、volatile、memory ordering/visibility、cooperative launch。修正 `__threadfence()` 的绝对化描述。

- [ ] **Step 5: 编写 occupancy 与异步基础题组**

覆盖 register/shared/block 限制、ILP/TLP、spill、stream/event/pinned memory、overlap、CUDA Graph 基础。删除固定 occupancy 经验阈值。

### Task 3: 编写 CUDA 性能主线（A）与专家加分（C）

**Files:**
- Modify: `docs/CUDA面试八股全集.md`

- [ ] **Step 1: 编写并行算法题组**

覆盖 reduction、scan、histogram、transpose、atomic aggregation；连接项目源码和实测。

- [ ] **Step 2: 编写 GEMM 题组**

覆盖 naive、shared、register tiling、vector load、double buffering、warp/instruction tile、cuBLAS/CUTLASS 边界；加入访存账、AI、资源权衡和项目深挖。

- [ ] **Step 3: 编写 Tensor Core 与 MMA 题组**

覆盖 WMMA、fragment、HMMA、`mma.sync`、`ldmatrix`、mixed precision 的硬件/指令边界；把训练/推理策略留给 AI 文档。

- [ ] **Step 4: 编写 cp.async 与流水题组**

覆盖 global→shared、cache hint、commit/wait group、barrier、2/3-stage、prologue/steady/epilogue 和 stage/occupancy 权衡。

- [ ] **Step 5: 编写 PTX/SASS 与编译链题组**

覆盖 CUDA C++→PTX→ptxas→cubin/SASS、JIT、fatbin、register/spill、反汇编取证。

- [ ] **Step 6: 编写 profiling、调试与工程题组**

覆盖 nsys/ncu、Roofline、scheduler/scoreboard/stall、compute-sanitizer 四工具、cuda-gdb、错误检查、RAII、benchmark 和性能回归。

- [ ] **Step 7: 编写多 GPU 与 Hopper 题组**

覆盖 PCIe/P2P/NVLink、拓扑、NCCL collective 机制；Hopper 覆盖 TMA、WGMMA、cluster、DSM、mbarrier 和 warp specialization。并行策略场景放 AI 文档。

### Task 4: 完成 CUDA 项目深挖、手写题与模拟面试

**Files:**
- Modify: `docs/CUDA面试八股全集.md`

- [ ] **Step 1: 添加项目深挖题**

围绕已有 A100 GEMM、bank conflict、occupancy 扫描、reduction/CUB、stream overlap、WMMA/HMMA 构造问题；答案必须与真实数据相符。

- [ ] **Step 2: 添加高频手写题**

列出 vector add、reduction、scan、transpose、histogram、tiled GEMM、WMMA 骨架、错误检查框架；给评分点和易错点，不展开成完整教程。

- [ ] **Step 3: 添加三档模拟题**

分别提供 B 常规 CUDA、A Kernel 性能、C 专家加分面试题组，并给面试官追问路径。

- [ ] **Step 4: 审计难度比例**

统计题目标签，调整到大致 B 45%、A 45%、C 10%；允许 ±5%，不能通过给同一道题重复标签凑数。

### Task 5: 创建 AI Infra 文档基础与算子题组

**Files:**
- Create: `docs/AI_Infra面试八股全集.md`

- [ ] **Step 1: 建立定位、目录和模板**

说明前置 CUDA 能力、交叉边界和 B/A/C 含义；链接 CUDA 面试文档，不复制其硬件基础。

- [ ] **Step 2: 编写 Tensor/shape/GEMM/GEMV 题组**

覆盖 batch、sequence、hidden/head dimension、layout、算术强度、小 batch GEMV/GEMM；连接 CUDA GEMM 机制但关注模型阶段。

- [ ] **Step 3: 编写 Softmax/Online Softmax 题组**

覆盖 stable softmax、online `(m,l)`、tile merge、数值稳定和并行归约；引用 Week4 教材。

- [ ] **Step 4: 编写 Attention/FlashAttention 题组**

覆盖 Q/K/V、shape、causal、FLOP/存储/IO、`m/l/O_acc`、exact 与浮点差异；避免把 FLOP 从平方降为线性的错误。

- [ ] **Step 5: 编写 Transformer、Prefill/Decode 与 KV 题组**

覆盖数据流、性能条件、KV cache 公式、MHA/MQA/GQA/MLA。所有公式明确变量和 dtype。

- [ ] **Step 6: 编写 Paged Attention 与显存管理题组**

覆盖逻辑块/物理块、block table、碎片、分页代价和调度关系，以官方 vLLM 文档为准。

### Task 6: 编写 AI Infra 系统、量化、并行与框架题组

**Files:**
- Modify: `docs/AI_Infra面试八股全集.md`

- [ ] **Step 1: 编写推理优化题组**

覆盖小 GEMM/GEMV、算子融合、kernel launch、CUDA Graph 场景、continuous batching、吞吐/延迟/公平性和调度。

- [ ] **Step 2: 编写精度与量化题组**

区分 FP16/BF16/TF32/FP8 与 INT8/INT4，覆盖 weight/activation/KV quant、scale 粒度、校准、误差和硬件落地。

- [ ] **Step 3: 编写并行策略与通信量题组**

覆盖 DP/TP/PP/CP/EP、all-reduce/all-gather/reduce-scatter/all-to-all 的场景、字节量、拓扑和重叠。

- [ ] **Step 4: 编写 MoE 题组**

覆盖 routing、top-k、dispatch/combine、容量、负载均衡、expert parallel 与 all-to-all。

- [ ] **Step 5: 编写框架边界题组**

基于官方资料解释 vLLM、TensorRT-LLM、SGLang 的职责；不写容易过时的功能对比表作为永久结论。

- [ ] **Step 6: 编写 DeepSeek 技术组件题组**

基于官方仓库解释 DeepGEMM、FlashMLA、DeepEP、DualPipe 分别解决的问题，明确 README/架构理解与源码实战的差别。

- [ ] **Step 7: 编写系统设计、容量估算与模拟题**

覆盖吞吐/延迟目标、显存预算、并行选择、batching、容错边界和观测指标；添加项目深挖和三档模拟面试。

### Task 7: 交叉去重与迁移审计

**Files:**
- Modify: `docs/CUDA面试八股全集.md`
- Modify: `docs/AI_Infra面试八股全集.md`

- [ ] **Step 1: 确认原文内容全部有归属**

逐项核对原文目录和快问快答：有价值内容必须在新 CUDA 或 AI 文档中出现；错误内容可删除但要由正确题目覆盖。

- [ ] **Step 2: 检查交叉边界**

搜索 Tensor Core、CUDA Graph、NCCL、GEMM/GEMV、Hopper、FP8，在两份文档中确认一边讲机制、一边讲场景，并使用相对链接而不是大段复制。

- [ ] **Step 3: 检查项目证据真实性**

所有“我做过/实测”必须对应实际文件和数据；资料性链接不能冒充实战。

- [ ] **Step 4: 检查技术风险句**

断言以下错误表述不存在：

```text
每个 SM 固定 4 scheduler
__threadfence 自动让其他线程看见
occupancy 达到固定百分比就足够
float4 必然生成一条 128-bit load
Long Scoreboard 就是 DRAM 慢
所有 prefill 都 compute-bound
所有 decode 都 memory-bound
FlashAttention 把计算复杂度变成 O(N)
```

### Task 8: 全文验证与提交

**Files:**
- Test: `docs/CUDA面试八股全集.md`
- Test: `docs/AI_Infra面试八股全集.md`

- [ ] **Step 1: 结构检查**

确认两份文档均有标题、目录、使用方法、B/A/C、15 秒回答、1 分钟展开、追问、误区、项目证据、模拟题和复习清单。

- [ ] **Step 2: 标签统计**

统计 CUDA 文档 B/A/C 题目，验证约 45%/45%/10%；输出实际数量和比例。

- [ ] **Step 3: Markdown 检查**

检查围栏成对、标题层级、表格、`$`/`$$` 数学定界符、本地链接、无意占位符和 `git diff --check`。

- [ ] **Step 4: 官方链接检查**

确认所有外部链接为实际使用的一手来源；框架类时效信息带版本/条件，不引用搜索结果页。

- [ ] **Step 5: 修改范围检查**

确认只修改/新增两份目标文档，不纳入用户其他未提交文件。

- [ ] **Step 6: 提交 CUDA 文档**

```bash
git add -- 'docs/CUDA面试八股全集.md'
git commit -m 'docs: rebuild CUDA interview guide'
```

- [ ] **Step 7: 提交 AI Infra 文档**

```bash
git add -- 'docs/AI_Infra面试八股全集.md'
git commit -m 'docs: add AI Infra interview guide'
```
