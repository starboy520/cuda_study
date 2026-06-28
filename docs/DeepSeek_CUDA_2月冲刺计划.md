# DeepSeek CUDA 岗位 · 2 个月冲刺计划

> 目标公司：DeepSeek（CUDA / GPU kernel 优化 / ML 系统 / AI Infra）
> 时间：8 周（约 2 个月）
> 起点：已经学过 CUDA 基础、内存系统、并行算法、shared memory GEMM、softmax、ncu/nsys/Roofline；C++ 经验较强
> 硬件：本地 T4(sm_75)，另一台 A100(sm_80)
>
> 核心原则：不要把这份计划当“阅读清单”，要当“训练营”。每天都要有代码、有数据、有解释。

---

## 0. 先说清楚：这份计划到底训练什么

DeepSeek / AI Infra / CUDA kernel 优化岗位，面试官不会只问：

```text
你看过哪些资料？
```

更可能问：

```text
这个 kernel 为什么慢？
你怎么测？
你怎么判断 memory-bound 还是 compute-bound？
你怎么优化 GEMM？
Tensor Core 为什么快？
FP8 为什么难？
FlashAttention 为什么省显存？
MoE 的通信瓶颈在哪里？
```

所以本计划的目标不是“学完所有名词”，而是训练四种能力：

```text
1. 能手写核心 kernel。
2. 能 benchmark，并知道数据是否可信。
3. 能用 ncu/nsys/Roofline 找瓶颈。
4. 能把优化过程讲给面试官听。
```

每个主题都必须产出四件东西：

```text
代码：
  一个能编译运行的 .cu / demo。

数据：
  至少一组 benchmark 表格。

分析：
  一段 profiler / Roofline / 访存账解释。

口述：
  一段面试风格的回答，能闭卷讲清楚。
```

如果某一天只看了资料，没有代码、数据或口述，那天不算真正完成。

---

## 1. 8 周总览

```text
Week 1  GEMM register tiling（一）：以 2D register tiling 为主，不再单独实现 1D
Week 2  GEMM register tiling（二）：vectorized load、double buffering、CUTLASS 对照
Week 3  Tensor Core 与混合精度：WMMA、TF32/FP16/BF16、FP8 思想
Week 4  Attention 与 FlashAttention：online softmax、分块 attention、MLA
Week 5  LLM 推理优化：KV cache、paged attention、量化、GEMV
Week 6  多 GPU 与 MoE：NCCL、通信重叠、专家并行、DualPipe
Week 7  作品化：GEMM 项目、Attention 项目、DeepSeek 开源研读
Week 8  面试冲刺：手写 kernel、性能分析问答、系统设计、模拟面试
```

优先级要非常明确。

必须打穿：

```text
GEMM register tiling
Tensor Core / WMMA
Roofline + Nsight Compute 分析
online softmax / FlashAttention 思想
KV cache / MLA / MoE 数据流
DeepGEMM / FlashMLA / DeepEP README 级理解
```

可以先理解，不强求完全手写工业级：

```text
完整 CUTLASS 内部实现
完整 NCCL kernel 细节
完整 FP8 训练系统
完整 DualPipe 工程实现
完整 vLLM / TensorRT-LLM 框架源码
```

---

## 2. 每日固定流程

每天按这个流程来，不要跳。

```text
1. 先写当天目标：
   今天要搞懂什么？
   今天要写哪个文件？
   今天要测哪些指标？

2. 写代码或读核心代码：
   至少 1 小时手写，不只看。

3. 跑 correctness：
   CPU reference / cuBLAS reference / 误差检查。

4. 跑 benchmark：
   至少 512、1024、2048 三档，能跑多大跑多大。

5. 跑 profiler：
   ncu 或 nsys，至少记录 2-3 个关键指标。

6. 写 worklog：
   今天做了什么？
   数据是什么？
   为什么变快或变慢？
   明天要验证什么？

7. 写一句面试口述：
   “如果面试官问 X，我会这样答……”
```

推荐每天最后写到：

```text
notes/deepseek/weekXX_dayYY.md
```

每个 kernel 推荐维护：

```text
README.md
benchmark.md
ncu_notes.md
```

---

## 3. Benchmark 与 Profiler 最低标准

每个 CUDA kernel 至少记录：

```text
GPU 型号
CUDA 版本
编译参数
矩阵/输入规模
block/grid 配置
运行时间 ms
GFLOPS 或 GB/s
正确性误差
```

GEMM 至少记录：

```text
M, N, K
time_ms
GFLOPS = 2*M*N*K / time / 1e9
与 naive/shared/cuBLAS 对比
```

访存型 kernel 至少记录：

```text
logical bytes
GB/s
是否 coalesced
是否有 bank conflict
```

ncu 入门先看：

```text
SM 利用率
DRAM throughput
L2 throughput
achieved occupancy
registers per thread
shared memory throughput
stall reason
Tensor Core pipe utilization
```

Roofline 最低要求：

```text
FLOP
bytes
AI = FLOP / bytes
measured_GFLOPS
roof_at_AI = min(peak_compute, AI * bandwidth)
下一步优化方向
```

---

## Week 1：GEMM Register Tiling（一）

> 本周是分水岭。你已经学过 2D register tiling，所以本周不再花时间单独实现 1D register tiling。  
> 1D 只保留为概念对照：它能帮助解释“一个 thread 可以算多个输出”，但真正要打磨的是 2D register tiling、profiling 和参数实验。

参考：

```text
cuda_deep_course/course/volume06_operators/02B_GEMM_Register_Tiling外积视角.md
cuda_deep_course/course/volume06_operators/02A_GEMM从零到Tiling小白版.md
siboehm CUDA-MMM Kernel 4/5
```

本周代码建议放在：

```text
week05_gemm_advanced/
  gemm_shared_baseline.cu
  gemm_2d_thread_tiling.cu
  benchmark.md
  ncu_notes.md
  roofline.md
```

### Day 1：复盘 shared memory GEMM

目标：

```text
重新手写或整理 shared tiled GEMM。
确认你真的知道 block tile、thread、shared tile 在干什么。
```

任务：

```text
1. 画出 BM x BN 的 C tile。
2. 画出 BK 方向每次搬 A tile 和 B tile。
3. 写清楚每个 thread 负责哪个 C[row, col]。
4. 跑 512/1024/2048。
5. 记录 GFLOPS。
```

重点问题：

```text
shared tiling 已经减少了 global memory 访问，为什么还不够？
```

你要能答：

```text
因为每个 thread 每算一个输出元素，仍然在 K 循环里反复从 shared memory 读 A 和 B。
global memory 压力下降后，瓶颈可能转移到 shared memory、指令吞吐和寄存器复用。
```

产出：

```text
gemm_shared_baseline.cu
benchmark.md 中的 baseline 表格
一段口述：shared tiling 解决了什么，没有解决什么
```

### Day 2：1D 快速对照 + 2D 外积复盘

目标：

```text
不再实现 1D。
只用 1D 帮你对照理解 2D register tiling。
```

1D 只需要知道：

```text
一个 thread 计算 TM 个输出。
它说明寄存器可以缓存部分数据，让一个 thread 复用 A 或 B。
```

但本计划不再要求：

```text
不写 gemm_1d_thread_tiling.cu。
不单独 profile 1D。
不把 1D 当作品重点。
```

真正要复盘的是 2D：

```text
每个 thread 计算 TM x TN 个 C 元素。
regM[TM] 来自 A 的一小列。
regN[TN] 来自 B 的一小行。
acc[TM][TN] 是当前 thread 负责的小 C tile。
```

外积核心：

```text
for k:
  load regM[0..TM-1]
  load regN[0..TN-1]

  for i in TM:
    for j in TN:
      acc[i][j] += regM[i] * regN[j]
```

产出：

```text
一张 1D vs 2D 对照图。
一段口述：为什么 1D 是过渡，2D 是主线。
```

### Day 3：手写或整理 2D register tiling GEMM

目标：

```text
写出或整理 gemm_2d_thread_tiling.cu，并跑通。
```

建议参数：

```text
BM = 64 或 128
BN = 64 或 128
BK = 8 或 16
TM = 4
TN = 4
```

最低要求：

```text
1. 支持 M=N=K=512、1024。
2. 和 CPU reference 或 cuBLAS 对比正确。
3. 输出 time_ms 和 GFLOPS。
4. 比 naive 明显快。
```

不要求第一天就极致。

重点是：

```text
索引正确。
regM[TM] / regN[TN] 正确。
寄存器数组 acc[TM][TN] 正确。
shared memory 加载正确。
边界处理正确。
```

产出：

```text
gemm_2d_thread_tiling.cu
benchmark.md 增加 naive/shared/2D 对比
```

### Day 4：profile 2D register tiling

目标：

```text
用数据证明 2D register tiling 为什么变快或为什么没变快。
```

ncu 重点看：

```text
GFLOPS
achieved occupancy
registers per thread
shared memory throughput
SM busy
stall reason
```

要回答：

```text
2D tiling 提高了哪些复用？
寄存器用量是否上升？
occupancy 是否下降？
如果 occupancy 下降了，为什么仍可能更快？
```

产出：

```text
ncu_notes.md
一张 naive/shared/2D 的表格
一段口述：性能变化原因
```

### Day 5：2D 参数实验

目标：

```text
理解 BM/BN/BK/TM/TN 对性能的影响。
```

实验维度：

```text
BM/BN:
  block 负责的 C tile 大小。

BK:
  每次 K 方向搬多少。

TM/TN:
  每个 thread 负责的小 C tile。

threads per block:
  通常约等于 BM*BN/(TM*TN)。
```

产出：

```text
一张参数实验表。
记录每组参数的 GFLOPS、registers/thread、occupancy。
```

### Day 6：shared/register 访存账 + Roofline

目标：

```text
从数据移动角度解释为什么 2D register tiling 有收益。
```

产出：

```text
计算 naive/shared/2D 的 FLOP、bytes、AI。
写一段 Roofline 判断。
结合 ncu 解释瓶颈从 DRAM 转向 shared/register/compute 的可能性。
```

### Day 7：Week 1 复盘

目标：

```text
把 naive -> shared -> 2D register tiling 讲成一条连续的优化链。
1D 只作为中间概念对照，不作为重点实现版本。
```

必须完成：

```text
1. 一张优化阶梯图。
2. 一张 benchmark 表。
3. 一张 Roofline / AI 解释表。
4. 一段 3 分钟口述。
```

口述模板：

```text
naive GEMM 的问题是 global memory 重复读取 A/B，AI 很低。
shared tiling 把 A/B tile 搬到 shared memory，提高了 global memory 复用。
但 shared GEMM 中每个 thread 仍频繁从 shared memory 读数据。
1D thread tiling 是过渡概念：它说明一个 thread 可以计算多个输出，并在寄存器里复用 A 或 B。
2D thread tiling 才是本周主线：一个 thread 计算 TM x TN 小块，用 regM/regN 做外积，增加寄存器级复用。
代价是寄存器更多、occupancy 可能下降，但如果减少了访存和指令瓶颈，整体仍可能更快。
```

本周验收：

```text
[ ] gemm_shared_baseline.cu
[ ] 1D vs 2D 概念对照图
[ ] gemm_2d_thread_tiling.cu
[ ] benchmark.md
[ ] ncu_notes.md
[ ] roofline.md
[ ] 能闭卷讲 GEMM 优化阶梯
```

---

## Week 2：GEMM Register Tiling（二）

> 本周不追求写出 CUTLASS，但要知道工业 GEMM 为什么还会继续做 vectorized load、double buffering、warp tiling 和 Tensor Core。

### Day 1：float4 向量化 global load

目标：

```text
减少 global -> shared 加载指令数量，提高 load/store 效率。
```

任务：

```text
1. 在 GEMM 的 A/B tile 加载处尝试 float4。
2. 确认地址 16-byte 对齐。
3. 处理边界。
4. 对比普通 float load 和 float4 load。
```

ncu 看：

```text
global load efficiency
memory transaction
L2 throughput
指令数量
```

产出：

```text
gemm_vectorized_load.cu
一段说明：float4 为什么可能更快，什么时候会出问题
```

### Day 2：shared memory 布局与 bank conflict

目标：

```text
理解为什么 shared memory 也会成为瓶颈。
```

任务：

```text
1. 对比 As[BM][BK] 与转置/填充布局。
2. 尝试看 shared memory padding。
3. 用 ncu 看 shared bank conflict 或 shared throughput。
```

重点：

```text
shared memory 不是“无限快”。
访问模式不好也会冲突。
```

产出：

```text
shared_layout_notes.md
```

### Day 3：double buffering 概念与手写小版本

目标：

```text
理解 load 当前 tile 和 compute 当前 tile 为什么不能完全串行。
```

普通流程：

```text
load tile 0
compute tile 0
load tile 1
compute tile 1
```

double buffering 目标：

```text
compute tile 0 的同时，提前准备 tile 1。
```

在 T4/A100 的手写入门版中，不一定能做到完美异步。

最低要求：

```text
1. 写出双 shared buffer 的结构。
2. 理解 ping-pong buffer。
3. 讲清为什么 Ampere 以后 cp.async 更适合做这个。
```

产出：

```text
gemm_double_buffering_sketch.cu
double_buffering_notes.md
```

### Day 4：CUTLASS 入门跑通

目标：

```text
不是读完 CUTLASS，而是建立映射关系。
```

任务：

```text
1. clone 或阅读 CUTLASS。
2. 跑通一个 SGEMM / HGEMM 示例。
3. 找到 Threadblock tile、Warp tile、Instruction tile。
4. 对照自己手写的 BM/BN/BK/TM/TN。
```

产出：

```text
cutlass_run_log.md
一张 CUTLASS 概念映射表
```

### Day 5：warp tiling 与三级 tile

目标：

```text
理解高性能 GEMM 为什么分成 block tile、warp tile、instruction tile。
```

你要能讲：

```text
block tile:
  一个 CTA 负责的大 C tile。

warp tile:
  一个 warp 负责 block tile 的一部分。

instruction tile:
  一条 MMA / FMA 指令负责的小矩阵。
```

产出：

```text
warp_tiling_notes.md
```

### Day 6：整理 GEMM 优化总表

目标：

```text
把 naive 到 CUTLASS 的每一步放进一张表。
```

表格列：

```text
版本
解决的问题
新增代价
主要瓶颈
GFLOPS
AI
ncu 证据
```

产出：

```text
gemm_optimization_ladder.md
```

### Day 7：Week 2 复盘

必须能讲：

```text
float4 解决什么？
shared memory bank conflict 是什么？
double buffering 解决什么？
CUTLASS 为什么要分层 tile？
为什么手写 GEMM 很难接近 cuBLAS？
```

本周验收：

```text
[ ] gemm_vectorized_load.cu
[ ] double_buffering_notes.md
[ ] cutlass_run_log.md
[ ] gemm_optimization_ladder.md
[ ] 能讲 naive -> shared -> register -> vectorized -> double buffering -> warp tiling -> Tensor Core
```

---

## Week 3：Tensor Core 与混合精度

> 本周要分清：T4/A100 能实践什么，FP8 主要先理解什么。

现实边界：

```text
T4:
  可练 FP16 Tensor Core。

A100:
  可练 TF32、FP16、BF16 Tensor Core。

FP8:
  真正高性能 FP8 Tensor Core 主要是 Hopper 以后。
  本阶段重点理解 E4M3/E5M2、scaling、amax、量化误差、DeepGEMM 思路。
```

### Day 1：混合精度基础

任务：

```text
整理 FP32 / TF32 / FP16 / BF16 / FP8 / INT8 的对照表。
```

表格列：

```text
格式
指数位
尾数位
范围
精度
常见用途
风险
```

产出：

```text
mixed_precision_table.md
```

### Day 2：Tensor Core 原理

目标：

```text
知道 Tensor Core 是做矩阵 MMA 的专用硬件。
```

要讲清：

```text
CUDA Core:
  标量 / 向量 FMA。

Tensor Core:
  矩阵块 MMA，例如 D = A*B + C。
```

产出：

```text
tensor_core_notes.md
```

### Day 3：WMMA FP16 GEMM

任务：

```text
手写 wmma_fp16_gemm.cu。
```

最低要求：

```text
1. 使用 nvcuda::wmma::fragment。
2. 跑通 256/512/1024。
3. 和 CPU 或 cuBLAS 对比正确。
4. 输出 GFLOPS。
```

产出：

```text
wmma_fp16_gemm.cu
```

### Day 4：profile Tensor Core

ncu 看：

```text
Tensor pipe utilization
HMMA / MMA 指令
SM busy
occupancy
global memory throughput
```

对比：

```text
FP32 CUDA core GEMM
FP16 WMMA GEMM
cuBLAS
```

产出：

```text
tensor_core_profile.md
```

### Day 5：TF32 / BF16 / FP8

任务：

```text
在 A100 上理解 TF32 默认行为。
整理 BF16 为什么适合训练。
整理 FP8 E4M3/E5M2 和 scaling。
```

重点：

```text
FP8 不是简单把 float 变成 8 bit。
它需要 scale、amax 统计、溢出控制和误差管理。
```

产出：

```text
fp8_scaling_notes.md
```

### Day 6：读 DeepGEMM

目标：

```text
不是读完源码，而是读懂 README 和核心设计词汇。
```

关注：

```text
FP8 GEMM
grouped GEMM
JIT compile
MoE 场景
scale / layout
```

产出：

```text
deepgemm_reading_notes.md
```

### Day 7：Week 3 复盘

必须能讲：

```text
Tensor Core 做什么？
WMMA fragment 是什么？
FP16 和 BF16 区别？
TF32 为什么对 A100 重要？
FP8 为什么省带宽，但为什么训练更难？
DeepGEMM 大概解决什么问题？
```

---

## Week 4：Attention 与 FlashAttention

> 本周要把 softmax、online softmax、attention、FlashAttention 连成一条线。

### Day 1：标准 Attention 账本

任务：

```text
写清 QK^T、softmax、PV 三步的 shape。
计算 FLOP 和 memory bytes。
说明为什么完整 attention matrix 是瓶颈。
```

产出：

```text
attention_memory_accounting.md
```

### Day 2：online softmax

任务：

```text
手写 online_softmax.cu。
```

必须理解：

```text
m_new = max(m_old, x)
s_new = s_old * exp(m_old - m_new) + exp(x - m_new)
```

产出：

```text
online_softmax.cu
online_softmax_notes.md
```

### Day 3：FlashAttention 思想

任务：

```text
画出 Q block、K block、V block 的数据流。
说明为什么不存完整 S = QK^T。
说明 online softmax 如何在 block 间合并。
```

产出：

```text
flash_attention_dataflow.md
```

### Day 4：简化版分块 Attention

任务：

```text
写一个教学版 tiled_attention.cu。
不追极致性能，追求逻辑正确。
```

最低要求：

```text
1. 小 shape 跑通。
2. 和 CPU/PyTorch 参考对齐。
3. 能讲 block 间 m/s 怎么更新。
```

### Day 5：FlashMLA / MLA

任务：

```text
读 FlashMLA README。
整理 MLA 为什么减少 KV cache。
```

产出：

```text
mla_flashmla_notes.md
```

### Day 6：profile softmax / attention

任务：

```text
用 ncu 看 softmax 或 tiled attention。
判断它是 bandwidth-bound 还是 compute-bound。
```

产出：

```text
attention_profile.md
```

### Day 7：Week 4 复盘

必须能讲：

```text
标准 attention 的显存瓶颈是什么？
online softmax 为什么需要重新缩放？
FlashAttention 为什么是 IO-aware？
MLA 省了什么？
```

---

## Week 5：LLM 推理优化

> 本周重点不是手写完整推理框架，而是能讲清推理瓶颈链路。

每日任务：

```text
Day 1  推理 vs 训练：prefill、decode、自回归、batch、latency、throughput
Day 2  KV cache：显存公式、长上下文、MQA/GQA/MLA 为什么省
Day 3  Paged Attention：分页管理、碎片、block table、vLLM 思想
Day 4  量化推理：INT8/INT4、per-tensor/per-channel、GPTQ/AWQ 大意
Day 5  GEMV / batch=1 GEMM：为什么 decode 常 memory-bound
Day 6  框架地图：vLLM / TensorRT-LLM / SGLang 分别解决什么
Day 7  复盘：画出一次 decode step 的数据流和瓶颈
```

本周产出：

```text
llm_inference_notes.md
kv_cache_accounting.md
paged_attention_notes.md
decode_step_dataflow.md
```

验收问题：

```text
为什么 decode 阶段通常比 prefill 更难吃满 GPU？
KV cache 显存怎么估算？
Paged Attention 解决什么问题？
为什么 batch=1 时 GEMV/GEMM 优化策略不同？
```

---

## Week 6：多 GPU / 通信 / MoE / 并行

> 本周不要陷入完整分布式框架源码，先抓通信原语和数据流。

每日任务：

```text
Day 1  并行方式地图：data / tensor / pipeline / expert / context parallel
Day 2  NCCL 原语：all-reduce、all-gather、reduce-scatter、all-to-all
Day 3  通信量估算：矩阵切分后通信在哪里发生
Day 4  通信计算重叠：stream、event、bucket、反向传播重叠
Day 5  MoE：token routing、top-k expert、dispatch/combine、负载均衡
Day 6  DeepEP / DualPipe：读 README，整理它们分别优化什么
Day 7  复盘：画出 MoE 一层的通信与计算数据流
```

本周产出：

```text
parallelism_map.md
nccl_collectives.md
moe_dataflow.md
deepep_dualpipe_notes.md
```

验收问题：

```text
all-reduce 和 reduce-scatter 区别？
Tensor parallel 的通信发生在哪里？
MoE 为什么需要 all-to-all？
DeepEP 主要优化什么？
DualPipe 主要减少什么？
```

---

## Week 7：作品化与 DeepSeek 开源研读

> 作品不能 Week 7 才开始补。本周是把前 6 周素材整理成能展示、能讲、能投递的形态。

### 作品一：GEMM 优化阶梯

目录建议：

```text
projects/gemm_optimization_ladder/
  README.md
  src/
    gemm_naive.cu
    gemm_shared.cu
    gemm_2d_thread_tiling.cu
    gemm_vectorized.cu
    wmma_fp16_gemm.cu
  docs/
    benchmark.md
    roofline.md
    ncu_analysis.md
    optimization_story.md
```

README 必须包含：

```text
项目目标
每个版本解决什么问题
benchmark 表格
Roofline 图或文字分析
ncu 关键证据
下一步可优化方向
```

### 作品二：Softmax / FlashAttention 教学版

目录建议：

```text
projects/attention_kernels/
  README.md
  src/
    softmax_warp.cu
    online_softmax.cu
    tiled_attention.cu
  docs/
    online_softmax.md
    flash_attention_dataflow.md
    profile.md
```

### DeepSeek 开源研读

研读顺序：

```text
1. DeepGEMM README：FP8 GEMM / grouped GEMM / JIT / MoE
2. FlashMLA README：MLA / KV cache / attention kernel
3. DeepEP README：expert parallel / all-to-all / dispatch/combine
4. DualPipe README：pipeline bubble / computation-communication overlap
5. DeepSeek-V3 技术报告：MLA / MoE / FP8 / DualPipe
```

本周验收：

```text
[ ] GEMM 项目 README 可展示
[ ] Attention 项目 README 可展示
[ ] DeepSeek 技术栈速记卡
[ ] 10 分钟项目讲解能讲顺
```

---

## Week 8：面试冲刺

### Day 1：手写基础 kernel

闭卷练：

```text
vector add
reduction warp shuffle
transpose shared memory
softmax warp/block
```

目标：

```text
每个 10-20 分钟能写出正确骨架。
```

### Day 2：手写 GEMM / WMMA 骨架

闭卷练：

```text
shared GEMM
1D register tiling 概念对照
2D register tiling 核心循环
WMMA GEMM fragment 骨架
```

### Day 3：性能优化问答

练这类问题：

```text
一个 kernel 慢，你怎么查？
Roofline 怎么判断？
occupancy 低一定不好吗？
memory-bound 怎么优化？
compute-bound 怎么优化？
```

### Day 4：GPU 架构快问快答

主题：

```text
SM / warp / lane / SIMT
memory hierarchy
coalescing
bank conflict
occupancy
stall reason
Tensor Core
```

### Day 5：AI Infra 系统问题

主题：

```text
prefill vs decode
KV cache
paged attention
MoE
NCCL collectives
communication overlap
```

### Day 6：C++ / 算法恢复

任务：

```text
LeetCode 中等 2-3 题。
C++ RAII、move、模板、并发、内存模型复盘。
准备 2 个过去项目深挖故事。
```

### Day 7：全真模拟面试

流程：

```text
5 分钟自我介绍
15 分钟项目深挖
20 分钟 CUDA kernel / 性能优化
15 分钟 AI Infra / LLM 系统
20 分钟 coding
5 分钟反问
```

产出：

```text
interview_cards.md
mock_interview_review.md
```

---

## 附录 A：每周复盘模板

每周末写：

```text
# Week X 复盘

## 本周完成

- [ ] 代码
- [ ] benchmark
- [ ] profiler
- [ ] notes
- [ ] 面试口述

## 最重要的 3 个收获

1.
2.
3.

## 还没真正懂的地方

1.
2.
3.

## 下周优先级

1.
2.
3.

## 面试口述卡

Q:
A:
```

---

## 附录 B：面试高频问题清单

架构类：

```text
warp 是什么？SIMT 是什么？
lane 是什么？
分支分歧为什么慢？
global / shared / register / L2 / constant memory 区别？
coalescing 是什么？
bank conflict 是什么？
occupancy 是什么？为什么 100% occupancy 不一定最快？
```

性能类：

```text
一个 kernel 慢，你怎么排查？
Roofline 怎么用？
怎么算 GFLOPS、GB/s、AI？
memory-bound 怎么优化？
compute-bound 怎么优化？
ncu 和 nsys 分别看什么？
```

GEMM 类：

```text
naive GEMM 为什么慢？
shared tiling 解决什么？
register tiling 解决什么？
1D tiling 和 2D tiling 区别？
为什么 2D tiling 是外积？
vectorized load 为什么可能更快？
double buffering 解决什么？
CUTLASS 的 threadblock/warp/instruction tile 是什么？
```

Tensor Core / 精度类：

```text
Tensor Core 做什么？
WMMA fragment 是什么？
FP16、BF16、TF32、FP8 区别？
FP8 为什么需要 scaling？
DeepGEMM 主要解决什么？
```

Attention / 推理类：

```text
标准 attention 为什么显存压力大？
online softmax 公式是什么？
FlashAttention 为什么 IO-aware？
KV cache 怎么算显存？
Paged Attention 解决什么？
MLA 为什么省 KV cache？
```

多 GPU / MoE 类：

```text
all-reduce / all-gather / reduce-scatter / all-to-all 区别？
Tensor parallel 怎么切矩阵？
MoE 为什么需要 expert parallel？
DeepEP 优化什么？
DualPipe 优化什么？
```

---

## 附录 C：两周里程碑

第 2 周末：

```text
能手写 2D register tiling 核心循环。
能讲 1D / 2D register tiling 的区别，但不要求单独实现 1D。
能讲 naive -> shared -> register -> vectorized -> double buffering。
有 GEMM benchmark 表和 ncu notes。
```

第 4 周末：

```text
能写 WMMA FP16 GEMM。
能讲 Tensor Core / TF32 / BF16 / FP8。
能写 online softmax。
能讲 FlashAttention 数据流。
```

第 6 周末：

```text
能讲 KV cache / paged attention / MLA。
能讲 NCCL collectives。
能画 MoE dispatch/combine 数据流。
能解释 DeepEP / DualPipe 大方向。
```

第 8 周末：

```text
GEMM 项目可展示。
Attention 项目可展示。
手写 kernel 不慌。
性能优化问答能形成闭环。
能完成一次全真模拟面试。
```

---

## 附录 D：心态与策略

```text
1. Week 1-4 是核心技术深度，不要赶进度牺牲质量。
2. 每个 kernel 都要“写出来、测出来、解释出来”。
3. 看到高级项目不要慌，先读 README 和数据流，再慢慢看源码。
4. DeepSeek 是冲刺目标，同时也可以投其他 CUDA / AI Infra 岗位。
5. 这套能力不是只为 DeepSeek，GEMM、Tensor Core、Attention、MoE、Profiler 是通用硬技能。
```

最后记住一句话：

```text
你的竞争力不是“我看过 CUDA”，而是：
我能把一个 kernel 从 naive 写到优化版，测出数据，解释瓶颈，并讲清下一步怎么优化。
```
