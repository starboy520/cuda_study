# AI Infra + CUDA 深水区四周聚焦学习计划

> **执行方式**：核心 kernel 必须由学习者亲手实现；AI 助手只负责概念讲解、三级提示、CPU/reference、测试与计时脚手架，以及完成后的 review 和验证。
>
> **配套能力地图**：[CUDA 核心能力补缺学习文档](CUDA核心能力补缺学习文档.md)。它用于查询七层能力、知识缺口和验收标准，不构成第二条并行日程。

**目标：** 用四周把现有分散知识收口成“一个求职定位、两个可讲项目、一条底层性能证据链和一个 PagedAttention 学习实验”。

**主线：** 求职落点是 AI Infra / C++ Model Serving / 推理性能；CUDA Kernel 是差异化壁垒。GEMM 是旗舰性能工程项目，FlashAttention 是推理算子副项目，PagedAttention 只学习两天、不作品化。

**环境：** NVIDIA A100 80GB（`sm_80`）、CUDA C++17、nvcc、Compute Sanitizer、Nsight Compute、cuobjdump、cuBLAS。

---

## 1. 这算不算聚焦

算，但前提是严格遵守以下边界。

### 1.1 四周只有一条职业叙事

> 约 12 年 C++ 高性能在线系统与推理工程经验，正在把性能工程方法迁移到 GPU；目标是 AI Infra / Model Serving / 推理性能岗位，同时具备可验证的 CUDA Kernel 优化能力。

不是同时准备五类互不相干的岗位，也不是四周内成为 CUTLASS、vLLM、多 GPU 和编译器专家。

### 1.2 四周只有四个学习对象

| 优先级 | 对象               | 定位                              | 时间预算                 |
| ------ | ------------------ | --------------------------------- | -----------------------: |
| P0     | FP32 GEMM          | 旗舰公开项目、性能方法论载体      | 7 天                     |
| P0     | PTX/SASS 与 ncu    | 底层证据链，不单独做项目          | 7 天                     |
| P1     | FlashAttention     | 第二个公开项目，限定教学/研究实现 | 7 天                     |
| P1     | PagedAttention     | 学习实验，不作品化                | 2 天                     |
| P0     | 面试转换与投递准备 | 把能力变成面试输出                | 其余 5 天 + 每日固定时间 |

### 1.3 明确冻结

四周内不新增以下主线：

- MoE kernel、Expert Parallel、DeepEP；
- NCCL、多 GPU 通信实测；
- vLLM、TensorRT-LLM、CUTLASS 大规模源码阅读；
- Hopper TMA/WGMMA 实现；
- 完整生产级 PagedAttention、block allocator、continuous batching 调度器；
- 新的零散算子收集；
- 为“知识完整”继续生成大而全的文档；
- 同时重构整个 `cuda_study` 目录。

需要这些知识时只允许记入“第四周后候选清单”，不准当日切换主线。

### 1.4 一句话判断是否跑偏

每天开始前问：

> 今天的任务能否直接增强 GEMM/FlashAttention 作品、PTX/SASS 证据链或 AI Infra 面试表达？

不能，就不做。

---

## 2. 四周结束后能学到什么

### 2.1 能独立完成的代码能力

四周结束后应能从空白或最小脚手架写出：

1. naive GEMM；
2. shared-memory tiled GEMM；
3. 2D register-tiled GEMM 的核心索引和外积循环；
4. 数值稳定的 online softmax；
5. 不物化 $N\times N$ scores 的 tiled attention；
6. 根据 block table 读取非连续 KV cache 的最小 paged decode attention；
7. reduction/softmax/GEMV 中至少两道限时面试 kernel。

“能独立”要求：不看已有完整实现，只允许查看接口、公式和自己写的提示卡。

### 2.2 能完成的性能分析能力

面对一个 CUDA kernel，能够按顺序回答：

1. 正确性是否可信，覆盖了哪些边界；
2. FLOP、bytes 和 arithmetic intensity 大致是多少；
3. 正常运行时间如何测，什么结果不能横比；
4. 是 latency、bandwidth、compute、occupancy、依赖链还是 launch 方向的问题；
5. PTX/SASS 中对应的数据流和关键指令类别在哪里；
6. 修改了什么单一变量；
7. 指标和墙钟是否共同支持假设；
8. 结论的硬件、shape、dtype 和实现边界是什么。

### 2.3 PTX/SASS 的现实目标

四周目标不是“会逐行翻译汇编”，而是达到以下水平：

- 理解 CUDA C++ → PTX → ptxas → cubin/SASS 的编译链；
- 能为指定 kernel 独立生成 PTX、ptxas 资源报告和 A100 SASS；
- 能从入口开始定位 thread index、地址计算、global/shared load、计算、同步和 store；
- 能识别 `float4` 是否真的形成宽加载；
- 能结合 ptxas、SASS 和 ncu 判断 register pressure、spill 和 stall；
- 能说明 WMMA、PTX `mma.sync`、SASS HMMA 分别属于哪一层；
- 不把某个助记符、某个 stall 百分比或高 occupancy 单独当成性能结论。

### 2.4 AI Infra 能力

能够把 CUDA 项目连接到真实推理系统：

- prefill/decode 的 shape 与瓶颈差异；
- GEMM/GEMV 在推理链中的位置；
- online softmax、FlashAttention 与 KV cache 的关系；
- PagedAttention 如何解决预留、碎片和搬迁；
- continuous batching 为什么需要灵活的 KV block 管理；
- 单 kernel 延迟、端到端延迟、系统吞吐和显存利用率不能混为一谈。

### 2.5 四周后仍然不会什么

必须诚实保留这些边界：

- 不会生产级 CUTLASS/FlashAttention 内核；
- 没有多 GPU/NCCL 实测；
- 没有完整 vLLM 调度器实现；
- 没有 Hopper TMA/WGMMA 实测；
- 不能仅凭四周训练宣称“精通 PTX/SASS”或“CUDA 专家”。

---

## 3. 每日固定执行模板

按每天约 8 小时设计；如果当天只有 4 小时，按相同比例缩放，但不能取消验收。

| 时间  | 内容           | 结束条件                                             |
| ----: | -------------- | ---------------------------------------------------- |
| 0.5 h | 回忆与计划     | 不看资料写出昨日三个结论和今日唯一目标               |
| 3.0 h | 核心手写       | 核心 kernel/实验由本人完成，能编译或留下最小失败用例 |
| 1.5 h | 正确性与边界   | reference、异常检查、非整齐 shape、sanitizer         |
| 1.5 h | 性能与底层证据 | 正常计时；按当天主题采集 ncu/PTX/SASS                |
| 1.0 h | AI Infra/面试  | 过往项目或当天 CUDA 主题的闭卷口述                   |
| 0.5 h | worklog        | 记录数据、结论、失败点和明日第一步                   |

### 3.1 每日四类产出

```text
概念：一张自己画的图或一页闭卷解释
代码：一个自己写的核心实现或最小实验
证据：正确性 + 计时/资源/指令中与主题相关的证据
口述：2～5 分钟“结论→原理→证据→边界”
```

### 3.2 防止再次发散的规则

1. **WIP=1**：同一时刻只允许一个未完成主任务。
2. **先写后看**：先从空白写 30～60 分钟，再查资料。
3. **资料按需读**：只读能解决当前失败点的章节。
4. **两小时止损**：同一 bug 两小时无新证据，就缩小到最小复现并请求提示。
5. **性能结论必须带条件**：GPU、shape、dtype、版本、计时方法。
6. **失败优化必须保留**：负结果也是性能工程证据。
7. **每天不创建新路线文档**：只更新本计划和当周 worklog。
8. **核心代码不由 AI 代写**：AI 可以提供三级提示，但最终实现必须本人完成。

### 3.3 三级提示规则

遇到困难按顺序使用：

- 一级提示：只提示检查方向或不变量；
- 二级提示：指出可能出错的数据流、索引或同步阶段；
- 三级提示：给伪代码或局部公式，不给完整 kernel；
- 最终 review：本人写完后再逐行审查、编译、对拍和 profile。

---

# Week 1：GEMM 收口——从“写过”到“能独立重建” ✅ 已完成

> 完成状态（2026-07-12）：独立公开仓已完成 Naive → Shared → Register → Vectorized → Async 16B，正式 benchmark、full sanitizer、ncu/SASS 证据均已发布并推送。

## 本周目标

只重建三个核心版本：naive、shared tiling、2D register tiling。`float4`、`cp.async` 和指令证据放到第二周，避免同时处理太多变量。

## 本周完成标准

- 三版均由本人从干净文件重写核心 kernel；
- 同一 runner、同一输入、同一 reference、同一计时口径；
- 覆盖矩形与非整齐 shape；
- 能解释每次优化改变了哪一级数据复用；
- 能在白板上推导 thread → output tile 映射；
- 公开数据只来自重建代码，不沿用历史数字。

## Day 1：冻结实验协议

### 概念目标

先定义“什么叫公平比较”，不急着优化。

### 动手任务

1. 确定接口：row-major FP32，$C=A\times B$；
2. 固定参数：`M/N/K`、seed、warmup、iterations、kernel 名称；
3. 准备 CPU double 累加 reference；
4. 大尺寸准备关闭 TF32 的 cuBLAS FP32 reference；
5. 统一误差输出：`max_abs`、`max_rel`、最差位置、NaN/Inf；
6. 测试 shape 至少包含：

```text
1×1×1
3×5×7
17×31×13
63×65×33
127×130×129
512×512×512
1024×1024×1024
2048×2048×2048
```

### 当日验收

- 空 kernel 必须被测试识别为失败；
- runner 能区分 validation 与 benchmark；
- 内存分配和 H2D/D2H 不计入 kernel 时间；
- 记录环境、编译参数和计时方法。

## Day 2：Naive GEMM

### 必须自己写

- thread 到 `(row,col)` 的映射；
- K 维循环；
- 边界判断；
- 输出写回。

### 必须解释

- 相邻线程分别读取 A/B 的什么地址；
- 哪一侧访问合并、哪一侧存在重复读取；
- 为什么 naive 是可信基线而不是“废代码”。

### 当日验收

- 全部小/非整齐 shape PASS；
- memcheck 0 error；
- 记录 512/1024/2048 的时间和 GFLOPS；
- 3 分钟闭卷写出 kernel。

## Day 3：Shared-memory Tiling

### 必须自己写

- A/B tile 协作加载；
- K tile 循环；
- 越界补零；
- 两个必要同步点；
- tile 内累加。

### 必须解释

- global memory 流量为什么下降；
- `__syncthreads()` 前后的生产者/消费者关系；
- 为什么最后一个 K tile 要补零；
- shared memory 解决了 global reuse，但还没有充分解决什么。

### 当日验收

- 任意 `M/N/K` 正确；
- memcheck、racecheck、synccheck 通过；
- 与 naive 同条件 benchmark；
- 若未变快，先检查规模、映射和计时，不直接继续叠加优化。

## Day 4：2D Register Tiling 索引推导

### 上午只做纸面推导

锁定一组参数，例如：

```text
BM=128, BN=128, BK=8
TM=8, TN=8
threads=(BM/TM)×(BN/TN)=256
```

画清楚：

- 一个 block 负责哪个 `BM×BN` C tile；
- 一个 thread 负责哪个 `TM×TN` micro tile；
- `regM[TM]`、`regN[TN]`、`acc[TM][TN]` 分别是什么；
- K 的每一步为什么形成外积。

### 下午实现

只实现 correctness，不追参数最优。

### 当日验收

给定随机 `blockIdx/threadIdx`，能不用看代码算出该线程负责的所有 C 坐标。

## Day 5：2D Register Tiling 正确性与资源

### 任务

- 完成边界路径；
- 检查 shared tile 布局；
- 跑全部 validation shape；
- 记录 ptxas registers/thread、shared memory/block；
- 检查是否发生 spill；
- 运行 sanitizer。

### 当日验收

- 不通过就不 benchmark；
- 能解释 register reuse 如何减少 shared load；
- 能解释 occupancy 为什么不是越高越好。

## Day 6：统一 benchmark 与参数小实验

只比较少量参数，不做无穷 sweep：

- 一个 shared tile 参数对照；
- 两个 `TM×TN` 对照；
- 512/1024/2048 三档；
- cuBLAS FP32 基线明确关闭 TF32。

记录：

```text
shape
time_ms
GFLOPS
speedup_vs_naive
speedup_vs_previous
percent_of_cublas
registers_per_thread
shared_memory_per_block
```

### 当日验收

每个数字都能对应到当前源码、当前命令和当前硬件。

## Day 7：闭卷重建与周复盘

### 闭卷任务

1. 30 分钟写 naive；
2. 60 分钟写 shared tiled 核心；
3. 90 分钟写 2D register tiled 核心索引和外积；
4. 5 分钟解释优化阶梯；
5. 记录卡住的具体位置，而不是写“还不熟”。

### 进入第二周的门槛

- 三版 correctness 全过；
- 至少 naive/shared 能从空白写出；
- register 版能独立推导映射；
- benchmark 协议已固定；
- 未完成项不超过半天工作量。

---

# Week 2：PTX/SASS 从零到能完成一次诊断 🔄 当前执行

> 当前唯一执行入口：[Week 2：PTX / SASS 与性能诊断执行清单](current/Week2_PTX_SASS与性能诊断.md)。本总计划不再复制第二套日程、命令或实验选择。

本周 Day 1–4 固定使用已发布的 FP32 GEMM 作品作为主诊断标本；Day 5–6 只增加一个写在 `/tmp` 的依赖链 microbenchmark，用于严格单变量实验；Day 7 只观察已有 WMMA 实现，补齐 Tensor Core API → PTX → SASS 三层证据。所有 PTX、cubin、SASS、ncu 报告和临时代码只写入 `/tmp/cuda_focus/week02/`，不写回公开作品仓。

通过标准：能独立生成三层产物、追踪 Naive 数据流、解释资源与 scheduler/scoreboard、完成一次严格单变量诊断，并讲清 WMMA / PTX MMA / SASS HMMA 的证据边界。

---

# Week 3：FlashAttention 作品化

## 本周定位

不是实现生产级 FlashAttention-2，而是把已有教学代码重建为一个诚实、可复现的 FP32 research/educational project。

## 首版范围

```text
单 GPU A100 sm_80
FP32 input/accumulate
单 batch、单 head
D <= 128
causal / non-causal
任意 N，含非 tile 整除
naive materialized attention
online-softmax tiled attention
cp.async pipelined attention
CPU reference + sanitizer + benchmark + ncu
```

## Day 1：数学、reference 与 naive baseline

- 闭卷写出 $QK^T/\sqrt{D}$、mask、softmax、PV；
- CPU double/float reference；
- naive 三阶段 CUDA baseline；
- 显式统计中间 $N\times N$ scores 的空间；
- 覆盖小尺寸、非整齐尺寸、causal/non-causal。

验收：能解释 softmax 沿 key 维做，causal mask 为什么必须是负无穷语义。

## Day 2：Online Softmax

必须自己推导和实现 running state：

$$
m_{new}=\max(m_{old},m_{block})
$$

$$
\alpha=e^{m_{old}-m_{new}}
$$

$$
l_{new}=\alpha l_{old}+\sum_j e^{s_j-m_{new}}
$$

$$
O_{acc,new}=\alpha O_{acc,old}+\sum_j e^{s_j-m_{new}}V_j
$$

验收：能解释新 max 出现时为什么分母和输出累加器必须同时重缩放，以及空状态如何避免 $-\infty-(-\infty)$ 产生 NaN。

## Day 3：Tiled Attention

核心 kernel 由本人从干净接口重写：

- 一个 block 处理一条 query；
- Q 放 shared；
- K/V 按 `BC` 分块；
- 当前 scores tile 放 shared；
- 只维护 running `m/l/O_acc`；
- 不物化完整 scores。

验收：随机打断点能解释当前 tile、当前 running state 和输出各自在什么地址空间。

## Day 4：并行归约与线程映射

清理当前教学版中线程 0 串行 max/sum 的主要限制：

- block/warp max reduction；
- block/warp sum reduction；
- 明确 score 轴和 output feature 轴不同；
- 对比修改前后的正确性、资源和墙钟。

如果当日未完成，保持教学版正确实现并把并行归约列为已知限制，不伪装完成。

## Day 5：`cp.async` 双缓冲

将第二周的指令层能力用于真实数据流：

- 画 prologue/steady-state/epilogue；
- 两个 K/V shared buffer；
- 预取 next tile 与计算 current tile 重叠；
- 正确处理最后一个 tile；
- 明确 wait、release 和 buffer 重用关系；
- 导出 SASS，验证当前编译结果中实际异步搬运路径。

验收：`N=37` 等非整齐尺寸必须通过；不能只测整齐 `N`。

## Day 6：统一 benchmark 与 ncu

对比：

- naive materialized；
- tiled online；
- pipelined tiled。

至少记录：

```text
N, D, causal
time_ms
intermediate_memory_bytes
registers/thread
shared_memory/block
SM throughput
DRAM/L2 throughput
long/short scoreboard
active/eligible warps
```

结论必须区分：

- 算法空间从 $O(N^2)$ 降到 tile/running-state 级；
- stall 改善；
- 实际墙钟是否改善；
- 小 grid 填不满 A100 时为什么指标改善不等于等比例加速。

## Day 7：作品清理与口述

- 去掉“你来写”、已完成 TODO 和教学对话；
- 写清实现边界；
- 记录失败优化；
- 所有公开数字重新跑；
- 5 分钟版：问题→online softmax→tiling→流水→证据→限制；
- 10 分钟版：加入公式、线程映射和 profiler 追问。

### 第三周通过标准

- naive/tiled/pipelined 三版 correctness；
- 至少 tiled 核心能独立重建；
- sanitizer 通过；
- 有公平 benchmark；
- 有一条 ncu + SASS 证据；
- 明确写“educational/research implementation”，不称生产级。

---

# Week 4：PagedAttention 两天 + 面试转换五天

## Day 1：PagedAttention 存储模型与 CPU 模拟

### 学习目标

理解 PagedAttention 主要解决 KV cache 管理，不改变 attention FLOP。

### 必须手推

给定 token `t` 与 block size `B`：

$$
logical\_block=\lfloor t/B\rfloor
$$

$$
offset=t\bmod B
$$

$$
physical\_block=block\_table[logical\_block]
$$

### CPU 模拟器范围

- 三个不同长度请求；
- physical block pool；
- 随机非连续 block 分配；
- append token；
- 释放请求；
- 最后一个不完整 block；
- 统计连续最大长度预留与分页方式的浪费；
- 可选：两个请求共享前缀 block。

### 当日闭卷

解释连续 KV 的预留浪费、内部/外部碎片和扩容搬迁，以及 block table 如何把逻辑连续和物理分散解耦。

## Day 2：最小 Paged Decode Attention CUDA 实验

### 限定范围

```text
Q:           [batch, heads, head_dim]
KV pool:     [physical_blocks, 2, heads, block_size, head_dim]
block table: [batch, max_logical_blocks]
seq_lens:    [batch]
Output:      [batch, heads, head_dim]
```

为降低学习门槛，可以先从 `batch=1, head=1` 写通，再扩展变长 batch；不做生产级 GQA 优化。

### 核心任务

- 一个 query 遍历自己的历史 token；
- 查 block table 得到 physical block；
- 计算块内 offset；
- gather K/V；
- 使用 online softmax；
- 与连续 KV CPU reference 对拍；
- 随机打乱 physical block，证明未依赖连续地址；
- 覆盖 `seq_len % block_size != 0`；
- memcheck 通过。

### 结束条件

正确性通过即结束，不做 ncu 调优，不做作品集 README，不与 vLLM 比性能。

### 两天后必须停止

即使还有兴趣，也只把以下内容记录到未来清单：GPU allocator、prefix cache 引用计数、copy-on-write、continuous batching、vLLM kernel、GQA/量化 KV。不得占用 Day 3。

## Day 3：限时 Kernel 手写

从以下抽两题：

- reduction；
- stable softmax；
- warp-per-row GEMV；
- shared tiled GEMM；
- online softmax combine。

每题流程：

```text
5 分钟讲接口与不变量
30～60 分钟写核心 kernel
15 分钟补边界与错误检查
15 分钟运行 reference/sanitizer
10 分钟复盘错误
```

## Day 4：未知 Kernel 性能诊断

由助手或历史代码提供一个未知慢 kernel，本人完成：

1. correctness；
2. FLOP/bytes 账本；
3. 正常 benchmark；
4. ptxas 资源；
5. ncu 高层到 stall；
6. PTX/SASS 关键路径；
7. 一个单变量修改；
8. 复测和反证。

输出一页诊断报告，不追求把 kernel 优化到极致。

## Day 5：项目表达

准备四个版本：

- 60 秒职业定位；
- 5 分钟 GEMM；
- 5 分钟 FlashAttention；
- 3 分钟 PagedAttention 原理；
- 10 分钟“CPU 性能工程如何迁移到 GPU”。

每个技术项目统一结构：

```text
问题
→ baseline
→ 假设
→ 修改
→ correctness
→ profiler/指令证据
→ 墙钟结果
→ 失败优化
→ 适用边界
```

## Day 6：模拟面试

至少两轮：

### 第一轮：AI Infra

- C++ Model Serving；
- 在线推理数据流；
- DAG 调度、异步执行、容量与稳定性；
- prefill/decode/KV cache；
- continuous batching 与 PagedAttention；
- 商业项目指标口径。

### 第二轮：CUDA 性能

- 手写 kernel；
- memory hierarchy；
- GEMM tiling；
- occupancy/ILP/TLP；
- ncu stall；
- PTX/SASS；
- Tensor Core；
- FlashAttention。

每轮记录：答错、答浅、答散、证据不足四类问题，只补高频缺口。

## Day 7：发布与投递准备

- 公开仓库只保留重新验证成果；
- 检查构建命令和干净环境；
- 检查公开数字与当前代码一致；
- 更新简历项目描述；
- 准备 AI Infra 版和 CUDA/推理性能版；
- 建立投递跟踪表；
- 开始第一批投递，不再以“还没学完”为理由延期。

---

## 7. 每周停机线与降级策略

### Week 1 延迟

优先保证 naive/shared/register correctness。参数 sweep 和 README 美化可以延后，不能带着索引错误进入第二周。

### Week 2 延迟

必须保住：编译链、naive PTX/SASS 数据流、资源报告、一次诊断。五版横向表和精细 Tensor Core mapping 可以降级。

### Week 3 延迟

必须保住：naive、online-softmax tiled、正确性和边界。并行 reduction 或 `cp.async` 若未完成，应如实标为下一里程碑，不能占用整个第四周。

### Week 4 延迟

PagedAttention 固定两天后停止。优先保住模拟面试、项目表达和投递，不继续优化学习实验。

### 砍任务顺序

从先砍到后砍：

1. 额外参数 sweep；
2. PagedAttention 可选 prefix sharing；
3. FlashAttention 并行 reduction 深化；
4. 3-stage pipeline；
5. 额外 SASS 指令细节；
6. README 美化。

绝不砍：correctness、边界、本人手写、真实计时、证据边界、模拟面试。

---

## 8. 每周评分表

每项 0～2 分：0=不会，1=看提示能做，2=闭卷能做。

| 能力              | Week 1 | Week 2 | Week 3 | Week 4 |
| ----------------- | -----: | -----: | -----: | -----: |
| 核心代码独立实现  |        |        |        |        |
| correctness/边界  |        |        |        |        |
| benchmark 可信度  |        |        |        |        |
| ncu 因果分析      |        |        |        |        |
| PTX/SASS 数据流   |        |        |        |        |
| 项目口述          |        |        |        |        |
| AI Infra 系统连接 |        |        |        |        |

判断规则：

- 12～14：本周通过；
- 9～11：进入下一周，但每天开场补考 30 分钟；
- 8 以下：不是继续读更多资料，而是缩小实现、重做最小验收。

---

## 9. 最终成果清单

四周结束时只检查以下成果，不按“看了多少页”评价：

### 代码与项目

- [ ] GEMM 重建版：naive/shared/register，进阶 vector/async 视实际验收；
- [ ] FlashAttention：naive/tiled/pipelined 或明确的首版边界；
- [ ] PagedAttention：两天学习实验，留在私人学习仓库；
- [ ] 至少两道核心 kernel 可以限时手写。

### 正确性与性能

- [ ] 每个公开 kernel 有 reference；
- [ ] 非整齐 shape 与 sanitizer；
- [ ] 同一 benchmark 口径；
- [ ] 至少一条“假设→修改→ncu/SASS→墙钟”的完整证据链；
- [ ] 至少一个失败优化案例。

### PTX/SASS

- [ ] 能独立生成 PTX、ptxas log、SASS；
- [ ] 能追踪 load→compute→store；
- [ ] 能解释 register/spill/local；
- [ ] 能说明 scheduler/scoreboard 诊断边界；
- [ ] 能说明 WMMA→PTX MMA→SASS HMMA 三层关系。

### 求职输出

- [ ] 60～90 秒自我介绍；
- [ ] GEMM 5/10 分钟项目版；
- [ ] FlashAttention 5 分钟项目版；
- [ ] PagedAttention 3 分钟原理版；
- [ ] 两轮模拟面试；
- [ ] 两个投递简历版本；
- [ ] 第一批正式投递。

---

## 10. 资料使用顺序

只在对应周按需使用：

### Week 1

- GEMM 现有实现与记录：[week04_gemm](../week04_gemm/)、[week05_gemm_advanced](../week05_gemm_advanced/)；
- GEMM 作品重建设计：[2026-07-10-gemm-portfolio-rebuild-design.md](../docs/superpowers/specs/2026-07-10-gemm-portfolio-rebuild-design.md)。

### Week 2

- 主教材：[CUDA 深水区：PTX、SASS、MMA、异步流水与 Hopper](../docs/topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)；
- 现有 14 天计划只作为验收题来源：[CUDA 工程师面试：14 天突击计划](../docs/interview/CUDA工程师面试_14天突击计划.md)；
- WMMA：[week06_tensorcore](../week06_tensorcore/)；
- ncu：[Nsight Compute ncu 详解](../docs/topics/performance/Nsight_Compute_ncu详解.md)。

### Week 3

- Attention 教材：[Week 4：Attention 与 FlashAttention 完整学习资料](../docs/courses/attention/Week4_Attention与FlashAttention完整学习资料.md)；
- Online Softmax 证明：[Online Softmax 正确性证明](../docs/proofs/Online_Softmax正确性证明.md)；
- 旧代码只用于完成后对照：[week04_attention](../week04_attention/)。

### Week 4

- PagedAttention：[PagedAttention 详解](../docs/topics/kv_cache/PagedAttention详解.md)；
- KV cache：[KV cache accounting](../docs/topics/kv_cache/kv_cache_accounting.md)；
- decode：[decode step dataflow](../docs/topics/kv_cache/decode_step_dataflow.md)；
- 面试：[CUDA 面试核心题库](../docs/interview/CUDA面试核心题库.md) 与私有职业叙事材料（私有，不提供仓库链接）。

资料纪律：先写、再查、最后对照。禁止直接复制旧 kernel 作为“重建”。

---

## 11. 最后一句执行口令

> 四周内不追求知识面更宽，只追求三件事：核心代码能独立写，性能结论有证据，项目故事经得起追问。