# Week 4 Attention 与 FlashAttention A100 教材 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 编写一份适配本仓库真实学习历史、以 A100 为实验平台、可从零学懂 Attention 并逐步实现 online softmax 和教学版 tiled Attention 的中文自包含教材。

**Architecture:** 使用一个 Markdown 长文档承载预备章和 Day 1–Day 7；概念按依赖顺序递进，每章统一包含直觉、数学、shape、手算、CUDA 映射、练习、验证与口述。可运行代码以内嵌 fenced code block 提供，通过脚本提取到临时目录后使用 `g++`/`nvcc -arch=sm_80` 做语法和运行验证；必须手写内容与参考实现分隔，避免学习者直接跳过练习。

**Tech Stack:** Markdown、C++17、CUDA C++17、CUDA Runtime、A100 SM 8.0、WMMA/FP16/BF16 概念、Nsight Compute、compute-sanitizer、Git。

---

## 文件结构

**创建：**

- `docs/Week4_Attention与FlashAttention完整学习资料.md`：唯一教材正文，包含零基础预备章、Day 1–Day 7、内嵌演示、手写骨架、提示、参考实现、自测答案和命令。

**引用但不修改：**

- `docs/DeepSeek_CUDA_2月冲刺计划.md`：Week 4 原始目标。
- `operator_practice/softmax/softmax.cu`：学习者已有三遍 stable softmax。
- `common/common.cuh`：已有 block reduce helper。
- `docs/Week3_TensorCore学习文档.md`：已有 WMMA、混合精度基础。
- `week06_tensorcore/tensor_core_profile.md`：A100 HMMA 与 Tensor pipe 实测。
- `docs/异步拷贝_pipeline_cooperative_groups学习文档.md`：已有 `cp.async`/pipeline 基础。
- `week05_gemm_advanced/gemm_optimization_ladder.md`：A100 GEMM、Roofline、ncu 经验。

**不预创建：**

- `week04_attention/*.cu`：这些是学习者按正文创建的作业文件；教材提供明确骨架和验收，不提前替学习者落盘答案。

---

### Task 1：建立教材骨架、使用说明与已有知识映射

**Files:**

- Create: `docs/Week4_Attention与FlashAttention完整学习资料.md`

- [ ] **Step 1：写标题、定位和学习成果**

正文开头明确：主要平台为 A100 SM 8.0；读者已经会 CUDA/GEMM/reduction/stable softmax/WMMA，但不假设懂 Attention；教学版实现优先正确与可解释，不宣称复刻工业 FlashAttention。

- [ ] **Step 2：写“怎么使用”规则**

加入五类标记：`【演示】`、`【必须手写】`、`【提示 1/2/3】`、`【参考实现】`、`【挑战】`。规定先独立尝试 30–60 分钟、PASS 后再看答案、次日闭卷重写核心循环。

- [ ] **Step 3：写已有知识映射表**

逐项连接 GEMM→`QK^T/PV`、reduction→softmax、分层归约→online state、shared tiling→Q/K/V tile、register accumulator→`O`、WMMA→Tensor Core、`cp.async`→K/V 预取、ncu→IO 证据。

- [ ] **Step 4：写七天路线和每日完成标准总表**

总表每行必须包含主题、必须手写产出、correctness 标准和口述问题，保证 Day 1–Day 7 与设计说明一致。

- [ ] **Step 5：检查骨架完整性**

Run:

```bash
rg -n '^#|^##|^###' docs/Week4_Attention与FlashAttention完整学习资料.md
```

Expected: 输出预备章、Day 1–Day 7、附录与答案区标题，顺序无跳跃。

- [ ] **Step 6：提交骨架**

```bash
git add docs/Week4_Attention与FlashAttention完整学习资料.md
git commit -m "docs: scaffold A100 attention week guide"
```

### Task 2：编写零基础预备章——token、embedding、shape 与内存布局

**Files:**

- Modify: `docs/Week4_Attention与FlashAttention完整学习资料.md`

- [ ] **Step 1：用三 token 例子解释输入**

固定贯穿示例：`N=3, D=2`，输入矩阵为：

```text
X = [[1,0],
     [0,1],
     [1,1]]
```

解释 token、token id、embedding、sequence length、hidden dimension，明确数字只为手算，不代表真实模型语义。

- [ ] **Step 2：解释 shape 层次**

从 `[N,D]` 推到 `[B,N,D]`、`[B,H,N,Dh]`，使用 `D=H×Dh`。每个符号首次出现时写中文含义、合法范围和一个具体数值。

- [ ] **Step 3：给出行主序压平公式**

至少给出：

```cpp
idx2(n, d)       = n * D + d;
idx4(b, h, n, d) = ((b * H + h) * N + n) * Dh + d;
```

用 `B=2,H=4,N=32,Dh=16` 手算一个非零索引，说明最右维连续。

- [ ] **Step 4：加入 shape 自测与答案**

题目覆盖：矩阵转置后的 shape、`D/H`、某个四维坐标的偏移、softmax 将来应沿哪个候选维度。答案置于本节末尾，并解释推理过程。

- [ ] **Step 5：人工校验术语首次出现**

Run:

```bash
rg -n 'token|embedding|sequence|hidden|head dimension|\[B,H,N,Dh\]' docs/Week4_Attention与FlashAttention完整学习资料.md
```

Expected: 每个英文词在首次出现处有中文解释，且包含 `[N,D]` 到 `[B,H,N,Dh]` 的递进。

- [ ] **Step 6：提交预备章**

```bash
git add docs/Week4_Attention与FlashAttention完整学习资料.md
git commit -m "docs: teach attention tensor prerequisites"
```

### Task 3：编写 Day 1——标准 Attention、CPU reference 与朴素 CUDA 三阶段

**Files:**

- Modify: `docs/Week4_Attention与FlashAttention完整学习资料.md`

- [ ] **Step 1：推导 Q、K、V 与 scaled dot-product attention**

正文必须写出并逐项解释：

```text
Q=XWq, K=XWk, V=XWv
S=QK^T/sqrt(Dh)
P=softmax_rows(S)
O=PV
```

解释 `Q[i]·K[j]` 是 query `i` 对 key `j` 的匹配分数；除以 `sqrt(Dh)` 用于控制点积方差，避免维度增长使 softmax 过饱和；softmax 对每个 query 的 key 轴执行。

- [ ] **Step 2：完整手算三 token 例子**

为降低无关复杂度，手算时先取 `Q=K=V=X`，计算 `S`、缩放后的 logits、每行概率和 `O`；随后解释真实模型通过三个投影矩阵获得 Q/K/V。

- [ ] **Step 3：写完整 CPU reference 代码块**

代码使用 `std::vector<float>`，函数固定为：

```cpp
void attention_cpu(const float* q, const float* k, const float* v,
                   float* out, int n, int d, bool causal);
```

函数内部逐行计算 score、应用 causal mask、减行最大值、求指数和、累计 `P·V`。代码包括 `N=3,D=2` 的 main、NaN 检查和输出打印。

- [ ] **Step 4：提取并运行 CPU reference**

从教材中给 CPU 程序的 fence 添加唯一标签注释 `// FILE: attention_cpu.cpp`，用 `awk` 提取到 `/tmp/week4_attention/attention_cpu.cpp`。

Run:

```bash
mkdir -p /tmp/week4_attention
awk '/^```cpp attention_cpu.cpp$/{p=1;next} /^```$/{if(p){exit}} p' \
  docs/Week4_Attention与FlashAttention完整学习资料.md \
  > /tmp/week4_attention/attention_cpu.cpp
g++ -O2 -std=c++17 /tmp/week4_attention/attention_cpu.cpp \
  -o /tmp/week4_attention/attention_cpu
/tmp/week4_attention/attention_cpu
```

Expected: 编译退出码 0，程序打印 `PASS`，每行概率和在 `1±1e-5`。

- [ ] **Step 5：写朴素 CUDA 必须手写骨架**

固定三个接口：

```cpp
__global__ void qk_scores(const float* q, const float* k, float* scores,
                          int n, int d, bool causal);
__global__ void row_softmax(float* scores, int n);
__global__ void pv_output(const float* probs, const float* v, float* out,
                          int n, int d);
```

host 端、分配、初始化、CPU 对照和误差函数完整提供；三个 kernel 只留核心循环空位，并给三层提示。参考实现放在练习验收之后。

- [ ] **Step 6：写 FLOP 和内存账本**

分别计算 `QK^T` 与 `PV` 约各 `2N²D` FLOP，完整 `S/P` 各 `N²` 元素。至少列出 FP16 下 `N=2048,8192,32768` 的单 head 中间矩阵字节数，并明确多 batch/head 会线性放大。

- [ ] **Step 7：提交 Day 1**

```bash
git add docs/Week4_Attention与FlashAttention完整学习资料.md
git commit -m "docs: add standard attention lesson and exercises"
```

### Task 4：编写 Day 2——从三遍 stable softmax 推导 online softmax

**Files:**

- Modify: `docs/Week4_Attention与FlashAttention完整学习资料.md`

- [ ] **Step 1：引用已有实现并复盘三遍算法**

链接 `operator_practice/softmax/softmax.cu`，只复盘 max、`sum(exp(x-max))`、归一化三遍，不复制整份旧代码。

- [ ] **Step 2：用具体数字推导元素级 online softmax**

使用输入 `[2,1,4,3]`，按顺序记录每一步 `(m,l)`，展示遇到 `4` 时旧 `l` 乘 `exp(2-4)`。最后与稳定 softmax 分母对照。

- [ ] **Step 3：推导 tile 级合并公式**

定义块统计量 `(m_a,l_a)`、`(m_b,l_b)`，给出：

```text
m = max(m_a,m_b)
l = exp(m_a-m)l_a + exp(m_b-m)l_b
```

解释结合性来自两组指数和统一到共同最大值，不宣称浮点计算在任意合并顺序下逐 bit 相同。

- [ ] **Step 4：写 CPU streaming 演示和 CUDA 手写骨架**

CPU 演示完整提供并打印与三遍参考的最大误差。CUDA 骨架复用已有 block max/sum 思维，要求学习者补 `(m,l)` 更新；明确单 kernel 中不同线程局部状态必须再做成对合并，不能把线程 0 的顺序公式机械复制到所有线程。

- [ ] **Step 5：写边界测试**

固定测试：`[2,1,4,3]`、全负数、`500..1499` 大值、长度 `1000`、长度 `1031`。验收为无 NaN/Inf、概率和误差 `<1e-4`、与 CPU reference 的最大绝对误差在 FP32 容差内。

- [ ] **Step 6：提交 Day 2**

```bash
git add docs/Week4_Attention与FlashAttention完整学习资料.md
git commit -m "docs: derive online softmax from stable softmax"
```

### Task 5：编写 Day 3——FlashAttention IO 数据流与 m/l/O 递推

**Files:**

- Modify: `docs/Week4_Attention与FlashAttention完整学习资料.md`

- [ ] **Step 1：画标准与分块数据流**

使用 Mermaid 或紧凑 ASCII 图，清楚标出 HBM、shared/register、`S/P` 是否写回。图旁写明 forward 的主导矩阵乘 FLOP 仍约 `4N²D`。

- [ ] **Step 2：建立 Q 外块、K/V 内块循环**

定义 `Br`、`Bc`、`Q_i`、`K_j`、`V_j`、局部分数 `S_ij`，说明每个 Q tile 遍历全部合法 K/V tile。

- [ ] **Step 3：推导完整状态更新**

给出维度明确的公式：

```text
m_new = max(m_old, rowmax(S_ij))
alpha = exp(m_old-m_new)
P_ij = exp(S_ij-m_new)
l_new = alpha*l_old + rowsum(P_ij)
O_acc_new = alpha*O_acc_old + P_ij*V_j
O_i = O_acc_final / l_final
```

强调这里保存的是未归一化输出累加器；若使用另一种已归一化 `O` 表示，递推公式会不同，全文只采用一种表示。

- [ ] **Step 4：用两块分数手算旧输出重缩放**

选择每块两个 score、`D=2` 的 V，用数字展示新块最大值变大后若不乘 `alpha`，旧块贡献会被错误放大。

- [ ] **Step 5：解释 causal tile**

区分完全位于未来的 tile（整块跳过）、完全合法 tile（正常计算）、跨对角线 tile（元素级 mask 为 `-∞`）。说明不能对全 mask 行执行无保护的 `-∞ - -∞`。

- [ ] **Step 6：写 forward 伪代码和误区表**

误区至少包含：减少主导 FLOP、近似 Attention、只做 kernel fusion、完全不需要同步、online softmax 只重缩放分母。

- [ ] **Step 7：提交 Day 3**

```bash
git add docs/Week4_Attention与FlashAttention完整学习资料.md
git commit -m "docs: explain flash attention IO and recurrence"
```

### Task 6：编写 Day 4——教学版 tiled Attention 手写路径

**Files:**

- Modify: `docs/Week4_Attention与FlashAttention完整学习资料.md`

- [ ] **Step 1：锁定教学实现边界**

第一版为 FP32、单 batch、单 head、无 causal；支持任意 `N,D`。明确它不使用 Tensor Core，目的为把 `m/l/O` 数据流写对。

- [ ] **Step 2：给出线程与内存映射**

逐项说明 block 负责哪个 Q tile、线程如何遍历 row/feature、K/V tile 怎样装入 shared、score 和输出累加器放在哪里。若为清晰而限制 `D≤128`，必须在接口检查和章节开头同时写明，并提供超限错误信息。

- [ ] **Step 3：提供分阶段必须手写骨架**

骨架中的七个空位固定为：加载 K/V、局部 score、局部 max、更新 `m/l`、重缩放旧输出、累加 `P·V`、最终归一化。每个空位前给 shape、索引和公式，后给单独的局部验收输出。

- [ ] **Step 4：提供三层提示和独立参考实现**

提示 1 只描述数据依赖；提示 2 给伪代码；提示 3 给关键索引。参考实现放在本日所有验收项之后，并标注不要直接复制为作业。

- [ ] **Step 5：写 host 测试框架**

固定测试：`N=3,D=2`、`N=8,D=8`、`N=37,D=24`、`N=128,D=64`。比较 CPU、naive CUDA、tiled CUDA 的最大绝对/相对误差并检查 NaN/Inf。

- [ ] **Step 6：提取参考实现并做 A100 编译检查**

为完整参考程序使用 fence 名 `cpp tiled_attention_reference.cu`。

Run:

```bash
awk '/^```cpp tiled_attention_reference.cu$/{p=1;next} /^```$/{if(p){exit}} p' \
  docs/Week4_Attention与FlashAttention完整学习资料.md \
  > /tmp/week4_attention/tiled_attention_reference.cu
nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo \
  /tmp/week4_attention/tiled_attention_reference.cu \
  -o /tmp/week4_attention/tiled_attention_reference
/tmp/week4_attention/tiled_attention_reference
```

Expected: 编译退出码 0，四组 shape 均打印 `PASS`。若当前机器没有 GPU，只运行 `nvcc` 编译并在教材交付中明确运行验证尚待 A100。

- [ ] **Step 7：运行内存检查**

Run:

```bash
compute-sanitizer --tool memcheck /tmp/week4_attention/tiled_attention_reference
```

Expected: `ERROR SUMMARY: 0 errors`。

- [ ] **Step 8：提交 Day 4**

```bash
git add docs/Week4_Attention与FlashAttention完整学习资料.md
git commit -m "docs: add tiled attention guided implementation"
```

### Task 7：编写 Day 5——A100 优化桥接

**Files:**

- Modify: `docs/Week4_Attention与FlashAttention完整学习资料.md`

- [ ] **Step 1：连接已有 WMMA 经验**

引用 `docs/Week3_TensorCore学习文档.md` 与 `week06_tensorcore/tensor_core_profile.md`，解释 `QK^T` 和 `PV` 可以使用 Tensor Core，但 softmax、mask、归约和状态更新仍需 CUDA Core/特殊函数/同步协作。

- [ ] **Step 2：写精度职责表**

建议教学路径：FP16/BF16 存 Q/K/V，FP32 计算 row max、指数和、`m/l` 与输出累加。解释输入转换已损失的信息不能由 FP32 累加恢复。

- [ ] **Step 3：连接 `cp.async` 与双缓冲**

引用已有异步拷贝文档，画时间线：计算当前 K/V tile 时预取下一 tile。明确 `cp.async` 只解决 global→shared 数据搬运，不会自动优化 softmax 或寄存器压力。

- [ ] **Step 4：写 tile 资源账本**

给出 shared memory 近似式：

```text
bytes ≈ sizeof(T) × (Br×D + Bc×D + optional Br×Bc)
```

比较是否显式保存 score tile 的资源差异，并说明寄存器中的 per-row `m/l/O` 也可能成为限制。

- [ ] **Step 5：写 A100/Hopper 边界表**

A100 可讲 WMMA/MMA、FP16/BF16/TF32 Tensor Core、`cp.async`；TMA/WGMMA 属于 Hopper，不作为代码路径。A100 40GB/80GB 带宽从实际 `nvidia-smi` 和设备资料确认，不在教材中硬套单一数值。

- [ ] **Step 6：提交 Day 5**

```bash
git add docs/Week4_Attention与FlashAttention完整学习资料.md
git commit -m "docs: connect attention to A100 optimization"
```

### Task 8：编写 Day 6——KV cache、MHA/MQA/GQA、MLA 与 FlashMLA

**Files:**

- Modify: `docs/Week4_Attention与FlashAttention完整学习资料.md`

- [ ] **Step 1：从 prefill 与 decode 建立问题**

用生成第 `t` 个 token 的过程说明历史 K/V 可以复用，而新 query 每步变化。明确 prefill 通常并行处理长序列，decode 常为 query length 1 且更受访存和 KV cache 影响。

- [ ] **Step 2：推导 KV cache 字节公式**

写出普通 MHA：

```text
bytes = 2(K和V) × B × N × H_kv × Dh × bytes_per_element
```

用至少一个具体模型 shape 手算，说明层数还会再次线性放大。

- [ ] **Step 3：对照 MHA、MQA、GQA**

表格必须比较 query head 数、KV head 数、cache 大小、表达能力与实现复杂度；用同一组 `Hq=32,Dh=128` 数字比较。

- [ ] **Step 4：讲 MLA 的 latent cache**

先给概念数据流，再给 shape。明确 MLA 的具体投影、解耦 RoPE 和矩阵吸收会随模型设计变化；教材聚焦 DeepSeek 相关直觉，但避免把低维 latent 简化成“完全不用存 K/V”。

- [ ] **Step 5：区分 MLA 与 FlashMLA**

MLA 是注意力架构/表示方式；FlashMLA 是面向该数据流的高性能 kernel/工程实现。列出 README 阅读问题：输入布局、支持 dtype、prefill/decode 场景、输出、硬件要求、benchmark 指标。

- [ ] **Step 6：加入 shape 与 cache 自测**

题目要求学习者计算 MHA/GQA/MLA 示例 cache、判断某 shape 属于 prefill 还是 decode、解释 FlashAttention 与 FlashMLA 的职责差异。提供标准答案。

- [ ] **Step 7：提交 Day 6**

```bash
git add docs/Week4_Attention与FlashAttention完整学习资料.md
git commit -m "docs: teach KV cache GQA MLA and FlashMLA"
```

### Task 9：编写 Day 7——benchmark、ncu、复盘与面试口述

**Files:**

- Modify: `docs/Week4_Attention与FlashAttention完整学习资料.md`

- [ ] **Step 1：写可靠 benchmark 模板**

规定 warmup 10 次、正式迭代至少 100 次、CUDA Event 只包 kernel 序列、同步 stop event、报告 GPU/编译参数/shape/dtype/causal。分开报告 kernel-only 与含分配/传输的端到端时间。

- [ ] **Step 2：定义对比表**

表格列固定为：版本、`B/H/N/Dh`、dtype、causal、time、logical bytes、是否 materialize `S/P`、最大误差、寄存器、occupancy、DRAM/L2/shared/Tensor pipe。

- [ ] **Step 3：写 ncu 命令与判读顺序**

提供：

```bash
ncu --set basic --kernel-name regex:attention ./attention_bench
ncu --set full -s 1 -c 1 -o attention_tiled ./attention_bench
```

判读顺序为：先确认 kernel/shape → Speed of Light → DRAM/L2/shared → occupancy/register → stall → Tensor pipe。再次提醒 ncu 重放下程序内时间不是真实 benchmark。

- [ ] **Step 4：解释 prefill/decode 和小 shape**

说明小 `N` 可能受 launch/并行度限制；prefill 具有大矩阵计算，decode query 很短且常被 KV cache 读取约束。不得把一个 shape 的结论推广到所有阶段。

- [ ] **Step 5：写最终自测和口述**

至少包含：15 道概念题、5 道 shape/字节计算题、3 道代码诊断题。提供 3 分钟 Attention、5 分钟 FlashAttention、2 分钟 MLA 口述模板。

- [ ] **Step 6：提交 Day 7**

```bash
git add docs/Week4_Attention与FlashAttention完整学习资料.md
git commit -m "docs: add attention profiling and review"
```

### Task 10：全书一致性、代码提取和交付验证

**Files:**

- Modify: `docs/Week4_Attention与FlashAttention完整学习资料.md`

- [ ] **Step 1：检查设计覆盖率**

逐项对照 `docs/superpowers/specs/2026-07-01-week4-attention-flashattention-a100-design.md` 第 9 节十条完成标准，在教材末尾内部验收表中逐项标记证据章节；交付前删除内部验收表，避免干扰学习正文。

- [ ] **Step 2：扫描未定义符号和占位内容**

Run:

```bash
rg -n 'T''BD|FIX''ME|待''补|以后''填写|未''完成' \
  docs/Week4_Attention与FlashAttention完整学习资料.md
```

Expected: 无输出。练习骨架使用明确的 `【必须手写】` 空位说明，不使用模糊占位词。

- [ ] **Step 3：检查公式符号一致性**

全文统一：sequence length=`N`、head dimension=`Dh`、query tile=`Br`、key tile=`Bc`、running max=`m`、running denominator=`l`、未归一化输出累加器=`O_acc`。运行：

```bash
rg -n 'D_h|d_head|head_dim|O_acc|m_new|l_new|Br|Bc' \
  docs/Week4_Attention与FlashAttention完整学习资料.md
```

Expected: 别名只在术语对照处出现，公式和代码使用统一命名。

- [ ] **Step 4：编译所有完整代码块**

重复 Task 3 与 Task 6 的提取命令，并对其余带文件名 fence 的完整程序逐一提取。Run:

```bash
g++ -O2 -std=c++17 /tmp/week4_attention/attention_cpu.cpp \
  -o /tmp/week4_attention/attention_cpu
nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo \
  /tmp/week4_attention/tiled_attention_reference.cu \
  -o /tmp/week4_attention/tiled_attention_reference
```

Expected: 全部编译退出码 0。

- [ ] **Step 5：运行正确性与 sanitizer**

Run:

```bash
/tmp/week4_attention/attention_cpu
/tmp/week4_attention/tiled_attention_reference
compute-sanitizer --tool memcheck /tmp/week4_attention/tiled_attention_reference
```

Expected: 所有 shape 打印 `PASS`，sanitizer 为 0 errors。无 A100 时记录只完成编译，不能声称 GPU 运行或性能通过。

- [ ] **Step 6：检查 Markdown 结构和链接**

Run:

```bash
rg -n '^#{1,4} ' docs/Week4_Attention与FlashAttention完整学习资料.md
rg -n '\]\([^)]*\)' docs/Week4_Attention与FlashAttention完整学习资料.md
```

Expected: 标题层级无倒跳；本地相对链接指向仓库中存在的文件；外部链接使用权威来源。

- [ ] **Step 7：检查最终 diff 范围**

Run:

```bash
git diff --check
git status --short
git log --oneline -10
```

Expected: `git diff --check` 无错误；只包含教材计划内变更，不纳入用户已有未跟踪文件。

- [ ] **Step 8：提交最终修订**

```bash
git add docs/Week4_Attention与FlashAttention完整学习资料.md
git commit -m "docs: finish A100 attention and FlashAttention guide"
```

---

## 执行检查点

执行时在以下节点回看学习体验，而不是只看篇幅：

1. Task 3 后：读者是否能在不知道 Transformer 其他结构的情况下解释标准 Attention；
2. Task 5 后：读者是否能用数字解释为什么 `l` 和 `O_acc` 都要重缩放；
3. Task 6 后：参考 CUDA 程序是否能在 A100 上编译、运行并通过非整除 shape；
4. Task 8 后：MLA 是否建立在 KV cache 与 MHA/MQA/GQA 之后，而非堆术语；
5. Task 10 后：所有成功声明是否有实际命令输出支持。
