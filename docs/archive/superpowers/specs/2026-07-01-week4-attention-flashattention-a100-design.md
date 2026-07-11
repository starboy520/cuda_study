# Week 4 Attention 与 FlashAttention 自包含教材设计

## 1. 目标

为 `docs/DeepSeek_CUDA_2月冲刺计划.md` 中的 Week 4 编写一份可直接顺序学习的中文长教材，目标读者是：已经完成本仓库 CUDA 基础、GEMM、归约、stable softmax、Tensor Core 与基础性能分析，但不了解 Attention 和大模型推理概念的学习者。

教材主文件为：

```text
docs/Week4_Attention与FlashAttention完整学习资料.md
```

教材按 Day 1 到 Day 7 组织，在 Day 1 前增加零基础预备章。代码以 CUDA C++ 为主，CPU reference 用于解释数学和验证正确性。主要实验平台为 NVIDIA A100（Ampere，SM 8.0）。

教材必须让学习者最终能够：

1. 从 token、embedding 开始解释 Q、K、V 和 scaled dot-product attention；
2. 手算并实现标准 Attention；
3. 从 stable softmax 推导并实现 online softmax；
4. 解释并实现教学版 tiled Attention 的核心递推；
5. 解释 FlashAttention 为什么是 IO-aware，以及它没有改变哪些数学结果；
6. 说明 A100 Tensor Core、FP16/BF16、shared memory 和 `cp.async` 在 Attention 中的位置；
7. 区分 prefill、decode、KV cache、MHA、MQA、GQA、MLA 和 FlashMLA；
8. 使用 benchmark 与 Nsight Compute 判断实现的瓶颈。

## 2. 已有基础与教材衔接

### 2.1 已经掌握、直接复用

仓库历史表明学习者已经完成或实际使用过：

- CUDA kernel launch、二维索引、边界判断和行主序压平；
- CPU reference、误差校验和 CUDA Event 计时；
- shared memory tiling、padding 和 bank conflict 分析；
- warp shuffle、block reduction、max/sum reduction；
- 三遍数值稳定 softmax；
- GEMM 外积、register tiling、向量化加载；
- Roofline、occupancy 和 Nsight Compute；
- A100 上的 FP16 WMMA GEMM、HMMA 指令检查和 Tensor pipe 分析；
- Ampere `cp.async`、pipeline 和 double buffering 的基本概念。

教材不重复完整教授这些内容，而是建立明确映射：

| 已学内容 | Week 4 中的用途 |
|---|---|
| GEMM 与矩阵 shape | `QK^T` 和 `PV` |
| max/sum reduction | row-wise softmax |
| 分层归约 | online softmax 状态合并 |
| shared tiling | Q/K/V 分块 |
| register accumulator | 在线保存输出 `O` |
| WMMA/Tensor Core | FP16/BF16 的 `QK^T` 与 `PV` |
| `cp.async`/pipeline | 预取下一块 K/V |
| ncu/Roofline | 验证 HBM IO、occupancy 和 Tensor pipe |

### 2.2 必须从零补齐

教材不得假设学习者知道以下概念：

- token、词表、embedding、sequence、hidden dimension；
- batch、head、head dimension；
- Q、K、V 的来源与作用；
- scaled dot-product attention；
- causal mask；
- multi-head attention；
- prefill、decode、KV cache；
- MQA、GQA、MLA、FlashMLA。

所有首次出现的术语必须就地解释，并同时给出直觉、数学定义、shape 和内存布局。

## 3. 教学方法

采用“依赖递进式内容 + Day 1 到 Day 7 打卡结构”。概念顺序为：

```text
token / embedding / shape
→ Q / K / V
→ 标准 Attention
→ stable softmax
→ online softmax
→ 分块统计量合并
→ tiled Attention
→ FlashAttention IO 数据流
→ A100 优化路径
→ KV cache / MQA / GQA / MLA / FlashMLA
→ ncu 分析与复盘
```

每个核心章节统一使用以下教学模板：

1. 为什么需要它；
2. 生活化直觉；
3. 严格数学定义；
4. shape 逐步变化；
5. 小数字手算；
6. 行主序索引与 CUDA 映射；
7. 演示或手写代码；
8. 正确性测试；
9. FLOP/bytes 或性能账本；
10. 常见错误；
11. 自测题与答案；
12. 面试口述。

教材使用一个 `N=3, D=2` 的微型单头例子贯穿前半部分，避免每章更换数据导致额外理解负担。

## 4. 章节设计

### 4.1 预备章：从一句话到张量

内容包括：

- token、token id、embedding；
- sequence length 与 hidden dimension；
- `[N,D]`、`[B,N,D]`、`[B,H,N,Dh]`；
- `D = H × Dh`；
- 多维 shape 的行主序压平；
- “reshape 改解释，不一定搬数据”的边界说明；
- 从一个三 token 句子构造数值 embedding。

### 4.2 Day 1：标准 Attention

递进推导：

```text
X
→ Q=XWq, K=XWk, V=XWv
→ S=QK^T/sqrt(Dh)
→ P=softmax(S)
→ O=PV
```

必须覆盖：

- Q、K、V 的直觉与严格含义；
- 为什么使用 `K^T`；
- 为什么除以 `sqrt(Dh)`；
- softmax 沿 key 维度执行；
- causal mask 的语义；
- 单头到多头的 shape；
- 完整 CPU reference；
- 三阶段 naive CUDA：`QK^T`、row softmax、`PV`；
- 标准 Attention 的 FLOP 和中间矩阵内存账本。

### 4.3 Day 2：online softmax

从现有三遍 stable softmax 出发，推导状态 `(m,l)`：

```text
m_new = max(m_old, m_block)
l_new = exp(m_old-m_new) * l_old
      + exp(m_block-m_new) * l_block
```

必须覆盖：

- 元素级递推和 tile 级合并的区别；
- 新最大值出现时旧分母为何重缩放；
- 具体数字逐项演算；
- 合并公式的等价性说明；
- CPU streaming reference；
- CUDA 单行 online softmax；
- 极大值、全负数、非整除长度测试；
- 为 Attention 增加输出状态 `O` 的动机。

### 4.4 Day 3：FlashAttention 思想

对比标准数据流和 tiled/Flash 数据流：

```text
标准：QK^T → S 写 HBM → softmax 读写 S/P → PV 再读 P
分块：Q/K/V tile → 片上算 score/softmax/PV → 仅写最终 O
```

必须覆盖：

- 数学结果不变；
- 主导 FLOP 不因 forward 分块而消失；
- 核心收益是避免完整 `S/P` 的 HBM materialization；
- Q 外块、K/V 内块的数据流；
- 状态 `(m_i,l_i,O_i)`；
- 新 max 出现时旧分母和旧输出都要重缩放；
- causal tile 的整块跳过与局部 mask；
- forward 伪代码；
- “FlashAttention 只是 kernel fusion”等常见误解纠正。

### 4.5 Day 4：教学版 tiled Attention

提供三层实现：

1. `attention_cpu`：权威正确性参考；
2. `attention_naive_cuda`：三个独立 kernel；
3. `attention_tiled_cuda`：不存完整 Attention matrix。

首版范围：

- FP32 输入与累加；
- 单 batch、单 head；
- 支持任意 `N`、`D` 的边界处理；
- 一个 block 负责一个 query tile 或教学上等价的清晰映射；
- shared memory 保存 K/V 或必要 tile；
- 在线维护 `m/l/O`；
- 不声称达到工业级性能。

渐进扩展：batch、多头、causal、FP16 输入与 FP32 累加。

### 4.6 Day 5：A100 优化路径

必须绑定已有 A100 知识：

- `-arch=sm_80`；
- FP16/BF16 输入与 FP32 softmax/累加；
- `QK^T`、`PV` 与 Tensor Core 的关系；
- WMMA 教学 API 与工业 MMA warp tiling 的差异；
- `cp.async` 预取下一块 K/V；
- double buffering 时间线；
- shared memory 容量如何限制 tile；
- register pressure、occupancy、tile size、并行度的权衡；
- `expf`、归约、同步如何影响纯 GEMM 流水；
- A100 40GB 与 80GB 的带宽配置不可混用，实验时读取实际设备信息。

不使用 Hopper 专属 TMA/WGMMA，不把它们混入 A100 实现要求。

### 4.7 Day 6：KV cache、MQA/GQA、MLA 与 FlashMLA

顺序为：

```text
prefill / decode
→ KV cache
→ MHA
→ MQA
→ GQA
→ MLA
→ FlashMLA
```

必须覆盖：

- decode 为何不能每步重算全部历史 K/V；
- KV cache shape 与显存公式；
- MHA/MQA/GQA 的 KV head 数量差异；
- MLA 的低维 latent cache 思路；
- 哪些量被缓存，哪些量在使用时投影或吸收；
- MLA 节省什么、不节省什么；
- FlashMLA 是面向 MLA 数据布局和解码/推理场景的高性能实现，不等同于 MLA 算法本身；
- README 级工程结构阅读，不要求完整复刻生产 kernel。

### 4.8 Day 7：性能分析与复盘

必须覆盖：

- warmup、CUDA Event、重复运行和同步边界；
- naive 与 tiled 的中间内存比较；
- FLOP 与 logical bytes；
- DRAM/L2/shared throughput；
- Tensor pipe；
- achieved occupancy、registers/thread；
- stall reason；
- 小 `N` 时 launch/并行度不足；
- prefill 与 decode 的不同瓶颈；
- 正确性、性能和 profiler 证据分开记录；
- 3 分钟 Attention、5 分钟 FlashAttention、MLA 口述模板。

## 5. 代码练习设计

### 5.1 标记体系

教材中的代码统一标记：

- `【演示】`：完整提供，可直接编译运行；
- `【必须手写】`：只给接口、测试框架和 TODO；
- `【提示 1/2/3】`：从方向到伪代码逐级解锁；
- `【参考实现】`：放在练习之后，要求先尝试；
- `【挑战】`：只给需求和验收标准。

### 5.2 完整提供

- 微型 CPU 手算示例；
- CPU Attention reference；
- CUDA 错误检查、内存分配、数据初始化；
- shape/索引/FLOP/bytes 工具；
- benchmark 框架；
- ncu 命令；
- MHA/GQA/MLA 的小型 CPU 数据流演示。

### 5.3 必须手写

作业一，标准 Attention：

```cpp
__global__ void qk_scores(...);
__global__ void row_softmax(...);
__global__ void pv_output(...);
```

作业二，online softmax：

- CPU streaming 版；
- CUDA 单行版；
- 与三遍 stable softmax 对照。

作业三，tiled Attention 主循环：

```text
for each Q tile:
    initialize m, l, O
    for each K/V tile:
        load K/V tile
        compute local scores
        reduce local max
        update m/l
        rescale old O
        accumulate P*V
    normalize and store O
```

### 5.4 挑战项

至少选择两项：

- causal mask；
- multi-head；
- FP16 输入、FP32 累加；
- 非整除 `N/D`；
- K/V 双缓冲；
- tile 参数扫描；
- ncu 对比分析。

### 5.5 参考答案使用规则

1. 先阅读概念、推导和骨架；
2. 独立尝试 30–60 分钟；
3. 保留并分析编译错误或错误输出；
4. 按需逐级查看提示；
5. correctness PASS 后再看参考实现；
6. 第二天关闭答案重写核心循环。

## 6. 正确性与测试设计

### 6.1 最小测试矩阵

至少包含：

| 类型 | 示例 | 目的 |
|---|---|---|
| 手算 | `N=3,D=2` | 检查数学与 shape |
| 最小 CUDA | `N=8,D=8` | 方便打印中间值 |
| 非整除 | `N=37,D=24` | 检查边界 |
| 常规 | `N=128/512,D=64` | benchmark |
| 大值 logits | 人工构造 | 检查数值稳定 |
| causal | `N=8,D=16` | 检查未来位置概率为零 |
| multi-head | `B=2,H=4,N=32,Dh=16` | 检查索引 |

### 6.2 校验标准

- 检查 NaN/Inf；
- softmax 每行和接近 1；
- 输出与 CPU reference 比较最大绝对误差和最大相对误差；
- FP32 与 FP16/BF16 使用不同容差，并解释容差来源；
- 运行 `compute-sanitizer --tool memcheck`；
- kernel launch error 与异步执行 error 均检查。

## 7. 性能实验边界

- 主要平台 A100 SM 8.0；
- benchmark 使用实际检测到的 GPU 型号和显存配置；
- 真实性能在无 ncu 插桩下使用 CUDA Event 测量；
- ncu 只用于指标和瓶颈证据，避免把重放后的进程内计时当成真实性能；
- 教学版目标是正确、可解释、可 profile，不设置“必须超过某工业库”的虚假目标；
- 可使用 cuBLAS 作为拆分 GEMM 的对照，但不把 cuBLAS 结果当作 fused Attention 的直接等价基线；
- 若扩展 Tensor Core，必须同时验证精度、SASS/pipe 证据和实际性能。

## 8. 预期产出

主教材：

```text
docs/Week4_Attention与FlashAttention完整学习资料.md
```

建议实验目录：

```text
week04_attention/
├── attention_cpu.cpp
├── attention_naive.cu
├── online_softmax.cu
├── tiled_attention.cu
├── benchmark.md
└── ncu_notes.md
```

本轮实现首先完成教材正文。实验文件可以根据教材中的“必须手写”规则由学习者创建；不预先把所有作业答案写成独立可复制文件，以免绕过练习过程。

## 9. 完成标准

教材完成后应满足：

1. 不阅读其他资料也能理解正文首次出现的概念；
2. 所有公式均说明变量、shape 和归约维度；
3. 标准 Attention 有完整可运行参考和必须手写骨架；
4. online softmax 有数字推导、CPU/CUDA 练习与参考实现；
5. tiled Attention 清楚解释 `m/l/O` 更新和旧输出重缩放；
6. A100 特性与 Hopper 特性边界准确；
7. MLA/FlashMLA 置于 KV cache 和 MHA/MQA/GQA 之后；
8. 每天都有学习目标、代码产出、正确性标准、自测和口述；
9. 明确区分教学实现、优化方向和工业实现；
10. 学习者能够关掉教材写出标准 Attention 三阶段、online softmax 更新式和 tiled Attention 核心循环。
