# Week5 LLM 推理优化自包含教材重写设计

## 目标

将 `docs/Week5增强版_LLM推理优化与decode.md` 从任务清单重写为一份可直接学习的 7 天高强度、自包含教材。

目标读者会 CUDA 基础和部分性能优化，但刚接触 ML/LLM 推理。教材不能默认读者已经理解 token、Transformer、prefill/decode、KV Cache、GEMV、RMSNorm、量化、Paged Attention 或推理框架。

课程保持较高强度，每天建议 3–5 小时；“高强度”通过增加有效学习和实验密度实现，而不是跳过概念解释。

## 教学主线

全书围绕“一次新 token 是怎样生成的，以及每层优化解决哪个瓶颈”展开：

```text
文本/token
→ prefill 建立上下文和 KV
→ decode 每步处理新 token
→ 线性层形成 GEMV/skinny GEMM
→ 小算子与 HBM 往返促成融合
→ 大量短 kernel 促成 CUDA Graph
→ 权重/KV bytes 促成量化
→ 动态请求和 KV 生命周期促成 Paged Attention
→ 框架把 kernel、KV 和调度组织起来
→ Hopper 用 TMA/WGMMA/cluster 继续改变内核组织
```

每个主题必须先说明它在这条数据流中的位置，再进入 CUDA 实验。

## 固定教学结构

每天包含：

1. 当天在完整 decode 流程中的位置；
2. 小白前置概念；
3. 最小 shape 和数字手算；
4. CPU/reference 版本；
5. CUDA 线程、数据和内存映射；
6. 可编译外围框架；
7. 学习者实现的核心 TODO；
8. 三级提示；
9. 编译和运行命令；
10. 正确性、边界和数值测试；
11. ncu/nsys 的观察目标；
12. 常见错误和错误结论；
13. 闭卷自测和面试口述。

若某天主要是系统概念而不是 CUDA kernel，CPU/reference 替换为可计算的容量、字节、时间线或 block-table 练习。

## 代码边界

教材提供：

- 完整 `main`、参数解析、host/device 内存管理；
- CPU/reference 实现；
- CUDA 错误检查、Event 计时和误差检查；
- kernel 签名、数据结构和线程映射外围框架；
- 编译命令、测试 shape、参考输出形态；
- TODO 的三级提示。

学习者实现：

- warp GEMV 的点积与归约；
- fused residual/RMSNorm/activation 核心；
- Graph capture/instantiate/replay 主流程；
- cooperative grid reduction 的 grid sync 核心；
- INT8 weight-only dequant GEMV 核心。

正文不直接给这些核心 TODO 的完整答案。若后续需要参考实现，应单独生成，避免学习时直接看到答案。

## Day 1：从零理解 LLM 推理与 KV Cache

### 内容

- token、token id、embedding、hidden state、logits 和采样的最小定义；
- 用一个微型 Transformer 层说明 RMSNorm、Attention、线性层、残差的位置；
- prefill 与 decode 的时间顺序、shape 和数据生命周期；
- batch、sequence length、hidden dimension、query heads、KV heads、head dimension；
- 为什么无 cache 会在每步重复计算历史 K/V；
- 从 shape 推导 KV bytes；
- MHA、MQA、GQA、MLA 的缓存差异只讲到 Week5 所需深度；
- latency、throughput、TTFT、TPOT；
- 为什么 prefill/decode 的瓶颈是条件性判断，而不是固定标签。

### 手算

至少两组 KV 配置，明确：

```text
KV bytes
= 2 × layers × batch × cached_tokens
  × kv_heads × head_dim × dtype_bytes
```

并说明公式不包含 metadata、padding、workspace 和 allocator fragmentation。

## Day 2：GEMV 从数学、CPU 到 CUDA

### 内容

- 从线性层 `y=Wx+b` 推出 GEMV shape；
- 用 3×4 小矩阵完整手算；
- GEMV 与 GEMM 的相同数学和不同复用；
- `W` 每个元素在单个 GEMV 中通常使用一次，但 `x` 跨输出行复用；
- CPU reference；
- CUDA v0：一个线程负责一行；
- CUDA v1：一个 warp 负责一行；
- lane-stride 点积、active mask、warp reduction；
- 行主序下如何同时考虑同一行点积和跨 warp global coalescing；
- FLOP、logical bytes、有效带宽和 GFLOPS。

### TODO

学习者完成 warp 点积、shuffle reduction 和尾部 mask。

## Day 3：GEMV 性能深化与工具

### 内容

- float4/half2 的对齐、尾部和最终 SASS 验证；
- `x` 的 shared/constant/cache 复用条件；
- 多行/多 warp/block 映射方案；
- register pressure、spill、ILP/TLP 和 occupancy；
- 理论 occupancy 的资源约束推导；
- 为什么 memory-bound kernel 仍不能使用固定带宽百分比验收；
- Roofline 对 GEMV 的作用与局限；
- ncu 看单 kernel，nsys 看循环和气泡；
- 建立 baseline→假设→修改→复测表。

### 实验

比较至少三个版本和多个 `N/K/batch`，不能只测单一大方阵。

## Day 4：RMSNorm、Residual、激活与融合

### 内容

- residual add、RMSNorm、SiLU/GELU 的数学和 shape；
- RMSNorm 与 LayerNorm 的区别；
- 稳定累计、epsilon 和 dtype；
- 未融合多 kernel 的 HBM 读写账本；
- 哪些中间量可留在 register/shared；
- 融合带来的 register、occupancy、并行度和维护成本；
- CPU reference、未融合 CUDA 和 fused CUDA；
- compute-sanitizer 的 memcheck、racecheck、initcheck、synccheck；
- 正确性测试包含非整除维度、零值、大值和随机输入。

### TODO

学习者实现 block/warp 归约、RMS inverse、融合写回。

## Day 5：CUDA Graph 与 Cooperative Grid Sync

### CUDA Graph

- kernel launch 的 CPU/GPU 时间线；
- capture、instantiate、replay、destroy 生命周期；
- 普通逐 launch 与 Graph replay 的对照；
- Graph 降低 launch 开销，不降低单 kernel FLOP/bytes；
- 动态 shape、地址、batch、控制流和 update 限制；
- nsys 观察 kernel gap；
- 什么规模下 Graph 收益可能被噪声淹没。

### Cooperative Grid Sync

- 普通 CTA 同步边界；
- 为什么 global counter 自旋可能死锁；
- cooperative launch、`grid_group::sync()` 和设备能力；
- 所有参与 block 同时驻留相关约束；
- 普通两阶段 reduction 与单 kernel grid-sync reduction 对比；
- persistent kernel 只讲必要联系，不把 grid-stride 等同 persistent scheduling。

### TODO

学习者完成 Graph 生命周期调用和 grid sync 归约核心。

## Day 6：INT8 Weight-only 量化 GEMV

### 内容

- signed INT8、量化范围、scale 和 round/clamp；
- 对称与非对称量化；
- per-tensor、per-channel、per-group 的误差/metadata/访存权衡；
- 用一组小权重完整手算 quantize→dequantize→dot product；
- weight-only 与 W8A8、KV cache quant 的区别；
- FP16/FP32 baseline 与 INT8 storage + dequant GEMV；
- 权重 bytes 减少不保证加速：反量化、scale、转换、指令路径和 shape；
- 正确性不能只与原浮点逐 bit 比较，应报告量化误差；
- ncu 比较 DRAM bytes、计算指令、occupancy 和时间。

### TODO

学习者实现 per-channel dequant GEMV 的核心加载、scale 和累加。

## Day 7：Paged Attention、推理框架与 Hopper

### Paged Attention

- 连续 KV buffer 的预留、增长、碎片和并发请求问题；
- logical block、physical block、block table、token offset；
- 用两个请求、四个 physical blocks 完整手推地址映射；
- block size 的内部浪费、metadata、间接寻址和并行度权衡；
- prefix sharing、copy-on-write/引用关系只讲必要概念；
- Paged Attention 优化 KV 管理，不把 dense Attention FLOP 自动变线性。

### Continuous Batching 与框架

- 请求在每个 decode step 动态加入/退出；
- KV capacity、prefill/decode 混合、吞吐、TPOT 和 P99；
- vLLM、TensorRT-LLM、SGLang 的职责边界以官方当前资料为准；
- 不制作容易过时的永久功能排行榜。

### Hopper 主线

- 先复习 A100 `cp.async + mma.sync`；
- `cp.async→TMA`：descriptor 驱动多维 bulk transfer；
- warp MMA→WGMMA：warp-group 异步矩阵计算；
- CTA→Thread Block Cluster；
- local shared→DSM；
- mbarrier 和 producer/consumer warp specialization；
- 没有 H100 时只做架构迁移理解，不声称真实性能。

Hopper 章节先自包含讲清在 decode/推理 kernel 中的作用，再链接 `docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md` 深挖。

### 最终产出

画出一次请求：

```text
prompt
→ prefill
→ KV block allocation
→ 多步 decode
→ GEMV/attention/fused ops
→ Graph/runtime scheduling
→ logits/sampling
```

并标注每一步的主要 FLOP、bytes、状态和可能瓶颈。

## 已有资料的连接规则

Week5 正文必须先讲到学习者能开始当天实验，再链接深度资料：

- Attention/Online Softmax：`docs/Week4_Attention与FlashAttention完整学习资料.md`；
- KV/Paged Attention/框架：`docs/AI_Infra面试八股全集.md` 与 KV 专项资料；
- Occupancy/ncu：已有专项文档和 A100 GEMM 记录；
- CUDA Graph/Cooperative Groups：异步系统课程和 Programming Guide 路径；
- Hopper：CUDA 深水区两周教材；
- Compute Sanitizer：性能卷和调试资料。

链接是用于继续深挖，不能代替当天必要解释。不得出现“先读完另外三份长文档才能开始”的依赖。

## 必须修正的原文表述

- 不写 `prefill=compute-bound`、`decode=memory-bound` 的无条件等式；
- 不把 decode 永远等同 `M=1`，说明 effective batch；
- 不写 GEMV “没有任何复用”或 “tiling 无用”；
- 不使用 `>70% HBM` 这类固定验收阈值；
- 不声称固定 shape 的 decode “完美适合 Graph”；
- 不把 Graph 说成消除所有 GPU 气泡；
- 不把 cooperative grid sync 说成普通 kernel 可任意使用；
- 不声称 INT8 bytes 减少必然带来同等加速；
- 不把 Paged Attention 只解释成 OS 分页类比；
- 不把 Hopper 新特性描述成自动加速开关。

## 验收标准

- 读者不依赖额外 ML 基础文档，也能理解 Day 1–7 主线；
- 每天都有固定教学结构，概念、shape、参考、CUDA、TODO、提示、命令、测试、profiling 和自测齐全；
- 核心 TODO 不泄露完整答案，但外围代码足以开始实现；
- 所有项目本地链接存在；
- 框架与架构事实以官方一手资料核对并注明条件；
- Markdown 围栏、表格、公式和 Mermaid 有效；
- 重写只修改目标文档，不修改现有 CUDA 源码或纳入其他未提交文件；
- 原文任务（GEMV、融合、Graph、grid sync、量化、Paged Attention、框架、Hopper）全部保留，但建立清晰的知识和实验阶梯。
