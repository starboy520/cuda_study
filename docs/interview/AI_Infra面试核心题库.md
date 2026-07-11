# AI Infra 面试八股全集：从算子数据流到推理与分布式系统

> 适用：LLM 推理、AI Infra、GPU 算子与分布式系统岗位。CUDA 机制本身见 [CUDA 面试八股全集](CUDA面试核心题库.md)。

## 0. 使用方法

- **B 必会**：模型/推理数据流基础；
- **A 主线**：性能、显存、调度与通信分析；
- **C 加分**：生产框架、前沿内核和复杂系统设计。

回答时始终问四件事：

```text
shape 是什么？
FLOP 与 bytes 是多少？
中间状态放在哪里？
瓶颈随 batch/sequence/hardware 怎样变化？
```

## 目录

1. Tensor、GEMM/GEMV 与模型数据流
2. Softmax、Online Softmax 与 Attention
3. FlashAttention
4. Prefill、Decode 与 KV Cache
5. MQA、GQA、MLA 与 Paged Attention
6. 推理调度、融合与 CUDA Graph
7. 精度与量化
8. 分布式并行与 NCCL 场景
9. MoE
10. 推理框架与 DeepSeek 组件
11. 系统设计、项目深挖与模拟面试

---

# 1. Tensor、GEMM/GEMV 与模型数据流

### Q1：LLM 中常见 tensor shape 怎样读？

**难度：B 必会**

**15 秒回答**

常见隐藏状态为 `[B,N,D]`，拆多头后为 `[B,H,N,Dh]`，且通常 `D=H×Dh`；性能分析必须同时知道逻辑 shape、物理 layout、stride 和 dtype。

**1 分钟展开**

`B` 是 batch，`N` 是 token 数，`D` 是隐藏维，`H` 是 query heads，`Dh` 是 head dimension。reshape 可能只改元数据，transpose 产生非连续 stride，后续 contiguous 可能真实搬运。面试时先写 shape，再讨论 GEMM 维度和 bytes。

**面试官继续追问**

1. `[B,N,D]→[B,H,N,Dh]` 一定发生拷贝吗？
2. Q/K/V 的 head 数为何可能不同？

**常见误区**

- 只会背 shape，不会算元素数、stride 和显存。

**项目证据**

- [Week4 Attention 教材的 shape 预备章](../courses/attention/Week4_Attention与FlashAttention完整学习资料.md)。

### Q2：为什么同一个线性层有时像 GEMM，有时像 GEMV？

**难度：B 必会**

**15 秒回答**

权重矩阵相同，但 token/batch 维决定左矩阵有多少行；prefill 通常行数多、易形成大 GEMM，低 batch decode 每步行数少，更接近 GEMV/skinny GEMM。

**1 分钟展开**

大 GEMM 能复用权重并提高算术强度；batch=1 decode 每步只处理少量 token，权重常需要反复从 HBM 读取，可能带宽受限。continuous batching、quantization 和 fusion 会改变 shape 与 bytes，因此不能把所有 decode 一概判为 memory-bound。

**面试官继续追问**

1. 提高 decode batch 为什么可能提高吞吐却增加延迟？
2. 权重量化为什么对低 batch decode 特别有价值？

**常见误区**

- “prefill 永远 compute-bound，decode 永远 memory-bound。”

**项目证据**

- CUDA 侧 GEMM 机制见 [CUDA 八股 GEMM 章节](CUDA面试核心题库.md)。

---

# 2. Softmax、Online Softmax 与 Attention

### Q3：Stable Softmax 为什么减 max？

**难度：B 必会**

**15 秒回答**

softmax 对整行同时减同一常数不改变概率；减行最大值让最大指数为 1，避免 `exp(大正数)` 上溢。

**1 分钟展开**

$$
\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}=\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

实现通常需要 row max 和 shifted exp sum 两次归约，再归一化。全 mask 行、`-∞-(-∞)` 和低精度累计是常见边界。

**面试官继续追问**

1. 为什么减 max 不能避免所有 underflow？
2. softmax 沿 Attention 的哪个维度？

**常见误区**

- 把 softmax 作用在 query 行之间，而不是固定 query 的 key 维。

**项目证据**

- [Softmax CUDA](../../week04_gemm/softmax/softmax.cu)。

### Q4：Online Softmax 保存什么状态？

**难度：A 主线**

**15 秒回答**

保存 running max `m` 和以该 max 为基准的指数和 `l`；新块提高 max 时，旧 `l` 必须乘 `exp(m_old-m_new)` 后再加入新贡献。

**1 分钟展开**

```text
m_new = max(m_old, m_block)
l_new = exp(m_old-m_new) l_old
      + exp(m_block-m_new) l_block
```

状态可结合合并，适合并行归约。只得到 `(m,l)` 仍不能永久写出所有概率；若要输出概率还需重读或保存中间值。

**面试官继续追问**

1. 为什么旧和必须重缩放？
2. 空状态 `(-∞,0)` 怎样避免 NaN？

**常见误区**

- 把 `l` 当成未经平移的 `Σexp(x)`。

**项目证据**

- [Online Softmax 正确性证明](../proofs/Online_Softmax正确性证明.md)。

### Q5：标准 Attention 的 shape、FLOP 和输出是什么？

**难度：B 必会**

**15 秒回答**

固定一个 head，`Q,K,V` 常为 `[N,Dh]`；`S=QK^T/√Dh` 是 `[N,N]`，按 key 维 softmax 得 `P`，输出 `O=PV` 为 `[N,Dv]`。

**1 分钟展开**

若 `Dv=Dh`，两次矩阵乘主导 FLOP 约 `4N²Dh`。朴素多 kernel 会物化 `S/P`，带来 `N²` 中间存储和 HBM 读写。causal mask 对未来 key 设为负无穷，而不是 0。

**面试官继续追问**

1. 为什么缩放 `1/√Dh`？
2. K 数学转置是否意味着必须先生成转置副本？

**常见误区**

- 把 Attention 输出理解为概率矩阵；输出是 Value 的加权向量。

**项目证据**

- [完整 Attention 教材](../courses/attention/Week4_Attention与FlashAttention完整学习资料.md)。

---

# 3. FlashAttention

### Q6：FlashAttention 为什么快，它减少了什么？

**难度：A 主线**

**15 秒回答**

FlashAttention 用 tiling 和 online softmax 在片上处理局部 score/probability，不把完整 `N×N` 的 `S/P` 写回 HBM；它主要减少 IO 和中间存储，不把 dense Attention 的主导 FLOP 变成线性。

**1 分钟展开**

固定 Q tile，流式读取 K/V tile，为每个 query 行维护 `m`、`l` 和未归一化向量分子 `O_acc`。新 max 出现时，旧 `l/O_acc` 同时乘 `alpha=exp(m_old-m_new)`。最终 `O=O_acc/l`，数学上仍是 exact dense Attention，浮点顺序可能不同。

**面试官继续追问**

1. 为什么不能每个 K tile 单独 softmax 后平均？
2. K/V 是否只从 HBM 读取一次？

**常见误区**

- “FlashAttention 把 `O(N²)` 计算变成 `O(N)`。”

**项目证据**

- [FlashAttention 渐进推导](../courses/attention/Week4_Attention与FlashAttention完整学习资料.md#day-3flashattention从普通加权平均到分块在线计算)。

### Q7：`m/l/O_acc` 为什么必须使用同一基准？

**难度：A 主线**

**15 秒回答**

`l` 是指数权重的标量和，`O_acc` 是同一权重下的向量分子；max 基准改变时，两者必须同时缩放，否则分子分母单位不一致。

**1 分钟展开**

```text
alpha = exp(m_old-m_new)
l_new = alpha*l_old + rowsum(exp(S_tile-m_new))
O_new = alpha*O_old + exp(S_tile-m_new) V_tile
```

全 mask 的局部 tile 应贡献 0 并保持旧状态，不能计算 `exp(-∞-(-∞))`。

**面试官继续追问**

1. `O_acc` 的 shape 是什么？
2. 为什么局部概率暂时不除以 `l_new`？

**常见误区**

- 只缩放 denominator，不缩放 numerator。

**项目证据**

- [教学版 tiled attention](../../week04_attention/tiled_attention.cu)。

---

# 4. Prefill、Decode 与 KV Cache

### Q8：Prefill 与 Decode 的计算形态有什么差异？

**难度：B 必会**

**15 秒回答**

prefill 同时处理 prompt 多个 token，矩阵较大、并行度高；decode 自回归每步新增少量 token，要读取权重和历史 KV，低 batch 时常更受带宽和 launch 影响。

**1 分钟展开**

瓶颈不是阶段名称决定的。长 prompt prefill 的 Attention 有平方交互；decode 的 KV 读取随上下文增长。continuous batching、chunked prefill、量化、speculative decoding 和并行方式都会改变利用率与延迟。

**面试官继续追问**

1. TTFT 与 TPOT 分别主要受什么影响？
2. chunked prefill 为什么能改善调度但可能改变延迟？

**常见误区**

- 用单请求 batch=1 的结论描述所有线上 decode。

**项目证据**

- [Week5 推理资料](../courses/inference/Week5增强版_LLM推理优化与decode.md)。

### Q9：KV Cache 为什么存在，显存怎样计算？

**难度：B 必会**

**15 秒回答**

KV cache 保存每层历史 token 的 K/V，避免 decode 时重复计算历史状态；MHA 的基本字节量为 `2×L×B×N×Hkv×Dh×bytes_per_elem`。

**1 分钟展开**

`2` 代表 K/V，`L` 层数，`B` batch，`N` 已缓存 token，`Hkv` KV heads。MHA 中 `Hkv=Hq`，MQA 常为 1，GQA 介于两者；MLA 存储的是潜在表示和位置相关部分，不能套同一公式。还要计 block metadata、对齐、临时 workspace 和碎片。

**面试官继续追问**

1. 为什么 cache 节省计算却增加带宽压力？
2. beam search/并发请求如何改变容量？

**常见误区**

- 公式里使用 query heads 代替 KV heads。

**项目证据**

- [KV Cache 学习指南](../topics/kv_cache/大模型KVCache系统学习指南.md)。

### Q10：MQA、GQA、MLA 分别怎样减少 KV？

**难度：A 主线**

**15 秒回答**

MQA 让所有 query heads 共享一组 K/V；GQA 让一组 query heads 共享一个 KV head；MLA 将 KV 信息压到低维潜在表示，并在计算时重构/吸收相关投影。

**1 分钟展开**

MQA/GQA 直接减少 `Hkv`，以表达能力和实现权衡换缓存/带宽。MLA 的缓存项和 attention 数据流不同，收益不能只用“KV heads 更少”解释；还要看 decoupled RoPE、矩阵吸收和 kernel 支持。

**面试官继续追问**

1. GQA 的 group size 怎样计算？
2. MLA 为什么需要专门 kernel 而不只是 reshape？

**常见误区**

- 把 MLA 说成 MQA 的另一个名称。

**项目证据**

- [Week4 MLA 章节](../courses/attention/Week4_Attention与FlashAttention完整学习资料.md)。

---

# 5. Paged Attention 与显存管理

### Q11：Paged Attention 解决什么问题？

**难度：A 主线**

**15 秒回答**

它把每个请求逻辑连续的 KV 序列映射到固定大小物理块，通过 block table 寻址，减少为最大长度预留和外部碎片，并支持动态请求生命周期。

**1 分钟展开**

逻辑 token block 不要求物理连续；kernel 根据请求、逻辑块和 offset 找到物理 KV。代价是间接寻址、metadata、块内浪费和更复杂调度。它不是虚拟内存硬件分页的简单复制，而是推理 runtime 与 attention kernel 的协同设计。

**面试官继续追问**

1. block 太大或太小分别有什么代价？
2. prefix sharing 如何利用块引用？

**常见误区**

- “Paged Attention 主要降低 Attention FLOP。”

**项目证据**

- vLLM 的具体实现会变化，应查 [vLLM 官方文档](https://docs.vllm.ai/)而不是背旧版本字段。

### Q12：KV block 分配、换出和复用怎样影响调度？

**难度：A 主线**

**15 秒回答**

调度器必须在每步考虑可用 KV blocks；容量不足时可能暂停、抢占、换出或拒绝请求，prefix reuse 则减少重新计算与内存。

**1 分钟展开**

容量调度与 token 调度不能分离。块的生命周期、引用计数、eviction policy、offload 路径和 attention window 都影响吞吐与尾延迟。例如 TensorRT-LLM 官方 KV cache 系统以 blocks 管理状态，并支持复用、offload 与优先级驱逐，但具体策略具有版本性。

**面试官继续追问**

1. eviction 与 recompute 如何权衡？
2. disaggregated prefill/decode 为什么需要 KV transfer？

**常见误区**

- 只按模型权重判断能支持多少并发请求。

**项目证据**

- [TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)。

---

# 6. 推理调度、融合与 CUDA Graph

### Q13：Continuous/Inflight Batching 为什么提高吞吐？

**难度：A 主线**

**15 秒回答**

它在每个推理步动态移除完成请求、加入新请求，让 GPU 持续形成更有效 batch，减少静态 batch 等最慢请求造成的空洞。

**1 分钟展开**

调度要平衡 KV 容量、prefill/decode 混合、优先级、抢占、公平性和 SLA。batch 大通常提高吞吐，但可能增加排队和 TPOT；chunked prefill 可限制一次占用，但引入更多调度与中间状态。

**面试官继续追问**

1. head-of-line blocking 怎样发生？
2. 为什么只优化平均吞吐可能恶化 P99？

**常见误区**

- “continuous batching 只是每轮把 tensor `cat` 一下。”

**项目证据**

- [TensorRT-LLM Scheduler](https://nvidia.github.io/TensorRT-LLM/torch/scheduler.html)展示了容量与 microbatch 调度的区分。

### Q14：算子融合和 CUDA Graph 分别减少什么？

**难度：A 主线**

**15 秒回答**

融合主要减少中间 HBM 读写和 launch；Graph 固化重复 launch/依赖序列，主要降低 CPU launch/调度开销。二者解决的问题不同且可以组合。

**1 分钟展开**

融合可能增加寄存器、降低 occupancy、减少并行选择；Graph 受动态 shape、地址、控制流和 update 约束。decode 中短小重复 kernel 更可能受益，但动态 batching 需要稳定的 bucket/graph 策略。

**面试官继续追问**

1. 哪些 elementwise 更适合融合进 epilogue？
2. Graph 为什么不降低权重读取 bytes？

**常见误区**

- 把 fusion、Graph、persistent kernel 混成一种优化。

**项目证据**

- CUDA Graph 机制见 [CUDA 面试八股](CUDA面试核心题库.md)。

---

# 7. 精度与量化

### Q15：FP16、BF16、TF32、FP8 的区别是什么？

**难度：B 必会**

**15 秒回答**

FP16 尾数相对多但指数范围窄；BF16 指数范围接近 FP32、尾数短；TF32 是 NVIDIA Tensor Core 计算格式；FP8 进一步压缩范围/精度，通常必须配 scaling。

**1 分钟展开**

要区分存储 dtype、乘法输入格式和 accumulator。硬件支持随架构不同；“A100 默认 TF32”必须限定到特定库/API 的 math mode，不能说所有 FP32 CUDA 运算自动变 TF32。

**面试官继续追问**

1. BF16 为什么训练中更不易 overflow？
2. FP8 E4M3/E5M2 为什么用途不同？

**常见误区**

- 只比较 bit 数，不比较指数、尾数和累计类型。

**项目证据**

- [混合精度与 Scaling](../../week06_tensorcore/mixed_precision_and_scaling.md)。

### Q16：Weight-only、W8A8、KV Cache 量化有什么差异？

**难度：A 主线**

**15 秒回答**

Weight-only 主要减权重带宽，激活保持高精度；W8A8 同时量化权重/激活以利用低精度计算；KV 量化减少随上下文增长的缓存容量和读取带宽。

**1 分钟展开**

要说明 scale 粒度（tensor/channel/group/token/block）、静态/动态、zero point、校准和反量化融合。支持矩阵依赖 GPU、kernel、模型和框架版本；不能仅凭格式存在就断言有高性能 kernel。

**面试官继续追问**

1. per-channel 为什么常优于 per-tensor？
2. KV 量化为什么对长上下文敏感？

**常见误区**

- “INT4 模型显存和端到端延迟都必然减半。”

**项目证据**

- 当前支持情况应查 [TensorRT-LLM Quantization](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html)。

---

# 8. 分布式并行与 NCCL 场景

### Q17：DP、TP、PP、CP、EP 分别切什么？

**难度：B 必会**

**15 秒回答**

DP 复制模型切 batch；TP 切层内张量/矩阵；PP 切层；CP 切长序列上下文；EP 切 MoE experts。每种并行减少不同资源压力，也引入不同通信。

**1 分钟展开**

DP 梯度 all-reduce；TP 常有 all-reduce/all-gather/reduce-scatter；PP 传 activation 并有 bubble；CP 需要跨设备 attention/KV 协作；EP 需要 token dispatch/combine all-to-all。实际系统常组合多维并行。

**面试官继续追问**

1. 为什么 TP 太大可能被通信延迟限制？
2. PP microbatch 怎样减少 bubble？

**常见误区**

- 只说“把模型分到多卡”，不说明切分维度和通信点。

**项目证据**

- [多 GPU 教材](../../cuda_deep_course/course/volume08_hpc_multigpu/README.md)。

### Q18：怎样估算 Tensor Parallel 通信量？

**难度：A 主线**

**15 秒回答**

先明确矩阵按行还是列切、每卡产生什么 partial/partition，再根据 collective 输入元素数、dtype、rank 数和算法估算每 rank 字节。

**1 分钟展开**

列并行可产生分片输出，后续若需要完整 activation 要 all-gather；行并行产生 partial sum，常需 all-reduce/reduce-scatter。不要直接背固定倍数，先画数据分布。通信能否隐藏取决于 bucket、依赖、stream、网络和剩余计算。

**面试官继续追问**

1. Reduce-Scatter 后为什么可以接分片计算？
2. latency-bound 与 bandwidth-bound collective 怎样区分？

**常见误区**

- 用总网络字节替代每 rank critical-path 字节。

**项目证据**

- NCCL 机制边界见 [CUDA 八股多 GPU 章节](CUDA面试核心题库.md)。

---

# 9. MoE

### Q19：MoE 一层的数据流是什么？

**难度：B 必会**

**15 秒回答**

router 为 token 选择 top-k experts，dispatch 把 token 发送到 expert 所在设备，experts 执行 grouped GEMM，combine 按权重汇总并恢复 token 顺序。

**1 分钟展开**

性能包括 routing、permute、all-to-all、local expert compute、combine。负载不均会让热点 expert/设备拖慢全局；capacity、drop/padding、expert placement 和通信拓扑共同影响效率。

**面试官继续追问**

1. 为什么 MoE 参数多但每 token FLOP 不按总参数同比增长？
2. grouped GEMM 为什么适合 experts？

**常见误区**

- 只讨论 expert GEMM，不计算 dispatch/combine。

**项目证据**

- 当前项目主要为资料层理解，没有真实多卡 MoE 实测。

### Q20：MoE 为什么需要 All-to-All，怎样优化？

**难度：A 主线**

**15 秒回答**

token 产生设备与 expert 所在设备通常不同，需要按目的 expert 重分布；优化方向是减少/重叠通信、改善负载、优化 layout/permute 和使用低延迟通信 kernel。

**1 分钟展开**

小 token 数可能 latency-bound，大 dispatch 可能 bandwidth-bound。应分开测 pack、network、unpack、expert compute。通信与计算重叠需要任务分块和依赖允许，不是多 stream 自动实现。

**面试官继续追问**

1. top-k 增大怎样影响通信量？
2. expert parallel 与 tensor parallel 怎样组合？

**常见误区**

- 把 all-to-all 当单个 NCCL 调用时间，不计布局转换。

**项目证据**

- [DeepEP 官方仓库](https://github.com/deepseek-ai/DeepEP)用于理解 expert-parallel 通信库目标。

---

# 10. 推理框架与 DeepSeek 组件

### Q21：vLLM、TensorRT-LLM、SGLang 的边界怎样回答？

**难度：A 主线**

**15 秒回答**

它们都是推理系统，但抽象、runtime、kernel/编译集成和服务能力不同；回答应描述核心职责和选择维度，不背容易过时的“谁支持某功能”列表。

**1 分钟展开**

vLLM 以高吞吐 serving、KV 管理/PagedAttention 生态著称；TensorRT-LLM 面向 NVIDIA GPU 的编译、kernel、量化和 C++ runtime 优化；SGLang 强调结构化生成/runtime、RadixAttention 等。功能持续演进，应按目标模型、硬件、延迟、吞吐、可维护性和版本实测选择。

**面试官继续追问**

1. 为什么功能支持不等于该组合性能成熟？
2. 怎样设计公平 benchmark？

**常见误区**

- 用某个旧版本博客做永久功能排名。

**项目证据**

- 官方入口：[vLLM](https://docs.vllm.ai/)、[TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/latest/)、[SGLang](https://docs.sglang.ai/)。

### Q22：DeepGEMM、FlashMLA、DeepEP、DualPipe 分别解决什么？

**难度：C 加分**

**15 秒回答**

DeepGEMM 面向高效低精度/分组 GEMM；FlashMLA 面向 MLA attention kernel；DeepEP 面向 expert-parallel 通信；DualPipe 是流水并行和计算通信重叠调度思想。

**1 分钟展开**

四者位于不同层：计算 kernel、attention kernel、通信库和训练 pipeline schedule。不能说“都用于加速 DeepSeek”就结束；要画出它们各自在一层模型执行中的输入输出、依赖和瓶颈。仓库会持续更新，具体支持以官方 README 为准。

**面试官继续追问**

1. DeepGEMM grouped GEMM 与 MoE 怎样连接？
2. DeepEP 输出 layout 如何影响后续 expert GEMM？

**常见误区**

- 读过 README 就声称掌握核心 CUDA mainloop。

**项目证据**

- 官方仓库：[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、[FlashMLA](https://github.com/deepseek-ai/FlashMLA)、[DeepEP](https://github.com/deepseek-ai/DeepEP)、[DualPipe](https://github.com/deepseek-ai/DualPipe)。

---

# 11. 系统设计、项目深挖与模拟面试

### Q23：怎样为模型服务估算 GPU 显存？

**难度：A 主线**

**15 秒回答**

至少拆成权重、KV cache、运行时 activation/workspace、通信/graph buffer、allocator 碎片与安全余量；不能只用参数量×dtype。

**1 分钟展开**

权重量化与 TP 改变每卡权重；KV 随并发和 token 数动态变化；prefill workspace 与 attention 实现有关；CUDA Graph 可能固定地址/预留 buffer；多卡通信有额外 workspace。最终用目标框架实际峰值校准理论模型。

**面试官继续追问**

1. 怎样从 SLA 反推最大并发 token？
2. 为什么 allocator reported free memory 不等于可分配最大连续块？

**常见误区**

- 不留碎片和 runtime 安全余量。

**项目证据**

- 可使用第 Q9 的 KV 公式作为动态项。

### Q24：如何设计一次可信的 LLM 推理性能评测？

**难度：C 加分**

**15 秒回答**

固定模型、精度、硬件、输入/输出长度分布和并发策略，同时报告 TTFT、TPOT、吞吐、P50/P95/P99、显存和正确性，区分离线吞吐与在线 SLA。

**1 分钟展开**

需要 warmup、固定版本/配置、请求分布和随机种子；分别测 prefill/decode、不同 batch/context、稳态和过载。框架比较必须对齐量化、kernel、max tokens、scheduler、prefix cache 和 speculative decoding，否则数字不可比。

**面试官继续追问**

1. goodput 与 raw throughput 有何区别？
2. 为什么平均延迟会掩盖排队崩溃？

**常见误区**

- 只测单个短 prompt 的 tokens/s。

**项目证据**

- CUDA benchmark 方法见 [实验完成标准](../../cuda_deep_course/course/实验方法与完成标准.md)。

## 项目深挖题

1. 从普通 Attention 推导 `m/l/O_acc`，解释为什么 exact。
2. 设计一个实验区分 FlashAttention 的 FLOP 收益与 IO 收益。
3. 给定模型参数，计算 MHA/GQA 的 KV bytes。
4. 设计 continuous batching 调度器的输入、状态和目标函数。
5. TP=8 性能不升反降时怎样拆通信、kernel 和调度时间？
6. MoE P99 抖动时怎样区分路由不均、all-to-all 和 expert GEMM？

## 高频容量计算

```text
权重 bytes ≈ parameters × effective bytes/parameter

MHA KV bytes
= 2 × layers × batch × cached_tokens × kv_heads × head_dim × dtype_bytes

Attention dense FLOP（单 head，Dv=Dh）
≈ 4 × N² × Dh
```

公式是起点，不包含 metadata、padding、workspace、fragmentation 或通信副本。

## 三档模拟面试

### B：基础数据流

1. Q/K/V shape 与 softmax 维度。
2. prefill/decode 区别。
3. KV cache 容量。
4. MQA/GQA。
5. DP/TP/PP。

### A：性能与系统

1. FlashAttention IO 账本。
2. Paged Attention block size 权衡。
3. continuous batching 的吞吐/P99。
4. TP 通信量与 overlap。
5. quantization 的 bytes、kernel 和误差闭环。

### C：架构深挖

1. disaggregated prefill/decode 与 KV transfer。
2. MoE all-to-all + grouped GEMM pipeline。
3. DeepGEMM/FlashMLA/DeepEP 的接口衔接。
4. 多租户推理的容量、抢占和隔离。

## 复习清单

- [ ] 所有性能结论都能说明适用的 batch/shape/hardware。
- [ ] 能手算 Attention 和 KV cache 账本。
- [ ] 能区分 kernel、runtime、scheduler、communication 四层问题。
- [ ] 框架题以官方当前文档为准，不背永久功能排名。
- [ ] 能把一次请求画成 prefill→KV→多步 decode→输出的数据流。

## 官方资料

- [vLLM Documentation](https://docs.vllm.ai/)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/latest/)
- [SGLang Documentation](https://docs.sglang.ai/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [DeepSeek-AI repositories](https://github.com/orgs/deepseek-ai/repositories)
