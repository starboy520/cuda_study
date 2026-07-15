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

---

# 附录 A：Q1-Q24 追问答案

> 使用方式：先闭卷回答主文中的追问，再到本附录核对。每条答案按“结论→原理/条件→边界”组织，不要求逐字背诵。

## Q1：Tensor shape

**1. `[B,N,D]→[B,H,N,Dh]` 一定发生拷贝吗？**

- 不一定。如果原 tensor 连续且新 shape 与原 stride 兼容，`reshape/view` 只修改元数据；`permute/transpose` 通常也只改 stride，但会产生非连续 layout。只有后续算子不支持该 stride、显式调用 `contiguous()`，或 reshape 无法用 stride 表达时，才发生真实搬运。

**2. Q/K/V 的 head 数为何可能不同？**

- MHA 通常有 `Hq=Hkv`；GQA 让一组 query heads 共享一个 KV head，MQA 让全部 query heads 共享一组 K/V，以减少 KV cache 容量和 decode 读取量。例如 `Hq=32,Hkv=8` 时，每 4 个 query heads 共享一个 KV head。

## Q2：GEMM 与 GEMV

**1. 提高 decode batch 为什么可能提高吞吐却增加延迟？**

- 更大 batch 能增加权重复用、并行度和 GPU 利用率，提高 tokens/s；但请求可能等待凑批，每步处理更多 token 也可能拉长 step time。因此吞吐提高时，排队延迟和单请求 TPOT 可能变差。

**2. 权重量化为什么对低 batch decode 特别有价值？**

- 低 batch 时每次加载的权重服务于很少 token，权重 HBM bytes 更容易主导。INT8/INT4 weight-only 可减少搬运量，若反量化与计算融合良好便能降低延迟；收益仍取决于量化粒度、硬件、kernel 和 shape，不能直接按位宽比例推断。

## Q3：Stable Softmax

**1. 为什么减 max 不能避免所有 underflow？**

- 减 max 保证最大指数为 1，并避免大正数上溢；但远小于最大值的元素仍可能使 `exp(x_i-m)` 下溢为 0。这通常表示其概率在当前精度下可忽略；若大量有效项下溢，则应检查输入尺度或使用更高精度累计。

**2. softmax 沿 Attention 的哪个维度？**

- 对 score `[B,H,Nq,Nk]`，softmax 沿最后的 key 维 `Nk`。即固定 batch、head 和 query，对所有可见 key 的权重归一化为 1；causal/padding mask 也作用在该维。

## Q4：Online Softmax

**1. 为什么旧和必须重缩放？**

- `l_old` 以 `m_old` 为指数基准，而合并后统一基准是 `m_new`，所以旧贡献必须乘 `exp(m_old-m_new)`。否则新块出现更大 max 时，新旧指数和不在同一尺度，结果错误。

**2. 空状态 `(-∞,0)` 怎样避免 NaN？**

- 不能无条件计算 `exp(-∞-(-∞))`。可显式维护有效标记：无有效元素的 tile 保持旧状态，第一次出现有效元素时直接初始化；全 mask 行按接口约定输出零或单独处理，避免 NaN 扩散。

## Q5：标准 Attention

**1. 为什么缩放 `1/√Dh`？**

- 当 Q/K 各维方差相近时，点积方差近似随 `Dh` 线性增长。除以 `√Dh` 能稳定 score 尺度，避免 softmax 过早饱和和梯度集中；它控制数值分布，不改变 shape。

**2. K 数学转置是否意味着必须先生成转置副本？**

- 不意味着。`QK^T` 首先是数学索引关系，kernel 可根据 K 的 layout/stride 直接按转置方式取数。只有当前 layout 访存很差且转置成本可被后续多次复用摊销时，显式转置才可能值得。

## Q6：FlashAttention IO

**1. 为什么不能每个 K tile 单独 softmax 后平均？**

- 每个 tile 的局部 softmax 有独立分母，直接平均会错误地给各 tile 相同总权重。正确合并必须保留局部 max、指数和与向量分子，并转换到同一个全局 max 基准。

**2. K/V 是否只从 HBM 读取一次？**

- 不能笼统说只读一次。不同 Q tiles 往往会重新流式读取同一 K/V tile；FlashAttention 的核心收益是避免完整 `N×N` score/probability 的 HBM 写回与重读，并增加片上复用。实际读取次数取决于 tiling、循环顺序和 cache。

## Q7：`m/l/O_acc`

**1. `O_acc` 的 shape 是什么？**

- 对含 `Br` 个 query 的 tile，`O_acc` 通常为 `[Br,Dv]`，每行维护一个长度为 `Dv` 的未归一化输出分子；`m/l` 则各是长度为 `Br` 的逐行状态。

**2. 为什么局部概率暂时不除以 `l_new`？**

- 后续 tile 可能提高 running max 并改变分母，过早归一化会迫使旧输出反复换尺度。保持未归一化 `O_acc`，让它与 `l` 用同一指数基准，最后统一做 `O=O_acc/l` 更直接。

## Q8：Prefill 与 Decode

**1. TTFT 与 TPOT 分别主要受什么影响？**

- TTFT 包含排队、prompt prefill、首 token 调度和必要传输，对输入长度和 prefill 竞争敏感；TPOT 反映稳态 decode step，受 batch、上下文长度、KV/权重读取和 launch 开销影响。二者还受量化、并行策略和服务负载共同影响。

**2. chunked prefill 为什么能改善调度但可能改变延迟？**

- 把长 prompt 拆成小块，可在块间插入 decode，缓解 head-of-line blocking；代价是更多调度、launch 和状态管理，也可能延后该 prefill 请求自身完成。它是在 TTFT、TPOT、公平性和吞吐间权衡。

## Q9：KV Cache

**1. 为什么 cache 节省计算却增加带宽压力？**

- KV cache 消除了历史 token 的 K/V 重算，但每个新 query 仍需读取历史 K/V。随着上下文增长，每步扫描 bytes 增加，而新增 query 很少，于是重复计算被转化为持续增长的状态读取。

**2. beam search/并发请求如何改变容量？**

- beam search 为多个候选序列维护状态；共同前缀可块级共享并 copy-on-write，但独立后缀仍增加 KV blocks。并发容量应按所有活跃请求/beam 的 cached tokens 总和计算，不能只看最大单请求长度。

## Q10：MQA/GQA/MLA

**1. GQA 的 group size 怎样计算？**

- 当 `Hq` 可被 `Hkv` 整除时，`group_size=Hq/Hkv`；常见映射为 `kv_head=q_head/group_size`。例如 `32/8=4`，每 4 个 query heads 对应一个 KV head。

**2. MLA 为什么需要专门 kernel 而不只是 reshape？**

- MLA 改变了缓存内容和计算图：缓存压缩 latent，并涉及低秩投影、decoupled RoPE、矩阵吸收等。它不是对相同元素改 shape；若沿用普通 MHA kernel，重构和中间 tensor 可能抵消收益。

## Q11：Paged Attention

**1. block 太大或太小分别有什么代价？**

- 太大会增加最后一个不满块的内部浪费，降低细粒度分配/驱逐/共享；太小会增加 block table、引用计数和间接寻址开销，也可能降低访问连续性。应结合 `Dh`、dtype、kernel tile 和序列分布实测。

**2. prefix sharing 如何利用块引用？**

- 相同前缀的只读 KV blocks 可被多个请求的 block table 共同引用，并通过引用计数管理；后续写入新后缀时分配新块或 copy-on-write。它减少重复 prefill 与 KV 存储，不降低 dense Attention 的 FLOP。

## Q12：KV block 调度

**1. eviction 与 recompute 如何权衡？**

- eviction 释放容量，但恢复时要 offload/回传或 recompute。前者依赖传输带宽，后者消耗 GPU 计算；选择取决于前缀长度、复用概率、链路和当前资源压力，还要计入 P99 与公平性。

**2. disaggregated prefill/decode 为什么需要 KV transfer？**

- prefill 节点生成历史 K/V，decode 节点必须持有这些状态。阶段位于不同 GPU/主机时，需要传输 KV、同步 block metadata 和请求状态；传输时间、目标 layout 及能否与计算重叠决定解耦是否有收益。

## Q13：Continuous Batching

**1. head-of-line blocking 怎样发生？**

- 长 prefill、大请求或资源不足请求长期占据执行队列时，后面的短请求被迫等待；静态 batching 中已完成序列也可能被最慢序列拖住。chunked prefill、迭代级调度和优先级可缓解，但需避免饥饿。

**2. 为什么只优化平均吞吐可能恶化 P99？**

- 增大 batch、延长凑批窗口可提高平均 tokens/s，却让少数请求排队更久；接近饱和后队列还会非线性增长。在线系统应同时约束 TTFT、TPOT、P99，并报告满足 SLA 的 goodput。

## Q14：Fusion 与 CUDA Graph

**1. 哪些 elementwise 更适合融合进 epilogue？**

- bias、简单 activation、scale、clamp、类型转换和部分 residual 只逐元素消费 GEMM 输出，适合在结果仍位于寄存器时完成。需要跨元素归约、复杂广播或显著增加寄存器压力的操作则需重新评估。

**2. Graph 为什么不降低权重读取 bytes？**

- Graph 降低的是重复 launch/依赖提交的 CPU 和驱动调度开销，不改变 kernel 内数据流。权重 HBM bytes 只有通过量化、融合、cache/持久化等策略才可能减少。

## Q15：低精度格式

**1. BF16 为什么训练中更不易 overflow？**

- BF16 的指数位宽与 FP32 相同，动态范围远大于 FP16，因此激活/梯度更不易因范围不足溢出；代价是尾数更短。实践中仍常用 FP32 accumulator/master weights 或 scaling 控制误差。

**2. FP8 E4M3/E5M2 为什么用途不同？**

- E4M3 尾数更多、精度较高但范围较小；E5M2 牺牲精度换更大动态范围。具体选择依赖数据分布、scaling recipe、硬件和框架，不能永久绑定某一种 tensor 类型。

## Q16：量化

**1. per-channel 为什么常优于 per-tensor？**

- per-tensor 共用 scale，异常通道会浪费其他通道的量化级别；per-channel 适应通道间幅值差异，通常误差更小。代价是更多 scale metadata 和更复杂的加载/广播。

**2. KV 量化为什么对长上下文敏感？**

- KV bytes 随 cached tokens 增长，长上下文下容量和带宽收益更明显；同时同一历史 KV 会被后续多步复用，误差会持续影响 Attention。应覆盖长上下文、不同层和任务指标评估。

## Q17：并行策略

**1. 为什么 TP 太大可能被通信延迟限制？**

- TP 增大使每卡矩阵分片变小，单卡 kernel 效率下降，同时 collective rank 数和同步成本上升。通信不能被剩余计算隐藏时，增加 GPU 反而可能变慢。

**2. PP microbatch 怎样减少 bubble？**

- 多个 microbatches 可让不同 stage 同时处理不同批次，填充等待空洞；但过多 microbatches 会增加调度、通信和小 kernel 开销。最优值取决于 stage 均衡、网络、显存和总 batch。

## Q18：TP 通信

**1. Reduce-Scatter 后为什么可以接分片计算？**

- Reduce-Scatter 完成跨 rank 归约并让每 rank 保留一段结果；若后续算子能直接消费该分片 layout，就无需先 all-gather 完整 activation。前后切分维兼容性决定能否直连。

**2. latency-bound 与 bandwidth-bound collective 怎样区分？**

- 可用 `T≈α×steps+bytes/effective_bandwidth` 分解固定启动和传输项。小消息对调用/rank 数敏感、对 bytes 不敏感时偏 latency-bound；大消息时间近似随 bytes 线性增长时偏 bandwidth-bound，仍需结合拓扑和算法实测。

## Q19：MoE 数据流

**1. 为什么 MoE 参数多但每 token FLOP 不按总参数同比增长？**

- router 每 token 只激活 top-k experts，主要 expert FLOP 由 k 而非全部 experts 决定。代价是更大参数存储以及 routing、dispatch、通信和负载不均。

**2. grouped GEMM 为什么适合 experts？**

- 各 expert 获得不同数量 token，执行结构相似但权重不同的 GEMM。grouped GEMM 可一次描述多个动态 `M` 问题，减少小 kernel launch 并统一调度；收益依赖 token 分布、layout 和负载均衡。

## Q20：MoE 通信

**1. top-k 增大怎样影响通信量？**

- 每 token 被发送到更多 experts，dispatch/combine 元素量、expert 计算和合并成本通常增加；实际跨设备流量还取决于 expert placement，本地 expert 不一定经过网络。

**2. expert parallel 与 tensor parallel 怎样组合？**

- EP 在设备组间分 experts，通过 all-to-all dispatch/combine；TP 可继续切分单个 expert 内权重和计算。组合执行包含 EP 通信、expert 内 TP collective 和 EP combine，规模需结合网络层级和 token 数选择。

## Q21：推理框架

**1. 为什么功能支持不等于该组合性能成熟？**

- “能运行”可能依赖 fallback kernel、受限 shape 或额外 layout conversion。高性能要求模型、硬件、量化、scheduler 和 kernel 路径同时匹配；支持状态具有版本性，必须用具体版本和 profile 证明。

**2. 怎样设计公平 benchmark？**

- 对齐模型、精度/量化 recipe、硬件、输入输出长度、并发、max tokens、prefix cache 和 speculative decoding。统一 warmup 和请求集，同时报告 TTFT、TPOT、吞吐、尾延迟、显存和正确性，并披露 OOM/fallback/失败。

## Q22：DeepSeek 组件

**1. DeepGEMM grouped GEMM 与 MoE 怎样连接？**

- router/dispatch 先按 expert 重排 token，并生成各 expert 的 token 数、offset 或指针；grouped GEMM 把每个 expert 的 token 矩阵与对应权重作为一组动态 `M` 问题执行。连接效率取决于 layout 是否可直接消费。

**2. DeepEP 输出 layout 如何影响后续 expert GEMM？**

- 若同一 expert 的 token 连续且有准确 count/offset，GEMM 可直接建立 group 描述；若对齐、tile 或 scale 排布不匹配，就需重打包并增加带宽/同步。具体接口具有版本性，以目标版本代码和 profile 为准。

## Q23：显存容量

**1. 怎样从 SLA 反推最大并发 token？**

- 先在目标硬件/框架上扫描 active tokens 和 prefill/decode 比例，找到仍满足 TTFT、TPOT、P99 的负载边界；再映射为 KV blocks、请求数和预留容量，并加入流量波动、碎片和恢复安全余量。理论 KV 公式只是初始上限。

**2. 为什么 allocator reported free memory 不等于可分配最大连续块？**

- 空闲内存可能分散，allocator 还存在 reserved blocks、内部碎片和不同 pool；Graph、通信库、workspace 可能固定地址或预留空间。大分配取决于最大可用块和 allocator 策略，而非总 free bytes。

## Q24：性能评测

**1. goodput 与 raw throughput 有何区别？**

- raw throughput 统计全部生成 token，即使超 SLA、失败或质量不可接受；goodput 只统计满足延迟、正确性和服务约束的有效工作。过度排队可能提高 raw throughput 却降低 goodput。

**2. 为什么平均延迟会掩盖排队崩溃？**

- 大量低延迟请求会稀释少数极慢请求；接近容量上限后，小幅服务时间波动可能使队列迅速累积，P95/P99 先恶化。应报告分布、队列长度和 offered load 扫描，而非只报均值。

---

# 附录 B：项目深挖题参考答案

## 1. 从普通 Attention 推导 `m/l/O_acc`，解释为什么 exact

固定一个 query 行，普通 Attention 为：

$$
O=\frac{\sum_j e^{s_j}V_j}{\sum_j e^{s_j}}
$$

将 key 分块后，维护已处理元素的最大值 `m`、以该基准计算的指数和 `l=Σexp(s_j-m)`，以及向量分子 `O_acc=Σexp(s_j-m)V_j`。新块到来时令 `m_new=max(m,m_block)`，用 `exp(m-m_new)` 同时缩放旧 `l/O_acc`，再加入新块贡献。因为分子分母只是统一乘了同一个非零尺度，最终 `O_acc/l` 与全量 stable softmax 数学等价。这里的 exact 指算法等价，浮点加法顺序不同仍会造成微小误差。

## 2. 设计实验区分 FlashAttention 的 FLOP 收益与 IO 收益

固定 `B/H/Dh/dtype` 并扫描序列长度，对比数学等价的朴素 Attention 与 FlashAttention。先按 `QK^T` 和 `PV` 计算两者 dense 主导 FLOP，确认 FlashAttention 没把平方主项降成线性；再记录正常墙钟、HBM 读写 bytes、DRAM throughput、中间 tensor 峰值和 achieved FLOP/s。加入“不物化 `S/P` 但计算量近似”的融合基线，可进一步区分 launch 与 IO 收益。若 FLOP 近似而 HBM bytes、显存峰值和时间下降，证据支持收益主要来自 tiling、融合和片上复用。

## 3. 给定模型参数，计算 MHA/GQA 的 KV bytes

先确认 `L/B/N/Hq/Hkv/Dh/dtype_bytes`。基础容量为：

$$
2\times L\times B\times N\times H_{kv}\times D_h\times dtype\_bytes
$$

其中 2 代表 K/V；MHA 通常 `Hkv=Hq`，GQA 使用更小 `Hkv`。例如 `Hq=32,Hkv=8` 时，其他条件相同的 GQA 基础 KV bytes 约为 MHA 的四分之一。工程预算还要加 block metadata、对齐、最后一块浪费、prefix/beam 共享关系、workspace、碎片和安全余量；MLA 缓存压缩表示，不能直接套该公式。

## 4. 设计 continuous batching 调度器的输入、状态和目标函数

输入包括到达时间、prompt 长度、已生成 token、最大输出长度、优先级、SLA、采样状态和模型标识。状态至少包含 waiting/running 队列、prefill/decode 阶段、每请求 KV block table、空闲 blocks、每轮 token budget，以及 graph/shape bucket。每个 iteration 回收完成请求，在显存和 token budget 内选择 decode token 与 prefill chunks，并处理抢占、恢复和公平性。目标不能只有 tokens/s，应在容量约束下最大化 SLA goodput，同时限制 TTFT、TPOT、P99 和饥饿。

## 5. TP=8 性能不升反降时怎样拆通信、kernel 和调度时间？

先建立 TP=1/2/4/8 的同模型、同精度、同请求分布基线，分别记录每层 GEMM/Attention、collective、layout conversion、同步空洞和 CPU 调度时间。kernel 侧检查每卡矩阵是否切得过小、Tensor Core/occupancy 是否下降；通信侧按消息大小区分启动延迟与带宽，并看 collective 是否真与计算重叠；调度侧检查 batch、graph bucket、跨 rank shape 和 straggler。当前没有多 GPU 生产实测时，应明确这是诊断方案，不能宣称已定位某个真实系统根因。

## 6. MoE P99 抖动时怎样区分路由不均、all-to-all 和 expert GEMM？

每步记录各 expert token 数、最大/均值比、空 expert 数、跨节点流量，并把 pack、all-to-all、unpack、grouped GEMM、combine 分段计时。若 P99 与最大 expert load 强相关，先看 router skew、capacity、placement 和 padding；若分布稳定而通信尾部增长，则按消息大小、节点内外链路和 rank straggler 分析 all-to-all；若通信稳定而 GEMM 尾部随小 `M` 或 group 数变化，则检查 grouped GEMM 调度与 layout。当前无真实多卡 MoE 实测，应强调该证据设计而不是虚构结论。
