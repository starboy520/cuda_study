# CUDA 与 AI Infra 面试八股拆分重写设计

## 目标

将现有 `docs/CUDA面试八股全集.md` 从 329 行的速记摘要，重构为面向 CUDA 性能工程/Kernel 岗的分层面试题库；同时把其中 ML、LLM 与分布式系统内容迁移并扩写为独立的 `docs/AI_Infra面试八股全集.md`。

两份文档都采用“15 秒结论 → 1 分钟展开 → 追问 → 误区 → 项目证据”的回答结构，既能快速背诵，也能抵御深入追问。

## 岗位与难度分布

CUDA 文档按以下比例设计：

- B 必会基础约 45%：常规 CUDA/HPC 开发岗位必须掌握；
- A 性能主线约 45%：CUDA 性能工程、GPU Kernel 与 AI Infra Kernel 岗的核心竞争力；
- C 专家加分约 10%：CUTLASS/CuTe、warp specialization、persistent kernel 和 Hopper 深层追问，只展示知识上限，不伪装成已有完整工业实践。

AI Infra 文档使用同样的 B/A/C 标签，但比例由主题自然决定：基础数据流与容量计算为 B，性能/通信分析为 A，生产级框架与高级内核组织为 C。

## 统一题目模板

每道完整题目使用：

```markdown
### Q：问题

**难度：** B 必会 / A 主线 / C 加分

**15 秒回答**

可以直接说出口的结论。

**1 分钟展开**

原理、边界、性能影响和必要例子。

**面试官继续追问**

1. 逐级深入的问题。
2. 要求推导、判断或联系硬件的问题。

**常见误区**

- 容易说错、说绝对或混淆层级的表述。

**项目证据**

- 指向仓库中实际完成的源码、benchmark、ncu 记录或学习资料。
```

短小同类题可组成“快问快答组”，但答案仍需包含边界，不用单句口号代替技术解释。

## 文档一：CUDA 面试八股全集

### 文件

重写：`docs/CUDA面试八股全集.md`

### 定位

纯 CUDA、GPU 硬件、性能分析、调试与工程化题库。ML 算法、KV cache、Paged Attention、推理框架和 MoE 场景不留在本文件。

### 章节

1. 使用方法、分层标签和面试回答策略；
2. GPU 硬件与架构；
3. CUDA 执行模型、SIMT 与分歧；
4. 内存层次、地址空间与缓存；
5. Coalescing、向量化、shared bank 与 layout；
6. 同步、原子、memory fence 与一致性；
7. Occupancy、寄存器、spill、ILP 与 TLP；
8. Stream、Event、异步执行与 CUDA Graph；
9. Reduction、Scan、Histogram、Transpose 等并行模式；
10. GEMM 优化阶梯与性能账本；
11. Tensor Core、WMMA、`mma.sync` 与 `ldmatrix`；
12. `cp.async`、async group 与多级流水；
13. CUDA C++、PTX、ptxas、cubin、fatbin 与 SASS；
14. Nsight Systems、Nsight Compute、Roofline、scheduler、scoreboard 与 stall；
15. Compute Sanitizer、cuda-gdb 与并发正确性；
16. 多 GPU 基础：P2P、PCIe、NVLink、拓扑和 NCCL collective 机制；
17. Hopper：TMA、WGMMA、Thread Block Cluster、DSM 与 mbarrier；
18. 工程化：错误检查、RAII、测试、benchmark、性能回归、架构兼容；
19. 项目深挖题；
20. 高频手写题；
21. B/A/C 模拟面试题组与复习清单。

### 必须修正的原文风险

- 不笼统声称每个 SM 都有 4 个 scheduler、每周期各发一个 warp 指令；明确随架构和指令类型变化；
- 不把 `__threadfence()` 描述为自动让其他线程“看到”数据；区分 ordering、visibility、cache 和 synchronization；
- 不把 cooperative groups 等同于天然 grid sync；说明 cooperative launch 与驻留约束；
- 不写“occupancy 60–70% 常够”之类无法泛化的经验阈值；
- 不把 `float4` 等同于必然一条 128-bit SASS load；要求检查对齐、别名和反汇编；
- 不把 PTX 当作最终机器码；
- 不把 Long Scoreboard 直接等同于 DRAM 延迟；
- 不把 stage 越多、Tensor Core、CUDA Graph、Hopper 新特性描述成自动加速。

### 项目证据

优先链接：

- Week 1 基础与计时；
- Week 2 内存、转置、bank conflict、stream；
- Week 3 reduction/scan/histogram；
- Week 4/5 GEMM 与 A100 ncu 记录；
- Week 6 WMMA、HMMA 与 Tensor pipe 记录；
- 新增的 CUDA 深水区两周教材；
- cuda_deep_course 中 profiling、async、工程化和硬件章节。

只有有源码/数据/提交证据的内容才写作“我做过”；课程资料中出现但没有学习闭环的内容写作“我理解/可以分析”，不冒充实战。

## 文档二：AI Infra 面试八股全集

### 文件

新增：`docs/AI_Infra面试八股全集.md`

### 定位

模型计算数据流、推理系统、显存管理、量化、分布式并行、通信场景和生产推理框架题库。它可以引用 CUDA 文档中的硬件机制，但不重复解释 warp、shared memory、PTX 或 SASS 基础。

### 章节

1. 使用方法与 AI Infra 性能分析框架；
2. Tensor、shape、GEMM/GEMV 与算子数据流；
3. Softmax、Online Softmax 与数值稳定性；
4. Attention 与 FlashAttention；
5. Transformer、Prefill 与 Decode；
6. KV Cache、MQA、GQA 与 MLA；
7. Paged Attention、块表、碎片与显存管理；
8. 小 GEMM、GEMV、算子融合与 kernel launch；
9. FP16/BF16/TF32/FP8、INT8/INT4 与量化；
10. CUDA Graph 在推理循环中的使用；
11. Continuous Batching、调度、吞吐和延迟；
12. Data/Tensor/Pipeline/Context/Expert Parallel；
13. All-Reduce、All-Gather、Reduce-Scatter、All-to-All 的场景与通信量；
14. MoE routing、dispatch/combine、负载均衡与通信；
15. vLLM、TensorRT-LLM、SGLang 的职责边界；
16. DeepGEMM、FlashMLA、DeepEP、DualPipe 的问题定位；
17. 容量估算、性能建模与系统设计；
18. 项目深挖题、模拟面试和复习清单。

### 边界

- 只给当前项目确实覆盖过的概念合理的“项目证据”；
- 不声称所有 prefill 都 compute-bound、所有 decode 都 memory-bound，必须结合 batch、shape、量化、并行和硬件；
- KV cache 公式必须说明 MHA/MQA/GQA/MLA、层数、KV head、head dimension、dtype 和 batch/sequence 的变量；
- FlashAttention 必须区分 FLOP、显式中间存储和 HBM IO；
- 框架和开源项目属于时效性内容，写作时以官方文档/仓库为准；
- 量化结论要区分权重、激活、KV cache、训练和推理。

## 两份文档的交叉边界

| 主题 | CUDA 文档 | AI Infra 文档 |
|---|---|---|
| Tensor Core | 硬件、WMMA/PTX/SASS | 模型精度和吞吐场景 |
| CUDA Graph | API、capture/update、launch 开销 | decode replay 和调度限制 |
| NCCL | collective 机制、拓扑和 CUDA stream | 并行策略中的通信量和重叠 |
| GEMM/GEMV | kernel 与硬件优化 | shape、batch 和模型阶段 |
| Hopper | TMA/WGMMA/cluster/DSM | 对推理/训练内核的可能影响 |
| FP8/INT8 | 硬件格式与指令能力 | scaling、量化策略和模型误差 |

通过本地相对链接交叉引用，不复制长段解释。

## 表达与准确性

- 先给结论，再给条件和例外；
- 明确区分架构事实、API 语义、编译器观察、项目实测和工程经验；
- 架构规格、工具指标和当前软件/框架信息以官方资料为准；
- 代码只使用短小、能支撑面试解释的片段；完整教学代码链接到现有项目；
- 公式使用当前 Markdown 预览兼容的 `$...$` 与 `$$...$$`；
- 不使用无法在不同 GPU 上泛化的裸数字，必须带具体卡型或说明只是示例；
- 问题之间尽量避免重复，基础定义由后续题链接引用。

## 验收标准

### CUDA 文档

- 只保留纯 CUDA/GPU 主题；
- B/A/C 比例大致符合 45%/45%/10%；
- 覆盖基础、性能主线和专家加分层；
- 原文所有高风险绝对化表述得到修正；
- 至少覆盖执行、内存、同步、并行算法、GEMM、Tensor Core、异步、PTX/SASS、profiling、调试、工程、多 GPU 与 Hopper；
- 项目证据链接存在且表述与真实完成度相符。

### AI Infra 文档

- 完整接收原文的 ML/LLM 内容，并扩写为独立题库；
- 覆盖 Attention、推理、KV cache、Paged Attention、量化、调度、并行、通信、MoE、框架与系统设计；
- 与 CUDA 文档边界清楚，没有大段重复；
- 所有容量公式、性能判断和框架信息有条件限定。

### 通用

- 每个核心章节同时具有 15 秒回答、深入展开、追问和误区；
- 两份文档均有目录、使用方法、模拟题和复习清单；
- Markdown 标题、围栏、表格、公式和本地链接有效；
- 不修改用户现有源码和其他未提交文件；
- 只提交两份目标文档及本规格/实施计划，不纳入用户已经暂存的无关变更。
