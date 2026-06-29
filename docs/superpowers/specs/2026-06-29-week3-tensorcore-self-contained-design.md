# Week 3 Tensor Core 自包含教材设计

## 目标

将 `docs/Week3_TensorCore学习文档.md` 重写为一份可独立完成 Week 3 学习和实验的教材。学习者无需阅读其他仓库文档或外部资料即可理解主线、运行示例、完成独立编码、验证性能并进行面试复盘。

外部链接只作为可选延伸和权威出处，不承担主线教学内容。

## 读者与硬件边界

- 读者已经掌握 CUDA 基础、shared-memory GEMM 和 register tiling。
- T4（SM 7.5）用于 FP16 Tensor Core 与 WMMA 实验。
- A100（SM 8.0）用于 FP16、BF16、TF32 对照实验。
- FP8 以数值原理、scaling 和 DeepGEMM 工程思路为主，不要求在 T4/A100 上手写 FP8 Tensor Core kernel。

## 文档结构

1. 本周目标、产出、硬件边界和使用方法。
2. 从标量 FMA 到 warp-level MMA，解释 Tensor Core 为什么快。
3. FP32、TF32、FP16、BF16、FP8、INT8 的格式、范围、精度、用途和风险对照。
4. WMMA 的抽象层级、fragment、warp 协作、布局、leading dimension、对齐和一致控制流约束。
5. 最小单-warp WMMA Demo，用于逐行理解 API 和数据流。
6. 可编译的多-tile FP16 GEMM Demo，包含 host 端初始化、启动、正确性检查、计时和 GFLOPS。
7. 独立重写训练：关掉完整答案，根据骨架、阶段检查点和常见错误自行完成 kernel。
8. cuBLAS 对照、benchmark 方法、误差标准和结果表格模板。
9. Nsight Compute 分析方法，包括命令、关键指标、指令证据和预期观察。
10. T4 与 A100 的实验差异，以及 TF32/BF16 的正确使用边界。
11. FP8 E4M3/E5M2、amax、per-tensor/per-block scaling、溢出与误差管理。
12. DeepGEMM、grouped GEMM、JIT、MoE 和 scaling layout 的阅读框架。
13. Day 1–7 执行清单、每日产出、面试口述和最终验收。
14. 可选权威链接。

## Demo 与独立练习的关系

完整代码定位为教学 Demo，不替代独立练习：

1. 先逐行读懂并编译运行 Demo。
2. 修改矩阵规模、布局或 block 配置，确认理解数据映射。
3. 关掉完整实现，仅看接口和检查点，独立重写核心 kernel。
4. 与 Demo 对照定位差异，再完成 cuBLAS 校验和 profiling。

文档会明确标记“理解用 Demo”和“必须独立完成”的边界。

## 正确性与技术口径

- 不把一次 WMMA API 调用简单等同为一条固定的底层 16×16×16 机器指令。
- BF16 描述为具有接近 FP32 的指数范围、比 FP16 更不易溢出，但仍存在溢出与舍入误差。
- TF32 是否启用必须绑定具体库、API 和 math mode；普通 FP32 CUDA kernel 不会自动使用 Tensor Core。
- fragment 的 lane/register 映射视为不透明实现细节，不进行非可移植索引。
- 所有 warp-level WMMA 操作必须强调全 warp 参与和一致控制流。
- 性能比较同时记录输入精度、累加精度、矩阵规模、GPU、编译参数和误差，避免不公平比较。

## 完成标准

- 文档中的代码块在对应架构上具有明确的编译命令，并应通过静态检查或实际编译验证。
- 学习者可以只依据本文完成 FP16 WMMA GEMM、正确性校验、benchmark 和 NCU 分析。
- 文档覆盖冲刺计划 Week 3 的全部产出和验收问题。
- 文档没有依赖其他本地章节才能理解的必读跳转。
- 外链失效不会破坏学习主线。

## 非目标

- 不实现 CUTLASS 级多级流水 Tensor Core GEMM。
- 不深入讲解全部 PTX `mma.sync` 形状或 Hopper WGMMA 编程。
- 不在不支持原生高性能 FP8 Tensor Core 的 T4/A100 上强行安排 FP8 kernel 实现。
- 不复制整章官方文档或 DeepGEMM 源码。
