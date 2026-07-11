# 面试准备入口

本目录收拢面向 CUDA、AI Infra 和性能工程岗位的面试材料。使用时优先沿当前学习计划推进，再用题库和专题做闭卷复述、手写实现与性能证据补强。

## 核心题库与计划

- **CUDA 核心题库**：当前使用 [CUDA面试八股全集](../CUDA面试八股全集.md)；后续合并目标为“CUDA 面试核心题库”，暂不作为当前入口。
- **AI Infra 核心题库**：[AI_Infra面试核心题库](AI_Infra面试核心题库.md)，覆盖算子数据流、Attention、KV Cache、推理调度、量化、分布式与系统设计。
- **14 天历史计划**：[CUDA 工程师面试 14 天突击计划](CUDA工程师面试_14天突击计划.md)，用于回顾面试转换节奏、验收方式和历史训练安排，不替代当前主计划。

## 专题入口

| 方向 | 材料 |
| --- | --- |
| GEMM / Tensor Core | [cuBLAS 与 CUTLASS 面试速成](../topics/gemm_tensorcore/cuBLAS与CUTLASS面试速成.md)、[cuBLAS 函数速查](../topics/gemm_tensorcore/cuBLAS函数速查.md) |
| 执行模型与运行时 | [CUDA 核心原语](../topics/execution/CUDA核心原语_场景驱动教程.md)、[Cooperative Groups 与 CUDA Graph](../topics/execution/CooperativeGroups与CUDAGraph深度教程.md) |
| 性能分析 | [CUDA 深水区：PTX/SASS/MMA/异步流水与 Hopper](../topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[Nsight Compute](../topics/performance/Nsight_Compute_ncu详解.md)、[Occupancy](../topics/performance/Occupancy详解_从入门到调优.md) |
| KV Cache / decode | [KV Cache 专题入口](../topics/kv_cache/README.md)、[PagedAttention](../topics/kv_cache/PagedAttention详解.md) |
| 分布式与 MoE | [MoE 与多卡并行](../topics/distributed/MoE与多卡并行_系统学习.md) |
| 课程材料 | [Week 3 Tensor Core](../courses/cuda/Week3_TensorCore学习文档.md)、[Week 4 Attention / FlashAttention](../courses/attention/Week4_Attention与FlashAttention完整学习资料.md)、[Week 5 推理优化与 decode](../courses/inference/Week5增强版_LLM推理优化与decode.md) |

## 当前学习计划入口

当前执行入口是 [study_plan/README.md](../../study_plan/README.md)。面试材料的使用顺序应服从该入口中的当前阶段安排；本目录只负责提供题库、历史计划和专题索引，不复制另一份学习计划。

> 一句话记忆：**先按当前 study_plan 学和做，再用 AI Infra/CUDA 题库闭卷表达，最后用专题材料补性能、推理和分布式追问。**
