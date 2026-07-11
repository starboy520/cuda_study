# CUDA / AI Infra 知识目录

> 当前执行入口是[四周聚焦计划](../study_plan/README.md)。本页只负责导航，不维护第二份进度表。

## 当前四周导航

| 阶段 | 核心材料 |
| --- | --- |
| Week 1：GEMM | [四周计划](../study_plan/四周聚焦计划_AIInfra与CUDA深水区.md)、[GEMM 优化阶梯](../week05_gemm_advanced/gemm_optimization_ladder.md)、[cuBLAS / CUTLASS 技术参考](topics/gemm_tensorcore/cuBLAS与CUTLASS面试速成.md) |
| Week 2：PTX/SASS + ncu + Tensor Core | [PTX/SASS/MMA 深水区](topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[Nsight Compute](topics/performance/Nsight_Compute_ncu详解.md)、[Tensor Core 学习文档](courses/cuda/Week3_TensorCore学习文档.md)、[Tensor Core 实验](../week06_tensorcore/README.md) |
| Week 3：Attention + Online proof | [Attention / FlashAttention 教材](courses/attention/Week4_Attention与FlashAttention完整学习资料.md)、[Online Softmax 正确性证明](proofs/Online_Softmax正确性证明.md)、[Attention pipeline 分析](../week04_attention/ncu_pipeline_notes.md) |
| Week 4：KV/Paged + 面试 | [KV Cache 专题](topics/kv_cache/README.md)、[PagedAttention](topics/kv_cache/PagedAttention详解.md)、[面试准备入口](interview/README.md) |

## 课程教材

- [CUDA 深度学习工程教材](../cuda_deep_course/README.md)：长期系统教材总入口。
- [教材卷目录](../cuda_deep_course/course/README.md)：按 GPU 基础、编程模型、内存、并行算法、性能和系统主题展开。
- [CUDA Programming Model 详解](courses/cuda/Programming_Model详解.md)：编程模型与执行层次。
- [ML 训练侧入门](courses/ml/ML基础_训练侧入门.md)：补充训练侧基础，不属于当前四周主线。

## 性能与底层

- [PTX/SASS/MMA/异步流水与 Hopper](topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)
- [Nsight Compute ncu 详解](topics/performance/Nsight_Compute_ncu详解.md)
- [Occupancy：从入门到调优](topics/performance/Occupancy详解_从入门到调优.md)
- [CUDA 核心原语场景驱动教程](topics/execution/CUDA核心原语_场景驱动教程.md)
- [Cooperative Groups 与 CUDA Graph](topics/execution/CooperativeGroups与CUDAGraph深度教程.md)

## GEMM / Tensor Core

- [GEMM 优化阶梯](../week05_gemm_advanced/gemm_optimization_ladder.md)与[实测记录](../week05_gemm_advanced/benchmark.md)
- [Week 3 Tensor Core 学习文档](courses/cuda/Week3_TensorCore学习文档.md)
- [Tensor Core 实验入口](../week06_tensorcore/README.md)
- [cuBLAS 与 CUTLASS 面试速成](topics/gemm_tensorcore/cuBLAS与CUTLASS面试速成.md)

## Attention / KV / 推理

- [Attention 与 FlashAttention 完整学习资料](courses/attention/Week4_Attention与FlashAttention完整学习资料.md)
- [Online Softmax 正确性证明](proofs/Online_Softmax正确性证明.md)
- [LLM 推理优化与 decode](courses/inference/Week5增强版_LLM推理优化与decode.md)
- [KV Cache 专题入口](topics/kv_cache/README.md)与[PagedAttention 详解](topics/kv_cache/PagedAttention详解.md)

## AI Infra / 多卡

- [AI Infra 面试核心题库](interview/AI_Infra面试核心题库.md)：连接 kernel、推理数据流、调度与系统设计。
- [HPC 与多 GPU 教材卷](../cuda_deep_course/course/volume08_hpc_multigpu/README.md)：长期教材入口。
- [MoE 与多卡并行](topics/distributed/MoE与多卡并行_系统学习.md)：**后续专题，当前四周冻结**；多卡实测同样不进入当前主线。

## 面试

- [面试准备入口](interview/README.md)
- [CUDA 面试核心题库](interview/CUDA面试核心题库.md)
- [AI Infra 面试核心题库](interview/AI_Infra面试核心题库.md)

## 参考

- [学习资料索引](reference/学习资料索引.md)
- [CUDA Programming Guide 学习路径](reference/Programming_Guide学习路径.md)
- [GPU 卡型专项学习指南](reference/GPU卡型专项学习指南.md)
- [GPU 架构图资源](reference/GPU架构图资源.md)
- [项目清单](reference/项目清单.md)

## 归档

[历史课程与指南归档](archive/README.md)保存旧路线、旧硬件资料、合并前原文和历史过程记录，仅供检索，不作为日常学习入口。
