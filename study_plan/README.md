# CUDA 学习计划

## 当前唯一执行计划

[AI Infra + CUDA 深水区四周聚焦学习计划](四周聚焦计划_AIInfra与CUDA深水区.md)

当前只执行这份计划，不按旧路线或目录编号推断进度。

配套知识地图与补缺验收手册：[CUDA 核心能力补缺学习文档](CUDA核心能力补缺学习文档.md)。该文档覆盖 Kernel 编程、Kernel 优化、编译与指令、执行调度、CUDA Graph、内存管理和性能工具，但不产生第二条并行日程。

## 当前阶段

**Week 2：PTX / SASS 与性能诊断**

- [本周独立执行清单](current/Week2_PTX_SASS与性能诊断.md)
- [本周 Worklog](../notes/week02_ptx_sass.md)

Week 1 GEMM 已完成首版、合并到公开作品仓 `main` 并推送；本周固定使用该项目作为实验对象，不再并行优化 GEMM。

## 当前目标

- 求职落点：AI Infra / Model Serving / 推理性能。
- 差异化能力：CUDA kernel 与性能工程。
- 产出标准：用可复现的正确性、性能数据和 PTX/SASS/ncu 证据支撑 GEMM、Attention 与推理系统叙事。

## 冻结主题

四周执行期间冻结 MoE、多卡通信实测、完整 vLLM/CUTLASS 源码阅读、Hopper 实现、生产级 PagedAttention 和新的零散算子收集。它们只进入后续候选，不形成并行主线。

## 配套入口

- [知识目录](../docs/README.md)：按四周与主题查资料。
- [长期深度教材](../cuda_deep_course/README.md)：不受当前冲刺节奏约束。
- [PTX/SASS 深水区教材](../docs/topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)：本周主教材。
- [Nsight Compute 详解](../docs/topics/performance/Nsight_Compute_ncu详解.md)：Day 6 按需查阅。
- [Tensor Core 学习文档](../docs/courses/cuda/Week3_TensorCore学习文档.md)：Day 7 按需查阅。

## 历史计划

旧八周路线及其他历史计划位于 [study plan 归档](archive/README.md)；旧课程、硬件与过程文档位于 [docs 归档](../docs/archive/README.md)。它们只用于检索，不构成第二条执行路线。
