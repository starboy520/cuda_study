# CUDA 学习计划

## 当前唯一执行计划

[AI Infra + CUDA 深水区四周聚焦学习计划](四周聚焦计划_AIInfra与CUDA深水区.md)

当前只执行这份计划，不按旧路线或目录编号推断进度。

## 当前目标

- 求职落点：AI Infra / Model Serving / 推理性能。
- 差异化能力：CUDA kernel 与性能工程。
- 产出标准：用可复现的正确性、性能数据和 PTX/SASS/ncu 证据支撑 GEMM、Attention 与推理系统叙事。

## 冻结主题

四周执行期间冻结 MoE、多卡通信实测、完整 vLLM/CUTLASS 源码阅读、Hopper 实现、生产级 PagedAttention 和新的零散算子收集。它们只进入后续候选，不形成并行主线。

## 配套入口

- [知识目录](../docs/README.md)：按四周与主题查资料。
- [长期深度教材](../cuda_deep_course/README.md)：不受当前冲刺节奏约束。
- [算子练习](../operator_practice/README.md)：实验与手写练习入口。
- [当前 worklog](../notes/week05.md)：记录实验数据、问题和结论。

## 历史计划

旧八周路线及其他历史计划位于 [study plan 归档](archive/README.md)；旧课程、硬件与过程文档位于 [docs 归档](../docs/archive/README.md)。它们只用于检索，不构成第二条执行路线。
