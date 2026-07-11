# CUDA / AI Infra 学习与实验仓库

这是私人 CUDA / AI Infra 学习与实验仓库，用于手写 kernel、正确性验证、性能分析和知识整理。当前实验环境为 NVIDIA A100 80GB（`sm_80`）。

## 当前状态

当前执行[四周聚焦路线](study_plan/四周聚焦计划_AIInfra与CUDA深水区.md)，唯一计划入口是 [study_plan/README.md](study_plan/README.md)。路线聚焦 GEMM、PTX/SASS 与 ncu、Attention、KV/PagedAttention 和面试转换，不以旧计划判断当前进度。

## 五个稳定入口

| 入口 | 用途 |
| --- | --- |
| [当前计划](study_plan/README.md) | 当前目标、冻结边界、四周计划与历史计划入口 |
| [知识目录](docs/README.md) | 按当前四周和技术主题检索课程、专题、面试与参考资料 |
| [长期深度教材](cuda_deep_course/README.md) | 不受四周冲刺节奏约束的系统 CUDA 教材与配套实验 |
| [实验代码](operator_practice/README.md) | 算子练习总入口；另见 [Tensor Core 实验](week06_tensorcore/README.md) 与 [Transpose 实验](week02_memory/transpose/README.md) |
| [当前 worklog](notes/week05.md) | 当前阶段的实验记录、数据、问题与结论 |

## 公开作品边界

本仓库是学习仓，允许保留探索过程、失败实验和阶段性笔记。独立公开作品仓完成并稳定后，再从这里提供固定链接；当前不把学习目录包装成已完成作品。

## 历史资料

历史 T4 环境资料与旧课程文档位于 [docs 归档](docs/archive/README.md)；旧八周路线及其他历史计划位于 [study plan 归档](study_plan/archive/README.md)。它们仅供检索，不构成当前执行主线。
