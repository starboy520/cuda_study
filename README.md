# CUDA / AI Infra 学习与实验仓库

这是私人 CUDA / AI Infra 学习与实验仓库，用于手写 kernel、正确性验证、性能分析和知识整理。当前实验环境为 NVIDIA A100 80GB（`sm_80`）。

## 当前状态

唯一计划入口是 [study_plan/README.md](study_plan/README.md)。Week 1 GEMM 已完成并发布；当前阶段为 **Week 2：PTX / SASS 与性能诊断**，执行清单见[本周计划](study_plan/current/Week2_PTX_SASS与性能诊断.md)。

## 五个稳定入口

| 入口 | 用途 |
| --- | --- |
| [当前计划](study_plan/README.md) | 当前目标、冻结边界、四周计划与历史计划入口 |
| [知识目录](docs/README.md) | 按当前四周和技术主题检索课程、专题、面试与参考资料 |
| [长期深度教材](cuda_deep_course/README.md) | 不受四周冲刺节奏约束的系统 CUDA 教材与配套实验 |
| [本周 Worklog](notes/week02_ptx_sass.md) | 记录 PTX/SASS、ptxas、ncu 和单变量诊断证据 |
| [本周主教材](docs/topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md) | 从编译链到 scheduler/scoreboard 的指定章节 |

## 公开作品边界

本仓库是学习仓，允许保留探索过程、失败实验和阶段性笔记。独立公开作品仓完成并稳定后，再从这里提供固定链接；当前不把学习目录包装成已完成作品。

## 历史资料

历史 T4 环境资料与旧课程文档位于 [docs 归档](docs/archive/README.md)；旧八周路线及其他历史计划位于 [study plan 归档](study_plan/archive/README.md)。它们仅供检索，不构成当前执行主线。
