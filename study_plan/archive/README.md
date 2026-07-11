# 旧学习计划归档

## 内容

这里保存原来的八周 CUDA 学习路线及其 Week 2.5 补缺桥接计划。它们记录了当时的学习顺序、实验安排和验收方式，适合查找历史实验或阅读映射。

## 历史假设

旧路线以 NVIDIA T4（Compute Capability 7.5、40 SM、16 GB 显存）为主要实验硬件，部分带宽、shared memory 和性能预期都沿用了 T4 的数字。当前环境与目标不应再由这些假设决定。

## 当前替代计划

当前唯一执行入口是 [四周聚焦计划](../四周聚焦计划_AIInfra与CUDA深水区.md)：面向 AI Infra / Model Serving，聚焦 GEMM、PTX/SASS 与 Nsight Compute、FlashAttention、PagedAttention 学习实验和面试转换。

## 使用规则

1. 旧计划只用于查实验步骤、概念顺序和历史阅读映射，不用于判断当前进度。
2. 需要复用旧实验时，先按当前硬件和当前计划重新确认编译架构、资源限制与性能基线。
3. 不从归档中恢复第二条主线；新增任务必须先符合当前四周计划的冻结边界。

## 归档目录

- [Week 1](legacy-8-week/week01/Week1详细步骤.md)：架构与编程模型。
- [Week 2](legacy-8-week/week02/Week2详细步骤.md)：内存层次与正确性。
- [Week 2.5](legacy-8-week/bridge/Week2.5_补缺学习计划.md)：卷一/卷二补缺桥接。
- [Week 3–8](legacy-8-week/)：并行模式、GEMM、性能工程、算子、作品集和面试路线。