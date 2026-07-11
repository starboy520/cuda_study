# Superpowers 过程文档归档

这里保存已经完成或被当前计划取代的设计与实施记录。状态索引优先链接同主题的归档 spec、归档 plan 和现有公开交付；私有交付不提供链接。

## 已完成

### Tensor Core — `completed`

- [设计](specs/2026-06-29-week3-tensorcore-self-contained-design.md)
- [实施计划](plans/2026-06-29-week3-tensorcore-self-contained.md)
- [最终公开文档](../../courses/cuda/Week3_TensorCore学习文档.md)

### Attention — `completed`

- A100 学习资料：[设计](specs/2026-07-01-week4-attention-flashattention-a100-design.md) / [实施计划](plans/2026-07-01-week4-attention-flashattention-a100.md)
- numerator/denominator 改写：[设计](specs/2026-07-04-attention-numerator-denominator-design.md) / [实施计划](plans/2026-07-04-attention-numerator-denominator-rewrite.md)
- FlashAttention 教学改写：[设计](specs/2026-07-04-flashattention-teaching-rewrite-design.md) / [实施计划](plans/2026-07-04-flashattention-teaching-rewrite.md)
- [最终公开文档](../../courses/attention/Week4_Attention与FlashAttention完整学习资料.md)

### ML — `completed`

- [设计](specs/2026-07-03-ml-training-foundations-pytorch-design.md)
- [实施计划](plans/2026-07-03-ml-training-foundations-pytorch.md)
- 最终公开文档：[训练侧入门](../../courses/ml/ML基础_训练侧入门.md) / [零基础记忆卡](../../courses/ml/ML零基础记忆卡.md)

### Week5 — `completed`

- [设计](specs/2026-07-05-week5-inference-self-contained-rewrite-design.md)
- [实施计划](plans/2026-07-05-week5-inference-self-contained-rewrite.md)
- [最终公开文档](../../courses/inference/Week5增强版_LLM推理优化与decode.md)

### cuBLAS/CUTLASS — `completed`

- [设计](specs/2026-07-07-cublas-cutlass-interview-crash-course-design.md)
- [实施计划](plans/2026-07-07-cublas-cutlass-interview-crash-course.md)
- [最终公开文档](../../topics/gemm_tensorcore/cuBLAS与CUTLASS面试速成.md)

### Transformer 扩写 — `completed`

- [设计](specs/2026-07-07-week5-transformer-block-expansion-design.md)
- [实施计划](plans/2026-07-07-week5-transformer-block-expansion.md)
- [最终公开文档](../../courses/inference/Week5增强版_LLM推理优化与decode.md)

### CUDA / AI Infra 面试资料 — `completed`

- [设计](specs/2026-07-05-cuda-ai-infra-interview-guides-design.md)
- [实施计划](plans/2026-07-05-cuda-ai-infra-interview-guides.md)
- 最终公开文档：[CUDA 面试核心题库](../../interview/CUDA面试核心题库.md) / [AI Infra 面试核心题库](../../interview/AI_Infra面试核心题库.md)

### 简历套件 — `private delivery completed`

- [设计](specs/2026-07-08-cuda-ai-infra-resume-suite-design.md)
- [实施计划](plans/2026-07-08-cuda-ai-infra-resume-suite.md)

## 已被当前计划取代

### CUDA 深水区两周 — `superseded by 当前四周计划`

- [设计](specs/2026-07-05-cuda-deep-dive-two-week-design.md)
- [实施计划](plans/2026-07-05-cuda-deep-dive-two-week.md)
- [保留的公开专题文档](../../topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)
- [当前四周计划](../../../study_plan/四周聚焦计划_AIInfra与CUDA深水区.md)

### 14 天面试冲刺 — `superseded by 当前四周计划`

- [设计](specs/2026-07-08-cuda-interview-14-day-sprint-design.md)
- [实施计划](plans/2026-07-08-cuda-interview-14-day-sprint.md)
- [保留的历史验收计划](../../interview/CUDA工程师面试_14天突击计划.md)
- [当前四周计划](../../../study_plan/四周聚焦计划_AIInfra与CUDA深水区.md)

## 当前活跃过程文档

- GEMM 作品集重建：[设计](../../superpowers/specs/2026-07-10-gemm-portfolio-rebuild-design.md) / [实施计划](../../superpowers/plans/2026-07-10-gpu-kernel-engineering-gemm.md)
- Docs 与 Study Plan 整理：[设计](../../superpowers/specs/2026-07-10-docs-study-plan-reorganization-design.md) / [实施计划](../../superpowers/plans/2026-07-10-docs-study-plan-reorganization.md)