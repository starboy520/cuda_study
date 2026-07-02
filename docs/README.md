# docs 文档导航

> 本目录学习文档索引。按「用途」分类，标注对应学习阶段。
> 外部资料（官方文档/书籍/课程）见 [学习资料索引.md](学习资料索引.md)。

---

## 🗺️ 计划与路线（先看这些）

| 文档 | 说明 |
|------|------|
| [DeepSeek_CUDA_2月冲刺计划.md](DeepSeek_CUDA_2月冲刺计划.md) | ⭐ 主计划，8 周训练营，每天代码+数据+口述 |
| [CUDA学习路线图.md](CUDA学习路线图.md) | 整体学习路线 |
| [学习资料索引.md](学习资料索引.md) | 外部资料（官方文档/书/课程/开源）索引 |
| [项目清单.md](项目清单.md) | 作品集项目清单 |

---

## 📅 每周学习材料（跟着计划走）

| 文档 | 阶段 | 说明 |
|------|------|------|
| [archive/Week1详细步骤.md](archive/Week1详细步骤.md) | Week1 | GPU 基础详细步骤（已归档） |
| [archive/Week1_Day2-Day5学习清单.md](archive/Week1_Day2-Day5学习清单.md) | Week1 | Day2-5 清单（已归档） |
| [Week3_TensorCore学习文档.md](Week3_TensorCore学习文档.md) | Week3 | ⭐ Tensor Core/WMMA 自包含手册 |
| [Week4_Attention与FlashAttention完整学习资料.md](Week4_Attention与FlashAttention完整学习资料.md) | Week4 | ⭐ Attention/FlashAttention 完整资料 |

> Week3 代码/笔记在 `week06_tensorcore/`；Week1-2 GEMM 在 `week05_gemm_advanced/`；算子练习在 `operator_practice/`。

---

## 🎯 面试准备

| 文档 | 说明 |
|------|------|
| [CUDA面试八股全集.md](CUDA面试八股全集.md) | ⭐ 15 章系统八股，闭卷口述粒度 |
| [CUDA面试完整准备指南.md](CUDA面试完整准备指南.md) | 面试准备策略 |

---

## 🔧 概念与原语参考（查阅用）

| 文档 | 说明 |
|------|------|
| [CUDA核心原语_场景驱动教程.md](CUDA核心原语_场景驱动教程.md) | 场景驱动：问题→代码→为什么（ballot/shuffle/atomicCAS/cp.async 等） |
| [异步拷贝_pipeline_cooperative_groups学习文档.md](异步拷贝_pipeline_cooperative_groups学习文档.md) | cp.async / pipeline / cooperative groups |
| [Programming_Model详解.md](Programming_Model详解.md) | CUDA 编程模型 |
| [Programming_Guide学习路径.md](Programming_Guide学习路径.md) | 官方 Programming Guide 阅读路径 |
| [Occupancy详解_从入门到调优.md](Occupancy详解_从入门到调优.md) | Occupancy 从入门到调优 |

---

## 📊 性能分析工具

| 文档 | 说明 |
|------|------|
| [Nsight_Compute_ncu详解.md](Nsight_Compute_ncu详解.md) | ncu 使用与指标解读 |

---

## 🖥️ 硬件与架构

| 文档 | 说明 |
|------|------|
| [GPU卡型专项学习指南.md](GPU卡型专项学习指南.md) | 各 GPU 卡型（T4/A100/H100 等） |
| [GPU架构图资源.md](GPU架构图资源.md) | 架构图资源 |
| [archive/T4实战指南.md](archive/T4实战指南.md) | T4 实战（已归档） |

---

## 🤖 LLM 系统（Week5+）

| 文档 | 说明 |
|------|------|
| [大模型KVCache系统学习指南.md](大模型KVCache系统学习指南.md) | KV cache 系统 |

---

## 🛠️ 自动生成的计划/设计（superpowers）

| 目录 | 说明 |
|------|------|
| [superpowers/plans/](superpowers/plans/) | Week3/Week4 的实施计划 |
| [superpowers/specs/](superpowers/specs/) | 对应的设计 spec |

---

## 📌 按学习阶段速查

```text
现在学 Tensor Core  → Week3_TensorCore学习文档.md
                      + week06_tensorcore/mixed_precision_and_scaling.md（精度/scaling）
准备学 Attention    → Week4_Attention与FlashAttention完整学习资料.md
查原语用法          → CUDA核心原语_场景驱动教程.md（学用法）
面试复习            → CUDA面试八股全集.md
调优/profiler       → Nsight_Compute_ncu详解.md + Occupancy详解.md
```
