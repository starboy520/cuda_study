# CUDA 系统学习项目

面向 **CUDA 相关工作岗位** 的 1–2 个月系统学习路线，配合 **Tesla T4 (16GB)** 实战。

## 深度教材入口

新的长期学习主线已经开始编写：

**[CUDA 深度学习教材](cuda_deep_course/README.md)**

这套教材不按固定周数压缩，以 GPU 算子开发和 HPC 为主要方向。每章包含
直觉解释、正式原理、手写代码、边界验证、故障注入、性能工具和面试复述。

## 快速开始

```bash
# 验证环境
nvidia-smi
nvcc --version

# 编译第一个程序（学完 Week 1 后）
cd week01_basics/vec_add
make && ./vec_add
```

## 学习笔记

| 笔记 | 说明 |
|------|------|
| [notes/week01.md](notes/week01.md) | Week 1 实验记录、性能表 |
| [notes/CUDA基础概念.md](notes/CUDA基础概念.md) | 概念速查（Grid/Block/SM/Warp 等） |
| [docs/Programming_Model详解.md](docs/Programming_Model详解.md) | **官方 Programming Model 易懂版**（v13.x Part 1–2，配合 vec_add） |

## 文档导航

| 文档 | 说明 |
|------|------|
| [study_plan/](study_plan/README.md) | **学习计划中枢**：8 周周计划(Week1-8)统一入口,从入门到面试 |
| [CUDA学习路线图.md](docs/CUDA学习路线图.md) | **主文档**：8 周课程、知识点、项目、面试方向（v13.x 阅读章节） |
| [CUDA面试完整准备指南.md](docs/CUDA面试完整准备指南.md) | **面试题库**：知识地图、50 概念题、手写 kernel、性能/系统设计、进阶主题 |
| [Occupancy详解_从入门到调优.md](docs/Occupancy详解_从入门到调优.md) | Occupancy 完整文档：基础(延迟隐藏)+ 进阶(精确计算/调优/profiler) |
| [Programming_Model详解.md](docs/Programming_Model详解.md) | Programming Model 白话详解（配合 v13.x Part 1–2） |
| [GPU架构图资源.md](docs/GPU架构图资源.md) | 官方架构图、Memory Hierarchy 等配图链接 |
| [Week1详细步骤.md](docs/Week1详细步骤.md) | **Week 1 逐步清单**（含 v13.x 阅读章节 + Legacy 对照） |
| [Week1_Day2-Day5学习清单.md](docs/Week1_Day2-Day5学习清单.md) | Week 1 **Day1–Day7** 可勾选清单（v13.x 对齐） |
| [学习资料索引.md](docs/学习资料索引.md) | 书籍、官方文档、课程、论文、工具链接 |
| [GPU卡型专项学习指南.md](docs/GPU卡型专项学习指南.md) | T4/A100 差异、卡相关学习内容、分阶段清单 |
| [T4实战指南.md](docs/T4实战指南.md) | T4 硬件特性、性能预期、实验建议 |
| [项目清单.md](docs/项目清单.md) | 12 个递进项目 + 作品集建议 |

## 学习节奏建议

- **全职学习（8 周）**：每天 6–8 小时，按主文档 Week 1–8 顺序推进
- **业余学习（8–10 周）**：每天 3–4 小时，Week 5–6 可合并部分章节
- **冲刺模式（4 周）**：有 C++/系统基础时，可压缩 Week 1–4，重点放在 Week 5–8 优化与项目

## 目录结构（随学习进度创建）

```
cuda_study/
├── docs/                    # 学习文档
├── week01_basics/           # 基础：向量加、矩阵乘
├── week02_memory/           # 内存：transpose、reduction
├── week03_advanced/         # 流、事件、统一内存
├── week04_libraries/        # cuBLAS、Thrust
├── week05_optimization/     # 优化：GEMM、Roofline
├── week06_profiling/        # Nsight 性能分析
├── week07_applications/     # 应用：图像、推理
└── week08_portfolio/        # 作品集项目
```

## 学习目标（2 个月后应达到）

- [ ] 理解 GPU 架构与 CUDA 编程模型，能独立写出正确且可调试的 kernel
- [ ] 掌握内存层次、合并访问、Bank Conflict、Occupancy 等优化手段
- [ ] 会用 Nsight Systems / Nsight Compute 定位瓶颈并量化优化效果
- [ ] 熟悉 cuBLAS / Thrust，了解 cuDNN、CUTLASS、TensorRT 生态
- [ ] 完成 3+ 个可展示的项目（含性能对比数据与 README）
- [ ] 能回答 CUDA 岗位常见面试题（内存、同步、warp、性能分析）

---

**下一步**：阅读 [docs/CUDA学习路线图.md](docs/CUDA学习路线图.md) 从 Week 1 开始。
