# CUDA 学习计划总览(study_plan)

> 这是你整个 CUDA 学习的**计划中枢**。用这个项目(`cuda_study`)作为主线,8 周从入门到能面试。
> 本目录放**前进路线的周计划**;每周都有"学什么 / 动手 / 完成标准",照着做即可。
>
> 当前进度:**Week 1 已完成 ✓**,下一步 Week 2。

---

## 总路线(8 周 + 面试)

```text
Week 1  架构与编程模型          ✓ 已完成
Week 2  内存层次与正确性        → transpose、reduction、边界、合并访问、bank conflict
Week 3  并行模式与同步          → atomic、warp shuffle、scan、histogram、stream
Week 4  GEMM 优化 + ncu 入门    → naive→tiling→register、cuBLAS 标尺、Nsight 初识
Week 5  性能工程与 Nsight 实战  → Roofline、occupancy 调优、ncu/nsys 系统学
Week 6  核心算子开发            → softmax、layernorm、kernel 融合
Week 7  作品集项目             → 挑一个算子做到极致 + 技术报告
Week 8  面试冲刺               → 四大手写题默写 + 概念题 + 模拟面试
```

> 这条路线是 [CUDA学习路线图](../docs/CUDA学习路线图.md) 的执行版。Week 4 起按"GEMM 提前 +
> ncu 绑定"的调整版编排(原因见路线图说明)。

---

## 各周计划文档

| 周 | 主题 | 计划文档 | 状态 |
|---|---|---|---|
| 1 | 架构与编程模型 | [docs/Week1详细步骤](../docs/Week1详细步骤.md) · [每日清单](../docs/Week1_Day2-Day5学习清单.md) | ✓ 完成 |
| 2 | 内存层次与正确性 | [Week2详细步骤](Week2详细步骤.md) · [每日清单](Week2_Day1-Day7学习清单.md) | 待开始 |
| 2.5 | 卷一/二补缺(可选) | [Week2.5补缺学习计划](Week2.5_补缺学习计划.md) | 按需 |
| 3 | 并行模式与同步 | [Week3每日清单](Week3_Day1-Day7学习清单.md) | 待开始 |
| 4 | GEMM 优化 + ncu | [Week4每日清单](Week4_Day1-Day7学习清单.md) | 待开始 |
| 5 | 性能工程与 Nsight | [Week5_性能工程与Nsight实战](Week5_性能工程与Nsight实战.md) | 待开始 |
| 6 | 核心算子开发 | [Week6_核心算子开发](Week6_核心算子开发.md) | 待开始 |
| 7 | 作品集项目 | [Week7_作品集项目.md](Week7_作品集项目.md) | 待开始 |
| 8 | 面试冲刺 | [Week8_面试冲刺.md](Week8_面试冲刺.md) | 待开始 |

> Week 2-4 的详细计划也在本目录。Week 5-8 是后续路线。整个前进路线由本索引统一串起。

---

## 配套资源(贯穿全程)

```text
教材正文:   cuda_deep_course/course/        十卷,深入原理时回查
可运行实验: cuda_deep_course/labs/          配套 lab
你的代码:   week01_basics/ week02_memory/ … 自己写的练习
面试题库:   docs/CUDA面试完整准备指南.md     Week8 主要用,平时查
occupancy:  docs/Occupancy详解_从入门到调优.md  Week5 深入用
笔记:       notes/weekXX.md                 每周记录目标/结果/问题
```

---

## 怎么用这套计划

```text
1. 每周按 Day1→Day7 走,每天有明确的"学什么/动手/完成标准"
2. 核心纪律(沿用 Week1/2):
   - 每个概念都落到【自己手写的代码】上,不只看
   - 先正确(CPU reference + 边界测试)再性能
   - 每个优化讲清"问题→假设→改法→快了多少"
3. 每天在 notes/weekXX.md 记三件事:目标 / 实验结果 / 遇到的问题
4. 每周末自测 + 复盘,答不出的回对应卷补
5. 阅读标注:📖 精读 · 👀 扫读 · ✍️ 必须自己写 · ⏭️ 跳过
```

---

## 阶段性里程碑(判断"能不能进下一阶段")

```text
Week 1-3 完成后:能独立写正确的 kernel(含边界/同步),会基础优化
  → 具备初级 CUDA 岗笔试能力

Week 4-5 完成后:能做 GEMM 优化阶梯,会用 ncu 定位瓶颈、量化加速
  → 具备性能工程能力,面试区分度最高的部分

Week 6-7 完成后:能写数值稳定的融合算子,有一个打磨过的作品集项目
  → 具备算子开发岗核心竞争力

Week 8 完成后:四大手写题能默写、概念题能讲清、能讲项目
  → 可以开始投简历、约面试
```

---

## 目标岗位方向(影响 Week6-8 的侧重)

```text
算子开发岗(推荐,最热):  重 Week6 算子 + Week7 GEMM/算子项目
性能优化岗:             重 Week5 profiler + Week7 优化报告
通用 CUDA 岗:           Week1-5 打扎实即可,Week6-7 适度
推理/框架岗:            Week6 + 进阶(PyTorch extension/量化,见面试指南 §7)
```

> 还没定方向也没关系:Week1-5 是所有方向的公共基础,先走完再说。
