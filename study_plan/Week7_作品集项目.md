# Week 7 作品集项目(Day1–Day7)

> **主题**:把前六周的能力打包成一个**可展示、有说服力的作品集项目**——面试时最硬的底气。
> **为什么这周**:简历写"精通 CUDA"没人信;一个有代码、有测试、有性能数据、有技术报告的
> 项目,能让面试官 5 分钟内判断你的真实水平。
> **前置**:Week1-6 完成(尤其 Week4 GEMM、Week5 profiler、Week6 算子)。
> **本周交付**:一个打磨到位的 GitHub 项目 + 技术报告 + 3 分钟讲解稿。

---

## 使用方式

本周不是"学新知识",而是"把已有能力做深做透成一个项目"。选一个项目方向,一周做到极致,
质量 > 数量。每天在 `notes/week07.md` 记进展。

---

## 选一个主打项目(三选一)

| 项目 | 难度 | 适合方向 | 用到 |
|---|---|---|---|
| **P-GEMM**:GEMM 优化与 cuBLAS 对标 | ⭐⭐⭐ | 性能/通用岗 | Week4-5 |
| **P-算子**:融合算子(softmax/layernorm)| ⭐⭐⭐ | 算子开发岗(推荐)| Week6 |
| **P-Reduce**:高性能归约库 | ⭐⭐ | 通用岗 | Week3-5 |

> 推荐 **P-GEMM 或 P-算子**(面试硬通货)。时间够可以做两个,但**一个做到极致 > 两个半成品**。

---

## 本周总览

| Day | 任务 |
|---|---|
| 1 | 选题 + 搭工程骨架(CMake + 目录结构) |
| 2 | 实现核心 + 完整测试体系(CPU reference + 边界 + sanitizer) |
| 3 | 优化阶梯 + 每步 profiler 验证 |
| 4 | 对标(cuBLAS/cuDNN/PyTorch)+ 性能表 |
| 5 | 写技术报告(README) |
| 6 | 打磨:代码质量、RAII、错误检查、可复现 |
| 7 | 录 3 分钟讲解 + 推 GitHub |

---

## 现成资料(直接照着做)

| 主题 | 文档 |
|---|---|
| 作品集模板与标准 | `cuda_deep_course/course/volume10_engineering_interview/10_作品集_三个项目与技术报告.md` |
| CMake 工程 | `.../volume10_engineering_interview/01_CMake与可复现构建.md` |
| RAII/错误包装 | `.../volume10_engineering_interview/02_错误包装_RAII与资源生命周期.md` |
| 测试体系 | `.../volume10_engineering_interview/03_测试体系_CPU参考_随机边界_误差容忍.md` |
| 性能基线 | `.../volume10_engineering_interview/04_性能基线与回归测试.md` |
| Code review 清单 | `.../volume10_engineering_interview/06_KernelCodeReview清单.md` |

---

## Day 1:选题 + 工程骨架

**动手** ✍️ `week07_project/`
```text
project/
├── CMakeLists.txt            # 可复现构建(卷十/01)
├── include/                  # 头文件
├── src/                      # kernel 实现(多版本)
├── tests/                    # 测试
├── bench/                    # 性能测量
└── README.md                 # 技术报告(逐步写)
```
1. 确定项目方向和目标(性能目标、对标对象)。
2. 搭好 CMake,`cmake -S . -B build && cmake --build build` 跑通一个空壳。

**完成标准**
- [ ] 项目能一键构建
- [ ] 目录结构清晰(实现/测试/bench 分开)

---

## Day 2:核心实现 + 测试体系

**动手** ✍️
1. 实现核心 kernel(先 naive 正确版)。
2. 建完整测试(卷十/03):CPU double reference + 多规模(含边界 n=1/31/257/非整除)+ 容差比较。
3. 跑 `compute-sanitizer memcheck/racecheck` 确认干净。

**完成标准**
- [ ] naive 版在所有规模 PASS
- [ ] sanitizer 无报错

---

## Day 3:优化阶梯 + profiler 验证

**动手** ✍️
1. 实现优化阶梯(如 GEMM:naive→tiled→register;算子:朴素→稳定→融合)。
2. 每一步用 ncu 验证(卷五/05):SoL 怎么变、哪个指标证明了优化有效。
3. 记录每版的时间和关键指标。

**完成标准**
- [ ] 每步优化都有"动机 + 量化收益 + profiler 证据"
- [ ] 有一张各版本性能对比表

---

## Day 4:对标 + 性能表

**动手** ✍️
1. 和工业库对比(cuBLAS/cuDNN/PyTorch)。
2. 做一张完整性能表:各版本 + 库的 GFLOPS/带宽/占库百分比。
3. 分析差距来源(向量化?双缓冲?Tensor Core?)。

**完成标准**
- [ ] 知道自己到了库的百分之几
- [ ] 能解释差距在哪

---

## Day 5:技术报告

**动手** ✍️ 按卷十/10 的模板写 `README.md`:
```text
1. 问题与目标
2. 环境(GPU/CUDA/编译参数,可复现关键)
3. 方法与优化阶梯(每步:动机 + 实现 + 解决什么瓶颈)
4. 正确性验证(reference/规模/容差/sanitizer)
5. 性能结果(对比表 + profiler 证据)
6. 分析与取舍(瓶颈、还差多少、差在哪)
7. 结论与未来工作
```

**完成标准**
- [ ] README 完整,§3 和 §5 有"动机→数据"的闭环
- [ ] 别人能照着 README 复现

---

## Day 6:打磨

**动手** ✍️ 对照卷十/06 code review 清单:
1. 所有 CUDA 调用 `CUDA_CHECK`;资源用 RAII 管理。
2. 边界判断、同步位置、合并访问都查一遍。
3. 去掉调试残留(printf、-G、热路径 deviceSync)。
4. 代码可读性:magic number 抽常量、注释解释"为什么"。

**完成标准**
- [ ] 过一遍 code review 清单,无明显隐患
- [ ] 代码整洁、可读

---

## Day 7:讲解 + 发布

**动手**
1. 录一段 **3 分钟讲解**(卷十/10 结构):
```text
10秒 这是什么、解决什么问题
30秒 最简基线和瓶颈
90秒 优化阶梯:每步动机 + 量化收益(核心,展示思维)
30秒 最终结果 + 对标 + 还差什么
20秒 学到什么/取舍
```
2. 推上 GitHub,README 作为门面。
3. 自听录音,改到流畅。

**完成标准**
- [ ] 项目推上 GitHub,README 完整
- [ ] 3 分钟讲解流畅,讲"问题→假设→改法→数据"而非罗列技术

---

## 本周交付清单

```text
week07_project/(或独立 repo)
  完整项目:多版本实现 + 测试 + bench + 技术报告
GitHub:已发布,README 是门面
讲解稿:3 分钟,录音过关
notes/week07.md:项目日志
```

> 完成这个项目 = 你有了面试时拿得出手的实证。它比任何"我会 XXX"的口头描述都有力。

---

**返回**:[study_plan/README.md](README.md) · 上一步:[Week6 核心算子](Week6_核心算子开发.md) · 下一步:[Week8 面试冲刺](Week8_面试冲刺.md)
