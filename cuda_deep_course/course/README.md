# CUDA 深度学习教材

这套教材的目标不是“看懂一些 CUDA 代码”，而是逐步达到：

- 能独立设计、实现和验证 CUDA kernel。
- 能从线程、访存和硬件执行角度解释程序行为。
- 能使用工具定位正确性和性能问题。
- 能实现经典并行算法、核心算子和 HPC 程序。
- 能用代码、数据和清晰表达准备 CUDA 相关岗位面试。

## 学习方法

每章都按下面的闭环学习：

```text
阅读与推演
  -> 亲手实现
  -> 正确性验证
  -> 故意制造问题
  -> 使用工具观察
  -> 优化并记录数据
  -> 面试式复述
```

只读完正文不算完成一章。详细要求见：

- [术语与符号约定](术语与符号约定.md)
- [实验方法与完成标准](实验方法与完成标准.md)

## 十卷路线

| 卷 | 主题 | 状态 |
|---|---|---|
| 1 | CUDA 入门所需的 GPU 基础 | 已深度扩写 |
| 2 | CUDA 编程模型 | 已完成并深度扩写 |
| 3 | CUDA 内存系统 | 已完成 |
| 4 | 同步与经典并行算法 | 已完成 |
| 5 | 性能工程与 Profiling | 已完成 |
| 6 | 核心算子开发 | 规划完成 |
| 7 | 异步执行与 CUDA 系统能力 | 核心章节完成（含可运行实验） |
| 8 | HPC 与多 GPU | 规划完成 |
| 9 | GPU 硬件架构深入与代际演进 | 规划完成 |
| 10 | 工程化、作品集与面试 | 规划完成 |

完成十卷后，再进入 AI 推理优化专项。

## 当前入口

### 卷一

[CUDA 入门所需的 GPU 基础](volume01_gpu_basics/README.md)

- [ ] CPU 与 GPU 为什么不同
- [ ] GPU、SM、Warp、Thread
- [ ] 一维、二维、三维 Block 与线性编号
- [ ] 内存层次第一印象
- [ ] 延迟隐藏与大量线程
- [ ] T4 设备观察实验
- [ ] 卷一复习与面试题

### 卷二

[CUDA 编程模型](volume02_programming_model/README.md)

- [ ] 第一个完整 CUDA 程序
- [ ] Grid、Block、Thread 索引
- [ ] CUDA 函数修饰符与执行空间
- [ ] 内存分配、复制与资源生命周期
- [ ] 异步执行、同步与错误模型
- [ ] NVCC、PTX 与编译流程
- [ ] CUDA Event 与正确计时
- [ ] 二维矩阵加法 Sample
- [ ] Naive GEMM 完整推导
- [ ] 卷二复习、练习答案与面试题

### 卷三

[CUDA 内存系统](volume03_memory_system/README.md)

- [ ] CUDA 内存空间
- [ ] 合并访问、对齐、AoS 与 SoA
- [ ] Shared Memory、Tile 与 Bank Conflict
- [ ] Cache、Host 传输与 Unified Memory
- [ ] 矩阵转置完整实验
- [ ] 卷三复习与面试题

### 卷四

[同步与经典并行算法](volume04_parallel_algorithms/README.md)

- [ ] Race、同步与内存可见性
- [ ] Atomic 与 Warp 级原语
- [ ] Reduction 从错误到优化
- [ ] Scan 与 Histogram
- [ ] Convolution、Stencil 与 SpMV
- [ ] 数值正确性、复习与面试

### 卷五

[性能工程与 Profiling](volume05_performance/README.md)

- [ ] APOD 与可靠 Benchmark
- [ ] 性能指标、Scaling 与 Roofline
- [ ] Occupancy、分歧与延迟隐藏
- [ ] Nsight Systems 系统时间线
- [ ] Nsight Compute 与 Compute Sanitizer
- [ ] 完整优化案例、复习与面试

## 配套实验

实验代码统一放在 [`labs/`](../labs/)：

```text
labs/
├── common/
├── 01_gpu_basics/
├── 02_programming_model/
├── 03_memory_system/
└── 04_parallel_algorithms/
```

所有实验都要求：

1. 能独立构建。
2. 有明确的正确性检查。
3. 覆盖非整除和边界输入。
4. 给出可重复的运行命令。
5. 区分实测结果、理论值和估算值。

## 主要资料

教材以 NVIDIA 官方资料为事实依据，以并行算法和实验重新组织内容：

- CUDA Programming Guide
- CUDA C++ Best Practices Guide
- CUDA Runtime API
- CUDA Samples
- Programming Massively Parallel Processors
- NVIDIA 官方架构白皮书与技术资料

正文会给出资料映射，但不会机械翻译原文。
