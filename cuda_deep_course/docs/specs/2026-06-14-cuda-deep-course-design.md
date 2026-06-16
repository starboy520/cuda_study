# CUDA 深度学习教材设计

日期：2026-06-14

## 1. 教材目标

本教材面向希望长期从事 GPU 算子开发、CUDA 性能工程和 HPC 的学习者。
不以固定周数或快速入门为约束，而以以下能力作为完成标准：

1. 能从硬件执行模型解释 CUDA 程序的行为。
2. 能独立编写正确的 CUDA kernel，并系统处理边界、同步和数值误差。
3. 能实现和分析经典并行算法与常见核心算子。
4. 能使用 Nsight Systems、Nsight Compute 和 Compute Sanitizer 定位问题。
5. 能用带宽、吞吐、算术强度、occupancy 和 Roofline 解释性能。
6. 能完成单 GPU 与多 GPU 的工程实验。
7. 能清楚回答 CUDA 岗位面试问题，并完成常见手写题。
8. 能在完成 CUDA 主线后自然衔接 AI 推理优化。

## 2. 读者前提

教材默认读者具备：

- C/C++ 基础语法、指针、数组、函数和基本面向对象知识。
- Linux 命令行和基本编译经验。
- 基础线性代数和计算机组成知识。

教材不假设读者已经理解 GPU 架构、并行算法或性能优化。涉及必要的 C++、
数学、浮点和体系结构知识时，在对应章节内补充。

## 3. 教学原则

### 3.1 先建立直觉，再给正式模型

每个复杂概念先使用图、具体数字和小规模例子解释，再引入正式术语、公式、
API 和硬件限制。

### 3.2 正确性先于性能

每个算法遵循：

```text
CPU 参考实现
  -> 最简单的正确 CUDA 实现
  -> 自动正确性验证
  -> 性能测量
  -> 分阶段优化
  -> Nsight 证据
```

不使用“看起来更快”作为优化结论。

### 3.3 从代码追踪到硬件

优化规则必须回答：

- 哪些线程在执行？
- 一个 warp 访问了哪些地址？
- 数据位于哪一级存储？
- 使用了哪些执行管线？
- 延迟靠什么隐藏？
- 瓶颈证据是什么？

### 3.4 稳定原理与代际特性分开

跨代稳定的执行、存储和并行原理作为主干。Turing、Ampere、Hopper、
Blackwell、Vera Rubin 的特性作为架构对比，不把特定产品规格误写成
CUDA 的永久规则。

### 3.5 不机械翻译资料

教材根据学习依赖重新组织官方资料和 PMPP 内容，使用原创讲解、图示、代码
和实验。官方文档作为行为定义与事实依据，不按原章节逐段翻译。

## 4. 权威资料与角色

### 4.1 CUDA Programming Guide

作为以下内容的首要依据：

- CUDA 编程与执行模型
- CUDA C++ 语言扩展
- 内存模型和同步语义
- 异步执行与高级特性
- Compute Capability 和架构能力

### 4.2 CUDA C++ Best Practices Guide

作为以下内容的首要依据：

- APOD 优化循环
- 正确性验证和性能测量
- 内存、执行配置、指令和控制流优化
- 部署与兼容性建议

### 4.3 Programming Massively Parallel Processors

作为以下内容的重要教学参考：

- 并行思维方式
- 数据并行算法分解
- convolution、reduction、scan、histogram、stencil、sparse 等算法
- 从 naive 到优化实现的教学顺序

### 4.4 CUDA Samples

作为 API 用法和可运行案例参考。教材只选择能服务章节目标的 sample，并对
代码进行重新解释和必要简化。

### 4.5 NVIDIA 架构资料

使用 NVIDIA 官方架构白皮书、技术博客和产品文档学习 Turing、Ampere、
Hopper、Blackwell 与 Vera Rubin。尚未公开的微架构细节不使用社区传闻补全。

## 5. 十卷总体结构

### 卷一：CUDA 入门所需的 GPU 基础

目标：只学习编写和理解第一批 CUDA 程序所必需的硬件直觉。

主要内容：

1. CPU 与 GPU 设计目标。
2. GPU、SM、Warp、Thread 的基本关系。
3. Warp 包含 32 个线程及其最基本的执行直觉。
4. Register、shared memory、global memory 的初步区别。
5. GPU 为什么需要大量线程隐藏延迟。
6. T4 设备属性查询和基础观察实验。

### 卷二：CUDA 编程模型

目标：能独立完成从 Host 到 Device 的正确程序。

主要内容：

1. 异构计算和 Host/Device 分工。
2. CUDA Runtime、Driver 和 Toolkit。
3. Kernel、Grid、Block、Thread。
4. 1D、2D、3D 索引与 grid-stride loop。
5. Block 形状、tile 形状和 warp 组织。
6. `dim3`、边界判断和非方阵问题。
7. `cudaMalloc`、`cudaMemcpy`、kernel launch 和释放。
8. 同步与异步的第一层认识。
9. 错误检查、launch error 和 runtime error。
10. NVCC、PTX、SASS、fat binary 和架构目标。
11. CUDA Event 正确计时。
12. deviceQuery、vector add、naive GEMM 综合实验。

### 卷三：CUDA 内存系统

目标：能根据线程访问模式判断数据流动效率。

主要内容：

1. 寄存器、local、shared、global、constant、texture memory。
2. 内存事务、cache line、对齐和合并访问。
3. 行主序、列主序、pitch 和多维数据。
4. AoS 与 SoA。
5. Shared memory 生命周期和 tile。
6. Bank、bank conflict 和 padding。
7. L1/L2 cache 与数据复用。
8. Pinned、mapped、zero-copy 和 Host-Device 传输。
9. Unified Memory 的基础定位。
10. Copy、transpose、tiled matrix multiply 实验。

### 卷四：同步与经典并行算法

目标：建立并行算法设计能力，而不只会套 kernel 模板。

主要内容：

1. Race condition、可见性、顺序和同步范围。
2. `__syncthreads`、`__syncwarp`、fence 和 cooperative groups。
3. Atomic 操作、竞争和聚合。
4. Warp vote、shuffle、match 和 active mask。
5. Reduction：global、shared、warp、分层归约。
6. Prefix sum：Hillis-Steele 与 Blelloch。
7. Histogram：global atomic、privatization 和合并。
8. Convolution 与 stencil：halo、tile 和边界。
9. Sparse matrix-vector multiplication。
10. 排序、compact 和并行队列的基础思想。
11. 正确性、确定性和浮点归约误差。

### 卷五：性能工程与 Profiling

目标：能够用测量证据完成性能优化闭环。

主要内容：

1. APOD：Assess、Parallelize、Optimize、Deploy。
2. 正确 benchmark：warmup、重复、median、同步和噪声。
3. 延迟、吞吐、GFLOPS、有效带宽和 speedup。
4. Amdahl、Gustafson、strong scaling 和 weak scaling。
5. Arithmetic intensity 和 Roofline。
6. Occupancy、active warp、寄存器压力和 shared-memory 限制。
7. Warp divergence、指令吞吐和延迟隐藏。
8. Nsight Systems 时间线分析。
9. Nsight Compute 指标、section 和 source correlation。
10. Compute Sanitizer、racecheck 和 initcheck。
11. 从 profile 到单一优化假设的完整案例。

### 卷六：核心算子开发

目标：掌握真实算子的分块、数据复用、数值稳定和优化方法。

主要内容：

1. GEMM 数学、布局和性能上限。
2. Naive GEMM、shared-memory tiling、register tiling。
3. 向量化加载、双缓冲和软件流水。
4. WMMA、Tensor Core 和混合精度。
5. Convolution 的直接、im2col 和 tiled 实现。
6. Softmax 的数值稳定与多级归约。
7. LayerNorm、RMSNorm 和融合思维。
8. Elementwise、broadcast 和 reduction fusion。
9. cuBLAS、cuDNN、cuSPARSE 与手写 kernel 的边界。
10. CUTLASS 的层次化设计思想。
11. 算子性能报告与代码审查。

### 卷七：异步执行与 CUDA 系统能力

目标：从单 kernel 优化扩展到端到端流水线优化。

主要内容：

1. Host 和 Device 异步语义。
2. Default stream、non-default stream 和 event。
3. `cudaMemcpyAsync`、pinned memory 和 overlap。
4. 多流分块流水线。
5. Concurrent kernels 和 stream priority。
6. Stream-ordered allocator 与 `cudaMallocAsync`。
7. CUDA Graph capture、instantiate、update 和 replay。
8. Unified Memory、prefetch 和 advice。
9. Device-side asynchronous copy、barrier 和 pipeline。
10. 架构相关的 `cp.async`、TMA、cluster 和 DSM 概念。
11. 端到端延迟和吞吐分析。

### 卷八：HPC 与多 GPU

目标：掌握数值程序和多 GPU 系统的基本设计与分析。

主要内容：

1. IEEE-754、FMA、舍入、误差传播和数值稳定性。
2. 可复现归约和精度性能权衡。
3. cuBLAS、cuFFT、cuSPARSE、cuSOLVER 基础。
4. GPU 拓扑、PCIe、NVLink 和 P2P。
5. 多 GPU 数据划分、负载均衡和同步。
6. NCCL collective：broadcast、reduce、all-reduce、all-gather。
7. 计算与通信重叠。
8. MPI 与 CUDA-aware MPI 的概念和实验路线。
9. Strong/weak scaling 与扩展效率。
10. Multi-GPU reduction、stencil 或 GEMM 项目。

### 卷九：GPU 硬件架构深入与代际演进

目标：在掌握 CUDA 编程、算法和性能分析后，系统打通软件与硬件。

主要内容：

1. GPU die、GPC、TPC、SM 的逻辑层次。
2. Warp scheduler、dispatch、scoreboard、issue、latency 和 throughput。
3. CUDA Core、Tensor Core、SFU、Load/Store Unit。
4. 寄存器文件、shared memory、L1、L2、GDDR/HBM。
5. 内存分区、控制器、cache 行为和数据通路。
6. 功耗、频率、散热和持续性能。
7. PCIe、NVLink、NVSwitch 与系统拓扑。
8. Turing/T4 深入分析和 microbenchmark。
9. Turing、Ampere、Hopper、Blackwell 的关键演进。
10. Vera Rubin 已公开的平台级架构。
11. 如何阅读规格表、Compute Capability、白皮书与 SASS。

### 卷十：工程化、作品集与面试

目标：把知识转化为稳定工程能力和求职表达。

主要内容：

1. CMake、目标架构和可复现构建。
2. CUDA 错误包装、RAII 和资源生命周期。
3. CPU reference、单元测试、随机测试和误差容忍。
4. 性能基线与回归测试。
5. Compute Capability、PTX JIT 和部署兼容性。
6. Kernel code review 清单。
7. CUDA 常见故障案例库。
8. 高频概念题、性能分析题和手写 kernel 题。
9. 系统设计题：多流、多 GPU、显存预算和吞吐目标。
10. 三个作品集项目及完整技术报告。

完成十卷后，另设 AI 推理优化专项，覆盖 PyTorch CUDA Extension、算子融合、
TensorRT、低精度推理和推理系统性能分析。

## 6. 每章固定结构

每个章节使用统一模板：

1. 本章解决什么问题。
2. 前置知识。
3. 直觉解释与图解。
4. 正式模型、术语和公式。
5. 最小正确代码。
6. 逐行与逐线程推演。
7. 常见错误和反例。
8. 从 naive 到 optimized 的版本阶梯。
9. 正确性测试。
10. 性能实验和预期现象。
11. Nsight 或 sanitizer 操作。
12. 架构差异与适用边界。
13. 本章小结。
14. 复习题。
15. 面试题。
16. 编程练习与验收标准。
17. 官方资料映射。

## 6.1 每章实践合同

实践不是课后选做内容，而是每章完成条件。每章必须包含以下五类实践：

1. **手写**：学习者先独立完成最小实现或关键代码，不直接复制最终答案。
2. **验证**：使用 CPU reference、随机输入、边界输入和数值误差检查。
3. **故障注入**：故意制造越界、竞态、错误同步、错误索引或低效访存。
4. **测量**：根据章节使用 CUDA Event、Nsight 或 Compute Sanitizer 收集证据。
5. **表达**：记录实验结论，并完成面试式口述和书面问答。

实践分为四级：

```text
L1 跟练：跟随文档补全局部代码
L2 独立实现：只提供接口、目标和验收标准
L3 优化实验：提出假设并用性能指标验证
L4 综合项目：独立设计、测试、profiling、优化和报告
```

阅读完成不代表章节完成。只有代码、测试、实验数据、工具证据和复述检查
全部达到验收标准，才可在总目录中勾选该章。

## 7. 图解体系

图解不作为装饰，而是用于解释空间和时间关系。至少包括：

- CPU、GPU、SM、Grid、Block、Warp、Thread 层次图。
- Warp 与 block 线性化顺序。
- SM 执行资源和 warp 调度示意图。
- 内存层次与数据移动路径。
- Coalescing 的地址访问图。
- Shared-memory bank conflict 图。
- Transpose 的 block 外层与 tile 内层转置图。
- Reduction、scan、histogram 的阶段图。
- GEMM 多层 tiling 图。
- Stream 和 event 时间线。
- CUDA Graph 生命周期图。
- PCIe、NVLink、NVSwitch 和多 GPU 拓扑图。
- Roofline 与 kernel 落点图。

架构图需要区分抽象教学图和厂商实际框图，避免把简化图误认为完整硬件布局。

## 8. 实验与代码体系

### 8.1 目录

计划新增：

```text
course/
  README.md
  volume01_hardware/
  volume02_programming_model/
  volume03_memory/
  volume04_parallel_algorithms/
  volume05_performance/
  volume06_operators/
  volume07_async_systems/
  volume08_hpc_multigpu/
  volume09_hardware_architecture/
  volume10_engineering_interview/

labs/
  common/
  01_hardware/
  02_programming_model/
  ...

assets/
  diagrams/
  profiling/
```

现有 `week01_basics`、`week02_memory` 和 `docs` 内容不删除。编写教材时选择性
迁移、修订或链接，避免破坏现有学习记录。

### 8.2 公共实验基础设施

所有实验逐步统一以下能力：

- CUDA API 和 kernel 错误检查。
- CPU reference 实现。
- 可配置问题规模。
- 随机与边界输入。
- CUDA Event 计时。
- warmup 与重复测量。
- 结果误差检查。
- GPU、CUDA、编译参数记录。
- CSV 或 Markdown 性能结果。

### 8.3 版本阶梯

复杂实验至少提供：

```text
v0_cpu_reference
v1_naive
v2_correct_and_tiled
v3_memory_optimized
v4_warp_or_register_optimized
v5_architecture_specific（必要时）
```

每个版本明确说明修改了什么、预期改善哪个指标、可能引入什么限制。

## 9. 验证标准

### 9.1 文档验证

- 术语和符号在全书一致。
- `x=column`、`y=row` 为默认二维约定，例外必须明确说明。
- 输入输出尺寸、行主序线性地址和边界判断均可手工验证。
- 硬件事实有官方资料依据和适用架构说明。
- 不把宣传峰值当作实测性能。
- 不把特定 GPU 的资源限制推广为所有 GPU 的固定限制。

### 9.2 代码验证

- 所有代码可以在目标环境构建。
- 包含正常、最小、非整除和非方阵输入。
- 与 CPU reference 对比。
- 运行 Compute Sanitizer 的适用检查。
- 性能实验在正确性通过后执行。

### 9.3 学习验收

学习者完成一章后应能：

- 不看文档复述核心模型。
- 手工追踪至少一个小例子。
- 独立完成基础实现。
- 预测一个性能瓶颈。
- 使用工具验证预测。
- 回答对应面试题。

## 10. 编写顺序

教材按依赖分批编写，而不是一次性生成全部内容：

1. 建立总目录、术语表、实验规范和资料映射。
2. 完成卷一入门所需硬件基础和卷二编程模型，并整合现有 Week 1 内容。
3. 完成卷三内存系统，并整合现有 transpose 内容。
4. 完成卷四并行算法。
5. 完成卷五性能工程。
6. 完成卷六核心算子。
7. 完成卷七异步系统。
8. 完成卷八 HPC 与多 GPU。
9. 完成卷九硬件架构深入与代际演进。
10. 完成卷十工程与面试。
11. 全书交叉链接、术语一致性和代码回归验证。

每完成一卷，先由学习者实际阅读和完成实验，再根据理解障碍修订后续卷册。

## 11. 非目标

本轮教材不试图：

- 完整翻译 NVIDIA 文档或 PMPP。
- 覆盖所有 CUDA Runtime 和 Driver API。
- 在没有对应硬件时声称完成新架构的真实性能验证。
- 把 CUDA Python、OpenACC、OpenMP offload 或其他 GPU DSL 作为主线。
- 在 CUDA 核心和 HPC 主线完成前展开完整 AI 推理优化课程。

## 12. 当前环境约束

当前工作目录不是 Git 仓库，因此设计文档可以保存和校验，但无法创建提交。
后续如果初始化 Git，应避免提交已有二进制文件和可视化 brainstorming 临时目录。
