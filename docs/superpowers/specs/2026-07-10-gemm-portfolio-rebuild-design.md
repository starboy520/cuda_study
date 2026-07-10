# GEMM 作品集重建设计

日期：2026-07-10

## 1. 背景

现有 `portfolio/01_gemm_optimization` 已包含 naive、shared-memory tiling、2D register tiling、`float4` 加载、双缓冲和 cuBLAS 对照，但它仍是学习阶段代码的集合，尚未达到公开作品集标准。

审查发现：

- `portfolio/` 当前未被 Git 跟踪，无法展示可信的演进过程。
- 源码保留大量“你来写”、教学说明和已完成的 `TODO`，公开展示时容易让实现归属不清。
- 各版本分别维护测试和计时逻辑，输入尺寸、warmup、迭代次数、误差标准并不统一。
- README 声明的完成度高于仓库可直接复现的状态。
- 部分程序忽略命令行尺寸。
- 实测非对齐尺寸 `M=N=K=130` 时，shared 和 2D 版本通过；`float4` 版本发生 misaligned/非法 16-byte 读取；双缓冲版本得到错误结果。
- 现有性能数据来自不同实验阶段和不同代码状态，不应直接作为首版公开结果。

因此，本项目不在旧文件上做表面清理，而是以旧代码为私人学习资料，从干净结构重新实现并重新采集证据。

## 2. 项目目标

构建一个可公开投递、可复现、可面试讲解的 FP32 GEMM 优化项目，展示以下能力：

1. 从数学定义独立实现正确的 CUDA kernel。
2. 通过数据复用和线程映射逐级优化 GEMM。
3. 使用统一 benchmark 公平比较不同版本。
4. 使用 Nsight Compute 和 SASS 建立“瓶颈—修改—指标—结果”的证据链。
5. 正确处理快速路径的适用条件和非对齐输入。
6. 保留负优化结果，并用数据解释原因。
7. 通过清晰的提交历史体现真实实现与实验过程。

核心 kernel 由作者亲手重新实现。辅助工作可以使用统一脚手架，包括 CPU reference、输入生成、误差统计、计时、结果导出、构建和测试命令。辅助脚手架不得隐藏 kernel 的线程映射、内存访问、同步或流水实现。

## 3. 首版范围

### 3.1 包含

- NVIDIA A100 80GB，目标架构 `sm_80`。
- FP32、row-major、`C = A × B`。
- 矩形 `M × K` 与 `K × N` 输入。
- 六级实现：
  1. naive
  2. shared-memory tiling
  3. 2D register tiling
  4. `float4` vectorized load
  5. `cp.async` double buffering
  6. cuBLAS FP32 baseline
- aligned shape 的优化快速路径。
- 非对齐 shape 的安全 fallback。
- CPU/cuBLAS 对拍、统一 benchmark、sanitizer、ncu 和 SASS 证据。

### 3.2 不包含

首版不实现以下功能：

- Tensor Core、WMMA 或 `mma.sync`
- FP16、BF16、TF32 kernel
- batched GEMM
- 转置输入
- `alpha`/`beta` BLAS 语义
- 多 GPU
- 面向生产库的稳定 ABI

这些内容可作为后续独立里程碑，不阻塞首版投递。

## 4. 方案比较与决策

### 4.1 代码处理方式

考虑过三种方式：

1. **原文件去教学痕迹**：最快，但无法解决测试口径漂移、边界缺陷和实现归属问题。
2. **在原结构中逐个重写**：保留独立可执行文件，可读性尚可，但公共逻辑继续重复。
3. **干净重建，统一 runner + 分版本 kernel**：重建成本最高，但可保证公平测试、接口一致和演进清晰。

采用方案 3。

### 4.2 工程复杂度

不做完整 CUDA 库，也不为首版引入复杂框架。项目保持“可读的研究型工程”定位：统一入口负责实验协议，各 kernel 保持独立、短小、可单独讲解。

## 5. 总体架构

### 5.1 组件边界

项目由以下逻辑组件组成：

- **Kernel implementations**：每个版本只负责设备端计算，不包含数据生成、性能统计和输出格式。
- **Kernel registry**：描述版本名称、launcher、快速路径约束和说明。
- **Runner**：解析参数，选择 kernel，执行 warmup、计时和验证。
- **Reference**：小尺寸使用 CPU double 累加；大尺寸使用明确关闭 TF32 的 cuBLAS FP32 结果。
- **Validation**：计算 `max_abs`、`max_rel`、最差元素位置，并检查 NaN/Inf。
- **Benchmark reporter**：输出延迟、GFLOPS、相对 cuBLAS 比例和相对上一版加速比，同时保存机器可读结果。
- **Profiling workflow**：固定 ncu 指标、SASS 提取方式和实验元数据。

### 5.2 统一 kernel 接口

所有手写版本对 runner 暴露相同的 launch 语义：

- 输入：设备端 `A`、`B`
- 输出：设备端 `C`
- 形状：`M`、`N`、`K`
- 执行流：runner 提供的 CUDA stream
- 返回信息：实际选择的执行路径，例如 `fast-float4` 或 `fallback-register-tiled`

launcher 负责检查当前 kernel 的前置条件，但不静默执行不安全访问。

### 5.3 数据流

一次完整运行按以下顺序执行：

1. 解析 shape、seed、warmup、iterations 和待运行版本。
2. 用固定 seed 生成同一份输入。
3. 根据规模生成 CPU 或 cuBLAS reference。
4. 对每个版本清空输出并检查快速路径约束。
5. 执行 warmup，检查 kernel launch 和同步错误。
6. 执行正确性验证。
7. 对 aligned benchmark shape 批量计时。
8. 汇总并打印统一结果表。
9. 可选输出 JSON/CSV 原始记录，供 README 表格生成。

## 6. 优化阶梯

### 6.1 Naive

- 一个线程计算一个 `C[row, col]`。
- 直接遍历 K 维读取 global memory。
- 目标是建立最简单、可信的正确性和性能基线。

### 6.2 Shared-memory tiling

- 一个 block 计算一个输出 tile。
- 协作加载 A/B tile 到 shared memory。
- 使用边界补零支持任意 `M/N/K`。
- 目标是证明 global-memory 数据复用带来的收益。

### 6.3 2D register tiling

- 一个线程计算 `TM × TN` 个输出。
- 用寄存器保存局部累加器并执行外积。
- 参数选择同时考虑寄存器压力、occupancy、ILP 和 shared-memory 访问。
- 目标是减少 shared-memory 读取并展示 occupancy 不是越高越好。

### 6.4 `float4` vectorized load

- 对满足地址与维度对齐条件的 tile 使用 16-byte global load。
- 明确列出快速路径约束。
- 非对齐输入自动调用安全 2D register-tiled fallback。
- 不允许把“越界但通常没崩”作为尾块处理方式。

### 6.5 `cp.async` double buffering

- 在两个 shared-memory stage 之间交替预取和计算。
- 对齐输入走异步快速路径；非对齐输入走安全 fallback。
- 明确 producer/consumer 次序、等待点和 block 同步。
- 双缓冲不预设一定快于 `float4` 版本；如果变慢，保留结果并分析指令开销、stage 粒度、occupancy 和 kernel 当前 bound 类型。

### 6.6 cuBLAS FP32 baseline

- 作为性能与大尺寸正确性对照。
- 明确设置 math mode，关闭 TF32，避免 CUDA Core FP32 与 Tensor Core TF32 混比。
- 记录 cuBLAS、CUDA driver/runtime 与 GPU 信息。

## 7. 形状与 fallback 策略

项目区分两类输入：

- **正确性 shape**：覆盖小尺寸、矩形、tile 尾块和非 4 对齐维度。
- **性能 shape**：选择满足快速路径约束的代表性大矩阵。

`float4` 与双缓冲 launcher 在运行前检查：

- 基地址与行起点是否满足 16-byte 对齐。
- K/N 以及所需 tile 边界是否满足快速加载条件。
- kernel 的 block/tile 配置是否满足 launch resource 限制。

约束不满足时，runner 打印 fallback 原因并调用安全版本。fallback 的性能不与快速路径混在同一优化阶梯结果中。

## 8. 正确性设计

### 8.1 小尺寸 CPU 对拍

至少覆盖：

- 极小尺寸：`1`、`2`、`3`
- warp/tile 边界：`17`、`31`、`32`、`33`、`63`、`64`、`65`
- 非对齐尺寸：`127`、`130`
- 矩形组合，而不只测试 `M=N=K`
- 固定随机、全零、常量和正负混合输入

CPU reference 使用 double 累加，最终转换为 float 比较。

### 8.2 大尺寸 cuBLAS 对拍

代表性规模包括 `512`、`1024`、`2048` 和资源允许时的 `4096`。大尺寸不运行低效 CPU reference，以 cuBLAS FP32 结果为准。

### 8.3 误差报告

每次验证至少输出：

- `max_abs`
- `max_rel`
- 最差元素的 row/column
- reference 与 actual 值
- 是否存在 NaN/Inf

容差按 FP32 GEMM 的累加长度设置，并在文档中固定，不针对单个失败案例临时放宽。

## 9. 性能实验协议

所有版本遵循同一协议：

- 使用相同输入与设备缓冲区。
- 内存分配和 H2D/D2H 不计入 kernel 时间。
- 先执行固定次数 warmup。
- 用 CUDA event 对多次迭代整体计时，报告平均延迟。
- 每个 shape 重复多轮，记录中位数或明确规定的稳定统计量。
- 记录 GPU 名称、时钟状态、CUDA 版本、编译器版本和完整编译参数。
- 输出毫秒、GFLOPS、相对上一版本加速比、相对 cuBLAS FP32 比例。
- benchmark 前后都检查 CUDA 错误。

原有性能数字仅作为历史参考。公开 README 的数字必须由重建后的代码和固定实验命令重新生成。

## 10. Profiler 与 SASS 证据

### 10.1 每一级固定回答

1. 上一版的主要瓶颈是什么？
2. 本版修改了什么数据复用、线程映射或执行流水？
3. 哪些 ncu 指标支持判断？
4. 墙钟性能如何变化，原因是什么？

### 10.2 ncu 证据

根据版本选取少量、直接相关的指标，例如：

- DRAM/L2/L1/shared throughput
- global load sectors 与请求效率
- shared bank conflict
- registers per thread
- theoretical/achieved occupancy
- SM throughput
- long/short scoreboard stall
- eligible/active warps

不以单一高百分比直接断言瓶颈，结论必须结合吞吐、stall、occupancy 和工作规模。

### 10.3 SASS 证据

只保存与结论直接相关的短摘录和统计，例如：

- 标量与 128-bit global load 的差异
- FFMA 指令
- `cp.async` 对应指令
- register spill 是否出现

完整二进制和大型 profiler 报告不进入 Git。

## 11. 错误处理与安全性

- 所有 CUDA Runtime 与 cuBLAS 调用统一检查返回值。
- kernel launch 后立即检查 launch error；同步点检查异步错误。
- 参数非法、维度溢出、分配失败或无可用 GPU 时返回非零退出码。
- 使用安全的尺寸乘法，避免 `M*N*sizeof(float)` 整数溢出。
- fallback 必须显式报告，禁止静默将快速路径错误归因于输入。
- benchmark 失败的版本不输出有效 GFLOPS。

## 12. 验证门槛

每一级按以下顺序验收：

1. 编译无错误，警告已解释或修复。
2. 所有规定 shape 正确性通过。
3. memcheck 通过。
4. shared-memory 版本执行 racecheck 和 synccheck。
5. aligned 性能 shape 完成 benchmark。
6. 采集与本次优化直接相关的 ncu 指标。
7. 提取必要的 SASS 证据。
8. 更新该级实验记录。
9. 仅提交该级相关代码和证据。

任何一项失败，都不把该版本标记为完成。

## 13. Git 与公开展示

### 13.1 提交原则

- 旧教学代码保留在私人备份中，不进入重建后的公开项目历史。
- 脚手架、每一级 kernel、benchmark、证据和文档分阶段提交。
- 每个提交只表达一个技术步骤。
- 不提交编译产物、`.ncu-rep` 大文件、临时日志或机器相关缓存。
- 不通过压平历史制造“一次性完成”的假象。

### 13.2 README 结构

最终 README 包含：

1. 项目一句话目标
2. 环境与复现方式
3. 优化阶梯结果表
4. 每一级瓶颈和改动摘要
5. 正确性与 sanitizer 状态
6. ncu/SASS 证据链接
7. 负优化案例
8. 限制与下一步

所有公开数字都必须能由仓库中的命令和数据文件重新得到。

## 14. 预期目录职责

首版采用以下目录边界；实施计划可以补充文件，但不得合并这些职责：

- `kernels/`：各优化版本的 kernel 与 launcher
- `runner/`：统一 CLI、kernel registry 和 benchmark 调度
- `common/`：错误检查、reference、validation、计时等辅助设施
- `tests/`：正确性 shape 清单和测试入口
- `scripts/`：sanitizer、benchmark、ncu、SASS 提取与结果生成
- `results/`：轻量 JSON/CSV 原始数据和生成后的 Markdown 摘要
- `docs/`：优化记录、指标解释和负结果分析

避免将全部实现塞入单个超大 `.cu` 文件，也避免为每个 kernel 复制一份 main 和 CPU reference。

## 15. 完成定义

首版达到以下条件时可公开投递：

- 六级阶梯均可通过统一构建命令生成。
- 所有规定正确性 shape 通过；快速路径和 fallback 都被覆盖。
- sanitizer 无错误。
- README 的 benchmark 表由重建代码重新采集。
- cuBLAS 对照明确为 FP32，未混入 TF32。
- 每个关键优化至少有一组 profiler 或 SASS 证据。
- 双缓冲无论提升或下降，都有可复现数据和合理解释。
- 新读者能在不阅读 runner 内部实现的情况下理解每个 kernel 的职责和约束。
- 作者能够独立讲清每个索引、同步点、资源权衡和性能结论。

## 16. 一句话叙事

从正确但低效的 naive GEMM 出发，通过 shared-memory 复用、2D register tiling、向量化加载和异步流水逐步移动瓶颈，并使用统一正确性测试、Nsight Compute 与 SASS 解释每一步收益和代价。
