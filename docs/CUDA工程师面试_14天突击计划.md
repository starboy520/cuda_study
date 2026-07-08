# CUDA 工程师面试：14 天突击计划

## 定位

这是一份面向 CUDA 工程师面试的短周期执行手册：每天约 8 小时，最多 14 天。目标不是继续铺开知识面，而是把仓库中的学习记录转换为四种可当场验证的能力：闭卷解释、独立编码、性能取证和项目表达。

14 天分成三段：

| 阶段 | 时间 | 主线 |
| --- | ---: | --- |
| 深水区 | Day 1～7 | PTX/SASS、寄存器与调度、MMA、异步流水，以及 Ampere/Hopper 边界 |
| 独立手写与项目 | Day 8～11 | 从空白实现核心 kernel，补齐正确性与性能证据，收口 A100 旗舰案例 |
| 面试转换 | Day 12～14 | 高频概念、限时编码、性能诊断、项目追问与模拟面试 |

本文件当前只规定共同基线、执行规则和开跑条件；每日正文按上述路线逐步写入。不要提前把阶段标题当作已完成事项。

## 使用说明

1. 每天开始前先读当日验收条件，再安排学习与实验；不要以“看了多久”衡量进度。
2. 阅读完成不等于验收。即使仓库已有完整笔记或可运行代码，也必须经过当天要求的闭卷解释和独立操作。
3. 每天必须留下四类可复查输出：
   - **概念输出**：不看资料解释当天主题，能说清机制、边界和取舍；
   - **代码输出**：从空白或最小脚手架写出核心逻辑，并覆盖约定边界；
   - **证据输出**：保留正确性、计时、资源、ncu、PTX/SASS 中与主题匹配的证据；
   - **面试输出**：形成一段 2～5 分钟可复述回答，采用“先结论、再边界、最后给项目证据”的结构。
4. 每项输出都标注日期、硬件、shape、编译参数和命令。性能数字必须区分正常计时与 profiler 重放结果。
5. 计划上限是 14 天。遇到难点使用补考机制，不无限挤占后续独立编码和面试训练。

## 基于仓库证据的能力画像

下表描述的是“仓库已经提供哪些起点”，不是对个人能力的自动认证。**仓库有文档、有代码或有历史性能数字，不等于个人已经掌握；只有闭卷复述、独立重写和重新取证通过，才能把该项改为已掌握。**

| 维度 | 仓库中的具体证据 | 当前判断与 14 天验收方向 |
| --- | --- | --- |
| 执行模型 | [Week 3 记录](../notes/week03.md#三个硬核结论面试素材)展示 grid-stride、block/warp 分层归约和单 block 上限；同一记录还区分硬件容量限制与算法上限。 | 有线程层次、同步和 grid-stride 实践基础；需闭卷解释 warp 调度、ILP/TLP、occupancy 与 latency hiding 的关系。 |
| 内存 | [Reduction 记录](../notes/week03.md#三个硬核结论面试素材)给出 128 MB、理论约 0.40 ms、实测 0.44 ms，即约 **91% 峰值带宽**；[GEMM 记录](../week05_gemm_advanced/benchmark.md#结论关键认知)展示 padding、bank conflict 与 float4 的取舍。 | 已有 coalescing、shared memory、padding 和向量化证据；需能从访问地址推导 transaction/bank 映射，并用指标而非口号判断瓶颈。 |
| 并行算法 | [Week3 性能总表（T4，全部 PASS）](../notes/week03.md)汇总 reduction 追平 CUB、scan 与 histogram 的优化结果。 | reduction/scan/histogram 已有多版实践；需从空白限时写核心版本，补齐任意尺寸、边界与数值正确性说明。 |
| GEMM | [Week2 Day1：float4 向量化 load（A100, 2048）](../week05_gemm_advanced/benchmark.md)记录 float4 global load 配合 padded shared 后达到 **12681 GFLOPS**，并保留去 padding 后性能倒退的反例。 | 已形成 tiling、寄存器复用、向量化、bank conflict 的优化链；需闭卷重建关键索引，并用 ncu 与资源数据解释为何有效或失效。 |
| Tensor Core | [教学版 WMMA benchmark](../week06_tensorcore/tensor_core_profile.md)记录 WMMA 在 1024 shape 达 **16484 GFLOPS**；[SASS 证据](../week06_tensorcore/tensor_core_profile.md#2-证据一sass-有-hmma-指令用了-tensor-core)找到 `HMMA.16816.F32`。 | 已接触 WMMA 与机器指令取证；需独立解释 fragment、`ldmatrix`/`mma.sync` 职责、精度语义和“用了 Tensor Core 仍可能被访存限制”。 |
| Attention | [Attention 流水记录](../week04_attention/ncu_pipeline_notes.md#怎么读)显示 `cp.async` 双缓冲后 long scoreboard 从 6.40 降至 0.03（**下降 99.5%**），同时 short scoreboard 基本不变。 | 已有 Online Softmax、教学版 Attention 与异步流水材料；需重建数据流、稳定性与 causal 边界，并诚实说明小 grid 下 stall 改善不等于墙钟同比加速。 |
| profiling | [GEMM ncu 笔记](../week05_gemm_advanced/ncu_notes.md#面试口述day-4)用 4.8-way bank conflict、吞吐和周期变化闭合“假设—定位—修复—验证”；[Attention 对比](../week04_attention/ncu_pipeline_notes.md#精确-stall-指标)保留精确 stall 指标。 | 有 ncu 使用与瓶颈迁移意识；需能独立选择指标、识别 replay/锁频干扰，并把指标变化连成因果链。 |
| 系统工程 | [Week 5 记录](../notes/week05.md#day-22026-07-07gemv-一线程一-warp-一行)包含 CPU 对拍、CUDA Event 计时和 memcheck 0 errors；设计基线还覆盖 stream、pinned memory 与错误检查。 | 已有正确性和计时框架意识；需统一边界 shape、重复运行、异步错误检查和 Compute Sanitizer 流程，并说明何时结果不可比。 |
| 底层指令 | [Tensor Core profile](../week06_tensorcore/tensor_core_profile.md#2-证据一sass-有-hmma-指令用了-tensor-core)已经通过 `cuobjdump` 找到 HMMA；[设计说明](superpowers/specs/2026-07-08-cuda-interview-14-day-sprint-design.md#2-用户与前置基础)将 PTX/SASS、warp MMA、调度与 stall 列为主要缺口。 | 目前证据点状、缺系统阅读能力；Day 1～7 必须建立 CUDA C++ → PTX → SASS 映射，并能解释寄存器、spill、scoreboard 与关键指令。 |
| 面试表达 | 多份记录已有“一段口述”，例如 [GEMM 反例复盘](../week05_gemm_advanced/benchmark.md#面试口述)和 [WMMA 三重证据](../week06_tensorcore/tensor_core_profile.md#7-面试口述)。 | 已有素材但未证明能脱稿应答；需形成 2～5 分钟日答、5/10 分钟项目版，并接受追问、反例和适用边界检查。 |
| 推理与 GEMV | [Week 5 GEMV 性能对比（N=4096 K=4096）](../notes/week05.md)显示 warp-per-row 相比 thread-per-row 约 **19 倍**提速，带宽从 72 GB/s 到 1360 GB/s。 | 已能用合并访问解释 decode 的 memory-bound 场景；需从空白写 warp-per-row、验证数值误差，并说明向量化为何只带来有限增益。 |

## 优先级规则

### 必须完成

直接决定当前面试通过率，不得用扩展阅读替代：每日四类输出、核心 kernel 独立手写、正确性与边界验证、ncu/PTX/SASS 证据链、旗舰案例和模拟面试。时间不足时仍不得取消每日验收。

### 尽量完成

用于展示性能深度：更多 shape、第二种实现、失败实验复盘、与库实现或理论上限对照、复杂 fragment 映射和更完整的流水参数扫描。只有“必须完成”已通过后才投入时间。

### 时间不足可跳过

短期内不会阻止参加初级或中初级 CUDA 面试：无法在 A100 上实测的 Hopper 性能实验、生产级 CUTLASS 内核复刻、额外 kernel 数量和过细的架构史。Hopper 仍需能做正确的概念对比，但不伪造实测结论。

## 每日时间盒

每天总计 8 小时，按结果而不是阅读页数推进：

| 时间 | 模块 | 当段结束条件 |
| ---: | --- | --- |
| 2 小时 | 概念 | 画出机制或数据流，并完成一次不看资料的口述 |
| 3 小时 | 代码 | 核心实现可编译、可运行，reference 对拍和主要边界通过 |
| 1.5 小时 | 证据 | 留下可复现命令、环境、结果和瓶颈解释 |
| 1.5 小时 | 闭卷 | 限时手写或复述、记录错误，并形成面试输出 |

时间盒到点仍未完成时，记录失败点、最小复现和下一次验证动作，不用无限加时掩盖问题。

## 补考规则

- 当天任一“必须完成”项未通过，就把失败项和判定证据写入次日开场清单。
- 次日补考最多 **90 分钟**，只重做未通过的最小验收，不重新泛读全部材料。
- 补考通过后立即返回当日主线；补考再次失败，则进入 **Day 12 薄弱清单**，注明缺口、已试方法和可复现证据。
- 同一难点不得连续吞掉后续日程。Day 12 再按面试频率和岗位相关性排序回补。
- “看懂答案”“照抄通过”和“历史记录曾通过”均不算补考通过。

## 突击前准备

### 环境边界

- 目标实测设备是 **A100（sm_80）**。Ampere 内容要求实际编译、运行和取证；Hopper 内容只做概念与架构对比，不把 A100 结果冒充 Hopper 数据。
- 准备 `nvcc`、`ncu`、`cuobjdump`、`compute-sanitizer`，并确认驱动可见 A100。
- 准备阶段只读取环境信息，或把临时产物写到 `/tmp/cuda_interview_sprint/`。**不整理工作区，不删除、覆盖、格式化或暂存现有文件。**

### 只读检查

```bash
nvidia-smi
nvcc --version
ncu --version
cuobjdump --version
compute-sanitizer --version
```

记录 GPU 型号、显存、驱动、CUDA Toolkit 和 Nsight Compute 版本。若命令不可用，先记录缺失项；不要为了“清洁环境”修改仓库。

### 临时目录冒烟测试

```bash
mkdir -p /tmp/cuda_interview_sprint
cd /tmp/cuda_interview_sprint
```

后续环境探测产生的可执行文件、profile 报告、PTX 和 SASS 文本都放在该目录。正式学习输出只有在每日任务明确指定时才进入仓库。

开跑前逐项确认：

- [ ] `nvidia-smi` 显示 A100，且当前进程有可用权限；
- [ ] `nvcc` 能以 `sm_80` 为目标编译最小程序；
- [ ] `ncu` 可采集一个最小 kernel，权限错误已单独记录；
- [ ] `cuobjdump` 可从临时可执行文件导出 SASS；
- [ ] `compute-sanitizer` 可运行 memcheck；
- [ ] `/tmp/cuda_interview_sprint/` 可写，仓库原有改动保持不动；
- [ ] 已建立每日四类输出的记录位置与命名规则。

## 总体路线

```text
Day 1～7   CUDA C++ → PTX → SASS → 寄存器/调度/stall → MMA → 异步流水 → 架构边界
Day 8～11  独立手写核心 kernel → 正确性与边界 → 性能证据 → A100 旗舰案例
Day 12～14 薄弱项回补 → 限时编码与诊断 → 项目追问 → 完整模拟面试
```

路线的顺序服务于最终闭环：能解释，能写，能证明，能在追问下讲清取舍。
