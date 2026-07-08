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

| 维度 | 仓库中的具体证据 | 仓库覆盖与个人验收 |
| --- | --- | --- |
| 执行模型 | [Week 3 记录](../notes/week03.md#三个硬核结论面试素材)展示 grid-stride、block/warp 分层归约和单 block 上限；同一记录还区分硬件容量限制与算法上限。 | 仓库已覆盖线程层次、同步和 grid-stride。个人验收：闭卷画出执行层次，并解释 warp 调度、ILP/TLP、occupancy 与 latency hiding 的关系。 |
| 内存 | [Reduction 记录](../notes/week03.md#三个硬核结论面试素材)显示 N=16M 的手写版在 T4 上约 0.44 ms，并追平当时的 CUB 对照；输入为 float，单次读取约 64 MiB，旧笔记把流量记为 128 MB 并据此声称 91% 峰值带宽，字节口径不恰当或未说明，不能作为掌握证据。[GEMM 记录](../week05_gemm_advanced/benchmark.md#结论关键认知)还包含 padding、bank conflict 与 float4 的取舍。 | 仓库已覆盖 coalescing、shared memory、padding 和向量化。个人验收：按实际读写流量重新跑 reduction 并计算有效带宽，再从地址推导 transaction/bank 映射；不得沿用旧 91% 结论。 |
| 并行算法 | [Week3 性能总表（T4，全部 PASS）](../notes/week03.md)汇总 reduction 追平 CUB、scan 与 histogram 的历史结果。 | 仓库已记录 reduction/scan/histogram 多版实现。个人验收：从空白限时写核心版本，完成 reference 对拍，并覆盖任意尺寸和边界输入。 |
| GEMM | [Week2 Day1：float4 向量化 load（A100, 2048）](../week05_gemm_advanced/benchmark.md)记录 A100、M=N=K=2048 下 float4 global load 配合 padded shared 达到 **12681 GFLOPS**，并保留去 padding 后性能倒退的反例。 | 仓库已覆盖 tiling、寄存器复用、向量化和 bank conflict 优化链。个人验收：闭卷重建关键索引，重新跑正确性与性能，并用 ncu 和资源数据解释有效或失效的原因。 |
| Tensor Core | [教学版 WMMA benchmark](../week06_tensorcore/tensor_core_profile.md)记录 A100 80GB、FP16 输入/FP32 累加、M=N=K=1024 下 **16484 GFLOPS**；[SASS 证据](../week06_tensorcore/tensor_core_profile.md#2-证据一sass-有-hmma-指令用了-tensor-core)找到 `HMMA.16816.F32`。 | 仓库已覆盖 WMMA 与 HMMA 取证。个人验收：重跑 API、SASS、性能三重证据，闭卷解释 fragment、`ldmatrix`/`mma.sync`、精度语义及访存瓶颈。 |
| Attention | [Attention 流水记录](../week04_attention/ncu_pipeline_notes.md#怎么读)记录 A100、N=128、D=64 的 ncu 对比：`cp.async` 双缓冲后 long scoreboard 指标从 6.40 降至 0.03（**下降 99.5%**），short scoreboard 基本不变；该小 grid 填不满 A100，指标改善不代表墙钟同比加速。 | 仓库已覆盖 Online Softmax、教学版 Attention 与异步流水材料。个人验收：从空白重建数据流、稳定性和 causal 边界，重跑对照并同时报告 stall、吞吐与墙钟限制。 |
| profiling | [GEMM ncu 笔记](../week05_gemm_advanced/ncu_notes.md#面试口述day-4)记录 A100、M=N=K=1024 profile 中 4.8-way bank conflict 及修复后的吞吐、周期变化；[Attention 对比](../week04_attention/ncu_pipeline_notes.md#精确-stall-指标)保留具体 stall 指标。 | 仓库已覆盖一条“假设—定位—修复—验证”链。个人验收：给定未知 kernel 独立选择指标，识别 replay/锁频干扰，并闭卷把指标变化连成可证伪的因果链。 |
| 系统工程 | [Week 5 记录](../notes/week05.md#day-22026-07-07gemv-一线程一-warp-一行)包含 A100 上的 CPU 对拍、CUDA Event 计时和 memcheck 0 errors；设计基线还覆盖 stream、pinned memory 与错误检查。 | 仓库已记录正确性与计时框架。个人验收：为新写 kernel 独立加入边界 shape、重复运行、异步错误检查和 Compute Sanitizer，并说明结果不可比的情形。 |
| 底层指令 | [Tensor Core profile](../week06_tensorcore/tensor_core_profile.md#2-证据一sass-有-hmma-指令用了-tensor-core)记录通过 `cuobjdump` 找到 HMMA；[设计说明](superpowers/specs/2026-07-08-cuda-interview-14-day-sprint-design.md#2-用户与前置基础)将 PTX/SASS、warp MMA、调度与 stall 列为主要缺口。 | 仓库的底层指令证据仍是点状材料。个人验收：从 CUDA C++ 沿数据流定位关键 PTX/SASS，并闭卷解释寄存器、spill、scoreboard 与关键指令。 |
| 面试表达 | 多份记录包含“一段口述”，例如 [GEMM 反例复盘](../week05_gemm_advanced/benchmark.md#面试口述)和 [WMMA 三重证据](../week06_tensorcore/tensor_core_profile.md#7-面试口述)。 | 仓库已提供口述素材，但不证明本人能脱稿应答。个人验收：录制 2～5 分钟日答和 5/10 分钟项目版，并接受追问、反例和适用边界检查。 |
| 推理与 GEMV | [Week 5 GEMV 性能对比（N=4096 K=4096）](../notes/week05.md)记录 A100、N=K=4096 下 warp-per-row 相比 thread-per-row 约 **19 倍**提速，带宽从 72 GB/s 到 1360 GB/s。 | 仓库已覆盖 decode memory-bound 与合并访问案例。个人验收：从空白写 warp-per-row、重新对拍和计时，并闭卷说明误差变化及继续向量化收益有限的条件。 |

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
- 次日补考最多 **90 分钟**，并计入次日总计 8 小时，不是在 8 小时之外加时；只重做未通过的最小验收，不重新泛读全部材料。
- 补考时间优先从扩展阅读和“尽量完成”项扣除；仍不足时可压缩概念学习或证据排版美化，但不得删除核心代码正确性验证，也不得省略概念、代码、证据、面试四类输出。
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

## Day 1：打通 CUDA C++ → PTX → ptxas → cubin/SASS

**当日目标**：以 naive GEMM 为唯一对象，亲手生成 PTX、ptxas 资源日志和 A100 SASS，能闭卷说明 `compute_80`、`sm_80` 以及离线 ptxas/JIT 的边界。仓库里已有讲解只作为教材，不算本人通过。

**优先级**：必须完成三层产物、命令记录和闭卷解释；尽量完成同时嵌入 cubin/PTX 的 fatbin 检查；时间不足可跳过 `nvdisasm` 与跨 toolkit 对比。

**约 8 小时时间表**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 0.5 小时 | 若有昨日失败项，按补考规则做最小补考；首日用于环境复核 | 补考结论或工具版本表 |
| 1.5 小时 | 阅读编译链，画 CUDA C++→PTX→cubin/SASS 图 | 一页闭卷重画稿 |
| 3 小时 | 在 `/tmp/cuda_interview_sprint/day01/` 编译 naive GEMM | `.ptx`、可执行文件、ptxas log、`.sass` |
| 1.5 小时 | 沿 kernel 找参数、load、FFMA、store，记录源行与指令类别 | 四段对照摘录 |
| 1.5 小时 | 闭卷验收与 3 分钟录音；失败则在本时段重做 | 答题纸、录音、失败清单 |

**必读仓库资料**：[主教材 3. Day 1：CUDA C++、PTX、SASS 不是同一层](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[naive GEMM](../week04_gemm/gemm_naive/gemm_naive.cu)。

**分步实验**：

1. `mkdir -p /tmp/cuda_interview_sprint/day01 && cd /tmp/cuda_interview_sprint/day01`，记录 `nvidia-smi`、`nvcc --version`；所有生成物留在此目录。
2. 生成虚拟 ISA：`nvcc -O3 -lineinfo -arch=compute_80 -ptx "$OLDPWD/week04_gemm/gemm_naive/gemm_naive.cu" -o naive.compute_80.ptx`。
3. 生成 A100 机器码路径并保留资源报告：`nvcc -O3 -lineinfo -arch=sm_80 -Xptxas=-v "$OLDPWD/week04_gemm/gemm_naive/gemm_naive.cu" -o naive.sm80 2> ptxas.log`。
4. 导出实际嵌入的机器指令：`cuobjdump --dump-sass ./naive.sm80 > naive.sm80.sass`；用 `cuobjdump --list-elf` 确认目标 code object。
5. 运行 `./naive.sm80 257 259 263` 做非整齐 shape 正确性检查；标注 PTX 是虚拟 ISA，SASS 才是该 `sm_80` code object 的机器指令。若当前驱动将来从嵌入 PTX JIT，新架构生成的 SASS 可能不同。

**必须产出**：`environment.txt`、`commands.sh`、`naive.compute_80.ptx`、`ptxas.log`、`naive.sm80.sass`、`day01_notes.md`、闭卷答题和录音索引；笔记必须写清硬件、shape、编译参数与每份证据证明/不能证明什么。

**闭卷验收题**：① `compute_80` 与 `sm_80` 分别指定什么，为什么前者不能直接等同 A100 最终机器码？② ptxas 在哪一步工作，驱动 JIT 又可能在哪一步工作？③ 如何证明可执行文件里有 `sm_80` code object？④ 从源码的一次 `C[row*N+col]` 写回，依次说出你在 PTX/SASS 中寻找的类别。四题全部说清且能现场重跑命令才通过。

**面试口述主题**：用 3 分钟解释“我如何从一段 CUDA C++ 得到可复核的 PTX、资源日志和 A100 SASS”，结尾主动给出 PTX 可移植性与 SASS 架构相关性边界。

**没通过时的降级/补救**：先把对象缩成只含一个 naive kernel 的 `/tmp` 副本，保留三条核心命令；次日开场最多 90 分钟补考并计入 Day 2 的 8 小时。仍失败就记录缺失工具/报错和最小复现，进入 Day 12 薄弱清单，不拿教材截图充当产物。

## Day 2：按数据流读五版 PTX/SASS

**当日目标**：对照 naive、shared、register tiling、`float4`、WMMA 五版，只追踪 global load→shared→计算→控制→store 的关键路径，形成可比较的观察表；不逐行翻译汇编，不把指令拼写当作跨架构承诺。

**优先级**：必须完成五版观察表和每版至少一条有行号/函数名的 PTX 或 SASS 证据；尽量同时保留 PTX/SASS 两层；时间不足可跳过非主循环指令和精确周期推断。

**约 8 小时时间表**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 1.5 小时 | Day 1 最小补考（若无需补考，则用于阅读方法） | 补考记录或数据流标注规则 |
| 1.5 小时 | 阅读五类观察点并先看 CUDA 源码 | 五版数据流草图 |
| 3 小时 | 在 `/tmp/cuda_interview_sprint/day02/` 生成/整理五版 PTX、SASS | 每版编译日志与摘录 |
| 1 小时 | 填观察表，横向比较 load/shared/FFMA/HMMA/control/store | `instruction_observation.md` |
| 1 小时 | 闭卷验收和 3 分钟口述 | 答题纸、录音索引 |

**必读仓库资料**：[主教材 4. Day 2：怎样开始读 PTX 与 SASS](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[naive](../week04_gemm/gemm_naive/gemm_naive.cu)、[shared tiled](../week04_gemm/gemm_tiled/gemm_tiled.cu)、[register tiling](../week05_gemm_advanced/gemm_2d_thread_tiling.cu)、[`float4`](../week05_gemm_advanced/gemm_vectorized_load.cu)、[WMMA](../week06_tensorcore/wmma_fp16_gemm.cu)。

**分步实验**：

1. 给五版画同一模板：输入地址生成、global load、可选 shared stage、register fragment/accumulator、FFMA/HMMA、边界控制、global store。
2. 把源文件复制到 `/tmp/cuda_interview_sprint/day02/src/`；高级版若有 TODO，只在临时副本补齐并保存 diff，不改仓库。每版用 `nvcc -O3 -lineinfo -arch=sm_80 -Xptxas=-v ...` 编译，再用 `cuobjdump --dump-sass` 导出；能生成 PTX 的版本再以 `-arch=compute_80 -ptx` 生成 PTX。
3. 用 kernel 名定位函数，然后只摘关键类别：global `LD*`、shared `LD*/ST*`、标量 `FFMA`、Tensor Core `HMMA`（PTX 层相应查 `mma`/WMMA 展开）、分支/谓词、global store。保留上下各少量上下文，不抄整文件。
4. 表格列固定为“版本、源码数据流、global load、shared、计算、control、store、资源、我能证明的结论、仍不能证明的结论”。`float4` 的宽 load 留到 Day 3 作严格验证。

**必须产出**：五版编译命令和 PASS/FAIL 状态、`instruction_observation.md`、五份短摘录、五张源码数据流图；任何不能编译的模板必须写明 TODO 阻塞，不得伪造指令。

**闭卷验收题**：① 为什么按数据流读比从第一行逐句翻译有效？② shared 版和 naive 版预期在哪两个指令类别上不同？③ `FFMA` 与 `HMMA` 分别支持什么结论、不能单独支持什么性能结论？④ 给一段未知 SASS，按什么顺序找 load→compute→store？现场随机抽两版复述且观察表无越界结论才通过。

**面试口述主题**：3 分钟讲“五版 GEMM 在机器指令层的数据流如何逐步变化”，强调观察结果只对当前源码、编译参数和 `sm_80` 构建成立。

**没通过时的降级/补救**：保底只完成 naive/shared/WMMA 三个锚点，但明确标红 register/`float4` 未通过；Day 3 开场最多 90 分钟补齐关键行。不得用“代码长得像向量化/WMMA”替代 SASS 证据。

## Day 3：寄存器、spill、local memory 与宽访存互证

**当日目标**：建立 ptxas 资源报告、SASS local 指令、ncu local traffic 三层证据；验证 `float4` 是否真正落成宽 global load，并解释为什么“寄存器更少”不等于更快。

**优先级**：必须完成压力前后 spill 三证据和 `float4` 宽 load 判定；尽量扫两个 tile/寄存器配置；时间不足可跳过 `__launch_bounds__` 调参。

**约 8 小时时间表**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 1.5 小时 | Day 2 补考或阅读三层证据 | 补考记录/证据清单 |
| 1 小时 | 推导寄存器压力、occupancy、ILP 和 spill 的可能关系 | 假设图 |
| 2.5 小时 | 临时副本制造压力，采集 ptxas 与 SASS | baseline/pressure 对照 |
| 1.5 小时 | ncu local traffic、`float4` 对齐与宽 load 验证 | `.ncu-rep`、宽访存结论 |
| 1.5 小时 | 复测墙钟、闭卷与口述 | 对照表、答题纸、录音 |

**必读仓库资料**：[主教材 5. Day 3：寄存器、spill 与向量化](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[register tiling](../week05_gemm_advanced/gemm_2d_thread_tiling.cu)、[`float4` 版本](../week05_gemm_advanced/gemm_vectorized_load.cu)、[GEMM ncu 笔记](../week05_gemm_advanced/ncu_notes.md)。

**分步实验**：

1. 在 `/tmp/cuda_interview_sprint/day03/` 建 baseline 与 pressure 两个临时源码；通过增加多个独立 accumulator/延长 live range 制造压力，保持数学结果和 shape 不变。
2. 两版均以 `nvcc -O3 -lineinfo -arch=sm_80 -Xptxas=-v` 构建，记录 registers、stack frame、spill stores/loads；导出 SASS，搜索 local load/store，但把“没搜到某个助记符”记为待 ncu 互证而非无 spill 定论。
3. 先运行正确性，再按本机 `ncu --query-metrics` 查可用名称，采集 local load/store sectors/bytes 及 LaunchStats；保存 `.ncu-rep`。三层只有互相一致时才下强结论。
4. 对 `float4` 检查指针与行首 16-byte 对齐、尾部处理、PTX 向量 load 及 SASS 实际 load 宽度；与标量版同 shape 对拍。源码出现 `float4` 但编译器拆成标量时，结论必须是“未形成宽 load”。
5. 同时记录正常 CUDA Event 墙钟、寄存器数、occupancy 和吞吐；明确 profiler 重放时间不可充当正常墙钟。比较只说明证据，不预设少寄存器版本更快。

**必须产出**：`register_spill_table.md`、baseline/pressure ptxas log、SASS 摘录、ncu 报告、`float4_width_verdict.md`、正常计时与正确性结果。

**闭卷验收题**：① local memory 在地址空间上“local”给谁，物理代价为何可能落到片外层次？② ptxas 显示 0 spill 是否足以证明所有 local traffic 为 0？③ 怎样证明 `float4` 最终是宽 load？④ 少寄存器为什么可能更慢？⑤ occupancy、ILP、spill 之间有哪些互相冲突的取舍？五题及现场指出三层证据位置才通过。

**面试口述主题**：4 分钟讲一次“我没有凭寄存器数猜性能，而是用资源、指令、local traffic、墙钟共同判定”的实验。

**没通过时的降级/补救**：把压力实验缩到单个 microkernel 和一个 shape，次日最多 90 分钟补考；ncu 权限缺失时保留 ptxas+SASS 并明确第三层缺证，不能宣称互证完成，列入 Day 12 复测。

## Day 4：从 scheduler/scoreboard 到可证伪的 stall 假设

**当日目标**：区分 active、eligible、issued warp，理解 scoreboard、ILP/TLP/occupancy 的关系；完成一份“指标→假设→单一修改→复测”短报告，禁止看到 stall 名称就直接宣布根因。

**优先级**：必须完成 scheduler 指标采集、单一修改和复测；尽量完成 dependent-chain vs multi-accumulator 对照；时间不足可跳过第三组 occupancy sweep。

**约 8 小时时间表**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 1.5 小时 | Day 3 补考或概念图 | 补考记录/四个 warp 数量图 |
| 1.5 小时 | 阅读 scheduler、scoreboard、stall 诊断顺序 | 指标→可能机制表 |
| 2 小时 | 采集基线：正常墙钟、资源、scheduler/warp state | baseline 报告 |
| 1.5 小时 | 提出一个假设，做唯一代码/参数修改并复测 | before/after 证据 |
| 1.5 小时 | 写短报告、闭卷、4 分钟口述 | `day04_diagnosis.md`、录音 |

**必读仓库资料**：[主教材 12. Day 10：为什么“有很多 warp”仍可能发不出指令](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[主教材 13. Day 11：怎样读 Warp Stall](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[GEMM ncu 笔记](../week05_gemm_advanced/ncu_notes.md)、[Attention 流水笔记](../week04_attention/ncu_pipeline_notes.md)。

**分步实验**：

1. 选 Day 3 可运行 GEMM；先记录 ptxas 资源与无 profiler 墙钟，再运行 `ncu --list-sections`，确认本机的 `SchedulerStats`、`WarpStateStats`、`LaunchStats`、`Occupancy`、`SpeedOfLight` 可用后采集，避免照抄跨版本 metric 名。
2. 按“高层吞吐→active/eligible/issued→主要 stall→对应 SASS/源码”顺序读报告。stall 只产生候选解释，例如 long scoreboard 可能来自 global/local load-use 距离或 spill，不能直接等于“DRAM 带宽瓶颈”。
3. 只选一个假设，例如“增加独立 accumulator 能提高 eligible/issued，隐藏依赖链延迟”；只改 accumulator 独立性，保持 shape、数据、编译参数和其余 tile 不变。
4. 复测正确性、正常墙钟、资源、scheduler 与 stall。若指标不支持假设，同样写成有效反证；不追加第二个改动救结果。

**必须产出**：`day04_diagnosis.md`，固定包含现象、高层瓶颈、active/eligible/issued、指令症状、单一假设、最小修改、正确性、复测、替代解释；附 baseline/variant 的命令、ptxas、SASS 定位和 ncu 报告。

**闭卷验收题**：① active、eligible、issued 分别是什么，为什么 active 高仍可能 issued 低？② scoreboard 跟踪什么，为什么不是缓存？③ ILP 与 TLP 各如何隐藏延迟？④ occupancy 高为什么不是性能充分条件？⑤ long scoreboard 出现后下一步查什么？能用自己的实验给出反证条件才通过。

**面试口述主题**：4 分钟复述一次真实诊断，结构必须是“观测—候选机制—单一修改—结果—仍存边界”，不得说“ncu 告诉我根因”。

**没通过时的降级/补救**：缩成教材中的 dependent-chain vs multi-accumulator microbenchmark，在 Day 5 开场最多 90 分钟重做；若只会背 stall 名，判定未通过并进入 Day 12，不能用一张 ncu 截图替代因果链。

## Day 5：固定一个 shape/type 下钻 fragment、`ldmatrix`、`mma.sync`

**当日目标**：只研究 Ampere `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 这一明确形状/类型。先画逻辑 A/B/C 矩阵到 lane/register 的映射，再做最小观察实验；目标是正确性和 SASS 证据，不追 cuBLAS 性能，也不把该映射推广到其他 shape/type。

**优先级**：必须完成纸上映射、最小正确性和 `LDMATRIX`/`HMMA` 证据；尽量比较 WMMA API 与 inline PTX 的层次；时间不足可跳过吞吐优化。

**约 8 小时时间表**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 1.5 小时 | Day 4 补考或固定 shape/type 并读语义 | 补考记录/指令契约卡 |
| 2 小时 | 纸上画逻辑矩阵→lane→register，并列 shared 地址约束 | 映射图与可核对样例 |
| 2.5 小时 | 在 `/tmp/cuda_interview_sprint/day05/` 做单 warp 最小实验 | reference PASS、源码、日志 |
| 1 小时 | 导出 PTX/SASS，定位 `ldmatrix`/`mma.sync`/HMMA | 指令摘录 |
| 1 小时 | 闭卷与 4 分钟口述 | 答题纸、录音 |

**必读仓库资料**：[主教材 6. Day 4：WMMA、PTX MMA、SASS HMMA 的关系](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[7. Day 5：`ldmatrix` 到底解决什么](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[8. Day 6：最小 `mma.sync` microkernel](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[WMMA GEMM](../week06_tensorcore/wmma_fp16_gemm.cu)。

**分步实验**：

1. 在笔记首行锁定 `m16n8k16.row.col.f32.f16.f16.f32`，列出 M/N/K、A/B 元素类型、accumulator 类型、layout 和每 warp 协作范围；所有映射结论都带此限定。
2. 先根据主教材/对应 PTX ISA 语义画 A(16×16)、B(16×8)、C/D(16×8) 的逻辑坐标，再标 lane 提供的 shared 地址与返回 registers；用几个唯一值元素做纸上追踪。若某个寄存器位映射未核准，明确留作实验验证，不猜。
3. 在 `/tmp` 写单 warp、单 tile microkernel：shared 装载已知 A/B，`ldmatrix` 取 fragment，执行一次指定 `mma.sync`，写回 D；CPU 逐元素 reference，覆盖零矩阵、单位/稀疏样例和一般小值。
4. 以 `sm_80` 构建并保留 ptxas log；导出 PTX/SASS，定位相应 `ldmatrix`、`mma.sync`/HMMA。正确性与指令证据都通过才验收；墙钟只记录，不与 cuBLAS 排名。

**必须产出**：`m16n8k16_contract.md`、lane/register 映射图、最小实验源码副本、CPU reference 与三组 PASS、ptxas log、PTX/SASS 摘录、未确认边界清单。

**闭卷验收题**：① fragment 是逻辑矩阵还是每线程连续小矩阵？② `ldmatrix` 搬运的源/目的地址空间是什么？③ `m16n8k16` 的 M/N/K 各对应哪一维归约？④ 为什么 shared layout 会影响 `ldmatrix`？⑤ 为什么不能把本日 lane mapping 套到另一 shape/type？能从零重画限定映射并解释 SASS 证据才通过。

**面试口述主题**：4 分钟从 WMMA API 下钻到 PTX `mma.sync` 和 SASS HMMA，并明确“API、虚拟 ISA、机器指令”三层证据各自边界。

**没通过时的降级/补救**：先保住 WMMA 版正确性+HMMA，再把 inline PTX/精确映射列为未通过；Day 6 开场最多 90 分钟只补单个固定 shape。绝不凭记忆推广 fragment 布局，也不以性能慢判定语义失败。

## Day 6：`cp.async` 状态机与 2-stage/3-stage 流水

**当日目标**：闭卷画出 copy→commit→wait→consume→reuse 状态机，明确 `wait_group` 不是 CTA barrier；比较同步、2-stage、3-stage 的正确性、资源、stall 和正常墙钟，正确处理 prologue/steady/epilogue 与边界 K。允许 3-stage 更慢，但必须用证据解释。

**优先级**：必须完成状态机、边界 K 正确性与三版证据表；尽量完成 Attention 或 GEMM 两种对象之一的 3-stage；时间不足可跳过参数扫描。

**约 8 小时时间表**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 1.5 小时 | Day 5 补考或状态机推演 | 补考记录/状态机图 |
| 1.5 小时 | 阅读 async group、2/3-stage 和排错表 | prologue/steady/epilogue 时间线 |
| 2.5 小时 | `/tmp` 临时副本完成 sync/2-stage/3-stage 并对拍 | 三版 PASS、资源日志 |
| 1.5 小时 | 采集正常墙钟与选定 ncu stall/occupancy | 对照表、`.ncu-rep` |
| 1 小时 | 闭卷和 5 分钟口述 | 答题纸、录音 |

**必读仓库资料**：[主教材 9. Day 7：`cp.async` 状态机](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[10. Day 8：2-stage 与 3-stage](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[11. Day 9：流水性能证据](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[double buffering 模板](../week05_gemm_advanced/gem_double_buffering.cu)、[Attention 流水笔记](../week04_attention/ncu_pipeline_notes.md)。

**分步实验**：

1. 画每个 stage 的状态：free→copies issued→committed group→wait satisfied→CTA 内需要的可见性/协作同步完成→consume→free。写明 `cp.async.wait_group` 只管理调用线程的 async group 完成关系，不等价于让整个 CTA 到达同一控制点；跨线程共同读 shared 时仍需正确 CTA 同步协议。
2. 复制 [double buffering 模板](../week05_gemm_advanced/gem_double_buffering.cu) 到 `/tmp/cuda_interview_sprint/day06/`，只改临时副本；先保留同步基线，再完成 2-stage ping-pong，最后用 `stage = tile % 3` 实现 3-stage 环形复用。
3. 分别标出 prologue 预取数量、steady 中 wait/consume/发下一组、epilogue drain；对 `K=1, 15, 16, 17, 31, 32, 33, 257` 与非整齐 M/N 做 CPU 对拍。尾 tile 使用合法边界谓词/zero fill，禁止越界读。
4. 三版记录 `-Xptxas=-v` 的 register/shared/spill、SASS 中 async copy/commit/wait 类别、正常 CUDA Event 墙钟；确认本机 metric 后用相同 ncu sections 对比 scheduler/stall/occupancy。stage 增多若提高 shared/register 压力、降低 occupancy 或工作量不足以摊薄 prologue，可合理更慢。

**必须产出**：状态机图、三版临时源码/diff、边界 K 测试矩阵、ptxas/SASS/ncu/正常墙钟对照表、`pipeline_explanation.md`；报告必须把 correctness、资源、stall、墙钟分栏，不能用单个 stall 下降代替加速结论。

**闭卷验收题**：① commit 的对象和 wait 的条件是什么？② 为什么 wait 不是 CTA barrier？③ 2-stage 与 3-stage 的 stage 复用条件有何不同？④ prologue、steady、epilogue 各负责什么？⑤ 边界 K 如何避免读非法地址并保证数学上补零？⑥ 3-stage 为什么可能更慢？状态机、边界测试和证据表全通过才验收。

**面试口述主题**：5 分钟解释一次异步流水设计，先讲正确性协议，再讲重叠收益，最后给出资源/occupancy/小 grid 让更多 stage 失效的反例。

**没通过时的降级/补救**：退回单 warp 或单 CTA copy→consume microtest，先验证 group 与同步语义；Day 7 开场最多 90 分钟补考。若 3-stage 未完成，保留 sync/2-stage 的有效证据并明确未通过，禁止从 2-stage 外推 3-stage 性能。

## Day 7：Hopper 对照与五层综合证据链

**当日目标**：上午只基于官方语义对照 Ampere `cp.async + mma.sync` 与 Hopper `TMA + WGMMA`，说清 Cluster/DSM、`mbarrier`、warp specialization；下午从现有 GEMM 或 Attention 完成“源码→资源→SASS→ncu→单一修改→复测”综合报告。A100 只实测 Ampere 路径，不伪造 Hopper 性能。

**优先级**：必须完成架构边界口述和一条 A100 综合证据链；尽量加入 Hopper 迁移判断；时间不足可跳过任何 H100 编译/性能实验。

**约 8 小时时间表**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 1.5 小时 | Day 6 补考或 Hopper 主循环语义对照 | 补考记录/对照图 |
| 1.5 小时 | TMA/WGMMA、Cluster/DSM、warp specialization 口述准备 | 架构边界卡 |
| 1 小时 | 选定 GEMM/Attention，冻结 shape、基线、指标与假设 | 实验契约 |
| 2.5 小时 | A100 基线→单一修改→复测，采集五层证据 | 两组可复核产物 |
| 1.5 小时 | 写综合报告、闭卷和 5 分钟口述 | `day07_integrated_report.md`、录音 |

**必读仓库资料**：[主教材 14. Day 12：Hopper 不是“把 A100 每项都加宽”](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[TMA](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[WGMMA](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[Thread Block Cluster 与 DSM](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[综合报告模板](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[GEMM ncu 笔记](../week05_gemm_advanced/ncu_notes.md)、[Attention ncu 笔记](../week04_attention/ncu_pipeline_notes.md)。

**分步实验**：

1. 画两条主循环：Ampere 由线程/warp 发 `cp.async` 到 shared、warp 以 `ldmatrix + mma.sync` 消费；Hopper 由 descriptor 驱动 TMA bulk tensor copy、以 transaction-aware barrier 管 stage、warp-group 发异步 WGMMA。注明具体 WGMMA 变体可能要求 `sm_90a`，不写未经本机验证的通用编译命令。
2. 口述 Cluster 保证同一 GPC 内一组 CTA 协作，DSM 是 cluster 内各 CTA shared 的分布式地址空间，不是全 GPU shared/自动缓存；warp specialization 是 producer/consumer 角色分工，不保证天然更快。
3. 下午二选一：GEMM 可用 [register tiling](../week05_gemm_advanced/gemm_2d_thread_tiling.cu)；Attention 可沿 [现有流水记录](../week04_attention/ncu_pipeline_notes.md) 选其对应对象。把可运行版本复制到 `/tmp/cuda_interview_sprint/day07/`，冻结硬件、shape、输入、编译参数、正常计时和 ncu sections。
4. 基线依次留源码数据流、`-Xptxas=-v` 资源、关键 SASS、正常墙钟与 ncu；基于证据提出一个可证伪假设，只改一个因素（如 accumulator 独立性、padding 或 stage 数），重新做正确性和同组采集。
5. 报告结尾写 Hopper 迁移判断：哪些 copy 可能适合 TMA、哪些 warp MMA 可能重构为 WGMMA、是否真的需要 Cluster/DSM；全部标为语义/设计判断，不写 H100 实测数字。

**必须产出**：Ampere/Hopper 对照图、架构边界卡、实验契约、baseline/variant 的源码 diff、正确性、资源、SASS、ncu 和正常墙钟、`day07_integrated_report.md`、5 分钟口述录音。报告必须区分仓库历史证据与本次本人重跑证据。

**闭卷验收题**：① TMA 为什么不是更宽的 `cp.async`？② `mma.sync` 与 WGMMA 的协作单位和同步模型有何变化？③ Cluster/DSM 的作用域和生命周期边界是什么？④ warp specialization 解决什么、付出什么？⑤ 为什么不能拿 A100 的 `cp.async` 数据声称 Hopper TMA 性能？⑥ 用自己的综合报告串起源码、资源、SASS、ncu、修改、复测。六题和报告证据链全部通过才结束深水区。

**面试口述主题**：5 分钟回答“如果把 A100 kernel 迁到 Hopper，我会如何判断 TMA/WGMMA/Cluster 是否值得用”，先讲语义差异，再讲本次 A100 证据，最后明确未在 H100 实测的边界。

**没通过时的降级/补救**：上午概念未过则只保留官方语义对照，不做任何 Hopper 性能声称；下午链路未过则缩成一个 kernel、一个 shape、一个修改，失败项进入 Day 12 薄弱清单。补考必须占当天 8 小时时间盒；若 Day 7 结束仍缺 ncu 权限或 H100，分别标为“证据缺失”和“计划外硬件边界”，不能用历史报告或 A100 数字冒充本人/Hopper 验收。
