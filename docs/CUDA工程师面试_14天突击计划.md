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

1. `mkdir -p /tmp/cuda_interview_sprint/day01 && cd /tmp/cuda_interview_sprint/day01`；执行 `nvidia-smi > environment.txt`、`nvcc --version >> environment.txt`、`ncu --version >> environment.txt`。`commands.sh` 只手工记录已经执行的命令，或用 `printf '%s\n' 'nvcc ...' >> commands.sh` 逐条写入安全的纯文字；不要把含 `$()`、反引号或未确认变量的命令通过 here-document 再执行。所有生成物留在此目录。
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

**优先级**：必须完成五版观察表，并为 naive、shared、register tiling、`float4`、WMMA 每一版同时保留有行号/函数名的 PTX 与 SASS 关键路径证据；时间不足只能跳过非主循环指令和精确周期推断，不能删减版本或任一指令层次。

**约 8 小时时间表**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 1.5 小时 | Day 1 最小补考；无需补考时阅读五版源码并画统一数据流框架 | 补考记录或一张统一框架图 |
| 4.5 小时 | 从完整 HEAD 快照编译五版，并逐版完成 PTX+SASS 双层取证 | 五版双层产物、日志与摘录 |
| 1 小时 | 填观察表，横向比较 load/shared/FFMA/HMMA/control/store | `instruction_observation.md` |
| 1 小时 | 闭卷验收和 3 分钟口述 | 答题纸、录音索引 |

**必读仓库资料**：[主教材 4. Day 2：怎样开始读 PTX 与 SASS](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[naive](../week04_gemm/gemm_naive/gemm_naive.cu)、[shared tiled](../week04_gemm/gemm_tiled/gemm_tiled.cu)、[register tiling](../week05_gemm_advanced/gemm_2d_thread_tiling.cu)、[`float4`](../week05_gemm_advanced/gemm_vectorized_load.cu)、[WMMA](../week06_tensorcore/wmma_fp16_gemm.cu)。

**分步实验**：

1. 只画一张五版共用的数据流框架：输入地址生成、global load、可选 shared stage、register fragment/accumulator、FFMA/HMMA、边界控制、global store；观察表中每版另写一句与框架的差异，不重复画五张图。
2. 为避免工作区变化和单文件遗漏，先执行 `mkdir -p /tmp/cuda_interview_sprint/day02/source`，再从仓库根目录执行 `git archive HEAD | tar -x -C /tmp/cuda_interview_sprint/day02/source` 导出完整提交树；五个当前源文件经检查可直接编译，正常使用该 HEAD 快照中的文件，所有修复也只发生在 `/tmp` 副本。
3. 五版都在 `/tmp/cuda_interview_sprint/day02/` 生成双层产物：用 `nvcc -O3 -lineinfo -arch=compute_80 -ptx <source> -o <name>.ptx` 生成 PTX；再用 `nvcc -O3 -lineinfo -arch=sm_80 -Xptxas=-v <source> -o <name>.sm80` 生成含 cubin 的 executable 和资源日志，最后用 `cuobjdump --dump-sass <name>.sm80 > <name>.sass` 导出 SASS。记录每条完整命令和目标 kernel 名。
4. 若确需回看历史版本，先人工检查候选 commit 的 diff、配套文件和可构建状态，得到 `verified_sha` 后用 `git archive <verified_sha> | tar -x -C /tmp/cuda_interview_sprint/day02/source-history` 导出完整树；不能假设“最近 commit”就是完成版，也不能用单文件 `git show` 丢失相对 include。若验证过的完整历史树仍不可用，当日判定未通过并进入补考，不虚构证据。
5. 对每一版分别在 PTX 和 SASS 中按数据流摘录实际存在的项：索引/地址计算、global load、shared load/store、主计算 `fma`/`FFMA` 或 `mma`/`HMMA`、控制/同步、global store。某版没有 shared 或同步指令时在对应项写“按该实现不适用”并给源码数据流理由，不强求不存在的类别；但 PTX、SASS 两个层次都必须实际观察并留定位信息。
6. 表格列固定为“版本、与统一数据流框架的一句差异、PTX 关键证据、SASS 关键证据、资源、我能证明的结论、仍不能证明的结论”，naive/shared/register tiling/`float4`/WMMA 五行必须齐全。`float4` 的宽 load 留到 Day 3 作严格验证。

**必须产出**：五版各自的 PTX、cubin/executable、SASS、ptxas log 和正确性 PASS，一张统一数据流框架图，以及含五行的 `instruction_observation.md`；每行必须有一句版本差异和“PTX 关键证据”“SASS 关键证据”两列。任何版本缺少任一层证据都判定当日未通过，不能以 TODO/FAIL 或另一版证据代替。

**闭卷验收题**：① 为什么按数据流读比从第一行逐句翻译有效？② shared 版和 naive 版预期在哪两个指令类别上不同？③ PTX 与 SASS 各证明哪一层，为什么不能相互替代？④ `FFMA` 与 `HMMA` 分别支持什么结论、不能单独支持什么性能结论？⑤ 给一段未知 PTX 和对应 SASS，分别按什么顺序找 load→compute→store？现场随机抽两版同时指出两层证据，且五版双层观察表齐全、无越界结论才通过。

**面试口述主题**：3 分钟讲“五版 GEMM 在机器指令层的数据流如何逐步变化”，强调观察结果只对当前源码、编译参数和 `sm_80` 构建成立。

**没通过时的降级/补救**：任一版本缺 PTX、SASS、正确性或关键路径定位，都把该版判为未通过；Day 3 开场最多 90 分钟只补缺失版本/层次并计入当日 8 小时，仍失败则进入 Day 12 薄弱清单。不得删成三版“保底”，不得用“代码长得像向量化/WMMA”、TODO/FAIL、历史截图或单层证据替代五版双层验收。

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
3. 先运行正确性与 `ncu --list-sections`，再按本机 `ncu --query-metrics` 查可用 section/metric 名称；选择本机存在的 section 后执行 `ncu --section <available_section> -o /tmp/cuda_interview_sprint/day03/local-traffic ./app`，必要时补充已确认存在的 local load/store sectors/bytes metric。ncu 通常生成 `.ncu-rep`，实际文件名以命令输出为准；三层只有互相一致时才下强结论。
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

1. 选 Day 3 可运行 GEMM；先记录 ptxas 资源与无 profiler 墙钟，再运行 `ncu --list-sections`，确认本机实际存在的 scheduler/warp state/launch/occupancy/SpeedOfLight section 名后，用 `ncu --section <available_section> -o /tmp/cuda_interview_sprint/day04/baseline ./app` 保存报告，variant 使用另一输出前缀；ncu 通常生成 `.ncu-rep`，实际文件名以命令输出为准。避免照抄跨版本 section/metric 名。
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
4. 明确分两条命令取证：`nvcc -O3 -arch=compute_80 -ptx source.cu -o file.ptx` 生成 PTX，在该文件中查 `mma.sync`/`ldmatrix`；`nvcc -O3 -lineinfo -arch=sm_80 -Xptxas=-v source.cu -o app` 生成 binary/cubin 并保留资源日志，再执行 `cuobjdump --dump-sass ./app > file.sass` 查 HMMA/LDSM 类机器指令，具体名称以实际输出为准。不要假设默认 binary 必然可 dump 出 PTX；正确性与双层指令证据都通过才验收，墙钟只记录、不与 cuBLAS 排名。

**必须产出**：`m16n8k16_contract.md`、lane/register 映射图、最小实验源码副本、CPU reference 与三组 PASS、ptxas log、PTX/SASS 摘录、未确认边界清单。

**闭卷验收题**：① fragment 是逻辑矩阵还是每线程连续小矩阵？② `ldmatrix` 搬运的源/目的地址空间是什么？③ `m16n8k16` 的 M/N/K 各对应哪一维归约？④ 为什么 shared layout 会影响 `ldmatrix`？⑤ 为什么不能把本日 lane mapping 套到另一 shape/type？能从零重画限定映射并解释 SASS 证据才通过。

**面试口述主题**：4 分钟从 WMMA API 下钻到 PTX `mma.sync` 和 SASS HMMA，并明确“API、虚拟 ISA、机器指令”三层证据各自边界。

**没通过时的降级/补救**：先保住 WMMA 版正确性+HMMA，再把 inline PTX/精确映射列为未通过；Day 6 开场最多 90 分钟只补单个固定 shape。绝不凭记忆推广 fragment 布局，也不以性能慢判定语义失败。

## Day 6：`cp.async` 状态机与 2-stage/3-stage 流水

**当日目标**：分别掌握 raw inline PTX 与 `cuda::pipeline<thread_scope_block>` 两层异步模型，明确二者的对应与不等价；完成正确的 3-stage，并比较同步、2-stage、3-stage 的正确性、资源、stall 和正常墙钟，正确处理 prologue/steady/epilogue 与边界 K。允许 3-stage 更慢，但必须用证据解释。

**优先级**：必须完成 raw PTX microtest、两层语义对照、正确的 3-stage、边界 K 验证和 sync/2-stage/3-stage 三版证据表；3-stage 不是“尽量完成”。时间不足只能跳过额外参数扫描，不能把 3-stage 降级为口头设计。

**约 8 小时时间表**：

正常模式（无 Day 5 补考，总计 8 小时）：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 1 小时 | 阅读并分开画 raw PTX 与 `cuda::pipeline` 两层状态机 | 两张状态机与对应/不等价表 |
| 1 小时 | 完成 inline PTX 最小 wrapper/microtest | raw copy/commit/wait/consume PASS |
| 4 小时 | 复用完整 sync/2-stage 基线，基于教材骨架实现并调通 3-stage | 三版 PASS、资源/SASS/计时/ncu |
| 1 小时 | 边界 K 对拍与 Compute Sanitizer | 边界矩阵、memcheck 0 errors |
| 1 小时 | 三版表格、闭卷和 5 分钟口述 | 对照表、答题纸、录音 |

Day 5 补考模式（总计 8 小时）：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 1.5 小时 | 只重做 Day 5 未通过的最小验收 | 补考结论与证据定位 |
| 0.5 小时 | Day 6 概念定向阅读，快速重画 raw PTX/API 边界 | 两层对应与不等价速查表 |
| 0.5 小时 | 复用教材 wrapper 完成 raw PTX microtest，不做扩展 | raw copy/commit/wait/consume PASS |
| 4 小时 | 复用完整 sync/2-stage 基线，实现并调通正确的 3-stage | 三版 PASS、资源/SASS/计时/ncu |
| 0.75 小时 | 边界 K 对拍与 Compute Sanitizer | 边界矩阵、memcheck 0 errors |
| 0.75 小时 | 三版证据表、闭卷和压缩口述 | 对照表、答题纸、录音 |

补考模式只压缩重复阅读、扩展实验和报告美化，不删除 raw PTX/API 区分、正确 3-stage、边界正确性或 sync/2-stage/3-stage 三版证据。压缩后任一硬验收仍未完成，Day 6 判定未通过并按通用补考规则处理。

**必读仓库资料**：[主教材 9. Day 7：`cp.async` 状态机](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[10. Day 8：2-stage 与 3-stage](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[11. Day 9：流水性能证据](CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)、[double buffering 模板](../week05_gemm_advanced/gem_double_buffering.cu)、[Attention 流水笔记](../week04_attention/ncu_pipeline_notes.md)。

**分步实验**：

1. 第一层只研究每线程 raw PTX：`cp.async` 发 copy，`cp.async.commit_group` 把调用线程此前未提交的 copy 组成 async group，`cp.async.wait_group` 追踪该调用线程发出的 group；画 copy→commit group→wait condition→consume 状态机，并写明 wait 不等于 CTA barrier，跨线程共同消费 shared 仍需满足正确的 CTA 可见性/会合协议。
2. 先在 `/tmp/cuda_interview_sprint/day06/` 完成 inline PTX 最小 wrapper/microtest，只验证 raw copy/commit/wait/consume、对齐和尾部 zero-fill 语义；用正确性与 SASS 确认理解，不先混入 GEMM 多级流水。
3. 第二层再分析现有 `cuda::pipeline<cuda::thread_scope_block>` / cooperative_groups 模板：它有 block 共享的 pipeline state 以及 producer acquire/commit、consumer wait/release 协议。编译器可能把 `cuda::memcpy_async` 降为 `cp.async` 及相关同步，但 API 的 commit/wait 不能直接等同于 PTX `commit_group/wait_group`；在对照表中分别写 API 保证与实际 SASS 观察。
4. 不从零再写 sync 和 2-stage：复用仓库中已完成的基线，或按 Day 2 的完整 `git archive` 快照方法把已人工验证的完整树导出到 `/tmp`。主实现只基于教材 3-stage 骨架，把 2-stage 的共享 pipeline state/buffer 扩为 3-stage，完成环形 stage 生命周期。
5. 分别标出 prologue 预取数量、steady 中 wait/consume/发下一组、epilogue drain；对 `K=1, 15, 16, 17, 31, 32, 33, 257` 与非整齐 M/N 做 CPU 对拍，并运行 `compute-sanitizer --tool memcheck ./app ...`。尾 tile 使用合法边界谓词/zero fill，禁止越界读。
6. 三版记录 `-Xptxas=-v` 的 register/shared/spill、SASS 中实际 async copy/同步类别、正常 CUDA Event 墙钟；先 `ncu --list-sections`，再用本机存在的相同 sections 对比 scheduler/stall/occupancy。stage 增多若提高 shared/register 压力、降低 occupancy 或工作量不足以摊薄 prologue，可合理更慢。

**必须产出**：raw PTX 与 `cuda::pipeline` 两张状态机、inline PTX microtest、两层对应/不等价表、正确的 3-stage 临时源码/diff、sync/2-stage/3-stage 边界 K 与 memcheck 结果、ptxas/SASS/ncu/正常墙钟对照表、`pipeline_explanation.md`；报告必须把 correctness、资源、stall、墙钟分栏，不能用单个 stall 下降代替加速结论。

**闭卷验收题**：① raw PTX 中每线程 commit 的对象和 wait 的条件是什么？② 为什么 raw wait 不是 CTA barrier？③ `cuda::pipeline<thread_scope_block>` 的共享状态和 producer/consumer 协议是什么，为什么 API commit/wait 不能直接等同 PTX group 指令？④ 2-stage 与 3-stage 的 stage 复用条件有何不同？⑤ prologue、steady、epilogue 各负责什么？⑥ 边界 K 如何避免非法地址并保证数学补零？⑦ 3-stage 为什么可能更慢？microtest、正确 3-stage、边界测试和三版证据表全通过才验收。

**面试口述主题**：5 分钟解释一次异步流水设计，先讲正确性协议，再讲重叠收益，最后给出资源/occupancy/小 grid 让更多 stage 失效的反例。

**没通过时的降级/补救**：raw microtest、两层语义、正确 3-stage、边界/sanitizer 或三版比较任一缺失，Day 6 都判定未通过；Day 7 开场最多 90 分钟补考并计入 8 小时。不得把 3-stage 降级为口头完成，也不得从 2-stage 外推其正确性或性能。

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

## Day 8：限时手写 Reduction 与 Shared Transpose

**当日目标**：不看完整答案，从空白分别写出 reduction 基础版与 shared-memory transpose；先用 CPU reference 证明任意尺寸正确，再按 reduction 的 warp/block 分层和 transpose 的 naive→shared→padding 阶梯优化。仓库 TODO 或现有文件只能提供编译、测试和计时框架，核心 kernel 必须本人写。

**优先级**：必须完成两个正确基础版、非规则边界、可靠计时和 sanitizer；尽量完成全部优化阶梯、吞吐与 Roofline/带宽上限解释；时间不足可跳过额外 block size 扫描，不能跳过从空白实现与正确性。

**正常模式（总计 8 小时）**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 0.5 小时 | 回忆接口、写 CPU reference 与测试矩阵 | 测试契约 |
| 2 小时 | 75 分钟基础 reduction，45 分钟 warp/block 优化 | 两版 reduction、对拍 |
| 2 小时 | 30 分钟 naive、60 分钟 shared、30 分钟 padding transpose | 三版 transpose、对拍 |
| 1.5 小时 | warmup+重复计时、memcheck/synccheck、吞吐 | 原始结果与命令 |
| 1 小时 | 线程映射、同步、bank conflict、带宽上限复盘 | 证据表与图 |
| 1 小时 | 闭卷重写关键段与 4 分钟口述 | 答题纸、录音 |

**必要时补考模式（总计 8 小时，含前日最多 1.5 小时）**：1.5 小时只补 Day 7 最小失败项；1 小时 reference/测试契约；1.75 小时 reduction 两版；1.75 小时 transpose naive/shared/padding；1 小时 sanitizer 与可靠计时；0.5 小时证据摘要；0.5 小时闭卷与口述。压缩来源是额外参数扫描、Roofline 排版和优化复测次数，不删除核心正确性或概念、代码、证据、面试四类输出。

**必读仓库资料**：[完整 reduction](../week03_parallel/reduction_sum_full/reduction_sum_full.cu)、[Week 3 记录](../notes/week03.md)、[operator practice transpose](../operator_practice/transpose/transpose.cu)、[Week 2 transpose](../week02_memory/transpose/transpose.cu)。只在首轮独立实现和失败定位之后对照，禁止边看边抄。

**分步练习**：

1. 在 `/tmp/cuda_interview_sprint/day08/` 新建个人练习副本；可以复制 main、错误检查、CPU reference 框架，但先移除 kernel 实现。测试至少含 `N=0,1,31,32,33,1000,1048583` 和 transpose 的 `1x1, 31x33, 1000x513`。
2. 限时 75 分钟写正确优先的多 block reduction 基础版，明确 partial sums 的收口方式；通过后再写 warp shuffle/block 分层版。每次优化只改一个因素，并重新对拍。
3. 限时完成 transpose：先写 naive 映射，再以协作 load/store、CTA 同步实现 shared tile，最后把 leading dimension 改为 `TILE_DIM+1`。边界线程也必须执行不会死锁的同步路径。
4. 每版先 warmup，再用 CUDA Event 重复至少 100 次；报告中写实际读写字节口径、有效 GB/s 或元素吞吐，并与设备可达带宽/Roofline 上限比较，不沿用旧数字。
5. 对适用版本运行 `compute-sanitizer --tool memcheck`；shared/sync 版本另运行 `--tool synccheck`。保存命令、shape、PASS/错误原文，异常不能被平均性能掩盖。
6. 画出线程到元素/矩阵坐标映射，逐个标注同步位置与必要性；用 bank 地址推导解释 padding，而不是仅凭“加 1 更快”。

**必须产出**：测试矩阵与 CPU reference、本人写的两版 reduction 和三版 transpose、边界对拍、memcheck/synccheck、warmup+重复计时原始数据、吞吐/带宽表、线程映射与 bank conflict 推导、一次优化前后证据、闭卷答题与录音索引。

**Day 8～11 共用数值测试契约**：CPU reference 使用 `double`/FP64 累加；逐元素先检查 GPU 输出为 finite，再以 `abs_err <= atol + rtol * abs(ref)` 判定，并同时报告 `max_abs` 与 `max_rel`。近零 reference 不能只看 relative error。每个 dtype/shape 必须声明自己的 `atol/rtol`；FP32 点式运算可从 `atol=1e-5, rtol=1e-4` 起步，但这不是普适阈值，长归约要结合 N、输入范围与累加顺序合理调宽，并报告误差如何随 N 变化。零维输入由 host 明确短路，不发起 0-grid launch。

**闭卷验收题**：① 从空白写出不会漏尾元素的 block reduction 核心；② 为什么 block 间不能靠 `__syncthreads()` 收口？③ transpose 的 load 和 store 各如何合并？④ shared tile 为什么需要同步，为什么同步不能放进分歧分支？⑤ `TILE_DIM+1` 怎样改变 bank 映射？⑥ 有效带宽字节数如何计算？两类 kernel 均通过非规则 shape 与 sanitizer 才通过。

**面试口述主题**：4 分钟讲“我怎样先得到正确基础版，再用 warp/block 归约与 shared padding 接近带宽上限”，必须包含一个边界 bug 或失败优化及其证据。

**没通过时的降级/补救**：把失败收缩为一个最小 shape，保留 sanitizer 报告和错误索引；Day 9 开场最多 90 分钟重写失败核心并计入 8 小时。不能复制仓库答案宣称通过；再次失败进入 Day 12 薄弱清单。

## Day 9：Softmax、GEMV 与 RMSNorm 基础正确版

**当日目标**：从空白独立写出稳定 softmax 基础核、warp-per-row GEMV 和 RMSNorm 基础核；三个主题各有 FP64 CPU reference、可审计边界、正确性与 sanitizer。thread-per-row GEMV、融合和完整性能证据只在三项基础验收通过后进行。

**优先级**：必须完成三个独立核心及稳定性、非 2 的幂长度、多 shape reference、finite/误差检查与 sanitizer；尽量完成 thread-per-row 对照、两版性能比较、RMSNorm 融合和完整性能证据；时间不足可跳过这些优化以及向量化、persistent kernel 与调参。

**正常模式（总计 8 小时）**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 0.5 小时 | 接口、CPU reference、误差与 shape 契约 | 测试矩阵 |
| 2 小时 | stable softmax：max→sum→normalize | 正确基础版、极值测试 |
| 1.75 小时 | warp-per-row GEMV：合并读取与 warp 归约 | 正确基础版 |
| 1.75 小时 | RMSNorm：平方和归约、epsilon、gamma | 正确基础版 |
| 1 小时 | 三项 sanitizer、多 shape 汇总；余时才做性能/融合 | 验收证据表 |
| 1 小时 | 闭卷核心段与 4 分钟口述 | 答题纸、录音 |

**必要时补考模式（总计 8 小时，含前日最多 1.5 小时）**：1.5 小时补 Day 8；0.5 小时测试契约；1.5 小时 softmax；1.5 小时 warp-per-row GEMV；1.5 小时 RMSNorm；1 小时三项 sanitizer/边界汇总；0.5 小时闭卷与口述。压缩掉 thread-per-row、性能比较、融合和排版；若无补考，释放的 1.5 小时才用于这些尽量项。

**必读仓库资料**：[softmax](../week04_gemm/softmax/softmax.cu)、[已跟踪 GEMV](../week05_inference/gemv.cu)、[Week 5 decode 教材](Week5增强版_LLM推理优化与decode.md)、[LayerNorm/RMSNorm 与融合教材](../cuda_deep_course/course/volume06_operators/05_LayerNorm_RMSNorm与融合.md)。只在独立版本完成或卡点复盘时阅读核心实现。

**分步练习**：

1. 在 `/tmp/cuda_interview_sprint/day09/` 建练习文件并独立写核心。若想使用本机 `week05_inference/fused_rmsnorm.cu`，先运行 `git ls-files --error-unmatch week05_inference/fused_rmsnorm.cu`；失败表示它是“本机可选、当前未纳入 Git”，只能在 `test -f` 成功后复制到 `/tmp` 作脚手架，不能成为计划依赖或被擅自提交。不存在时依据已跟踪教材伪代码在 `/tmp` 新建个人练习。
2. softmax 完成每行 max、`exp(x-max)` 求和和 normalize；shape 至少让 rows 与 D 分别覆盖 `1,31,32,33`、一个其他非 2 的幂和一个大值，输入含 `-1000,0,1000`、全相等和随机极值。按共用契约检查有限性、逐元素误差及每行和。
3. 独立写 warp-per-row GEMV；M、K 分别覆盖 `1,31,32,33` 和奇数，从地址推导合并读取及 shuffle 归约。thread-per-row 实现、两版计时/带宽比较均为尽量项，不阻塞基础验收。
4. RMSNorm 写 `sum(x²)`、`rsqrt(mean+epsilon)` 和 gamma；rows 与 D 分别覆盖 `1,31,32,33`、奇数及较大值，并测试全零/很小方差、不同 epsilon。基础版通过后才考虑 residual 融合。
5. 三组代码均运行 memcheck，含共享同步的版本运行 synccheck；按 dtype/shape 记录 `atol/rtol/max_abs/max_rel/finite/PASS`。零维由 host 短路。性能与融合慢或未做不影响基础验收，但错误结果不能进入性能表。

**必须产出**：三个 FP64 CPU reference、稳定 softmax、warp-per-row GEMV、RMSNorm 基础版、多 shape/极值对拍、finite/误差表、sanitizer、地址/归约图、闭卷答题和录音。thread-per-row、两版性能表和融合若完成则作为附加证据。至此仍明确包含一个本人独立 GEMV 实现。

**闭卷验收题**：① 不减 max 的 softmax 为何会溢出/下溢？② 非 2 的幂归约怎样保证不漏元素？③ warp-per-row GEMV 的相邻 lane 分别访问哪里？④ GEMV 为何常受带宽限制？⑤ RMSNorm 与 LayerNorm 差什么？⑥ epsilon 放在哪里？能现场重写三个核心索引并通过约定 shape、finite、误差与 sanitizer 才通过。

**面试口述主题**：4 分钟串讲 decode 中 softmax、GEMV、RMSNorm 的数值稳定性、线程映射和 memory-bound 特征，明确基础正确版与已验证优化的边界。

**没通过时的降级/补救**：任一基础版失败都记录最小输入、CPU/GPU 差值和 sanitizer；Day 10 开场最多 90 分钟只补失败基础版。融合未完成可列为 Day 10 工程收口，但不能用融合 TODO 掩盖 RMSNorm 核心未通过。

## Day 10：闭卷 Tiled GEMM 与推理 TODO 工程收口

**当日目标**：闭卷完成边界安全的 tiled GEMM，完成 CUDA Graph 基本 capture/instantiate/replay/cleanup 状态机，并独立写 Dequant GEMV 基础正确版与双层指标。Fused RMSNorm 只在 Day 9 基础核已通过且本机可选脚手架存在时做接口验收，不重复要求核心核。

**优先级**：必须完成 GEMM 正确性、Graph 基本生命周期、Dequant GEMV 实现误差与量化损失分栏、sanitizer 及四类输出；尽量完成 Fused RMSNorm 接口实装、性能调参和三项 ncu。时间不足可跳过 Graph 节点更新、更多量化格式及全部深度性能优化。

**正常模式（总计 8 小时）**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 2.5 小时 | 90 分钟闭卷 GEMM，60 分钟边界/对拍修复 | tiled GEMM、测试 |
| 1.5 小时 | CUDA Graph capture→instantiate→replay→destroy | 生命周期正确实现 |
| 0.5 小时 | Fused RMSNorm 接口验收（可选实装） | 接口检查表 |
| 2 小时 | Dequant GEMV 与双层 reference | 实现误差与量化损失表 |
| 0.75 小时 | sanitizer、warmup+重复计时 | 证据摘要 |
| 0.75 小时 | 闭卷与 5 分钟口述 | 答题纸、录音 |

**必要时补考模式（总计 8 小时，含前日最多 1.5 小时）**：1.5 小时补 Day 9；2 小时 tiled GEMM 正确性；1.25 小时 Graph 基本状态机；1.75 小时 Dequant GEMV 与双层指标；0.75 小时三项边界/sanitizer；0.25 小时 RMSNorm 接口检查；0.5 小时闭卷与口述。压缩性能调参、ncu、融合实装和排版，不删除最低正确性或四类输出。

**必读仓库资料**：[tiled GEMM](../week04_gemm/gemm_tiled/gemm_tiled.cu)、[Week 5 decode 教材（RMSNorm/Graph/Dequant 核心）](Week5增强版_LLM推理优化与decode.md)、[RMSNorm 与融合教材](../cuda_deep_course/course/volume06_operators/05_LayerNorm_RMSNorm与融合.md)、[CUDA Graph 教材](../cuda_deep_course/course/volume07_async_system/04_CUDA_Graph.md)。这些均为已跟踪主依赖。

本机的 `week05_inference/decode_graph.cu`、`fused_rmsnorm.cu`、`dequant_gemv.cu` 仅是**本机可选、当前未纳入 Git**的脚手架。每个都先用 `git ls-files --error-unmatch <path>` 判断；若失败，不得把它当作克隆后必然存在的资料。仅在 `test -f <path>` 成功时复制到 `/tmp` 使用；不存在时依据上述已跟踪教材伪代码在 `/tmp` 新建个人练习，绝不提交或覆盖这些本机文件。

**分步练习**：

1. 限时 90 分钟从空白写 GEMM 核心：`row/col` 索引、A/B 协作加载、越界补零、同步、tile 累加、边界写回；测试 `1x1x1, 31x33x35, 127x129x65, 256x256x256`，用 CPU 与可用时 cuBLAS 双对照。
2. CUDA Graph 为每个 CUDA API 做 `CUDA_CHECK`/状态检查。capture 一旦开始，即使被 invalidated，也要在同一 stream 调用 EndCapture 收口并处理 graph 可能为 `nullptr`；instantiate 成功后可销毁 graph。replay warmup 与正式计时分开，launch 后先完成 stream/event 同步，再 destroy executable graph。使用单出口和“已创建”状态位依次清理 exec/graph/event/stream，检查关键 launch、sync、destroy 返回，任何对象只销毁一次。
3. Fused RMSNorm 的最低要求只是检查 Day 9 基础核与 residual/gamma 接口的输入输出别名、epsilon、D 尾部和 launch 错误；只有可选脚手架存在且基础核已通过才实装，否则写接口验收表，不阻塞本日三项必须任务。
4. Dequant GEMV 的**实现误差**以相同 Q/scale 的 FP64 CPU dequant+GEMV 为 reference，按 `abs/rel/finite` 判 PASS。**量化损失**以原 FP 权重输出为 reference，至少报告 RMSE、`NRMSE=RMSE/(ref RMS+epsilon)`（写明 epsilon）和 max_abs，可选 cosine；质量指标不参与 kernel 对错判定。
5. Dequant 边界必须覆盖：全零行的 scale 约定（例如 `scale=1,Q=0`，或明确自己的实现选择）、scale 正且 finite、int8 signed 极值及对称量化通常使用 `[-127,127]`、`K=1,31,32,33`、输出行 N 非 warp/block 整倍数。全部核心跑 memcheck，适用时 synccheck；性能计时与三项 ncu 均为尽量项，若做则把 capture/instantiate 一次性成本与 replay 延迟分开。

**必须产出**：本人闭卷 tiled GEMM 与 CPU/可用时 cuBLAS 对拍、Graph 状态机和多次 replay/cleanup、Dequant GEMV 基础核与双层指标、RMSNorm 接口检查表、sanitizer、命令/环境、闭卷答题与录音。最低验收不要求性能调参、三项 ncu 或 Fused RMSNorm 实装。

**闭卷验收题**：① 两个 shared tile 的索引如何推导？② 越界 load 为何补零？③ 两次同步各保护什么？④ invalidated capture 如何 EndCapture 收口且避免二次销毁？⑤ graph 与 executable graph 生命周期如何区分？⑥ 实现误差与 RMSE/NRMSE 量化损失为何不能混判？现场通过非规整 GEMM、三次 Graph replay/cleanup、Dequant 边界与双层指标才通过。

**面试口述主题**：5 分钟回答“我如何把教学 kernel 变成可重复推理路径”，从 GEMM 正确性讲到 Graph 生命周期、融合边界与量化误差归因。

**没通过时的降级/补救**：优先保留一个正确 GEMM、最短 Graph replay 链和一种量化格式；失败项带最小复现进入 Day 11 开场最多 90 分钟补考。不得删除核心正确性、四类输出或把未完成 TODO 说成工程收口。

## Day 11：A100 旗舰项目收口（GEMM 或 Attention 二选一）

**当日目标**：默认选择 GEMM；也可在开场改选 Attention，但当天只能做一个。把单一项目整理成可复现的 baseline→版本阶梯→正确性矩阵→可靠计时→Roofline→ncu→PTX/SASS→失败优化→差距与下一步证据链，并形成 5/10 分钟项目表达。

**优先级**：必须冻结一个项目、重跑全部关键数字、保留至少一次失败优化并完成报告；尽量与 cuBLAS/CUTLASS 实测对照；无法构建库时必须给可审计的理论上限与差距，时间不足可跳过新优化版本和额外 shape。

**正常模式（总计 8 小时）**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 0.5 小时 | 二选一、冻结环境/shape/dtype/版本阶梯 | 实验契约 |
| 1.5 小时 | baseline、正确性矩阵、warmup+重复计时 | 可复现基线 |
| 2 小时 | 逐版重跑与一次失败优化 | 版本性能表 |
| 1.5 小时 | Roofline、ncu、PTX/SASS 取证 | 瓶颈证据链 |
| 1.5 小时 | 库/理论上限差距、下一步、报告 | 完整 Markdown 报告 |
| 1 小时 | 闭卷追问与 5/10 分钟双版本口述 | 录音与问题清单 |

**必要时补考模式（总计 8 小时，含前日最多 1.5 小时）**：1.5 小时补 Day 10；0.5 小时冻结项目；1.25 小时 baseline/正确性矩阵；1.5 小时两级版本与失败优化复测；1.25 小时 Roofline+ncu+PTX/SASS；1.25 小时报告与差距；0.75 小时 5/10 分钟口述。压缩额外版本、shape 扫描和排版，绝不复用旧数字代替重跑。

**必读仓库资料**：GEMM 路线读 [benchmark](../week05_gemm_advanced/benchmark.md)、[ncu notes](../week05_gemm_advanced/ncu_notes.md)、[Roofline](../week05_gemm_advanced/roofline.md)及 [tiled GEMM](../week04_gemm/gemm_tiled/gemm_tiled.cu)；Attention 路线读 [naive](../week04_attention/naive_attention.cu)、[tiled](../week04_attention/tiled_attention.cu)、[pipelined](../week04_attention/tiled_attention_pipelined.cu)、[ncu pipeline notes](../week04_attention/ncu_pipeline_notes.md)。这些只提供真实仓库起点与历史证据；当天必须在当前 A100、当前构建重新确认，不能复制历史数字。

**分步练习**：

1. 开场写实验契约并锁定 GEMM 或 Attention。GEMM 建议阶梯为 naive→shared tiled→register/vectorized；Attention 建议 naive→tiled/online softmax→pipelined。每次只解释相邻版本的主要变化。
2. 建正确性矩阵：至少三个规则/非规则 shape、固定随机种子、dtype、逐 dtype/shape 的 `atol/rtol`、CPU/cuBLAS 或可信 reference；同时报告 finite、max_abs、max_rel 与 PASS，任何失败版不进入性能排名。
3. 计时契约固定并记录 warmup 次数和正式样本数（例如 warmup 20、samples 100）；CUDA Event 结果统一标明 ms 或 us，至少报告 median、p5、p95（若改用 mean/std，必须预先声明且保持全表一致）。GEMM 按 `FLOP=2*M*N*K`、`GFLOP/s=FLOP/(time_seconds*1e9)`；Attention 若选，必须依据当前实现逐项列 QK、softmax、PV 等实际 FLOP 模型，不能套通用常数。禁止把 profiler replay 时间当正常墙钟。
4. 分开计算源码模型的 algorithmic bytes 与 ncu measured DRAM bytes，并标注 arithmetic intensity 单位 `FLOP/byte`。Roofline 写成 `min(datasheet compute peak, datasheet BW * AI)`，再与同 shape/dtype/计时条件的 cuBLAS/参考库实测可达值分栏；任何“达到 X%”都写明分母是 datasheet peak、Roofline 上限还是库实测。用 ptxas、ncu、PTX/SASS 连接源码修改、指标和墙钟。
5. 至少保留一次失败优化，例如去 padding、过大 tile/stage 或不利向量化；写原假设、唯一改动、正确性、资源、指标、墙钟以及为什么失败。
6. 与当前环境可用的 cuBLAS/CUTLASS 对照；若没有公平可运行对照，写清库缺失和理论峰值/带宽上限，量化百分比差距。结尾列一个最可能的下一步、预期指标变化和证伪条件。
7. 用下方模板在当天 `/tmp` 报告中填写；本计划只提供模板，不创建实际 README。

**必须产出**：实验契约、baseline 与至少两级版本、正确性矩阵、可靠计时、Roofline、ncu、PTX/SASS、一次失败优化、库或理论上限差距、下一步实验、完整 Markdown 报告、5/10 分钟口述录音与追问清单。

**可复制的 Markdown 报告/README 模板**：

```markdown
# A100 旗舰项目：<GEMM 或 Attention，只填一个>

## 环境与复现
- GPU/显存/时钟策略：
- Driver/CUDA/nvcc/ncu：
- Git SHA、编译参数：
- dtype、shape、随机种子、每组 atol/rtol：
- 计时 warmup、samples、统计口径与时间单位：
- FLOP 模型（GEMM=`2MNK`；Attention 按本实现逐项列式）：
- 构建/正确性/计时/ncu/PTX/SASS 命令：

## 正确性矩阵
| 版本 | dtype/shape | reference | atol | rtol | max_abs | max_rel | finite | sanitizer | PASS |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |

## 可靠计时与版本阶梯
| 版本 | 唯一主要变化 | warmup/samples | 单位 | median | p5 | p95 | GFLOP/s 或有效吞吐 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |

## Roofline 与瓶颈证据
- algorithmic bytes（源码模型）与 ncu measured DRAM bytes：
- FLOP 口径、AI（FLOP/byte）：
- datasheet peak、Roofline `min(compute peak, BW*AI)`、同条件库实测可达：
- 当前百分比及明确分母：
- ncu 指标、ptxas 资源、关键 PTX/SASS：
- 源码→指标→墙钟因果解释及替代解释：

## 失败实验
- 原假设与唯一改动：
- 正确性、资源、指标、墙钟结果：
- 失败原因与学到什么：

## 与 cuBLAS/CUTLASS 或理论上限的差距
- 对照是否公平、命令与版本：
- 差距百分比及不能下的结论：

## 下一步
- 一个修改、预期指标变化、证伪条件：

## 面试口述
### 5 分钟版
问题→baseline→关键优化→结果→边界。
### 10 分钟版
再加入正确性矩阵、Roofline/ncu/PTX/SASS、失败实验、库差距和下一步。
```

**闭卷验收题**：① 为什么选择该项目而不是另一项？② baseline 的瓶颈证据是什么？③ 每一级优化改变了什么数据流或资源？④ 正确性矩阵为何足以覆盖主要边界？⑤ Roofline、ncu 与墙钟怎样互证？⑥ 失败优化推翻了什么假设？⑦ 与库/上限差多少，下一步如何证伪？能在不看稿时给出 5 分钟版，并在追问下扩展为 10 分钟版才通过。

**面试口述主题**：5 分钟版强调问题、两级优化、量化结果和边界；10 分钟版加入测试设计、瓶颈证据、失败实验、库差距与下一步。所有数字都注明本日重跑环境。

**没通过时的降级/补救**：不切换到另一项目救场；收缩为一个 baseline、一个有效版本、一个失败版本和三个 shape，保留四类输出。未完成项进入 Day 12 薄弱清单；历史 benchmark 只能作背景，不能代替本人本日证据。

## Day 12：知识体系闭卷压缩与薄弱项回补

**当日目标**：把执行模型、内存、同步、性能、异步、Tensor Core、编译部署七组知识压缩为可追问的闭卷回答；每题先在 60～120 秒内回答，再查资料修正。所有回答统一采用“结论→条件/边界→仓库实验/证据”，禁止背诵“总是、一定、越高越好”等绝对化结论；至少用 90 分钟回补 Day 1～11 的真实薄弱项。

**优先级**：必须完成七组各 2 题、Day 1～11 薄弱项回补、错题卡、可审计评分和口述；尽量完成每组第 3 题与交叉追问；时间不足可减少扩展题，不能删除七组覆盖、先闭卷后修正、四类输出或 90 分钟回补。

**正常模式（总计 8 小时）**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 0.5 小时 | 汇总 Day 1～11 失败项、录音和证据缺口，按风险排序 | 薄弱项清单与抽题种子 |
| 1.5 小时 | 回补最高风险薄弱项：重答、最小实验或重写核心段 | 90 分钟回补记录 |
| 3.5 小时 | 七组闭卷题：每组 2 题，60～120 秒首答后查资料修正 | 14 张评分答题卡 |
| 1 小时 | 每组抽 1 个交叉追问，整理错题卡 | 7 个追问与错因分类 |
| 1 小时 | 随机复测错题并做 7 分钟综合口述 | 二次得分与录音 |
| 0.5 小时 | 审计评分、决定 Day 13 抽题权重 | 总分表与薄弱项权重 |

**必要时补考模式（总计 8 小时，含 Day 1～11 回补 2.5 小时）**：2.5 小时补最高风险失败项（其中至少 90 分钟用于重做而非阅读）；3.5 小时完成七组各 2 题闭卷首答、查证和修正；0.75 小时错题卡与交叉追问；0.75 小时随机复测和口述；0.5 小时评分审计及 Day 13 抽题权重。压缩每组扩展题和排版，不牺牲概念、代码/实验、证据、面试四类输出，补考本身计入当天 8 小时。

**必读仓库资料**：[CUDA 复习资料知识体系](CUDA复习资料_知识体系.md)、[CUDA 面试八股全集](CUDA面试八股全集.md)、[面试八股追问答案](CUDA面试八股_追问答案.md)、[面试八股追问答案续](CUDA面试八股_追问答案_续.md)、[概念、性能分析与手写 kernel 面试题](../cuda_deep_course/course/volume10_engineering_interview/08_面试题_概念_性能分析_手写kernel.md)。每题必须先闭卷作答并留时间戳，之后才能查阅；资料答案也要结合硬件、shape、CUDA 版本和仓库证据限定边界。

**分步执行**：

1. 从 Day 1～11 的答题纸、sanitizer、ncu、代码和录音建立薄弱项表，字段为“来源日/症状/错误结论/证据缺口/回补动作/复测结果”。优先处理会导致错误结果、死锁、错误计时或证据冒充的项。
2. 七组精选题不做题海：执行模型抽 warp/CTA 调度与 occupancy 边界；内存抽 coalescing/cache/shared bank；同步抽 barrier/原子/内存可见性；性能抽 Roofline/延迟隐藏/ncu；异步抽 stream/event/Graph/`cp.async`；Tensor Core 抽 fragment/数据布局/精度与流水；编译部署抽 nvcc→PTX→SASS、架构兼容与运行时错误。每组固定 2 题，余时才抽第 3 题。
3. 每题流程固定：计时 60～120 秒闭卷口答并写关键词；再查上述资料和追问答案；用不同颜色或 diff 写“原答、资料依据、修正版”。修正版只有同时包含“结论→条件/边界→仓库实验/证据”才可得结构分。
4. 每题 10 分：结论 2、条件/边界 3、仓库实验/证据 3、追问应对 1、限时与表达 1。评分者逐项勾选并写证据页码/录音时间戳；自己评分时必须保留原答和修订 diff，不能只填总分。硬错误（越界/死锁、把 profiler replay 当墙钟、把旧记录冒充本日重跑、用单指标作唯一因果结论）该题最高 4 分。
5. 首轮通过阈值为总分至少 112/140、七组均至少 14/20，且无硬错误；未达标的题进入错题卡，字段为“错误原句/反例或边界/可信证据/正确三段式/下次复测时间”。隔至少 60 分钟随机复测 5 题，须达到 40/50 且无同类硬错误。
6. Day 13 的两道 kernel 从代码/边界最弱组加权抽取，三道 ncu 症状从性能证据最弱组加权抽取；把抽题规则、随机种子或请朋友抽取的结果提前封存，避免挑熟题。

**必须产出**：Day 1～11 薄弱项表、90 分钟以上回补证据、七组 14 张闭卷原答/修正版、逐项评分表、错题卡、随机复测卷、资料定位、7 分钟口述录音与时间戳、Day 13 抽题权重。四类输出分别落到概念答题、最小代码/实验、证据表、口述录音。

**闭卷验收题**：从七组各随机 1 题，共 7 题；每题 90 秒按“三段式”回答，并接受一个反例追问。总分至少 56/70、每组不低于 7/10、无硬错误才通过；修正后的答案不能覆盖原答，朋友或复听者能按 rubric 复算得分。

**面试口述主题**：7 分钟串讲“一个 kernel 从执行映射、内存与同步，到 Roofline/ncu、异步流水、Tensor Core，再到编译部署的完整证据链”，每个结论至少说一个成立条件，并区分本人当天重跑证据与仓库旧记录。

**没通过时的降级/补救**：不继续刷几百道题；只保留未达 7/10、出现硬错误或无法给证据的卡片。Day 13 开场最多 90 分钟补最危险的 3 张卡并计入 8 小时，随后仍完成两道限时 kernel、三道诊断、评分和口述；其余错题进入 Day 14 后的 3～7 天最小补救清单。

## Day 13：限时编码与性能诊断

**当日目标**：在不看答案的条件下完成两道 kernel 题，并对三道 ncu 症状题给出可证伪诊断。编码必须先澄清输入、保证正确与边界安全，再解释同步、复杂度/bytes、测试和优化；诊断固定采用“现象→多个候选原因→需要的补充证据→一个可证伪的单一实验→复测指标”，禁止从单一指标直接下结论。

**优先级**：必须完成两道限时 kernel、三道 ncu 症状题、正确性/测试证据、评分和口述；尽量实际编译运行并采集一个诊断实验；时间不足可缩减参数扫描和第二轮优化，不能牺牲两题正确基础版、三类诊断链或四类输出。

**正常模式（总计 8 小时）**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 0.5 小时 | 按 Day 12 权重封存抽题、写统一接口和测试契约 | 两道 kernel、三道症状题 |
| 2 小时 | kernel 题 1：15 分钟澄清/设计，60 分钟编码，30 分钟测试，15 分钟复盘 | 代码、测试、复杂度/bytes |
| 2 小时 | kernel 题 2：同一时间盒 | 代码、测试、优化假设 |
| 2 小时 | 三道 ncu 症状题，每题 40 分钟 | 三张可证伪诊断卡 |
| 1 小时 | 实跑一个单一实验并复测，或对给定证据做纸面复测设计 | 前后指标表 |
| 0.5 小时 | 审计评分、5 分钟口述与失败归档 | 总分、录音、补救项 |

**必要时补考模式（总计 8 小时，含 Day 12 最多 1.5 小时）**：1.5 小时补 Day 12 三张高风险卡；1.75 小时 kernel 题 1；1.75 小时 kernel 题 2；1.5 小时三道 ncu 诊断（每题 30 分钟）；0.75 小时一个单一实验/复测设计；0.5 小时统一测试与评分；0.25 小时口述。压缩代码美化、额外优化和参数扫描，不删除两道正确性、三道诊断、证据或面试输出。

**必读仓库资料**：[概念、性能分析与手写 kernel 面试题](../cuda_deep_course/course/volume10_engineering_interview/08_面试题_概念_性能分析_手写kernel.md)、[CUDA 复习资料知识体系](CUDA复习资料_知识体系.md)、[CUDA 面试八股全集](CUDA面试八股全集.md)、[面试八股追问答案](CUDA面试八股_追问答案.md)、[面试八股追问答案续](CUDA面试八股_追问答案_续.md)。只能在题目提交、计时停止后对照；若题目来自仓库现有实现，先把答案路径交给朋友隐藏，或只复制接口/测试框架到 `/tmp`。

**分步执行**：

1. 两道题从 reduction、transpose、softmax、GEMV、RMSNorm、tiled GEMM 中按 Day 12 薄弱项加权无放回抽取；至少一道含非规则边界或归约，不能临时换成最熟题。记录抽题人/随机种子和开始、停止时间。
2. 开写前口头澄清 dtype、shape、layout、别名、误差、设备约束和性能目标；写线程映射、边界与同步草图。先交正确基础版，再谈 vectorization、shared tiling、warp primitive 或融合。
3. 每题必须说明 work/span 或主要运算复杂度，以及 algorithmic bytes 与预期 arithmetic intensity；测试至少含空/最小值、31/32/33 邻域、奇数大 shape、极值或近零数值场景，使用可信 reference、finite、max_abs/max_rel 和适用的 sanitizer。未通过正确性不得计优化分。
4. ncu 症状从以下五组按弱项抽 3 组且不得重复：高 long scoreboard；高 occupancy 但 eligible warps 低；shared bank conflicts；register spill/local traffic；launch-bound。每题先复述 shape、kernel 阶段和采集条件，缺上下文就明确索要证据。
5. 每张诊断卡至少列 2 个竞争候选原因、区分它们所需的源码/资源/SASS/相关 ncu 指标，然后只选一个改动做可证伪实验。例如针对 long scoreboard，只改变访问合并/预取中的一个因素，而不是同时改 tile 和 block；预先写若假设成立哪些指标和墙钟应怎样变，若不变就否定该假设。
6. 诊断的复测指标同时包含正常 CUDA Event 墙钟和与假设相关的指标；保持 shape、输入、编译参数、时钟策略和采集 section 一致。occupancy、eligible、stall、bank conflict 或 spill 任一单指标都只能缩小假设空间，不能独自证明根因。
7. 评分共 100 分：kernel 各 30 分（澄清 3、正确性/边界 12、同步 4、复杂度/bytes 3、测试 5、优化与表达 3）；诊断三题各 12 分（现象 1、候选原因 3、补充证据 2、单一可证伪实验 4、复测指标 2）；审计与限时 4 分。通过阈值 80/100、每道 kernel 至少 22/30、每道诊断至少 8/12，且无硬失败。

**必须产出**：封存抽题记录、两道独立 kernel 源码/伪代码、接口澄清表、线程映射、边界与同步说明、复杂度/bytes、reference/测试/sanitizer、三张 ncu 诊断卡、一个实验前后或完整复测设计、逐项评分表、5 分钟录音。原始代码、命令和输出必须可供朋友复核。

**闭卷验收题**：随机选一道 kernel，10 分钟讲清接口、线程映射、边界、同步、复杂度/bytes 和测试；再随机选一道症状题，5 分钟完整走完五段诊断链。出现越界/死锁、无法给 reference、把单指标直接判根因或无法提出单一实验即失败。

**面试口述主题**：5 分钟回答“面对一个慢 kernel，我如何先守住正确性，再把 profiler 症状变成可证伪实验”，引用当天一项本人实跑证据；若只有纸面数据，必须明确说是诊断设计而非实测结论。

**没通过时的降级/补救**：硬失败包括任一 kernel 错误/死锁/无边界测试、任一诊断缺竞争原因或单一实验、伪造/混淆计时证据。立即保留最小失败输入和原始输出，不无限重写；Day 14 前最多用 60 分钟复述失败根因并重做关键段，这 60 分钟计入 Day 14 的 8 小时。其他低分项进入模拟后的 3～7 天补救。

## Day 14：两轮模拟面试与投递判断

**当日目标**：完成两轮各 45 分钟的可审计模拟面试，把知识、手写 kernel、性能诊断、架构/系统取舍和旗舰项目转成真实面试表达；依据评分和硬失败项作 ready 判断。达到阈值当天开始投递，未达到则只安排 3～7 天最小补救，不以“再准备一点”为由无限延迟。

**优先级**：必须完成两轮 45 分钟、项目四档口述、逐项评分、硬失败审计和投递/补救决定；尽量由朋友提问并追问；找不到朋友时可提前封存题目、自录音录像，间隔至少 90 分钟后再做第二轮并按同一 rubric 复听。

**正常模式（总计 8 小时）**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 0.5 小时 | 冻结题单、评分者、录音和环境，做设备检查 | 模拟契约 |
| 0.75 小时 | 第一轮 45 分钟：概念+手写+项目 | 原始录音与答题纸 |
| 0.75 小时 | 独立评分，不改答案 | 第一轮评分表 |
| 1.5 小时 | 项目口述 30 秒/2 分钟/5 分钟/10 分钟四版及追问 | 四档录音与证据索引 |
| 0.75 小时 | 第二轮 45 分钟：性能追问+架构取舍+系统题 | 原始录音与设计草图 |
| 0.75 小时 | 独立评分与硬失败审计 | 第二轮评分表 |
| 1.5 小时 | 针对最低项做一次最小重答/重画/重写 | 补测证据 |
| 1.5 小时 | ready 判断；通过则准备并开始投递，未过则写 3～7 天补救 | 投递记录或限时补救表 |

**必要时补考模式（总计 8 小时，含 Day 13 最多 1 小时）**：1 小时补 Day 13 硬失败关键段；0.5 小时冻结题单；0.75 小时第一轮；0.5 小时第一轮评分；1.25 小时项目四档口述；0.75 小时第二轮；0.5 小时第二轮评分；1.25 小时最低项补测；1.5 小时 ready 决策并启动投递或制定 3～7 天补救。压缩录音排版和扩展追问，不删除两轮 45 分钟、项目表达、证据、评分或决定。

**必读仓库资料**：[概念、性能分析与手写 kernel 面试题](../cuda_deep_course/course/volume10_engineering_interview/08_面试题_概念_性能分析_手写kernel.md)、[系统设计题](../cuda_deep_course/course/volume10_engineering_interview/09_系统设计题.md)、[CUDA 面试八股全集](CUDA面试八股全集.md)、[面试八股追问答案](CUDA面试八股_追问答案.md)、[面试八股追问答案续](CUDA面试八股_追问答案_续.md)、[CUDA 复习资料知识体系](CUDA复习资料_知识体系.md)。项目环节另读 Day 11 当天生成的旗舰项目报告和原始证据；若该文件仅在 `/tmp`，先复制路径索引，不把它误写成仓库已跟踪资料。

**分步执行**：

1. 第一轮 45 分钟逐分钟时间轴：0～3 分钟自我介绍/项目定位；3～13 分钟执行模型、内存、同步概念；13～28 分钟手写一个抽取 kernel 并讲边界/同步/复杂度；28～40 分钟项目 2 分钟摘要加正确性、计时、瓶颈追问；40～45 分钟候选人反问与总结。面试官按时打断，模拟真实压力。
2. 第二轮 45 分钟逐分钟时间轴：0～5 分钟复述一个 ncu 现象；5～17 分钟从竞争原因到单一可证伪实验；17～28 分钟 Ampere/Hopper、Tensor Core、TMA/WGMMA/Cluster 等架构取舍，明确未实测边界；28～41 分钟系统设计题，覆盖目标/SLO、数据流、并发与内存预算、调度/回压、故障与可观测性；41～45 分钟权衡总结与反问。
3. 朋友提问时只给题干和允许提供的证据，不提示答案；朋友在原始评分表签名或写 initials。自测时由 Day 12 封存题单或随机数抽题，首轮后不立即查答案，至少间隔 90 分钟再做第二轮；录音、屏幕/纸面答案和时间戳都保留。
4. 项目口述模板必须形成四版：30 秒“我解决了什么、对象/平台、一个可信结果和边界”；2 分钟“问题→baseline→关键优化→结果→证据”；5 分钟加入正确性、计时、瓶颈和失败优化；10 分钟加入可移植性、cuBLAS/CUTLASS 等库对比、Hopper 迁移及下一步。追问必须覆盖正确性、计时、瓶颈、失败优化、可移植性、库对比、Hopper 迁移。
5. 所有项目数字在证据索引中标记 `TODAY-RERUN` 或 `REPO-HISTORICAL`；前者附当天命令、环境和原始输出，后者附仓库路径/SHA 且只作背景。无法当天重跑就说“仓库旧记录，未由我今天复现”，不得混说成个人当日结果。
6. 每轮 100 分：技术正确性 25、条件/边界 15、推理与取舍 15、代码/系统结构 15、实验与证据 15、表达/时间管理 10、追问与诚实边界 5。两位评分者可独立打分后记录分歧；只有自己时，通过录音时间戳逐项引用原话，不能凭印象填总分。
7. ready 阈值为两轮均至少 80/100、合计至少 165/200，项目 5 分钟版在对应条目至少 16/20，且无硬失败。硬失败为：kernel 明显越界/死锁且无法自纠；把错误结果用于性能结论；用单一 ncu 指标断言根因；伪造或混淆本人当天重跑与仓库旧记录；系统设计无正确性/容量/故障边界；无法说明项目中本人贡献。
8. 达到阈值就在当天建立投递清单并完成首批真实投递，不承诺“完成计划就必过面试”。未达到时按硬失败映射 3～7 天：正确性/同步 3 天最小 kernel 重写；性能证据 3 天诊断卡+两次单变量复测；项目证据 3～5 天重跑和口述；架构/系统 5 天两次白板；两轮均低于 70 则 7 天综合补救。到期再次做两轮精简复测后开始投递，不无限延期。

**必须产出**：模拟契约、两轮各 45 分钟完整录音/录像、手写代码或系统图、两份逐项评分表与时间戳、硬失败清单、项目 30 秒/2 分钟/5 分钟/10 分钟四版、七类项目追问答案、当天/历史证据索引、最低项补测、首批投递记录或 3～7 天最小补救表。

**闭卷验收题**：① 90 秒解释一个 CUDA 概念及边界；② 15 分钟手写并测试一个 kernel 核心；③ 从一个 ncu 症状提出竞争原因和单一实验；④ 5 分钟讲旗舰项目并应对失败优化、库对比、Hopper 迁移；⑤ 8 分钟画一个带容量与故障边界的 GPU 推理系统。按正式两轮评分计入，不另开无压力自评替代。

**面试口述主题**：项目表达严格使用 30 秒定位、2 分钟摘要、5 分钟优化链、10 分钟深挖四档；根据面试官打断随时收束。回答的目标是可验证、边界诚实和取舍清楚，而不是背出唯一标准答案。

**没通过时的降级/补救**：当天仍写明确的 not-ready 原因和截止日期，不追加无边界题海。只修触发硬失败或低于 70% 的维度，选择上述 3～7 天最小包；保留原模拟证据以便复测对比。补救结束后无论结果如何都用同一 rubric 复测并开始有针对性的投递，同时在真实面试反馈中继续迭代。
