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

**闭卷验收题**：① 从空白写出不会漏尾元素的 block reduction 核心；② 为什么 block 间不能靠 `__syncthreads()` 收口？③ transpose 的 load 和 store 各如何合并？④ shared tile 为什么需要同步，为什么同步不能放进分歧分支？⑤ `TILE_DIM+1` 怎样改变 bank 映射？⑥ 有效带宽字节数如何计算？两类 kernel 均通过非规则 shape 与 sanitizer 才通过。

**面试口述主题**：4 分钟讲“我怎样先得到正确基础版，再用 warp/block 归约与 shared padding 接近带宽上限”，必须包含一个边界 bug 或失败优化及其证据。

**没通过时的降级/补救**：把失败收缩为一个最小 shape，保留 sanitizer 报告和错误索引；Day 9 开场最多 90 分钟重写失败核心并计入 8 小时。不能复制仓库答案宣称通过；再次失败进入 Day 12 薄弱清单。

## Day 9：Softmax、GEMV 与 RMSNorm 基础正确版

**当日目标**：从空白独立写出稳定 softmax、thread-per-row 与 warp-per-row GEMV、RMSNorm 三组核心（合计至少四个 kernel）；三个主题均先达到基础正确版，优化按级别推进，不用追求一天内把每版做到峰值。

**优先级**：必须完成稳定性、非 2 的幂长度、多 shape reference 与 sanitizer；尽量完成 GEMV 合并访问和 RMSNorm gamma/残差融合；时间不足可跳过向量化、persistent kernel 与大规模调参。

**正常模式（总计 8 小时）**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 0.5 小时 | 接口、CPU reference、误差与 shape 契约 | 测试矩阵 |
| 2 小时 | stable softmax：max→sum→normalize | 正确基础版、极值测试 |
| 2 小时 | GEMV：thread-per-row→warp-per-row | 两版正确性与带宽 |
| 1.5 小时 | RMSNorm：平方和归约、epsilon、gamma；残差融合为进阶 | 基础版及融合版/设计 |
| 1 小时 | sanitizer、warmup+重复计时、多 shape 汇总 | 证据表 |
| 1 小时 | 闭卷核心段与 4 分钟口述 | 答题纸、录音 |

**必要时补考模式（总计 8 小时，含前日最多 1.5 小时）**：1.5 小时补 Day 8；0.5 小时测试契约；1.5 小时 softmax；1.75 小时两版 GEMV；1.25 小时 RMSNorm 基础版；1 小时 sanitizer/计时；0.5 小时闭卷与口述。压缩掉融合实测、额外 shape 和优化排版；若无补考，释放的 1.5 小时用于 RMSNorm 残差融合与 GEMV 优化。

**必读仓库资料**：[softmax](../week04_gemm/softmax/softmax.cu)、[GEMV](../week05_inference/gemv.cu)、[fused RMSNorm](../week05_inference/fused_rmsnorm.cu)、[Week 5 记录](../notes/week05.md)。只在独立版本完成或卡点复盘时阅读核心实现。

**分步练习**：

1. 在 `/tmp/cuda_interview_sprint/day09/` 建练习文件，保留仓库脚手架时清空核心 kernel。CPU reference 使用更高精度累积；每项明确绝对/相对误差，不能只看首元素。
2. softmax 分别完成每行 max、`exp(x-max)` 求和和 normalize；测试长度 `1,31,32,33,1000,1025`，输入含 `-1000,0,1000`、全相等和随机极值，检查有限性及每行和接近 1。
3. GEMV 先写一线程一行，再写一 warp 一行；测试非规整 `M,K`，从地址推导 warp-per-row 对权重的合并访问与 shuffle 归约，分别报告正确性和有效带宽。
4. RMSNorm 写 `sum(x²)` 归约、`rsqrt(mean+epsilon)` 和 gamma；覆盖很小方差、奇数 hidden size、不同 epsilon。基础版通过后才做 `residual+x` 与 gamma 融合，并写清输出语义和边界。
5. 三组代码均运行 memcheck，含共享同步的版本运行 synccheck；均 warmup 后重复计时，至少覆盖小、中、大 shape。性能较慢不影响基础版通过，但错误结果不能进入性能表。

**必须产出**：三个 CPU reference、稳定 softmax、两版 GEMV、RMSNorm 基础版（若完成则附融合版）、多 shape/极值对拍、sanitizer、可靠计时与带宽、地址/归约图、闭卷答题和录音。至此 Day 8～9 至少独立写出 reduction、transpose、softmax、两版 GEMV、RMSNorm 六类核心 kernel。

**闭卷验收题**：① 不减 max 的 softmax 为何会溢出/下溢？② 非 2 的幂归约怎样保证不漏元素？③ 两种 GEMV 映射的相邻 lane 分别访问哪里？④ GEMV 为何常受带宽限制？⑤ RMSNorm 与 LayerNorm 差什么？⑥ epsilon 放在哪里，gamma/残差融合改变哪些读写？能现场重写每组核心索引并通过基础测试才通过。

**面试口述主题**：4 分钟串讲 decode 中 softmax、GEMV、RMSNorm 的数值稳定性、线程映射和 memory-bound 特征，明确基础正确版与已验证优化的边界。

**没通过时的降级/补救**：任一基础版失败都记录最小输入、CPU/GPU 差值和 sanitizer；Day 10 开场最多 90 分钟只补失败基础版。融合未完成可列为 Day 10 工程收口，但不能用融合 TODO 掩盖 RMSNorm 核心未通过。

## Day 10：闭卷 Tiled GEMM 与推理 TODO 工程收口

**当日目标**：闭卷完成一个边界安全的 tiled GEMM，并按 CUDA Graph→Fused RMSNorm→Dequant GEMV 的顺序收口推理练习；Day 9 已通过 RMSNorm 核心时，本日只做融合接口、工程验证与证据，不重复抄写同一基础 kernel。

**优先级**：必须完成 tiled GEMM 的非规整 shape 对拍、CUDA Graph 生命周期和 dequant kernel 正确性；尽量完成 RMSNorm 残差融合及三项性能对照；时间不足可跳过 Graph 节点更新、更多量化格式和深度调参。

**正常模式（总计 8 小时）**：

| 时间 | 任务 | 到点产物 |
| ---: | --- | --- |
| 2.5 小时 | 90 分钟闭卷 GEMM，60 分钟边界/对拍修复 | tiled GEMM、测试 |
| 1.5 小时 | CUDA Graph capture→instantiate→replay→destroy | 生命周期正确实现 |
| 1 小时 | Fused RMSNorm 工程收口 | 融合正确性/接口证据 |
| 1.5 小时 | Dequant GEMV 与量化 reference | kernel 与量化误差表 |
| 0.75 小时 | sanitizer、warmup+重复计时 | 证据摘要 |
| 0.75 小时 | 闭卷与 5 分钟口述 | 答题纸、录音 |

**必要时补考模式（总计 8 小时，含前日最多 1.5 小时）**：1.5 小时补 Day 9；2 小时 tiled GEMM；1.25 小时 CUDA Graph；0.75 小时 Fused RMSNorm 工程检查；1.25 小时 Dequant GEMV；0.75 小时 sanitizer/计时；0.5 小时闭卷与口述。压缩来源为 GEMM 性能调参、Graph 扩展能力和报告美化，不删除四类输出或三项工程正确性。

**必读仓库资料**：[tiled GEMM](../week04_gemm/gemm_tiled/gemm_tiled.cu)、[CUDA Graph](../week05_inference/decode_graph.cu)、[Fused RMSNorm](../week05_inference/fused_rmsnorm.cu)、[Dequant GEMV](../week05_inference/dequant_gemv.cu)、[Week 5 记录](../notes/week05.md)。TODO 文件可作为测试框架；先在 `/tmp` 副本实现，不覆盖用户文件。

**分步练习**：

1. 限时 90 分钟从空白写 GEMM 核心：`row/col` 索引、A/B 协作加载、越界补零、同步、tile 累加、边界写回；测试 `1x1x1, 31x33x35, 127x129x65, 256x256x256`，用 CPU 与可用时 cuBLAS 双对照。
2. CUDA Graph 按顺序实现 stream capture、结束 capture 得 graph、instantiate 得 executable graph、warmup/replay/同步；验证多次 replay 输出稳定，并在正确时机 destroy graph executable、graph、event/stream，错误路径也不能泄漏或使用已销毁对象。
3. 若 Day 9 RMSNorm 已通过，直接检查 residual+RMSNorm+gamma 融合的输入输出别名、epsilon、hidden 尾部和 launch 错误；若未通过，先完成补考基础版，再做最小融合收口。
4. Dequant GEMV 分两层 reference：先用同一量化权重的 CPU dequant+GEMV 检查 kernel 实现误差，再与原始浮点权重结果比较量化损失。两者必须分栏，不能把量化误差归罪于 kernel。
5. 全部核心跑非规则 shape、memcheck，适用时 synccheck；正常计时使用 warmup+重复运行，并把 Graph capture/instantiate 一次性成本与 replay 延迟分开。

**必须产出**：本人闭卷 tiled GEMM、CPU/cuBLAS 对拍、Graph 生命周期图与多次 replay、Fused RMSNorm 工程验收、Dequant GEMV 双层误差表、sanitizer、正常计时、命令/环境、闭卷答题与录音。

**闭卷验收题**：① 两个 shared tile 的索引如何推导？② 为什么越界 load 要写零而非跳过后继续使用旧 shared 值？③ 两次同步各保护什么？④ Graph 与 executable graph 生命周期如何区分？⑤ capture/instantiate 成本为何不应混入稳态 replay？⑥ 怎样区分 dequant kernel bug 和量化损失？现场通过非规整 GEMM、三次 Graph replay 与双层 dequant 对拍才通过。

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
2. 建正确性矩阵：至少三个规则/非规则 shape、固定随机种子、dtype、容差、CPU/cuBLAS 或可信 reference；任何失败版不进入性能排名。
3. 每版做 warmup、至少 100 次 CUDA Event 计时，报告 median 或均值及波动；GEMM 报 FLOP/s，Attention 同时报延迟和有效工作量，禁止混用 profiler replay 时间。
4. 计算 arithmetic intensity 和 Roofline 上限；用 `-Xptxas=-v`、ncu 的本机可用 sections、PTX/SASS 解释资源、访存、stall 和关键计算指令。结论必须连接源码修改、指标和墙钟。
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
- dtype、shape、随机种子、容差：
- 构建/正确性/计时/ncu/PTX/SASS 命令：

## 正确性矩阵
| 版本 | shape | reference | max abs/rel error | sanitizer | 结论 |
| --- | --- | --- | ---: | --- | --- |

## 可靠计时与版本阶梯
| 版本 | 唯一主要变化 | warmup/重复 | 延迟 | GFLOP/s 或有效吞吐 | 波动 |
| --- | --- | ---: | ---: | ---: | ---: |

## Roofline 与瓶颈证据
- 实际字节/FLOP 口径、arithmetic intensity：
- 理论/可达上限与当前百分比：
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
