# Week 2：PTX / SASS 与性能诊断执行清单

> 当前状态：**执行中**
> 周期：7 天，每天约 4～8 小时
> 硬件：NVIDIA A100 80GB PCIe（`sm_80`）
> 主项目标本：[GPU Kernel Engineering / FP32 GEMM](https://github.com/starboy520/gpu-kernel-engineering/tree/main/projects/gemm)
> 本地仓库：`/home/qichengjie/workspace/gpu-kernel-engineering`
> 本周记录：[Week 2 PTX/SASS Worklog](../../notes/week02_ptx_sass.md)

## 0. 本周定位

Week 1 已完成并发布：Naive、Shared、Register、Vectorized、Async 16B、cuBLAS FP32、正式 benchmark、sanitizer、ncu 与 SASS 证据均已进入公开仓库。

本周**不再优化 GEMM 性能**，也不继续做 swizzle。Day 1–4 把已经熟悉的公开 GEMM 当作固定主标本；Day 5–6 只增加一个写在 `/tmp` 的依赖链 microbenchmark，用来保证单变量实验严格可控；Day 7 只观察学习仓中已有的 WMMA 实现，用于补齐 Tensor Core 三层证据。这两个小实验都不形成新的项目或优化主线。训练下面这条证据链：

```text
CUDA C++
→ PTX 虚拟 ISA
→ ptxas 资源分配
→ sm_80 cubin / SASS
→ ncu scheduler 与 stall
→ 可证伪假设
→ 单变量实验
→ correctness + wall-clock + 指令 + profiler 复测
```

### 本周结束后应该会什么

- 独立生成 PTX、cubin、ptxas log 和 SASS；
- 沿 Naive GEMM 的 `load → compute → store` 追踪一条数据流；
- 比较五个 CUDA Core GEMM 版本的机器数据流；
- 解释 register、spill、local memory、occupancy、ILP、TLP 的取舍；
- 区分 active、eligible、issued warp 和 scoreboard；
- 完成一次严格的单变量实验；
- 讲清 WMMA API、PTX `mma.sync`、SASS HMMA 三层关系。

### 明确不做

- 不逐行翻译整份 SASS；
- 不背全部 PTX/SASS 助记符；
- 不继续调 GEMM tile 参数或 swizzle；
- 不手写完整生产级 inline PTX MMA；
- 不学习 Hopper TMA、WGMMA、Cluster、DSM；
- 不把 profiler replay 时间当 wall-clock；
- 不从 stall 名称直接宣布根因。

---

## 1. 固定目录与实验纪律

### 1.1 每天开始时

```bash
export GEMM_REPO=/home/qichengjie/workspace/gpu-kernel-engineering
export CUDA_STUDY=/home/qichengjie/workspace/cuda_study
export WEEK2_TMP=/tmp/cuda_focus/week02
mkdir -p "$WEEK2_TMP"
```

确认基线：

```bash
cd "$GEMM_REPO"
git status --short
git rev-parse --short HEAD
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvcc --version
ncu --version
cuobjdump --version
```

预期：

```text
GEMM_REPO 位于 main
工作树干净
HEAD 至少包含 f92e95e
GPU 为 NVIDIA A100 80GB PCIe
```

### 1.2 产物只放 `/tmp`

```text
/tmp/cuda_focus/week02/
├── day01/
├── day02/
├── day03/
├── day04/
├── day05/
├── day06/
└── day07/
```

每个 Day 目录至少保留：

```text
environment.txt
commands.sh
notes.md
*.ptx
*.cubin
*.sass
ptxas.log
*.ncu-rep（需要时）
```

`/tmp` 产物不提交。每天真正沉淀的结论写入 [Week 2 PTX/SASS Worklog](../../notes/week02_ptx_sass.md)。

### 1.3 每日完成定义

每天必须留下四类输出：

```text
概念：一张图或一段闭卷解释
实验：真实命令和产物
证据：行号、指令、资源或 metric
口述：2～5 分钟“结论→证据→边界”
```

只阅读、只运行已有脚本、只找到一条指令，都不算完整完成。

---

# Day 1：CUDA C++ → PTX → cubin / SASS

## 今日目标

建立编译链心智模型。今天不分析性能，不解释大段指令。

## 阅读范围

1. [深水区 §3：CUDA C++、PTX、SASS 不是同一层](../../docs/topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md#3-day-1cuda-cptxsass-不是同一层)
2. 本文 Day 1，不扩展阅读其他章节。

## 实验对象

```text
$GEMM_REPO/projects/gemm/kernels/naive.cu
```

## 实验步骤

```bash
mkdir -p "$WEEK2_TMP/day01"
cd "$WEEK2_TMP/day01"

{
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  nvcc --version
  ncu --version
  cuobjdump --version
  git -C "$GEMM_REPO" rev-parse HEAD
} > environment.txt

cat > commands.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
: "${GEMM_REPO:?请先设置 GEMM_REPO}"

nvcc -O3 -lineinfo -arch=compute_80 -ptx \
  -I"$GEMM_REPO/projects/gemm/include" \
  "$GEMM_REPO/projects/gemm/kernels/naive.cu" \
  -o naive.compute80.ptx

nvcc -O3 -lineinfo -arch=sm_80 -cubin \
  -I"$GEMM_REPO/projects/gemm/include" \
  -Xptxas=-v \
  "$GEMM_REPO/projects/gemm/kernels/naive.cu" \
  -o naive.sm80.cubin 2> ptxas.log

cuobjdump --list-elf naive.sm80.cubin > elf_list.txt
cuobjdump --dump-sass naive.sm80.cubin > naive.sm80.sass
EOF

chmod +x commands.sh
./commands.sh
```

以上命令已在当前 A100 环境冒烟验证，可生成：

```text
naive.compute80.ptx
naive.sm80.cubin
naive.sm80.sass
ptxas.log
elf_list.txt
```

## 必须画出的图

```text
CUDA C++
  ↓ nvcc front-end / NVVM
PTX（compute_80 虚拟 ISA）
  ↓ ptxas
cubin（sm_80 机器码）
  ↓ cuobjdump 反汇编
SASS

PTX 也可能嵌入 fatbin
  ↓ 目标机器无兼容 cubin 时
Driver JIT
  ↓
目标架构机器码
```

## 必答五题

1. PTX 为什么不是 A100 最终执行的机器码？
2. `compute_80` 和 `sm_80` 分别约束哪一层？
3. ptxas 做什么，资源日志说明什么？
4. 已有兼容 cubin 时，为什么通常不走 PTX JIT？
5. `-lineinfo` 与 `-G` 有什么区别，为什么性能实验不用 `-G`？

## 当日验收

- [x] 不看资料重画编译链；
- [x] 独立重新执行四条核心编译/反汇编命令；
- [x] `elf_list.txt` 能证明存在 `sm_80` code object；
- [x] 从 `ptxas.log` 读出 registers、shared、spill；
- [x] 录一段 3 分钟口述。

## 一句话记忆

> PTX 是虚拟 ISA；ptxas 为具体 `sm_xx` 做指令选择和寄存器分配；SASS 是机器码的反汇编表示。

---

# Day 2：沿 Naive GEMM 读 PTX 数据流

## 今日目标

只追一条输出元素的数据流，不逐行翻译 PTX。

## 阅读范围

1. [深水区 §4.1：先读数据流](../../docs/topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md#41-先读数据流不要逐字符翻译)
2. [深水区 §4.2：PTX 最小词典](../../docs/topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md#42-ptx-最小词典)

## 使用产物

```text
$WEEK2_TMP/day01/naive.compute80.ptx
$GEMM_REPO/projects/gemm/kernels/naive.cu
```

## 阅读顺序

```text
.entry naive_kernel
→ kernel 参数
→ %ctaid / %ntid / %tid
→ row / column
→ 边界 predicate
→ A[row,k] 地址
→ B[k,column] 地址
→ ld.global
→ fma
→ 循环 branch
→ st.global C[row,column]
```

## 建立数据流表

在 `$WEEK2_TMP/day02/notes.md` 填写：

| 源码动作 | PTX label / 行号 | 输入寄存器 | 输出寄存器 | 说明 |
|---|---|---|---|---|
| 读取 block/thread id | | | | |
| 计算 row | | | | |
| 计算 column | | | | |
| 计算 A 地址 | | | | |
| 计算 B 地址 | | | | |
| 加载 A/B | | | | |
| FMA 累加 | | | | |
| 写回 C | | | | |

## 重点认识的类别

```text
ld.param
mov %ctaid / %ntid / %tid
mul.lo / mul.wide
mad.lo / mad.wide
cvta
setp
@predicate bra
ld.global
fma.rn.f32
st.global
```

实际编译结果可能发生合并或使用不同变体，以真实 PTX 为准。

## 当日验收

- [ ] 指向一条 A/B `ld.global`，能向前追到地址来源；
- [ ] 从该 load 向后追到 FMA；
- [ ] 从 `st.global` 反向追到 accumulator；
- [ ] 说清 PTX 虚拟寄存器数量不等于物理 registers/thread；
- [ ] 3 分钟口述完整数据流。

## 禁止事项

- 不解释每条控制指令；
- 不推断精确 cycle；
- 不因看到 FMA 就说 compute-bound；
- 不把 PTX 寄存器编号当 ptxas 物理寄存器数量。

---

# Day 3：从 PTX 对照到 A100 SASS

## 今日目标

理解 PTX 与 SASS 不一一对应，但能按数据流类别建立对应。

## 阅读范围

1. [深水区 §4.3：SASS 看类别](../../docs/topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md#43-sass-阅读时看类别不背跨代拼写)
2. [深水区 §4.5：实验记录模板](../../docs/topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md#45-实验记录模板)

## 使用产物

```text
$WEEK2_TMP/day01/naive.sm80.sass
$WEEK2_TMP/day01/naive.compute80.ptx
```

## SASS 阅读顺序

```text
Function: naive_kernel
→ S2R：thread/block 信息
→ 整数地址计算
→ LDG：global load
→ FFMA：计算
→ STG：global store
→ predicate / branch
→ EXIT
```

## 对照表

| 源码动作 | PTX 证据 | SASS 证据 | 能证明 | 不能证明 |
|---|---|---|---|---|
| thread mapping | | | 索引生成路径 | 映射一定高效 |
| A/B global load | | | 实际 load 类别 | 一定 DRAM-bound |
| K 累加 | | | 实际 FFMA | 已接近算力峰值 |
| C store | | | 实际 store | store 是主瓶颈 |
| 边界与循环 | | | predicate/branch | 分支一定严重 divergence |

## 当日验收

- [ ] 能在 SASS 中找到 load、compute、store、control；
- [ ] 能解释一条 PTX 可能对应多条 SASS，或被合并/消除；
- [ ] 能说清 SASS 结论只对当前源码、编译参数和 `sm_80` 生效；
- [ ] 不看资料完成一张 PTX/SASS 对照表。

## 一句话记忆

> PTX 适合看语义，SASS 适合确认机器实际执行路径；两者按数据流对应，不按文本逐行对应。

---

# Day 4：横向比较五版 CUDA Core GEMM

## 今日目标

比较优化如何改变机器数据流，不做新的性能优化。

## 阅读范围

1. [深水区 §4.4：五版 GEMM 观察点](../../docs/topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md#44-五版-gemm-应该观察什么)
2. 公开 GEMM 项目中五份 kernel 源码和阶段文档。

## 实验对象

| 版本 | 源码 |
|---|---|
| Naive | `projects/gemm/kernels/naive.cu` |
| Shared | `projects/gemm/kernels/shared_tiled.cu` |
| Register | `projects/gemm/kernels/register_tiled.cu` |
| Vectorized | `projects/gemm/kernels/vectorized.cu` |
| Async 16B | `projects/gemm/kernels/double_buffer.cu` |

## 生成 PTX、cubin 和资源日志

在 `$WEEK2_TMP/day04` 中，对五个 translation unit 重复 Day 1 的命令。文件名映射：

```text
naive.cu          → naive
shared_tiled.cu   → shared
register_tiled.cu → register
vectorized.cu     → vectorized
double_buffer.cu  → async-16b
```

公开仓已经提交了 Vectorized 与 Async 的紧凑 SASS 证据，直接读取：

```text
$GEMM_REPO/projects/gemm/results/evidence/vectorized-sass.md
$GEMM_REPO/projects/gemm/results/evidence/async-16b-sass.md
```

本周不要运行会写回公开仓 `results/evidence/` 的提取脚本。需要重新观察完整 SASS 时，直接对 `$WEEK2_TMP/day04/*.cubin` 使用 `cuobjdump --dump-sass`，输出仍放 `/tmp`。

## 必填对照表

| 版本 | Global load | Shared 路径 | 同步 | 主计算 | registers/thread | spill | 一句话变化 |
|---|---|---|---|---|---:|---:|---|
| Naive | | 无 | 无 block barrier | | | | |
| Shared | | | | | | | |
| Register | | | | | | | |
| Vectorized | | | | | | | |
| Async 16B | | | | | | | |

## 必须确认的现有证据

- Vectorized 是否真实出现 `LDG.E.128`；
- Async 16B 是否真实出现 `LDGSTS.E.BYPASS.128`；
- 五个版本计算主体是否仍是 CUDA Core `FFMA`；
- Register/Vectorized/Async 的 ptxas 资源差异；
- 是否出现 `LDL`/`STL` 静态风险信号。

## 当日验收

- [ ] 每个版本都能用一句话说明机器数据流变化；
- [ ] 每句结论至少有一个 PTX、SASS 或 ptxas 定位；
- [ ] 能解释“源码用了 `float4`”为什么不等于“一定生成宽 load”；
- [ ] 能解释“出现 `LDGSTS`”为什么不等于 Async 一定更快。

---

# Day 5：寄存器、spill、local memory 与 ILP

## 今日目标

建立 register reuse、ILP、occupancy、spill 的取舍模型。

## 阅读范围

1. [深水区 §5：寄存器、spill 与向量化](../../docs/topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md#5-day-3寄存器spill-与向量化)
2. [Occupancy 专题](../../docs/topics/performance/Occupancy详解_从入门到调优.md)仅按问题查阅，不通读。

## 先解释已有真实数据

```text
Register：   约 72 registers/thread
Vectorized：约 83 registers/thread
Async 16B：约 58 registers/thread
```

结合正式结果回答：

1. 为什么 Vectorized 寄存器更多，却是最快手写版本？
2. 为什么 Async 寄存器更少、achieved occupancy 更高，仍比 Vectorized 慢？
3. 0 spill stores/loads 能证明什么，不能证明什么？
4. SASS 无 `LDL/STL` 又增加了哪一层证据？

## 单变量实验准备

在 `$WEEK2_TMP/day05/ilp_dependency.cu` 中由你手写一个独立的最小依赖链 microbenchmark，做：

```text
baseline：一条长依赖 accumulator 链
variant：  相同 FLOP，使用多个独立 accumulator
```

要求：

- 数学工作量一致；
- grid/block/输入规模一致；
- 只改变 accumulator 独立性；
- baseline 与 variant 必须保存在同一份源码、同一提交外临时目录中；
- 在 `notes.md` 记录源码 SHA-256、完整编译命令和输入规模，使 Day 6 可复现；
- 核心实验代码由你自己写；
- 用 CPU reference 或解析公式验证输出；
- 记录 ptxas registers、spill、SASS 和正常 wall-clock。

如果当天写不完，只完成 baseline、接口、reference 和实验假设，Day 6 继续；不允许直接复制完整答案。

## 当日验收

- [ ] 能解释 local address space 为什么“线程私有”但可能落到片外；
- [ ] 能解释 registers 少不保证更快；
- [ ] 能解释 occupancy 高不保证 eligible/issued 高；
- [ ] baseline 与 variant 的唯一变量写清楚；
- [ ] 给出实验反证条件。

## 反证条件示例

> 如果多个 accumulator 增加寄存器却没有提高 eligible/issued 或 wall-clock，说明当前 kernel 并不主要受单条依赖链限制，或资源代价抵消了 ILP 收益。

---

# Day 6：Scheduler、Scoreboard 与可证伪诊断

## 今日目标

完成一次“现象 → 假设 → 单变量 → 复测”的完整闭环。

## 阅读范围

1. [深水区 §12：Scheduler 与 Scoreboard](../../docs/topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md#12-day-10为什么有很多-warp仍可能发不出指令)
2. [深水区 §13：怎样读 Warp Stall](../../docs/topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md#13-day-11怎样读-warp-stall而不被指标牵着走)
3. [ncu §5：判读总流程](../../docs/topics/performance/Nsight_Compute_ncu详解.md#5-判读总流程背下来)
4. [ncu §9：Warp State / Scheduler](../../docs/topics/performance/Nsight_Compute_ncu详解.md#9-warp-state--schedulerlatency-bound-看这里)
5. [ncu §15：常见坑](../../docs/topics/performance/Nsight_Compute_ncu详解.md#15-常见坑)

## 先复述 GEMM 现有负结果

```text
现象：Async 16B 比 Vectorized 慢约 4.7%

支持证据：
  long scoreboard：      1.88 → 0.05
  short scoreboard：     0.49 → 1.87
  shared conflict：      1.3-way → 2.2-way
  achieved occupancy：  26.78% → 38.49%

结论：
  cp.async 隐藏了大部分 global-memory dependency，
  但 shared-memory 冲突和 MIO 等待上升，
  瓶颈转移后墙钟没有改善。
```

这是一条真实项目诊断，但它不是严格单变量 microbenchmark。因此仍需完成 Day 5 的依赖链实验。

## ncu 诊断顺序

```text
正确性 + 正常 wall-clock
→ SpeedOfLight
→ LaunchStats / registers / shared
→ SchedulerStats：active / eligible / issued
→ Warp State：主要 stall
→ 对应源码和 SASS
→ 单一假设
→ variant
→ correctness + wall-clock + resources + ncu 复测
```

Metric 名称随 ncu 版本变化，先执行：

```bash
ncu --list-sections
ncu --query-metrics | grep -E 'eligible|issued|warps_active|scoreboard' | head -80
```

不要未经本机查询直接照抄其他版本的 metric 名称。

## 诊断报告模板

写入 `$WEEK2_TMP/day06/diagnosis.md`，再把结论压缩到 worklog：

```text
现象：
高层 bound：
active / eligible / issued：
主要 stall：
源码/SASS 对应位置：
候选机制：
反证条件：
唯一修改：
correctness：
资源变化：
指标变化：
正常 wall-clock：
结论：支持 / 否定 / 证据不足
适用边界：GPU / shape / dtype / build / metric set
```

## 当日验收

- [ ] 能区分 active、eligible、issued；
- [ ] 能解释 scoreboard 是依赖跟踪，不是 cache；
- [ ] 不把 long scoreboard 直接等同于 DRAM 带宽瓶颈；
- [ ] baseline/variant 正确性均通过；
- [ ] 正常 wall-clock 与 profiler replay 时间分开；
- [ ] 无论假设成功或失败，都写出完整结论。

---

# Day 7：WMMA → PTX MMA → SASS HMMA

## 今日目标

理解 Tensor Core 三层证据，不追求高性能 Tensor Core GEMM。

## 阅读范围

1. [Tensor Core §1：CUDA Core 到 Tensor Core](../../docs/courses/cuda/Week3_TensorCore学习文档.md#1-从-cuda-core-到-tensor-core)
2. [Tensor Core §3：WMMA、MMA、WGMMA](../../docs/courses/cuda/Week3_TensorCore学习文档.md#3-wmmammawgmma不要混成一个词)
3. [深水区 §6：WMMA、PTX MMA、SASS HMMA](../../docs/topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md#6-day-4wmmaptx-mmasass-hmma-的关系)
4. [深水区 §7：`ldmatrix`](../../docs/topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md#7-day-5ldmatrix-到底解决什么)只读概念；当前标本不以 shared memory 为 fragment 来源，因此不要求生成 `ldmatrix`/LDSM。

## 观察对象（本周唯一例外）

```text
$CUDA_STUDY/week06_tensorcore/wmma_fp16_gemm.cu
```

Day 7 不再使用 CUDA Core GEMM 做诊断，而是只观察已有正确 WMMA 实现，不在今天重写或优化完整 WMMA GEMM。该例外只服务于 API → PTX → SASS 三层关系。

## 三层关系

```text
CUDA C++ API
  wmma::load_matrix_sync
  wmma::mma_sync
  wmma::store_matrix_sync

PTX 虚拟 ISA
  wmma.load / fragment lowering
  wmma.mma.sync（当前标本为 m16n16k16）

A100 SASS
  HMMA 机器指令
```

## 必须理解

- fragment 是 warp 协作持有的逻辑矩阵片段；
- 不能把 fragment 当成每线程连续二维数组；
- 当前 WMMA 标本的逻辑 shape 是 `m16n16k16`；M/N/K 分别对应输出行、输出列、归约维；
- `ldmatrix` 解决 shared memory 到 warp fragment 的协作搬运，但当前标本直接从 global memory 加载 fragment，SASS 中不出现 LDSM 是符合实现数据流的结果；
- shared layout、对齐、bank mapping 仍影响 Tensor Core 数据供给；
- 看到 HMMA 证明走了 Tensor Core 指令路径，不证明性能高。

## 当日验收

- [ ] 从 WMMA 源码生成 PTX、cubin、SASS；
- [ ] 在 PTX 中定位 `wmma.mma.sync...m16n16k16` 类指令；
- [ ] 在 SASS 中定位 HMMA，并解释当前实现为什么没有 LDSM；
- [ ] 能画 API → PTX → SASS 三层图；
- [ ] 5 分钟口述三层证据和性能边界。

## 本周不做

- 不手写完整 inline PTX `mma.sync`；
- 不推导所有 lane/register 精确映射；
- 不做 3-stage Tensor Core pipeline；
- 不和 cuBLAS 排名。

---

# 2. 周末总验收

## 2.1 90 分钟闭卷

1. 10 分钟：画 CUDA C++ → PTX → ptxas → cubin/SASS → JIT；
2. 15 分钟：从 Naive PTX 找 thread mapping、A/B load、FMA、C store；
3. 15 分钟：从 Naive SASS 找 `S2R/LDG/FFMA/STG/control`；
4. 10 分钟：解释 ptxas registers/shared/spill；
5. 15 分钟：解释 active/eligible/issued 与 scoreboard；
6. 15 分钟：复述 Day 6 单变量诊断；
7. 10 分钟：解释 WMMA → PTX MMA → SASS HMMA。

## 2.2 通过标准

| 能力 | 通过条件 |
|---|---|
| 编译链 | 不看命令说明生成 PTX/cubin/SASS |
| PTX | 独立追踪 Naive 一条输出数据流 |
| SASS | 按类别找到 load/compute/store/control |
| 资源 | 解释 registers/shared/spill/local |
| Scheduler | 区分 active/eligible/issued 与 scoreboard |
| 实验 | 完成一次严格单变量、可证伪实验 |
| Tensor Core | 讲清 API/PTX/SASS 三层，不夸大性能 |

全部通过后进入 Week 3。任一项不过：记录为 Week 3 每天开场最多 30 分钟补考，不重学整周。

## 2.3 本周最终交付

```text
/tmp/cuda_focus/week02/day01..day07/   临时完整实验产物
notes/week02_ptx_sass.md               永久 worklog
一张五版 GEMM 指令观察表
一份 ptxas/register/spill 对照
一份单变量诊断报告
一段 Tensor Core 三层口述
```

---

# 3. 每日最小版（只有 4 小时时）

| 模块 | 时间 |
|---|---:|
| 指定阅读 | 45 分钟 |
| 本人动手实验 | 90 分钟 |
| 证据整理 | 45 分钟 |
| 闭卷/口述 | 30 分钟 |
| worklog | 30 分钟 |

时间不足时按以下顺序砍：

1. 额外指令细节；
2. 多 shape；
3. 排版美化；
4. 扩展阅读。

不得砍：真实命令、正确性、证据定位、口述和 worklog。

## 本周一句话

> 不背汇编；沿数据流读 PTX/SASS，用 ptxas 和 ncu 提出可证伪假设，再回到 correctness 与 wall-clock 判断。
