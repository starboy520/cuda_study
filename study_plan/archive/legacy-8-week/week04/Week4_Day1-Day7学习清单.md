# Week 4 每日详细安排（Day1–Day7）

> **主题**：GEMM 优化阶梯 + cuBLAS 当标尺 + Nsight Compute（ncu）入门
> **为什么这样排**：GEMM 是面试手写题天花板、也是练 profiler 的最佳载体。把"手写 GEMM 优化"
> 和"用 ncu 验证每一步"绑在一起学，比把工具单独留到后面干学高效得多。cuBLAS 在这里只当
> **性能标尺**（看你离工业库多远），不当主课。
> **硬件**：Tesla T4（sm_75）。FP32 峰值约 8.1 TFLOP/s、显存带宽约 320 GB/s（Roofline 用）。
> **前置**：Week 3 已完成 atomic / warp shuffle / scan / histogram / stream（都 PASS）。
> **本周交付**：`week04_gemm/` 下 naive / tiled / register-tiled 三版 GEMM + cuBLAS 对比 +
> 每版的 ncu 关键指标，`notes/week04.md`。

> **目标岗位假设**：本计划偏 **算子开发 / CUDA 性能工程**（GEMM 手写 + 实测为重）。若你更偏
> 通用 CUDA 应用岗，可把 Day3 register tiling 压缩、Day6 多花点在库使用上——告诉我即可调整。

---

## 使用方式

- 按 Day1 → Day7 顺序做；每天有「学什么 / 看什么 / 动手 / 完成标准」。
- 每个优化都要落到**自己手写的代码**，并且**先说动机、再写、最后用数据验证**（这是本周和
  前几周最大的不同：每一步优化都要有 ncu 数据背书，不能只说"我加了 tiling"）。
- 每天在 `notes/week04.md` 记三件事：**目标 / 实验结果（含 GFLOPS 和关键指标）/ 遇到的问题**。
- 阅读标注：📖 精读 · 👀 扫读 · ✍️ 必须自己写 · ⏭️ 本周跳过。

---

## 本周总览（一张表看完）

| Day | 主题 | 动手产出 | 关键概念 |
|-----|------|----------|----------|
| 1 | GEMM baseline + Roofline 定位 | `gemm_naive` + GFLOPS 记录 | AI / Roofline / 为什么 naive 受限 |
| 2 | Shared-memory tiling | `gemm_tiled` | tile / 数据复用 / 提升 AI |
| 3 | Register tiling | `gemm_reg` | 每线程多输出 / ILP |
| 4 | cuBLAS 当标尺 | `gemm_cublas` 对比 | 何时调库 vs 手写 |
| 5 | ⭐ ncu 入门（本周重点） | 每版的 SoL/occupancy/bank 指标 | 指标→瓶颈→验证 |
| 6 | Thrust / 算子封装（轻量） | 一个封装好的 GEMM 接口 | 库边界 / 接口设计 |
| 7 | 复盘 + 性能表 + 讲解 | `notes/week04.md` 定稿 | 优化叙事 / 3 分钟讲清 |

---

## 本周用得上的现成资料（你工作区里已有）

| 主题 | 教材正文（读） | 可运行实验（对照） |
|------|---------------|-------------------|
| GEMM 数学与上限 | `cuda_deep_course/.../volume06_operators/01_GEMM数学_布局与性能上限.md` | `week01_basics/mat_mul_naive/` |
| GEMM 优化阶梯 | `.../volume06_.../02_GEMM优化阶梯_Tiling与寄存器分块.md` | `cuda_deep_course/labs/06_operators/gemm_tiled/` |
| 向量化/双缓冲/Tensor Core | `.../volume06_.../03_向量化_双缓冲与Tensor_Core.md` | — |
| Roofline（T4 实算） | `.../volume05_performance/02_性能指标_Scaling与Roofline.md` §5.1 | — |
| ncu 指标→决策 | `.../volume05_performance/05_Nsight_Compute与Compute_Sanitizer.md` §5.1 | — |
| 库的边界 | `.../volume06_.../06_库的边界与算子工程化.md` | — |

官方文档：
- 👀 **cuBLAS** `cublasSgemm` 用法（API 参考，用到再查，不必通读）
- 👀 **Thrust** 快速上手（`thrust::device_vector` / `transform` / `reduce`）
- 📖 **Best Practices Guide**：Memory Optimizations、Execution Configuration（配合 tiling 看）

⏭️ **本周跳过**（留到后面）：Tensor Core / WMMA（卷六/03 后半，Week5）、双缓冲软件流水、
CUTLASS 深入、混合精度。先把 FP32 的 tiling + register tiling 吃透。

---

## ⚠️ 关于 ncu 权限（先确认，否则 Day5 卡住）

`ncu` 采集硬件计数器需要权限。先跑一次试探：

```bash
ncu --version          # 确认能用
ncu --set basic ./week01_basics/mat_mul_naive/mat_mul   # 试采集
```

如果报 `ERR_NVGPUCTRPERM`（无权限访问 GPU 性能计数器）：
- 这**不是你代码的错**，是当前用户无权读硬件计数器。
- 解决要管理员按 NVIDIA 文档开放 profiling 权限（加载 nvidia 模块时设
  `NVreg_RestrictProfilingToAdminUsers=0`），**不要自己在生产机乱改**。
- 若暂时无法解决：Day5 仍可用 **CUDA Event 测耗时 + GFLOPS** 做对比（退而求其次），ncu 部分
  等有权限再补。先把"手写优化 + 量化加速"这条主线走完。

---

## Day 1：GEMM baseline 与 Roofline 定位

**学什么**
- GEMM 定义：`C[M×N] = A[M×K] × B[K×N]`，约 `2·M·N·K` FLOP（FMA 算 2）。
- 为什么 naive GEMM 慢：每个输出元素都从 global 重新读整行整列，**数据复用为零 → AI 极低 → memory-bound**。
- 用 Roofline 定位：算出 naive 的 AI，落在 T4 拐点（≈25 FLOP/byte）左边，确认受带宽限制。

**看什么**
- 📖 教材 `volume06_.../01_GEMM数学_布局与性能上限.md`（数学、布局、AI 推导）。
- 📖 教材 `volume05_.../02_性能指标_Scaling与Roofline.md` §5.1（T4 拐点 25、GEMM 为什么靠 tiling 逃离带宽墙）。

**动手** ✍️ 新建 `week04_gemm/gemm_naive/gemm_naive.cu`（可参考 `week01_basics/mat_mul_naive/`）
1. 写 naive GEMM：每个线程算一个 `C[i][j]`，循环 K 累加。
2. 加 CPU reference + 正确性校验（非方阵、非整除也要测）。
3. 用 CUDA Event 计时，算 **GFLOPS = 2·M·N·K / 时间 / 1e9**，记录基线（如 1024³）。

**完成标准**
- [ ] naive GEMM 结果正确（和 CPU 对比，含非方阵）。
- [ ] 记录 baseline GFLOPS。
- [ ] 能口述：为什么 naive 是 memory-bound？（复用为零、AI 极低）

---

## Day 2：Shared-memory Tiling（核心）

**学什么**
- tiling 的本质：把 A、B 的子块搬进 shared，让一块数据**读一次、复用多次** → AI 上升 → 逃离带宽墙。
- 一个 tile 的流程：协作加载 A/B 子块到 shared → `__syncthreads()` → 每线程对 tile 做多次乘加 → 滑到下一个 K 方向 tile。
- 边界处理：M/N/K 不是 tile 整数倍时的越界判断。

**看什么**
- 📖 教材 `volume06_.../02_GEMM优化阶梯_Tiling与寄存器分块.md`（tiling 推导）。
- 对照实验 `labs/06_operators/gemm_tiled/gemm_tiled.cu`（现成 tiled 版，读懂每一步）。

**动手** ✍️ 新建 `week04_gemm/gemm_tiled/gemm_tiled.cu`
1. 实现 shared-memory tiled GEMM（如 TILE=16 或 32）。
2. 正确性校验（重点测非整除规模，验证边界）。
3. 测 GFLOPS，和 Day1 naive 对比（参考：tiled 通常有 1.5~2x+ 提升）。

**完成标准**
- [ ] tiled GEMM 正确（含非整除规模）。
- [ ] 记录 GFLOPS，和 naive 对比有明显提升。
- [ ] 能解释 tiling 如何提升 AI（一块数据复用 TILE 次）。

---

## Day 3：Register Tiling（每线程算多个输出）

**学什么**
- shared tiling 之后的下一步：让**每个线程算多个 C 元素**（如 4×4 微块），把中间结果攒在**寄存器**里。
- 为什么更快：减少 shared 访问次数、提升 ILP（指令级并行）、更好地摊薄加载成本。
- 代价：寄存器压力上升，可能降 occupancy —— 又一个权衡（Day5 用 ncu 验证）。

**看什么**
- 📖 教材 `volume06_.../02_...Tiling与寄存器分块.md` 后半（register tiling）。

**动手** ✍️ 新建 `week04_gemm/gemm_reg/gemm_reg.cu`
1. 在 Day2 基础上，让每线程算一个小微块（如 4×4），中间和放寄存器数组。
2. 正确性校验。
3. 测 GFLOPS，和 tiled 对比。

**完成标准**
- [ ] register-tiled GEMM 正确。
- [ ] 记录 GFLOPS（可能比 tiled 再快一截）。
- [ ] 能说出 register tiling 省了什么、代价是什么（寄存器压力 vs occupancy）。

---

## Day 4：cuBLAS 当标尺 + 库的边界

**学什么**
- cuBLAS 怎么用：`cublasCreate` → `cublasSgemm`（注意它是**列主序**，行主序数据要转置或调换参数）→ `cublasDestroy`。
- 它不是要你"学算法"，而是当**性能上限的标尺**：你的手写版到了 cuBLAS 的百分之几？
- 何时调库、何时手写：库覆盖标准算子且接近峰值；手写用于库没有的融合算子、特殊形状、特殊精度。

**看什么**
- 📖 教材 `volume06_.../06_库的边界与算子工程化.md`（手写 vs 库的边界）。
- 👀 cuBLAS `cublasSgemm` API 参考（用到再查）。

**动手** ✍️ 新建 `week04_gemm/gemm_cublas/gemm_cublas.cu`
1. 用 cuBLAS 跑同样的 GEMM（注意列主序，先在小矩阵上验证结果对）。
2. 测 cuBLAS 的 GFLOPS。
3. 做一张表：naive / tiled / reg / cuBLAS 的 GFLOPS 和"占 cuBLAS 百分比"。

```bash
# 编译时链接 cuBLAS
nvcc -arch=sm_75 gemm_cublas.cu -o gemm_cublas -lcublas
```

**完成标准**
- [ ] cuBLAS 结果与 CPU 一致（列主序处理对）。
- [ ] 四版 GFLOPS 对比表完成，知道自己手写版到了 cuBLAS 的百分之几。
- [ ] 能回答"什么时候该调库、什么时候该手写"。

---

## Day 5：⭐ Nsight Compute（ncu）入门 —— 本周重点

**学什么**（不求系统精通，先学会"看几个关键指标判断瓶颈"）
- ncu 基本用法：`--set`、`--kernel-name`、`--section`、`-o` 存报告。
- 三个最该看的：**Speed of Light**（memory 还是 compute bound）、**Occupancy**（理论 vs 实际）、
  **shared bank conflict**（验证 tiling 有没有冲突）。
- 用"指标 → 瓶颈 → 验证假设"的闭环，把前四天的优化用数据解释。

**看什么**
- 📖 教材 `volume05_.../05_Nsight_Compute与Compute_Sanitizer.md` §5（阅读顺序）+ §5.1（指标→决策表）。
- 📖 教材 `volume05_.../04_Nsight_Systems系统时间线.md`（系统级，扫读，知道和 ncu 的分工）。

**动手** ✍️ 对 Day1-3 三个 GEMM 各跑一次 ncu
```bash
# 编译时加 -lineinfo，ncu 才能关联源码行
nvcc -arch=sm_75 -lineinfo gemm_tiled.cu -o gemm_tiled

# 跑 Speed of Light + Occupancy（按你程序的实际参数）
ncu --set basic --kernel-name regex:gemm ./gemm_tiled <你的参数>

# 存完整报告，之后用 ncu-ui 打开细看
ncu --set full -o reports/gemm_tiled --kernel-name regex:gemm ./gemm_tiled <你的参数>
```
1. naive：看 SoL 是否 memory-bound（验证 Day1 的假设）。
2. tiled：看 DRAM throughput 是否下降、SoL 是否往 compute 移动（验证 tiling 提升了 AI）。
3. reg：看 occupancy 有没有因寄存器压力下降、性能是否仍提升（验证 Day3 的权衡）。

**完成标准**
- [ ] 能用 ncu 跑出三版的 Speed of Light 并读懂 memory/compute 占比。
- [ ] 能指出 naive→tiled 时哪个指标按预期变化（DRAM throughput / AI）。
- [ ] 给三版各写一行："瓶颈是什么、哪个指标证明的"。

> 若 ncu 无权限（见上面 ⚠️）：用 GFLOPS + Roofline 推断瓶颈，把 ncu 部分标记为"待权限"，
> 不阻塞本周主线。

---

## Day 6：Thrust 与算子封装（轻量，半天即可）

**学什么**
- Thrust 快速上手：`thrust::device_vector`、`thrust::reduce` / `transform` / `sort` —— 高层 C++ 抽象，
  适合快速搭原型、做正确性参考。
- 算子封装思路：把你的 GEMM 包成一个干净接口（输入/输出指针、尺寸、stream），像库一样可调用。
- 知道"手写 kernel / Thrust / cuBLAS"各自的位置。

**看什么**
- 👀 Thrust 文档快速上手（不必深入）。
- 📖 教材 `volume06_.../06_库的边界与算子工程化.md`（接口与工程化）。

**动手** ✍️
1. 用 Thrust 写一个 reduce 或 transform，对比你 Week3 手写版（体会"高层抽象省事但不可控"）。
2. 把你 Day3 的 register-tiled GEMM 封装成一个函数接口 `myGemm(C, A, B, M, N, K, stream)`，
   内部处理 launch 配置，外部像调库一样用。

**完成标准**
- [ ] 跑过一个 Thrust 例子，能说出它和手写 kernel 的取舍。
- [ ] GEMM 封装成干净接口，main 里一行调用。

---

## Day 7：复盘 + 性能表定稿 + 3 分钟讲解

**动手**
1. 整理 `notes/week04.md`：
   - GEMM 四版（naive/tiled/reg/cuBLAS）的 GFLOPS + 占 cuBLAS 百分比表。
   - 每版的 ncu 关键指标（SoL / occupancy / bank conflict）。
   - 每一步优化的"问题 → 改法 → 快了多少 → 哪个指标证明的"。
2. 写一段 **3 分钟口述**：从 naive 到 register-tiled，你怎么一步步优化、每步为什么有效、
   离 cuBLAS 还差多少、差在哪。这是面试最常考的"讲一个你做过的优化"。

**本周核心问题自测**（能口头答出就算掌握）
- [ ] naive GEMM 为什么 memory-bound？tiling 为什么能改善？
- [ ] register tiling 在 shared tiling 基础上又省了什么？代价是什么？
- [ ] 怎么用 ncu 判断一个 kernel 是 memory-bound 还是 compute-bound？
- [ ] 你的手写 GEMM 到了 cuBLAS 的百分之几？还差的部分可能差在哪？
- [ ] 什么时候该调 cuBLAS、什么时候该手写 kernel？

---

## 本周交付清单

```text
week04_gemm/
├── gemm_naive/gemm_naive.cu      (Day1)
├── gemm_tiled/gemm_tiled.cu      (Day2)
├── gemm_reg/gemm_reg.cu          (Day3)
├── gemm_cublas/gemm_cublas.cu    (Day4)
└── reports/                       (Day5：ncu 报告 .ncu-rep)

notes/week04.md  —— 四版 GFLOPS 对比表 + ncu 指标 + 优化叙事 + 3 分钟讲解稿
```

---

## 和原计划的差异说明

```text
原计划 Week4：库与抽象层（cuBLAS / Thrust / 封装）单独占一周
本计划 Week4：GEMM 优化阶梯为主线，cuBLAS 当标尺、ncu 提前入门，Thrust/封装降为半天

原因：GEMM 手写 + profiler 实测是面试最该早练的硬技能；cuBLAS/Thrust 本身是 API 调用，
不值得单独占整周。把它们融进 GEMM 周，省下的时间给"手写 + 实测"这两个真正的重点。

连锁调整（建议）：
  Week5  性能优化方法论 + Nsight 系统深化（把 Day5 入门的 ncu 系统学完）
  Week6  卷六算子：softmax → layernorm → 融合
  Week7  作品集：挑一个算子做到极致 + 技术报告
  Week8  面试冲刺：四大手写题默写 + 概念题
```
