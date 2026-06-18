# Week 5 每日详细安排(Day1–Day7)

> **主题**:性能工程方法论 + Nsight 系统实战(把 Week4 入门的 ncu 系统学完)
> **为什么这周**:Week4 你已会手写 GEMM 优化阶梯、初识 ncu。Week5 把 profiler 从"会跑命令"
> 提升到"会判读、会定位、会量化"——这是面试区分度最高的能力。
> **硬件**:Tesla T4(sm_75)。FP32 8.1 TFLOP/s、带宽 320 GB/s、Roofline 拐点≈25。
> **前置**:Week4 完成(GEMM naive/tiled/reg + cuBLAS 对比 + ncu 入门)。
> **本周交付**:`week05_perf/` 下对 reduction/transpose/GEMM 的完整 profiler 分析报告 +
> `notes/week05.md`(每个 kernel 的 bound 类型、瓶颈、优化前后指标)。

---

## 使用方式

- 按 Day1→Day7;每天有「学什么 / 看什么 / 动手 / 完成标准」。
- 本周核心是**判读 + 量化**:每个结论都要有 profiler 指标背书,不能"我觉得"。
- 每天在 `notes/week05.md` 记:目标 / 实验数据 / 遇到的问题。
- 阅读标注:📖 精读 · 👀 扫读 · ✍️ 必须动手 · ⏭️ 跳过。

---

## 本周总览

| Day | 主题 | 动手产出 | 关键概念 |
|---|---|---|---|
| 1 | Roofline 与 bound 判断 | 三个 kernel 的 AI + Roofline 落点 | AI / 拐点 / memory vs compute bound |
| 2 | Nsight Compute 深入 | 各 kernel 的 SoL/Memory/Occupancy 报告 | SoL / section / 指标判读 |
| 3 | Occupancy 调优实战 | 扫 block size + 寄存器调优 | 理论/实际 occupancy / spill |
| 4 | Warp divergence 与 stall | 找出并消除一个 divergence | divergence / stall reason |
| 5 | Nsight Systems 时间线 | 一个程序的系统级时间线分析 | 传输/计算重叠 / GPU 空洞 |
| 6 | Compute Sanitizer | 抓 race/越界/未初始化 | memcheck/racecheck/synccheck |
| 7 | 完整优化案例 + 报告 | 一份"问题→证据→优化"闭环报告 | 因果验证 / 量化 |

---

## 本周用得上的现成资料

| 主题 | 教材正文 |
|---|---|
| APOD 与 benchmark | `cuda_deep_course/course/volume05_performance/01_APOD与可靠Benchmark.md` |
| Roofline(T4 实算) | `.../volume05_performance/02_性能指标_Scaling与Roofline.md` §5.1 |
| Occupancy 调优 | `docs/Occupancy详解_从入门到调优.md`(进阶篇,本周主力) |
| Nsight Compute 判读 | `.../volume05_performance/05_Nsight_Compute与Compute_Sanitizer.md` §5.1 |
| Nsight Systems | `.../volume05_performance/04_Nsight_Systems系统时间线.md` §5.1 |
| 完整优化案例 | `.../volume05_performance/06_完整优化案例_复习与面试.md` |

官方:Nsight Compute / Systems User Guide、Best Practices Guide。

> ⚠️ ncu 权限:若 `ERR_NVGPUCTRPERM`,见 Week4 计划的权限说明。无权限时用 Event 计时 +
> Roofline 推断代替,不阻塞主线。

---

## Day 1:Roofline 与 bound 判断

**学什么**
- arithmetic intensity(AI)= FLOP / bytes,Roofline 横轴。
- T4 拐点 ≈ 25 FLOP/byte:AI 低于它 → memory-bound,高于 → 才可能 compute-bound。
- 怎么从代码估算一个 kernel 的 AI。

**看什么** 📖 `volume05/02` §5.1(T4 拐点 + 四 kernel AI 表)。

**动手** ✍️ 对你 Week2-4 的三个 kernel 算 AI 和 Roofline 落点
1. reduction:AI≈?(每读一个 float 做一次加)→ memory-bound
2. transpose:AI≈0(纯搬运)→ memory-bound
3. tiled GEMM:AI 随 tile 上升 → 接近/超过拐点
4. 画一张简单的 Roofline,把三个 kernel 标上去。

**完成标准**
- [ ] 三个 kernel 都算出 AI 并判断 bound 类型
- [ ] 能解释为什么 GEMM 靠 tiling 把 AI 抬过拐点

---

## Day 2:Nsight Compute 深入

**学什么**
- ncu 的 `--set` / `--section` / `--kernel-name` / `-o` 用法。
- 三个最该看的 section:Speed of Light、Memory Workload、Occupancy。
- 用 SoL 的 memory% / compute% 验证 Day1 的 bound 判断。

**看什么** 📖 `volume05/05` §5(阅读顺序)+ §5.1(指标→决策表)。

**动手** ✍️ 对三个 kernel 各跑一次 ncu(记得 `-lineinfo` 编译)
```bash
ncu --set full -o reports/reduction --kernel-name regex:reduce ./reduction 16777216
ncu --section SpeedOfLight --kernel-name regex:transpose ./transpose 4096 4096
```
1. 看每个 kernel 的 SoL,确认 memory/compute 占比和 Day1 预测一致吗?
2. memory-bound 的看 Memory Workload:gld/gst_efficiency、DRAM throughput。
3. 记录关键指标到 `notes/week05.md`。

**完成标准**
- [ ] 三个 kernel 的 SoL 都跑出来,bound 类型和 Day1 对上
- [ ] transpose naive 的 gst_efficiency 低(验证写不合并)

---

## Day 3:Occupancy 调优实战

**学什么**
- 理论 vs 实际 occupancy,差异原因(尾部效应)。
- 用 `-Xptxas=-v` 看寄存器/spill;用 ncu Block Limit 定位 occupancy 瓶颈。
- 什么时候提 occupancy 有用(latency-bound)、什么时候没用。

**看什么** 📖 `docs/Occupancy详解_从入门到调优.md` 进阶篇(§11-§22,本周重点)。

**动手** ✍️
1. 对 GEMM 扫 block size {128,256,512},记录时间 + 理论/实际 occupancy + 寄存器。
2. 用 `__launch_bounds__` 或 `--maxrregcount` 调一次寄存器,观察 occupancy 和时间变化。
3. 验证:occupancy 升了,时间一定降吗?(不一定——记录反例)

**完成标准**
- [ ] 有一张 block size × (时间/occupancy/寄存器) 的对比表
- [ ] 能说出你的 GEMM 是被哪个资源限制了 occupancy(Block Limit)
- [ ] 理解"occupancy 不是越高越快"(有数据支撑)

---

## Day 4:Warp Divergence 与 Stall

**学什么**
- divergence:warp 内 lane 走不同路径 → 串行 → 最坏 32x。
- ncu Warp State 的 stall reason(Long Scoreboard/Barrier/MIO...)的含义。
- 怎么从 source counters 定位 divergence 到具体行。

**看什么** 📖 `volume05/03`(divergence + stall)、`volume05/05` §5.1(stall→动作)。

**动手** ✍️
1. 写一个故意有 divergence 的 kernel(如 `if (threadIdx.x % 2)`),用 ncu 看 warp
   execution efficiency 低。
2. 改成无 divergence 版(按 warp 对齐分支),对比效率和时间。
3. 看一个真实 kernel 的主要 stall reason,解释它意味着什么。

**完成标准**
- [ ] 能复现 divergence 导致的效率下降,并消除它
- [ ] 能读懂至少 3 个 stall reason 的含义

---

## Day 5:Nsight Systems 时间线

**学什么**
- nsys 看系统级时间线:CPU/GPU 活动、memcpy、kernel、同步、空洞。
- 判断瓶颈在 kernel 还是在传输/同步(Amdahl)。
- 传输与计算是否重叠。

**看什么** 📖 `volume05/04` §5(时间线阅读)+ §5.1(实战判读)。

**动手** ✍️
```bash
nsys profile --trace=cuda,osrt -o reports/sys ./your_program
nsys stats reports/sys.nsys-rep
```
1. profile 一个"H2D→kernel→D2H"循环的程序,看时间线。
2. 找出 GPU 空洞、传输占比、是否过度同步。
3. (可选)改成 pinned + 多 stream,看时间线是否出现重叠。

**完成标准**
- [ ] 能读懂系统时间线,指出最大瓶颈段
- [ ] 能区分"kernel 慢"和"传输/同步慢"两种情况

---

## Day 6:Compute Sanitizer

**学什么**
- 四个工具:memcheck(越界)、racecheck(竞争)、synccheck(barrier 误用)、initcheck(未初始化)。
- 正确顺序:reference → 测试 → sanitizer → benchmark → profiler。

**看什么** 📖 `volume05/05` §6-§7。

**动手** ✍️
```bash
compute-sanitizer --tool memcheck  ./your_kernel
compute-sanitizer --tool racecheck ./your_kernel
```
1. 故意制造越界,用 memcheck 抓出具体行。
2. 故意漏一个 `__syncthreads()`,用 racecheck 抓竞争。
3. 把 sanitizer 纳入你的测试流程。

**完成标准**
- [ ] 能用对应工具抓出越界和 race
- [ ] 理解"sanitizer 没报错≠一定没 bug"

---

## Day 7:完整优化案例 + 报告

**动手**
1. 挑一个 kernel(建议 transpose 或 reduction),走完整闭环并写进 `notes/week05.md`:
```text
症状 → 测基线 → 判 bound(SoL)→ 形成假设 → 上工具验证指标 →
优化 → 复测(指标按预期变化 + 时间下降)→ 结论
```
2. 这份报告要有:环境信息、优化前后指标对比、"为什么变快"的证据链。
3. 写一段 3 分钟口述:这个 kernel 你怎么定位瓶颈、怎么优化、快了多少。

**本周自测**
- [ ] 怎么判断 memory-bound 还是 compute-bound?
- [ ] 理论 occupancy 高但实际低,什么原因?
- [ ] divergence 发生在哪、代价多大?
- [ ] Nsight Systems 和 Compute 各看什么?
- [ ] 一次优化怎么证明"确实解决了目标瓶颈"?

---

## 本周交付清单

```text
week05_perf/
└── reports/                  ncu/nsys 报告 + 指标记录

notes/week05.md  —— 三个 kernel 的 bound 分析 + occupancy 调优表 +
                    一份完整优化闭环报告 + 3 分钟讲解稿
```

---

**返回**:[study_plan/README.md](README.md) · 下一步:[Week6 核心算子](Week6_核心算子开发.md)
