# Nsight Compute(ncu)详解:从命令到判读

> 一份独立、面向实操的 ncu 手册。从"怎么跑"到"指标怎么读、瓶颈怎么定位"。是你做性能优化
> (Week5 起)的主力工具文档。配合 [Occupancy详解](Occupancy详解_从入门到调优.md) 和课程
> [卷五/05](../cuda_deep_course/course/volume05_performance/05_Nsight_Compute与Compute_Sanitizer.md) 使用。
>
> 硬件基准:Tesla T4(sm_75)。命令通用,指标名随 ncu 版本略有差异,以 `--list-sections` 为准。

---

## 目录

1. [ncu 是什么、和 nsys 的分工](#1-ncu-是什么和-nsys-的分工)
2. [权限问题(第一道坎)](#2-权限问题第一道坎)
3. [命令体系:最常用的几条](#3-命令体系最常用的几条)
4. [Section 与 Set:采集什么](#4-section-与-set采集什么)
5. [判读总流程(背下来)](#5-判读总流程背下来)
6. [Speed of Light:第一刀分流](#6-speed-of-light第一刀分流)
7. [Memory Workload:memory-bound 看这里](#7-memory-workloadmemory-bound-看这里)
8. [Occupancy section](#8-occupancy-section)
9. [Warp State / Scheduler:latency-bound 看这里](#9-warp-state--schedulerlatency-bound-看这里)
10. [Source / Compute Workload](#10-source--compute-workload)
11. [关键指标速查表](#11-关键指标速查表)
12. [指标 → 动作对照表](#12-指标--动作对照表)
13. [完整实战案例](#13-完整实战案例)
14. [ncu-ui 图形界面](#14-ncu-ui-图形界面)
15. [常见坑](#15-常见坑)
16. [面试题](#16-面试题)

---

## 1. ncu 是什么、和 nsys 的分工

**Nsight Compute(ncu)= 单个 CUDA kernel 的"显微镜"。** 它深入一个 kernel 的微架构层面:
访存效率、计算吞吐、occupancy、warp 为什么停顿、指令对应到哪行源码。

和 **Nsight Systems(nsys)** 的分工要分清(面试常考):

```text
nsys(系统时间线):看"宏观"——CPU/GPU 活动、kernel/memcpy 时序、传输和计算是否重叠、
                  GPU 有没有空洞、是不是过度同步。回答"时间花在哪个阶段"。
ncu(单 kernel):  看"微观"——一个 kernel 内部的访存/计算/occupancy/stall 细节。
                  回答"这个 kernel 为什么慢"。

正确顺序:先 nsys 定位到慢的 kernel,再 ncu 深挖它。先系统、后 kernel。
```

> 一句话:**nsys 找"哪个 kernel 慢",ncu 答"它为什么慢"。**

---

## 2. 权限问题(第一道坎)

ncu 要读 GPU 硬件性能计数器,**需要权限**。先跑一次试探:

```bash
ncu --version                      # 确认装了
ncu --set basic ./your_program     # 试采集
```

如果报这个错:

```text
ERR_NVGPUCTRPERM: The user does not have permission to access
NVIDIA GPU Performance Counters on the target device.
```

含义:**不是你代码的错**,kernel 照样能跑,只是当前用户无权读硬件计数器。

解决(需要管理员):

```text
- 临时:用 sudo 跑 ncu(若策略允许)
- 永久:管理员在加载 nvidia 模块时设
        options nvidia NVreg_RestrictProfilingToAdminUsers=0
        (写入 /etc/modprobe.d/ 然后重启或重载模块)
- 不要自己在生产/共享服务器乱改安全配置
```

**没权限怎么办**(别卡住):

```text
退而求其次:用 CUDA Event 测耗时 + Roofline 推断 bound 类型(卷五/02)。
能定大方向(memory/compute bound),只是拿不到细粒度指标。等有权限再补 ncu。
```

---

## 3. 命令体系:最常用的几条

### 3.1 最小用法

```bash
ncu ./program                      # 采集所有 kernel 的基础指标(kernel 多会很慢)
```

### 3.2 过滤 kernel(几乎必用)

复杂程序有很多 kernel/launch,要用正则过滤,否则采集慢、输出乱:

```bash
ncu --kernel-name regex:reduce ./program <args>     # 只测名字含 reduce 的 kernel
ncu --launch-count 1 ./program                       # 只测第一个 launch
ncu --launch-skip 5 --launch-count 1 ./program       # 跳过前 5 个,测第 6 个
```

### 3.3 控制采集多少(set / section)

```bash
ncu --set basic   --kernel-name regex:reduce ./program    # 基础集(快)
ncu --set full    --kernel-name regex:reduce ./program    # 完整集(慢但全)
ncu --section SpeedOfLight --kernel-name regex:reduce ./program   # 只采一个 section
```

### 3.4 存报告(之后用 ncu-ui 看)

```bash
ncu --set full -o reports/reduce \
    --kernel-name regex:reduce ./program <args>      # 生成 reports/reduce.ncu-rep
ncu-ui reports/reduce.ncu-rep                         # 图形界面打开
```

### 3.5 编译配合:加 -lineinfo

要让 ncu 把指标关联到**源码行**,编译时加 `-lineinfo`(几乎不影响性能,卷二/06):

```bash
nvcc -lineinfo -arch=sm_75 kernel.cu -o kernel
```

> 不要用 `-G`(device debug)编译再测性能——它关优化,数字没参考价值(卷二/06、卷五/01)。

---

## 4. Section 与 Set:采集什么

ncu 把指标分成 **section**(一组相关指标),多个 section 组成 **set**。

```bash
ncu --list-sets        # 列出可用集合(basic/full/...)
ncu --list-sections    # 列出所有 section
```

常用 section(按判读顺序):

| Section | 看什么 | 何时重点看 |
|---|---|---|
| **SpeedOfLight** | memory% / compute% 谁接近上限 | **永远第一个看**(分流)|
| **MemoryWorkloadAnalysis** | DRAM/L2/L1、访存效率、sectors | memory-bound 时 |
| **Occupancy** | 理论/实际 occupancy、Block Limit | occupancy 可疑时 |
| **SchedulerStatistics** | warp 发射、eligible warp | latency-bound 时 |
| **WarpStateStatistics** | stall reason 占比 | latency-bound 时 |
| **ComputeWorkloadAnalysis** | 各执行管线利用率 | compute-bound 时 |
| **SourceCounters** | 指标关联到源码行 | 定位具体行 |

```text
set 选择:
  --set basic  快,够日常判 bound 类型
  --set full   全,深入分析或存报告用(慢,会重放 kernel 多次)
```

---

## 5. 判读总流程(背下来)

拿到一个慢 kernel,**永远按这个顺序**,不要一上来盯某个数字:

```text
1. Duration:先看 kernel 实际耗时(确认这个 kernel 值得优化)
2. Speed of Light:memory% vs compute% → 分流(第 6 节)
   ├─ memory 高 → memory-bound → 看 Memory Workload(第 7 节)
   ├─ compute 高 → compute-bound → 看 Compute Workload(第 10 节)
   └─ 两个都低 → latency-bound → 看 Warp State + Occupancy(第 8、9 节)
3. 按 bound 类型深挖对应 section,找根因
4. Source Counters:定位到具体源码行
5. 形成假设 → 改 → 复测:那个指标按预期变了吗?(卷五/06 因果验证)
```

> 核心纪律:**先分流(SoL),再对症**。不同 bound 看不同 section,别用错地图。

---

## 6. Speed of Light:第一刀分流

SoL 是最重要的 section,它把 kernel 的内存和计算吞吐**和硬件峰值比**,告诉你被谁限制:

```text
SoL Memory  [%]:  内存吞吐占峰值的百分比
SoL Compute [%]:  计算吞吐占峰值的百分比

判读:
  Memory 85%, Compute 30%   → memory-bound(被带宽限制)
  Compute 85%, Memory 30%   → compute-bound(被算力限制)
  Memory 30%, Compute 30%   → latency-bound(都没用满,在空等)← 最易误判
```

**两个都低**是最常见也最关键的情况:**不是"还很空闲可以加活",而是延迟没被隐藏住、执行单元
在干等**。这时去堆计算只会更糟,要去 Warp State 找它在等什么(第 9 节)。

> Roofline 视角(卷五/02):SoL 就是 Roofline 在告诉你 kernel 落在拐点哪侧。Memory 高 = 在带宽
> 屋顶下;Compute 高 = 在算力屋顶下。

---

## 7. Memory Workload:memory-bound 看这里

确认 memory-bound 后,这个 section 告诉你**访存哪里浪费了**:

| 指标 | 含义 | 不好时怎么办 |
|---|---|---|
| **DRAM Throughput [%]** | 显存带宽用了多少 | 已接近峰值 → 到墙了,减少数据搬运/融合 |
| **L2 Hit Rate** | L2 命中率 | 高 → 数据多在 cache,不那么吃 DRAM |
| **gld_efficiency**(global load 效率)| 读的有效率 | 低 → 读不合并,改访问模式(卷三/02)|
| **gst_efficiency**(global store 效率)| 写的有效率 | 低 → 写不合并(如 transpose naive)|
| **Sectors/Req** | 每次请求触发几个 32B sector | 偏高 → 不合并/不对齐,合并或向量化 |
| **Shared bank conflicts** | shared 冲突次数 | 非 0 → padding 消冲突(卷三/03)|

典型判读:

```text
transpose naive:gst_efficiency 很低(如 12%)→ 写跨步不合并(卷五/06 的经典假设)
                → shared tile 把跨步写换成合并写 → 复测 gst_efficiency 接近 100%

DRAM throughput 90%+ → 带宽打满了 → 这就是物理上限,别再纠结访存模式,
                       想办法减少总搬运量(数据复用/kernel 融合)
```

---

## 8. Occupancy section

详细机制见 [Occupancy详解](Occupancy详解_从入门到调优.md),这里只讲 ncu 里怎么看:

```text
Theoretical Occupancy:  理论上限(资源算出)
Achieved Occupancy:     实际达到(运行时平均)

Block Limit Registers:  寄存器限制的 block 数
Block Limit Shared Mem: shared 限制的 block 数
Block Limit Warps:      warp 数限制的 block 数
Block Limit SM:         SM block 上限
  → 这四个里最小的,就是 occupancy 的瓶颈资源!
```

判读:

```text
理论低 → 资源限制 → 看四个 Block Limit 哪个最小 → 调那个资源
理论高但实际低 → 尾部效应/grid 太小/负载不均 → 增大 grid、grid-stride、均衡负载
occupancy 低但已 memory-bound → 别管它(提 occupancy 不增带宽)
```

---

## 9. Warp State / Scheduler:latency-bound 看这里

SoL 两个都低(latency-bound)时,来这里找 **warp 为什么发不出指令(stall)**:

| 主要 stall reason | 含义(硬件根源见卷九/02)| 怎么办 |
|---|---|---|
| **Stall Long Scoreboard** | 等 global memory 返回 | 提并发隐藏延迟、减依赖链、shared 复用 |
| **Stall Barrier** | 卡在 `__syncthreads()` | 减同步、均衡负载、warp shuffle 收尾 |
| **Stall MIO Throttle** | shared/特殊单元排队 | 减 shared 访问、消 bank conflict |
| **Stall Short Scoreboard** | 等 shared / 较快依赖 | 减 shared 依赖 |
| **Stall Not Selected** | 有就绪 warp 但没轮到 | 通常是好现象(并发足),别动 |
| **Stall Wait** | 等固定延迟指令 | 增加 ILP |

Scheduler Statistics 补充:

```text
Eligible Warps Per Scheduler:每周期有几个 warp 可发射
  → 很低(<1)说明 scheduler 经常没 warp 可发 → occupancy/ILP 不足
Issued Warp Per Scheduler:实际发射率
```

> 纪律:**看主要 stall 占比,但别看到一个高就立刻改**。它可能是真瓶颈,也可能是别的瓶颈的
> 副作用。要和 SoL、occupancy 交叉验证。

---

## 10. Source / Compute Workload

### Source Counters(定位到行)

需要 `-lineinfo` 编译。它把指标(如哪行 stall 最多、哪行访存最多)对应到**源码行**,让你
精确知道改哪里。ncu-ui 里能看到源码和 SASS 并排、每行的指标热度。

### Compute Workload(compute-bound 时)

```text
各执行管线(pipe)的利用率:
  FMA pipe、ALU pipe、Tensor pipe、SFU 等
  → 看哪个 pipe 接近饱和,就是它限制了计算吞吐
例:大量 expf 的 kernel → SFU pipe 高(卷九/03);
   纯 FP32 FMA → FMA pipe 高
```

---

## 11. 关键指标速查表

```text
【分流】
SoL Memory % / Compute %        谁高被谁限制;都低=latency-bound

【访存】
DRAM Throughput %               显存带宽利用率(90%+=到墙)
gld_efficiency / gst_efficiency 读/写合并效率(低=不合并)
L2 Hit Rate                     L2 命中
Sectors/Req                     每请求 sector 数(高=浪费)
Shared bank conflicts           shared 冲突(非0=加padding)

【并发】
Achieved / Theoretical Occupancy 实际/理论占用率
Block Limit (4个)               哪个最小=occupancy 瓶颈
Eligible Warps Per Scheduler    可发射 warp 数(低=并发不足)

【停顿】
Stall Long Scoreboard           等显存(最常见)
Stall Barrier                   等同步
Stall MIO Throttle              shared/特殊单元排队

【计算】
各 pipe 利用率                  哪个 pipe 饱和=计算瓶颈
寄存器数 / spill                -Xptxas=-v 也能看
```

---

## 12. 指标 → 动作对照表

这是 ncu 判读的核心——**看到什么数,做什么事**:

```text
SoL Memory 高 + gst_efficiency 低     → 写不合并 → shared tile / 改布局(卷三)
SoL Memory 高 + DRAM 90%+             → 带宽到墙 → 减搬运 / kernel 融合(卷六)
SoL Memory 高 + bank conflict 非0     → shared 冲突 → padding(卷三/03)
SoL Compute 高 + FMA pipe 饱和        → 算力到墙 → 减指令 / Tensor Core
SoL 都低 + Long Scoreboard 高         → 延迟没藏住 → 提 occupancy / ILP(卷五/03)
SoL 都低 + Barrier 高                 → 同步太多 → 减 __syncthreads / 均衡负载
理论 occ 低 + Block Limit Reg 最小    → 寄存器限制 → __launch_bounds__(防 spill)
理论 occ 高但实际低                   → 尾部效应 → 增大 grid / grid-stride
Eligible Warps 很低                   → 并发不足 → 提 occupancy 或 ILP
```

---

## 13. 完整实战案例

**目标:用 ncu 优化一个 transpose kernel。** 走完整闭环:

```text
第0步 编译:nvcc -lineinfo -arch=sm_75 transpose.cu -o transpose

第1步 跑 SoL:
  ncu --section SpeedOfLight --kernel-name regex:transpose ./transpose 4096 4096
  结果:Memory 80%, Compute 5% → memory-bound

第2步 看 Memory Workload:
  ncu --section MemoryWorkloadAnalysis --kernel-name regex:transpose ./transpose 4096 4096
  结果:gld_efficiency 100%(读合并),gst_efficiency 12%(写不合并!)
  → 假设:瓶颈是 global store 跨步不合并

第3步 优化:上 shared tile,让读写两端都合并(卷五/06)

第4步 复测:
  gst_efficiency 升到 ~100%,时间下降 → 假设验证 ✓

第5步 但发现新问题:看 Memory Workload,shared bank conflicts 非 0
  → tile 是 [32][32],列访问 32 路冲突(卷三/03)

第6步 再优化:tile 改 [32][33] padding

第7步 复测:bank conflicts 降到 ~0,时间再降 → 完成

第8步 记录(卷十/04):每步的指标 + 时间 + "为什么变快"的证据链
```

> 这就是 ncu 的标准用法:**SoL 分流 → 对应 section 找根因 → 改 → 复测验证指标变化**。每步都有
> 数据背书,这正是面试讲优化最有说服力的方式。

---

## 14. ncu-ui 图形界面

命令行存的 `.ncu-rep` 用 `ncu-ui` 打开,图形界面更直观:

```bash
ncu-ui reports/transpose.ncu-rep
```

界面要点:

```text
- Details 页:所有 section 的指标,带颜色标注(红=瓶颈)
- Source 页:源码 + SASS 并排,每行的指标热度(需 -lineinfo)
- 多个 kernel/launch 可下拉切换
- 可以 baseline 对比:把优化前后两个报告对比,直接看指标差异
```

> baseline 对比功能特别有用:优化前存一份、优化后存一份,ncu-ui 并排显示,一眼看出哪个指标
> 改善了——天然适合写优化报告(卷十/04)。

---

## 15. 常见坑

```text
✗ 不过滤 kernel 直接 ncu ./program → 程序有几百个 launch,采集极慢
  → 用 --kernel-name regex: 或 --launch-count 过滤

✗ 用 ncu 的时间当 benchmark → ncu 会重放 kernel、插桩,有 overhead,时间不准
  → 性能数字另用 Event 测;ncu 只看指标(卷五/06)

✗ 没加 -lineinfo → Source 页看不到源码关联
  → 编译加 -lineinfo

✗ 用 -G 编译再分析 → 关了优化,指标不代表 release
  → 用 -lineinfo,别用 -G

✗ 只看一个指标就下结论 → stall reason 高可能是别的瓶颈的副作用
  → 按总流程(SoL→对应 section)交叉验证

✗ 输入规模太小 → kernel 跑几微秒,指标噪声大、不代表真实
  → 用足够大的输入(如 4096² / 16M 元素)

✗ memory-bound 还去提 occupancy → 带宽已满,提 occupancy 没用
  → 先看 SoL 判 bound,对症下药
```

---

## 16. 面试题

**Q1:Nsight Compute 和 Nsight Systems 区别?**
ncu 看单个 kernel 的微架构指标(访存/计算/occupancy/stall),回答"kernel 为什么慢";nsys 看
系统级时间线(传输/计算重叠、GPU 空洞、同步),回答"时间花在哪个阶段"。先 nsys 定位、再 ncu
深挖。

**Q2:怎么用 ncu 判断 memory-bound 还是 compute-bound?**
看 Speed of Light section 的 Memory% 和 Compute%,哪个接近 100% 就被哪个限制。两个都低则是
latency-bound,要去 Warp State 看 stall。

**Q3:transpose 慢,你用 ncu 怎么排查?**
先 SoL 确认 memory-bound;再看 Memory Workload 的 gst_efficiency,很低说明写不合并;上 shared
tile 改成合并写后复测,该指标应升到接近 100%;再检查 bank conflicts,非 0 就加 padding。

**Q4:ncu 报 ERR_NVGPUCTRPERM 怎么办?**
是性能计数器权限问题,不是代码错误。需管理员设
`NVreg_RestrictProfilingToAdminUsers=0` 或用 sudo。无权限时用 Event 计时 + Roofline 推断 bound
类型代替。

**Q5:为什么不能用 ncu 的时间当 benchmark?**
ncu 为采集指标会插桩、重放 kernel、串行化,有显著 overhead,时间不准。要分开:Event 测时间,
ncu 只看指标。

**Q6:occupancy 低,ncu 里怎么定位是哪个资源限制的?**
看 Occupancy section 的四个 Block Limit(Registers/SharedMem/Warps/SM),最小的那个就是瓶颈
资源。寄存器最常见,配合 `-Xptxas=-v` 看每线程寄存器数。

---

## 配套阅读

- [Occupancy详解_从入门到调优.md](Occupancy详解_从入门到调优.md) —— occupancy section 的深入
- 课程 [卷五/05 Nsight Compute 与 Compute Sanitizer](../cuda_deep_course/course/volume05_performance/05_Nsight_Compute与Compute_Sanitizer.md)
- 课程 [卷五/04 Nsight Systems 系统时间线](../cuda_deep_course/course/volume05_performance/04_Nsight_Systems系统时间线.md)
- 课程 [卷五/06 完整优化案例](../cuda_deep_course/course/volume05_performance/06_完整优化案例_复习与面试.md)
- 课程 [卷九/02 Warp 调度与延迟隐藏](../cuda_deep_course/course/volume09_hardware_architecture/02_Warp调度_scheduler_scoreboard与延迟隐藏.md) —— stall 的硬件根源
- 官方:Nsight Compute CLI / Profiling Guide、Kernel Profiling Guide

---

**返回**:[study_plan/Week5 性能工程](../study_plan/Week5_性能工程与Nsight实战.md) · [CUDA学习路线图](CUDA学习路线图.md)
