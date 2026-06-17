# 05 Nsight Compute 与 Compute Sanitizer

## 1. Nsight Compute 回答什么

它分析单个 CUDA kernel：

- 内存吞吐和事务。
- SM/计算吞吐。
- Occupancy。
- Warp stall。
- 指令和 source correlation。
- Shared-memory 行为。

## 2. 最小使用

```bash
ncu ./program
```

默认收集基础 section。复杂程序应过滤 kernel，避免 profile 太多 launch。

```bash
ncu --kernel-name regex:transposeShared \
  ./labs/03_memory_system/transpose/transpose 4096 4096
```

## 3. Sets 与 Sections

查看可用集合：

```bash
ncu --list-sets
ncu --list-sections
```

常用：

```bash
ncu --set full ...
ncu --section SpeedOfLight ...
ncu --section MemoryWorkloadAnalysis ...
ncu --section Occupancy ...
ncu --section SourceCounters ...
```

名称和可用内容随版本变化，以 `--list-sections` 为准。

## 4. 保存报告

```bash
ncu --set full -o reports/transpose_shared \
  --kernel-name regex:transposeShared \
  ./labs/03_memory_system/transpose/transpose 4096 4096
```

使用 `ncu-ui` 打开 `.ncu-rep`。

源码关联需要编译：

```bash
nvcc -lineinfo ...
```

实验 Makefile 已为 transpose/reduction 加入 `-lineinfo`。

## 4.1 Performance Counter 权限

服务器上可能出现：

```text
ERR_NVGPUCTRPERM
The user does not have permission to access NVIDIA GPU Performance Counters
```

这表示 kernel 仍能运行，但当前用户无权采集硬件计数器，不是 CUDA 计算错误。

处理方式由机器管理员和安全策略决定，通常需要管理员按照 NVIDIA 的
performance-counter 权限说明开放 profiling。不要为了绕过限制擅自修改生产
服务器配置。

## 5. 阅读顺序

1. Duration 和 launch 配置。
2. Speed of Light：memory 还是 compute 更接近上限。
3. Memory Workload：DRAM/L2/L1、sectors、访问效率。
4. Occupancy：理论与达到值。
5. Warp State：主要 stall。
6. Source Counters：定位源代码行。

不要只凭一个指标下结论。

## 5.1 从指标到决策：看到这个数，该怎么办

光知道指标名字没用，关键是**看到一个数能推出下一步动作**。下面是最常用的判读表，
括号里是 Nsight Compute 里的大致字段名（随版本略有差异，以 `--list-sections` 为准）。

### 第一刀：先用 Speed of Light 分流

```text
SoL: Memory % 和 Compute % 哪个高？
  ├─ Memory 高（如 85%）、Compute 低     -> memory-bound -> 去看 Memory Workload（下方）
  ├─ Compute 高、Memory 低               -> compute-bound -> 看指令组合 / 是否能用更快指令
  └─ 两个都低（如都 30%）               -> latency-bound -> 看 Warp State 的 stall（下方）
```

两个都低是最常见也最容易误判的情况：**不是"还很空闲可以加活"，而是 warp 不够多/延迟
没被隐藏**，GPU 在干等。这时去堆计算只会更糟，应该去 Warp State 找它在等什么。

### memory-bound：看 Memory Workload

| 看到的指标 | 含义 | 该做什么 |
|---|---|---|
| `gld_efficiency` / `gst_efficiency` 低（如 12%）| 访存不合并，一个 warp 的请求散到很多 sector | 改访问模式：合并访问、SoA、shared tile 重排（见卷三）|
| DRAM throughput 已接近峰值（如 90%+）| 真的打满带宽了 | **到墙了**，再优化访存模式收益有限；考虑减少数据搬运总量 / kernel fusion |
| L2 hit rate 高、DRAM 不高 | 数据大多命中 cache | 当前 kernel 其实不那么吃 DRAM，别盯着带宽 |
| sectors per request 偏高 | 每次请求触发过多 32B sector | 对齐、合并、向量化加载（`float4`）|

### latency-bound：看 Warp State（主要 stall reason）

| 主要 stall | 含义 | 该做什么 |
|---|---|---|
| `Stall Long Scoreboard` | 在等 global memory 返回 | 提高并发隐藏延迟（更多 warp / ILP）、减少依赖链、用 shared 复用 |
| `Stall Barrier` | 卡在 `__syncthreads()` | 减少同步、让各 warp 负载均衡、warp shuffle 收尾（见卷四 reduction v4）|
| `Stall MIO Throttle` | shared/特殊功能单元排队 | 减少 shared 访问、消除 bank conflict |
| `Stall Not Selected` | 有合格 warp 但没轮到它 | 通常是好现象（并发充足），别乱动 |

> 关键纪律：**先看主要 stall 占比，但别看到一个高就立刻改代码。** 它可能是真瓶颈，
> 也可能是另一个瓶颈的副作用。要和 SoL、occupancy 交叉验证（第 5 节最后一句）。

### occupancy：理论 vs 实际

```text
理论 occupancy 低           -> 被寄存器/shared/block 大小限制 -> -Xptxas=-v 看是谁
理论高但实际(achieved)低    -> 负载不均、尾部效应、block 数不足 -> 调 grid/block、grid-stride
理论已经不低               -> 别再盲目提 occupancy，去看带宽/stall 才是墙
```

### 一个完整判读示例（transpose naive）

```text
SoL:           Memory 80%, Compute 5%      -> memory-bound
Memory:        gst_efficiency 12%          -> 写不合并（正是卷五 06 的假设）
决策:          上 shared tile 把跨步写换成合并写
改完复测:      gst_efficiency ~100%, 时间下降 -> 假设被验证 ✅
```

这就是"指标 → 假设 → 动作 → 复测验证"的闭环，也是面试时讲优化最有说服力的讲法。

## 6. Compute Sanitizer

### Memcheck

```bash
compute-sanitizer --tool memcheck ./program
```

检查越界、misaligned 和非法访问。

### Racecheck

```bash
compute-sanitizer --tool racecheck ./program
```

重点检查 shared-memory hazards。

### Initcheck

```bash
compute-sanitizer --tool initcheck ./program
```

检查未初始化 device global memory 读取。

### Synccheck

```bash
compute-sanitizer --tool synccheck ./program
```

检查 barrier 和 warp 同步使用问题。

## 7. 正确顺序

```text
CPU reference
-> 普通测试
-> memcheck/racecheck/synccheck
-> benchmark
-> profiler
-> 优化
```

不要 profile 已经越界或 race 的 kernel。

## 8. 当前实验命令

```bash
compute-sanitizer --tool memcheck \
  ./labs/03_memory_system/transpose/transpose 1003 769

compute-sanitizer --tool memcheck \
  ./labs/04_parallel_algorithms/reduction/reduction 1000003

ncu --set full --kernel-name regex:reduceWarpShuffle \
  ./labs/04_parallel_algorithms/reduction/reduction 16777216
```

## 9. 实践报告

每个优化结论附：

- Before/after 时间。
- 关键 profiler 指标。
- Source 行。
- 正确性和 sanitizer。
- 为什么指标变化支持假设。

## 10. 资料映射

- Nsight Compute 13.3 CLI 与 Profiling Guide。
- Compute Sanitizer Documentation。
