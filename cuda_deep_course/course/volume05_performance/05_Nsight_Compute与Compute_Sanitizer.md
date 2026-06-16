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
