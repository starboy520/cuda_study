# 01 APOD 与可靠 Benchmark

## 1. APOD

```text
Assess      评估热点和可加速性
Parallelize 并行化正确实现
Optimize    基于测量优化
Deploy      部署、回归和持续验证
```

这是循环，不是一次性瀑布流程。

## 2. Assess

先回答：

- 端到端时间花在哪里？
- 问题规模是否足够？
- 数据传输占多少？
- 优化目标是 latency 还是 throughput？
- 正确性和精度要求是什么？

没有 profile 就直接优化最显眼的 kernel，可能优化错地方。

为什么 Assess 要排在最前、且必须先 profile？因为加速的**上限被你没动的那部分
锁死**（Amdahl，卷五第 02 章）。举个具体数字：假设端到端 100 ms，其中某个
kernel 占 5 ms、数据传输和 CPU 预处理占 95 ms。

```text
把这个 5 ms 的 kernel 优化到 0（极限）：
  端到端 100 ms -> 95 ms，仅快 5%

把 95 ms 的传输/预处理减半：
  端到端 100 ms -> 52.5 ms，快近一倍
```

你可能花一周把 kernel 调快 10 倍，端到端却几乎没变——因为它从一开始就不是
瓶颈。这就是"没 profile 先优化最显眼 kernel"的典型翻车。Assess 的任务正是用
数据找出那个 95 ms 在哪，避免把力气花在 5% 上。

## 3. Benchmark 的测量范围

必须明确：

```text
kernel-only
H2D + kernel + D2H
完整应用请求
steady-state throughput
first-run latency
```

它们回答不同问题。

## 4. Warmup

首次 CUDA 工作可能包含：

- Context 初始化。
- Module 加载/JIT。
- 内存首次触碰。
- Cache 和频率状态变化。

因此稳定 benchmark 通常先运行若干 warmup。

首次延迟本身也可能是业务指标，不能简单丢弃，要单独报告。

## 5. CUDA Event

```cpp
cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&milliseconds, start, stop);
```

Event 记录在 stream 时间线上，适合 GPU 工作测量。

注意：

- Start/stop 应在正确 stream。
- Event 区间是否包含 memcpy 要明确。
- 不要在每个 kernel 后不必要地 `cudaDeviceSynchronize()`。

## 6. CPU Timer

测端到端时间可使用 `std::chrono`，但结束前必须有能保证 GPU 工作完成的同步：

```cpp
auto start = clock::now();
kernel<<<...>>>(...);
CUDA_CHECK(cudaDeviceSynchronize());
auto stop = clock::now();
```

否则只测到 launch 开销。

## 7. 重复和统计

推荐记录：

- Iteration 数。
- 中位数。
- 最小值。
- P90/P99（延迟服务场景）。
- 标准差或离散程度。

平均值可能被少量异常值影响。只有一次运行不能支持稳定结论。

## 8. 控制变量

记录：

```text
GPU 型号
驱动与 CUDA
编译参数
输入规模和分布
数据类型
block/grid
warmup 和 iteration
功耗/频率环境
是否同时有其他 GPU 进程
```

每次只改变一个主要因素，才能解释结果。

## 9. 防止编译器消除工作

结果应被读取或验证。纯 Host benchmark 中编译器可能消除无用计算；GPU kernel
也应写出可观察结果并进行正确性检查。

## 10. 实践

为 `transpose` 建立表：

```text
首次运行
warmup 后单次
50 次平均
50 次中位数
kernel-only
包含复制
```

解释这些数字为什么不同。

## 11. 资料映射

- CUDA C++ Best Practices Guide：APOD、Performance Metrics。
- CUDA Runtime API：Event Management。

