# 01 APOD 与可靠 Benchmark

性能优化第一课不是“怎么优化”，而是：

> 怎么避免优化错地方。

很多初学者会这样做：

```text
看到一个 kernel
觉得它慢
改 block size
加 shared memory
再跑一下
```

这叫猜测式优化。

真正的性能工程要走闭环：

```text
Assess -> Parallelize -> Optimize -> Deploy
```

简称 APOD。

## 1. APOD 是什么

```text
Assess:
  评估。先找到瓶颈在哪里，确认值不值得优化。

Parallelize:
  并行化。先写出正确的并行实现。

Optimize:
  优化。基于测量和证据做修改。

Deploy:
  部署。把优化放进真实流程，并持续验证没有退化。
```

它是循环，不是一次性流程：

```text
Assess -> Parallelize -> Optimize -> Deploy
   ^                                      |
   |______________________________________|
```

每次优化后都要重新测量，因为瓶颈会转移。

## 2. 为什么 Assess 必须在最前

假设完整应用一次请求是：

```text
总时间 = 100 ms
```

其中：

```text
数据传输 + CPU 预处理 = 80 ms
kernel A = 15 ms
kernel B = 5 ms
```

如果你把 kernel B 优化到 0：

```text
总时间 100 ms -> 95 ms
只快 5%
```

如果你把数据传输和 CPU 预处理减半：

```text
总时间 100 ms -> 60 ms
快 40%
```

所以：

```text
不先 Assess，就可能把力气花在不重要的地方。
```

这就是 Amdahl 定律在真实优化中的样子。

## 3. Assess 阶段要问什么

先回答这些问题：

```text
1. 端到端时间花在哪里？
2. 是 CPU 慢、数据传输慢，还是 GPU kernel 慢？
3. 哪个 kernel 占比最高？
4. 输入规模是否足够大？
5. 优化目标是 latency 还是 throughput？
6. 正确性和精度要求是什么？
7. baseline 是什么？
```

没有 profile 就直接优化最显眼的 kernel，通常不靠谱。

工具选择：

```text
端到端看时间线：
  Nsight Systems

单个 kernel 内部：
  Nsight Compute

正确性和越界：
  Compute Sanitizer
```

## 4. Parallelize：先正确，再谈快

Parallelize 阶段不是“立刻追求最快”。

它的目标是：

```text
把问题正确地搬到 GPU 上。
```

先做到：

```text
结果正确
边界正确
非整除规模正确
大输入正确
小输入正确
没有越界
没有 race
```

再谈性能。

一个错误的快速 kernel 没有意义。

所以性能优化前先跑：

```bash
compute-sanitizer --tool memcheck ./your_program
compute-sanitizer --tool racecheck ./your_program
```

## 5. Optimize：一次只改一个主要因素

Optimize 阶段应该像实验：

```text
提出假设
修改一个因素
复测
判断假设是否成立
```

错误做法：

```text
同时改 block size、shared memory、unroll、数据布局。
```

这样就算变快，你也不知道是哪一项起作用。

正确做法：

```text
假设：naive transpose 慢是因为写不合并。
修改：使用 shared tile，把写变成合并。
复测：GB/s 是否上升？ncu 里 store efficiency 是否改善？
结论：假设成立或不成立。
```

性能优化最重要的是：

```text
每一步都有证据。
```

## 6. Deploy：别让优化只存在于一次实验

Deploy 阶段要关心：

```text
优化是否在真实输入下仍然有效？
是否影响正确性？
是否影响其他 GPU 架构？
是否需要回归 benchmark？
是否有 fallback？
```

例如：

```text
某个 block size 在 T4 上最快，
但在 A100/H100 上未必最快。
```

所以部署时要记录：

```text
GPU 型号
CUDA 版本
输入规模
数据类型
benchmark 方法
性能阈值
```

## 7. Benchmark 到底测什么范围

你必须明确测量范围。

常见范围：

```text
kernel-only:
  只测 kernel 执行时间。

H2D + kernel + D2H:
  包含 host/device 数据传输。

end-to-end:
  包含完整请求、CPU 预处理、GPU、后处理。

steady-state throughput:
  预热后稳定吞吐。

first-run latency:
  第一次请求延迟，包含初始化等。
```

这些不是谁对谁错，而是回答的问题不同。

例子：

```text
研究 kernel 优化：
  用 kernel-only。

研究推理服务延迟：
  用 end-to-end latency。

研究 batch 训练吞吐：
  用 steady-state throughput。

研究冷启动：
  用 first-run latency。
```

报告里必须写清楚。

## 8. Warmup：为什么要预热

第一次 CUDA 工作可能包含：

```text
CUDA context 初始化
module 加载
JIT
内存首次触碰
cache 冷启动
GPU 频率还没升上来
```

所以稳定 benchmark 通常：

```text
先 warmup 若干次
再正式记录多次 timing
```

但是：

```text
first-run latency 本身也可能是业务指标。
```

所以：

```text
warmup 后性能
first-run latency
```

应该分开报告。

## 9. CUDA Event：测 GPU 时间

CUDA Event 记录在 GPU stream 时间线上，适合测 GPU 工作。

模板：

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);

float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
```

注意：

```text
start 和 stop 要在同一个 stream 或明确的同步关系里。
event 区间是否包含 cudaMemcpyAsync 要说清。
不要在每个 kernel 后随手 cudaDeviceSynchronize，除非它就是测量边界。
```

## 10. CPU Timer：测端到端时间

CPU timer 可以用 `std::chrono`。

但是 CUDA kernel 异步，所以结束前必须同步：

```cpp
auto start = clock::now();

kernel<<<grid, block>>>(...);
CUDA_CHECK(cudaDeviceSynchronize());

auto stop = clock::now();
```

如果没有同步：

```text
你测到的是 kernel launch 返回时间，
不是 GPU 真正执行时间。
```

CPU timer 适合：

```text
端到端延迟
包含 H2D/D2H 的流程
包含多个 kernel 的完整 pipeline
```

## 11. 重复次数和统计

只跑一次不可信。

建议记录：

```text
warmup 次数
正式 iteration 次数
min
median
mean
p90 / p99
标准差或波动范围
```

不同场景看不同指标：

```text
benchmark kernel 极限性能：
  常看 min 或 median。

线上服务延迟：
  必须看 p90 / p99。

稳定吞吐：
  看长时间平均和波动。
```

平均值可能被少数异常值影响，所以不要只报 mean。

## 12. 控制变量

每次实验要记录：

```text
GPU 型号
驱动版本
CUDA 版本
编译器和编译参数
输入规模
数据类型
block/grid 配置
warmup 次数
iteration 次数
是否有其他 GPU 进程
功耗/频率/温度状态
```

每次只改一个主要变量。

否则你无法解释性能变化。

## 13. 防止编译器消除工作

如果结果没人用，编译器可能优化掉某些工作。

所以 benchmark 要：

```text
写出结果
读回结果
验证结果
避免 dead code
```

GPU kernel 通常写 global memory，比较不容易被完全消除，但 host 侧 benchmark 和小函数测试仍然要注意。

## 14. 正确性先于性能

性能报告必须有正确性证明。

常见方式：

```text
和 CPU reference 对比
和 naive GPU 对比
检查误差容忍
检查边界输入
compute-sanitizer
```

例如 GEMM：

```text
不能要求浮点结果 bitwise 一样。
要用相对误差或绝对误差容忍。
```

如果没有正确性，性能数字没有意义。

## 15. 一个最小 Benchmark 报告长什么样

```text
实验名称：
  transpose naive vs shared

环境：
  GPU: Tesla T4
  CUDA:
  编译参数:

输入：
  width x height:
  dtype:

测量范围：
  kernel-only

计时方法：
  CUDA Event
  warmup:
  iterations:
  statistic:

正确性：
  CPU reference / naive reference:
  compute-sanitizer:

结果：
  naive time:
  shared time:
  speedup:
  effective GB/s:

结论：
  是否支持优化假设？
  下一步是什么？
```

## 16. 实践：从 transpose 开始

构建：

```bash
cd /home/qichengjie/workspace/cuda_study/cuda_deep_course
make -C labs/03_memory_system/transpose clean all
./labs/03_memory_system/transpose/transpose
```

为 transpose 建表：

```text
首次运行
warmup 后单次
50 次平均
50 次中位数
kernel-only
包含复制
```

解释这些数字为什么不同。

## 17. 本章检查

你应该能回答：

```text
1. APOD 四个阶段是什么？
2. 为什么不能没 profile 就优化？
3. kernel-only 和 end-to-end 有什么区别？
4. CUDA Event 为什么适合测 kernel？
5. CPU timer 测 CUDA 为什么要同步？
6. warmup 是为了解决什么？
7. 为什么只跑一次不可信？
8. benchmark 报告必须记录哪些环境信息？
9. 为什么正确性先于性能？
```

## 18. 资料映射

- CUDA C++ Best Practices Guide：APOD、Performance Metrics。
- CUDA Runtime API：Event Management。
- 配套：[卷五第 02 章性能指标](02_性能指标_Scaling与Roofline.md)、[卷五第 04 章 Nsight Systems](04_Nsight_Systems系统时间线.md)。
