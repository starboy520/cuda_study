# 卷七：异步执行与 CUDA 系统能力

> 前六卷把"单个 kernel 写对、写快"讲透了。卷七把视野从**单 kernel** 拉到
> **端到端流水线**：当 GPU 算得很快时，瓶颈往往跑到了"数据传输"和"CPU 发指令"上。
> 这一卷教你把 PCIe、SM、CPU 三者的空闲时间互相填满。

## 学习目标

- 理解 CUDA 的异步语义：launch / `cudaMemcpyAsync` 何时返回、何时真正执行。
- 用 stream 表达并发，用 event 表达依赖，区分"同一 stream 顺序"与"跨 stream 并发"。
- 用 pinned memory + 多 stream 分块，**重叠传输与计算**，并能实测加速。
- 让多个小 kernel 并发执行，填满未占满的 GPU。
- 用 CUDA Graph **录制一次、重放多次**，消除重复 launch 开销。
- 用 Nsight Systems 时间线**验证**重叠是否真的发生。

## 为什么需要这一卷（一张图）

```text
端到端时间 = H2D 传输 + kernel 计算 + D2H 传输 + CPU 调度开销
                ↑ PCIe          ↑ SM         ↑ PCIe      ↑ CPU launch

朴素串行：每一段忙时，其它三者都在空等 → 利用率低
卷七目标：用 stream/event/graph 让它们重叠起来 → 填满空闲
```

## 章节

1. [Stream 与异步执行模型](01_Stream与异步执行模型.md) —— 异步语义、stream、默认 stream 的坑
2. [传输与计算重叠](02_传输与计算重叠.md) —— pinned + 多流分块流水线（核心，配实测 demo）
3. [Event、并发 Kernel 与依赖](03_Event_并发Kernel与依赖.md) —— event 计时/依赖、并发 kernel、stream 优先级
4. [CUDA Graph](04_CUDA_Graph.md) —— capture / instantiate / replay，消除 launch 开销（配实测 demo）

> 本卷聚焦"最常用、收益最大"的核心能力。`cudaMallocAsync`（stream-ordered
> allocator）、Unified Memory prefetch/advice、device-side `cp.async`/TMA/cluster
> 等进阶主题，在卷三/卷九和官方文档已有铺垫，工程中按需深入。

## 配套实验

```bash
cd /home/qichengjie/workspace/cuda_study/cuda_deep_course

# 传输与计算重叠：串行 vs 多流重叠 vs pageable 反例
make -C labs/07_async_system/overlap_pipeline clean all
./labs/07_async_system/overlap_pipeline/overlap_pipeline

# CUDA Graph：逐个 launch vs 图重放
make -C labs/07_async_system/cuda_graph clean all
./labs/07_async_system/cuda_graph/cuda_graph
```

参考实测（T4，仅供量级参考，每台机器会不同）：

```text
overlap_pipeline (64M 元素, 16 块, 4 流)：
  串行 (pinned)            99.43 ms
  重叠 (pinned, 4 streams) 34.77 ms   加速 2.86x
  分块 (pageable)         100.20 ms   加速 0.99x  ← 无 pinned 不重叠

cuda_graph (iters=1000, chain=50)：
  逐个 launch  232.60 ms
  Graph 重放   124.55 ms   加速 1.87x
```

## 用 Nsight Systems 看时间线（验证重叠）

```bash
nsys profile --trace=cuda -o overlap_report \
  ./labs/07_async_system/overlap_pipeline/overlap_pipeline
```

在时间线上确认：不同 stream 的 H2D / kernel / D2H 是否**在时间上重叠**。
串行版应看到三段首尾相接；重叠版应看到拷贝条和 kernel 条彼此交叠。

## 完成标准

- [ ] 能解释为什么 `cudaMemcpyAsync` 在 pageable 内存上会退化成同步。
- [ ] 能写出 pinned + 多 stream 分块流水线，并实测出重叠加速。
- [ ] 能用 event 让一个 stream 等待另一个 stream 的某个节点。
- [ ] 能说明"同一 stream 内顺序、跨 stream 间并发"的规则。
- [ ] 能用 stream capture 录制并重放一张 CUDA Graph。
- [ ] 能用 Nsight Systems 时间线判断重叠是否真的发生。

## 资料映射

- CUDA C++ Programming Guide：Asynchronous Concurrent Execution、Streams、Events、CUDA Graphs。
- CUDA C++ Best Practices Guide：Asynchronous Transfers and Overlapping。
- Nsight Systems User Guide：Timeline、CUDA trace。
