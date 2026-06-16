# 卷五：性能工程与 Profiling

## 学习目标

- 使用 APOD 建立优化闭环。
- 写出可重复、可信的 GPU benchmark。
- 计算 latency、throughput、有效带宽、GFLOPS 和算术强度。
- 使用 Amdahl、Gustafson 和 Roofline 判断优化方向。
- 正确理解 occupancy、寄存器压力、分歧和 stall。
- 使用 Nsight Systems、Nsight Compute 和 Compute Sanitizer。
- 从证据提出单一优化假设并验证。

## 章节

1. [APOD 与可靠 Benchmark](01_APOD与可靠Benchmark.md)
2. [性能指标、Scaling 与 Roofline](02_性能指标_Scaling与Roofline.md)
3. [Occupancy、分歧与延迟隐藏](03_Occupancy_分歧与延迟隐藏.md)
4. [Nsight Systems 系统时间线](04_Nsight_Systems系统时间线.md)
5. [Nsight Compute 与 Compute Sanitizer](05_Nsight_Compute与Compute_Sanitizer.md)
6. [完整优化案例、复习与面试](06_完整优化案例_复习与面试.md)

## 实践对象

卷五复用前两卷实验：

```text
memory_access
transpose
reduction
vector_add
```

这样可以把精力放在测量和推理，而不是重新写无关代码。

## 完成标准

- [ ] 能解释一次 benchmark 的同步边界。
- [ ] 能为一个 kernel 算有效带宽或 GFLOPS。
- [ ] 能计算算术强度并画 Roofline 落点。
- [ ] 能从 Nsight Systems 判断端到端瓶颈。
- [ ] 能从 Nsight Compute 提出具体而非模糊的优化假设。
- [ ] 能区分 occupancy 问题与 memory/instruction 问题。
- [ ] 完成 transpose 或 reduction 优化报告。

