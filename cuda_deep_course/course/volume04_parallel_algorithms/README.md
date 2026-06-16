# 卷四：同步与经典并行算法

## 学习目标

- 识别 data race、错误 barrier 和错误同步范围。
- 区分 barrier、memory fence 和 atomic。
- 正确使用 active mask、vote 与 warp shuffle。
- 从串行算法推导 reduction、scan 和 histogram。
- 理解 convolution、stencil、SpMV 的数据划分与瓶颈。
- 处理浮点非结合性、误差和非确定性。

## 章节

1. [Race、同步与内存可见性](01_Race_同步与内存可见性.md)
2. [Atomic 与 Warp 级原语](02_Atomic与Warp级原语.md)
3. [Reduction：从错误到优化](03_Reduction从错误到优化.md)
4. [Scan 与 Histogram](04_Scan与Histogram.md)
5. [Convolution、Stencil 与 SpMV](05_Convolution_Stencil与SpMV.md)
6. [数值正确性、复习与面试](06_数值正确性_复习与面试.md)

## 配套实验

```bash
cd /home/qichengjie/workspace/cuda_study/cuda_deep_course
make -C labs/04_parallel_algorithms/reduction clean all
./labs/04_parallel_algorithms/reduction/reduction
./labs/04_parallel_algorithms/reduction/reduction 31
./labs/04_parallel_algorithms/reduction/reduction 1000003
```

## 完成标准

- [ ] 能解释 `__syncthreads`、fence 和 atomic 的不同职责。
- [ ] 能说明 divergent barrier 为什么危险。
- [ ] 能推导多阶段 reduction。
- [ ] 能解释 shuffle mask。
- [ ] 能手工完成小数组的 Blelloch scan。
- [ ] 能设计 histogram privatization。
- [ ] 能解释浮点归约为什么顺序变化会改变结果。

