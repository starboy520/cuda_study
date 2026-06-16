# 卷一：CUDA 入门所需的 GPU 基础

这一卷只讲开始写 CUDA 前必须知道的硬件直觉。更深入的 warp scheduler、
执行管线、缓存细节和架构代际演进放在卷九。

## 学习目标

完成本卷后，你应该能够：

1. 解释 CPU 和 GPU 的设计目标为什么不同。
2. 区分 GPU、SM、Block、Warp、Thread。
3. 说清哪些概念是硬件，哪些是 CUDA 编程模型中的逻辑组织。
4. 建立 Register、Shared Memory、Global Memory 的第一层认识。
5. 解释 GPU 为什么需要大量线程隐藏延迟。
6. 读取自己 GPU 的关键设备属性。

## 章节

1. [CPU 与 GPU 为什么不同](01_CPU与GPU为什么不同.md)
2. [GPU、SM、Warp、Thread](02_GPU_SM_Warp_Thread.md)
3. [一维、二维、三维 Block 与线性编号](02A_一维二维三维Block与线性编号.md)
4. [内存层次第一印象](03_内存层次第一印象.md)
5. [延迟隐藏与大量线程](04_延迟隐藏与大量线程.md)
6. [T4 设备观察实验](05_T4设备观察实验.md)
7. [卷一复习与面试题](06_卷一复习与面试题.md)

## 配套实验

```bash
cd /home/qichengjie/workspace/cuda_study/cuda_deep_course
make -C labs/01_gpu_basics/device_query clean all
./labs/01_gpu_basics/device_query/device_query
```

## 完成标准

- [ ] 能从记忆画出 `GPU -> SM -> Warp -> Thread` 的硬件观察图。
- [ ] 能说明 `Grid/Block` 为什么不是 GPU 上固定存在的物理盒子。
- [ ] 能把一维、二维和三维 `threadIdx` 转成 block 内线性编号。
- [ ] 能区分 block 内线性 thread ID、全局 thread 位置和数据下标。
- [ ] 能解释 register、shared、global 的共享范围。
- [ ] 能解释“隐藏延迟”而不是只背“GPU 线程多”。
- [ ] 运行设备观察实验并保存自己的输出。
- [ ] 完成卷末问题和口述练习。
