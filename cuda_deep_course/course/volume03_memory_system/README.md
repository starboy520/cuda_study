# 卷三：CUDA 内存系统

## 学习目标

完成本卷后，你应能：

- 判断数据位于哪种 CUDA 内存空间，以及由谁共享。
- 按 warp 画出地址并判断 global memory 是否合并访问。
- 解释对齐、跨步访问、AoS/SoA 和多维布局。
- 正确使用 shared memory、同步、tile 和 padding。
- 区分 cache、pinned memory、mapped memory 与 Unified Memory 的作用。
- 独立实现并分析非方阵矩阵转置。

## 章节

1. [CUDA 内存空间](01_CUDA内存空间.md)
2. [合并访问、对齐、AoS 与 SoA](02_合并访问_对齐_AoS与SoA.md)
3. [Shared Memory、Tile 与 Bank Conflict](03_Shared_Memory_Tile与Bank_Conflict.md)
4. [Cache、Host 传输与 Unified Memory](04_Cache_Host传输与Unified_Memory.md)
5. [矩阵转置完整实验](05_矩阵转置完整实验.md)
6. [卷三复习与面试题](06_卷三复习与面试题.md)

## 配套实验

```bash
cd /home/qichengjie/workspace/cuda_study/cuda_deep_course

make -C labs/03_memory_system/memory_access clean all
./labs/03_memory_system/memory_access/memory_access

make -C labs/03_memory_system/transpose clean all
./labs/03_memory_system/transpose/transpose
./labs/03_memory_system/transpose/transpose 1024 768
```

## 完成标准

- [ ] 能对一个 warp 的地址列表计算需要覆盖多少 32-byte 区间。
- [ ] 能解释为什么连续访问通常优于 stride 访问。
- [ ] 能判断何时 shared memory 有真实价值。
- [ ] 能画出 bank conflict 和 padding 的关系。
- [ ] 能独立写出支持非方阵、非整除尺寸的 shared transpose。
- [ ] 使用 Nsight Compute 对比至少两个内存访问版本。

