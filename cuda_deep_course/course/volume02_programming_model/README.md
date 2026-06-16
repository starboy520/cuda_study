# 卷二：CUDA 编程模型

本卷学习如何把一个计算问题组织成 CUDA 程序。

## 章节

1. [第一个完整 CUDA 程序](01_第一个完整CUDA程序.md)
2. [Grid、Block、Thread 索引](02_Grid_Block_Thread索引.md)
3. [CUDA 函数修饰符与执行空间](03_CUDA函数修饰符与执行空间.md)
4. [内存分配、复制与资源生命周期](04_内存分配_复制与资源生命周期.md)
5. [异步执行、同步与错误模型](05_异步执行_同步与错误模型.md)
6. [NVCC、PTX 与编译流程](06_NVCC_PTX与编译流程.md)
7. [CUDA Event 与正确计时](07_CUDA_Event与正确计时.md)
8. [二维矩阵加法 Sample](08_二维矩阵加法Sample.md)
9. [Naive GEMM 完整推导](09_Naive_GEMM完整推导.md)
10. [卷二复习、练习答案与面试题](10_卷二复习_练习答案与面试题.md)

## 当前配套实验

### Vector Add

```bash
cd /home/qichengjie/workspace/cuda_study/cuda_deep_course
make -C labs/02_programming_model/vector_add clean all
./labs/02_programming_model/vector_add/vector_add
```

### Index Mapping

```bash
make -C labs/02_programming_model/index_mapping clean all
./labs/02_programming_model/index_mapping/index_mapping
```

### 更多 Sample

```bash
make -C labs/02_programming_model/function_qualifiers clean all
make -C labs/02_programming_model/memory_lifecycle clean all
make -C labs/02_programming_model/async_errors clean all
make -C labs/02_programming_model/compile_inspection clean all
make -C labs/02_programming_model/event_timing clean all
make -C labs/02_programming_model/matrix_add_2d clean all
make -C labs/02_programming_model/gemm_naive clean all
```

## 当前完成标准

- [ ] 能从头写出 vector-add 的 Host/Device 完整流程。
- [ ] 能解释为什么 kernel launch 后要检查两类错误。
- [ ] 能推导一维全局索引和 block 数量。
- [ ] 能使用 `x=col, y=row` 写二维索引。
- [ ] 能解释 block 可以是 `m x n`。
- [ ] 能正确处理不能整除 block 的输入。
- [ ] 能解释四种 CUDA 函数修饰符。
- [ ] 能管理分配、复制和释放生命周期。
- [ ] 能区分 launch error 与 execution error。
- [ ] 能解释 NVCC、PTX、cubin/SASS。
- [ ] 能正确使用 CUDA Event。
- [ ] 能独立实现非方阵矩阵加法和非整除 GEMM。
