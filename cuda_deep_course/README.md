# CUDA 深度学习工程

本目录包含新 CUDA 教材的全部内容，旧的 Week 学习材料不属于本教材。

## 目录

```text
cuda_deep_course/
├── README.md       # 总入口
├── course/         # 教材正文
├── labs/           # 可编译、可运行的 CUDA 实验
├── docs/           # 教材设计规范和实施计划
└── assets/         # 后续加入图解和 profiling 结果
```

## 开始学习

[进入教材总目录](course/README.md)

建议顺序：

1. 阅读 [术语与符号约定](course/术语与符号约定.md)。
2. 阅读 [实验方法与完成标准](course/实验方法与完成标准.md)。
3. 开始 [卷一：CUDA 入门所需的 GPU 基础](course/volume01_gpu_basics/README.md)。
4. 完成卷一实验后进入 [卷二：CUDA 编程模型](course/volume02_programming_model/README.md)。
5. 继续学习 [卷三：CUDA 内存系统](course/volume03_memory_system/README.md)。
6. 继续学习 [卷四：同步与经典并行算法](course/volume04_parallel_algorithms/README.md)。
7. 继续学习 [卷五：性能工程与 Profiling](course/volume05_performance/README.md)。

## 构建当前实验

```bash
cd /home/qichengjie/workspace/cuda_study/cuda_deep_course

make -C labs/01_gpu_basics/device_query clean all
make -C labs/02_programming_model/vector_add clean all
make -C labs/02_programming_model/index_mapping clean all
make -C labs/02_programming_model/function_qualifiers clean all
make -C labs/02_programming_model/memory_lifecycle clean all
make -C labs/02_programming_model/async_errors clean all
make -C labs/02_programming_model/compile_inspection clean all
make -C labs/02_programming_model/event_timing clean all
make -C labs/02_programming_model/matrix_add_2d clean all
make -C labs/02_programming_model/gemm_naive clean all
make -C labs/03_memory_system/memory_access clean all
make -C labs/03_memory_system/transpose clean all
make -C labs/04_parallel_algorithms/reduction clean all
make -C labs/06_operators/gemm_tiled clean all
make -C labs/06_operators/softmax clean all
make -C labs/07_async_system/overlap_pipeline clean all
make -C labs/07_async_system/cuda_graph clean all
```

## 运行当前实验

```bash
./labs/01_gpu_basics/device_query/device_query
./labs/02_programming_model/vector_add/vector_add
./labs/02_programming_model/index_mapping/index_mapping
./labs/02_programming_model/function_qualifiers/function_qualifiers
./labs/02_programming_model/memory_lifecycle/memory_lifecycle
./labs/02_programming_model/async_errors/async_errors
./labs/02_programming_model/compile_inspection/compile_inspection
./labs/02_programming_model/event_timing/event_timing
./labs/02_programming_model/matrix_add_2d/matrix_add_2d
./labs/02_programming_model/gemm_naive/gemm_naive
./labs/03_memory_system/memory_access/memory_access
./labs/03_memory_system/transpose/transpose
./labs/04_parallel_algorithms/reduction/reduction
./labs/06_operators/gemm_tiled/gemm_tiled 2048 2048 2048
./labs/06_operators/softmax/softmax
./labs/07_async_system/overlap_pipeline/overlap_pipeline
./labs/07_async_system/cuda_graph/cuda_graph
```

## 章节 ↔ 实验对照（速查）

| 卷 | 章节 | 实验 |
|---|---|---|
| 1 | 05 设备观察 | `01_gpu_basics/device_query` |
| 2 | 01–09 各章 | `02_programming_model/`（vector_add / index_mapping / function_qualifiers / memory_lifecycle / async_errors / compile_inspection / event_timing / matrix_add_2d / gemm_naive） |
| 3 | 02 合并访问 | `03_memory_system/memory_access` |
| 3 | 05 转置 | `03_memory_system/transpose` |
| 4 | 03 Reduction | `04_parallel_algorithms/reduction` |
| 4 | 01/02/04/05 | ⏳ 待补（留作动手练习，写在 `week03_parallel/`） |
| 5 | 全卷 | 复用 `memory_access` / `transpose` / `reduction` / `vector_add` 做分析 |
| 6 | 02 GEMM Tiling | `06_operators/gemm_tiled` |
| 6 | 04 Softmax | `06_operators/softmax` |
| 6 | 01/03/05/06 | ⏳ 待补 |
| 7 | 02 重叠 | `07_async_system/overlap_pipeline` |
| 7 | 04 CUDA Graph | `07_async_system/cuda_graph` |

> 完整对照（含每章说明、纯概念章节标注）见 [course/README.md 配套实验](course/README.md#配套实验)。
> 标 ⏳ 的章节正文有完整原理 + 代码骨架，但没有现成 lab——这些是留给你**自己写**的练习。

## 教材设计

- [完整教材设计](docs/specs/2026-06-14-cuda-deep-course-design.md)
- [第一批实施计划](docs/plans/2026-06-14-cuda-course-foundation.md)
- [卷三至卷五实施计划](docs/plans/2026-06-14-volumes-03-05.md)
