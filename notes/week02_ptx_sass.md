# Week 2：PTX / SASS 与性能诊断 Worklog

> 执行计划：[Week 2 PTX/SASS 与性能诊断](../study_plan/current/Week2_PTX_SASS与性能诊断.md)\
> 实验对象：[GPU Kernel Engineering / FP32 GEMM](https://github.com/starboy520/gpu-kernel-engineering/tree/main/projects/gemm)\
> 硬件：NVIDIA A100 80GB PCIe（`sm_80`）

## 本周目标

```text
CUDA C++ → PTX → ptxas → SASS → ncu → 单变量诊断
```

## 环境

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA A100 80GB PCIe |
| Driver | 610.43.02 |
| CUDA / nvcc | 13.3|
| Nsight Compute |13.3 |
| cuobjdump | 13.3|
| GEMM commit | |

---

# Day 1：编译链

## 今日目标

- [x] 生成 PTX
- [x] 生成 sm_80 cubin
- [x] 保存 ptxas 资源日志
- [x] 导出 SASS
- [x] 闭卷画编译链

## 产物

```text
/tmp/cuda_focus/week02/day01/
```

## 命令记录

```text
commands.sh：
```

## Code Object 证据

```text
elf_list.txt 中的 sm_80 条目：ELF file    1: naive.sm80.sm_80.cubin
```

## 五题答案

### 1. PTX 为什么不是最终机器码？

ptx是英伟达定义的虚拟ISA, 表达CUDA底层语义，但是不某一代GPU直接执行的机器指令，它还需要通过tpxas离线编译，或者通过JIT 生成面向具体架构的机器码， A100最终执行的是sm_80机器码，  SASS只是这种机器码的反汇编表示。

### 2. `compute_80` 与 `sm_80` 的区别

* compute_80是虚拟架构目标，规定生成PTX使用哪些compute capability 8.0特性
* sm_80是具架目标，

### 3. ptxas 做什么？

* ptxas 把 PTX编程具体的sm_xx的机器码同事完成一些执行选择优化，-Xptxas=-v 能 看到一些register、thread,shared memmory等，

### 4. Driver JIT 何时工作？

驱动加载fatbin的时候， 优先选择与当前GPU兼容的cubin，因为里面有具体机器架构吗，如果兼容的cubin，但是保留了可兼容的ptx， 驱动使用JIT生成目标吗

### 5. `-lineinfo` 与 `-G` 的区别

 -lininfo  是帮助ncu， 反汇编的时候关联源码，保留优化， -G是调试构建， 会关闭所有的优化， 可以理解为debug model，

## ptxas 资源

| Kernel | registers/thread | shared memory | spill stores | spill loads |
| --- | ---: | ---: | ---: | ---: |
| Naive |32 | 0| 0| 0|

## 一句话口述

>

## 今日错误与明日第一步

- 错误：
- 明日第一步：

---

# Day 2：Naive PTX 数据流

逐段阅读辅助：[Naive GEMM PTX 逐段对照注释](week02_naive_ptx_annotated.md)

## 数据流定位

| 源码动作 | PTX label / 行号 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- | --- |
| thread/block id |40-42,45-47 |`%ctaid.{x,y}`, `%ntid.{x,y}`, `%tid.{x,y}`| `%r15-20`|读取 blockdim, blockidx, threadidx |
| row |43 | `%r15-17`| `%r1`|计算row |
| column | 48|`%r18-20`|`%r2` | 计算column |
| 边界 predicate |50-52 |`%r1-r2`, `%r12`, `%r14`|`%p1`、`%p2`、`%p3` | 计算 row>= m || column >=n|
| A/B 基址 | 31–32、38–39 | kernel 参数 `%rd18`、`%rd19` | `%rd2`（A global base）、`%rd1`（B global base） | 读取 A/B 指针并转换到 global address space |
| A 地址 | 68-69， 79， 79 | `a base k, k, row, 循环偏移`|`%rd21` |   `a[row*k+i] |
| B 地址 | 32|`_Z12naive_kernelPKfS0_Pfiii_param_1` | `rd19`| |
| A/B load |38-39 | `%rd18-19`|`%rd102`| |
| FMA |82,86,90,94 | `%f12,13,29`|`%f14` | |
| C store |137 | `%f29`|`rd29` | |

## 我能证明什么

* kernel 索引， 边界， 加载， 四次循环展开， FMA

## 我不能证明什么

* 合并访问
* memory-bound
* 性能
* bank conflict

## 三分钟口述

>

## 今日错误与明日第一步

- 错误：
- 明日第一步：

---

# Day 3：PTX 与 SASS 对照

| 源码动作 | PTX 证据 | SASS 证据 | 能证明 | 不能证明 |
| --- | --- | --- | --- | --- |
| thread mapping | | | | |
| global load | | | | |
| FMA | | | | |
| global store | | | | |
| control | | | | |

## PTX 与 SASS 为什么不一一对应

## 三分钟口述

>

## 今日错误与明日第一步

- 错误：
- 明日第一步：

---

# Day 4：五版 GEMM 指令观察

| 版本 | Global load | Shared 路径 | 同步 | 主计算 | registers | spill | 一句话变化 |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| Naive | | 无 | 无 block barrier | | | | |
| Shared | | | | | | | |
| Register | | | | | | | |
| Vectorized | | | | | | | |
| Async 16B | | | | | | | |

## 宽加载证据

- Vectorized：
- Async 16B：

## “生成了指令”为什么不等于“性能更快”

## 今日错误与明日第一步

- 错误：
- 明日第一步：

---

# Day 5：Register / Spill / Local / ILP

## 现有资源对照

| 版本 | registers/thread | shared memory | spill | occupancy | wall-clock |
| --- | ---: | ---: | ---: | ---: | ---: |
| Register | | | | | |
| Vectorized | | | | | |
| Async 16B | | | | | |

## 单变量实验

### 假设

### 反证条件

### Baseline

### Variant

### 唯一改变

### Microbenchmark 复现信息

```text
source: /tmp/cuda_focus/week02/day05/ilp_dependency.cu
sha256:
compile command:
shape / launch config:
```

## 今日错误与明日第一步

- 错误：
- 明日第一步：

---

# Day 6：Scheduler / Scoreboard 诊断

## 诊断报告

| 项 | Baseline | Variant |
| --- | --- | --- |
| Correctness | | |
| Normal wall-clock | | |
| registers/thread | | |
| shared memory | | |
| achieved occupancy | | |
| active warps | | |
| eligible warps | | |
| issued warps | | |
| long scoreboard | | |
| short scoreboard | | |
| 关键 SASS | | |

### 现象

### 候选机制

### 复测结论

- [ ] 支持假设
- [ ] 否定假设
- [ ] 证据不足

### 适用边界

```text
GPU：
shape：
dtype：
build：
ncu metric set：
```

## 五分钟口述

>

## 今日错误与明日第一步

- 错误：
- 明日第一步：

---

# Day 7：WMMA → PTX MMA → SASS HMMA

## 三层证据

| 层 | 观察对象 | 实际证据 | 能证明 | 不能证明 |
| --- | --- | --- | --- | --- |
| CUDA C++ | WMMA API | | | |
| PTX | `wmma.mma.sync...m16n16k16` | | | |
| SASS | HMMA；当前实现预期无 LDSM | | | |

## 必答

### Fragment 是什么？

### 当前 `m16n16k16` 的 M/N/K 是什么？

### `ldmatrix` 解决什么？

### 当前标本为什么没有 LDSM？

### 为什么看到 HMMA 仍不能证明性能高？

## 五分钟口述

>

---

# 周末闭卷验收

| 能力 | 结果 | 卡点 |
| --- | --- | --- |
| 独立生成 PTX/cubin/SASS | ⬜ | |
| 追踪 Naive PTX 数据流 | ⬜ | |
| 按类别阅读 SASS | ⬜ | |
| 解释 registers/shared/spill/local | ⬜ | |
| 区分 active/eligible/issued | ⬜ | |
| 完成单变量诊断 | ⬜ | |
| 讲清 WMMA/PTX MMA/HMMA | ⬜ | |

## 本周三个最重要结论

1.
2.
3.

## 最有价值的失败

## Week 3 开场补考项（每项最多 30 分钟）

- [ ] 无
