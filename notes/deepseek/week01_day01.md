# Week 01 · Day 01 — shared memory GEMM baseline 复盘

> 日期：2026-06-26
> 主题：Week 1 Day 1 — 复盘 shared memory GEMM，建立 baseline
> 硬件：Tesla T4 (sm_75)

---

## 1. 今天目标

- [x] 重新整理 shared tiled GEMM，确认 block tile / thread / shared tile 各自在干什么
- [x] 跑 512 / 1024 / 2048 三档，记录 GFLOPS
- [x] 用 ncu 采集关键指标，判断瓶颈
- [x] 写出「shared tiling 解决了什么、没解决什么」的口述

## 2. 今天做了什么

- 修正了 `gemm_shared_baseline.cu` 的若干 bug（shared tile 类型 int→float、`column` 计算用错运算符、输出索引 `row*M`→`row*N`、越界写、TILEB else 下标、内层循环变量遮蔽）。
- 补全 `main()`：命令行读 M/N/K、host/device 内存、随机初始化、warmup + 正式计时、CPU reference 校验、`CUDA_CHECK`。
- launcher 加入 GFLOPS 计算（`2*M*N*K / time_s / 1e9`）。
- 采集 ncu（`--set full -s 1 -c 1`，M=N=K=1024）。

文件：
- 代码：`week05_gemm_advanced/gemm_shared_baseline.cu`
- 数据：`week05_gemm_advanced/benchmark.md`
- 分析：`week05_gemm_advanced/ncu_notes.md`

## 3. 数据是什么

| M=N=K | time (ms) | GFLOPS | 正确性 |
| --- | --- | --- | --- |
| 512  | 0.742  | 361.86 | PASS |
| 1024 | 4.138  | 518.97 | PASS |
| 2048 | 30.497 | 563.33 | PASS |

ncu 关键指标（1024）：

| 指标 | 值 |
| --- | --- |
| L1/TEX Cache Throughput | 85.51%（SOL 最高项） |
| DRAM Throughput | 12.40% |
| Compute (SM) Throughput | 83.97% |
| Achieved Occupancy | 100% |
| Registers Per Thread | 35 |

## 4. 为什么是这个结果（瓶颈分析）

- DRAM 只占 12% → shared tiling 成功消除 global memory 重复读取，**不是 DRAM-bound**。
- L1/TEX 高达 85.5%（最高项）→ shared memory 访问走 L1/TEX，每个 thread 在 K 循环里反复读 A/B 把它压满。
- occupancy 已 100% → 性能问题不是并行度不足。
- 结论：瓶颈 = **shared memory / L1 throughput bound**。当前 2048 仅约 T4 FP32 峰值的 7%。

## 5. 明天要验证什么

- 2D thread tiling：一个 thread 算 TM×TN 个输出，用 regM/regN 做外积，减少 shared memory 读取次数。
- 预期：L1/TEX 占比下降、Compute(SM) 上升、GFLOPS 提高；寄存器上升、occupancy 可能下降。
- 先纸上推演索引，再动手写。

## 6. 面试口述

> shared tiling 把 A/B 的 tile 搬进 shared memory，让 block 内线程复用，从而大幅减少 global memory 重复访问、提高算术强度。但它没解决的是：每个线程只算一个输出，在 K 方向内层循环里仍反复从 shared memory 读 A/B，shared memory 带宽成为新瓶颈。ncu 上表现为 DRAM 只有 12% 而 L1/TEX 高达 85%、occupancy 已 100%。下一步用 register tiling 让一个线程算多个输出，把数据读进寄存器复用，减少 shared 访问，把瓶颈从访存转向计算。
