# Week 01 · Day 02 — 2D thread tiling + profiling + padding

> 日期：2026-06-27 ~ 06-28
> 主题：Week 1 Day 2-4 合并 — 手写 2D register tiling，ncu profile，发现并修复 bank conflict
> 硬件：Tesla T4 (sm_75)
> 注：1D thread tiling 已降级为概念对照，不单独实现。

---

## 1. 今天目标

- [x] 推清 2D thread tiling：一个 thread 算 TM×TN 个 C，regM/regN 外积
- [x] 手写 `gemm_2d_thread_tiling.cu` 并跑通 512 / 1024 / 2048
- [x] correctness PASS，记录 GFLOPS、寄存器数量
- [x] ncu profile，找瓶颈
- [x] 发现 bank conflict 并用 padding 修复，验证收益

## 2. 今天做了什么

- 手写 2D tiling kernel（BM=BN=64, BK=32, TM=TN=8, 64 线程，外积累加）。
- 调试加载索引：sa row 连续/column 跳步，sb 相反（对称覆盖）。
- ncu 发现 L1/TEX 不降反升到 91%，定位到 4.8-way shared bank conflict。
- 给 shared 数组加 `+1` padding（sa[BM][BK+1] / sb[BK][BN+1]）消冲突。
- 更新 benchmark.md、ncu_notes.md。

## 3. 纸上推演（外积）

2D 外积（TM=2, TN=2）：

```text
regM = [a0, a1]   # 来自 A 的一小列
regN = [b0, b1]   # 来自 B 的一小行
acc00 += a0*b0   acc01 += a0*b1
acc10 += a1*b0   acc11 += a1*b1
```

复用关键：每步读 TM+TN 个数，做 TM*TN 次乘加。1D 只是过渡概念（一个 thread 算多个输出），2D 才是主线。

## 4. 数据是什么

性能（无 profiling，GFLOPS）：

| 版本 | M=N=K=2048 GFLOPS | 相对 baseline |
| --- | --- | --- |
| shared baseline | 563 | 1.0x |
| 2D tiling（无 padding） | 884 | 1.57x |
| 2D tiling + padding | 1284 | 2.28x |

ncu（1024）padding 前后：

| 指标 | padding 前 | padding 后 |
| --- | --- | --- |
| shared load bank conflict | 4.8-way / 66.68% | 1.5-way / 33.40% |
| L1/TEX Throughput | 90.98% | 83.09% |
| Compute (SM) Throughput | 20.49% | 41.95% |
| Elapsed Cycles | 2,104,074 | 1,254,785 |
| Registers / thread | 167 | 168 |
| Achieved Occupancy | 21.78% | 17.06% |

## 5. 为什么变快 / 变慢

- 2D tiling 比 baseline 快：每步读 16 个数做 64 次乘加，shared 访问量被摊薄。
- 但 L1/TEX 反升 → 发现是 bank conflict：BK=32 让 sa 列读取全落同一 bank（4.8-way）。
- padding 错开 stride（33）后，不同 row 落到不同 bank，冲突 4.8→1.5，L1/TEX 松开，compute 翻倍 → 再快 1.45x。
- 副作用：padding 破坏 16B 对齐，shared load 无法向量化，请求数上升，但冲突减少占主导，净赢。

## 6. 明天要验证什么

- 当前瓶颈：occupancy 仅 17%（被 shared mem + 168 寄存器限制），全局 load 未 coalesced（每 sector 只用 7.1/32 字节）。
- 下一步（Week 2 Day 1）：vectorized / coalesced global load，看能否进一步提速。

## 7. 面试口述

> 我给 2D tiling 做 profile，发现 L1/TEX 不降反升到 91%，一开始预测错了。但 Memory Workload Analysis 直接指出 shared load 有 4.8-way bank conflict，根因是 BK=32 让 shared 数组的列读取全部映射到同一个 bank。加了 +1 padding 把 stride 错开后，bank conflict 从 4.8 路降到 1.5 路，L1/TEX 降到 83%，compute throughput 翻倍，2048 从 884 涨到 1284 GFLOPS，相对最初 baseline 是 2.28x。这是一次完整的「假设→profile 发现真因→针对性修复→数据验证」闭环。

> （写一段：为什么 2D tiling 是外积？相比 shared baseline 复用了什么？代价是什么？）
