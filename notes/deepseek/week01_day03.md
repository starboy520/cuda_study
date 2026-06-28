# Week 01 · Day 03 — 迁移到 A100 + 重新 profile

> 日期：2026-06-28
> 主题：从 T4 迁移到 A100，重新编译/基准化 2D tiling，用 ncu 复测瓶颈画像
> 硬件：从 Tesla T4 (sm_75) → NVIDIA A100 80GB PCIe (sm_80, 108 SM)，CUDA 13.3

---

## 1. 今天目标

- [x] 把 T4 上的工作迁到 A100（git 同步 + 环境核对）
- [x] `-arch=sm_80 -O3` 重新编译 baseline / 2D tiling
- [x] 拿到 A100 真实 GFLOPS（脱离 ncu）
- [x] 配好 ncu 权限并复测 2D kernel（512 / 2048）
- [x] 更新 benchmark.md / ncu_notes.md

## 2. 今天做了什么

- 核对环境：A100 80GB PCIe，sm_80，CUDA 13.3。
- 发现迁移过来的二进制是 T4 的 sm_75 旧产物，重新用 `-arch=sm_80` 编译。
- 跑真实基准（2048）：baseline ~4194 GFLOPS、2D+padding ~11313 GFLOPS。
- 排查一个困惑：ncu 下打印的 `0.09 GFLOPS` 是假值（`--set full` 重放 52 passes + 锁频 + 插桩），真实性能要脱离 ncu 看。
- 用干净命令（`-o report.ncu-rep`，不用 tee）重采 512 和 2048，导入分析。

## 3. 数据是什么

真实性能（无 profiling）：

| 版本 | 2048 GFLOPS (A100) | 占峰值(~19.5T) | T4 对照 |
| --- | --- | --- | --- |
| shared baseline | ~4194 | ~21% | 563 |
| 2D + padding | ~11313 | ~58% | 1284 |

ncu 关键指标（2D kernel，512 vs 2048）：

| 指标 | @512 | @2048 |
| --- | --- | --- |
| Grid (blocks) | 64 | 1024 |
| Waves Per SM | <1 | 1.58 |
| DRAM Throughput | 1.11% | 1.53% |
| Compute (SM) | 39.12% | 56.25% |
| L1/TEX | 49.32% | 73.96% |
| Registers / thread | 168 | 168 |
| Theoretical Occupancy | 18.75% | 18.75% |
| Achieved Occupancy | 6.46% | 14.53% |
| shared load bank conflict | 1.5-way | 1.5-way |

## 4. 为什么 A100 的瓶颈画像和 T4 不同

- T4：L1/TEX 满 + occupancy 100% → shared/L1 bound。
- A100：DRAM 全闲(1.5%) + occupancy 仅 14.5% → **寄存器卡死 occupancy**。
- 理论 occupancy 只有 18.75%：168 寄存器/线程 → 每 SM 只能放 6 个 block（Block Limit Registers=6）。
- 512 实测更低（6.46%）：grid 才 64 个 block，填不满 108 个 SM；放大到 2048（1024 block）后回到 14.53%。
- padding 在 A100 同样有效，bank conflict 仍是 1.5-way，不是瓶颈。

## 5. 两个踩坑记录

1. **迁移后必须按目标架构重编译**：sm_75 二进制在 A100 上靠 JIT 跑，且旧 profile 文件里 18 GFLOPS 是 ncu 下的垃圾值，丢弃重采。
2. **ncu 打印的 GFLOPS 不能信**：`--set full` 重放几十 passes + 锁频，进程内 cudaEvent 计时会爆炸。真实性能脱离 ncu 跑；ncu 只看指标（occupancy / bank conflict / throughput），不看它打印的时间。
3. 采集别用 `... | tee file` 叠加重定向（会重复两遍），直接 `-o report.ncu-rep` 导出。

## 6. 明天要验证什么

- Day 5 参数实验：在 A100 上扫 TM/TN（8x8 → 4x4 降寄存器）、BK、BM/BN，
  在「寄存器↓ → occupancy↑」与「单线程复用↓」之间找 A100 最优点。
- 目标：把 occupancy 从 14.5% 往理论上限上方推（降寄存器换更多驻留 block）。

## 7. 面试口述

> 把同一个 2D register tiling kernel 从 T4 迁到 A100 后重新 profile，瓶颈画像完全变了。T4 上 L1/TEX 和 occupancy 都打满，卡在 shared 带宽；A100 上 DRAM 只有 1.5%、occupancy 只有 14.5%，根因是每线程 168 个寄存器把理论 occupancy 卡死在 18.75%。我还做了规模对照：512 时 grid 只有 64 个 block，填不满 108 个 SM，occupancy 掉到 6.5%；放大到 2048 后 grid 1024 个 block，occupancy 回到 14.5%、compute 利用率到 56%。所以 A100 上的优化方向不是访存而是降寄存器换 occupancy。这也提醒我：同一个 kernel 在不同卡上瓶颈可能完全不同，优化必须基于实测 profile，不能照搬结论。
