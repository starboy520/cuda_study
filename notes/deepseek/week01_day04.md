# Week 01 · Day 04 — Day 5 参数实验（A100）

> 日期：2026-06-28
> 主题：Week 1 Day 5 — 通用 grid-stride 加载改造 + 扫 TM/TN/BM/BN/BK + ncu 验证 occupancy
> 硬件：NVIDIA A100 80GB PCIe (sm_80, 108 SM)
> 注：worklog 文件名按节奏排 day04，对应计划里的 Week 1 Day 5。

---

## 1. 今天目标

- [x] 把 shared 加载从「位置驱动」改成「grid-stride 编号驱动」，与计算解耦
- [x] 参数 `#ifndef` 守卫化，支持 `-D` 编译时覆盖
- [x] 写扫描脚本 `sweep.sh`，扫 TM/TN/BM/BN/BK
- [x] 记录每组 寄存器数 + GFLOPS + 正确性
- [x] ncu 验证「降寄存器 → occupancy 上升」

## 2. 今天做了什么

- 重写 sa/sb 加载为 grid-stride（`for i=tid; i<总数; i+=nthreads`），下标只用 BK/BN，不含 TM/TN。
- 旧的位置驱动加载注释保留作对照；两种写法整理进 `shared_load_patterns.md`。
- 参数改 `#ifndef` 守卫，写 `sweep.sh` 用 `-D` 扫 7 组组合。
- 重写后寄存器 168→122，仍全部 PASS。
- ncu 对比 8x8 基线 vs 8x4 赢家。

## 3. 数据是什么

扫描结果（2048）：

| TMxTN | BMxBN | BK | reg | GFLOPS | chk |
| --- | --- | --- | --- | --- | --- |
| **8x4** | 64x64 | 16 | **78** | **11382** | PASS 🥇 |
| 8x8 | 128x64 | 16 | 127 | 10249 | PASS |
| 4x4 | 64x64 | 16 | 72 | 9981 | PASS |
| 8x8 | 64x64 | 32 | 122 | 9625 | 基线 |
| 8x8 | 64x64 | 16 | 122 | 9506 | PASS |
| 8x8 | 128x128 | 16 | 122 | 8285 | PASS |
| 4x4 | 128x128 | 16 | 72 | — | 启动失败 |

ncu（8x8 基线 vs 8x4 赢家）：

| 指标 | 8x8 | 8x4 |
| --- | --- | --- |
| reg/thread | 122 | 78 |
| Achieved Occupancy | 14.53% | 30.10% |
| SM Throughput | 56.25% | 66.28% |
| GFLOPS | 9625 | 11382 |

## 4. 为什么变快（因果链）

```text
TN 8→4 → reg_c[8][4] 累加器减半 → 寄存器 122→78
→ 每 SM 驻留 block 翻倍 → occupancy 14.5%→30.1%
→ 活跃 warp 增多，藏延迟更强 → SM throughput 56%→66%
→ GFLOPS 9625→11382（+18%）
```

## 5. 两个反例 / 边界（同样重要）

1. **甜点效应**：4x4（72 reg）= 9981 < 8x4（78 reg）= 11382。TM/TN 砍过头 → 单线程复用/ILP 不足 → occupancy 收益被抵消。**occupancy 不是越高越好**。
2. **启动失败边界**：`4x4 128x128` → block 1024 线程 × 72 reg = 73728 > 65536（A100 每 SM 寄存器上限）→ "too many resources requested for launch"。约束：`线程数 × 寄存器/线程 ≤ 65536`。
3. **大 tile 反慢**：128x128（8285）→ 2048 下只有 256 个 block，喂不饱 108 SM，occupancy 反降。

## 6. 明天要验证什么

- 当前最优 8x4 的 occupancy 30%，仍不算高 → 还能否进一步（如 vectorized load / 调 BK）？
- Day 6：访存账 + Roofline（用 A100 峰值 ~19.5T / 带宽 ~1935 GB/s），判断各版本 memory/compute-bound。
- 顺带：grid-stride 改造后 global load 是否更 coalesced（对比旧版 7.1/32 字节）。

## 7. 面试口述

> 我在 A100 上做了 GEMM 参数扫描，写了个脚本用 `-D` 编译不同 tile 配置，自动记录寄存器数、GFLOPS 和正确性。发现把 TN 从 8 降到 4（reg_c 累加器减半），寄存器从 122 降到 78，achieved occupancy 从 14.5% 翻倍到 30.1%，SM 利用率从 56% 升到 66%，2048 GEMM 快了 18%。但不是越激进越好：TM/TN 都砍到 4 反而更慢，因为单线程复用和 ILP 不够，存在甜点。我还撞到一个边界——4x4 配 128x128 时 block 要 1024 线程、每线程 72 寄存器，超过每 SM 65536 的寄存器上限，直接启动失败。这套实验把「occupancy / 寄存器 / 单线程复用」的权衡用数据讲清楚了。

> 另一条（加载解耦）：我还把 shared 加载从「每个线程搬自己要算的数据」改成「线程按一维编号 grid-stride 搬运」，让加载和计算解耦。好处是任意 tile 参数都正确（否则改参数就越界）、且相邻线程访问连续地址天然 coalesced。这是工业级 GEMM（CUTLASS）的通用写法。
