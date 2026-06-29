# Week 01 · Day 05 — Day 6 访存账 + Roofline（A100）

> 日期：2026-06-29
> 主题：Week 1 Day 6 — 算 naive/shared/2D 的 FLOP/bytes/AI，建 A100 Roofline，判 bound，解释「2D 的好处为何 DRAM-Roofline 看不出来」
> 硬件：NVIDIA A100 80GB PCIe (sm_80, 108 SM, FP32 峰值 19.5T, 带宽 1935 GB/s)
> 产出：`week05_gemm_advanced/roofline.md`

---

## 1. 今天目标

- [x] 算 FLOP = 2*M*N*K（M=N=K=2048）
- [x] 算 naive / shared / 2D 三版 global bytes 与 AI
- [x] 算 Roofline 拐点 AI* = P/B
- [x] 三版本标到 Roofline 判 bound
- [x] 解释 shared vs 2D 同 AI 却快 2.7× → Roofline 盲区
- [x] 整理 roofline.md + 口述

## 2. 今天做了什么

- FP32、方阵 2048、tile BM=BN=64 为前提，按「数每个数组从 global 读几遍」估 bytes。
- naive：每输出读 2K 元素无复用；shared/2D：tile 复用，DRAM bytes 降 64×。
- 关键认知：2D 与 shared 的 DRAM 流量相同，省的是 shared/L1 读取，不是 DRAM。
- 结论写入 roofline.md，含面试口述。

## 3. 数据是什么

| 版本 | global bytes | AI | vs 拐点10 | bound(DRAM层) | 实测 GFLOPS(2048) |
| --- | --- | --- | --- | --- | --- |
| naive | 68.7 GB | 0.25 | ≪ | memory | （未测） |
| shared | ~1.07 GB | 16 | > | compute侧 | ~4194 |
| 2D+pad | ~1.07 GB | 16 | > | compute侧 | ~11313 |

```text
FLOP = 2*2048^3 = 17,179,869,184
shared 元素 = 2 * (2048^3/64) = 268,435,456 → *4 ≈ 1.07GB
拐点 AI* = 19.5e12 / 1935e9 ≈ 10
```

## 4. 三个结论

1. naive→shared：复用把 DRAM bytes 降 64×，AI 0.25→16，越拐点上算力屋顶。
2. shared vs 2D：global AI 都=16，Roofline 在 DRAM 层无法区分；2D 砍 shared 读取，实测仍快 2.7×。
3. A100 上已是 compute/occupancy 受限，下一步：vectorized load / double buffering / Tensor Core。

## 5. 盲区（高级点）

Roofline 只刻画 DRAM。两版 DRAM bytes 同→落点同，实测差 2.7×→须用 shared 访存账或 ncu(occupancy/shared throughput) 才能解释。呼应 Day5 的 occupancy 实验。

## 6. 明天要验证什么

- Day 7：画优化阶梯图 + 3 分钟口述，Week1 收尾。
- Week2 Day1：float4 向量化 global load，迁移到 GEMM A/B tile 加载。

## 7. 面试口述

> 我用 AI=FLOP/bytes 对拐点判 bound。naive AI=0.25 远低于 A100 拐点 10，重度 memory-bound；tiling 复用把 DRAM bytes 降 64× → AI=16 越拐点进 compute 侧。但 shared 和 2D 的 global AI 都是 16，Roofline 区分不了，实测却差 2.7×，说明瓶颈已从 DRAM 转到 shared/occupancy——这时改看 ncu 的 shared throughput 和 occupancy，下一步用向量化、double buffering、Tensor Core 逼近峰值。

---

## 8. GEMM 优化阶梯（Day7 收尾）

阶梯：`naive → shared tiling → 2D register tiling`

| 版本 | 复用层级 | 砍的流量 | AI | bound |
| --- | --- | --- | --- | --- |
| naive | 无，全程读 DRAM | — | 0.25 | memory |
| shared tiling | block 内复用 shared | 砍 DRAM | 16 | 转 shared/指令 |
| 2D register tiling | 线程内寄存器外积 | 砍 shared 读取 | 16 | compute/occupancy |

```text
naive  : 每线程算 1 个 C[i][j]，K 循环每步读 global A/B，无复用 → AI 0.25
shared : A/B tile 搬进 shared，block 内复用 → DRAM bytes 降 64×，AI=16
2D     : 一线程算 TM*TN，regM[TM] 读 A 列、regN[TN] 读 B 行 → 外积
         acc[i][j] += regM[i]*regN[j]
         读 TM+TN 个、算 TM*TN 次 → 砍 shared 流量，DRAM 不变（AI 仍 16）
代价   : 2D 寄存器多、occupancy 降，但复用收益更大 → 实测快 2.7×
（1D 只是过渡：一线程算 TM 个、单边复用，不单独实现）
```

一句话区分：
```text
naive  重复读 DRAM
shared DRAM→shared，砍 DRAM 流量
2D     shared→寄存器外积，砍 shared 流量
```

