# GEMM Nsight Compute Notes

## 环境

| 项目 | 值 |
| --- | --- |
| GPU | Tesla T4 (sm_75, 40 SM, 16 GB) |
| CUDA | release 13.3, V13.3.33 |
| 采集命令 | `ncu --set full -s 1 -c 1 ./gemm_shared_baseline 1024` |
| 形状 | M=N=K=1024，FP32 |
| 配置 | block 32x32 (1024 线程)，grid 32x32，TILE=32，每线程算 1 个 C |

> 注意：ncu 会把 kernel 重放多个 pass 并锁频，报告里 `1139 ms / 1.88 GFLOPS` 是 profiling 干扰值，不代表真实性能。真实性能看无 profiling 时的 ~519 GFLOPS。

---

## Day 1：shared baseline 关键指标

### 命令
```text
ncu --set basic -s 1 -c 1 ./gemm_shared_baseline 1024
```

### Speed of Light（SOL）

| 指标 | 值 | 说明 |
| --- | --- | --- |
| Compute (SM) Throughput | 83.97% | SM 计算管线繁忙 |
| Memory Throughput | 83.97% | 内存子系统繁忙 |
| **L1/TEX Cache Throughput** | **85.51%** | **最高项 = 瓶颈** |
| DRAM Throughput | 12.40% | DRAM 几乎空闲 |
| L2 Cache Throughput | 6.57% | L2 空闲 |

### Launch / 寄存器

| 指标 | 值 | 说明 |
| --- | --- | --- |
| Block Size | 1024 | 32 个 warp/block |
| Registers Per Thread | 35 | 寄存器压力不高 |
| Static Shared Mem / Block | 8.45 KB | 两个 32x33 float tile |
| Waves Per SM | 25.6 | 任务量足够摊薄开销 |

### Occupancy

| 指标 | 值 | 说明 |
| --- | --- | --- |
| Achieved Occupancy | 100% | 占用率已拉满 |
| Block Limit Warps | 1 | 一个 block 32 warp 已占满 SM（T4 上限 32 warp/SM） |
| Block Limit Registers | 1 | 寄存器只够放 1 个 block |
| Block Limit Shared Mem | 3 | shared memory 还能放 3 个 block |

---

## 瓶颈判断

```text
DRAM 闲（12%）  +  L1/TEX 满（85.5%）  +  occupancy 满（100%）
=> 不是 DRAM-bound
=> 不是 occupancy 不足
=> 是 shared memory / L1 throughput bound
```

推理链：

- **DRAM 只有 12%** → shared tiling 成功消除了 global memory 的重复读取，已不是 global/DRAM 瓶颈。
- **L1/TEX 高达 85.5%（SOL 最高项）** → T4 上 shared memory 访问走 L1/TEX 单元；每个 thread 在 K 循环里反复从 shared memory 读 A/B，把 L1/TEX 压满。
- **Occupancy 已 100%** → 提升并行度这条路走到头，性能问题不是「线程不够多」。

结论：瓶颈已从 global memory 转移到 **shared memory / L1 throughput**，与 benchmark.md 中的预测一致。

---

## 下一步优化方向

```text
目标：减少每个 thread 对 shared memory 的读取次数。
方法：register tiling（1D / 2D thread tiling）。
原理：一个 thread 计算多个输出（TM 或 TM x TN），
      把 A/B 数据读进寄存器后复用多次，
      降低 L1/TEX 压力，把瓶颈从 shared 带宽转向计算。
```

预期：register tiling 后 L1/TEX throughput 占比下降，Compute(SM) throughput 上升，GFLOPS 提高；寄存器用量上升、occupancy 可能从 100% 下降，但因访存压力减小，整体仍更快。

---

## 面试口述

> 我用 ncu 看 baseline 的 Speed of Light，发现 DRAM throughput 只有 12%，但 L1/TEX cache throughput 高达 85%，是所有子系统里最高的，同时 achieved occupancy 已经 100%。这说明 shared tiling 已经把 global memory 重复访问消掉了，瓶颈不再是 DRAM，也不是并行度不足，而是 shared memory / L1 的带宽。每个线程只算一个输出，在 K 循环里反复读 shared memory 把 L1/TEX 压满了。所以下一步是 register tiling：让一个线程算多个输出，把数据读进寄存器复用，减少 shared memory 访问次数，把瓶颈从访存转向计算。

---

## Day 4：2D register tiling 的 ncu（含 bank conflict 发现 + padding 验证）

kernel：`gemm_2d_thread_tiling`（BM=BN=64, BK=32, TM=TN=8, 64 线程），profile shape M=N=K=1024。

> 同样注意：profile 里 `46 GFLOPS` 是 ncu 锁频值，真实性能看下面无 profiling 的 GFLOPS。

### Padding 前后对比（1024）

| 指标 | Padding 前 | Padding 后 (BK+1/BN+1) | 说明 |
| --- | --- | --- | --- |
| shared load bank conflict | 4.8-way / 66.68% | 1.5-way / 33.40% | 冲突大幅下降 |
| shared store bank conflict | 6.1-way / 83.44% | 1.6-way / 35.49% | 冲突大幅下降 |
| L1/TEX Throughput | 90.98% | 83.09% | 松开 |
| Compute (SM) Throughput | 20.49% | 41.95% | 翻倍 |
| Elapsed Cycles | 2,104,074 | 1,254,785 | -40% |
| 主导 stall | MIO 44% | fixed-latency dep 30% | MIO 不再主导 |
| Registers / thread | 167 | 168 | 基本不变 |
| Achieved Occupancy | 21.78% | 17.06% | 仍低，受 shared mem 限制 |

### 真实性能（无 profiling，GFLOPS）

| 版本 | M=N=K=2048 GFLOPS | 相对 baseline |
| --- | --- | --- |
| shared baseline | 563 | 1.0x |
| 2D tiling（无 padding） | 884 | 1.57x |
| 2D tiling + padding | 1284 | 2.28x |

约为 T4 FP32 峰值（~8.1 TFLOPS）的 16%。

### 关键发现：L1/TEX 反而升高的原因是 bank conflict

```text
预测「2D tiling 后 L1/TEX 下降」没有立刻兑现，反而升到 91%。
原因：BK=32 让 sa[row][k] 的地址 = row*32+k，bank = (row*32+k)%32 = k，
      一个 warp 里不同 row 的线程全落到同一个 bank → 4.8-way bank conflict。
L1/TEX 高不是因为有效吞吐，而是被冲突重放的无效 wavefront 塞满（66% 是冲突）。
修复：把 shared 数组 padding 成 [BK+1]/[BN+1]，stride 错开成 33，
      bank = (row*33+k)%32 = (row+k)%32，不同 row 落到不同 bank，冲突消失。
结果：冲突 4.8→1.5 way，L1/TEX 91→83%，compute 翻倍，2048 从 884→1284 GFLOPS（1.45x）。
```

### bank conflict 的「N-way」含义

```text
N-way = 一次 shared 访问被 bank 冲突拆成 N 个串行批次（wavefront），慢约 N 倍。
N = 总 wavefronts / 总 requests。
1 是理想（无冲突），padding 前 4.8，padding 后 1.5。
```

### 副作用（记录用）

padding 后 shared load 请求数反而上升（5.24M → 8.39M）。推测：stride=33*4=132 字节不再 16 字节对齐，
编译器无法把 reg_a 的读取合并成宽的向量化 shared load，只能拆成更多窄 load。
但每次访问的冲突重放从 4.8→1.5，净效果仍大赢。属于 padding 的经典 tradeoff：消冲突 vs 牺牲向量化。

### 当前还剩的瓶颈（下一步）

```text
1. Occupancy 仅 17%（被 shared memory + 168 寄存器限制）→ warp 太少，藏不住延迟。
2. 全局 load 未 coalesced（每 sector 只用 7.1/32 字节）→ Week 2 的 vectorized/coalesced load 课题。
3. partial wave 尾效应（grid 配置）。
```

### 面试口述（Day 4）

> 我给 2D tiling 做 profile，发现 L1/TEX 不降反升到 91%，一开始预测错了。但 Memory Workload Analysis 直接指出 shared load 有 4.8-way bank conflict，占 66% 的 wavefront。根因是 BK=32 让 shared 数组的列读取全部映射到同一个 bank。加了 +1 padding 把 stride 错开后，bank conflict 从 4.8 路降到 1.5 路，L1/TEX 降到 83%，compute throughput 翻倍，2048 从 884 涨到 1284 GFLOPS。这是一次完整的「假设→profile 发现真因→针对性修复→数据验证」闭环。

---

## Day 4（A100 复测）：迁移到 A100 后重新 profile

> 硬件换成 **NVIDIA A100 80GB PCIe (sm_80, 108 SM)**，CUDA 13.3，kernel 用 `-arch=sm_80 -O3` 重新编译。ncu 权限已配好（无需 sudo）。
> kernel：`gemm_2d_thread_tiling`（BM=BN=64, BK=32, TM=TN=8, 64 线程，已含 padding）。
> 采集：`ncu --set full -s 1 -c 1 -o <报告> ./gemm_2d_thread_tiling <N>`。

### 关键指标：512 vs 2048（A100）

| 指标 | @512 | @2048 | 说明 |
| --- | --- | --- | --- |
| Grid (blocks) | 64 | 1024 | 512 填不满 108 SM |
| Waves Per SM | <1 | 1.58 | 512 大量 SM 空转 |
| Duration | 394.6 us | 2.19 ms | |
| DRAM Throughput | 1.11% | 1.53% | 始终全闲，非 memory-bound |
| Compute (SM) | 39.12% | 56.25% | 放大后利用率↑ |
| L1/TEX Throughput | 49.32% | 73.96% | 放大后被压上来 |
| L2 Throughput | 7.48% | 9.23% | |
| Registers / thread | 168 | 168 | |
| **Theoretical Occupancy** | 18.75% | 18.75% | 被 168 寄存器卡死 |
| **Achieved Occupancy** | 6.46% | 14.53% | 放大后逼近理论上限 |
| Active Warps / SM | 4.14 | 9.30 | |
| shared load bank conflict | 1.5-way | 1.5-way | padding 在 A100 同样有效 |

### A100 上的瓶颈画像（与 T4 不同）

```text
T4：  L1/TEX 满 + occupancy 满(100%) → shared/L1 bound
A100：DRAM 全闲(1.5%) + L1/TEX 74% + occupancy 仅 14.5% → 寄存器卡死 occupancy
```

两层原因：

1. **理论 occupancy 只有 18.75%**：168 寄存器/线程 → Block Limit Registers = 6，每 SM 最多驻留 6 个 block；block=64 线程=2 warp，6×2=12 warp = 64 上限的 18.75%。这是 register tiling 的固有代价。
2. **小规模实测更低（512 → 6.46%）**：512 的 grid 只有 64 个 block，A100 有 108 个 SM，超过一半 SM 分不到 block；放大到 2048（1024 个 block，Waves/SM 1.58）后实测回到 14.53%，逼近理论上限。

### 真实性能（无 profiling，A100）

| 版本 | 2048 GFLOPS | 占 A100 峰值(~19.5T) |
| --- | --- | --- |
| shared baseline | ~4194 | ~21% |
| 2D + padding | ~11313 | ~58% |

> 对比 T4：同一个 2D kernel T4 只到峰值 16%，A100 到 58%。A100 SM 多、带宽大，register tiling 的并行度/带宽收益放得更开。

### 下一步（Day 5 参数实验方向）

```text
A100 的天花板是寄存器卡死 occupancy(18.75%)，不是访存。
→ 扫 TM/TN（8x8 → 4x4 降寄存器）、BK、BM/BN，
  在「寄存器↓ occupancy↑」与「单线程复用↓」之间找 A100 最优点。
```

### 面试口述（A100）

> 把同一个 2D tiling kernel 迁到 A100 后重新 profile，瓶颈画像完全变了。T4 上是 L1/TEX 和 occupancy 都打满、卡在 shared 带宽；A100 上 DRAM 只有 1.5%、occupancy 只有 14.5%。根因是每线程 168 个寄存器，把理论 occupancy 卡死在 18.75%。我还做了个规模对照：512 时 grid 只有 64 个 block，填不满 108 个 SM，实测 occupancy 掉到 6.5%；放大到 2048 后 grid 1024 个 block，occupancy 回到 14.5%、compute 利用率升到 56%。所以 A100 上的优化方向不是访存，而是降寄存器换 occupancy，这就是下一步参数实验要扫的。
