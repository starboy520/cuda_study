# Roofline 分析：GEMM naive → shared → 2D register tiling

> 平台：A100，FP32，方阵 M=N=K=2048，tile BM=BN=64
> 结论一句话：naive 是重度 memory-bound；shared/2D 在 DRAM 层已是 compute 侧，但实测差 2.7×，瓶颈转移到 shared/occupancy，Roofline 看不出。

## 固定前提

```text
FP32 峰值算力  P = 19.5 TFLOPS = 19.5e12 FLOP/s
显存带宽       B = 1935 GB/s   = 1935e9 byte/s
FLOP（三版相同）= 2*M*N*K = 2*2048^3 = 17,179,869,184
拐点 AI* = P/B = 19.5e12 / 1935e9 ≈ 10 FLOP/byte
```

## 三版本 FLOP / bytes / AI

| 版本 | global bytes | AI = FLOP/bytes | vs 拐点(10) | bound（DRAM层） | 实测 GFLOPS(2048) |
|------|--------------|-----------------|-------------|------------------|--------------------|
| naive  | 2*M*N*K*4 = 68.7 GB | 0.25 | 远小于 | memory-bound | （未测） |
| shared | 268,435,456*4 ≈ 1.07 GB | 16 | 大于 | compute 侧 | ~4194 |
| 2D+pad | ≈ 1.07 GB（同 shared） | 16 | 大于 | compute 侧 | ~11313 (8x8) / 11382 (8x4) |

计算依据：
```text
naive ： 每个输出读 2K 个元素，M*N 输出 → bytes = 2*M*N*K*4，AI=1/4（与规模无关）
shared： A 读 M*K*(N/BN)=134,217,728；B 读 M*N*(K/BM)=134,217,728
         总 268,435,456 元素 *4 ≈ 1.07GB；比 naive 降 64×；AI=16
2D    ： DRAM 流量与 shared 相同（tile 搬运一样），AI 同为 16
```

## 三个结论

### 1. naive → shared：AI 为什么暴涨？
naive 每用一次 A/B 都从 global 读，无复用 → bytes 巨大、AI=0.25。shared tiling 把 tile 搬进 shared 复用 BM/BN 次，DRAM bytes 降 64×，AI 从 0.25 跳到 16，越过拐点 10，从带宽斜线爬上算力屋顶。

### 2. shared vs 2D：global AI 相同，为何 2D 快 2.7×？
两版从 DRAM 搬的 tile 一样，**global AI 都是 16，Roofline 在 DRAM 层无法区分**。但 shared 版每个 thread 每算一个输出都反复读 shared，瓶颈在 shared/L1 吞吐与指令；2D register tiling 让一个 thread 算 TM×TN，用 regM×regN 外积，把 shared 读取次数砍掉数倍 → shared/指令压力降，实测 4194→11313。代价是寄存器更多、occupancy 略降，但收益更大。

### 3. A100 上下一步往哪优化？
shared/2D 已是 compute/occupancy 受限，不是 DRAM 受限。继续提速靠：vectorized load (float4)、double buffering (cp.async)、参数调优(occupancy/寄存器平衡)、warp tiling，最终上 Tensor Core。

## Roofline 的盲区（高级点）
Roofline 只刻画 DRAM 层。当两版 DRAM bytes 相同(shared vs 2D)，AI 一样、落点一样，却实测差 2.7×——必须用 shared 层访存账或 ncu(occupancy/shared throughput) 才能解释。

## 面试口述
"我用 AI=FLOP/bytes 对拐点判 bound。naive AI=0.25 远低于 A100 拐点 10，重度 memory-bound；tiling 复用把 DRAM bytes 降 64× → AI=16 越过拐点，进 compute 侧。但 shared 和 2D 的 global AI 都是 16，Roofline 区分不了，实测却差 2.7×，说明瓶颈已从 DRAM 转到 shared/occupancy——这时改看 ncu 的 shared throughput 和 occupancy，下一步用向量化、double buffering、Tensor Core 继续逼近峰值。"
