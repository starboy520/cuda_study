# 02 性能指标、Scaling 与 Roofline

## 1. Speedup

```text
speedup = baseline_time / optimized_time
```

必须说明 baseline 是：

- CPU 单线程。
- CPU 多线程。
- Naive GPU。
- 工业库。
- 上一个版本。

## 2. 有效带宽

Copy kernel：

```text
读 N bytes + 写 N bytes = 2N bytes
effective bandwidth = 2N / time
```

Transpose 同样至少读写各一次。

这是算法请求的有效字节，不一定等于硬件实际 DRAM 事务字节。

## 3. GFLOPS

GEMM `M x K` 乘 `K x N`：

```text
约 2 * M * N * K FLOP
GFLOPS = FLOP / seconds / 1e9
```

要明确 FMA 是按 2 FLOP 计。

## 4. Arithmetic Intensity

```text
AI = FLOP / bytes moved from target memory level
```

“bytes”必须说明针对 DRAM、L2、shared 还是其他层次。Roofline 常使用 DRAM
字节作为第一层模型。

Vector add：

```text
1 FLOP
约 12 bytes
AI 约 1/12 FLOP/byte
```

通常 memory-bound。

## 5. Roofline

性能上限：

```text
min(峰值计算吞吐, AI * 峰值内存带宽)
```

拐点：

```text
ridge point = peak FLOPS / peak bandwidth
```

AI 低于拐点通常受带宽上限约束，高于拐点才更可能受计算上限约束。

Roofline 是上限模型，不保证 kernel 能达到线。

## 5.1 用 T4 的真实数字算一遍 Roofline

抽象公式看完就忘，代入 **Tesla T4** 的规格走一遍，印象会深很多。T4 关键峰值（近似，
以本机实测为准）：

```text
FP32 峰值算力    ≈ 8.1 TFLOP/s   = 8100 GFLOP/s
显存带宽         ≈ 320 GB/s
```

先求**拐点（ridge point）**——AI 超过它才可能 compute-bound：

```text
ridge point = peak FLOPS / peak BW = 8100 / 320 ≈ 25 FLOP/byte
```

也就是说：在 T4 上，**只有每搬 1 字节能干 25 次以上浮点运算的 kernel，才有机会被算力
限制**；否则一律是 memory-bound。把三个常见 kernel 代进去：

| kernel | AI（FLOP/byte，近似）| 在 T4 上 | 上限性能 |
|---|---|---|---|
| vector add | 1 FLOP / 12 byte ≈ 0.08 | 远低于 25 → **memory-bound** | `0.08 × 320 ≈ 26 GFLOP/s`（带宽墙）|
| transpose | 0（纯搬运）| → **memory-bound** | 看带宽：目标逼近 320 GB/s |
| 朴素 GEMM | 每元素 2 FLOP，但反复从 DRAM 读 | 仍偏低 → 多半 memory-bound | 需靠 shared tiling 抬高 AI |
| 分块 GEMM | tiling 后复用高，AI 可超 25 | → 有望 **compute-bound** | 逼近 8100 GFLOP/s |

**这张表的实战意义**：它直接告诉你"优化该往哪个方向使劲"。

```text
AI 在拐点左边（如 vector add / transpose）：
  -> 性能被带宽锁死，堆计算没用
  -> 优化方向：合并访问、减少数据搬运、提高 AI（如 GEMM 的 tiling）

AI 在拐点右边（如分块 GEMM）：
  -> 才轮到关心指令效率、tensor core、FMA 利用率
```

**GEMM 为什么靠 tiling 就能从带宽墙逃到算力墙？** 朴素 GEMM 每算一个输出都从 DRAM 重新
读整行整列，AI 很低；shared memory 分块让一块数据**读一次、复用 tile 宽度次**，搬运字节
数大降而 FLOP 不变，AI 被抬到拐点右边——这就是 GEMM 优化的核心，也是面试必考点。

> 提醒：拐点 25 是 FP32 的。换数据类型峰值就变——T4 的 FP16/Tensor Core 算力高得多，
> 拐点右移，对 AI 的要求更高（见第 6 节"不同数据类型有不同计算峰值"）。

## 6. 实测 Roofline

使用厂商峰值可做规格 Roofline；更可靠的机器分析可以使用：

- 实测内存带宽。
- 实测计算吞吐 microbenchmark。
- Nsight Compute 的 SpeedOfLight 分析。

不同数据类型有不同计算峰值。

## 7. Amdahl

固定问题规模，串行比例为 `s`：

```text
speedup <= 1 / (s + (1-s)/p)
```

即使并行部分无限快，上限也是 `1/s`。

## 8. Gustafson

当处理器增加时扩大问题规模，串行部分占比可能降低。Gustafson 更适合解释
weak scaling 的价值。

## 9. Strong 与 Weak Scaling

Strong scaling：

```text
固定总问题，增加资源
```

Weak scaling：

```text
每个资源工作量固定，资源和总问题一起增加
```

单 GPU block 数变化也能观察饱和，但正式 scaling 常用于多 GPU/HPC。

## 10. 实践

1. 为 memory_access 算 logical GB/s。
2. 为 transpose 算有效 GB/s。
3. 为 naive GEMM 算 GFLOPS 和 AI。
4. 使用 T4 规格与实测带宽画简单 Roofline。
5. 解释为什么 transpose shared 仍远低于规格峰值。

## 11. 资料映射

- Best Practices Guide：Performance Metrics、Scaling。
- Roofline Model 论文与 Nsight Compute SpeedOfLight。

