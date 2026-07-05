# cp.async 双缓冲 vs 教学版 —— ncu stall 对比

> 对象：`week04_attention/tiled_attention.cu`（Day4 基线）vs `tiled_attention_pipelined.cu`（Day5 cp.async 双缓冲）
> 硬件：A100 80GB PCIe（sm_80）
> 采集：`ncu --launch-skip 5 --launch-count 1`（两个二进制里 N=128 D=64 causal=0 都是第 6 次启动，保证同 shape 公平对比）

## 采集命令

```bash
# 编译两个版本
nvcc -O3 -std=c++17 -arch=sm_80 tiled_attention.cu           -o tiled_attention
nvcc -O3 -std=c++17 -arch=sm_80 tiled_attention_pipelined.cu -o tiled_attention_pipelined

# warp stall + SoL 概览
ncu --launch-skip 5 --launch-count 1 --section WarpStateStats --section SpeedOfLight ./tiled_attention
ncu --launch-skip 5 --launch-count 1 --section WarpStateStats --section SpeedOfLight ./tiled_attention_pipelined

# 精确 stall 指标
ncu --launch-skip 5 --launch-count 1 --metrics \
  smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
  smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
  smsp__average_warp_latency_per_inst_issued.ratio,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed ./<bin>
```

## 结果（N=128 D=64 causal=0）

| 指标 | Day4 基线 | Pipelined | 变化 |
|------|----------|-----------|------|
| **long_scoreboard stall**（等 global 访存） | **6.40** | **0.03** | **↓99.5%** ⭐ |
| short_scoreboard stall（等 shared） | 0.60 | 0.59 | ≈ 不变 |
| warp latency / inst issued | 15.15 cyc | 6.46 cyc | ↓57%（2.35×） |
| SM Throughput | 7.47% | 16.79% | ↑2.25× |

## 怎么读

- **long_scoreboard**：warp 卡在"等一个 global load 结果"上的周期。Day4 里是头号瓶颈（6.4 cyc，占 15.15 的 40%+）。pipelined 用 `cp.async` 提前把下一块 K/V 预取到 shared，真正要用时数据已就位 → 这个 stall 塌到 0.03，**降 99.5%**。这就是"延迟被藏掉"的量化。
- **对照组 short_scoreboard 不变**（0.60→0.59）：没动 shared 用法，所以等 shared 的 stall 保持。一降一稳，精确证明**藏的是 global 访存延迟，而非别的**。
- 连锁收益：每指令等待 15.15→6.46 cyc（少的约 8.7 cyc 主要就是那 6.4 cyc long scoreboard），SM 吞吐翻倍。

## 诚实的 caveat

- 这个 launch 只有 N=128 个 block，**填不满 A100**（ncu 提示 0.13~0.30 wave）。所以**墙钟没有 2× 加速**——大量 SM 空闲，瓶颈变成占用率不足。
- 但 stall **组成**的变化真实清晰，证明 cp.async 双缓冲机制有效。要把它变成大幅墙钟加速，需同时解决占用率（多 query tile 填满 GPU）——属 Tensor Core 专项。

## 一句话（面试话术）

> 我用 ncu 对比双缓冲前后：long_scoreboard（global 访存等待）从每指令 6.4 cycle 降到 0.03，几乎归零，而 short_scoreboard（shared 等待）保持 0.6 不变——一降一稳，精确证明 cp.async 预取把 global 访存延迟藏进了计算里，warp 平均等待从 15 cycle 砍到 6.5，SM 吞吐翻倍。小 N 下墙钟没大改是因为 grid 填不满 GPU，瓶颈转成了占用率。
