# Reduce（归约求和）

> 第一个练习算子。目标：把一个长数组求和，逐步从 naive 优化到 warp shuffle 版本。

## 优化阶梯（建议自己逐版实现）

```text
v1  naive：每个 block 内 tree reduction（interleaved addressing，有 warp divergence）
v2  sequential addressing：消除 bank conflict 和 divergence
v3  first add during load：grid-stride 累加，减少一半空闲线程
v4  warp shuffle：最后 32 个元素用 __shfl_down_sync，无 shared memory
v5  完全 warp-level + grid-stride：每 block 一个值，再二次归约
```

## 关键问题（实现时自问）

```text
1. reduce 是 memory-bound 还是 compute-bound？
   -> 几乎纯 memory-bound，理论上限 = 数组字节数 / DRAM 带宽

2. interleaved addressing 为什么慢？
   -> warp divergence + shared memory bank conflict

3. warp shuffle 为什么快？
   -> 寄存器间直接交换，不走 shared memory，无需 __syncthreads

4. 怎么做最终的跨 block 归约？
   -> 二次 kernel / atomicAdd / grid sync
```

## 完成标准

```text
[ ] CPU reference 校验通过（注意 float 累加误差，用相对误差）
[ ] benchmark：N = 1<<20, 1<<24, 1<<26
[ ] 记录 GB/s 并和 DRAM 峰值带宽对比
[ ] ncu 看 dram throughput / achieved occupancy
[ ] 一段口述：reduce 优化思路
```

## 参考

```text
week03_parallel/reduction_sum_full/   <- 你之前写过的版本，可对照
NVIDIA "Optimizing Parallel Reduction in CUDA"（Mark Harris 经典 slides）
```
