# 03 Occupancy、分歧与延迟隐藏

## 1. Occupancy

```text
occupancy =
active warps per SM / maximum warps per SM
```

它描述驻留并发，不直接描述：

- Warp 是否可运行。
- 内存带宽是否饱和。
- 指令效率。
- Cache 命中。

## 2. 限制因素

每个 SM 有限：

- Threads/warps。
- Blocks。
- Registers。
- Shared memory。

一个 block 的资源使用决定同时能驻留多少 block。

## 3. 寄存器压力

每 thread 寄存器增加：

```text
可能减少 spilling
可能提高 instruction-level parallelism
也可能降低 resident warps
```

强制降低寄存器上限可能引入 local-memory spill，反而更慢。

使用：

```bash
nvcc -Xptxas=-v ...
```

和 Nsight Compute 一起判断。

## 4. Shared Memory 压力

更大 tile 可增加复用，但也消耗 shared memory，减少每 SM block 数。

Tile 调参必须同时看：

- 数据复用。
- Occupancy。
- Barrier。
- Bank conflict。
- 边界浪费。

## 5. Warp Divergence

```cpp
if (predicate) {
  pathA();
} else {
  pathB();
}
```

同一 warp 中 lane 走不同路径时，路径可能分阶段执行，部分 lane 被 predicate
关闭。

要理解代价，先记住 warp 的硬性约束：一个 warp 的 32 条 lane **共用一个程序
计数器**，同一时刻只能执行**一条**指令。当 lane 在 `if/else` 上分裂，硬件没法
让两半同时各走各的，只能**先执行 `pathA`（关闭走 else 的 lane），再执行
`pathB`（关闭走 if 的 lane）**——两条路径**串行**跑完，被关闭的 lane 在对应
阶段空转。

```text
无分歧（32 lane 同走 A）：  执行 A          = 1 份时间
两路分歧（部分 A 部分 B）： 执行 A + 执行 B = 最多 2 份时间，且总有 lane 在空转
```

最坏情况是一个 32 路 `switch`，每条 lane 走不同分支：32 条路径全部串行，这次
执行慢约 **32 倍**，任一时刻只有 1/32 的 lane 在干活。这就是 divergence 的本质
代价——不是"分支判断慢"，而是**并行度被路径数量除掉**。

不同 warp 走不同路径不叫 warp divergence（它们本就独立调度），代价为零。所以
优化方向是让分歧**发生在 warp 之间而非 warp 内部**：例如按条件排序数据、或让
同一 warp 的 32 条 lane 尽量走同一分支。

## 6. 分歧不等于所有 if 都慢

- Predicate 对整个 warp 一致，没有分歧。
- 很短分支可能被 predication。
- 边界分支只影响少数 warp。
- 消除分支可能增加更多指令。

必须查看实际控制流和 source counters。

## 7. Stall

Warp 可能因以下原因无法发射：

- 等待 memory。
- 等待依赖结果。
- Barrier。
- 执行管线繁忙。
- 没有合格 warp。

不要看到一个 stall reason 高就立刻修改代码。它可能是主要问题，也可能是其他
瓶颈的结果。

## 8. Occupancy API

```cpp
cudaOccupancyMaxActiveBlocksPerMultiprocessor(...)
cudaOccupancyMaxPotentialBlockSize(...)
```

API 可计算资源约束下的理论驻留，但仍需实际 benchmark。

## 9. 调参流程

1. 正确性固定。
2. 扫描 block size。
3. 记录时间、寄存器、shared memory、occupancy。
4. 对最好和最差版本做 profiler 对比。
5. 解释因果，不只保留最快数字。

## 10. 面试问题

- Occupancy 100% 为什么不保证最快？
- Register 使用多为什么可能既有利又有害？
- Divergence 的作用范围是什么？
- 为什么 `block=1024` 合法但常不是默认最佳？

## 11. 资料映射

- Best Practices Guide：Execution Configuration、Control Flow。
- Programming Guide：Occupancy、SIMT Execution。

