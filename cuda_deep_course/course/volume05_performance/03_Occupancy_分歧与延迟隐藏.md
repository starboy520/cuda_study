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

## 1.5 为什么需要 occupancy：延迟隐藏与 Little 定律

这一节回答全章最根本的问题：**occupancy 高到底有什么用？** 答案是——**用足够多的
warp 把访存/指令延迟藏起来，让计算单元别空转**。GPU 没有大乱序、没有大分支预测，它
隐藏延迟的唯一武器就是“换一个 warp 接着干”。

### 核心机制：stall 了就切 warp

一条 global load 大约要 **400~800 个时钟周期**才能拿到数据。CPU 会用乱序执行和大
cache 硬扛这段延迟；GPU 的做法完全不同：

```text
warp A 发一条 load -> 数据没回来，A 进入 stall
  -> 调度器立刻切到 warp B 发指令
     -> B 也 stall -> 切到 warp C ...
        -> 等轮回到 A 时，A 的数据可能已经回来了
```

只要**随时有别的 warp 可发射**，那 400 周期延迟就被其他 warp 的有效工作填满了，对外
表现为“延迟被隐藏”。如果活跃 warp 太少，轮一圈回来数据还没到，SM 就只能干等——这就是
latency-bound。

### 用 Little 定律量化：到底需要多少 warp

Little 定律说：**在途请求数 = 延迟 × 吞吐率**。套到延迟隐藏上：

```text
需要的在途并发 = 要隐藏的延迟(周期) × 每周期要发射的指令数
```

代入一个粗略例子（数量级感受即可）：

```text
访存延迟 L        ≈ 400 周期
想让访存单元每周期都有 1 个请求在途（不空闲）
=> 需要约 400 个独立的访存请求同时在飞
```

T4 每个 SM 最多驻留 **32 个 warp**。如果每个 warp 同一时刻只有 1 个独立访存请求在飞，
那 32 个 warp 只能提供 32 个在途请求，**离 400 差得远**——这正是很多 memory-bound
kernel 即便 occupancy 拉满也打不到带宽峰值的原因。两条出路：

```text
提高 occupancy(TLP)：更多 warp -> 更多在途请求   <- occupancy 的价值就在这
提高 ILP：           每个 warp 一次发多个独立 load（循环展开、float4）
                     -> 单 warp 也能贡献多个在途请求
```

**这就解释了两个看似矛盾的现象**：

- 为什么 occupancy 重要——warp 是隐藏延迟的“燃料”，太少就藏不住。
- 为什么 occupancy 不必 100%——只要在途并发够盖住延迟就行；Volkov 的经典反例用很高
  的 ILP（每线程多个独立操作）在低 occupancy 下照样跑满，因为 `TLP × ILP` 的乘积才是
  真正的在途并发量。

> 一句话面试答法：occupancy 的作用是提供足够的并发 warp 来隐藏访存/指令延迟；够用即
> 可，多余的 occupancy 不再带来收益，此时瓶颈通常已转移到带宽或指令吞吐。

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

## 4.5 手算 occupancy：用 T4 规格走一遍

光说“寄存器多了 occupancy 会降”太抽象。下面用 **Tesla T4（Turing）单个 SM 的真实
上限**手算一遍，你就能自己判断瓶颈是谁，而不必每次都开 profiler。T4 每个 SM 的硬限制：

```text
寄存器       65536 个（32-bit）
最大线程     1024 / SM
最大 warp    32 / SM
最大 block   16 / SM
shared mem   64 KB / SM（opt-in 上限）
```

**occupancy 的算法**：分别算出“寄存器/shared/线程/block 各自允许多少个 block 驻留”，
取**最小值**就是真正能驻留的 block 数，再换算成 warp 占比。

### 例 1：32 寄存器/线程，block = 256

```text
按线程上限：   1024 / 256              = 4 block
按 warp 上限： 32 / (256/32) = 32 / 8  = 4 block
按寄存器：     65536 / (32 × 256)      = 65536 / 8192  = 8 block
按 block 上限： 16                      = 16 block
-------------------------------------------------------------
取最小 = 4 block  -> 4 × 256 = 1024 线程 = 32 warp = 32/32 = 100% occupancy
```

此时**线程/warp 数是限制因素**，寄存器还很宽裕。

### 例 2：把寄存器压到 72/线程（其余不变）

```text
按寄存器：     65536 / (72 × 256) = 65536 / 18432 ≈ 3.55  -> 向下取整 = 3 block
其余仍是 4 block
-------------------------------------------------------------
取最小 = 3 block  -> 3 × 256 = 768 线程 = 24 warp = 24/32 = 75% occupancy
```

**寄存器一涨，它就成了限制因素**，occupancy 从 100% 掉到 75%。这正是“register pressure
降低 resident warps”的具体数字。用 `nvcc -Xptxas=-v` 能看到每线程寄存器数，自己代进去
就知道会掉到几档。（注意真实硬件还有寄存器分配粒度，手算是上界，精确值以 profiler 为准。）

### 例 3：shared memory 成为限制因素

转置 tile `float[32][33]` 每 block 用 `32×33×4 ≈ 4.2 KB`：

```text
按 shared： 64 KB / 4.2 KB ≈ 15 block  -> 不是瓶颈（还没到 16 block 上限）
```

但若把 tile 放大到每 block 用 48 KB：

```text
按 shared： 64 / 48 = 1 block  -> 只能驻留 1 个 block！
```

大 tile 提高了数据复用，却把 occupancy 砸到地板——**这就是第 4 节说的“tile 调参要同时
看复用和 occupancy”的量化版本**。

> 手算的价值：拿到一个 kernel，先用 `-Xptxas=-v` 看寄存器/shared 用量，代进上面三组
> 除法，30 秒就能判断 occupancy 卡在哪个资源上，再决定要不要动 block size 或 tile 大小。

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

手算适合快速估计，运行时让 CUDA 帮你算更准、更省事。两个核心 API：

```cpp
cudaOccupancyMaxActiveBlocksPerMultiprocessor(...)  // 给定 block size，每 SM 能驻留几个 block
cudaOccupancyMaxPotentialBlockSize(...)             // 直接建议一个 occupancy 最优的 block size
```

### 可直接运行的示例

```cpp
__global__ void myKernel(const float* in, float* out, int n) { /* ... */ }

void reportOccupancy() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // 1) 让 CUDA 推荐一个 occupancy 友好的 block size
    int minGridSize = 0, blockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, myKernel, /*dynamicSMem=*/0, /*blockSizeLimit=*/0);

    // 2) 在该 block size 下，每个 SM 能驻留多少 block
    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM, myKernel, blockSize, /*dynamicSMem=*/0);

    // 3) 换算成理论 occupancy（活跃 warp / SM 上限 warp）
    int activeWarps   = maxBlocksPerSM * blockSize / prop.warpSize;
    int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    float occupancy   = (float)activeWarps / maxWarpsPerSM;

    printf("建议 blockSize = %d\n", blockSize);
    printf("每 SM 驻留 block = %d\n", maxBlocksPerSM);
    printf("理论 occupancy   = %.0f%%\n", occupancy * 100.0f);
}
```

要点：

- `cudaOccupancyMaxPotentialBlockSize` 适合“懒人起步”——不确定 block size 时先用它给的值，
  再围绕它做 scan。
- 若 kernel 用了**动态 shared memory**，把每 block 字节数传进 `dynamicSMem` 参数，否则
  算出来偏乐观。
- 这套数字是**理论上界**，不含尾部效应、负载不均、实际 stall。最终仍要 benchmark + 用
  Nsight Compute 的 achieved occupancy 对照（理论高但 achieved 低，往往是 grid 太小或负载
  不均，见第 05 章判读表）。

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

