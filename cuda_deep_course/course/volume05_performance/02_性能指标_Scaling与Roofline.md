# 02 性能指标、Scaling 与 Roofline

这一章专门解决一个问题：

> 一个 CUDA kernel 跑完以后，我到底该怎么看它快不快？下一步该往哪里优化？

初学性能分析时最容易犯的错是：

```text
只看时间。
```

时间当然重要，但只看时间会让你不知道：

```text
它是被显存带宽卡住？
还是被计算单元卡住？
还是 launch / 同步 / 数据传输卡住？
优化方向应该是减少访存、提高复用，还是用 Tensor Core？
```

所以本章按下面顺序慢慢来：

```text
时间
  -> speedup
  -> 有效带宽 GB/s
  -> GFLOPS
  -> 算术强度 AI
  -> Roofline
  -> scaling
```

## 0. 先建立一张“指标地图”

不同指标回答不同问题：

| 指标 | 回答的问题 | 常见单位 |
|---|---|---|
| Time | 这个 kernel 花了多久？ | ms / us |
| Speedup | 优化后比基线快几倍？ | x |
| Effective Bandwidth | 每秒搬了多少逻辑数据？ | GB/s |
| GFLOPS | 每秒做了多少浮点运算？ | GFLOP/s |
| Arithmetic Intensity | 每搬 1 byte 能做多少计算？ | FLOP/byte |
| Roofline | 理论上是带宽卡还是算力卡？ | 图 / 上限 |
| Scaling | 加更多资源后是否继续变快？ | strong / weak |

一个简单判断：

```text
搬数据为主的 kernel：
  重点看 GB/s。

浮点计算为主的 kernel：
  重点看 GFLOPS。

想判断优化方向：
  看 AI 和 Roofline。

想判断多 GPU / 多线程扩展：
  看 scaling。
```

## 0.1 单位先说清：GB、GiB、ms、秒

性能指标最怕单位混乱。

本章默认：

```text
1 GB = 1e9 bytes
1 GiB = 1024^3 bytes
1 ms = 1e-3 seconds
1 us = 1e-6 seconds
```

很多 GPU 性能报告使用：

```text
GB/s = bytes / seconds / 1e9
GFLOPS = FLOP / seconds / 1e9
```

注意：

```text
显存容量常用 GiB。
带宽和吞吐常用 GB/s。
```

例如：

```text
16 GiB 显存
320 GB/s 带宽
```

这两个单位不是同一个进制。

写报告时最好明确：

```text
logical bandwidth = 240 GB/s, using 1 GB = 1e9 bytes
```

这样不会和别人对不上。

## 1. Time：所有指标的起点

性能分析第一步永远是可靠计时。

CUDA kernel 是异步 launch 的，所以不能这样测：

```cpp
auto start = now();
kernel<<<grid, block>>>();
auto end = now();
```

这测到的大多是 launch 开销，不是 kernel 真正执行时间。

CUDA kernel 推荐用 CUDA Event：

```cpp
cudaEventRecord(start);
kernel<<<grid, block>>>();
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

你要记住：

```text
后面所有 GB/s、GFLOPS、speedup 都依赖这个 time。
时间测不准，后面全错。
```

## 2. Speedup：快了几倍

公式：

```text
speedup = baseline_time / optimized_time
```

例如：

```text
naive_time = 10 ms
optimized_time = 2 ms

speedup = 10 / 2 = 5x
```

这说明优化版比 naive 快 5 倍。

## 3. Speedup 一定要说明 baseline

“快 10 倍”没有意义，除非说清楚和谁比。

baseline 可以是：

```text
CPU 单线程
CPU 多线程
Naive GPU
上一个 GPU 版本
cuBLAS / cuDNN / Thrust 这种工业库
另一个硬件平台
```

同一个结果，含义完全不同：

```text
比 CPU 单线程快 100x：
  可能只是 GPU 并行了。

比 naive GPU 快 5x：
  说明 CUDA 优化有效。

达到 cuBLAS 的 70%：
  说明这个 GEMM 实现已经很强。
```

写报告时建议格式：

```text
baseline: naive GPU kernel
optimized: shared-memory tiled kernel
speedup: 3.2x
input: M=N=K=2048
GPU: Tesla T4
```

## 4. 有效带宽：搬数据的速度

很多 CUDA kernel 本质上主要在搬数据，比如：

```text
copy
vector add
transpose
memset
很多 elementwise kernel
```

这类 kernel 光看 GFLOPS 没意义，因为它们本来就没多少浮点运算。

要看：

```text
effective bandwidth
```

也就是：

```text
每秒逻辑上搬了多少数据。
```

关键词是：

```text
逻辑上
```

有效带宽不是硬件真实 DRAM 事务带宽，而是你按算法认为“应该读写”的字节数除以时间。

它适合用来比较同一个算法不同实现：

```text
transpose naive vs transpose shared
copy version A vs copy version B
memory_access contiguous vs strided
```

但如果要看硬件实际搬了多少，要用 Nsight Compute 里的 memory throughput / DRAM bytes。

## 5. Copy kernel 的有效带宽

假设：

```cpp
out[i] = in[i];
```

每个元素：

```text
读 in[i]
写 out[i]
```

如果一共有 `N` 个 float：

```text
读 N * 4 bytes
写 N * 4 bytes
总逻辑数据 = 2 * N * 4 bytes
```

公式：

```text
effective_bandwidth = logical_bytes / time_seconds
```

换成 GB/s：

```text
GB/s = logical_bytes / time_seconds / 1e9
```

例子：

```text
N = 100,000,000 float
bytes_per_array = 100,000,000 * 4 = 400 MB
logical_bytes = 800 MB
time = 4 ms = 0.004 s

GB/s = 800 MB / 0.004 s = 200 GB/s
```

## 6. Vector Add 的有效带宽

代码：

```cpp
c[i] = a[i] + b[i];
```

每个元素：

```text
读 a[i] 4 bytes
读 b[i] 4 bytes
写 c[i] 4 bytes
```

所以：

```text
logical_bytes = 3 * N * sizeof(float)
```

如果你测到：

```text
N = 100,000,000
time = 5 ms
```

那么：

```text
logical_bytes = 3 * 100,000,000 * 4 = 1.2 GB
GB/s = 1.2 / 0.005 = 240 GB/s
```

这说明这个 kernel 每秒逻辑上完成了 240 GB 的读写。

## 7. Transpose 的有效带宽

矩阵转置：

```cpp
out[col][row] = in[row][col];
```

至少：

```text
读 input 一次
写 output 一次
```

所以：

```text
logical_bytes = 2 * width * height * sizeof(float)
```

如果 naive transpose 和 shared transpose 都做同样的逻辑工作，那么可以比较：

```text
谁的 GB/s 更高，谁更接近硬件带宽上限。
```

注意：

```text
有效带宽是“算法逻辑字节”。
硬件实际 DRAM 事务字节可能更多。
```

例如不合并访问会导致：

```text
逻辑上只写 4 bytes
硬件可能为了一个 cache line / memory transaction 搬更多 bytes
```

所以有效带宽低，通常说明：

```text
访问模式没有充分利用硬件带宽。
```

## 7.1 有效带宽 vs Nsight 里的 DRAM Throughput

这两个名字很像，但不是一回事。

有效带宽：

```text
logical_bytes / time
```

DRAM throughput：

```text
硬件实际从 DRAM 读写的字节 / time
```

举个极端例子：

```text
每个 thread 只需要 4 bytes，
但访问很分散，硬件为了这些访问搬了很多 cache line。
```

那么可能出现：

```text
有效带宽低
DRAM throughput 高
```

这表示：

```text
硬件很忙，但搬了很多没用数据。
```

这就是为什么 Nsight Compute 里还要看：

```text
memory efficiency
sector utilization
global load/store efficiency
```

初学时先记：

```text
有效带宽回答：算法有用工作做得多快？
DRAM throughput 回答：硬件内存系统实际有多忙？
```

## 8. GFLOPS：计算速度

GFLOPS 回答：

```text
每秒做了多少十亿次浮点运算？
```

先分清词：

```text
FLOP   = Floating-point Operation，一次浮点运算
FLOPS  = FLOP per Second，每秒多少次浮点运算
GFLOPS = Giga FLOPS，每秒 10 亿次浮点运算
```

公式：

```text
GFLOPS = total_FLOP / time_seconds / 1e9
```

它需要两个数：

```text
1. 这个 kernel 总共做了多少 FLOP。
2. 这个 kernel 跑了多少秒。
```

## 9. 怎么数 FLOP：从简单例子开始

### Vector add

```cpp
c[i] = a[i] + b[i];
```

每个元素：

```text
1 次浮点加法 = 1 FLOP
```

总 FLOP：

```text
N
```

但 vector add 通常不重点看 GFLOPS，因为它主要是搬数据。

### SAXPY

```cpp
y[i] = a * x[i] + y[i];
```

每个元素：

```text
1 次乘法 + 1 次加法 = 2 FLOP
```

总 FLOP：

```text
2 * N
```

### GEMM

```text
C[M,N] = A[M,K] x B[K,N]
```

结果有：

```text
M * N 个元素
```

每个元素：

```text
K 次乘法 + K 次加法 ≈ 2K FLOP
```

所以：

```text
total_FLOP = 2 * M * N * K
```

这是 GEMM 最常用的 FLOP 公式。

## 10. FMA 为什么算 2 FLOP

GPU 常用 FMA：

```cpp
acc += a * b;
```

硬件可能执行成：

```text
acc = a * b + acc
```

这包含：

```text
1 次乘法
1 次加法
```

所以：

```text
1 条 FMA = 2 FLOP
```

厂商标称的 FP32 峰值通常也按 FMA 算 2 FLOP。

如果你把 FMA 算 1 FLOP：

```text
你的 GFLOPS 会少一半。
和规格表峰值、Roofline 都对不上。
```

## 11. GEMM GFLOPS 手算例子

假设：

```text
M = N = K = 1024
time = 2 ms = 0.002 s
```

总 FLOP：

```text
2 * 1024 * 1024 * 1024
= 2,147,483,648 FLOP
≈ 2.15e9 FLOP
```

GFLOPS：

```text
2.15e9 / 0.002 / 1e9
= 1073.7 GFLOPS
```

如果 T4 FP32 峰值约 8100 GFLOPS，那么：

```text
1074 / 8100 ≈ 13.3%
```

说明：

```text
距离 FP32 峰值还很远。
```

但注意：这不直接告诉你该优化计算还是访存。这个要靠后面的 AI 和 Roofline。

## 12. Arithmetic Intensity：算术强度

算术强度简称 AI：

```text
AI = FLOP / bytes
```

读成一句人话：

```text
每搬 1 byte 数据，能做多少次浮点运算。
```

AI 高：

```text
数据搬进来以后被反复使用。
搬一次，算很多。
```

AI 低：

```text
搬了很多数据，只做一点点计算。
```

所以 AI 是判断：

```text
这个 kernel 更像“搬运工”，还是更像“计算工”。
```

## 13. Vector Add 的 AI

代码：

```cpp
c[i] = a[i] + b[i];
```

每个元素：

```text
FLOP:
  1 次加法 = 1 FLOP

bytes:
  读 a[i] = 4 bytes
  读 b[i] = 4 bytes
  写 c[i] = 4 bytes
  总共 = 12 bytes
```

所以：

```text
AI = 1 / 12 ≈ 0.083 FLOP/byte
```

这非常低。

意思是：

```text
每搬 12 bytes 才做 1 次浮点运算。
```

所以 vector add 通常是 memory-bound。

## 14. GEMM 的 AI 为什么可以很高

GEMM 的魔力是：

```text
A/B 的元素可以被很多 C 输出复用。
```

naive GEMM 没有好好复用，所以 AI 不高。

shared-memory tiled GEMM 会把一小块 A/B 读进 shared memory，然后被一个 block 内很多 thread 复用。

直觉：

```text
vector add:
  一个数据读进来，用一次。

GEMM tiling:
  一个 A/B 元素读进来，可能被一个 tile 里的多个输出使用。
```

FLOP 没变，但 DRAM bytes 变少：

```text
AI = FLOP / bytes
```

bytes 变少，AI 就上升。

这就是 GEMM 能从 memory-bound 走向 compute-bound 的根。

## 14.1 Naive GEMM 的 AI 近似怎么算

先用最朴素的角度估算。

每个 C 元素：

```text
计算:
  K 次乘法 + K 次加法 ≈ 2K FLOP

访存:
  读 A 的 K 个 float = 4K bytes
  读 B 的 K 个 float = 4K bytes
  写 C 一个 float    = 4 bytes
```

所以每个输出：

```text
FLOP ≈ 2K
bytes ≈ 8K + 4
```

当 K 很大时：

```text
AI ≈ 2K / 8K = 0.25 FLOP/byte
```

这就是为什么 naive GEMM 虽然 FLOP 很多，但仍可能被 DRAM 拖住：

```text
每个 thread 都反复从 DRAM 读 A/B。
数据没有被 block 充分复用。
```

Shared tiling 后：

```text
同一个 A/B tile 从 DRAM 读一次，
在 shared memory 中被很多输出复用。
```

FLOP 没变，DRAM bytes 下降，AI 上升。

注意：

```text
这是入门估算。
真实 AI 会受 cache、L2、编译器、边界、数据复用和实现方式影响。
```

## 15. bytes 必须说明是哪一层

AI 里的 bytes 不是永远同一个数。

你可以站在不同内存层看：

```text
DRAM bytes:
  从显存搬了多少。

L2 bytes:
  从 L2 搬了多少。

shared bytes:
  从 shared memory 搬了多少。
```

同一个 kernel：

```text
站在 DRAM 看，AI 可能很高。
站在 shared 看，AI 可能没那么高。
```

Roofline 入门版通常用：

```text
DRAM bytes
```

因为显存带宽通常是最稀缺的瓶颈。

写报告时建议说：

```text
AI = 0.083 FLOP/byte, counted at DRAM logical bytes.
```

这样别人知道你怎么算的。

## 16. Roofline：一张图判断优化方向

Roofline 是一张图，用来回答：

```text
这个 kernel 理论上更可能被带宽限制，还是被计算限制？
```

图的横轴：

```text
AI = FLOP/byte
```

图的纵轴：

```text
Performance = FLOP/s
```

图有两条“屋顶”：

```text
带宽屋顶：
  performance <= AI * memory_bandwidth

算力屋顶：
  performance <= peak_compute
```

最终上限是：

```text
performance <= min(peak_compute, AI * memory_bandwidth)
```

### 16.1 带宽屋顶到底是什么意思

先看这条：

```text
performance <= AI * memory_bandwidth
```

这条公式刚看会很抽象。

我们把单位写出来：

```text
AI:
  FLOP / byte

memory_bandwidth:
  byte / second
```

两者相乘：

```text
(FLOP / byte) * (byte / second)
= FLOP / second
```

也就是：

```text
每搬 1 byte 能做多少 FLOP
乘以
每秒最多能搬多少 byte

等于
每秒最多能支撑多少 FLOP
```

这就是“带宽屋顶”。

它不是说 GPU 算力只有这么多。

它说的是：

```text
如果数据只能这么快地从显存搬进来，
那么这些数据最多只能喂出这么多计算。
```

举一个很小的数字例子。

假设：

```text
AI = 2 FLOP/byte
memory_bandwidth = 100 byte/s
```

意思是：

```text
每搬 1 byte 数据，可以做 2 次浮点计算。
每秒最多搬 100 byte 数据。
```

那么一秒内，显存最多能提供多少计算需要的数据？

```text
2 FLOP/byte * 100 byte/s = 200 FLOP/s
```

所以：

```text
performance <= 200 FLOP/s
```

哪怕 GPU 的计算单元很强，比如理论上能做：

```text
800 FLOP/s
```

这个 kernel 也不可能跑到 800 FLOP/s。

因为数据喂不上。

这就是 memory-bound 的直觉：

```text
不是计算器不够强，
而是计算器等数据。
```

### 16.2 算力屋顶到底是什么意思

再看这条：

```text
performance <= peak_compute
```

这个更直接。

如果一块 GPU 对某种数据类型的峰值算力是：

```text
peak_compute = 800 FLOP/s
```

那不管数据喂得多好，kernel 理论上也不能超过：

```text
800 FLOP/s
```

因为计算单元最多一秒就只能做这么多计算。

这就是“算力屋顶”。

比如：

```text
AI = 20 FLOP/byte
memory_bandwidth = 100 byte/s
```

带宽能支撑的性能是：

```text
AI * bandwidth = 20 * 100 = 2000 FLOP/s
```

但是如果 GPU 算力峰值只有：

```text
peak_compute = 800 FLOP/s
```

最终还是：

```text
performance <= min(2000, 800) = 800 FLOP/s
```

这时候不是数据不够，而是计算单元已经到上限。

这就是 compute-bound 的直觉：

```text
数据已经够喂了，
但计算器自己吃不下更多。
```

### 16.3 为什么最终取 min

一个 kernel 同时受两个限制：

```text
限制一：数据搬运最多能支撑多少计算
限制二：计算单元最多能执行多少计算
```

所以最终上限一定是两者中更低的那个：

```text
performance <= min(AI * memory_bandwidth, peak_compute)
```

举例：

```text
AI * bandwidth = 200 FLOP/s
peak_compute = 800 FLOP/s
```

最终：

```text
performance <= 200 FLOP/s
```

因为数据不够。

另一个例子：

```text
AI * bandwidth = 2000 FLOP/s
peak_compute = 800 FLOP/s
```

最终：

```text
performance <= 800 FLOP/s
```

因为计算单元到顶了。

一句话：

```text
带宽屋顶：
  数据最多能喂出多少计算量。

算力屋顶：
  计算单元最多能吃下多少计算量。

最终性能上限：
  看谁更低。
```

## 17. Roofline 图怎么读

简化图：

```text
性能 GFLOP/s
  ^
  |
峰|                 ───────────────  算力屋顶 peak compute
值|               ╱
  |             ╱
  |           ╱  带宽屋顶 AI * bandwidth
  |         ╱
  |       ╱
  +──────┴──────────────────────────> AI FLOP/byte
        拐点
```

这张图可以分成两段看。

左边这段斜线：

```text
带宽屋顶 = AI * bandwidth
```

为什么是斜线？

因为 bandwidth 固定时，AI 越大，带宽能支撑的 FLOP/s 就越大。

假设：

```text
memory_bandwidth = 100 byte/s
```

那么：

```text
AI = 1  -> AI * bandwidth = 100 FLOP/s
AI = 2  -> AI * bandwidth = 200 FLOP/s
AI = 4  -> AI * bandwidth = 400 FLOP/s
AI = 8  -> AI * bandwidth = 800 FLOP/s
```

AI 变大，性能上限跟着变大，所以图上就是往右上方走的斜线。

右边这段横线：

```text
算力屋顶 = peak_compute
```

为什么是横线？

因为 GPU 的峰值算力是固定的。

假设：

```text
peak_compute = 800 FLOP/s
```

那么即使 AI 继续增加：

```text
AI = 8   -> 带宽可支撑 800 FLOP/s，最终上限 800 FLOP/s
AI = 16  -> 带宽可支撑 1600 FLOP/s，最终上限还是 800 FLOP/s
AI = 32  -> 带宽可支撑 3200 FLOP/s，最终上限还是 800 FLOP/s
```

因为计算单元最多只能做到 800 FLOP/s。

所以图上右边是一条水平线。

把数字放进表里：

```text
memory_bandwidth = 100 byte/s
peak_compute = 800 FLOP/s
```

| AI | 带宽能支撑的性能 AI*BW | 算力峰值 | 最终上限 |
|---:|---:|---:|---:|
| 1 | 100 FLOP/s | 800 FLOP/s | 100 FLOP/s |
| 2 | 200 FLOP/s | 800 FLOP/s | 200 FLOP/s |
| 4 | 400 FLOP/s | 800 FLOP/s | 400 FLOP/s |
| 8 | 800 FLOP/s | 800 FLOP/s | 800 FLOP/s |
| 16 | 1600 FLOP/s | 800 FLOP/s | 800 FLOP/s |
| 32 | 3200 FLOP/s | 800 FLOP/s | 800 FLOP/s |

所以这张图的意思是：

```text
左边：
  AI 太低。
  每搬 1 byte 做不了多少计算。
  GPU 大概率在等数据。

右边：
  AI 足够高。
  每搬 1 byte 可以做很多计算。
  数据才有机会把计算单元喂饱。
```

拐点左边：

```text
AI 太低。
带宽屋顶低于算力屋顶。
kernel 通常 memory-bound。
优化方向：减少访存、合并访问、提高复用、提高 AI。
```

拐点右边：

```text
AI 足够高。
算力屋顶成为上限。
kernel 才更可能 compute-bound。
优化方向：提高计算单元利用率、Tensor Core、减少指令开销、提升 occupancy/ILP。
```

## 18. 拐点 ridge point

拐点公式：

```text
ridge_point = peak_compute / peak_bandwidth
```

单位：

```text
(FLOP/s) / (byte/s) = FLOP/byte
```

它表示：

```text
AI 至少要达到多少，才有机会被算力限制。
```

如果 AI 小于拐点：

```text
带宽卡。
```

如果 AI 大于拐点：

```text
有机会算力卡。
```

注意是“有机会”，不是保证。

因为实际性能还会受：

```text
latency
occupancy
分支分歧
bank conflict
cache miss
指令 mix
同步
```

影响。

## 19. 用 T4 算一次 Roofline

用近似规格：

```text
T4 FP32 peak_compute ≈ 8.1 TFLOP/s = 8100 GFLOP/s
T4 memory_bandwidth ≈ 320 GB/s
```

拐点：

```text
ridge_point = 8100 / 320 ≈ 25 FLOP/byte
```

意思是：

```text
在 T4 上，一个 FP32 kernel 的 AI 要超过约 25 FLOP/byte，
才有机会成为 compute-bound。
```

## 20. 把常见 kernel 放到 T4 Roofline 上

| Kernel | AI 近似 | 和 25 比 | 判断 |
|---|---:|---:|---|
| vector add | 1 / 12 = 0.083 | 远小于 25 | memory-bound |
| copy | 0 | 远小于 25 | bandwidth benchmark |
| transpose | 0 | 远小于 25 | memory-bound |
| naive GEMM | 约 0.25 | 远小于 25 | 需要 tiling |
| tiled GEMM | 可以显著提高 | 有机会超过 | 可能 compute-bound |

对于 vector add：

```text
AI * bandwidth = 0.083 * 320 ≈ 26.6 GFLOP/s
```

这说明：

```text
哪怕 T4 有 8100 GFLOP/s FP32 峰值，
vector add 也不可能接近这个算力峰值。
因为它根本没有足够计算量。
```

所以别用 GFLOPS 评价 vector add。

应该用：

```text
GB/s
```

## 21. Roofline 对优化方向的指导

如果 kernel 在拐点左边：

```text
优先优化内存：
  合并访问
  减少重复读写
  shared memory tiling
  register tiling
  避免不必要的 global memory 往返
  改数据布局
```

如果 kernel 在拐点右边：

```text
优先优化计算：
  用 Tensor Core
  提高 FMA / MMA 利用率
  减少控制流和同步
  提高 ILP
  调整 tile size
  减少指令开销
```

一句话：

```text
Roofline 不是告诉你最终性能是多少。
Roofline 是告诉你下一步该往哪里努力。
```

## 21.1 怎么把“实测点”放到 Roofline 上

Roofline 不只是算理论线，还要把你的 kernel 实测点放上去。

一个点需要两个坐标：

```text
x 坐标 = AI = FLOP / bytes
y 坐标 = measured performance = FLOP / second
```

也就是：

```text
(AI, measured_GFLOPS)
```

例如 vector add：

```text
AI = 0.083 FLOP/byte
measured = 20 GFLOPS
```

T4 带宽上限预测：

```text
AI * BW = 0.083 * 320 = 26.6 GFLOPS
```

那么：

```text
20 / 26.6 ≈ 75%
```

说明：

```text
它已经达到带宽 roof 的 75% 左右。
继续优化计算指令意义不大，
应该看访存合并、对齐、事务效率。
```

如果某个 kernel：

```text
AI 很高，但 measured_GFLOPS 离 peak_compute 很远
```

就要去看：

```text
occupancy
stall reason
instruction mix
Tensor Core 是否用上
shared bank conflict
同步开销
```

所以 Roofline 的实际用法是：

```text
1. 算 AI。
2. 算 measured GFLOPS。
3. 算同一 AI 下的 roof 上限。
4. 看离 roof 多远。
5. 决定下一步用 Nsight Compute 看什么。
```

## 22. 实测 Roofline 与规格 Roofline

两种 Roofline：

```text
规格 Roofline:
  用厂商规格峰值算。

实测 Roofline:
  用你自己 microbenchmark 测出来的 peak bandwidth 和 peak compute。
```

规格 Roofline 的优点：

```text
容易算。
适合建立上限概念。
```

缺点：

```text
真实应用很难达到规格峰值。
```

实测 Roofline 更实用，因为：

```text
它用的是你这台机器、这个环境、这个编译器下能达到的实际上限。
```

Nsight Compute 的 Speed Of Light 分析也会帮助你判断：

```text
离 SM 峰值多远
离 memory 峰值多远
哪个 pipe 或 memory level 更忙
```

## 22.1 多层 Roofline：DRAM、L2、Shared

入门 Roofline 通常只画 DRAM roof。

更深入时，可以画多层：

```text
DRAM Roofline
L2 Roofline
Shared Memory Roofline
Compute Roofline
```

为什么需要多层？

```text
一个 kernel 可能已经不受 DRAM 限制，
但受 L2 或 shared memory 带宽限制。
```

例如 optimized GEMM：

```text
DRAM 复用已经很好。
下一步瓶颈可能变成 shared memory 读、register pressure、Tensor Core 利用率。
```

这时只看 DRAM Roofline 会觉得：

```text
AI 很高，应该 compute-bound。
```

但实际可能是：

```text
shared memory bandwidth 或 instruction issue 成为瓶颈。
```

所以初学阶段先会 DRAM Roofline；进阶阶段再结合 Nsight Compute 看多层 memory workload。

## 23. 不同数据类型有不同 Roofline

同一块 GPU 对不同数据类型峰值不同：

```text
FP32 CUDA Core peak
FP16 Tensor Core peak
BF16 Tensor Core peak
TF32 Tensor Core peak
FP8 Tensor Core peak
```

所以：

```text
同一个 kernel，如果从 FP32 改成 FP16 Tensor Core，
peak_compute 会变大很多。
```

拐点：

```text
ridge_point = peak_compute / bandwidth
```

peak_compute 变大，拐点也变大。

这意味着：

```text
Tensor Core 很强，但也更难喂饱。
你需要更高 AI、更好的 tiling、更好的数据复用。
```

这就是为什么高性能 Tensor Core GEMM 仍然很复杂。

## 23.1 Roofline 为什么值得单独认真学

Roofline 很重要，不是因为它数学复杂。

恰恰相反，它的公式很简单：

```text
performance <= min(peak_compute, AI * bandwidth)
```

它真正重要的地方在于：

```text
它能帮你在优化前先判断方向。
```

刚开始学 CUDA 时，我们很容易进入一种状态：

```text
看到慢，就想加 shared memory。
看到慢，就想调 block size。
看到慢，就想提高 occupancy。
看到慢，就想上 Tensor Core。
```

这些方向都可能对，但不是每次都对。

Roofline 要你先问一个更根本的问题：

```text
这个 kernel 现在到底缺什么？

缺数据？
缺计算吞吐？
还是理论上应该很快，但实际没有把硬件喂饱？
```

如果你不先问这个问题，优化会很像乱试。

而性能优化最怕乱试，因为一个 kernel 变快或变慢，背后可能有很多原因：

```text
访存合并变好了
缓存命中变好了
寄存器变多导致 occupancy 下降
shared memory bank conflict 变严重
指令数量减少
同步次数减少
Tensor Core 用上了
数据规模变了
计时方式变了
```

Roofline 的价值就是帮你把这些可能性先分成两大类：

```text
第一类：你主要被“搬数据”限制。
第二类：你主要被“做计算”限制。
```

然后你再用 Nsight Compute 去看更细的原因。

所以你可以这样理解：

```text
Roofline:
  决定大方向。

Nsight Compute:
  找具体证据。

代码优化:
  根据证据改 kernel。
```

这三步顺序非常重要。

## 23.2 用生活直觉理解 Roofline：厨房模型

我们先不用 GPU，换一个更直观的比喻。

假设你开了一家厨房。

厨房里有两种能力：

```text
厨师做菜的速度 = compute peak
食材送进厨房的速度 = memory bandwidth
```

一道菜需要：

```text
搬进一些食材
做一些加工
```

如果一道菜很简单：

```text
拿一个苹果
切一刀
装盘
```

那么厨师再强也没用。

瓶颈是：

```text
食材送进来有多快。
```

这就像 vector add：

```cpp
c[i] = a[i] + b[i];
```

每个元素只做 1 次加法，但要读 `a[i]`、读 `b[i]`、写 `c[i]`。

它主要是在搬数据。

如果另一道菜很复杂：

```text
食材不多
但每份食材要切、炒、调味、摆盘很多步骤
```

这时瓶颈可能变成：

```text
厨师加工速度。
```

这就像优化好的 GEMM：

```text
A 和 B 的元素搬进来以后，
在 shared memory / register 里反复复用，
每搬 1 byte 能做很多 FLOP。
```

Roofline 的横轴 `AI = FLOP/byte`，其实就是在问：

```text
每搬进来 1 byte “食材”，你能做多少“加工”？
```

AI 低：

```text
搬一点，算一点。
厨房大概率在等食材。
```

AI 高：

```text
搬一点，算很多。
厨房才有机会让厨师忙起来。
```

注意这里说的是“有机会”。

AI 高不代表一定快，因为可能还有：

```text
厨师排班不好        -> occupancy / scheduling 问题
工具没用对          -> Tensor Core 没用上
厨师互相挡路        -> shared memory bank conflict
步骤依赖太强        -> dependency stall
等人一起做下一步    -> __syncthreads() 同步开销
```

这就是为什么 Roofline 是第一层判断，不是最终诊断。

## 23.3 Roofline 实战步骤：拿到一个 kernel 后怎么做

以后你拿到一个 CUDA kernel，不要一上来就改代码。

先按这个顺序走：

```text
第 1 步：确认这个 kernel 做了什么。
第 2 步：数 FLOP。
第 3 步：数 logical bytes。
第 4 步：测时间。
第 5 步：算 GFLOPS、GB/s、AI。
第 6 步：查硬件 peak compute 和 bandwidth。
第 7 步：算 ridge point。
第 8 步：把实测点放到 Roofline 上。
第 9 步：判断下一步该优化访存还是计算。
第 10 步：用 Nsight Compute 找证据。
```

我们一项一项讲。

### 第 1 步：确认 kernel 做了什么

这一步听起来简单，但很重要。

你要先知道 kernel 的“有用工作”是什么。

例如 vector add：

```cpp
c[i] = a[i] + b[i];
```

有用工作是：

```text
每个元素做 1 次浮点加法。
```

例如 GEMM：

```cpp
C = A * B
```

有用工作是：

```text
每个 C[row, col] 做 K 次乘法和 K 次加法。
```

例如 transpose：

```cpp
out[col, row] = in[row, col];
```

有用工作主要是：

```text
读矩阵、写矩阵、改变数据位置。
```

它几乎没有浮点计算。

如果你连有用工作是什么都没定义清楚，后面的 FLOP、bytes、GFLOPS、AI 都会乱。

### 第 2 步：数 FLOP

FLOP 是 floating-point operations，也就是浮点操作次数。

常见规则：

```text
float add:
  1 FLOP

float mul:
  1 FLOP

FMA:
  a * b + c
  通常按 2 FLOP
```

为什么 FMA 按 2 FLOP？

因为数学上它包含：

```text
1 次乘法
1 次加法
```

虽然硬件可能用一条 FMA 指令完成，但性能报告通常按 2 FLOP 统计。

vector add：

```cpp
c[i] = a[i] + b[i];
```

每个元素：

```text
1 次加法 = 1 FLOP
```

总共 `N` 个元素：

```text
total_FLOP = N
```

GEMM：

```cpp
for row in M:
  for col in N:
    sum = 0
    for k in K:
      sum += A[row, k] * B[k, col]
```

每个输出元素：

```text
K 次乘法
K 次加法
约 2K FLOP
```

总共有 `M * N` 个输出：

```text
total_FLOP = 2 * M * N * K
```

transpose：

```text
基本没有浮点计算。
```

所以 transpose 通常不看 GFLOPS，而看 GB/s。

### 第 3 步：数 logical bytes

logical bytes 是你按算法“有用读写”数出来的数据量。

它不是硬件真实 DRAM transaction。

例如 vector add：

```cpp
c[i] = a[i] + b[i];
```

每个 float 是 4 bytes。

每个元素：

```text
读 a[i]  -> 4 bytes
读 b[i]  -> 4 bytes
写 c[i]  -> 4 bytes
总共     -> 12 bytes
```

所以：

```text
total_bytes = 12 * N
```

copy：

```cpp
out[i] = in[i];
```

每个元素：

```text
读 in[i]  -> 4 bytes
写 out[i] -> 4 bytes
总共      -> 8 bytes
```

所以：

```text
total_bytes = 8 * N
```

transpose：

```cpp
out[col, row] = in[row, col];
```

如果是 `m * n` 的 float 矩阵：

```text
读 input  -> m * n * 4 bytes
写 output -> m * n * 4 bytes
总共      -> 2 * m * n * 4 bytes
```

所以：

```text
total_bytes = 8 * m * n
```

GEMM 的 bytes 更容易让人绕。

最理想的 logical bytes 可以写成：

```text
读 A 一次 -> M * K * 4
读 B 一次 -> K * N * 4
写 C 一次 -> M * N * 4
```

所以理想 bytes：

```text
bytes_ideal = 4 * (M*K + K*N + M*N)
```

但 naive GEMM 实际上会反复从 global memory 读 A 和 B。

对 naive GEMM 来说，每个 `C[row, col]` 都会读：

```text
A[row, 0..K-1]
B[0..K-1, col]
```

也就是：

```text
读 A: K 个 float
读 B: K 个 float
写 C: 1 个 float
```

每个输出元素 bytes 约：

```text
K * 4 + K * 4 + 4 = 8K + 4 bytes
```

总共有 `M*N` 个输出：

```text
bytes_naive_approx = M * N * (8K + 4)
```

这就是 naive GEMM AI 很低的原因。

shared memory tiled GEMM 的核心目标就是：

```text
让 A 和 B 从 global memory 读进来以后，被一个 block 里的很多 thread 反复使用。
```

FLOP 没变：

```text
还是 2*M*N*K
```

但 global memory bytes 下降。

所以：

```text
AI = FLOP / bytes
```

会提高。

### 第 4 步：测时间

时间要用 CUDA event 测。

例如：

```cpp
cudaEventRecord(start);
kernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

然后：

```text
time_seconds = ms / 1000
```

不要忘记 warmup。

第一次运行可能包含：

```text
上下文初始化
JIT
cache 冷启动
频率还没稳定
```

所以通常：

```text
先 warmup 几次
再重复测多次取平均或中位数
```

### 第 5 步：算 GFLOPS、GB/s、AI

三个公式：

```text
GFLOPS = total_FLOP / time_seconds / 1e9
GB/s   = total_bytes / time_seconds / 1e9
AI     = total_FLOP / total_bytes
```

注意：

```text
GFLOPS 看计算速度。
GB/s 看数据搬运速度。
AI 看每 byte 做多少计算。
```

这三个指标不要互相替代。

vector add 的 GFLOPS 低不代表写得差。

因为它本来就没有多少 FLOP。

GEMM 的 GB/s 不高也不一定差。

因为优化好的 GEMM 目标是复用数据，让 DRAM bytes 变少，然后把算力打满。

### 第 6 步：查硬件 peak compute 和 bandwidth

你需要两个硬件上限：

```text
peak_compute
peak_bandwidth
```

例如 T4 FP32：

```text
peak_compute ≈ 8100 GFLOP/s
peak_bandwidth ≈ 320 GB/s
```

注意：

```text
不同数据类型的 peak_compute 不一样。
```

FP32、TF32、FP16 Tensor Core、BF16 Tensor Core、FP8 Tensor Core 都不是同一个 roof。

如果你分析的是 FP32 CUDA core kernel，就不要拿 FP16 Tensor Core 峰值当屋顶。

如果你分析的是 Tensor Core GEMM，就不要拿普通 FP32 CUDA core 峰值当屋顶。

### 第 7 步：算 ridge point

公式：

```text
ridge_point = peak_compute / peak_bandwidth
```

T4 FP32：

```text
ridge_point = 8100 / 320 ≈ 25 FLOP/byte
```

意思是：

```text
AI 小于 25：
  理论上更容易被显存带宽限制。

AI 大于 25：
  才有机会被计算峰值限制。
```

这句话非常重要：

```text
AI 大于 ridge point 只是“有机会 compute-bound”，不是一定 compute-bound。
```

因为实际 kernel 可能还有别的问题。

### 第 8 步：把实测点放到 Roofline 上

一个 kernel 的实测点是：

```text
(AI, measured_GFLOPS)
```

其中：

```text
x = AI
y = measured_GFLOPS
```

同一个 AI 下的理论上限：

```text
roof_at_AI = min(peak_compute, AI * peak_bandwidth)
```

然后看：

```text
roof_utilization = measured_GFLOPS / roof_at_AI
```

例如 vector add：

```text
AI = 0.083 FLOP/byte
measured_GFLOPS = 20
peak_bandwidth = 320 GB/s
peak_compute = 8100 GFLOP/s
```

同一个 AI 下的 roof：

```text
AI * bandwidth = 0.083 * 320 = 26.56 GFLOP/s
roof_at_AI = min(8100, 26.56) = 26.56 GFLOP/s
```

实测利用率：

```text
20 / 26.56 ≈ 75%
```

解释：

```text
这个 vector add 已经接近它自己的带宽 roof。
继续纠结 FMA、Tensor Core、算术指令没有意义。
应该看 global memory load/store 是否合并、是否对齐、是否有多余访存。
```

## 23.4 四种常见 Roofline 点位

实际分析时，你会经常遇到下面四种情况。

### 情况一：AI 低，而且接近带宽 roof

形状：

```text
AI 很小
measured_GFLOPS 接近 AI * bandwidth
```

典型 kernel：

```text
copy
vector add
简单 elementwise
优化后的连续访问 transpose
```

这说明：

```text
kernel 主要受显存带宽限制，而且已经把带宽用得还可以。
```

优化方向：

```text
检查是否还有多余读写。
检查 global memory 是否 coalesced。
检查是否可以融合 kernel，减少中间结果落 global memory。
检查数据类型能不能变小，比如 float -> half。
```

不优先做：

```text
提高 FLOP 指令吞吐。
上 Tensor Core。
追求更高 GFLOPS。
```

因为这个 kernel 不缺算力。

### 情况二：AI 低，但离带宽 roof 很远

形状：

```text
AI 很小
理论带宽 roof 不高
但 measured_GFLOPS / measured_GB/s 仍然很差
```

典型原因：

```text
global memory 没有合并访问
stride 访问导致 transaction 浪费
访问没有对齐
cache line 利用率低
分支导致 warp 内部分 thread 不工作
数据规模太小，SM 没吃满
```

优化方向：

```text
先不要急着加复杂计算优化。
先把 memory access pattern 改好。
```

你应该看 Nsight Compute：

```text
DRAM throughput
L2 throughput
global load/store efficiency
memory transaction
warp stall memory dependency
active warps
```

一个典型例子是 naive transpose。

读可能是连续的，但写是跨 stride 的。

所以 logical bytes 看起来不多，但硬件实际 transaction 很浪费。

### 情况三：AI 高，但离 compute roof 很远

形状：

```text
AI 大于 ridge point
理论上有机会 compute-bound
但 measured_GFLOPS 离 peak_compute 很远
```

这时不能简单说：

```text
这个 kernel 是 memory-bound。
```

更准确的说法是：

```text
从 DRAM Roofline 看，它不应该主要被 DRAM bandwidth 限制。
但它没有把计算单元用好。
```

常见原因：

```text
occupancy 太低，延迟藏不住
寄存器太多，active warps 少
instruction mix 不好，很多非 FMA 指令
没有用 Tensor Core
shared memory bank conflict
同步太频繁
load-use dependency 太强
tile size 不合适
```

优化方向：

```text
看 SM 利用率。
看 issue stall。
看 FMA / Tensor Core 指令占比。
看 register pressure。
看 shared memory 访问效率。
```

GEMM 优化后经常进入这种区域。

这也是为什么 GEMM 不只是 shared memory tiling，还会继续做：

```text
register tiling
double buffering
warp-level MMA
Tensor Core
更精细的数据布局
```

### 情况四：AI 高，而且接近 compute roof

形状：

```text
AI 大于 ridge point
measured_GFLOPS 接近 peak_compute
```

这说明：

```text
kernel 已经非常接近这类计算的硬件上限。
```

这时再优化会很难。

可能的方向：

```text
换更合适的数据类型。
使用 Tensor Core。
减少不必要的 epilogue 操作。
融合前后处理 kernel。
换算法。
```

但你要有心理预期：

```text
越接近 roof，每提升一点都越难。
```

面试时如果能讲到这一点，会显得你不是只会背公式，而是真的理解性能优化。

## 23.5 四个 kernel 手算 Roofline 点

这一节我们从最简单到最重要，算四个 kernel。

### 例子一：copy

代码：

```cpp
out[i] = in[i];
```

假设：

```text
N = 100,000,000 float
time = 3 ms
```

FLOP：

```text
没有浮点计算
total_FLOP = 0
```

Bytes：

```text
读 in  = N * 4
写 out = N * 4
total_bytes = 8 * N = 800,000,000 bytes = 0.8 GB
```

GB/s：

```text
GB/s = 0.8 / 0.003 = 266.7 GB/s
```

AI：

```text
AI = 0 / 0.8e9 = 0
```

结论：

```text
copy 不适合用 GFLOPS 评价。
它就是带宽测试。
```

### 例子二：vector add

代码：

```cpp
c[i] = a[i] + b[i];
```

假设：

```text
N = 100,000,000 float
time = 5 ms
```

FLOP：

```text
total_FLOP = N = 100,000,000
```

Bytes：

```text
读 a = N * 4
读 b = N * 4
写 c = N * 4
total_bytes = 12 * N = 1.2 GB
```

GFLOPS：

```text
GFLOPS = 1e8 / 0.005 / 1e9 = 20 GFLOP/s
```

GB/s：

```text
GB/s = 1.2 / 0.005 = 240 GB/s
```

AI：

```text
AI = 1e8 / 1.2e9 = 0.083 FLOP/byte
```

如果 T4 的 bandwidth 是 320 GB/s：

```text
roof_at_AI = 0.083 * 320 = 26.6 GFLOP/s
```

所以：

```text
20 GFLOP/s 已经是 26.6 的 75% 左右。
```

结论：

```text
vector add 的 GFLOPS 看起来很低，
但这是正常的。
它主要看 GB/s。
```

### 例子三：naive GEMM

代码逻辑：

```cpp
for row in M:
  for col in N:
    sum = 0
    for k in K:
      sum += A[row, k] * B[k, col]
    C[row, col] = sum
```

假设：

```text
M = N = K = 1024
time = 10 ms
```

FLOP：

```text
total_FLOP = 2 * M * N * K
           = 2 * 1024^3
           ≈ 2.147e9 FLOP
```

GFLOPS：

```text
GFLOPS = 2.147e9 / 0.010 / 1e9
       ≈ 214.7 GFLOP/s
```

naive bytes 近似：

```text
每个 C 元素：
  读 A: K float
  读 B: K float
  写 C: 1 float

bytes_per_C ≈ 4K + 4K + 4 = 8K + 4
```

总 bytes：

```text
total_bytes ≈ M * N * (8K + 4)
            = 1024 * 1024 * (8192 + 4)
            ≈ 8.59e9 bytes
```

AI：

```text
AI ≈ 2.147e9 / 8.59e9
   ≈ 0.25 FLOP/byte
```

如果 T4 ridge point 约 25：

```text
0.25 远小于 25
```

结论：

```text
naive GEMM 虽然是矩阵乘法，
但因为 global memory 复用太差，
从 DRAM Roofline 看仍然很像 memory-bound。
```

优化方向：

```text
shared memory tiling。
让 A/B 读进来以后被更多 thread 复用。
提高 AI。
```

### 例子四：tiled GEMM

假设 tiled GEMM 仍然计算：

```text
M = N = K = 1024
total_FLOP ≈ 2.147e9
```

但通过 shared memory tiling，把 global memory 读写大幅减少。

理想情况下，DRAM logical bytes 接近：

```text
读 A 一次 -> M*K*4
读 B 一次 -> K*N*4
写 C 一次 -> M*N*4

bytes_ideal = 4 * (M*K + K*N + M*N)
            = 4 * 3 * 1024^2
            ≈ 12.58 MB
```

理想 AI：

```text
AI_ideal = 2.147e9 / 12.58e6
         ≈ 170.7 FLOP/byte
```

这个 AI 已经大于 T4 的 25。

但是注意：

```text
这只是理想 DRAM bytes 视角。
真实 tiled GEMM 可能还会受 L2、shared memory、register、指令吞吐影响。
```

如果 time = 2 ms：

```text
GFLOPS = 2.147e9 / 0.002 / 1e9
       ≈ 1073.5 GFLOP/s
```

它的 AI 很高，但 GFLOPS 仍远低于 T4 FP32 peak 8100。

这说明：

```text
DRAM 可能已经不是最大瓶颈。
下一步要看计算单元有没有吃满。
```

优化方向：

```text
register tiling
减少 shared memory bank conflict
提高 occupancy 和 ILP
减少同步
使用 Tensor Core
```

这就是 GEMM 优化阶梯的原因：

```text
naive GEMM:
  global memory 复用差，AI 低。

shared tiled GEMM:
  DRAM AI 提高，但可能还没打满 SM。

register tiled GEMM:
  进一步减少 shared memory 读，增加寄存器复用。

Tensor Core GEMM:
  使用更高的矩阵乘硬件吞吐。
```

## 23.6 Roofline 到 Nsight Compute：下一步看什么

Roofline 判断大方向以后，下一步要用 Nsight Compute 找证据。

不要只停留在：

```text
我觉得它 memory-bound。
我觉得它 compute-bound。
```

你要能说：

```text
我根据 AI 和 ridge point 判断它倾向 memory-bound。
然后 Nsight Compute 里 DRAM/L2 throughput、memory stall、load/store efficiency 支持这个判断。
```

或者：

```text
我根据 AI 判断它不应该主要被 DRAM bandwidth 限制。
然后 Nsight Compute 里看到 SM busy 不高、eligible warps 少、register pressure 高，所以问题在执行效率。
```

下面是一个实用对照表。

| Roofline 现象 | 优先怀疑 | Nsight Compute 重点看 |
|---|---|---|
| AI 低，性能接近带宽 roof | 正常 memory-bound | DRAM throughput、L2 throughput、global load/store efficiency |
| AI 低，远离带宽 roof | 访存模式差 | coalescing、sector/request、stride、replay、memory dependency stall |
| AI 高，远离 compute roof | SM 没吃满 | SM busy、eligible warps、issue stall、occupancy、registers/thread |
| AI 高，但 Tensor Core GEMM 低 | 没用好 Tensor Core | tensor pipe utilization、MMA 指令、数据类型、tile shape |
| shared tiled 后仍慢 | shared/register 瓶颈 | shared bank conflict、shared throughput、register pressure、同步 stall |
| 小输入性能差 | GPU 没饱和 | grid size、active blocks、launch overhead、SM occupancy |

记住：

```text
Roofline 说的是上限。
Nsight Compute 告诉你为什么离上限还有距离。
```

## 23.7 面试里怎么讲 Roofline

面试里被问 Roofline，不要只背公式。

你可以这样讲：

```text
Roofline 是一个用算术强度和硬件峰值来判断性能上限的模型。

横轴是 arithmetic intensity，也就是 FLOP/byte。
纵轴是 performance，比如 GFLOP/s。

它有两个上限：
一个是 memory roof，等于 AI * memory bandwidth。
一个是 compute roof，等于 peak compute。

实际性能不可能超过二者的较小值。
拐点 ridge point = peak_compute / bandwidth。

如果 kernel 的 AI 小于 ridge point，通常优先考虑访存优化；
如果 AI 大于 ridge point，才有机会 compute-bound，接下来要看计算单元利用率。

但 Roofline 只是上限模型，不会解释所有 stall。
实际优化还需要结合 Nsight Compute 看 occupancy、memory transaction、stall reason、shared bank conflict、Tensor Core utilization 等指标。
```

如果要举例，可以用 vector add 和 GEMM：

```text
vector add 每个元素读两个 float 写一个 float，只做一次加法，
AI 约 1/12，所以几乎一定是 memory-bound。

naive GEMM 理论 FLOP 很多，但如果每个输出都反复从 global memory 读 A/B，
DRAM AI 仍然很低，所以需要 shared memory tiling 提高数据复用。

优化后的 GEMM AI 变高，DRAM 可能不再是主要瓶颈，
之后要继续看 register tiling、Tensor Core、instruction throughput 和 shared memory 行为。
```

这段回答就比较完整。

它体现了三层能力：

```text
知道公式。
知道怎么用公式判断方向。
知道公式之外还要用 profiler 找证据。
```

## 24. Amdahl：固定问题规模的扩展上限

Amdahl 讨论：

```text
总问题规模固定，增加并行资源，最多能快多少？
```

假设串行比例为 `s`，并行资源为 `p`：

```text
speedup <= 1 / (s + (1 - s) / p)
```

如果：

```text
s = 0.1
```

即 10% 代码无法并行。

就算并行部分无限快：

```text
最大 speedup = 1 / 0.1 = 10x
```

这告诉你：

```text
固定问题规模下，串行部分会限制最终加速比。
```

CUDA 里可能的串行/非并行部分：

```text
CPU 预处理
数据拷贝
kernel launch
同步
串行 reduction 后处理
I/O
```

## 25. Gustafson：问题规模也变大时怎么看

Gustafson 讨论：

```text
资源增加时，问题规模也一起增加。
```

例如：

```text
1 张 GPU 训练 batch=32
8 张 GPU 训练 batch=256
```

每张 GPU 工作量差不多，总问题变大。

这时串行比例可能没有 Amdahl 看起来那么致命。

所以：

```text
Amdahl:
  固定总问题规模，看加资源能快多少。

Gustafson:
  每个资源工作量固定，总问题规模随资源增加。
```

## 26. Strong Scaling

Strong scaling：

```text
固定总问题规模，增加资源。
```

例子：

```text
同一个 1 TB 数据集
用 1 张 GPU 处理
用 2 张 GPU 处理
用 4 张 GPU 处理
```

理想情况：

```text
GPU 数翻倍，时间减半。
```

现实中不一定，因为：

```text
通信开销
同步开销
负载不均
单卡工作量太小，GPU 吃不满
```

Strong scaling 常用于回答：

```text
我能不能用更多 GPU 更快完成同一个任务？
```

Strong scaling 常用效率指标：

```text
parallel_efficiency = speedup / number_of_resources
```

例如：

```text
1 GPU: 100 s
4 GPU: 30 s

speedup = 100 / 30 = 3.33x
efficiency = 3.33 / 4 = 83%
```

这说明 4 张 GPU 没有达到理想 4x，但效率还不错。

## 27. Weak Scaling

Weak scaling：

```text
每个资源工作量固定，资源增加，总问题规模也增加。
```

例子：

```text
1 张 GPU 处理 100 GB
2 张 GPU 处理 200 GB
4 张 GPU 处理 400 GB
```

理想情况：

```text
总时间不变。
```

Weak scaling 常用于回答：

```text
系统变大后，能不能处理更大的问题，同时保持每张 GPU 的效率？
```

AI 训练里：

```text
global batch 随 GPU 数增加
每张 GPU 的 micro batch 保持不变
```

就是一种 weak scaling 思路。

Weak scaling 常看：

```text
资源增加后，时间是否保持不变。
```

例如：

```text
1 GPU 处理 100 GB，用 10 s
4 GPU 处理 400 GB，也用 11 s
```

说明 weak scaling 还不错。

如果变成：

```text
4 GPU 处理 400 GB，用 25 s
```

说明随着系统变大，通信、同步或 I/O 开销变大了。

## 28. 单 GPU 里也有“饱和”问题

虽然 strong/weak scaling 常用于多 GPU/HPC，但单 GPU 里也有类似现象。

例如 block 数太少：

```text
GPU 上很多 SM 没活干。
```

增加问题规模或 block 数后：

```text
SM 被填满，性能上升。
```

再继续增加：

```text
达到带宽或算力上限，性能不再上升。
```

所以 benchmark 要注意：

```text
小输入可能测不到真实性能。
```

## 29. 常见误区

### 误区一：时间变短就一定是优化成功

不一定。

可能是：

```text
输入规模太小
少算了东西
没有同步导致计时错误
缓存热身影响
```

必须同时看正确性、输入规模、计时方法。

### 误区二：GFLOPS 越高就一定越好

不一定。

对 memory-bound kernel：

```text
GFLOPS 本来就不该高。
```

比如 vector add，重点应该是 GB/s。

### 误区三：达到 50% 峰值就一定差

不一定。

要看：

```text
这个 kernel 的 AI
它理论上是否可能接近计算峰值
是否受带宽、分支、同步、访存模式限制
```

### 误区四：Roofline 说 compute-bound 就一定能达到峰值

不一定。

Roofline 是上限模型。

它不完整建模：

```text
warp stall
occupancy
instruction mix
bank conflict
cache behavior
同步开销
```

所以 Roofline 是方向盘，不是最终判决书。

## 30. 手算模板

以后拿到一个 kernel，按这个表算：

```text
1. 它主要是搬数据还是做浮点计算？

2. 数 FLOP:
   total_FLOP = ?

3. 数 logical bytes:
   read bytes = ?
   write bytes = ?
   total_bytes = ?

4. 测时间:
   time_seconds = ?

5. 算指标:
   GB/s = total_bytes / time_seconds / 1e9
   GFLOPS = total_FLOP / time_seconds / 1e9
   AI = total_FLOP / total_bytes

6. 查硬件:
   peak_compute = ?
   peak_bandwidth = ?
   ridge_point = peak_compute / peak_bandwidth

7. 判断:
   AI < ridge_point  -> memory-bound 倾向
   AI > ridge_point  -> compute-bound 倾向

8. 算当前 AI 下的理论 roof:
   roof_at_AI = min(peak_compute, AI * peak_bandwidth)

9. 算离 roof 多远:
   roof_utilization = measured_GFLOPS / roof_at_AI

10. 决定优化方向:
   memory-bound  -> 优化访存和复用
   compute-bound -> 优化计算单元利用率

11. 用 Nsight Compute 找证据:
   memory-bound  -> 看 DRAM/L2 throughput、coalescing、memory stall
   compute-bound -> 看 SM busy、issue stall、occupancy、Tensor Core utilization
```

## 31. 五个练习

### 练习一：vector add

```text
N = 100,000,000 float
time = 5 ms
```

求：

```text
logical GB/s
GFLOPS
AI
```

答案：

```text
bytes = 3 * N * 4 = 1.2 GB
GB/s = 1.2 / 0.005 = 240 GB/s

FLOP = N = 1e8
GFLOPS = 1e8 / 0.005 / 1e9 = 20 GFLOPS

AI = 1e8 / 1.2e9 = 0.083 FLOP/byte
```

解释：

```text
GB/s 是更有意义的指标。
```

### 练习二：GEMM

```text
M = N = K = 1024
time = 2 ms
```

答案：

```text
FLOP = 2 * 1024^3 ≈ 2.15e9
GFLOPS = 2.15e9 / 0.002 / 1e9 ≈ 1074 GFLOPS
```

如果它是 naive GEMM，下一步要问：

```text
它是不是 DRAM 读太多？
能不能用 shared tiling 提高 AI？
```

### 练习三：T4 Roofline

```text
peak_compute = 8100 GFLOP/s
peak_bandwidth = 320 GB/s
```

答案：

```text
ridge_point = 8100 / 320 ≈ 25 FLOP/byte
```

解释：

```text
AI 远小于 25 的 kernel 优先看内存。
AI 大于 25 的 kernel 才有机会看计算峰值。
```

### 练习四：把 vector add 放到 Roofline 上

已知：

```text
AI = 0.083 FLOP/byte
measured_GFLOPS = 20
peak_compute = 8100 GFLOP/s
peak_bandwidth = 320 GB/s
```

求：

```text
roof_at_AI
roof_utilization
下一步优化方向
```

答案：

```text
memory_roof = AI * peak_bandwidth
            = 0.083 * 320
            ≈ 26.6 GFLOP/s

roof_at_AI = min(8100, 26.6)
           = 26.6 GFLOP/s

roof_utilization = 20 / 26.6
                 ≈ 75%
```

解释：

```text
它已经比较接近带宽 roof。
下一步重点不是优化浮点计算，而是看访存合并、对齐、多余读写、kernel fusion。
```

### 练习五：比较 naive GEMM 和 tiled GEMM

已知：

```text
M = N = K = 1024
T4 FP32 ridge point ≈ 25 FLOP/byte
```

naive GEMM 近似：

```text
AI ≈ 0.25 FLOP/byte
```

tiled GEMM 理想 DRAM 视角：

```text
AI_ideal ≈ 170 FLOP/byte
```

问题：

```text
为什么两者都是 GEMM，但 Roofline 判断完全不同？
```

答案：

```text
因为 FLOP 一样，差别在 bytes。

naive GEMM 每算一个 C 元素都反复从 global memory 读 A 和 B，
所以 DRAM bytes 很大，AI 很低。

tiled GEMM 把 A 和 B 搬进 shared memory 后复用，
global memory bytes 大幅下降，AI 上升。
```

结论：

```text
GEMM 优化的第一目标不是减少 FLOP，
而是减少 global memory 重复读，提高数据复用，提高 AI。
```

## 32. 实践

1. 为 `labs/03_memory_system/memory_access` 算 logical GB/s。
2. 为 `labs/03_memory_system/transpose` 算有效 GB/s。
3. 为 `labs/02_programming_model/gemm_naive` 算 GFLOPS 和 AI。
4. 为 `labs/06_operators/gemm_tiled` 对比 naive / tiled 的 GFLOPS。
5. 使用 T4 规格与实测带宽画一张简单 Roofline。
6. 解释为什么 transpose shared 仍可能远低于规格峰值。
7. 为 vector add 计算 `(AI, measured_GFLOPS)`，再计算 `roof_at_AI`。
8. 对一个你自己写的 kernel 做一次完整 Roofline 分析：FLOP、bytes、AI、GFLOPS、roof、Nsight Compute 证据。

## 33. 面试题

**Q1：有效带宽和硬件实际带宽有什么区别？**

有效带宽按算法逻辑读写字节计算，比如 copy 是读 N 写 N。但硬件实际 DRAM 事务可能因为不合并、
cache line、重放等原因搬更多字节。有效带宽低说明算法没有充分利用硬件带宽。

**Q2：GFLOPS 怎么算？GEMM 为什么是 2MNK？**

GFLOPS = 总 FLOP / 秒 / 1e9。GEMM 有 M*N 个输出，每个输出做 K 次乘和 K 次加，所以约
`2*M*N*K` FLOP。FMA 按 2 FLOP 计。

**Q3：算术强度是什么？**

算术强度是 FLOP/byte，表示每搬 1 byte 数据做多少计算。AI 低通常 memory-bound，AI 高才可能
compute-bound。要说明 bytes 是按 DRAM、L2 还是 shared 统计。

**Q4：Roofline 有什么用？**

Roofline 用 AI、峰值算力和峰值带宽估算性能上限，判断 kernel 更可能被带宽还是算力限制。它主要
指导优化方向，不保证实际性能能达到线。

**Q5：为什么 vector add 不可能接近 GPU FP32 峰值？**

因为 vector add 的 AI 很低。每个元素读两个 float、写一个 float，只做 1 次加法，所以 AI 约
`1/12 = 0.083 FLOP/byte`。在 T4 这种 ridge point 约 25 FLOP/byte 的 GPU 上，它远在拐点左边，
主要受显存带宽限制，而不是 FP32 算力限制。

**Q6：为什么 naive GEMM FLOP 很多，但仍可能 memory-bound？**

因为 naive GEMM 对 global memory 的数据复用很差。每个输出元素都会重复读取一行 A 和一列 B，
导致 DRAM bytes 很大，AI 可能只有约 0.25 FLOP/byte。shared memory tiling 的目的就是减少这些
重复 global load，提高 AI。

**Q7：Roofline 说 AI 很高，为什么实际 GFLOPS 仍然可能很低？**

因为 Roofline 只是上限模型。AI 高只说明从 DRAM 带宽角度看有机会 compute-bound，但实际还可能受
occupancy、寄存器压力、指令 mix、依赖 stall、shared memory bank conflict、同步、Tensor Core 未使用等因素限制。
这时要用 Nsight Compute 找具体证据。

**Q8：Strong scaling 和 weak scaling 区别？**

Strong scaling 固定总问题规模，增加资源，看时间能否下降。Weak scaling 保持每个资源工作量固定，
随着资源增加扩大总问题规模，看时间能否保持稳定。

## 34. 本章小结

```text
Time:
  所有指标的基础，必须可靠计时。

Speedup:
  快了几倍，但必须说明 baseline。

GB/s:
  搬数据型 kernel 的关键指标。

GFLOPS:
  计算型 kernel 的关键指标。

AI:
  每搬 1 byte 做多少 FLOP，是 Roofline 横轴。

Roofline:
  用 AI 判断优化方向，memory-bound 先优化访存，compute-bound 再优化计算。

Scaling:
  看增加资源后是否继续有效。
```

## 35. 资料映射

- CUDA C++ Best Practices Guide：Performance Metrics、Memory Optimizations、Scaling。
- Nsight Compute：Speed Of Light / Roofline 相关分析。
- Roofline Model 论文：性能上限模型。
- 配套：[卷五第 01 章 APOD 与可靠 Benchmark](01_APOD与可靠Benchmark.md)、[卷五第 05 章 Nsight Compute](05_Nsight_Compute与Compute_Sanitizer.md)、[卷六 GEMM](../volume06_operators/README.md)。
