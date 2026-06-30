# GEMM Benchmark

## 环境

| 项目 | T4 | A100 |
| --- | --- | --- |
| GPU | Tesla T4 (sm_75, 16 GB) | NVIDIA A100 80GB PCIe (sm_80) |
| FP32 峰值 | ~8.1 TFLOPS | ~19.5 TFLOPS |
| 显存带宽 | ~320 GB/s | ~1935 GB/s |
| SM 数 | 40 | 108 |
| CUDA | release 13.3 | release 13.3 |
| 编译参数 | `-arch=sm_75` | `-arch=sm_80` |
| 数据类型 | FP32 | FP32 |
| 计时方式 | cudaEvent，warmup 1 次后取正式一次 | 同左 |
| GFLOPS 公式 | `2*M*N*K / time_s / 1e9` | 同左 |

> 注意：T4 与 A100 的 GFLOPS 不可直接同列比较，分别看「占各自峰值百分比」。

---

## A100 版本对比（sm_80, FP32, 实测）

| 版本 | 512 | 1024 | 2048 | 2048 占峰值 |
| --- | --- | --- | --- | --- |
| shared baseline | 2979 | 3957 | 4200 | ~22% |
| 2D + padding | 2703 | 7109 | **11305** | **~58%** |
| 2D 相对 baseline | 0.91x | 1.80x | **2.69x** | |

观察：
- 2048 时 2D = 11305 GFLOPS，约 A100 FP32 峰值的 58%（T4 上同 kernel 仅 16%），register tiling 在 A100 上收益更大。
- **512 时 2D 反而比 baseline 慢**：2D block 仅 64 线程，512 时 grid 只有 (512/64)²=64 个 block，连 A100 的 108 个 SM 都填不满；baseline 512 是 256 个 block ×1024 线程，反而喂得饱。
- 结论：A100 并行资源多，小规模 GEMM 下 2D 的低 occupancy + 少 block 暴露不足 → Day 5 在 A100 上要试更大的 BM/BN 或更多线程。

---

## Week2 Day1：float4 向量化 load（A100, 2048）

| 版本 | global 读 | shared 布局 | 写 shared | 2048 GFLOPS | 说明 |
| --- | --- | --- | --- | --- | --- |
| 2D + padding（基线） | 标量 | `[BK+1]` padding | 标量 | 11305 | Week1 最优 |
| 向量化 + 无 padding | float4 | 无 padding | 标量 | **8995** | ❌ 反而慢 |
| 向量化 + padding | float4 | `[BK+1]` padding | 标量 | **12681** | ✅ 最快 🥇 |

### ncu 定位（无 padding 向量化版，2048）
| 指标 | 值 |
| --- | --- |
| shared **load** bank conflict | 134,288,803 |
| shared **store** bank conflict | 26,453,941 |
| registers/thread | 126 |
| SM throughput | 53.12% |
| achieved occupancy | 19.38% |

### 结论（关键认知）
- **盲目去 padding 换 float4 写 shared → 1.3 亿次 bank conflict**，把向量化省下的加载收益全吃掉，2048 反而从 11305 跌到 8995。
- 正确组合：**float4 只用于读 global（连续、对齐），shared 仍保留 padding + 标量读写防冲突** → 兼得两个收益，2048 冲到 12681（约 A100 FP32 峰值 65%）。
- 印证 Day6 Roofline 结论：此 kernel 瓶颈在 shared / occupancy，不在 global 加载；向量化要对症下药。

### 面试口述
> 我在 2D register tiling 上加 float4 向量化 global load，第一版顺手把 shared 的 padding 去了，结果 2048 反而从 11305 掉到 8995。用 ncu 一看 shared load bank conflict 高达 1.3 亿——去 padding 导致计算阶段读 shared 严重 bank 冲突。把 padding 加回、float4 只用于读 global、shared 仍标量读写，2048 提到 12681。教训是：向量化加速的是 global 读，shared 防冲突靠 padding，两者别混用。

### 约束
向量化 load 要求 **K%4==0 且 N%4==0**（A/B 的 float4 沿 K/N 方向，需行起址 16B 对齐），M 任意（已用非规整 M=100/200/300 验证 PASS）。

---

## Week2 Day3：cp.async + pipeline 双缓冲（A100, 2048）

kernel：`gemm_2d_double_buffering_tiling`，用 `cuda::memcpy_async` + `cuda::pipeline`（depth=2）+ 两块 shared buffer 做 ping-pong。

| 版本 | 加载方式 | 2048 GFLOPS | 说明 |
| --- | --- | --- | --- |
| 2D + padding（基线） | 标量 load | 11305 | Week1 最优 |
| float4 向量化 + padding | float4 读 global，标量写 shared | **12681** | 当前最快 🥇 |
| cp.async 双缓冲 | 标量 cp.async（每次 1 float）+ padding | **7612** | 结构正确但偏慢 |

512/1024/2048 三档均 PASS（max abs err 1.5e-05 ~ 3e-05）。

### 为什么双缓冲反而慢
```text
1. 加载粒度太小：cp.async 每次只搬 sizeof(float)=4B → 一个 tile 发 BM*BK 条 memcpy_async
   指令数是 float4 向量化版的 4 倍。
2. 此 kernel 是 compute/occupancy-bound（Day6 已证 DRAM 只占 1.5%），
   双缓冲"藏 global 延迟"的收益本就有限。
```

### padding vs float4 cp.async 的矛盾（关键认知）
```text
cp.async 直写 shared 16B（float4）→ 目标必须连续 16B 对齐 → 不能 padding → bank conflict
当前选择：保留 padding + 标量 cp.async → 无 bank conflict，但加载指令多
三种组合：
  A 标量 cp.async + padding（当前双缓冲）：无冲突，指令多 → 慢
  B float4 cp.async 直写（去 padding）   ：指令少，有 bank conflict
  C float4 读 global + 标量写 padded shared（向量化版）：无冲突 + 加载少 → 最优(12681)
要兼得"向量化 + 无冲突 + 异步重叠"，工业做法是 cp.async float4 + swizzle 布局（非 padding），
即 cuBLAS/CUTLASS 的方案，索引复杂，留作进阶。
```

### 面试口述
> 我用 cuda::pipeline + 两块 shared buffer 写了 cp.async 双缓冲 GEMM，ping-pong 结构正确，三档都 PASS，但 2048 只有 7612，反而比 float4 向量化版的 12681 慢。原因有两个：一是我的 cp.async 每次只搬一个 float，加载指令是向量化版的四倍；二是这个 kernel 本就 compute-bound，藏 global 延迟收益有限。想让双缓冲更快得用 float4 cp.async，但那要求 shared 连续对齐、不能 padding，会引入 bank conflict——工业上靠 swizzle 布局同时拿到向量化、无冲突和异步重叠，这是 CUTLASS 的做法。

---

## Day 1：shared memory GEMM baseline

kernel：`gemm_shared_baseline`
配置：`block = 32x32`，`TILE = 32`，每个 thread 算 1 个 C 元素。

| M=N=K | block / grid | time (ms) | GFLOPS | 正确性 (max abs err) |
| --- | --- | --- | --- | --- |
| 512  | 32x32 / 16x16   | 0.742  | 361.86 | PASS (1.5e-05) |
| 1024 | 32x32 / 32x32   | 4.138  | 518.97 | PASS (1.5e-05) |
| 2048 | 32x32 / 64x64   | 30.497 | 563.33 | PASS (1.5e-05) |

> 注：time/GFLOPS 取 warmup 后第二次运行；CPU reference 逐元素三重循环校验。

### 观察

- 规模越大 GFLOPS 越高（512 → 2048：362 → 563），说明小规模时 kernel launch / 尾部效应占比更大，大规模更能摊薄开销、复用更充分。
- T4 FP32 峰值约 8.1 TFLOPS，当前 baseline 在 2048 仅约 563 GFLOPS，约为峰值的 **7%**，还有很大优化空间。
- shared tiling 已经把 global memory 的重复读取降下来了，但每个 thread 在 K 循环里仍频繁从 shared memory 读 A/B，瓶颈转移到 shared memory 访问、指令吞吐和寄存器复用 —— 这正是后面 1D/2D register tiling 要解决的。

### 一段口述（面试用）

> shared tiling 把 A、B 的 tile 先搬进 shared memory，让一个 block 内的线程复用这块数据，从而大幅减少 global memory 的重复访问、提高算术强度。但它没有解决的是：每个线程只算一个输出元素，在 K 方向的内层循环里仍要反复从 shared memory 读 A 和 B，shared memory 带宽和指令发射成为新瓶颈。下一步的 register tiling 让一个线程计算多个输出，把数据读进寄存器后复用多次，进一步提高复用率。

---

## Day 2-3：2D register (thread) tiling

kernel：`gemm_2d_thread_tiling`
配置：`BM=BN=64`，`BK=32`，`TM=TN=8`，`block = 8x8 (64 线程)`，每个 thread 算 8x8=64 个 C 元素。
核心：K 方向每步把 A 的一小列读进 `regM[TM]`、B 的一小行读进 `regN[TN]`，做外积累加到 `acc[TM][TN]`。

| M=N=K | block / grid | time (ms) | GFLOPS | 正确性 (max abs err) |
| --- | --- | --- | --- | --- |
| 512  | 8x8 / 8x8    | 0.524  | 512.00 | PASS (1.5e-05) |
| 1024 | 8x8 / 16x16  | 2.511  | 855.30 | PASS (1.5e-05) |
| 2048 | 8x8 / 32x32  | 19.442 | 883.66 | PASS (1.5e-05) |

> 注：time/GFLOPS 取 warmup 后第二次运行。

### 观察

- 相比 shared baseline，2048 从 563 → 884 GFLOPS（**1.57x**），约为 T4 FP32 峰值的 **11%**（baseline 7%）。
- 复用关键：每步读 `TM+TN=16` 个数，做 `TM*TN=64` 次乘加，把 shared memory 的读取量摊薄，缓解了 baseline 的 L1/TEX 瓶颈。
- 代价：每个 thread 持有 `acc[8][8]` + `regM[8]` + `regN[8]`，寄存器用量上升，block 变小（64 线程），occupancy 可能下降 —— 待 ncu 验证。

---

## Day 4：2D + shared padding（消 bank conflict）

改动：把 shared 数组改成 `sa[BM][BK+1]`、`sb[BK][BN+1]`，stride 错开避免 bank conflict。

| M=N=K | time (ms) | GFLOPS | 正确性 |
| --- | --- | --- | --- |
| 2048 | 13.375 | 1284.45 | PASS |

### 观察（ncu 1024，详见 ncu_notes.md）

- ncu 发现 2D tiling 的 L1/TEX 不降反升到 91%，根因是 `BK=32` 让 sa 列读取全部落到同一个 bank → **4.8-way bank conflict**（66% 的 wavefront 是冲突）。
- 加 `+1` padding 后：bank conflict 4.8 → 1.5 way，L1/TEX 91% → 83%，Compute(SM) 20% → 42%，Elapsed Cycles -40%。
- 2048 实测 884 → **1284 GFLOPS（1.45x）**，相对 baseline **2.28x**，约 T4 FP32 峰值的 16%。
- 副作用：padding 破坏 16 字节对齐，shared load 无法向量化，请求数上升，但冲突减少占主导，净赢。

---

## 版本对比

| 版本 | M=N=K=2048 GFLOPS | 相对 baseline | 备注 |
| --- | --- | --- | --- |
| shared baseline | 563 | 1.0x | 每 thread 算 1 个 C |
| 2D register tiling | 884 | 1.57x | 每 thread 算 8x8，外积复用 |
| 2D + padding | 1284 | **2.28x** | 消 bank conflict |
| vectorized load | - | - | 待测 |
| cuBLAS 参考 | - | - | 待测 |

---

## Day 5：参数实验（A100, sm_80, 2048, 通用 grid-stride 加载）

> 前置：把加载从「位置驱动」改成「grid-stride 编号驱动」（详见 shared_load_patterns.md），
> 加载与计算解耦后才能自由扫参数。重写后寄存器 168→122，仍全部 PASS。
> 扫描脚本：`./sweep.sh`（`-D` 覆盖 TM/TN/BM/BN/BK，自动抓寄存器 + GFLOPS + 正确性）。

### 扫描结果（2048）

| TMxTN | BMxBN | BK | reg/thread | GFLOPS | 正确性 |
| --- | --- | --- | --- | --- | --- |
| **8x4** | 64x64 | 16 | **78** | **11382** | PASS 🥇 |
| 8x8 | 128x64 | 16 | 127 | 10249 | PASS |
| 4x4 | 64x64 | 16 | 72 | 9981 | PASS |
| 8x8 | 64x64 | 32 | 122 | 9625 | PASS（基线） |
| 8x8 | 64x64 | 16 | 122 | 9506 | PASS |
| 8x8 | 128x128 | 16 | 122 | 8285 | PASS |
| 4x4 | 128x128 | 16 | 72 | — | 启动失败 |

### 关键观察

- **最优 = `8x4 64x64 BK16`：11382 GFLOPS，78 寄存器**，比 8x8 基线（9625）快 **~18%**。
- ncu 验证（详见 ncu_notes.md）：8x4 vs 8x8 → 寄存器 122→78，**achieved occupancy 14.5%→30.1%（翻倍）**，SM throughput 56%→66%。
- **甜点效应**：4x4（72 reg）= 9981，反而比 8x4 慢 → TM/TN 太小，单线程复用/ILP 不够，occupancy 收益被抵消。occupancy 不是越高越好。
- **128x128 变慢（8285）**：大 block → 2048 下只有 16×16=256 个 block，喂不饱 108 个 SM，occupancy 反降。
- **`4x4 128x128` 启动失败**（非真实 GFLOPS）：block = 32×32 = 1024 线程，1024×72 = 73728 > 65536（A100 每 SM 寄存器上限）→ "too many resources requested for launch"。边界教训：`线程数 × 寄存器 ≤ 65536`。

### 因果链

```text
TN 8→4 → reg_c[8][4] 累加器减半 → 寄存器 122→78
→ 每 SM 驻留 block 翻倍 → occupancy 14.5%→30.1%
→ 活跃 warp 增多，藏延迟更强 → SM throughput 56%→66%
→ GFLOPS 9625→11382（+18%）
但 TM/TN 砍过头(4x4) → 单线程复用不足 → 回落 → 存在甜点
```

