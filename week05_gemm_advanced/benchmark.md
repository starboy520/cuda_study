# GEMM Benchmark

## 环境

| 项目 | 值 |
| --- | --- |
| GPU | Tesla T4 (sm_75, 16 GB) |
| CUDA | release 13.3, V13.3.33 |
| 编译参数 | `nvcc -arch=sm_75 gemm_shared_baseline.cu -o gemm_shared_baseline` |
| 数据类型 | FP32 |
| 计时方式 | cudaEvent，warmup 1 次后取正式一次 |
| GFLOPS 公式 | `2*M*N*K / time_s / 1e9` |

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

