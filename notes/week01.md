# Week 1 学习笔记

## 环境信息

| 项 | 值 |
|----|-----|
| GPU | Tesla T4 |
| compute capability | 7.5 |
| Driver | 595.71.05（`nvidia-smi`） |
| CUDA Toolkit | 13.3（`nvcc --version`） |
| CUDA device count | 1 |
| 编译架构 | `-arch=sm_75` |

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
# Tesla T4, 7.5, 16384 MiB

cd week01_basics/device_query && nvcc -O3 -arch=sm_75 -o device_query device_query.cu && ./device_query
```

---

## 设备属性（T4，`device_query` 输出）

| 属性 | 值 | 备注 |
|------|-----|------|
| SM count | 40 | `multiProcessorCount`；nvidia-smi 查不到 |
| compute capability | 7.5 | `major=7`, `minor=5` |
| warp size | 32 | SIMT 基本单位 |
| max threads per block | 1024 | |
| max threads dim (x,y,z) | 1024, 1024, 64 | `maxThreadsDim[3]` |
| max blocks per SM | 16 | `maxBlocksPerMultiProcessor` |
| max grid size (x,y,z) | 2147483647, 65535, 65535 | 实际 launch 远小于此 |
| shared memory / block | 49152 B（48 KB） | `sharedMemPerBlock` |
| shared memory / SM | 65536 B（64 KB） | `sharedMemPerMultiprocessor`，≠ per block |
| global memory | 16704405504 B（≈15.6 GB） | API 值略小于 smi 的 16384 MiB（保留区） |
| constant memory | 65536 B（64 KB） | `totalConstMem` |

**和 nvidia-smi 对照**：名称、CC、显存容量一致；SM 数只能用 `cudaGetDeviceProperties`。

---

## 概念笔记（详见 CUDA基础概念.md）

- [x] Host / Device
- [x] Kernel = `__global__` + `<<<grid, block>>>`
- [x] 一次 launch = 一个 Grid
- [x] Thread → Block → Grid 层次
- [x] compute_cap 7.5 = sm_75
- [x] cudaEvent 计时（Step 04）
- [x] device_query / cudaGetDeviceProperties（Step 05）
- [x] mat_mul_naive GPU + GFLOPS（Step 09–10）
- [ ] SIMT / Warp 手绘（Step 06）
- [ ] Occupancy（Step 11）

**我的三句话总结**：
1. 一次 launch = 一个 Grid，里面很多 Block，每个 Block 很多 Thread
2. vec_add 里每个 Thread 算一个 `c[i]`
3. SM 是 GPU 硬件执行单元，T4 有 40 个；Grid 大小由我根据 n 决定

---

## 每日记录

### Day 1

**目标**：环境确认、读 Programming Guide **1. Introduction to CUDA**、跑通 vec_add

**概念**：
- CUDA = Host(CPU) + Device(GPU) 协同
- `h_` = host 内存，`d_` = device 显存

**实验**：
```bash
cd week01_basics/vec_add && make && ./vec_add
```

**问题**：
- `nvidia-smi` 没有 `multiprocessor_count` 字段 → 用 CUDA API 查 SM
- Programming Model 第一遍绕 → 对照 vec_add 读，见 `CUDA基础概念.md`

---

### Day 2

**目标**：完成 `vec_add` 的 H2D / Kernel / D2H 分段计时实验

**概念**：
- H2D = Host to Device，把 CPU 内存复制到 GPU global memory
- Kernel time = GPU 执行 `vec_add<<<blocks, threads>>>` 的时间
- D2H = Device to Host，把 GPU 结果复制回 CPU 内存
- `cudaEventSynchronize(stop)` 等到 stop event 完成即可，不需要单独同步 start

**实验**：
```bash
cd week01_basics/vec_add
make clean && make
./vec_add 1048576
./vec_add 16777216
```

结果：
- `n=1048576`：H2D 1.148192 ms，Kernel 0.144704 ms，D2H 0.684640 ms，Total 1.977536 ms，PASS
- `n=16777216`：H2D 11.672992 ms，Kernel 0.840352 ms，D2H 6.760608 ms，Total 19.273952 ms，PASS

**问题**：
- 计时范围要放准：`cudaMalloc` 是分配开销，不算 H2D；H2D 应该只包住 Host→Device 的 `cudaMemcpy`
- 当前 `vec_add` 计算很简单，整体时间主要花在 H2D/D2H 数据传输上，Kernel 时间明显更小
- `n` 从 1M 增加到 16M 后，三段时间都增加，其中拷贝时间增长最明显，因为传输数据量直接变大

---

### Day 3

**目标**：Step 05 `device_query` + Step 06 线程层次 / SIMT 入门

**概念**：
- `cudaGetDeviceCount` 查 GPU 数量；`cudaGetDeviceProperties` 查单卡属性
- SM = 流多处理器，T4 有 40 个；一个 block 会被调度到某个 SM 上执行
- Warp = 32 线程，SIMT 以 warp 为单位执行同一条指令
- `sharedMemPerBlock`（48 KB）和 `sharedMemPerMultiprocessor`（64 KB）不是同一个数

**实验**：
```bash
cd week01_basics/device_query
nvcc -O3 -arch=sm_75 -std=c++14 -o device_query device_query.cu
./device_query
```

**device_query 要点**：
- 先用 `cudaGetDeviceCount` 打印设备数，再循环 `print_device(i)`
- T4 验证：`warpSize=32`，`maxThreadsPerBlock=1024`，`multiProcessorCount=40`

**手算 blocks（复习）**：
- `n=1000, threads=256` → `blocks = (1000+256-1)/256 = 4`，共 1024 线程，最后一 block 只有 232 个有效线程
- 2D 约定：`x→col`，`y→row`；`col = blockIdx.x*blockDim.x + threadIdx.x`

**问题**：
- `totalGlobalMem`（API）和 smi 的 `memory.total` 数值不完全相同 → API 是可用显存，smi 是物理容量
- transpose 写回阶段 x/y 对调是为了合并写 → 见 `week02_memory/transpose/转置优化详解.md`（Week 2 提前练）

---

### Day 5–6（Step 08–10：mat_mul CPU/GPU + GFLOPS）

**目标**：`matmul_cpu` / `matmul_gpu` 正确性 + kernel 单独计时 + GFLOPS 基线

**实验**：
```bash
cd week01_basics/mat_mul_naive
nvcc -O3 -arch=sm_75 -std=c++14 -o mat_mul mat_mul.cu
./mat_mul --m 256 --n 256 --k 256
./mat_mul --m 512 --n 512 --k 512
./mat_mul --m 1024 --n 1024 --k 1024
```

**要点**：
- GPU 与 CPU 对比 PASS（相对误差 < 1e-4）
- `cudaEvent` **只包** `matmul_gpu<<<>>>`，不含 H2D/D2H/malloc
- GFLOPS = `2*M*N*K / kernel_ms / 1e6`（`cudaEventElapsedTime` 返回毫秒）
- block `(16,16)`，grid `((N+15)/16, (M+15)/16)`

**问题**：
- 初版公式用 `/1e9` 且包了整段 `test_matmul_gpu` → GFLOPS 只有 ~0.15，修正后 ~400 量级
- 512³ GFLOPS 高于 1024³：小矩阵 launch/固定开销占比大，大矩阵更稳；1024³ 作 Week 5 基线

---

## 性能记录

### vec_add 计时（Step 04）

| 规模 n | H2D (ms) | Kernel (ms) | D2H (ms) | 合计 (ms) |
|--------|----------|-------------|----------|-----------|
| 1M | 1.148192 | 0.144704 | 0.684640 | 1.977536 |
| 16M | 11.672992 | 0.840352 | 6.760608 | 19.273952 |

### mat_mul_naive GFLOPS（Step 10）

测试条件：T4，`nvcc -O3 -arch=sm_75`，block `(16,16)`，仅 kernel 时间。

| 规模 | Kernel (ms) | GFLOPS |
|------|-------------|--------|
| 256³ | 0.203 | 165.2 |
| 512³ | 0.617 | 435.2 |
| 1024³ | 5.269 | **407.5** |

> **Week 5 基线**：1024³ = **407.5 GFLOPS**（朴素 naive，远低于 cuBLAS ~3000+，正常）。

---

## Occupancy 实验（Step 11）

详见 [Occupancy详解_从入门到调优.md](../docs/Occupancy详解_从入门到调优.md)

| blockDim | 线程数 | GFLOPS (1024³) |
|----------|--------|----------------|
| (16,16) | 256 | |
| (8,8) | 64 | |
| (32,32) | 1024 | |

---

### 内存层次（Step 06 / Week 2 预习）

- **Global Memory**：GPU 主显存，所有 thread 可访问；`cudaMalloc` 分配；带宽高但延迟大
- **Shared Memory**：block 内共享，物理在 SM 上；`__shared__` 声明；比 global 快，容量小（T4 每 block 最多 48 KB）
- **Register**：每个 thread 私有，最快，数量有限

---

## 本周总结

- **掌握**：Host/Device、kernel launch、cudaEvent 分段计时、device_query、2D 映射（x→col, y→row）、mat_mul naive GPU PASS、GFLOPS 基线（1024³ ≈ 407 GFLOPS）
- **薄弱**：transpose 写回 x/y 对调（已手写 `my_transpose.cu`，逻辑正确）；SIMT/Occupancy 待 Step 06/11 深化
- **下周重点**：Week 2 内存合并、shared memory transpose、reduction
