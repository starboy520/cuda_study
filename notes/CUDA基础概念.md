# CUDA 基础概念笔记

> 学习过程中的概念整理，配合 `vec_add.cu` 和 Programming Guide 阅读。  
> 代码路径：`week01_basics/vec_add/vec_add.cu`

---

## 1. 环境信息（T4）

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
# Tesla T4, 7.5, 16384 MiB

nvcc --version
# CUDA 13.x
```

**注意**：`nvidia-smi` **没有** `multiprocessor_count` 字段，SM 数量要用 `cudaGetDeviceProperties`。

---

## 2. compute_cap（计算能力）

- `compute_cap` = **Compute Capability**，格式 `Major.Minor`
- T4 上是 **7.5** → Turing 架构 → 编译用 `-arch=sm_75`
- 作用：标识 GPU 架构版本，决定可用特性和编译目标
- 与代码对应：`prop.major = 7`, `prop.minor = 5`

| GPU | compute_cap | nvcc |
|-----|-------------|------|
| T4 | 7.5 | sm_75 |
| A100 | 8.0 | sm_80 |
| H100 | 9.0 | sm_90 |

---

## 3. 设备属性：SM / threads / memory

| 属性 | T4 实测 | 含义 | 怎么查 |
|------|---------|------|--------|
| SM count | 40 | 流多处理器数量，GPU 执行 block 的硬件单元 | `cudaGetDeviceProperties` → `multiProcessorCount` |
| max threads per block | 1024 | 每个 block 最多线程数 | `maxThreadsPerBlock` |
| warp size | 32 | 硬件调度单位，固定 32 | `warpSize` |
| shared memory / block | 48 KB | block 内线程共享的快速内存 | `sharedMemPerBlock` |
| shared memory / SM | 64 KB | 每个 SM 上 shared memory 总量 | `sharedMemPerMultiprocessor` |
| global memory | ~16 GB | 主显存，`cudaMalloc` 在这里 | `totalGlobalMem` 或 `nvidia-smi` |

**记忆**：
- **SM**：GPU 上的「车间」，T4 有 40 个
- **shared memory**：班组内小仓库（快、小），Week 2 再用
- **global memory**：整个工厂大仓库（慢、大），`d_a` 在这

---

## 4. GPU / SM / Grid / Block / Thread 关系

```
GPU（T4，整块显卡）
 └── 40 个 SM（硬件固定）
      └── 调度执行多个 Block
           └── 每个 Block 内有多个 Thread

一次 kernel launch
 └── 1 个 Grid（逻辑，本次全部工作）
      └── 很多 Block
           └── 每个 Block 很多 Thread
```

| 概念 | 性质 | vec_add (n=1M) |
|------|------|----------------|
| GPU | 硬件 | T4 |
| SM | 硬件，40 个 | 调度 4096 个 block |
| Grid | 逻辑，每次 launch 定义 | 4096 blocks |
| Block | 逻辑 | 每 block 256 threads |
| Thread | 逻辑 | 每 thread 算一个 `c[i]` |

**公式（1D）**：
```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

---

## 5. Kernel 与 Grid

- **一次 `<<<>>>` launch = 一个 Grid**（固定对应）
- **一次 launch 不能产生多个 Grid**
- **一个程序可以 launch 多次** → 多个 Grid（来自不同次启动）

```cuda
vec_add<<<blocks, threads>>>(...);  // Grid 1
vec_mul<<<blocks, threads>>>(...);  // Grid 2
```

- 默认单 stream：Grid 依次执行
- 多 stream（Week 3）：不同 kernel 可能重叠

---

## 6. Grid 是固定的吗？

**不固定。** 每次 launch 自己指定，随问题规模变化。

```cuda
int blocks = (n + threads - 1) / threads;  // n 不同，grid 不同
vec_add<<<blocks, 256>>>(...);
```

| 固定（硬件） | 不固定（你定义） |
|--------------|------------------|
| SM 数 = 40 | Grid 有多少 block |
| warp = 32 | 每 block 多少 thread（≤1024） |
| max threads/block = 1024 | 1D/2D/3D 布局 |

---

## 7. Grid / Block 的 1D、2D、3D

> Thread blocks and grids may be 1, 2, or 3 dimensional.

**意思**：线程组织可以是 1/2/3 维，让 thread 索引直接对应数据坐标。

| 数据 | 常用布局 | thread 映射 |
|------|----------|-------------|
| 一维数组 | 1D grid + 1D block | `i` → `a[i]`（vec_add） |
| 二维矩阵 | 2D grid + 2D block | `(row,col)` → `C[row,col]`（mat_mul） |
| 三维体数据 | 3D grid + 3D block | `(x,y,z)` → `data[x,y,z]` |

```cuda
// 2D 示例
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

限制：`blockDim.x * blockDim.y * blockDim.z ≤ 1024`

---

## 8. Warp（Week 1 了解即可）

- **1 Warp = 32 个 thread**（T4 固定）
- GPU 按 **Warp** 调度（SIMT），不是单个 thread
- **写代码仍按 thread 写**，Week 1 不必深入
- `blockDim` 最好用 **32 的倍数**（256 ✅）

| 阶段 | 要求 |
|------|------|
| Week 1 | 知道定义即可 |
| Week 3 | warp shuffle reduction |
| Week 5+ | divergence、优化 |

---

## 9. Programming Model（官方 1.2 消化版）

> **更完整版本**见 [docs/Programming_Model详解.md](../docs/Programming_Model详解.md)

### 主线

```
CPU (Host) 准备数据
  → cudaMemcpy (H2D)
  → launch Kernel（= 1 Grid）
  → 大量 Thread 并行
  → cudaMemcpy (D2H)
  → CPU 校验
```

### 术语对照

| 文档术语 | 白话 | vec_add |
|----------|------|---------|
| Heterogeneous | CPU+GPU 协同 | main 在 CPU，kernel 在 GPU |
| Host | CPU 侧 | `h_a`, `main()` |
| Device | GPU 侧 | `d_a`, `vec_add` |
| Kernel | GPU 上跑的函数 | `__global__ void vec_add` |
| Grid/Block/Thread | 线程层次 | `<<<4096, 256>>>` |
| Global memory | 主显存 | `cudaMalloc(&d_a)` |
| SIMT | 以 warp 为单位执行 | Week 1 知道即可 |

### 第一遍可跳过

- SIMT / Warp divergence 细节
- Shared memory 优化
- Unified Memory、CUDA Graphs
- 详细内存性能分析（Week 2+）

---

## 10. vec_add 与概念对照

```
main()                         → Host
h_a, h_b, h_c                  → Host memory
cudaMalloc(d_*)                → Device global memory
cudaMemcpy(H2D)                → 数据搬到 GPU
vec_add<<<blocks, threads>>>   → launch = 1 Grid
  i = blockIdx.x * blockDim.x + threadIdx.x
  c[i] = a[i] + b[i]           → 每 Thread 一份 work
cudaDeviceSynchronize()        → 等 GPU 完成
cudaMemcpy(D2H)                → 结果搬回 CPU
```

---

## 11. 快递中心类比（帮助记忆）

| CUDA | 类比 |
|------|------|
| GPU | 整个分拣中心 |
| SM (40) | 40 条分拣线 |
| Grid | 今天这一批货 |
| Block | 一包货（一个班组） |
| Thread | 工人，处理 1 个包裹 |

---

## 12. 学习建议

1. **先抓主线**，Grid/SM/Warp 第二层慢慢消化
2. **以代码为主**：开着 `vec_add.cu` 对照文档
3. **笔记三句话够用**：一次 launch = 一个 Grid；每 thread 算一个元素；SM 是硬件执行单元
4. **概念会重复出现**：Week 2 mat_mul、transpose 会巩固 Thread 层次

---

**相关文档**：`docs/Week1详细步骤.md` · `docs/GPU卡型专项学习指南.md`
