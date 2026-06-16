# Week 1 详细步骤清单

> **主题**：GPU 架构与 CUDA 编程模型  
> **预计用时**：5–7 天（每天 3–4 小时业余 / 2–3 天全职）  
> **本周交付**：`week01_basics/` 三个可运行程序 + `notes/week01.md`（含设备表 + GFLOPS 表）

**使用方式**：按 Step 01 → 12 顺序做；每步有「阅读」「动手」「完成标志」；做完一步再勾 `[ ]`。

> **官方文档说明（CUDA 13.x）**  
> NVIDIA 已将主文档重组为 [CUDA Programming Guide v13.x](https://docs.nvidia.com/cuda/cuda-programming-guide/)（5 部分结构）。  
> 下文 **Programming Guide** 均指 **新版 v13.x**，不再使用旧版 Legacy 的 `Ch.1–3` 编号。  
> 旧版对照见文末 **附录 D**；本地白话版见 [Programming_Model详解.md](Programming_Model详解.md)。

---

## Week 1 学习路径（对齐 v13.x）

**原则**：先读 **Part 1 编程模型**（语言无关），再用 **2.1** 写代码；**2.3** 在 Week 1 只读「执行模型 + Global Memory 名字」，优化细节留 Week 2。

### 7 天安排（业余节奏）

| Day | Step | 官方文档（v13.x） | 代码/笔记 |
|-----|------|------------------|-----------|
| 1 | 01–03 | **1. Introduction**（1.1–1.3）+ [Programming_Model详解](Programming_Model详解.md) §1–§8 + **2.1** | 环境确认、跑通 `vec_add` |
| 2 | 04 | **2.5 Asynchronous Execution**（扫读）+ [Event API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html) | `vec_add` H2D/Kernel/D2H 计时 |
| 3 | 05–06 | **5.1 Compute Capabilities** + **1.2 Programming Model** + **2.3**（SIMT/Warp 部分） | `device_query`、手算 blocks、SIMT 笔记 |
| 4 | 07 | **2.7 NVCC** + **2.1/2.3**（页内搜 `cudaGetLastError`） | 故意 launch 错误、`CUDA_CHECK` |
| 5 | 08 | PMPP / CUDA by Example 矩阵乘 + **2.1**（页内搜 `dim3`，可选） | `mat_mul_naive` CPU 基线 |
| 6 | 09–10 | **2.1/2.3** 多维 block + **2.3** Matrix Transpose（Global 版，选读） | `mat_mul_naive` GPU + GFLOPS 表 |
| 7 | 11–12 | **2.3** Kernel Launch and Occupancy | block 对比实验、Week 1 复盘 |

可勾选每日清单：[Week1_Day2-Day5学习清单.md](Week1_Day2-Day5学习清单.md)（已扩展为 Day1–Day7）。

### 2.3 在 Week 1 读什么 / 不读什么

| Week 1 要读 | Week 1 跳过（以后读） |
|-------------|----------------------|
| 页内搜 `threadIdx` / grid / block | Shared Memory 优化实现 |
| SIMT / Warp 执行模型 | Memory Coalescing 深入 |
| Global Memory 是什么 | Shared Memory 版 Transpose |
| Kernel Launch and Occupancy（Step 11） | Atomics、Cooperative Groups |
| Matrix Transpose **Global Memory 版**（Step 09 选读，学 2D 映射） | **2.4 Writing Tile Kernels**（Week 5） |

### Week 1 整章跳过

- **2.2** Intro to CUDA Python
- **2.4** Writing Tile Kernels
- **2.6** Unified Memory（Week 3 浏览）
- **3.** Advanced CUDA 及以后

> 说明：`vec_add.cu` 里的 `reduce_sum` 属于 Week 2 预习，Week 1 知道 shared memory 存在即可，不必深入。

---

## 总览


| Step | 主题                | 预计时间   | 产出                   |
| ---- | ----------------- | ------ | -------------------- |
| 01   | 环境确认              | 30 min | 环境记录                 |
| 02   | 宏观理解 GPU          | 2 h    | 笔记 5 条概念             |
| 03   | 读懂 vec_add        | 2 h    | 能口述数据流               |
| 04   | vec_add 计时实验      | 2 h    | 拷贝/kernel 耗时表        |
| 05   | 设备信息打印            | 2 h    | `device_query` 程序    |
| 06   | 线程层次与 SIMT        | 3 h    | 手绘 grid/block 图      |
| 07   | 错误处理与编译流程         | 1.5 h  | 故意触发一次错误             |
| 08   | mat_mul CPU 基线    | 2 h    | CPU 正确性验证            |
| 09   | mat_mul_naive GPU | 3 h    | GPU PASS + 首次 GFLOPS |
| 10   | GFLOPS 多规模测试      | 2 h    | 性能记录表                |
| 11   | Occupancy 入门      | 2 h    | 能解释 block 大小影响       |
| 12   | 周复盘与交付            | 2 h    | `notes/week01.md` 定稿 |


---

## Step 01：环境确认

### 阅读

- 无需读文档，先看 [T4实战指南.md](T4实战指南.md) 第一节「T4 硬件速览」（10 分钟）

### 动手

```bash
nvidia-smi
nvcc --version
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
# SM 数量 nvidia-smi 不提供，见 Step 05 device_query（cudaGetDeviceProperties）
```

### 完成标志

- [ ] 确认 GPU 为 Tesla T4，CC = 7.5，显存 16384 MiB
- [ ] 确认 `nvcc` 可用（CUDA 13.x）
- [ ] 在 `notes/week01.md` 顶部记下：日期、驱动版本、CUDA 版本

### 交付物

`notes/week01.md` 开头环境信息（可先复制 [week01_template.md](../notes/week01_template.md)）

---

## Step 02：宏观理解 GPU 与 CUDA 是什么

### 阅读（约 1.5–2 h）

**必读**


| 资料                                                                                   | 章节（v13.x）                                              | 重点                                      |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------- | --------------------------------------- |
| [Programming_Model详解.md](Programming_Model详解.md)                                     | **§1–§8 全文**                                            | 推荐先读，建立编程模型整体图                         |
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)       | **1. Introduction to CUDA**（1.1–1.3）                    | 编程模型抽象、CUDA 平台（语言无关）                   |
| 同上                                                                                   | **2.1 Intro to CUDA C++** 前半                            | Host/Device、Kernel、`<<<grid, block>>>` 入门 |


**选读（二选一，快速过）**


| 资料                                          | 章节     |
| ------------------------------------------- | ------ |
| 《CUDA by Example》                           | Ch.1–2 |
| 《Programming Massively Parallel Processors》 | Ch.1–2 |


### 动手

- 用一句话写下：CPU 和 GPU 各自擅长什么
- 列出 CUDA 程序典型 5 步：分配 → 拷贝 H2D → launch → 同步 → 拷贝 D2H

### 完成标志

- [ ] 能说出 **Host** 和 **Device** 的区别
- [ ] 能说出 **Kernel** 是什么（在 GPU 上并行执行的函数）
- [ ] 知道 `<<<grid, block>>>` 是 launch 语法（细节 Step 06 再深入）
- [ ] `notes/week01.md` Day1 写下 3 个新概念

### 交付物

笔记片段：「CUDA 程序数据流 5 步」

---

## Step 03：读懂并跑通 vec_add

### 阅读（约 1 h）


| 资料                | 章节（v13.x）                                         | 重点                                             |
| ----------------- | ------------------------------------------------- | ---------------------------------------------- |
| Programming Guide | **2.1 Intro to CUDA C++**                         | `__global__`、launch、`cudaMalloc` / `cudaMemcpy` |
| 同上                | **2.1** 中 Thread indexing（页内搜 `threadIdx`）        | `threadIdx`、`blockIdx`、`blockDim`、`gridDim` 公式 |
| 本地代码              | [vec_add.cu](../week01_basics/vec_add/vec_add.cu) | 逐行对照文档                                         |


### 动手

```bash
cd /home/qichengjie/workspace/cuda_study/week01_basics/vec_add
make clean && make
./vec_add           # 默认 1M
./vec_add 10000000  # 1000 万
```

对照代码理解：

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程 ID
if (i < n) { ... }                               // 边界处理
```

### 完成标志

- [ ] 程序输出 `result=PASS`
- [ ] 能解释：为何 `blocks = (n + threads - 1) / threads`
- [ ] 能解释：`cudaMalloc` / `cudaMemcpy` / `cudaFree` 各做什么
- [ ] 能解释：为何 kernel 后要有 `cudaGetLastError()` + `cudaDeviceSynchronize()`

### 交付物

在 `notes/week01.md` 画一张简单图：CPU 数组 → H2D → GPU kernel → D2H → 校验

---

## Step 04：vec_add 计时实验（H2D / kernel / D2H）

### 阅读


| 资料                                                                 | 章节（v13.x）              | 重点                                   |
| ------------------------------------------------------------------ | ---------------------- | ------------------------------------ |
| Programming Guide                                                  | **2.5 Asynchronous Execution**（先扫读） | 同步 vs 异步；Stream/Event 概念，Week 3 再深入 |
| [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html) | **Event Management** | `cudaEventCreate/Record/ElapsedTime` |


### 动手

在 `vec_add.cu` 基础上增加 `cudaEvent` 计时，分别测量：

1. H2D 拷贝（两次 `cudaMemcpy` 可合并测或分开）
2. Kernel 执行
3. D2H 拷贝

建议接口：

```bash
./vec_add 1048576   # 1M
./vec_add 16777216  # 16M
```

### 完成标志

- [ ] 输出三段耗时（毫秒）
- [ ] 至少测 2 种规模（1M 和 16M）
- [ ] 能回答：数据变大时，哪一段增长最快？为什么？

### 交付物

`notes/week01.md` 性能表：


| 规模 n | H2D (ms) | Kernel (ms) | D2H (ms) | 合计 (ms) |
| ---- | -------- | ----------- | -------- | ------- |
| 1M   |          |             |          |         |
| 16M  |          |             |          |         |


---

## Step 05：设备信息打印（device_query）

### 阅读


| 资料                               | 章节（v13.x）                            | 重点               |
| -------------------------------- | ------------------------------------ | ---------------- |
| Programming Guide                | **5.1 Compute Capabilities**（Technical Appendices，浏览） | CC 7.5 对应 Turing |
| Runtime API                      | **cudaGetDeviceProperties**          | 结构体各字段含义         |
| [GPU卡型专项学习指南.md](GPU卡型专项学习指南.md) | 第三节 CC 速查                            | sm_75            |


### 动手

新建 `week01_basics/device_query/device_query.cu`，打印至少：


| 字段                       | 对应属性                              |
| ------------------------ | --------------------------------- |
| GPU 名称                   | `prop.name`                       |
| 计算能力                     | `prop.major` / `prop.minor`       |
| SM 数量                    | `prop.multiProcessorCount`        |
| 全局显存                     | `prop.totalGlobalMem`             |
| 最大 block 维度              | `prop.maxThreadsDim[3]`           |
| 每 block 最大线程数            | `prop.maxThreadsPerBlock`         |
| 每 SM 最大 block 数          | `prop.maxBlocksPerMultiProcessor` |
| Warp 大小                  | `prop.warpSize`                   |
| 每 block 最大 shared memory | `prop.sharedMemPerBlock`          |


编译：

```bash
nvcc -O3 -arch=sm_75 -o device_query device_query.cu
./device_query
```

也可对照：

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
```

### 完成标志

- [ ] 程序运行，输出与 `nvidia-smi` 一致（SM 数、显存、CC）
- [ ] T4 上 `warpSize == 32`，`maxThreadsPerBlock == 1024`
- [ ] `notes/week01.md` 设备信息表填满

### 交付物

`week01_basics/device_query/` 目录 + Makefile

---

## Step 06：深入线程层次 — SIMT、Warp、SM

### 阅读（约 2–3 h，本周理论重点）


| 资料                     | 章节（v13.x）                             | 重点                            |
| ---------------------- | ------------------------------------- | ----------------------------- |
| Programming Guide      | **1.2 Programming Model** + **2.3 Writing SIMT Kernels** 前半 | Grid → Block → Thread、SIMT、Warp |
| Programming Guide      | **2.3** 中 Global Memory 小节（浏览）       | 先知道有哪些内存类型，Week 2 深入          |
| [GPU架构图资源.md](GPU架构图资源.md) | Memory Hierarchy 归档配图                 | 可视化辅助（新版正文图较少）                |
| PMPP 或 CUDA by Example | SIMT / Warp 相关小节                      | 一个 Warp = 32 线程，同 warp 执行同一指令 |
| [T4实战指南.md](T4实战指南.md) | 第一节                                   | T4：40 SM、2560 CUDA Cores      |


### 核心概念（读完要能复述）


| 概念                | 一句话                                   |
| ----------------- | ------------------------------------- |
| **SIMT**          | 以 Warp 为单位执行同一条指令，线程有独立数据             |
| **Warp**          | 32 个线程；`threadIdx.x` 连续时同一 warp 相邻    |
| **SM**            | 流多处理器，调度 block；T4 有 40 个              |
| **Occupancy**     | 活跃 warp 数 / SM 最大 warp 容量（Step 11 量化） |
| **block=(256,1)** | 每 block 8 个 warp（256/32）              |


### 动手

1. 对 `vec_add`：若 `threads=256`，`n=1000`，计算 `blocks`、总线程数、最后 block 有多少线程真正工作
2. 在纸上画：1 个 Grid、4 个 Block、每 Block 8 个 Warp 的关系图

### 完成标志

- [ ] 能手写公式：`global_id = blockIdx.x * blockDim.x + threadIdx.x`（1D 情况）
- [ ] 能解释：为何总线程数要 ≥ n，且需要 `if (i < n)` 处理尾部
- [ ] 能解释：Warp 大小 32 对选 `blockDim` 的意义（宜为 32 的倍数）
- [ ] 笔记中有手绘或 ASCII 线程层次图

### 交付物

`notes/week01.md` 中「SIMT / Warp / SM」三节，每节 2–3 句自己的话

---

## Step 07：错误处理与编译链接流程

### 阅读


| 资料                     | 章节（v13.x）                                  | 重点            |
| ---------------------- | ------------------------------------------ | ------------- |
| Programming Guide      | **2.7 NVCC: The NVIDIA CUDA Compiler**     | 编译流程、`-arch=sm_75` |
| 同上                     | **2.1** 或 **2.3**（页内搜 `cudaGetLastError`） | 异步错误检查        |
| Runtime API            | **cudaGetLastError / cudaPeekAtLastError** |               |
| [T4实战指南.md](T4实战指南.md) | 第二节 编译与架构标志                                | `-arch=sm_75` |


### 动手

**A. 理解编译流程**

```bash
# 观察编译产物
cd week01_basics/vec_add
nvcc -O3 -arch=sm_75 -cuda -o /tmp/vec_add.ptx vec_add.cu   # 可选：看 PTX
nvcc -O3 -arch=sm_75 -c -o vec_add.o vec_add.cu
nvcc -O3 -arch=sm_75 -o vec_add vec_add.o
```

**B. 故意制造一次错误**（然后修复）

- 把 `cudaMalloc` 改成故意传错大小或漏掉 `cudaFree` 前访问
- 或 launch `<<<0, 256>>>` 非法配置，观察 `cudaGetLastError()` 报错

**C. 统一 `CUDA_CHECK` 宏**

- 确认三个项目都使用同一套错误检查（可从 `vec_add.cu` 复制到公共头文件 `week01_basics/common/cuda_check.h`）

### 完成标志

- [ ] 能说出：`nvcc` 编译 `.cu` → 生成 host 代码 + device 代码
- [ ] 知道 T4 编译必须带 `-arch=sm_75`（或 CMake `CMAKE_CUDA_ARCHITECTURES 75`）
- [ ] 成功捕获并读懂至少 1 次 CUDA 运行时错误
- [ ] 理解 kernel launch 错误可能异步上报，需要 `cudaGetLastError`

### 交付物

`week01_basics/common/cuda_check.h`（可选但推荐）

---

## Step 08：矩阵乘法 — CPU 基线与正确性

### 阅读


| 资料                     | 章节（v13.x）     | 重点     |
| ---------------------- | ------------- | ------ |
| PMPP / CUDA by Example | 矩阵乘法 introductory 节 | 三重循环语义 |
| Programming Guide      | **2.1** 中多维索引（可选，页内搜 `dim3`） | 为 Step 09 预热 |


### 动手

新建 `week01_basics/mat_mul_naive/`：

1. 先写 **CPU 版本** `matmul_cpu(A, B, C, M, N, K)`
  - `C[M×N] = A[M×K] × B[K×N]`
2. 小规模验证：`M=N=K=4` 手算对比
3. 支持命令行参数：`./mat_mul --m 512 --n 512 --k 512`

### 完成标志

- [ ] CPU 版本 128³ / 512³ 结果正确
- [ ] 理解 GFLOPS 公式：`2*M*N*K / time_seconds / 1e9`
- [ ] 记录 CPU 512³ 耗时（作对比参考，GPU 应快很多）

### 交付物

`mat_mul_naive.cu` 中 CPU 函数 + 单元测试 PASS

---

## Step 09：mat_mul_naive — GPU 实现

### 阅读


| 资料                | 章节（v13.x）                       | 重点                                |
| ----------------- | ------------------------------- | --------------------------------- |
| Programming Guide | **2.1** 或 **2.3** 中 multidimensional blocks（页内搜 `dim3`） | 2D `threadIdx.x/y`、`blockIdx.x/y` |
| 同上                | **2.3** Matrix Transpose Example（Global Memory 版，选读） | 理解 2D 线程映射；transpose 优化留 Week 2 |
| 本地 vec_add        | 对照 Host 端流程                     | 矩阵更大，注意内存分配                       |


### 动手

实现 naive GPU kernel（每线程算 C 的一个元素）：

```cuda
// 每个线程负责 C[row][col] 的一个元素
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
```

建议第一版：

- `blockDim = (16, 16)` → 每 block 256 线程（T4 建议起步配置）
- `gridDim = (ceil(N/16), ceil(M/16))`
- 行主序存储：`A[row*K + k]`，`B[k*N + col]`，`C[row*N + col]`

Host 流程：分配 → H2D → launch → sync → D2H → 与 CPU 结果对比（允许小误差 `1e-2` FP32）

### 完成标志

- [ ] GPU 与 CPU 结果一致（512³）
- [ ] 能跑 512³ 不 OOM（约需 3×512²×4B ≈ 3MB，远小于 16GB）
- [ ] 输出首次 GFLOPS（朴素版通常较低，正常）

### 交付物

`week01_basics/mat_mul_naive/` 可运行，`make && ./mat_mul` PASS

---

## Step 10：GFLOPS 多规模基准测试

### 阅读


| 资料                     | 章节         | 重点           |
| ---------------------- | ---------- | ------------ |
| [T4实战指南.md](T4实战指南.md) | 第三节 性能基准参考 | 朴素 GEMM 预期区间 |
| [项目清单.md](项目清单.md)     | P02 矩阵乘    | 验收标准         |


### 动手

用 `cudaEvent` 只计 **kernel 时间**（不含 H2D/D2H），测试：


| M=N=K | 预期（朴素）                |
| ----- | --------------------- |
| 256   | 记录                    |
| 512   | 记录                    |
| 1024  | 记录，**作为 Week 5 优化基线** |


每个规模跑 3 次取中位数；打印 GFLOPS。

```bash
./mat_mul 256
./mat_mul 512
./mat_mul 1024
```

### 完成标志

- [ ] `notes/week01.md` GFLOPS 表填满
- [ ] 知道 1024³ 朴素 GEMM 在 T4 上大约几十 GFLOPS 量级（远低于 cuBLAS）
- [ ] 能解释：为何矩阵越大，GFLOPS 通常会变化（访存/缓存/占用）

### 交付物


| 规模    | Kernel (ms) | GFLOPS |
| ----- | ----------- | ------ |
| 256³  |             |        |
| 512³  |             |        |
| 1024³ |             |        |


> **重要**：1024³ 的 GFLOPS 数字务必保存，Week 5 优化要对比「提升了多少倍」。

---

## Step 11：Occupancy 基本概念（入门）

**本地详解**：[Occupancy入门详解.md](Occupancy入门详解.md)（概念 + T4 数字 + mat_mul 实验 + API 示例）

### 阅读


| 资料                                                                                         | 章节（v13.x）                                           | 重点        |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------- | --------- |
| Programming Guide                                                                          | **2.3 Writing SIMT Kernels** 中 Kernel Launch and Occupancy | 概念即可      |
| [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | **Occupancy** 章节（前半）                                | 不追求公式推导   |
| Runtime API                                                                                | **cudaOccupancyMaxActiveBlocksPerMultiprocessor**   | 可选 API 实验 |


### 核心理解（本周不要求调优到极致）

- **Occupancy** = SM 上活跃 warp 比例，高 occupancy 有利于隐藏内存延迟
- 但 **高 occupancy ≠ 一定更快**（Week 5 再深入）
- 限制因素：每 block 线程数、寄存器用量、shared memory 用量

### 动手（二选一）

**A. 简单实验**（推荐）  
对 `mat_mul_naive`，试 3 种 block 配置，记录 GFLOPS：


| blockDim | 每 block 线程 | GFLOPS (1024³) |
| -------- | ---------- | -------------- |
| (16,16)  | 256        |                |
| (8,8)    | 64         |                |
| (32,32)  | 1024       |                |


**B. API 查询**（可选）  
调用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 打印不同 block size 的理论最大 active blocks。

### 完成标志

- [ ] 能用一句话解释 Occupancy
- [ ] 知道 T4 `maxThreadsPerBlock = 1024`，故 `(32,32)` 是上限
- [ ] 观察到 block 大小变化会导致性能变化（记录现象即可）
- [ ] 笔记写下：Week 1 对 Occupancy 的理解（3–5 句）

### 交付物

`notes/week01.md` Occupancy 小节 + block 对比小表

---

## Step 12：周复盘与正式交付

### 阅读

- 回顾 [CUDA学习路线图.md](CUDA学习路线图.md) Week 1 验收项
- 扫一眼 [项目清单.md](项目清单.md) P01、P02

### 动手：目录自检

```
week01_basics/
├── vec_add/           ✅ PASS + 计时
├── device_query/      ✅ 设备表打印
├── mat_mul_naive/     ✅ PASS + GFLOPS
└── common/            ⭕ cuda_check.h（推荐）
notes/
└── week01.md          ✅ 定稿
```

### 自测题（闭卷能答再进入 Week 2）

1. CUDA 程序中 Host 和 Device 各指什么？
2. 写出 1D 全局线程 ID 公式。
3. 一个 Warp 多少线程？T4 有多少个 SM？
4. `cudaMemcpy` H2D 和 D2H 方向分别是什么？
5. 为何 kernel  launch 后要 `cudaDeviceSynchronize`？
6. GFLOPS 怎么算？你 1024³ 基线是多少？
7. Occupancy 是什么？高 occupancy 一定更快吗？
8. T4 编译架构标志是什么？

### 完成标志

- [ ] 三个程序均可 `make` 通过
- [ ] `notes/week01.md` 含：环境信息、设备表、vec_add 计时表、GFLOPS 表、本周总结
- [ ] 8 道自测题能答对 ≥ 6
- [ ] 知道 Week 2 主题：内存层次、transpose、reduction

### 交付物

**Week 1 正式结案** — 可打勾：

- [ ] Step 01–12 全部完成
- [ ] 代码在 `week01_basics/`
- [ ] 笔记在 `notes/week01.md`

---

## 附录 A：推荐阅读顺序（与 Step 对照，v13.x）


| Step  | Programming Guide（v13.x）                         | 其他                      |
| ----- | ------------------------------------------------ | ----------------------- |
| 02    | 1.1–1.3, 2.1 前半                                  | Programming_Model详解 §1–§8 |
| 03    | 2.1 Intro to CUDA C++                            | vec_add 源码              |
| 04    | 2.5 Asynchronous Execution（扫读）                   | cudaEvent API           |
| 05    | 5.1 Compute Capabilities（浏览）                   | cudaGetDeviceProperties |
| 06    | 1.2 Programming Model, 2.3 SIMT Kernels 前半       | GPU架构图资源、PMPP          |
| 07    | 2.7 NVCC, 2.1/2.3 错误检查                           | T4 编译节                  |
| 08–09 | 2.1/2.3 多维 block（`dim3`）                         | mat mul 示例              |
| 10    | —                                                | T4 基准表                  |
| 11    | 2.3 Kernel Launch and Occupancy                  | Best Practices Guide    |
| 12    | 复盘 1. Introduction + 2.1–2.3（Week 1 范围）        | 项目清单 P01–P02            |

---

## 附录 D：新版 vs Legacy 章节对照（查阅用）

> 旧链接 [cuda-c-programming-guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 仍可用，但章节结构与新版不同。  
> 若看到第三方教程写 `Ch.3.3 Thread Hierarchy`，请对照下表找新版位置。

| 旧版 Legacy（常见引用） | 新版对应位置 | Week 1 哪步用 |
| ---------------- | -------------- | ---------- |
| Ch.1 Introduction | **1. Introduction to CUDA** | Step 02 |
| Ch.3.1 Programming Model Overview | **1.2 Programming Model** + **2.1** | Step 02–03 |
| Ch.3.2 Kernel Function | **2.1 Intro to CUDA C++** | Step 03, 07 |
| Ch.3.3 Thread Hierarchy | **2.1** + **2.3 Writing SIMT Kernels** | Step 03, 06, 09 |
| Ch.3.4 Memory Hierarchy | **2.3** Global/Shared Memory 各小节 | Step 06（Week 2 深入） |
| Ch.3.6 Compute Capabilities | **5.1 Compute Capabilities** | Step 05 |
| Ch.5 Memory Hierarchy（优化） | **2.3** Coalescing / Shared Memory | Week 2 |
| Ch.6–7 Async / Unified Memory | **2.5**、**2.6**（入门）、**4.1** Unified Memory（完整） | Week 3+ |
| Occupancy Calculator | **2.3** Kernel Launch and Occupancy | Step 11 |
| NVCC 编译 | **2.7 NVCC** | Step 07 |
| C Language Extensions（`__syncthreads`/atomic/shuffle/WMMA） | **5.4 C/C++ Language Extensions**（5.4.4/5.4.5/5.4.6/5.4.11） | Week 2–5 |

> **2026-05 官方更新提示**：Part 3（Advanced CUDA）与 Part 4（CUDA Features，共 20 节）现已完整发布。
> 旧计划里写「Atomics/Warp shuffle 在 2.3」「WMMA 在 5.1」**不再准确**——
> 这些**内联函数的权威定义集中在 5.4**：原子 5.4.5、warp 函数（shuffle/vote/reduce）5.4.6、WMMA/Tensor Core 5.4.11、`__syncthreads`/`__syncwarp`/memory fence 5.4.4。
> Stream-Ordered Allocator、Cooperative Groups、CUDA Graphs、Pipelines/异步拷贝分别在 4.3 / 4.4 / 4.2 / 4.10–4.11。

**Week 1 暂时跳过的新版章节**：

- **2.2 Intro to CUDA Python** — C++ 路线跳过
- **2.4 Writing Tile Kernels** — Week 5 再读（新版 tile 编程模型 `__tile__` / `cuda::tiles`）
- **3. Advanced CUDA / 4. CUDA Features** — Week 3+ 再读


---

## 附录 B：常见卡点


| 现象                        | 可能原因                | 处理                    |
| ------------------------- | ------------------- | --------------------- |
| `invalid device function` | 架构不对                | 加 `-arch=sm_75`       |
| 结果全错                      | 索引公式错 / 行列主序混用      | 先测 4×4 手算             |
| GFLOPS 极低                 | 朴素实现正常              | 记录基线，Week 5 优化        |
| `cudaMalloc` OOM          | n 过大                | `nvidia-smi` 看占用      |
| launch 失败                 | grid/block 为 0 或超上限 | 检查 `(32,32)` 是否超 1024 |


---

## 附录 C：下一步

Week 1 完成后 → [CUDA学习路线图.md](CUDA学习路线图.md) **Week 2**（transpose、reduction、合并访问）

如需生成 `device_query` 和 `mat_mul_naive` 代码骨架，可在对话中说：**帮我生成 Week 1 Step 05/09 代码骨架**。

---

**返回**：[CUDA学习路线图.md](CUDA学习路线图.md)