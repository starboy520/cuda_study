# Week 1 每日学习清单（Day1–Day7）

> **使用方式**：按 Day1 → Day7 顺序勾选。每 Day 对应 [Week1详细步骤.md](Week1详细步骤.md) 中的 Step，并标明 v13.x 官方章节。  
> **官方文档**：[CUDA Programming Guide v13.x](https://docs.nvidia.com/cuda/cuda-programming-guide/)  
> **Legacy 对照**：[Week1详细步骤.md 附录 D](Week1详细步骤.md#附录-d新版-vs-legacy-章节对照查阅用)

### v13.x 阅读顺序（Week 1 总览）

```text
Day1  Part 1 (1.1–1.3) + Programming_Model详解 + 2.1
Day2  2.5（扫读）+ Event API
Day3  5.1 + 1.2 + 2.3（SIMT/Warp）
Day4  2.7 NVCC
Day5  2.1（dim3 可选）+ 书籍矩阵乘
Day6  2.1/2.3 2D + Transpose Global（选读）+ matmul GPU
Day7  2.3 Occupancy + 复盘
```

**Week 1 跳过**：2.2 Python、2.4 Tile Kernels、2.6、Part 3–4

---

## 本周总览

- [ ] Day1（Step 01–03）：环境 + 编程模型 + 跑通 `vec_add`
- [ ] Day2（Step 04）：`vec_add` H2D / Kernel / D2H 计时
- [ ] Day3（Step 05–06）：设备属性 + Grid/Block/Thread/Warp/SM
- [ ] Day4（Step 07）：错误处理 + `nvcc` 编译流程
- [ ] Day5（Step 08）：`mat_mul_naive` CPU 基线
- [ ] Day6（Step 09–10）：`mat_mul_naive` GPU + GFLOPS 记录
- [ ] Day7（Step 11–12）：Occupancy 入门 + Week 1 复盘
- [ ] `notes/week01.md` 每天都有：目标、实验、问题

---

## Day1：环境 + 编程模型 + vec_add

**对应 Step**：01–03

### 今天要学什么

- [ ] Host / Device、Kernel、`<<<grid, block>>>`
- [ ] CUDA 程序 5 步：分配 → H2D → launch → 同步 → D2H
- [ ] 1D 全局线程 ID：`blockIdx.x * blockDim.x + threadIdx.x`

### 需要看的资料

- [ ] [Programming_Model详解.md](Programming_Model详解.md) §1–§8
- [ ] Programming Guide：**1. Introduction to CUDA**（1.1–1.3）
- [ ] Programming Guide：**2.1 Intro to CUDA C++**
- [ ] [T4实战指南.md](T4实战指南.md) 第一节

### 动手步骤

```bash
nvidia-smi && nvcc --version
cd week01_basics/vec_add && make && ./vec_add
```

- [ ] `notes/week01.md` 填写环境信息
- [ ] 能口述 `vec_add` 数据流

### 完成标准

- [ ] `result=PASS`
- [ ] 能解释 `cudaMalloc` / `cudaMemcpy` / `cudaFree`

---

## Day2：`vec_add` 计时实验

**对应 Step**：04

### 今天要学什么

- [ ] 理解 CUDA 程序里的三段耗时：
  - H2D：Host to Device，把 CPU 内存复制到 GPU 显存。
  - Kernel：GPU 执行 `vec_add<<<blocks, threads>>>`。
  - D2H：Device to Host，把 GPU 结果复制回 CPU 内存。
- [ ] 理解 `cudaEvent` 的基本用法：
  - `cudaEventCreate`
  - `cudaEventRecord`
  - `cudaEventSynchronize`
  - `cudaEventElapsedTime`
  - `cudaEventDestroy`
- [ ] 理解同步计时的边界：要测哪段，就把 start / stop event 放在哪段前后。

### 需要看的资料

- [ ] `docs/Week1详细步骤.md`：Step 04
- [ ] [CUDA Programming Guide v13.x](https://docs.nvidia.com/cuda/cuda-programming-guide/)：**2.5 Asynchronous Execution**（先扫读）
- [ ] [CUDA Runtime API — Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
- [ ] 本地代码：`week01_basics/vec_add/vec_add.cu`

### 动手步骤

- [ ] 打开 `week01_basics/vec_add/vec_add.cu`，找到三段位置：
  - `cudaMemcpy(d_a, ...)` 和 `cudaMemcpy(d_b, ...)`
  - `vec_add<<<blocks, threads>>>(...)`
  - `cudaMemcpy(h_c.data(), ...)`
- [ ] 在 H2D 前后加 event，测两次 H2D 拷贝的总耗时。
- [ ] 在 kernel launch 前后加 event，测 kernel 耗时。
- [ ] 在 D2H 前后加 event，测结果拷回耗时。
- [ ] 编译运行默认规模：

```bash
cd /home/qichengjie/workspace/cuda_study/week01_basics/vec_add
make clean && make
./vec_add 1048576
```

- [ ] 编译运行 16M 规模：

```bash
./vec_add 16777216
```

- [ ] 确认输出仍然包含 `result=PASS`。
- [ ] 把 1M 和 16M 的耗时填入 `notes/week01.md` 的 `vec_add 计时（Step 04）` 表格。

### 建议输出格式

```text
n=1048576, H2D=..., Kernel=..., D2H=..., Total=..., result=PASS
n=16777216, H2D=..., Kernel=..., D2H=..., Total=..., result=PASS
```

### 完成标准

- [ ] 程序输出 H2D / Kernel / D2H 三段耗时。
- [ ] 1M 和 16M 都运行并记录。
- [ ] `notes/week01.md` 中 `cudaEvent 计时（Step 04）` 可以打勾。
- [ ] 能回答：数据量变大时，哪一段增长最明显？为什么？

### 自测问题

- [ ] `cudaMemcpyHostToDevice` 和 `cudaMemcpyDeviceToHost` 分别是什么意思？
- [ ] 为什么 kernel 计时后通常要同步？
- [ ] `cudaEventElapsedTime` 的单位是什么？
- [ ] H2D 和 D2H 是计算还是数据传输？

---

## Day3：设备信息与线程层次

**对应 Step**：05–06

> **2.3 今日只读**：SIMT、Warp、Global Memory 名字。**不读** Shared Memory 优化、Coalescing、Shared 版 Transpose。

### 今天要学什么

- [ ] 记住 T4 的几个关键数字：
  - Compute Capability：7.5
  - SM 数：40
  - Warp size：32
  - Max threads per block：1024
  - 常用起步 block size：`256` 或 `(16,16)`
- [ ] 理解 1D 线程索引：

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

- [ ] 理解为什么需要边界判断：

```cuda
if (i < n) {
  c[i] = a[i] + b[i];
}
```

- [ ] 理解 SIMT / Warp / SM：
  - SIMT：一个 warp 内线程执行同一条指令，但处理不同数据。
  - Warp：32 个线程，是 GPU 调度执行的重要单位。
  - SM：Streaming Multiprocessor，负责调度和执行 block。

### 需要看的资料

- [ ] `docs/Week1详细步骤.md`：Step 05、Step 06
- [ ] `docs/T4实战指南.md`：T4 硬件速览、编译架构标志
- [ ] `docs/Programming_Model详解.md`：线程层次相关章节
- [ ] [CUDA Programming Guide v13.x](https://docs.nvidia.com/cuda/cuda-programming-guide/)：**1.2 Programming Model** + **2.3 Writing SIMT Kernels** 前半（页内搜 `threadIdx`）
- [ ] **2.3** 中 Global Memory 小节（浏览，Week 2 深入）
- [ ] **5.1 Compute Capabilities**（Appendix，配合 CC 7.5）
- [ ] `notes/week01.md`：设备属性表和概念笔记

### 动手步骤

- [ ] 检查当前设备信息：

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
```

- [ ] 对照 `vec_add.cu` 的 `print_device_shared_memory()` 输出，确认 shared memory per block 和 per SM 的区别。
- [ ] 手算：`n=1000, threads=256` → `blocks=4`，总线程 1024，最后 block 232 个线程真正工作
- [ ] 在 `notes/week01.md` 里补一段自己的话：
  - SIMT 是什么
  - Warp 是什么
  - SM 是什么
  - 为什么 block size 常选 32 的倍数
- [ ] 画一个简单 ASCII 图或手绘图，表达：

```text
Grid
├── Block 0: 256 threads = 8 warps
├── Block 1: 256 threads = 8 warps
├── Block 2: 256 threads = 8 warps
└── ...
```

### 完成标准

- [ ] 能写出 1D 全局线程 ID 公式。
- [ ] 能解释 `blocks = (n + threads - 1) / threads`。
- [ ] 能解释为什么 `if (i < n)` 必须存在。
- [ ] `notes/week01.md` 有 SIMT / Warp / SM 三段自己的总结。

### 自测问题

- [ ] 一个 warp 有多少线程？
- [ ] T4 有多少个 SM？
- [ ] `threadIdx.x` 是 block 内索引还是 grid 内索引？
- [ ] `blockIdx.x * blockDim.x + threadIdx.x` 为什么能得到全局索引？
- [ ] 为什么 `threads=250` 通常不如 `threads=256` 合适？

---

## Day4：错误处理与编译流程

**对应 Step**：07

### 今天要学什么

- [ ] 理解 `CUDA_CHECK` 宏的作用：每次 CUDA runtime API 调用后立刻检查错误。
- [ ] 理解 kernel launch 之后为什么要写：

```cuda
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

- [ ] 理解 CUDA 的异步错误：kernel launch 本身可能先返回，真正执行错误要同步后才暴露。
- [ ] 理解 `nvcc -arch=sm_75`：为 T4 的 compute capability 7.5 编译合适的 device code。

### 需要看的资料

- [ ] `docs/Week1详细步骤.md`：Step 07
- [ ] `docs/T4实战指南.md`：编译与架构标志
- [ ] CUDA Runtime API：`cudaGetLastError`、`cudaPeekAtLastError`、`cudaDeviceSynchronize`
- [ ] [CUDA Programming Guide v13.x](https://docs.nvidia.com/cuda/cuda-programming-guide/)：**2.7 NVCC** + **2.1/2.3**（页内搜 `cudaGetLastError`）
- [ ] 本地代码：`week01_basics/vec_add/vec_add.cu` 中的 `CUDA_CHECK`

### 动手步骤

- [ ] 正常编译一次，确认当前程序可运行：

```bash
cd /home/qichengjie/workspace/cuda_study/week01_basics/vec_add
make clean && make
./vec_add
```

- [ ] 观察 `nvcc` 生成中间文件：

```bash
nvcc -O3 -arch=sm_75 -cuda -o /tmp/vec_add.cpp.ii vec_add.cu
nvcc -O3 -arch=sm_75 -c -o /tmp/vec_add.o vec_add.cu
nvcc -O3 -arch=sm_75 -o /tmp/vec_add /tmp/vec_add.o
```

- [ ] 故意制造一次 launch 错误，例如临时把：

```cuda
vec_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
```

改成：

```cuda
vec_add<<<0, threads>>>(d_a, d_b, d_c, n);
```

- [ ] 重新编译运行，观察错误信息。
- [ ] 恢复正确代码。
- [ ] 再次编译运行，确认程序恢复 `PASS`。
- [ ] 在 `notes/week01.md` 的 Day4 记录这次错误：
  - 你故意改了什么
  - 报错是什么
  - 哪一行检查发现了错误
  - 恢复后如何确认正常

### 完成标准

- [ ] 能说出 `nvcc` 编译 `.cu` 时同时处理 host code 和 device code。
- [ ] 能说出为什么 T4 使用 `-arch=sm_75`。
- [ ] 成功捕获并读懂至少 1 次 CUDA runtime error。
- [ ] 能解释 `cudaGetLastError()` 和 `cudaDeviceSynchronize()` 的作用区别。

### 自测问题

- [ ] `cudaGetLastError()` 检查的是哪类错误？
- [ ] `cudaDeviceSynchronize()` 为什么会让异步错误暴露出来？
- [ ] 如果漏掉错误检查，CUDA 程序可能出现什么问题？
- [ ] `invalid configuration argument` 通常意味着什么？

---

## Day5：`mat_mul_naive` CPU 基线

**对应 Step**：08

> Day5 只做 CPU 三重循环；GPU kernel 放到 Day6。

### 今天要学什么

- [ ] 矩阵乘法三重循环语义
- [ ] 行主序索引：`A[row*K+k]`、`C[row*N+col]`
- [ ] GFLOPS 公式（先理解，Day6 再测 GPU）

### 需要看的资料

- [ ] `docs/Week1详细步骤.md`：Step 08
- [ ] PMPP 或 CUDA by Example：矩阵乘法入门
- [ ] Programming Guide **2.1**（页内搜 `dim3`，可选预习 2D）

### 动手步骤：CPU baseline only

- [ ] 新建 `week01_basics/mat_mul_naive/`
- [ ] 写 `matmul_cpu`，`M=N=K=4` 手算验证
- [ ] `./mat_mul 512` CPU 结果正确

### 完成标准

- [ ] CPU 512³ PASS
- [ ] 理解 GFLOPS 公式

---

## Day6：mat_mul GPU + GFLOPS

**对应 Step**：09–10

### 今天要学什么

- [ ] 2D grid/block：`col` 用 x 算，`row` 用 y 算
- [ ] naive GPU matmul：每线程算 `C` 的一个元素
- [ ] 用 `cudaEvent` 只测 kernel 时间

### 需要看的资料

- [ ] `docs/Week1详细步骤.md`：Step 09、Step 10
- [ ] Programming Guide **2.1/2.3**（页内搜 `dim3`）
- [ ] Programming Guide **2.3** Matrix Transpose（Global Memory 版，选读）
- [ ] `week01_basics/vec_add/vec_add.cu`：复用 Host 端流程

### 动手步骤：GPU naive kernel

- [ ] 写 GPU kernel：

```cuda
__global__ void matmul_naive(const float* A, const float* B, float* C,
                             int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}
```

- [ ] `dim3 block(16,16)`，`grid` 按 M/N 向上取整
- [ ] Host 流程：H2D → launch → sync → D2H → 与 CPU 对比

### 动手步骤：GFLOPS 记录

```bash
cd week01_basics/mat_mul_naive
make clean && make
./mat_mul 256
./mat_mul 512
./mat_mul 1024
```

- [ ] 填入 `notes/week01.md` GFLOPS 表

### 完成标准

- [ ] GPU 512³ 与 CPU 一致
- [ ] 记录 256³ / 512³ / 1024³ kernel GFLOPS（1024³ 作 Week 5 基线）

---

## Day7：Occupancy + Week 1 复盘

**对应 Step**：11–12

### 今天要学什么

- [ ] Occupancy 是什么：SM 上活跃 warp 比例
- [ ] 高 occupancy ≠ 一定更快
- [ ] Week 1 自测 8 题（见 Week1详细步骤 Step 12）

### 需要看的资料

- [ ] Programming Guide **2.3** Kernel Launch and Occupancy
- [ ] [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) Occupancy 前半
- [ ] `docs/Week1详细步骤.md`：Step 11–12

### 动手步骤

- [ ] 对 `mat_mul_naive` 试 block `(16,16)` / `(8,8)` / `(32,32)`，记录 1024³ GFLOPS
- [ ] 目录自检：`vec_add/`、`device_query/`、`mat_mul_naive/`、`notes/week01.md`
- [ ] 闭卷自测 ≥ 6/8 题

### 完成标准

- [ ] Occupancy 笔记 3–5 句
- [ ] Week 1 正式结案勾选

---

## 本周最终交付

- [ ] `notes/week01.md` Day1–Day7 日志完整
- [ ] vec_add 计时表（1M / 16M）已填
- [ ] SIMT / Warp / SM 有自己的解释
- [ ] 至少一次 CUDA 错误制造与恢复记录
- [ ] `mat_mul_naive` GPU PASS + 1024³ GFLOPS 基线
- [ ] Step 12 自测 ≥ 6/8 题

---

## 时间不够时的取舍

1. **必做**：Day1–Day3（编程模型 + 计时 + 线程层次）
2. **必做**：Day6 GPU matmul + 1024³ GFLOPS 基线
3. **尽量做**：Day4 错误处理
4. **可后补**：Day7 Occupancy 实验（概念 Step 11 要知道）

---

## 学习节奏建议

- [ ] 每完成一个 Day，在 `notes/week01.md` 写：学了什么 / 实验结果 / 卡在哪里
- [ ] 代码改完就运行，不攒问题
- [ ] `vec_add.cu` 里的 `reduce_sum` 是 Week 2 预习，Week 1 不深入
