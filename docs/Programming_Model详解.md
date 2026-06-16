# CUDA Programming Model 详解（易懂版）

> 本文是对 [CUDA Programming Guide v13.x](https://docs.nvidia.com/cuda/cuda-programming-guide/) **Part 1 + Part 2 编程模型** 的重新讲解，  
> 用白话 + 图示 + `vec_add` 对照，降低第一遍阅读官方文档的难度。  
>  
> **配套代码**：`manual_write/vec_add.cu` 或 `week01_basics/vec_add/vec_add.cu`

---

## 如何使用本文

| 阅读方式 | 建议 |
|----------|------|
| 第一遍 | 只读「必读章节」+ 对照 vec_add |
| 第二遍 | Week 1 Step 06 时精读 **2.3 Writing SIMT Kernels**（页内搜 `threadIdx`） |
| 查概念 | 用目录跳转到具体小节 |

**必读章节**：§1–§6、§8（一次读完约 1.5–2 小时）  
**Week 2 再读**：§7 内存层次细节（对照官方 **2.3** Global/Shared Memory）  
**Week 3 再读**：§9 异步与 Stream（对照官方 **2.5 Asynchronous Execution**）

---

## §1  Programming Model 到底是什么？

官方说的 **Programming Model**，不是某一段 API，而是一套 **「你怎么组织并行计算」的规则**。

可以概括成 **三个问题**：

```
1. 代码跑在哪？        → Host (CPU)  vs  Device (GPU)
2. 并行怎么组织？      → Grid → Block → Thread
3. 数据放在哪、怎么搬？ → Host 内存 vs Device 显存，cudaMemcpy
```

你写的每一个 CUDA 程序，都是在回答这三个问题。`vec_add` 是最小完整答案。

---

## §2  异构计算：为什么需要 Host 和 Device？

### 2.1 Heterogeneous Computing（异构计算）

一台有 NVIDIA GPU 的机器里，有两种不同风格的处理器：

| | CPU (Host) | GPU (Device) |
|---|------------|--------------|
| 核心数 | 少（几十核） | 多（数千计算单元） |
| 单个核心 | 很强，复杂逻辑 | 较弱 |
| 擅长 | if/else、调度、IO、串行 | **大量相同的小计算** |
| 内存 | 系统 RAM | 显存 VRAM |
| 你的代码 | `main()` | `vec_add` kernel |

**CUDA 的工作**：让 CPU 和 GPU **分工协作**，而不是只用其中一个。

### 2.2 分工模式（vec_add）

```
CPU 负责：
  - 准备输入数据 h_a, h_b
  - 分配 GPU 显存、拷贝数据
  - 下令「开始算」
  - 取回结果、检查结果

GPU 负责：
  - 收到命令后，百万个 thread 同时做 a[i]+b[i]
```

### 2.3 快递中心类比（帮助记忆）

| CUDA | 类比 |
|------|------|
| CPU (Host) | 仓库管理员：备货、下单、收货验货 |
| GPU (Device) | 分拣车间：大量工人并行处理包裹 |
| 显存 | 车间旁边的货架 |
| Kernel | 今天这批货的分拣规则 |
| cudaMemcpy | 货车在仓库和车间之间运货 |

---

## §3  Host 与 Device

### 3.1 定义

- **Host**：CPU 及其内存（你电脑的主内存）
- **Device**：GPU 及其显存

它们是 **两套独立的内存**，地址不通用。

### 3.2 命名习惯

```cpp
std::vector<float> h_a;   // h = host，在 CPU 内存
float* d_a;               // d = device，在 GPU 显存
cudaMalloc(&d_a, bytes);  // 在显存里分配
```

### 3.3 关键规则（非常重要）

> **Kernel 里访问的指针，必须指向 Device 能访问的内存。**

```cuda
vec_add<<<...>>>(d_a, d_b, d_c, n);   // ✅ d_* 在显存
vec_add<<<...>>>(h_a.data(), ...);     // ❌ h_* 在 CPU 内存，GPU 读不到
```

数据从 CPU 到 GPU，必须经过：
- 显式 `cudaMemcpy`（你现在学的），或
- Unified Memory 等高级方式（Week 3 了解）

### 3.4 vec_add 对照

| 代码 | 跑在哪 | 内存 |
|------|--------|------|
| `main()` | Host | — |
| `h_a`, `h_b`, `h_c` | Host 使用 | CPU RAM |
| `cudaMalloc(&d_a)` | Host 调用，分配在 | GPU VRAM |
| `vec_add<<<...>>>` | Host 发起，在 | GPU 执行 |

---

## §4  Kernel：在 GPU 上跑的函数

### 4.1 什么是 Kernel？

**Kernel** = 用 `__global__` 声明、通过 `<<<grid, block>>>` 启动、在 GPU 上并行执行的函数。

```cuda
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
  // 这段代码会被成千上万个 thread 各执行一遍
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
```

### 4.2 函数修饰符（Week 1 先记两个）

| 修饰符 | 调用位置 | 执行位置 | 示例 |
|--------|----------|----------|------|
| `__global__` | Host 调用 | Device 执行 | `vec_add` kernel |
| `__device__` | Device 调用 | Device 执行 | device 辅助函数（以后用） |

普通 `void foo()` 没有修饰符 → Host 函数，给 `main` 用。

### 4.3 Launch 语法

```cuda
vec_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
//       └──grid──┘ └block┘
//       几个 block    每 block 几个 thread
```

等价于：

```cuda
dim3 grid(blocks, 1, 1);
dim3 block(threads, 1, 1);
vec_add<<<grid, block>>>(d_a, d_b, d_c, n);
```

**一次 `<<<>>>` = 启动一个 Kernel = 创建一个 Grid。**

### 4.4 Kernel 执行后的同步

```cuda
vec_add<<<blocks, threads>>>(...);  // 异步：发出命令
cudaDeviceSynchronize();            // 等待 GPU 算完
cudaMemcpy(h_c.data(), d_c, ...);   // 再取结果
```

Launch 后 CPU 默认可以继续往下走；要读 GPU 结果前，必须先同步。

---

## §5  可扩展编程模型（Scalable Programming Model）

### 5.1 官方想说什么？

你只需要写：

```cuda
int blocks = (n + 255) / 256;
vec_add<<<blocks, 256>>>(...);
```

**不用关心** T4 有 40 个 SM、A100 有 108 个 SM——硬件调度器会把 block 分到可用 SM 上。

### 5.2 对你意味着什么？

| 你写的（逻辑） | 硬件做的（物理） |
|----------------|------------------|
| 我要 4096 个 block | 40 个 SM 轮流/并行消化这些 block |
| 每 block 256 thread | 每个 SM 上挂多个 block（受资源限制） |
| 同一份代码 | T4、A100 都能跑（改编译架构） |

### 5.3 注意

「可扩展」指 **block/thread 逻辑结构** 可适配不同 GPU，不是指 **性能自动一样**。大 GPU 通常更快，但代码结构可以不变。

---

## §6  线程层次：Thread、Block、Grid

这是 Programming Model 的 **核心**，建议对照 vec_add 反复读。

### 6.1 三层结构

```
一次 Kernel Launch
        │
        ▼
      Grid          ← 本次启动的全部工作（逻辑）
        │
   ┌────┼────┬────┐
   ▼    ▼    ▼    ▼
 Block Block Block ...   ← 一组 Thread（逻辑）
   │    │    │
 256T 256T 256T          ← Thread：最小执行单位
```

| 层级 | 谁定义 | vec_add (n=1M) | 固定吗？ |
|------|--------|----------------|----------|
| **Grid** | `<<<blocks, ...>>>` 第一参数 | 4096 blocks | 每次 launch 可变 |
| **Block** | 隐式，grid 的每个单元 | 4096 个 | 随 grid 变 |
| **Thread** | `<<<..., threads>>>` 第二参数 | 每 block 256 个 | 你设定（≤1024） |

### 6.2 内置变量

每个 thread 在 kernel 里自动拥有：

| 变量 | 含义 | vec_add 中 |
|------|------|------------|
| `threadIdx.x` | block 内第几个 thread | 0 ~ 255 |
| `blockIdx.x` | 第几个 block | 0 ~ 4095 |
| `blockDim.x` | 每 block 多少 thread | 256 |
| `gridDim.x` | 一共多少 block | 4096 |

（还有 `.y`、`.z` 用于 2D/3D，vec_add 只用 `.x`。）

### 6.3 全局下标公式（1D，必背）

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

**含义**：把「第几个 block + block 内位置」换算成「数组全局下标」。

**例子**：

| blockIdx | threadIdx | 计算 | i | 负责 |
|----------|-----------|------|---|------|
| 0 | 0 | 0×256+0 | 0 | c[0] |
| 0 | 255 | 0×256+255 | 255 | c[255] |
| 1 | 0 | 1×256+0 | 256 | c[256] |
| 2 | 10 | 2×256+10 | 522 | c[522] |

图示：

```
数组:  [0 ... 255][256 ... 511][512 ... 767] ...
       └──Block 0──┘└─Block 1──┘└─Block 2──┘
```

### 6.4 边界处理

```cuda
int blocks = (n + threads - 1) / threads;  // 向上取整
// ...
if (i < n) { ... }  // 防止 i 超出数组
```

总 thread 数 `blocks × threads` 可能 ≥ n，多出来的 thread 必须跳过。

### 6.5 Block 之间 vs Block 之内

| | Block **之内** | Block **之间** |
|---|----------------|----------------|
| 同步 | ✅ `__syncthreads()` | ❌ 不能直接同步 |
| 共享内存 | ✅ 可共享（Week 2） | ❌ 不共享 |
| 协作 | ✅ 适合队内合作 | 需 atomic 等（Week 3） |

vec_add 里每个 thread **独立**算一个 `c[i]`，block 之间无合作，是最简单模式。

### 6.6 1D / 2D / 3D

Grid 和 Block 都可以是 1、2 或 3 维，目的是 **让 thread 坐标对齐数据形状**。

**1D（vec_add）**——一维数组：

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

**2D（mat_mul，Week 1 后面）**——二维矩阵：

```cuda
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
// thread (row, col) 负责 C[row, col]
```

**3D**——体数据 `[z][y][x]`，以后再用。

限制：`blockDim.x × blockDim.y × blockDim.z ≤ 1024`（T4）。

### 6.7 Grid 固定吗？

**不固定。** 每次 launch 你指定；`n` 不同，`blocks` 就不同。  
**SM 数量才固定**（T4 = 40）。

---

## §7  内存层次（Programming Model 中的预告）

官方 **1.2 Programming Model** / **2.3** 会提前列出内存类型，第一遍 **知道名字和位置** 即可，优化细节 Week 2 学。

### 7.1 层次图

```
每个 Thread
  ├── Registers（寄存器，最快，私有）
  └── Local Memory（线程私有，实际常在慢内存）

每个 Block
  └── Shared Memory（块内共享，很快，48KB/block on T4）

整个 GPU
  └── Global Memory（显存，16GB，cudaMalloc 在这）
        ↑ vec_add 的 d_a, d_b, d_c
```

### 7.2 Week 1 只需掌握

| 类型 | vec_add 用了吗 | 一句话 |
|------|----------------|--------|
| **Global** | ✅ `d_a` | 主显存，kernel 读写的主战场 |
| **Shared** | ❌ | block 内共享，Week 2 transpose 用 |
| **Registers** | 自动 | 编译器把 `i` 等放寄存器，不用你管 |

### 7.3 Host 与 Device 内存搬运

```
h_a (Host RAM)  --cudaMemcpy H2D-->  d_a (Global VRAM)
                                         │
                                    vec_add kernel
                                         │
h_c (Host RAM)  <--cudaMemcpy D2H--  d_c (Global VRAM)
```

**方向**：

| 宏 | 方向 |
|----|------|
| `cudaMemcpyHostToDevice` | CPU → GPU |
| `cudaMemcpyDeviceToHost` | GPU → CPU |
| `cudaMemcpyDeviceToDevice` | GPU → GPU（以后 pipeline 用） |

---

## §8  SIMT 与 Warp（第一遍了解即可）

### 8.1 SIMT 是什么？

**SIMT** = Single Instruction, Multiple Threads  
硬件以 **Warp（32 个 thread）** 为单位，执行同一条指令，各自操作自己的数据。

```
Warp 里 32 个 thread 同时做：c[i] = a[i] + b[i]
但每个 thread 的 i 不同
```

### 8.2 和你写代码的关系

- 你 **按 thread 写代码**，不必写 warp
- `blockDim = 256` = 8 个 warp（256÷32），是好习惯
- vec_add 无 `if` 分支差异 → 无 **warp divergence** 问题

### 8.3 Week 1 要记什么

| 现在记 | 以后学 |
|--------|--------|
| 1 warp = 32 threads | warp shuffle（Week 3） |
| blockDim 用 32 的倍数 | divergence 优化（Week 5） |

---

## §9  异步执行与 Stream（Week 3 再读）

Programming Model 会提到：kernel launch 是 **异步** 的。

```cuda
vec_add<<<...>>>();     // CPU 发出命令后可能立即返回
// CPU 可以干别的...
cudaDeviceSynchronize(); // 显式等待 GPU
```

**Stream** = 一串按顺序执行的 GPU 操作队列。  
多 Stream 可以让 **拷贝和计算重叠**（Week 3 实战）。  
Week 1 用 `cudaDeviceSynchronize()` 就够。

---

## §10  完整 vec_add 流程图（Programming Model 全景）

```
┌─────────────────────────────────────────────────────────────┐
│                        Host (CPU)                           │
│                                                             │
│  1. 准备 h_a, h_b        ← 数据在 Host Memory               │
│  2. cudaMalloc(d_*)      ← 在 Device Global Memory 分配     │
│  3. cudaMemcpy H2D       ← Host → Device 搬运               │
│  4. vec_add<<<g,b>>>()   ← 启动 Kernel，创建 Grid           │
│  5. cudaDeviceSynchronize()                                 │
│  6. cudaMemcpy D2H       ← Device → Host 取结果             │
│  7. 校验 h_c                                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ launch
┌─────────────────────────────────────────────────────────────┐
│                       Device (GPU)                          │
│                                                             │
│  Grid: 4096 blocks × 256 threads/block                      │
│    每个 Thread: i = blockIdx.x * blockDim.x + threadIdx.x │
│    if (i < n) c[i] = a[i] + b[i]                           │
│                                                             │
│  数据: d_a, d_b, d_c 在 Global Memory                       │
│  调度: 40 个 SM 执行各 Block                                │
└─────────────────────────────────────────────────────────────┘
```

---

## §11  逐行对照表（vec_add）

### Host 代码

| 行 | 代码 | Programming Model 含义 |
|----|------|------------------------|
| 13 | `int n = 1<<20` | 问题规模 |
| 14 | `h_a, h_b, h_c` | Host 内存中的数据 |
| 27-29 | `cudaMalloc` | Device Global Memory 分配 |
| 30-31 | `cudaMemcpy H2D` | Host → Device 数据搬运 |
| 33-34 | `threads=256, blocks=...` | 定义 Grid/Block 规模 |
| 35 | `vec_add<<<blocks, thread>>>` | Kernel Launch = 创建 Grid |
| 37 | `cudaDeviceSynchronize` | Host 等待 Device 完成 |
| 40 | `cudaMemcpy D2H` | Device → Host 取结果 |

### Device 代码（Kernel）

| 行 | 代码 | 含义 |
|----|------|------|
| 4 | `__global__ void vec_add` | 这是 Kernel |
| 5 | `blockIdx * blockDim + threadIdx` | Thread 全局下标 |
| 6 | `if (i < n)` | 边界保护 |
| 7 | `c[i] = a[i] + b[i]` | 每 Thread 一份独立工作 |

---

## §12  官方术语 → 白话 速查表

| 官方英文 | 白话 | vec_add |
|----------|------|---------|
| Heterogeneous computing | CPU+GPU 一起干活 | main + kernel |
| Host | CPU 侧 | `main`, `h_a` |
| Device | GPU 侧 | `vec_add`, `d_a` |
| Kernel | GPU 并行函数 | `__global__ vec_add` |
| Grid | 一次 launch 的全部 block | 4096 blocks |
| Block | 一组 thread | 256 threads/block |
| Thread | 最小执行单位 | 算一个 `c[i]` |
| Warp | 32 个 thread 的硬件调度单位 | 每 block 8 个 warp |
| SM | GPU 上执行 block 的硬件单元 | T4 有 40 个 |
| Global memory | 主显存 | `d_a` |
| SIMT | 以 warp 为单位执行指令 | 自动，不用手写 |
| Scalable model | 同代码适配不同 GPU | T4/A100 都能跑 |

---

## §13  常见问题 FAQ

### Q1: 为什么不能让 kernel 直接用 h_a？

Host 指针指向 CPU 内存，GPU 在显存地址空间里访问会出错。必须 `d_a` 或在特殊内存（Unified）里。

### Q2: 一次 launch 几个 Grid？

**一个。** 多次 launch 才有多个 Grid。

### Q3: thread 和 CUDA Core 是一一对应吗？

不是。Thread 是逻辑概念；硬件用 Warp 调度，一个 SM 上同时活跃很多 thread。

### Q4: block 和 SM 是一一对应吗？

不是。多个 block 会被分配到 40 个 SM 上，一个 SM 可同时挂多个 block。

### Q5: 一定要先 cudaMalloc 吗？

经典路径：**是**。数据要在 Global Memory 里 kernel 才能高效访问。

### Q6: `-arch=sm_75` 和 Programming Model 什么关系？

Programming Model 是逻辑规则；`-arch=sm_75` 是把 kernel **编译成 T4 能执行的指令**，属于实现层。

---

## §14  学习检验（自测）

能答对下面 8 题，说明 Programming Model 第一遍已过关：

1. Host 和 Device 分别是什么？
2. `__global__` 函数在哪里执行？
3. 写出 1D 全局 thread 下标公式。
4. 一次 `<<<>>>` 对应几个 Grid？
5. 为什么 vec_add 要 `cudaMemcpy`？
6. `blockIdx.x`、`threadIdx.x`、`blockDim.x` 各表示什么？
7. `if (i < n)` 为什么需要？
8. Global memory 和 Shared memory 区别（一句话）？

---

## §15  与官方文档的对应关系（v13.x）

| 本文章节 | Programming Guide v13.x | Legacy 旧版（对照） |
|----------|-------------------------|---------------------|
| §1–§2 | **1. Introduction to CUDA**（1.1–1.3） | Ch.1 Introduction |
| §3–§4 | **2.1 Intro to CUDA C++** | Ch.3.1–3.2 |
| §5 | **1.2 Programming Model** | Ch.1.3 Scalable Programming Model |
| §6 | **2.1** + **2.3 Writing SIMT Kernels**（页内搜 `threadIdx`） | Ch.3.3 Thread Hierarchy |
| §7 | **2.3** Global/Shared Memory 各小节 | Ch.3.4 Memory Hierarchy |
| §8 | **2.3** SIMT / **5.8** Execution Model（选读） | Ch.3.2 / Appendix SIMT |
| §9 | **2.5 Asynchronous Execution** | Ch.6 Async |

**建议**：读完本文 §1–§8 后，再回官方 **2.1 + 2.3** 读一遍，会轻松很多。  
Legacy 对照详见 [Week1详细步骤.md 附录 D](Week1详细步骤.md#附录-d新版-vs-legacy-章节对照查阅用)。

---

## §16  第一遍 vs 后续：什么时侯再读

| 内容 | 第一遍 | 何时深入 |
|------|--------|----------|
| Host/Device/Kernel | ✅ 掌握 | — |
| Grid/Block/Thread | ✅ 掌握 | mat_mul 2D 巩固 |
| Global + cudaMemcpy | ✅ 掌握 | — |
| Shared memory | 知道名字 | Week 2 |
| Warp/SIMT | 知道定义 | Week 3 |
| Stream 异步 | 知道即可 | Week 3 |
| Occupancy | 知道即可 | Week 5 |
| Unified Memory | 跳过 | Week 3 了解 |

---

**相关文档**：[Week1详细步骤.md](Week1详细步骤.md) · [notes/CUDA基础概念.md](../notes/CUDA基础概念.md) · [CUDA学习路线图.md](CUDA学习路线图.md)
