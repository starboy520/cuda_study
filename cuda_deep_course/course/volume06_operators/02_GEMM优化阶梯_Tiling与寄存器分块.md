# 02 GEMM 优化阶梯：Tiling 与寄存器分块

> 上一章算出：naive GEMM 的算术强度只有 0.25，严重带宽受限。本章爬第一段优化阶梯
> ——用 **shared-memory tiling** 和 **register tiling** 把算术强度抬上去，让 GEMM
> 从"等数据"变成"忙着算"。每一步都讲清动机、机制和提升量级。

## 0. 这章先怎么读

这章很容易被几个词卡住：

```text
tile
shared memory
K 维分段
As / Bs
register tiling
BM / BN / BK / TM
```

先不要急着背。你只需要抓住一条主线：

```text
GEMM 优化 = 让读进来的数据多用几次。
```

每一级优化都在问同一个问题：

```text
我花时间把一个 A 或 B 元素读进来了，
能不能别只用一次？
能不能给更多 C 输出一起用？
```

本章两级优化的区别是：

| 优化 | 数据先放在哪里 | 谁复用它 | 主要解决什么 |
|---|---|---|---|
| Shared-memory tiling | Shared memory | 一个 block 的很多 thread | 少读 global memory |
| Register tiling | Register | 一个 thread 自己 | 少读 shared memory |

所以阅读顺序是：

```text
先看 naive 为什么重复读
再看 shared memory 为什么能让 block 共同复用
再看一个 block 如何算一个 C tile
最后看一个 thread 如何用寄存器算多个 C 输出
```

## 1. 回顾 Naive：每个线程太“单打独斗”

### 1.1 GEMM 到底在算什么

GEMM 公式：

```text
C[M,N] = A[M,K] × B[K,N]
```

一个输出元素是：

```text
C[row][col] =
    A[row][0] * B[0][col]
  + A[row][1] * B[1][col]
  + A[row][2] * B[2][col]
  + ...
  + A[row][K-1] * B[K-1][col]
```

也就是：

```text
A 的一行
乘
B 的一列
得到
C 的一个点
```

Naive kernel 的想法很直接：

```text
一个 thread 负责一个 C[row][col]
这个 thread 自己去 global memory 读 A 的一行、B 的一列
自己算完
自己写回 C
```

```cpp
// Naive GEMM（卷二第 09 章）
__global__ void gemmNaive(const float* A, const float* B, float* C,
                          int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];   // 每个输出独立读 A 一行、B 一列
    }
    C[row * N + col] = sum;
  }
}
```

### 1.2 Naive 的病根一：A 被重复读

假设同一行有 4 个输出：

```text
C[row][0]
C[row][1]
C[row][2]
C[row][3]
```

它们分别由 4 个 thread 计算。每个输出都需要 A 的同一行：

```text
C[row][0] 需要 A[row][0..K-1]
C[row][1] 需要 A[row][0..K-1]
C[row][2] 需要 A[row][0..K-1]
C[row][3] 需要 A[row][0..K-1]
```

也就是说：

```text
A[row][0] 被多个 thread 反复从 global memory 读取。
A[row][1] 也被多个 thread 反复从 global memory 读取。
...
```

这很浪费，因为 global memory 远、慢、贵。

### 1.3 Naive 的病根二：B 也缺少显式复用

在 naive 代码里：

```cpp
B[k * N + col]
```

对单个 thread 来说，`k` 每次加 1，它会沿着 B 的一列往下走：

```text
B[0][col]
B[1][col]
B[2][col]
...
```

这一列数据会被很多 C 的行重复使用。

例如同一列上有 4 个输出：

```text
C[row0][col]
C[row1][col]
C[row2][col]
C[row3][col]
```

它们都会用到：

```text
B[k][col]
```

Naive 做法中，这些不同 thread 会各自从 global memory 读取自己需要的 B。即使某些
warp 访问在某个时刻可以合并，整体上仍然没有像 shared tiling 那样把 B 的一个小块
显式放到 shared memory 里给多行、多列输出反复复用。

### 1.4 两个病根合起来

Naive GEMM 的问题不是数学错了，而是数据搬运方式太笨：

```text
病根① 重复读：C 的一行 N 个输出，每个都把 A 的【同一行】完整读一遍 → A 被读 N 次
病根② B 缺少显式复用：同一个 B[k][col] 会被多个输出需要，naive 中常被重复读取
```

两个病根都指向同一个解法：**把数据先搬进 shared memory，让一个 block 协作复用。**

## 2. Shared-Memory Tiling：先把“共享厨房”讲清楚

### 2.1 Shared Memory 在这里扮演什么角色

Shared memory 是一个 block 内线程共享的、比较快的片上存储。

你可以先把它理解成：

```text
global memory = 远处仓库
shared memory = 当前 block 的工作台
register      = 每个 thread 自己手里的小纸条
```

一个 block 内的线程可以合作：

```text
先一起从 global memory 搬一小块 A、B 到 shared memory
再从 shared memory 反复读取这小块数据来计算
```

这件事的关键不是“shared memory 神奇地让数学变少”，而是：

```text
global memory 读少了
同一份数据被更多计算复用
```

### 2.2 Tiling 一句话

```text
Tiling = 把"每个线程各自反复从 global 读同一份数据"
        改成"一个 block 协作把数据读一次进 shared，大家共用"
```

生活类比：一个小组要做 16 道菜，每道都要切同一种洋葱。

```text
naive 做法：16 个人，每人自己跑超市买一份洋葱(16 次往返超市)→ 累在路上
tiling 做法：派 1 个人买一大份洋葱放厨房台面(1 次往返)，16 人从台面取(快)
→ "超市" = global memory(远、慢)；"台面" = shared memory(近、快)
→ 关键：同一份数据，买一次，大家共用很多次 → 省掉重复的远程访问
```

为什么 GEMM 特别适合这招？因为它的数据复用度极高：

```text
算 C 的一行(N 个输出)，每个输出都要读 A 的【同一行】
→ A 的一行被 N 个输出各读一遍(naive 重复 N 次)
→ 如果把 A 这一行(的一段)放进 shared，N 个输出共用 → 省 N 倍 global 读
```

### 2.3 从 C 的一个小块开始想

先不要想整张大矩阵。只看 C 的一个小块：

```text
C tile:

        col0  col1
row0    C00   C01
row1    C10   C11
```

假设这个 tile 是 `2×2`。一个 block 负责算这 4 个输出。

这 4 个输出需要的数据有重叠：

```text
C00 和 C01 都需要 A[row0][k]
C10 和 C11 都需要 A[row1][k]

C00 和 C10 都需要 B[k][col0]
C01 和 C11 都需要 B[k][col1]
```

所以一个 block 可以这样合作：

```text
先把这 4 个 C 输出共同需要的一小块 A 放到 shared
再把共同需要的一小块 B 放到 shared
然后 4 个 thread 一起从 shared 里取数据算
```

这就是 shared-memory tiling 的根：

```text
不是每个 thread 自己去 global memory 找材料
而是一个 block 先把大家都会用的材料搬到 shared memory
```

### 2.4 为什么要沿 K 方向分段

GEMM 的求和方向是 K：

```text
C[row][col] = Σ A[row][k] * B[k][col]
```

如果 K 很大，比如 1024，那么算一个 C tile 需要：

```text
A 的 TILE 行 × 1024 列
B 的 1024 行 × TILE 列
```

这太大，不可能一次全放进 shared memory。

所以做法是：

```text
把 K 切成很多小段。
每次只处理一段。
这一段处理完，把部分和累加到 sum。
再处理下一段。
```

例如 `TILE=4, K=12`：

```text
完整 K:  0 1 2 3 | 4 5 6 7 | 8 9 10 11
          t=0段   |  t=1段  |   t=2段
```

每一段做同样的事情：

```text
1. 从 global 搬 A 的一个小块到 As
2. 从 global 搬 B 的一个小块到 Bs
3. __syncthreads() 等大家搬完
4. 用 As 和 Bs 算这一段对 C 的贡献
5. __syncthreads() 等大家算完，再覆盖 shared 装下一段
```

这就是代码里外层 `t` 循环的含义。

### 2.5 As 和 Bs 是什么

代码里会看到：

```cpp
__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];
```

先把它们读成：

```text
As = A shared tile
Bs = B shared tile
```

它们不是新矩阵，也不是最终结果。它们只是当前 block 的临时工作台：

```text
As 暂时存 A 的一个小块
Bs 暂时存 B 的一个小块
```

每处理一个 K 段，As/Bs 会装入新内容：

```text
t=0: As/Bs 装 K=0..TILE-1 这一段
t=1: As/Bs 装 K=TILE..2*TILE-1 这一段
t=2: As/Bs 装下一段
```

所以 shared memory 是反复复用的临时空间。

### 2.6 为什么这样能提升算术强度

关键在**复用**：一个 `TILE×TILE` 的 A 子块被 load 进 shared 一次后，block 内
`TILE` 列的输出都用它——**一次 global 读，TILE 次复用**。算术强度因此提升约 TILE 倍：

```text
naive:        每个 A 元素从 global 读 1 次，用 1 次  → AI ≈ 0.25
tiling(TILE): 每个 A 元素从 global 读 1 次，用 TILE 次 → AI ≈ 0.25 × TILE

TILE=16 → AI ≈ 4 ；TILE=32 → AI ≈ 8 ... 离拐点 25 更近了
```

如果“算术强度”这个词还不熟，先把它读成：

```text
每搬 1 字节数据，能做多少计算。
```

数值越高，说明数据搬来以后被用得更充分。

### 2.7 代码骨架

先看变量表，再看代码会轻松很多：

| 名字 | 读成 | 意义 |
|---|---|---|
| `TILE` | tile size | C 小块的边长，也是每次 K 段的长度 |
| `As` | A shared tile | A 当前 K 段的小块，放在 shared memory |
| `Bs` | B shared tile | B 当前 K 段的小块，放在 shared memory |
| `row` | output row | 当前 thread 负责的 C 行 |
| `col` | output col | 当前 thread 负责的 C 列 |
| `t` | K tile index | 当前正在处理第几个 K 段 |
| `k` | inner K index | 当前 K 段内部的第几个元素 |
| `sum` | accumulator | 当前 C[row][col] 的部分和，放在寄存器里 |

这一版代码有一个故意简化：

```text
一个 thread 只算一个 C 输出。
```

后面的 register tiling 才会改成：

```text
一个 thread 算多个 C 输出。
```

> **⚠️ 前提与约束（这一版骨架成立的隐含契约，不满足会越界/算错）**
>
> ```text
> 1. block 必须是 TILE×TILE：dim3 block(TILE, TILE)
>    → 因为 As/Bs 是 TILE×TILE，用 threadIdx.y/x 当下标，
>      block 大了(如 32×32 而 TILE=16)→ As[31][..] 越界崩；小了 → tile 填不满
>    → 线程数(TILE×TILE) == tile 格子数 → 每线程正好搬 1 个 A + 1 个 B，一一对应
> 2. grid 按"覆盖整个 C"算：
>    dim3 grid((N+TILE-1)/TILE, (M+TILE-1)/TILE)  // x 管列(N)、y 管行(M)
> 3. 一个 block ↔ C 的一个 TILE×TILE 输出块 ↔ As/Bs 一块，三者尺寸完全一致。
> 4. 每线程只算 1 个 C 输出(到 §3 register tiling 才会"一个线程算多个")。
> 5. M/N/K 可以不是 TILE 整数倍 → 靠加载/写回的边界判断 + 填 0 兜底。
> ```
>
> 对应的 host 启动：
> ```cpp
> dim3 block(TILE, TILE);
> dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
> gemmTiled<<<grid, block>>>(dA, dB, dC, M, N, K);
> ```

```cpp
#define TILE 16
// 约定：调用方必须用 dim3 block(TILE, TILE) 启动（见上方"前提与约束"）
__global__ void gemmTiled(const float* A, const float* B, float* C,
                          int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;   // 该线程算的 C 行（依赖 blockDim.y==TILE）
  int col = blockIdx.x * TILE + threadIdx.x;   // 该线程算的 C 列（依赖 blockDim.x==TILE）
  float sum = 0.0f;

  // 沿 K 维一段一段处理
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {   // 每个 t = K 方向的一个 tile（一个"段/括号"）
    // ① 协作加载：每个线程搬一个 A 元素、一个 B 元素进 shared
    int aCol = t * TILE + threadIdx.x;
    int bRow = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
    __syncthreads();                         // ② 等整块加载完

    // ③ 从 shared 做部分点积（复用！）—— 算"这一段(这个 tile)"的部分和
    for (int k = 0; k < TILE; ++k) {         // 全局 K 下标 = t*TILE + k
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];   // sum 不清零 → 跨 t 持续累加
    }
    __syncthreads();                         // ④ 等大家算完再 load 下一块
  }

  if (row < M && col < N) C[row * N + col] = sum;
}
```

### 2.8 把加载索引讲清楚（最容易晕的地方）

这段代码最难的是 `aCol`、`bRow` 这两个加载索引。我们用"线程负责哪个 C 元素 →
它该读 A 哪行、B 哪列"的思路一步步推。

```text
一个 block 算 C 的一个 TILE×TILE 块；线程 (tx,ty) 算这块里的 C[row][col]：
  row = blockIdx.y*TILE + ty   (C 的行)
  col = blockIdx.x*TILE + tx   (C 的列)
```

第 t 步处理 K 维的第 t 段（K 下标范围 [t*TILE, t*TILE+TILE)）：

```text
加载 A 子块 As[ty][tx]：
  它该装 A[row][t*TILE + tx]  → A 的第 row 行、第 (t*TILE+tx) 列
  压平：A[row*K + (t*TILE+tx)]，所以 aCol = t*TILE + tx ✓
  (As[ty][tx] 的第二维 tx 对应 A 的列方向)

加载 B 子块 Bs[ty][tx]：
  它该装 B[t*TILE + ty][col]  → B 的第 (t*TILE+ty) 行、第 col 列
  压平：B[(t*TILE+ty)*N + col]，所以 bRow = t*TILE + ty ✓
  (Bs[ty][tx] 的第一维 ty 对应 B 的行方向)
```

**为什么 A 用 tx、B 用 ty？** 因为 As/Bs 都是 `[ty][tx]` 索引，但：
```text
A 子块是"TILE 行 × TILE 列"，列方向(K 方向)要随 tx 走 → aCol 含 tx
B 子块是"TILE 行 × TILE 列"，行方向(K 方向)要随 ty 走 → bRow 含 ty
→ A、B 的 K 方向"长在不同维度上"(A 的 K 是列、B 的 K 是行)，所以一个用 tx 一个用 ty
```

用 `TILE=2` 看一眼会更直观。四个 thread 的加载任务是：

```text
thread (tx=0, ty=0):
  As[0][0] = A[row0][t*2 + 0]
  Bs[0][0] = B[t*2 + 0][col0]

thread (tx=1, ty=0):
  As[0][1] = A[row0][t*2 + 1]
  Bs[0][1] = B[t*2 + 0][col1]

thread (tx=0, ty=1):
  As[1][0] = A[row1][t*2 + 0]
  Bs[1][0] = B[t*2 + 1][col0]

thread (tx=1, ty=1):
  As[1][1] = A[row1][t*2 + 1]
  Bs[1][1] = B[t*2 + 1][col1]
```

所以：

```text
As 的每一行来自 A 的一行
Bs 的每一行来自 B 的一行
```

只是计算时会用：

```text
As[ty][k] 取 A 的当前输出行
Bs[k][tx] 取 B 的当前输出列
```

### 2.9 为什么计算是 As[ty][k] * Bs[k][tx]

```text
C[row][col] = Σ_K A[row][K] * B[K][col]
本段(第 t 段)的部分和 = Σ_{k=0..TILE-1} A[row][t*TILE+k] * B[t*TILE+k][col]

而 As[ty][k] 装的是 A[row][t*TILE+k]   (上面 aCol=t*TILE+tx，这里内层用 k)
   Bs[k][tx] 装的是 B[t*TILE+k][col]   (上面 bRow=t*TILE+ty，这里内层用 k)
→ As[ty][k] * Bs[k][tx] = A[row][t*TILE+k] * B[t*TILE+k][col] ✓
→ 内层 k 循环把这一段的部分点积累加进 sum，K/TILE 段后得完整 C[row][col]
```

> 验证索引对不对的通用方法(和转置那篇一样)：写出每个数组下标的代数表达式，
> 代入化简，对照定义 `C=Σ A·B`。不用追踪具体线程。

### 2.10 最该想透的 t 与 K：每个 t 段 = K 的一个 tile

外层 `t` 和内层 `k` 看着是两层循环，其实合起来就是"把 K 从 0 走到 K-1"。
最好懂的方式：**把完整求和按 TILE 拆成一段一段，每段就是一个 t、就是一个 K 方向的 tile。**

```text
完整定义：
  C[row][col] = Σ_{K=0}^{K-1} A[row][K] * B[K][col]    (K 项相加)

按 TILE 分段拆开（加法结合律，把 K 项分组）：
  C[row][col]
    = ( A[row][0]·B[0][col]        + ... + A[row][TILE-1]·B[TILE-1][col] )    ← t=0 段
    + ( A[row][TILE]·B[TILE][col]  + ... + A[row][2TILE-1]·B[2TILE-1][col] )  ← t=1 段
    + ( A[row][2TILE]·...                                                  )  ← t=2 段
    + ...
      └──────────── 每个括号 = 一个 K 方向的 tile = 一次外层 t ────────────┘

每个括号内部的那串加法  →  就是内层 k 循环（k=0..TILE-1）算的"部分和"
把所有括号相加          →  就是外层 t 循环把每段部分和累进 sum（sum 不清零）
```

**对应关系一眼看清：**

```text
  数学上的"第 t 个括号"   ⇄   代码里"外层第 t 次迭代"   ⇄   K 方向第 t 个 tile
        段内第 k 项        ⇄        内层第 k 次乘加        ⇄   该 tile 内第 k 列/行

  全局 K 下标 = t * TILE + k   ← t 选哪一段，k 选段内哪一项
```

> **一句话**：分段求和 = 整段求和（加法结合律），所以"每段算一点、累加起来"和
> "一口气全算"结果完全一样。**一个 t 段 = K 的一个 tile = 一个括号**，
> 外层 t 负责"换段"，内层 k 负责"把这段加完"，`sum` 全程不清零把所有段攒在一起。

### 2.11 一步 tile 的完整流程（具体走查）

以 TILE=2、block(0,0)、算 C 左上角 2×2 块为例，看第 t=0 步：

```text
4 个线程 (tx,ty)：(0,0)(1,0)(0,1)(1,1)，分别算 C[0][0] C[0][1] C[1][0] C[1][1]

① 协作加载(每线程搬 1 个 A + 1 个 B 元素)：
   As[0][0]=A[0][0]  As[0][1]=A[0][1]   ← A 左上 2×2
   As[1][0]=A[1][0]  As[1][1]=A[1][1]
   Bs[0][0]=B[0][0]  Bs[0][1]=B[0][1]   ← B 左上 2×2
   Bs[1][0]=B[1][0]  Bs[1][1]=B[1][1]
② __syncthreads()  等 8 个元素都搬完
③ 每线程内层 k=0,1 算部分和(全从 shared 读，复用！)：
   C[0][0] += As[0][0]*Bs[0][0] + As[0][1]*Bs[1][0]
   C[0][1] += As[0][0]*Bs[0][1] + As[0][1]*Bs[1][1]
   ... → As[0][0] 被 C[0][0] 和 C[0][1] 都用了(复用 2 次=TILE 次)
④ __syncthreads()  等大家算完，再 load 下一段 K
→ t=1,2... 滑过 K 方向所有段，累加得最终 C
```

### 2.12 跨 block：整张 C 是怎么拼出来的（全局视角）

前面 §2.8~2.11 都聚焦"**一个** block 怎么算出 C 的**一个** TILE×TILE 块"。
但 C 是 M×N 的大矩阵，一个 tile 只是其中一小格——整张 C 靠**很多 block 各算各的、
拼起来**。这就是"跨 block"。

**① C 被切成网格，一个 block 包一格：**

```text
C (M×N) 被切成 (M/TILE) × (N/TILE) 个 TILE×TILE 小块：

         col 方向 (N) →  grid.x = N/TILE 个
        ┌──────┬──────┬──────┬──────┐
        │block │block │block │block │   ← blockIdx.y=0 这一排
row(M)  │(0,0) │(1,0) │(2,0) │(3,0) │     算 C 的第 0 条 TILE 高的横带
  ↓     ├──────┼──────┼──────┼──────┤
grid.y= │block │block │block │block │   ← blockIdx.y=1
M/TILE  │(0,1) │(1,1) │(2,1) │(3,1) │
个      ├──────┼──────┼──────┼──────┤
        │ ...  │      │      │      │
        └──────┴──────┴──────┴──────┘

block(bx,by) 负责 C 的行 [by*TILE, by*TILE+TILE)、列 [bx*TILE, bx*TILE+TILE)
→ 这就是 row = blockIdx.y*TILE+ty、col = blockIdx.x*TILE+tx 的来源
```

**② 块与块之间互不通信、各算各的（关键！）**

```text
block(0,0) 和 block(1,0) 算的是 C 不同的格子 → 输出不重叠 → 不需要互相等、互相读
每个 block：
  - 有自己的 As/Bs（shared 是 block 私有的，别的 block 看不到）
  - 自己沿 K 从头滑到尾（§2.11 那套 t 循环）
  - 自己把结果写回 C 自己那一格
→ __syncthreads() 只同步"块内" TILE×TILE 个线程，从不跨 block
→ 几百上千个 block 由 GPU 调度器铺到各 SM 上并行/排队跑，顺序无所谓
```

**③ 对比"块内 K 滑动" vs "跨块"两个维度（别混了）：**

```text
                何时发生            谁参与            靠什么协作
块内沿 K 滑动   一个 block 内部      该 block 的线程    shared + __syncthreads
(§2.11 的 t)    累加出 1 个 C tile

跨 block        不同 block 之间      不同 SM 上的 block  不协作！各写各的 C 格子
(本节)          拼出整张 C          完全独立并行
```

> **一句话**：K 方向的"分期累加"是**块内**的事（一个 block 把自己那格的 K 全吃完）；
> 而"算完整张 C"是**跨 block**的事（每个 block 包一格，互不干扰地并行）。
> 二者正交：先有跨 block 切出"我负责哪格"，再有块内沿 K 滑动"把这格算完"。

**④ 为什么 GEMM 能跨 block 完全独立？** 因为 C 的每个输出元素只属于一个 block，
没有"两个 block 往同一个 C 位置写"的情况 → 无需原子操作、无需跨 block 同步。
（对比 reduction/histogram 那种多 block 往同一处累加，才需要 atomicAdd 或二次 kernel。）

### 2.13 三个关键点（每个都对应前面学的）

- **加载阶段更规整**：`A[row*K + aCol]` 按 A 的行段搬，`B[bRow*N + col]` 按 B 的行段搬；
  更重要的是，搬进 shared 后，A/B 的同一小块会被整个 block 反复使用。
- **两个 `__syncthreads()` 缺一不可**：第一个保证 load 完才算（不然读到没加载的）；
  第二个保证算完才覆盖 shared（不然下一轮 load 冲掉还在用的数据）。这是卷四的 race。
- **边界用 0 填充**：非 TILE 整除的尺寸，越界位置填 0（加法单位元，不影响结果）。

### 2.14 Shared Tiling 小白检查表

读完 shared tiling，你至少要能回答：

```text
1. 一个 block 负责 C 的哪一块？
2. 为什么不能把整个 K 都放进 shared？
3. 外层 t 循环在换什么？
4. As 和 Bs 是最终结果吗？
5. 为什么 load 完要 __syncthreads()？
6. 为什么算完一段后还要 __syncthreads()？
7. sum 为什么不能在每个 t 里清零？
```

答案用最短的话说：

```text
1. 一个 TILE×TILE 的 C tile。
2. K 可能很大，shared 放不下整条 A/B 带。
3. 换 K 方向的下一段。
4. 不是，是当前 block 的临时 shared 小块。
5. 防止有人还没搬完，别人就开始读。
6. 防止有人还没算完，别人就覆盖 shared 装下一段。
7. 每个 t 只算部分和，sum 要跨 t 累加成完整点积。
```

## 3. Register Tiling：再上一个台阶

### 3.1 shared tiling 还不够

shared tiling 把 AI 提到了几倍，但**每个线程仍只算 1 个输出**。瓶颈转移了：现在
线程频繁从 **shared memory** 读数据（虽然比 global 快，但仍有延迟和 bank 压力）。

### 3.2 先从 1D 点积开始：一个输出到底怎么算

先忘掉 block、warp、shared memory，只看数学：

```text
C[row][col] =
    A[row][0] * B[0][col]
  + A[row][1] * B[1][col]
  + A[row][2] * B[2][col]
  + ...
```

这就是一个点积：

```text
A 的一行 · B 的一列 = C 的一个元素
```

用代码写就是：

```cpp
float sum = 0.0f;
for (int k = 0; k < K; ++k) {
  sum += A[row * K + k] * B[k * N + col];
}
C[row * N + col] = sum;
```

这个 `sum` 一般会放在寄存器里。也就是说，最朴素的 GEMM 里，每个 thread 至少已经
在用一个寄存器保存自己的部分和。

### 3.3 从一个输出变成两个输出

现在看两个相邻的输出：

```text
C[row0][col] = A[row0][0]*B[0][col] + A[row0][1]*B[1][col] + ...
C[row1][col] = A[row1][0]*B[0][col] + A[row1][1]*B[1][col] + ...
```

注意它们有一个共同点：

```text
它们是同一列 col
所以都会用到同一个 B[k][col]
```

把第 `k` 步单独拿出来：

```text
C[row0][col] += A[row0][k] * B[k][col]
C[row1][col] += A[row1][k] * B[k][col]
```

这里 `B[k][col]` 完全一样。如果一个 thread 同时算这两个 C 元素，就可以：

```cpp
float acc0 = 0.0f;   // C[row0][col] 的累加器，寄存器
float acc1 = 0.0f;   // C[row1][col] 的累加器，寄存器

for (int k = 0; k < K; ++k) {
  float b = B[k * N + col];      // 读一次 B，放进寄存器
  acc0 += A[row0 * K + k] * b;   // 复用 b
  acc1 += A[row1 * K + k] * b;   // 再复用 b
}
```

这就是 register tiling 的最小直觉：

```text
让一个 thread 多算几个相关输出
把共同需要的数据先读进寄存器
然后在寄存器里复用
```

为什么叫 register tiling？

```text
tile      = 小块
register  = 寄存器

register tiling = 一个 thread 在寄存器里维护一个很小的 C 输出块
```

如果这个 thread 算 2 个输出，就需要 2 个累加器：

```cpp
float acc[2];
```

如果算 8 个输出，就需要 8 个累加器：

```cpp
float acc[8];
```

如果算 `4×4` 个输出，就需要 16 个累加器：

```cpp
float acc[4][4];
```

这些累加器就是 register tiling 里最核心的“寄存器小块”。

### 3.4 为什么先学 1D Register Tiling

`TM×TN` 的 2D register tiling 一上来会很绕，因为一个 thread 同时负责一个小矩形：

```text
4 行 × 4 列 = 16 个 C 输出
```

我们先学 1D 版本：

```text
一个 thread 只负责同一列上的 TM 个输出
```

例如 `TM=4`：

```text
同一个 thread 负责：

C[row+0][col]
C[row+1][col]
C[row+2][col]
C[row+3][col]
```

画成图：

```text
        同一列 col
           ↓
        C[0][col]   ← acc[0]
        C[1][col]   ← acc[1]
        C[2][col]   ← acc[2]
        C[3][col]   ← acc[3]
```

在某个 `k` 上，它们都需要同一个：

```text
B[k][col]
```

所以代码核心会长这样：

```cpp
float b = B[k * N + col];       // 读一次 B 到寄存器

acc[0] += A[(row + 0) * K + k] * b;
acc[1] += A[(row + 1) * K + k] * b;
acc[2] += A[(row + 2) * K + k] * b;
acc[3] += A[(row + 3) * K + k] * b;
```

一句话：

```text
B 读 1 次，用 4 次。
```

如果 `TM=8`：

```text
B 读 1 次，用 8 次。
```

这就是 1D register tiling 最重要的收益。

### 3.5 再推广到 2D Register Tiling

等 1D 版看懂后，2D 版只是再同时复用 A：

```text
1D register tiling:
  同一列多个行
  主要复用 B

2D register tiling:
  多个行 × 多个列
  A 和 B 都复用
```

例如一个 thread 算 `TM×TN = 4×4` 个输出：

```text
        col0   col1   col2   col3
row0    acc00  acc01  acc02  acc03
row1    acc10  acc11  acc12  acc13
row2    acc20  acc21  acc22  acc23
row3    acc30  acc31  acc32  acc33
```

某个 `k` 上：

```text
需要 4 个 A:
  A[row0][k], A[row1][k], A[row2][k], A[row3][k]

需要 4 个 B:
  B[k][col0], B[k][col1], B[k][col2], B[k][col3]
```

然后做 `4×4=16` 次乘加：

```cpp
for (int i = 0; i < TM; ++i) {
  for (int j = 0; j < TN; ++j) {
    acc[i][j] += regA[i] * regB[j];
  }
}
```

这一步的读写账是：

```text
从 shared 读 4 个 A + 4 个 B = 8 次读
做 16 次 FMA
```

所以“每次 shared 读取换来的计算量”更高。

本章后面先实现 1D 版本，因为它最容易写对；2D 版本放到你真正理解 1D 后再爬。

### 3.6 代价与权衡

register tiling 不是免费的——**它吃寄存器**。`TM×TN` 个累加器 + 临时寄存器，会让
每线程寄存器用量上升，从而**降低 occupancy**（卷五）：

```text
TM=TN=4 → 16 个累加器 + 临时 → 每线程几十个寄存器
→ 每 SM 能驻留的线程变少 → occupancy 下降
但：每线程干的活变多（16 个输出），用 ILP（指令级并行）也能隐藏延迟
→ 即使 occupancy 降了，整体仍可能更快
```

这是 GEMM 优化最微妙的权衡：**occupancy ↓ 但每线程效率 ↑**，要靠实测找平衡点。这
也解释了卷五为什么反复强调"高 occupancy ≠ 一定更快"。

### 3.7 动手实现：1D Thread Tiling 一步步（最易写对的第一步）

> §3.2~3.5 已经把 register tiling 拆成了 1D 和 2D 两种直觉。我们先做
> **最经典的 1D 版**：
> **每个线程算一【列】TM 个输出**(纵向叠 TM 个)。这就是 siboehm 教程的 Kernel 4，
> 一步就能从 SMEM tiling 再快 2~3 倍。先把这个写对，2D 只是它的自然推广。
>
> 参考：siboehm《How to Optimize a CUDA Matmul Kernel》
> https://siboehm.com/articles/22/CUDA-MMM （代码 github.com/siboehm/SGEMM_CUDA）

#### 第 0 步：和上一版(基础 tiling)的唯一区别

```text
基础 tiling：一个线程算 1 个 C 输出
1D tiling： 一个线程算 TM 个 C 输出(竖着排成一列)
→ 关键收益:把 B 的一个值读进【寄存器】,复用给这一列的 TM 个输出
  → SMEM 读次数 / 计算次数 从 2:1 降到约 1:1 → 逃离"SMEM 读取"瓶颈
```

#### 第 1 步：参数与"一个 block 算多大"（设计契约）

```text
参数:  BM=64   BN=64   BK=8   TM=8
含义:  一个 block 算 C 的 BM×BN = 64×64 输出块
       K 维一次切 BK=8 厚
       每个线程算 TM=8 个输出(BM 方向上连续的 8 个)

线程数 = (BM/TM) × BN = (64/8) × 64 = 8 × 64 = 512 个线程
       → 注意:线程数 512 < 输出块 64×64=4096,所以每线程算 4096/512=8 个 ✓
shared: As[BM][BK]=64×8=512 个,  Bs[BK][BN]=8×64=512 个
```

> **为什么 1D 好写**：BM×BK=512、BK×BN=512，正好都等于线程数 512
> → 加载阶段【每个线程刚好搬 1 个 As + 1 个 Bs】，不需要加载循环（2D 才需要）。

这些参数的名字可以这样读：

| 名字 | 慢慢读成 | 表示什么 |
|---|---|---|
| `BM` | Block M | 一个 block 负责 C 的多少行 |
| `BN` | Block N | 一个 block 负责 C 的多少列 |
| `BK` | Block K | 每次沿 K 方向吃多厚的一段 |
| `TM` | Thread M | 一个 thread 沿 M 方向算多少个输出 |

所以：

```text
BM×BN = 一个 block 负责的 C 输出块大小
BK    = 每次搬进 shared 的 K 方向厚度
TM    = 一个 thread 竖着算几个 C 输出
```

本章 1D 版本固定：

```text
一个 block 算 C 的 64 行 × 64 列
一个 thread 算同一列上的 8 行
所以 block 内需要：
  64 列 × (64 行 / 每 thread 8 行)
= 64 × 8
= 512 个 thread
```

#### 第 2 步：把 512 个线程"摆"成两种视角

线程用一维 `threadIdx.x`（0~511）启动，但要在两个场合解释成二维：

```text
【计算视角】这个线程算 C 块里哪一列、哪一组行?
  threadCol = threadIdx.x % BN     // 0~63,它负责的列
  threadRow = threadIdx.x / BN     // 0~7, 它负责的"第几组"(每组 TM=8 行)
  → 它算 C 的列 threadCol、行 [threadRow*TM, threadRow*TM+TM) 这 8 行

【加载视角】这个线程搬 As/Bs 的哪一格?
  As(64行×8列): innerRowA = threadIdx.x / BK;  innerColA = threadIdx.x % BK
  Bs(8行×64列): innerRowB = threadIdx.x / BN;  innerColB = threadIdx.x % BN
  → 512 线程正好铺满 As(512格)和 Bs(512格),一人一格
```

这里有一个非常重要的点：

```text
同一个 threadIdx.x
在“计算 C”时有一套解释
在“加载 shared”时有另一套解释
```

这不是矛盾。因为同一个 thread 可以先做搬运工，再做计算工。

#### 计算视角：我负责哪些 C

假设：

```text
BN = 64
TM = 8
threadIdx.x = 130
```

计算：

```text
threadCol = 130 % 64 = 2
threadRow = 130 / 64 = 2
```

意思是：

```text
这个 thread 负责 C tile 中第 2 列
负责第 2 组行
每组有 TM=8 行
```

所以它负责的行是：

```text
threadRow * TM + 0 = 16
threadRow * TM + 1 = 17
...
threadRow * TM + 7 = 23
```

也就是：

```text
C tile 内：
  C[16][2]
  C[17][2]
  C[18][2]
  C[19][2]
  C[20][2]
  C[21][2]
  C[22][2]
  C[23][2]
```

它们都在同一列，所以后面可以复用同一个 `Btmp`。

#### 加载视角：我负责搬 shared 的哪一格

同一个 `threadIdx.x=130`，加载 A tile 时：

```text
innerRowA = 130 / BK = 130 / 8 = 16
innerColA = 130 % BK = 130 % 8 = 2
```

所以它搬：

```text
As[16][2]
```

加载 B tile 时：

```text
innerRowB = 130 / BN = 130 / 64 = 2
innerColB = 130 % BN = 130 % 64 = 2
```

所以它搬：

```text
Bs[2][2]
```

一开始会觉得“同一个 thread 怎么又是 C[16..23][2]，又是 As[16][2]，
又是 Bs[2][2]”。答案是：

```text
这是同一个人分时做三件事：
1. 搬一个 A 元素到 shared
2. 搬一个 B 元素到 shared
3. 用 shared 里的整块数据计算自己负责的 8 个 C 输出
```

#### 第 3 步：累加器放寄存器

```text
float threadResults[TM] = {0.0f};   // TM=8 个累加器,在寄存器里(最快)
→ 这一列 8 个输出,各自有一个累加器,沿 K 一路累加
```

#### 第 4 步：计算核心（最关键，体会"B 读一次复用 TM 次"）

```cpp
for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {        // 遍历这段 K(BK=8)
    float Btmp = Bs[dotIdx * BN + threadCol];        // ★从 SMEM 读 1 个 B → 寄存器
    for (int resIdx = 0; resIdx < TM; ++resIdx) {     // 这一列的 TM=8 个输出
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;  // 复用 Btmp!
    }
}
```

```text
精髓:Btmp 只从 SMEM 读 1 次,被内层 TM=8 个 FMA 复用
  → 原来算 8 个输出要读 8 次 B,现在只读 1 次 → SMEM 压力砍 8 倍
  As 仍读 8 次(每个输出对应不同行),但 B 的复用已经把总访存大幅降低
```

#### 第 5 步：完整 kernel 骨架

```cpp
#define BM 64
#define BN 64
#define BK 8
#define TM 8
// 启动：dim3 block((BM*BN)/TM);  即 512 个线程(一维)
//       dim3 grid(N/BN, M/BM);
__global__ void gemm1DTiling(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    int cRow = blockIdx.y;   // 这个 block 负责 C 的第几个 BM 行带
    int cCol = blockIdx.x;   // 第几个 BN 列带

    // 计算视角:这个线程在 block 内负责哪列、哪组行
    int threadCol = threadIdx.x % BN;
    int threadRow = threadIdx.x / BN;

    // 加载视角:这个线程搬 As/Bs 的哪一格
    int innerColA = threadIdx.x % BK;
    int innerRowA = threadIdx.x / BK;
    int innerColB = threadIdx.x % BN;
    int innerRowB = threadIdx.x / BN;

    // 把 A/B 指针移到本 block 负责的起点
    A += cRow * BM * K;            // A 第 cRow*BM 行起
    B += cCol * BN;               // B 第 cCol*BN 列起
    C += cRow * BM * N + cCol * BN;

    float threadResults[TM] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {   // 沿 K 一段段(每段 BK)
        // ① 协作加载:每线程搬 1 个 As + 1 个 Bs
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        A += BK;            // A 右移 BK 列
        B += BK * N;        // B 下移 BK 行

        // ② 计算:B 读一次复用 TM 次
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float Btmp = Bs[dotIdx * BN + threadCol];
            for (int resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] +=
                    As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
            }
        }
        __syncthreads();
    }

    // ③ 写回这一列的 TM 个输出
    for (int resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
    }
}
```

> 上面骨架假设 M/N/K 都是 BM/BN/BK 的整数倍(省掉边界判断,先跑通)。
> 测试用 512/1024/2048 这种 64 整数倍的尺寸即可,不会触发边界。

#### 第 6 步：为什么快了（访存账，面试能讲）

```text
基础 tiling(每线程 1 个输出):
  每个输出的 SMEM 访问 ≈ K*2 次(每步读 1A+1B)

1D tiling(每线程 TM=8 个输出):
  每个输出的 SMEM 访问 ≈ K*9/8 次(B 复用,8 个输出共享 1 次 B 读)
  → SMEM 访问/输出 从 2K 降到 ~1.1K,几乎砍半
  → siboehm 实测:Kernel3 2980 → Kernel4 8475 GFLOPS,2.8x
```

#### 第 7 步：完成标准

```text
✅ gemm1DTiling 正确(512/1024 PASS)
✅ 比基础 tiling 明显更快(目标 1.5~2.5x)
✅ 能口述:每线程算 TM 个输出,B 读 1 次复用 TM 次 → SMEM 访存减半
✅ 理解 threadRow/threadCol(计算视角) 和 innerRow/innerCol(加载视角)是两套
```

### 3.8 Register Tiling 易混点对照

| 容易混的点 | 正确理解 |
|---|---|
| `threadResults[TM]` 是 shared memory 吗？ | 不是。它是每个 thread 自己的寄存器累加器。 |
| 一个 thread 为什么能算多个 C？ | 因为这些 C 输出共享一部分 A 或 B 数据，适合在寄存器里复用。 |
| 1D register tiling 主要复用谁？ | 同一列上多个输出共享 `B[k][col]`，所以主要复用 B。 |
| `threadRow/threadCol` 和 `innerRowA/innerColA` 为什么不同？ | 前者描述这个 thread 算哪些 C；后者描述这个 thread 搬 shared 的哪一格。 |
| occupancy 降了为什么还可能更快？ | 每个 thread 做更多有效计算，寄存器复用和指令级并行可能弥补更少的驻留线程。 |

一句话区分 shared tiling 和 register tiling：

```text
shared tiling:
  一个 block 复用 global memory 读来的数据。

register tiling:
  一个 thread 复用 shared memory 读来的数据。
```

## 4. 优化阶梯总览（量级感）

```text
版本                  算术强度    相对 naive    瓶颈
─────────────────────────────────────────────────────
naive                 ~0.25       1×            global 带宽 + 缺少显式复用
shared tiling         ~0.25×TILE  数倍          shared 读取 + 同步
register tiling       更高        十几倍         寄存器/occupancy 权衡
+ 向量化/双缓冲(下章)  接近峰值    数十倍         逼近 cuBLAS
─────────────────────────────────────────────────────
cuBLAS（参考天花板）   —          ~峰值          —
```

> 具体加速比依赖矩阵规模、硬件和调参。建议在 `labs/06_operators/gemm_tiled/`
> 里**亲手实测**每一步的 GFLOPS，像卷七那样把"理论提升"变成"真实数字"——一次短运
> 行不能下结论，要用大矩阵（如 2048³）多次测。

参考实测（T4，2048×2048×2048）：

```text
naive   441 GFLOPS
tiled   736 GFLOPS   ← 约 1.67x（TILE=16）
```

注意这只是 tiling 第一级的提升；继续上 register tiling、向量化、Tensor Core（下章）
还能把 GFLOPS 推高数倍。你机器上的具体数字会不同，重点是看**趋势和量级**。

## 5. 怎么验证你的优化真的有效

每加一级优化，按卷五的方法验证因果：

```text
1. 正确性优先：和 CPU double 参考比，相对误差 < 1e-3 才算数。
2. 测 GFLOPS：2·M·N·K / 秒 / 1e9，和 naive 对比。
3. 用 Nsight Compute 看：
   - DRAM throughput（tiling 后应下降，因为复用减少了 global 读）
   - shared bank conflict（register tiling 时注意）
   - achieved occupancy（register tiling 后会降，验证你的权衡判断）
   - Speed of Light：是否从带宽受限转向算力受限
```

如果某级优化没带来预期提速，**那正是最值得深挖的学习点**——回到对应小节追问"我哪个
假设错了"。

## 6. 本章小结

```text
naive 两病根：A/B 都缺少跨输出复用，大量时间花在 global memory
shared tiling：tile 进 shared 复用 → AI 提升约 TILE 倍
  关键：两个 __syncthreads()、规整加载、边界填 0
register tiling：每线程算 TM×TN 输出 → 寄存器复用 → AI 再升
  代价：吃寄存器、occupancy 降，靠 ILP 补偿（要实测权衡）
本质：每一级都在把算术强度往 Roofline 拐点推，从带宽受限走向算力受限
```

## 7. 本章通关标准

不要用“我看完了”判断是否学会。用下面这些问题检查：

```text
基础理解：
  能画出 C tile、A tile、B tile 的关系。
  能解释为什么 K 要分段。
  能解释 As/Bs 每轮 t 都会被覆盖。

代码理解：
  能说清 row、col、aCol、bRow 分别是什么。
  能说清 As[ty][k] 和 Bs[k][tx] 为什么正好对应 A[row][K] 和 B[K][col]。
  能解释两个 __syncthreads() 分别防什么错误。

优化理解：
  能说清 shared tiling 复用 global 数据。
  能说清 register tiling 复用 shared 数据。
  能用自己的话解释 Btmp 为什么可以被 TM 个输出复用。
```

如果这些还说不顺，先不要急着看下一章 Tensor Core。GEMM 后面的所有优化，基本都在
这三个问题上继续叠复杂度：

```text
数据放哪里？
谁来复用？
复用多少次？
```

下一章：向量化加载、双缓冲（软件流水）和 Tensor Core，把 GEMM 推到接近峰值。

## 8. 资料映射

- PMPP：Tiled Matrix Multiplication、Thread Coarsening。
- CUDA C++ Programming Guide：Shared Memory、Writing Tile Kernels。
- CUTLASS 文档：threadblock / warp / thread 三级 tiling 的层次化设计。
- 配套：[卷三第 03 章 Shared Memory、Tile 与 Bank Conflict](../volume03_memory_system/03_Shared_Memory_Tile与Bank_Conflict.md)、[卷五第 03 章 Occupancy](../volume05_performance/03_Occupancy_分歧与延迟隐藏.md)。
