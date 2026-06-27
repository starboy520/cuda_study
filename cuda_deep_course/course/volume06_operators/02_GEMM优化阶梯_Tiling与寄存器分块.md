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

### 3.2 先从点积开始：一个输出到底怎么算

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

### 3.4 Register Tiling：2D 外积视角

register tiling 让一个 thread 算多个 C 输出，把 A、B 读进寄存器后复用。最有用的是
**2D 外积**：一个 thread 负责一个 `TM×TN` 的小矩形，A 和 B 都在寄存器里复用。

```text
register tiling（2D 外积）:
  一个 thread 算 TM×TN 个输出（一个小矩形）
  A 和 B 都在寄存器里复用
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

动手时建议先用小参数（如 `BM=BN=64, BK=8, TM=TN=4`）把外积的索引写对，再调大调优。

### 3.5 用行主序地址手推 2×2 外积（0 → 1）

光看 `acc[i][j] += regA[i]*regB[j]` 还是抽象。我们用**行主序的真实地址**手推一个
`TM=TN=2` 的最小例子，把"外积"落到内存上。先记死行主序地址公式：

```text
A 是 M×K：A[row][k]   → A[row*K + k]
B 是 K×N：B[k][col]   → B[k*N + col]
C 是 M×N：C[row][col] → C[row*N + col]
关键：同一行里下标 +1，地址 +1（连续）；换行地址跳一整行宽。
```

设这个 thread 负责 C 的起点 `(row, col)`，算 `2×2` 小块。沿 K 走，每步：

```cpp
// 一个 thread：4 个累加器,从头到尾住在寄存器
float c00=0, c01=0, c10=0, c11=0;

for (int k = 0; k < K; ++k) {
  // 取 A 的 2 个（同列 k,不同行 → 地址跨行,相差 K）
  float a0 = A[(row+0)*K + k];
  float a1 = A[(row+1)*K + k];
  // 取 B 的 2 个（同行 k,不同列 → 地址相差 1,连续！）
  float b0 = B[k*N + (col+0)];
  float b1 = B[k*N + (col+1)];
  // 外积：2 个 A × 2 个 B = 4 次乘加
  c00 += a0*b0;  c01 += a0*b1;
  c10 += a1*b0;  c11 += a1*b1;
}
// K 跑完,4 个寄存器才写回(只写一次)
C[(row+0)*N + col+0]=c00;  C[(row+0)*N + col+1]=c01;
C[(row+1)*N + col+0]=c10;  C[(row+1)*N + col+1]=c11;
```

```text
账：取 4 个数（2A+2B）→ 做 4 次乘加 → 计算密度比"取2算1"翻 4 倍
行主序的礼物：取 B 的 b0、b1 地址连续 → 硬件可用向量加载(LDG.128)一次抓多个
```

### 3.6 为什么每线程算 8×8：寄存器是天花板

把 `TM=TN=2` 放大到 `8×8`：取 8 个 A、8 个 B，做 `8×8=64` 次乘加。为什么是 8，
不是 16？因为**寄存器数量有物理上限**：

```text
NVIDIA GPU 每个线程最多约 255 个寄存器
TM×TN=8×8 → 64 个累加器 + 8 个 A 暂存 + 8 个 B 暂存 ≈ 80 个 → 放得下 ✓
TM×TN=16×16 → 256 个累加器 → 直接超 255 → 寄存器溢出(register spill)
  → 溢出的值落到 local memory(其实在显存)→ 慢上百倍 → tiling 失效
```

```text
→ register tiling 的本质 = 在"寄存器数量"这顶紧箍咒下,
  把"每次 shared 读取换来的计算量(算术强度)"压到最大
→ 8×8 是常见甜点:64 次乘加只需 16 次 shared 读,且寄存器不溢出
```

这与 §3.5 的"代价与权衡"是同一件事的两面：tile 越大复用越多，但寄存器/occupancy
是硬约束。

### 3.7 完整 kernel：global → shared → register 全貌

前面 §2 讲了 shared tiling、§3.4~3.6 讲了 register 外积。**真正的 GEMM 是两者叠加**：
一个 block 把 A/B 大块搬进 shared（省 global 访问），block 内每个 thread 再从 shared
取小块进寄存器做 8×8 外积（省 shared 访问）。下面是带注释的完整骨架：

```cpp
#define TM 8        // 每线程算 C 的 8 行
#define TN 8        // 每线程算 C 的 8 列
#define BM 64       // 一个 block 算 C 的 64 行 = TM × blockDim.y(8)
#define BN 64       // 一个 block 算 C 的 64 列 = TN × blockDim.x(8)
#define BK 8        // K 维一次切多厚搬进 shared(教学用小值,实战可更大)
// 启动：dim3 block(8,8);  dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);

__global__ void gemmRegTiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
  __shared__ float sA[BM][BK];          // A 的 64×BK 块
  __shared__ float sB[BK][BN];          // B 的 BK×64 块

  int tx = threadIdx.x, ty = threadIdx.y;        // block 内 0~7
  int rowBase = blockIdx.y * BM + ty * TM;       // 该线程 8 行的起点
  int colBase = blockIdx.x * BN + tx * TN;       // 该线程 8 列的起点

  float acc[TM][TN] = {0.0f};            // ★64 个累加器,全程住寄存器,永不清零

  for (int k0 = 0; k0 < K; k0 += BK) {   // 沿 K 一大段一大段(宏观分块)
    // ① block 协作把 A/B 这一段搬进 shared(每线程搬几格,省略边界判断)
    //    (real code 这里要按 threadIdx 分工铺满 sA/sB)
    loadTileToShared(sA, sB, A, B, k0, ...);
    __syncthreads();                     // 等搬完

    // ② block 内每线程从 shared 取小块做 8×8 外积
    for (int k = 0; k < BK; ++k) {       // 段内 K
      float a[TM], b[TN];
      for (int i = 0; i < TM; ++i) a[i] = sA[ty*TM + i][k];  // 取 8 个 A
      for (int j = 0; j < TN; ++j) b[j] = sB[k][tx*TN + j];  // 取 8 个 B
      for (int i = 0; i < TM; ++i)       // 8×8 外积,累加到寄存器
        for (int j = 0; j < TN; ++j)
          acc[i][j] += a[i] * b[j];
    }
    __syncthreads();                     // 等大家算完再覆盖 shared 装下一段
  }

  // ③ 所有 K 段跑完,acc 才是最终结果,一次性写回
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j) {
      int r = rowBase + i, c = colBase + j;
      if (r < M && c < N) C[r * N + c] = acc[i][j];
    }
}
```

**三个一定要记住的全局点：**

```text
1. acc[TM][TN] 跨所有 K 段【永不清零】→ 一段段累加,最后才写回(只写一次)
2. 数据流三级:global(慢) → shared(快,block 共用) → register(最快,thread 私有)
   每降一级都靠"复用"换"少访问上一级"
3. 两个 __syncthreads():搬完一个、算完一个,缺一个就 race(读到没搬好/被提前覆盖)
```

**BK 选多大？(shared 容量算出来的)**

```text
shared 用量 = (BM×BK + BK×BN) × 4 字节,必须 ≤ 每 block 的 shared 上限
  BM=BN=64 → (64×BK + BK×64)×4 = 512×BK 字节
  若 shared 上限 64KB:512×BK ≤ 65536 → BK ≤ 128
→ BK 越大,K 维"搬运次数(K/BK)"越少 → global 访问和同步越少 → 越快
→ 所以经典配置常取 BK=128(刚好卡满 64KB);老卡 shared 小就降到 64/32
→ 注意:T4/A100 shared 上限不同,实战要按你的卡算(可 cudaFuncSetAttribute 调大)
```

**grid 怎么算？(覆盖整个 C,要向上取整)**

```text
dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);  // ceil,别用 N/BN 整除(会漏边)
  x 管列(N)、y 管行(M)
M/N 非 BM/BN 整数倍 → 最后一排/一列是"尾块",靠 ① 加载填 0 + ③ 写回边界判断兜底
(高性能库常把 M/N padding 到 64 倍数,让所有 block 满载,免尾块损失)
```

### 3.8 代价与权衡

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

### 3.9 Register Tiling 易混点对照

| 容易混的点 | 正确理解 |
|---|---|
| `acc[TM][TN]` 是 shared memory 吗？ | 不是。它是每个 thread 自己的寄存器累加器（小矩形）。 |
| 一个 thread 为什么能算多个 C？ | 因为这些 C 输出共享一部分 A 或 B 数据，适合在寄存器里复用。 |
| 2D register tiling 复用了什么？ | regA[TM] 和 regB[TN] 读进寄存器后，做 TM×TN 次外积，A、B 都被复用。 |
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
  能用自己的话解释 2D 外积里 regA[i]、regB[j] 为什么能复用算出 TM×TN 个输出。
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
