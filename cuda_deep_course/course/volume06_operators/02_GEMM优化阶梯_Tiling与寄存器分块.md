# 02 GEMM 优化阶梯：Tiling 与寄存器分块

> 上一章算出：naive GEMM 的算术强度只有 0.25，严重带宽受限。本章爬第一段优化阶梯
> ——用 **shared-memory tiling** 和 **register tiling** 把算术强度抬上去，让 GEMM
> 从"等数据"变成"忙着算"。每一步都讲清动机、机制和提升量级。

## 1. 回顾 Naive 的两个病根

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

两个病根：

```text
病根① 重复读：C 的一行 N 个输出，每个都把 A 的【同一行】完整读一遍 → A 被读 N 次
病根② B 不合并：B[k*N+col]，k 变化时地址跨 N → warp 访问 B 不连续（卷一布局）
```

两个病根都指向同一个解法：**把数据先搬进 shared memory，让一个 block 协作复用。**

## 2. Shared-Memory Tiling：核心思想

### 2.0 一句话 + 生活类比（先抓直觉）

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

为什么 GEMM 特别适合这招？因为它的数据**复用度极高**：

```text
算 C 的一行(N 个输出)，每个输出都要读 A 的【同一行】
→ A 的一行被 N 个输出各读一遍(naive 重复 N 次)
→ 如果把 A 这一行(的一段)放进 shared，N 个输出共用 → 省 N 倍 global 读
```

### 2.1 分块的直觉

不再让每个线程独立算一个输出、各自读内存，而是：

```text
把 C 切成 TILE×TILE 的小块，一个 block 负责算一块。
计算这一块需要 A 的对应"行带"和 B 的对应"列带"。
沿 K 维分段：每次取 A、B 的一个 TILE×TILE 子块load 进 shared，
            block 内所有线程复用这两个子块做部分点积，累加。
```

```text
       B 的列带
        ┌───┐
        │tile│
        └───┘
A 行带  ┌───┐ ┌───┐   每一步：加载 A、B 各一个 tile 到 shared
 ┌───┐  │tile│→│ C │   block 内线程从 shared 反复读，算部分和
 │tile│→└───┘ │tile│   K 维走完所有 tile，累加得最终 C tile
 └───┘        └───┘
```

**为什么要"沿 K 维分段"？** 因为算一个 C tile 需要 A 一整条"行带"(TILE 行 × 整个 K)
和 B 一整条"列带"(整个 K × TILE 列)，整条带子放不进 shared(K 可能上千)。

```text
解法：把 K 切成 TILE 长的小段，一段一段处理：
  C_tile = Σ_t (A 的第 t 个 K-段子块) × (B 的第 t 个 K-段子块)
  每一步只把"一个 TILE×TILE 的 A 子块 + 一个 TILE×TILE 的 B 子块"放进 shared
  算完这段的部分点积，累加，再滑到下一段
→ 像"分期付款"：整条带子分 K/TILE 期，每期只搬两小块进 shared
```

### 2.2 为什么这样能提升算术强度

关键在**复用**：一个 `TILE×TILE` 的 A 子块被 load 进 shared 一次后，block 内
`TILE` 列的输出都用它——**一次 global 读，TILE 次复用**。算术强度因此提升约 TILE 倍：

```text
naive:        每个 A 元素从 global 读 1 次，用 1 次  → AI ≈ 0.25
tiling(TILE): 每个 A 元素从 global 读 1 次，用 TILE 次 → AI ≈ 0.25 × TILE

TILE=16 → AI ≈ 4 ；TILE=32 → AI ≈ 8 ... 离拐点 25 更近了
```

### 2.3 代码骨架

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

### 2.3.1 把加载索引讲清楚（最容易晕的地方）

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

### 2.3.2 为什么计算是 As[ty][k] * Bs[k][tx]

```text
C[row][col] = Σ_K A[row][K] * B[K][col]
本段(第 t 段)的部分和 = Σ_{k=0..TILE} A[row][t*TILE+k] * B[t*TILE+k][col]

而 As[ty][k] 装的是 A[row][t*TILE+k]   (上面 aCol=t*TILE+tx，这里内层用 k)
   Bs[k][tx] 装的是 B[t*TILE+k][col]   (上面 bRow=t*TILE+ty，这里内层用 k)
→ As[ty][k] * Bs[k][tx] = A[row][t*TILE+k] * B[t*TILE+k][col] ✓
→ 内层 k 循环把这一段的部分点积累加进 sum，K/TILE 段后得完整 C[row][col]
```

> 验证索引对不对的通用方法(和转置那篇一样)：写出每个数组下标的代数表达式，
> 代入化简，对照定义 `C=Σ A·B`。不用追踪具体线程。

### 2.3.2.1 ★最该想透的★ t 与 K：每个 t 段 = K 的一个 tile，分段求和 = 整段求和

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

### 2.3.3 一步 tile 的完整流程（具体走查）

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

### 2.3.4 跨 block：整张 C 是怎么拼出来的（全局视角）

前面 §2.3.1~2.3.3 都聚焦"**一个** block 怎么算出 C 的**一个** TILE×TILE 块"。
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
  - 自己沿 K 从头滑到尾（§2.3.3 那套 t 循环）
  - 自己把结果写回 C 自己那一格
→ __syncthreads() 只同步"块内" TILE×TILE 个线程，从不跨 block
→ 几百上千个 block 由 GPU 调度器铺到各 SM 上并行/排队跑，顺序无所谓
```

**③ 对比"块内 K 滑动" vs "跨块"两个维度（别混了）：**

```text
                何时发生            谁参与            靠什么协作
块内沿 K 滑动   一个 block 内部      该 block 的线程    shared + __syncthreads
(§2.3.3 的 t)   累加出 1 个 C tile

跨 block        不同 block 之间      不同 SM 上的 block  不协作！各写各的 C 格子
(本节)          拼出整张 C          完全独立并行
```

> **一句话**：K 方向的"分期累加"是**块内**的事（一个 block 把自己那格的 K 全吃完）；
> 而"算完整张 C"是**跨 block**的事（每个 block 包一格，互不干扰地并行）。
> 二者正交：先有跨 block 切出"我负责哪格"，再有块内沿 K 滑动"把这格算完"。

**④ 为什么 GEMM 能跨 block 完全独立？** 因为 C 的每个输出元素只属于一个 block，
没有"两个 block 往同一个 C 位置写"的情况 → 无需原子操作、无需跨 block 同步。
（对比 reduction/histogram 那种多 block 往同一处累加，才需要 atomicAdd 或二次 kernel。）

### 2.4 三个关键点（每个都对应前面学的）

- **加载阶段合并**：`A[row*K + aCol]` warp 内 `aCol` 连续 → 合并；`B[bRow*N + col]`
  warp 内 `col` 连续 → **合并！** 这就治好了病根②——通过 shared 中转，B 也变合并了。
- **两个 `__syncthreads()` 缺一不可**：第一个保证 load 完才算（不然读到没加载的）；
  第二个保证算完才覆盖 shared（不然下一轮 load 冲掉还在用的数据）。这是卷四的 race。
- **边界用 0 填充**：非 TILE 整除的尺寸，越界位置填 0（加法单位元，不影响结果）。

## 3. Register Tiling：再上一个台阶

### 3.1 shared tiling 还不够

shared tiling 把 AI 提到了几倍，但**每个线程仍只算 1 个输出**。瓶颈转移了：现在
线程频繁从 **shared memory** 读数据（虽然比 global 快，但仍有延迟和 bank 压力）。

### 3.2 思路：一个线程算多个输出（thread tiling）

让**每个线程负责一个 `TM×TN` 的输出小块**（比如 4×4=16 个输出），而不是 1 个：

```text
每线程算 1 个输出（shared tiling）：
  从 shared 读 1 个 A、1 个 B → 1 次 FMA

每线程算 4×4 个输出（register tiling）：
  从 shared 读 4 个 A、4 个 B（共 8 次读）→ 做 16 次 FMA
  → 读 shared 的次数 / 计算次数 = 8/16，比 1/1 好一倍
```

核心收益：**把 A、B 的元素从 shared 读进寄存器后，在寄存器里复用算多个输出**。
寄存器是最快的存储（卷三），读寄存器几乎零延迟。算术强度（相对 shared）再次提升。

### 3.3 代码思想（简化）

```cpp
// 每个线程算 TM×TN 个输出，累加器在寄存器数组里
float acc[TM][TN] = {0};
float regA[TM], regB[TN];

for (每个 K 维 tile) {
  load A、B tile 进 shared; __syncthreads();
  for (int k = 0; k < TILE; ++k) {
    // 从 shared 把这一列/行读进寄存器（少量读）
    for (int i = 0; i < TM; ++i) regA[i] = As[...];
    for (int j = 0; j < TN; ++j) regB[j] = Bs[...];
    // 在寄存器里做 TM×TN 次 FMA（大量算）
    for (int i = 0; i < TM; ++i)
      for (int j = 0; j < TN; ++j)
        acc[i][j] += regA[i] * regB[j];
  }
  __syncthreads();
}
// 把 acc 写回 C
```

### 3.4 代价与权衡

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

## 4. 优化阶梯总览（量级感）

```text
版本                  算术强度    相对 naive    瓶颈
─────────────────────────────────────────────────────
naive                 ~0.25       1×            global 带宽 + B 不合并
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
naive 两病根：重复读 A、B 不合并
shared tiling：tile 进 shared 复用 → AI 提升约 TILE 倍 + B 变合并
  关键：两个 __syncthreads()、加载合并、边界填 0
register tiling：每线程算 TM×TN 输出 → 寄存器复用 → AI 再升
  代价：吃寄存器、occupancy 降，靠 ILP 补偿（要实测权衡）
本质：每一级都在把算术强度往 Roofline 拐点推，从带宽受限走向算力受限
```

下一章：向量化加载、双缓冲（软件流水）和 Tensor Core，把 GEMM 推到接近峰值。

## 7. 资料映射

- PMPP：Tiled Matrix Multiplication、Thread Coarsening。
- CUDA C++ Programming Guide：Shared Memory、Writing Tile Kernels。
- CUTLASS 文档：threadblock / warp / thread 三级 tiling 的层次化设计。
- 配套：[卷三第 03 章 Shared Memory、Tile 与 Bank Conflict](../volume03_memory_system/03_Shared_Memory_Tile与Bank_Conflict.md)、[卷五第 03 章 Occupancy](../volume05_performance/03_Occupancy_分歧与延迟隐藏.md)。
