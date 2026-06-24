# 02A GEMM 从零到 Tiling 小白版

这份文档先不追求“像高手一样讲 GEMM”。它只追求一件事：

> 你能把 GEMM 从一个小点、一个小 tile、一个 block、一个 grid，完整串起来。

旧文档里很多地方是在讲局部技巧，比如 shared memory、register tiling、`BM/BN/BK/TM`。
局部技巧本身不难，真正难的是：

```text
我知道一个 tile 怎么算，
但它怎么放进 block？
block 又怎么放进 grid？
grid 又怎么拼出完整 C？
```

所以这一版按“从小到大”讲：

```text
一个 C 元素
  -> 一个 2x2 C tile
  -> 一个 block 负责一个 C tile
  -> 很多 block 组成 grid 拼出完整 C
  -> shared memory tiling
  -> 1D register tiling
  -> 2D register tiling 的直觉
```

## 0. 先记住一张总图

GEMM 是：

```text
C = A x B

A: M 行 K 列
B: K 行 N 列
C: M 行 N 列
```

也就是：

```text
A[M,K] x B[K,N] = C[M,N]
```

在 CUDA 里，我们最终想做的是：

```text
Grid 负责整张 C
Block 负责 C 的一个小块
Thread 负责小块里的一个或多个 C 元素
```

画成一条线：

```text
整张 C
  被切成很多 C tile
    每个 C tile 交给一个 block
      block 内 thread 合作算这个 tile
        thread 最后写回自己负责的 C 元素
```

如果你读到后面又绕了，就回到这张图。

## 1. 一个 C 元素到底怎么算

先看一个输出：

```text
C[row][col]
```

它来自：

```text
A 的第 row 行
B 的第 col 列
```

例如：

```text
A[row] = [a0, a1, a2, a3]
B[col] = [b0, b1, b2, b3]
```

那么：

```text
C[row][col] = a0*b0 + a1*b1 + a2*b2 + a3*b3
```

这叫点积。

写成 GEMM 的下标形式：

```text
C[row][col] =
    A[row][0] * B[0][col]
  + A[row][1] * B[1][col]
  + A[row][2] * B[2][col]
  + ...
  + A[row][K-1] * B[K-1][col]
```

代码就是：

```cpp
float sum = 0.0f;
for (int k = 0; k < K; ++k) {
  sum += A[row * K + k] * B[k * N + col];
}
C[row * N + col] = sum;
```

这里默认矩阵是行主序：

```text
A[row][k]   -> A[row * K + k]
B[k][col]   -> B[k * N + col]
C[row][col] -> C[row * N + col]
```

## 2. Naive CUDA GEMM：一个 thread 算一个 C 元素

最直接的 CUDA 写法是：

```text
一个 thread 负责一个 C[row][col]
```

二维 grid 和二维 block 常用来映射矩阵：

```text
x 方向 -> col
y 方向 -> row
```

所以：

```cpp
const int row = blockIdx.y * blockDim.y + threadIdx.y;
const int col = blockIdx.x * blockDim.x + threadIdx.x;
```

完整 kernel：

```cpp
__global__ void gemmNaive(const float* A,
                          const float* B,
                          float* C,
                          int M,
                          int N,
                          int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}
```

Host 侧启动：

```cpp
dim3 block(16, 16);
dim3 grid((N + block.x - 1) / block.x,
          (M + block.y - 1) / block.y);
gemmNaive<<<grid, block>>>(A, B, C, M, N, K);
```

这里的完整关系是：

```text
grid.x 覆盖 C 的列方向 N
grid.y 覆盖 C 的行方向 M

block.x 表示一个 block 有多少列方向 thread
block.y 表示一个 block 有多少行方向 thread

threadIdx.x 决定当前 thread 在 block 内的列偏移
threadIdx.y 决定当前 thread 在 block 内的行偏移
```

## 3. Naive 为什么慢

Naive 不是错，它只是浪费数据。

看 C 的同一行：

```text
C[row][0]
C[row][1]
C[row][2]
C[row][3]
```

这 4 个输出都需要 A 的同一行：

```text
A[row][0], A[row][1], A[row][2], ...
```

如果 4 个 thread 各算一个输出，那么它们会各自去 global memory 读 A 的同一行。

也就是说：

```text
A[row][k] 被多个 thread 重复读取。
```

再看 C 的同一列：

```text
C[0][col]
C[1][col]
C[2][col]
C[3][col]
```

这些输出都会用到同一列 B 里的元素，比如：

```text
B[k][col]
```

Naive 里不同 thread 也会各自去 global memory 读自己需要的 B。

所以 naive 的问题可以一句话概括：

```text
每个 thread 太独立，大家明明需要相同的数据，却没有先合作复用。
```

优化 GEMM 的核心就是：

```text
把 A/B 的一小块数据读进来以后，让它被更多计算复用。
```

## 4. 先从一个 2x2 C Tile 开始

不要一上来就想 `16x16`、`32x32`。先想一个最小例子：

```text
C tile = 2 行 x 2 列
```

它包含 4 个输出：

```text
        col0   col1
row0    C00    C01
row1    C10    C11
```

假设一个 block 负责这个 `2x2` 的 C tile。

最简单的安排是：

```text
thread(0,0) 算 C00
thread(1,0) 算 C01
thread(0,1) 算 C10
thread(1,1) 算 C11
```

注意坐标：

```text
threadIdx.x -> 列方向
threadIdx.y -> 行方向
```

所以：

```text
threadIdx=(x=0,y=0) -> C tile 内第 0 行第 0 列
threadIdx=(x=1,y=0) -> C tile 内第 0 行第 1 列
threadIdx=(x=0,y=1) -> C tile 内第 1 行第 0 列
threadIdx=(x=1,y=1) -> C tile 内第 1 行第 1 列
```

## 5. 这个 2x2 C Tile 需要哪些 A/B 数据

每个 C 元素都是 A 的一行乘 B 的一列。

对于这个 `2x2` C tile：

```text
C00 需要 A[row0][k] 和 B[k][col0]
C01 需要 A[row0][k] 和 B[k][col1]
C10 需要 A[row1][k] 和 B[k][col0]
C11 需要 A[row1][k] 和 B[k][col1]
```

你会看到复用：

```text
A[row0][k] 同时被 C00、C01 使用
A[row1][k] 同时被 C10、C11 使用

B[k][col0] 同时被 C00、C10 使用
B[k][col1] 同时被 C01、C11 使用
```

这就是 tiling 的根：

```text
一个 C tile 里的多个输出，会共享一小块 A 和一小块 B。
```

## 6. 为什么还要沿 K 分段

完整点积要从 `k=0` 算到 `k=K-1`。

如果 K 很大，比如 1024，那么一个 C tile 需要：

```text
A 的 2 行 x 1024 列
B 的 1024 行 x 2 列
```

这太大，不适合一次全搬进 shared memory。

所以我们把 K 切成小段。

如果 `TILE=2`，`K=6`，那么：

```text
完整 K:

k = 0 1 | 2 3 | 4 5
    t=0 | t=1 | t=2
```

每次只处理一个 K 段：

```text
t=0: 处理 k=0,1
t=1: 处理 k=2,3
t=2: 处理 k=4,5
```

每一段都做：

```text
1. 把 A 的当前小块搬进 shared memory 的 As
2. 把 B 的当前小块搬进 shared memory 的 Bs
3. block 内线程同步
4. 用 As 和 Bs 算这一段的部分和
5. block 内线程同步
6. 进入下一个 K 段，覆盖 As/Bs
```

这就是 shared-memory tiling。

## 7. As 和 Bs 是什么

代码里常见：

```cpp
__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];
```

读成：

```text
As = A shared tile
Bs = B shared tile
```

它们不是完整 A/B。

它们只是：

```text
当前 block 的临时工作台
```

对于 `TILE=2`，某一轮 `t` 中：

```text
As 存 A 的 2x2 小块
Bs 存 B 的 2x2 小块
```

下一轮 `t`：

```text
As/Bs 会被覆盖成下一段 K 的 A/B 小块
```

所以：

```text
As/Bs 是循环复用的 shared memory 缓冲区。
```

## 8. 从 2x2 Tile 推到真实 TILE x TILE

真实代码里通常用：

```cpp
constexpr int TILE = 16;
dim3 block(TILE, TILE);
```

意思是：

```text
一个 block 有 16 x 16 = 256 个 thread
一个 block 负责 C 的一个 16 x 16 tile
每个 thread 负责这个 tile 里的一个 C 输出
```

对应关系：

```text
threadIdx.y -> C tile 内的行
threadIdx.x -> C tile 内的列
```

所以当前 thread 负责的全局 C 坐标是：

```cpp
const int row = blockIdx.y * TILE + threadIdx.y;
const int col = blockIdx.x * TILE + threadIdx.x;
```

这句话非常重要：

```text
blockIdx 决定这个 block 负责整张 C 的哪一个 tile。
threadIdx 决定这个 thread 负责 tile 里的哪一个点。
```

## 9. 一个 Block 怎么负责一个 C Tile

假设：

```text
TILE = 16
blockIdx = (2, 3)
```

那么这个 block 负责：

```text
C 的第 3 个 tile 行
C 的第 2 个 tile 列
```

因为：

```text
blockIdx.x -> C tile 的列编号
blockIdx.y -> C tile 的行编号
```

这个 block 覆盖的 C 范围是：

```text
row = 3 * 16 ... 3 * 16 + 15 = 48 ... 63
col = 2 * 16 ... 2 * 16 + 15 = 32 ... 47
```

如果 block 内某个 thread：

```text
threadIdx = (x=5, y=7)
```

那么它负责：

```text
row = 3 * 16 + 7 = 55
col = 2 * 16 + 5 = 37
```

也就是：

```text
C[55][37]
```

## 10. Grid 怎么拼出完整 C

整张 C 是 `M x N`。

如果每个 block 负责一个 `TILE x TILE` 的 C tile，那么 grid 需要：

```cpp
dim3 grid((N + TILE - 1) / TILE,
          (M + TILE - 1) / TILE);
```

为什么：

```text
grid.x 覆盖列方向 N
grid.y 覆盖行方向 M
```

例如：

```text
M = 5 行
N = 7 列
TILE = 2
```

那么：

```text
grid.x = ceil(7 / 2) = 4
grid.y = ceil(5 / 2) = 3
```

Grid 有 `4 x 3 = 12` 个 block。

它们拼 C：

```text
        blockIdx.x →
          0       1       2       3
        +-------+-------+-------+-------+
y=0     | Ctile | Ctile | Ctile | Ctile |
        +-------+-------+-------+-------+
y=1     | Ctile | Ctile | Ctile | Ctile |
        +-------+-------+-------+-------+
y=2     | Ctile | Ctile | Ctile | Ctile |
        +-------+-------+-------+-------+

blockIdx.y ↓
```

最后一行、最后一列 tile 可能超出真实矩阵范围，所以 kernel 里必须有边界判断：

```cpp
if (row < M && col < N) {
  C[row * N + col] = sum;
}
```

加载 A/B 时也要边界判断，越界就填 0。

## 11. Shared-Memory Tiled GEMM 完整代码

先看完整代码，再慢慢拆：

```cpp
constexpr int TILE = 16;

__global__ void gemmTiled(const float* A,
                          const float* B,
                          float* C,
                          int M,
                          int N,
                          int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;

  float sum = 0.0f;

  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    const int aCol = t * TILE + threadIdx.x;
    const int bRow = t * TILE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] =
        (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

    Bs[threadIdx.y][threadIdx.x] =
        (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}
```

启动：

```cpp
dim3 block(TILE, TILE);
dim3 grid((N + TILE - 1) / TILE,
          (M + TILE - 1) / TILE);
gemmTiled<<<grid, block>>>(A, B, C, M, N, K);
```

## 12. 代码第一层：row 和 col

```cpp
const int row = blockIdx.y * TILE + threadIdx.y;
const int col = blockIdx.x * TILE + threadIdx.x;
```

这两行回答：

```text
当前 thread 负责 C 的哪个元素？
```

分解：

```text
blockIdx.y * TILE:
  当前 block 前面已经有多少个完整 C tile 行。

threadIdx.y:
  当前 thread 在这个 C tile 内的行偏移。

blockIdx.x * TILE:
  当前 block 前面已经有多少个完整 C tile 列。

threadIdx.x:
  当前 thread 在这个 C tile 内的列偏移。
```

所以：

```text
blockIdx 决定大位置
threadIdx 决定小位置
```

## 13. 代码第二层：t 表示当前 K 段

```cpp
for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
```

这不是遍历 C tile。

这是在遍历 K 方向的分段。

如果：

```text
K = 40
TILE = 16
```

那么：

```text
t=0 -> k = 0..15
t=1 -> k = 16..31
t=2 -> k = 32..39，剩下位置填 0
```

每个 `t` 都让当前 block 处理一段 K。

`sum` 不能在 `t` 循环里清零，因为：

```text
每个 t 只贡献一段部分和。
所有 t 的部分和加起来，才是完整 C[row][col]。
```

## 14. 代码第三层：加载 A 到 As

```cpp
const int aCol = t * TILE + threadIdx.x;

As[threadIdx.y][threadIdx.x] =
    (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
```

这两行回答：

```text
当前 thread 负责搬 A 的哪一个元素到 As？
```

`As` 是一个 `TILE x TILE` 小块：

```text
As[localRow][localCol]
```

这里：

```text
localRow = threadIdx.y
localCol = threadIdx.x
```

A 的来源是：

```text
A[row][aCol]
```

其中：

```text
row  = 当前 C 输出的行
aCol = 当前 K 段里的列
```

为什么 A 用 `threadIdx.x` 推出 `aCol`？

因为 A 的 K 方向是列方向：

```text
A[row][k]
```

同一个 C tile 中，A tile 是：

```text
TILE 行 x TILE 个 K 列
```

所以 thread 的 x 坐标负责铺开 A 的 K 列。

## 15. 代码第四层：加载 B 到 Bs

```cpp
const int bRow = t * TILE + threadIdx.y;

Bs[threadIdx.y][threadIdx.x] =
    (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
```

这两行回答：

```text
当前 thread 负责搬 B 的哪一个元素到 Bs？
```

B 的来源是：

```text
B[bRow][col]
```

其中：

```text
bRow = 当前 K 段里的行
col  = 当前 C 输出的列
```

为什么 B 用 `threadIdx.y` 推出 `bRow`？

因为 B 的 K 方向是行方向：

```text
B[k][col]
```

同一个 C tile 中，B tile 是：

```text
TILE 个 K 行 x TILE 列
```

所以 thread 的 y 坐标负责铺开 B 的 K 行。

这就是很多人最容易晕的地方：

```text
A 的 K 在列方向，所以用 threadIdx.x。
B 的 K 在行方向，所以用 threadIdx.y。
```

## 16. 为什么要第一个 __syncthreads()

加载完：

```cpp
As[threadIdx.y][threadIdx.x] = ...
Bs[threadIdx.y][threadIdx.x] = ...
```

每个 thread 只搬了 As 的一个元素、Bs 的一个元素。

但是计算时，一个 thread 会读整行/整列：

```cpp
sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
```

这意味着：

```text
我不只读自己搬的元素。
我还会读别的 thread 搬的元素。
```

所以必须等整个 block 都搬完：

```cpp
__syncthreads();
```

否则有的 thread 可能读到还没被写入的 As/Bs。

## 17. 代码第五层：用 As/Bs 算部分和

```cpp
for (int k = 0; k < TILE; ++k) {
  sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
}
```

这个循环只算当前 K 段。

当前 thread 负责：

```text
C[row][col]
```

所以它需要：

```text
A[row][当前段里的 k]
B[当前段里的 k][col]
```

而在 shared memory 里：

```text
As[threadIdx.y][k] 存的是 A[row][t*TILE + k]
Bs[k][threadIdx.x] 存的是 B[t*TILE + k][col]
```

所以：

```text
As[threadIdx.y][k] * Bs[k][threadIdx.x]
```

正好就是：

```text
A[row][Kglobal] * B[Kglobal][col]
```

其中：

```text
Kglobal = t * TILE + k
```

## 18. 为什么要第二个 __syncthreads()

算完当前 K 段之后，下一轮 `t` 会覆盖：

```text
As
Bs
```

但是同一个 block 里，不同 thread 的执行进度可能不同。

如果没有第二个：

```cpp
__syncthreads();
```

可能出现：

```text
某些 thread 还在读当前 As/Bs 做计算
另一些 thread 已经进入下一轮，把 As/Bs 覆盖了
```

这就是 race。

所以第二个同步的作用是：

```text
确认大家都用完当前 As/Bs，才能装下一段。
```

## 19. Shared Tiling 的完整故事

现在把完整链路串起来：

```text
1. Grid 覆盖整张 C。
2. 每个 block 负责一个 C tile。
3. block 内每个 thread 负责一个 C 元素。
4. 为了算这个 C tile，block 沿 K 方向分段。
5. 每段中，block 合作把 A tile 和 B tile 搬到 shared memory。
6. __syncthreads() 等大家搬完。
7. 每个 thread 从 shared memory 读数据，算自己 C 元素的部分和。
8. __syncthreads() 等大家算完。
9. 进入下一段 K，重复。
10. 所有 K 段完成后，thread 把 sum 写回 C[row][col]。
```

这就是 shared-memory tiled GEMM。

## 20. Shared Tiling 到底快在哪里

Naive：

```text
每个 thread 自己从 global memory 读 A/B。
相同 A/B 数据会被不同 thread 反复读。
```

Shared tiling：

```text
一个 block 先把 A/B 小块搬进 shared memory。
这个 block 内很多 thread 反复使用这小块。
```

所以它快在：

```text
global memory 读少了
数据复用多了
```

这里“算术强度”可以理解成：

```text
每搬一次数据，做多少计算。
```

shared tiling 让数据搬进来以后被使用更多次，所以算术强度提高。

## 21. Register Tiling 为什么还要继续优化

Shared tiling 后，global memory 压力小了。

但现在每个 thread 仍然是：

```text
只算一个 C 输出
```

每次内层循环：

```cpp
sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
```

它从 shared memory 读：

```text
1 个 A
1 个 B
```

然后做：

```text
1 次 FMA
```

也就是：

```text
读 shared 的次数仍然很多。
```

Register tiling 的想法是：

```text
让一个 thread 算多个 C 输出。
这样它从 shared 读来的某些数据，可以在自己的寄存器里复用多次。
```

## 22. 1D Register Tiling：一个 thread 竖着算多个输出

先不要看 `4x4`。

先看最简单的 1D register tiling：

```text
一个 thread 算同一列上的多个 C 输出。
```

例如一个 thread 算：

```text
C[row+0][col]
C[row+1][col]
C[row+2][col]
C[row+3][col]
```

它们共同点是：

```text
列 col 相同。
```

在某个 K 上：

```text
C[row+0][col] += A[row+0][k] * B[k][col]
C[row+1][col] += A[row+1][k] * B[k][col]
C[row+2][col] += A[row+2][k] * B[k][col]
C[row+3][col] += A[row+3][k] * B[k][col]
```

你会看到：

```text
B[k][col] 是同一个。
```

所以可以：

```cpp
float b = Bs[k][colLocal];  // 从 shared 读一次，放进寄存器

acc0 += As[rowLocal + 0][k] * b;
acc1 += As[rowLocal + 1][k] * b;
acc2 += As[rowLocal + 2][k] * b;
acc3 += As[rowLocal + 3][k] * b;
```

这就是 1D register tiling 的核心：

```text
B 读一次，用多次。
```

这些 `acc0/acc1/acc2/acc3` 通常都在寄存器里。

如果写成数组：

```cpp
float acc[TM] = {0.0f};
```

其中：

```text
TM = Thread M
```

意思是：

```text
一个 thread 沿 M 方向，也就是行方向，算多少个输出。
```

## 23. 1D Register Tiling 怎么放回 Block

这一步是最容易断掉的地方。

shared tiling 里：

```text
一个 block 算 16x16 的 C tile
一个 thread 算 1 个 C 输出
所以需要 16x16 = 256 个 thread
```

1D register tiling 里，如果：

```text
一个 block 算 64x64 的 C tile
一个 thread 竖着算 8 个输出
```

那么 block 需要多少 thread？

```text
C tile 有 64 x 64 = 4096 个输出
每个 thread 算 8 个输出
需要 4096 / 8 = 512 个 thread
```

可以这样摆：

```text
threadCol = 0..63       -> 负责哪一列
threadRowGroup = 0..7   -> 负责哪一组 8 行
```

也就是：

```text
512 个 thread = 64 列 x 8 个行组
```

一个 thread 的计算任务是：

```text
列 = threadCol
行 = threadRowGroup * 8 + 0..7
```

这就是：

```cpp
int threadCol = threadIdx.x % 64;
int threadRowGroup = threadIdx.x / 64;
```

如果：

```text
threadIdx.x = 130
```

那么：

```text
threadCol = 130 % 64 = 2
threadRowGroup = 130 / 64 = 2
```

它负责：

```text
C tile 内第 2 列
第 2 组 8 行
```

也就是：

```text
C[16][2]
C[17][2]
C[18][2]
C[19][2]
C[20][2]
C[21][2]
C[22][2]
C[23][2]
```

## 24. 1D Register Tiling 的计算核心

假设：

```text
TM = 8
BK = 8
```

其中：

```text
TM = 一个 thread 算 8 个输出
BK = 当前 K 段厚度是 8
```

核心计算：

```cpp
float acc[TM] = {0.0f};

for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
  float b = Bs[dotIdx][threadCol];

  for (int i = 0; i < TM; ++i) {
    acc[i] += As[threadRowGroup * TM + i][dotIdx] * b;
  }
}
```

读法：

```text
dotIdx:
  当前 K 段里的第几个 k。

b:
  当前 B[k][col]，读一次放进寄存器。

i:
  当前 thread 负责的第几个行输出。

acc[i]:
  当前 thread 第 i 个 C 输出的累加器。
```

关键收益：

```text
一个 b 被 TM 个 acc 复用。
```

所以 shared memory 的 B 读取压力下降。

## 25. 2D Register Tiling 只先理解直觉

1D register tiling：

```text
一个 thread 算同一列上的多个行。
主要复用 B。
```

2D register tiling：

```text
一个 thread 算一个小矩形。
同时复用 A 和 B。
```

例如一个 thread 算 `4x4` 个输出：

```text
        col0   col1   col2   col3
row0    acc00  acc01  acc02  acc03
row1    acc10  acc11  acc12  acc13
row2    acc20  acc21  acc22  acc23
row3    acc30  acc31  acc32  acc33
```

某个 K 上：

```text
需要 4 个 A:
  A[row0][k], A[row1][k], A[row2][k], A[row3][k]

需要 4 个 B:
  B[k][col0], B[k][col1], B[k][col2], B[k][col3]
```

然后做：

```text
4 x 4 = 16 次 FMA
```

也就是说：

```text
读 4 个 A + 4 个 B
做 16 次计算
```

数据复用更强，但代码也更复杂。

先把 1D register tiling 吃透，再看 2D。

## 26. 三层 Tiling 的完整位置

现在把所有层级放回一张图：

```text
Grid 级：
  整张 C 被切成很多 block tile。

Block 级：
  一个 block 算一个 C tile。
  block 把 A/B 小块搬进 shared memory。
  这是 shared-memory tiling。

Thread 级：
  一个 thread 算一个或多个 C 输出。
  如果一个 thread 算多个输出，就把部分数据放在寄存器里复用。
  这是 register tiling。
```

一张更短的图：

```text
C matrix
  -> C block tile
      -> C thread tile
```

对应：

```text
Grid 负责 C matrix
Block 负责 C block tile
Thread 负责 C thread tile
```

## 27. 最后用一句话串起来

Naive GEMM：

```text
每个 thread 自己读 global，自己算一个 C。
```

Shared-memory tiled GEMM：

```text
一个 block 合作把 A/B 小块读到 shared，
再让 block 内很多 thread 复用这些数据。
```

1D register tiled GEMM：

```text
一个 thread 算同一列上的多个 C，
把一个 B 值读到寄存器后复用给多个输出。
```

2D register tiled GEMM：

```text
一个 thread 算一个小矩形，
A 和 B 都在寄存器里复用。
```

## 28. 你应该能回答的问题

如果下面这些问题能说顺，这章就过了。

```text
1. 一个 C[row][col] 怎么由 A 和 B 算出来？
2. Naive GEMM 中，一个 thread 负责什么？
3. 一个 C tile 和一个 block 是什么关系？
4. 为什么 K 要分段？
5. As 和 Bs 是什么？它们会保存完整 A/B 吗？
6. 第一个 __syncthreads() 防什么？
7. 第二个 __syncthreads() 防什么？
8. Grid.x 为什么覆盖 N，Grid.y 为什么覆盖 M？
9. shared tiling 复用的是哪里的数据？
10. register tiling 复用的是哪里的数据？
11. 1D register tiling 为什么主要复用 B？
12. block tiling 和 thread tiling 分别处在哪一层？
```

最重要的答案是：

```text
Grid 拼完整 C。
Block 算一个 C tile。
Thread 算 tile 里的一个或多个 C 输出。
Shared memory 让 block 复用 global 数据。
Register 让 thread 复用 shared 数据。
```

## 29. 配套实验

当前已有实验：

```bash
cd /home/qichengjie/workspace/cuda_study/cuda_deep_course
make -C labs/06_operators/gemm_tiled clean all
./labs/06_operators/gemm_tiled/gemm_tiled 512 512 512
./labs/06_operators/gemm_tiled/gemm_tiled 2048 2048 2048
```

这个实验包含：

```text
naive GEMM
shared-memory tiled GEMM
```

读实验时建议先只看这两个 kernel：

```text
gemmNaive
gemmTiled
```

不要先看 timing、CPU 验证、GFLOPS 输出。那些是工程验证部分，等 kernel 看懂后再看。

## 30. 下一步学习顺序

建议顺序：

```text
1. 手算 TILE=2、K=4 的一个 C tile。
2. 对照 gemmTiled 代码，标出 row、col、aCol、bRow。
3. 运行 gemm_tiled 实验，看 naive 和 tiled 的 GFLOPS。
4. 再回来看 1D register tiling。
5. 最后再看原来的 02 文档和下一章向量化、双缓冲、Tensor Core。
```

不要急着跳到 Tensor Core。GEMM 的根在这里：

```text
数据在哪里？
谁复用它？
复用几次？
```
