# 02B GEMM Register Tiling：外积视角小白版

这份文档只讲一件事：

> 用“外积”理解 GEMM register tiling。

如果你用“点积”视角看 GEMM，容易看到的是：

```text
一个 thread 算一个 C[row][col]
```

但 register tiling 不是让一个 thread 只算一个点，而是：

```text
一个 thread 算一个 C 的小矩形。
```

这时外积视角会更自然。

## 0. 先记一句话

GEMM 可以有两种看法。

点积视角：

```text
C[row][col] = A 的一行 · B 的一列
```

外积视角：

```text
C += A 的一列 × B 的一行
```

更具体一点：

```text
C = A[:,0] × B[0,:]
  + A[:,1] × B[1,:]
  + A[:,2] × B[2,:]
  + ...
```

每个 `k` 都贡献一张“小矩阵”到 C。

register tiling 的本质就是：

```text
一个 thread 手里拿着一个很小的 C 小矩阵 acc。
每次取一小列 A 和一小行 B。
做一次外积，更新 acc。
```

## 1. 什么是外积

先看一个很小的例子。

有一个列向量：

```text
a =
  a0
  a1
```

有一个行向量：

```text
b = b0  b1
```

外积是：

```text
a × b =

        b0      b1
a0   a0*b0   a0*b1
a1   a1*b0   a1*b1
```

也就是：

```text
结果是一个 2x2 小矩阵。
```

写成更新 C：

```text
C00 += a0 * b0
C01 += a0 * b1
C10 += a1 * b0
C11 += a1 * b1
```

这四行很重要。register tiling 后面本质上就是这个。

## 2. GEMM 为什么可以看成很多次外积

GEMM 的定义：

```text
C[row][col] = Σ_k A[row][k] * B[k][col]
```

换个角度，不固定 `row,col`，而是固定 `k`。

当 `k=0` 时：

```text
A[:,0] 是 A 的第 0 列
B[0,:] 是 B 的第 0 行
```

它们做外积，得到一张和 C 同形状的贡献矩阵：

```text
C += A[:,0] × B[0,:]
```

当 `k=1` 时：

```text
C += A[:,1] × B[1,:]
```

一直加到 `k=K-1`：

```text
C = Σ_k A[:,k] × B[k,:]
```

这和点积公式是同一件事，只是视角不同。

## 3. 先看一个 2x2 C Tile

假设某个 block 或 thread 正在更新一个 `2x2` 的 C 小块：

```text
acc =

        col0   col1
row0    acc00  acc01
row1    acc10  acc11
```

这里叫 `acc`，因为它不是最终 C，而是“正在累加中的 C 小块”。

现在固定某个 `k`。

它需要：

```text
A 的一小列：

regA[0] = A[row0][k]
regA[1] = A[row1][k]

B 的一小行：

regB[0] = B[k][col0]
regB[1] = B[k][col1]
```

然后做外积：

```text
        regB[0]       regB[1]
regA[0] acc00 += A0B0 acc01 += A0B1
regA[1] acc10 += A1B0 acc11 += A1B1
```

代码：

```cpp
acc[0][0] += regA[0] * regB[0];
acc[0][1] += regA[0] * regB[1];
acc[1][0] += regA[1] * regB[0];
acc[1][1] += regA[1] * regB[1];
```

这就是最小的 register tile。

## 4. 这为什么叫 Register Tiling

因为这个 `acc` 小矩阵放在 thread 自己的寄存器里。

比如一个 thread 算 `2x2` 个输出：

```cpp
float acc[2][2] = {
  {0.0f, 0.0f},
  {0.0f, 0.0f},
};
```

如果一个 thread 算 `4x4` 个输出：

```cpp
float acc[4][4] = {0.0f};
```

这就是：

```text
thread tile = 一个 thread 负责的 C 小矩形
```

它通常放在寄存器里，所以叫：

```text
register tile
```

## 5. 外积视角下的核心循环

假设：

```text
TM = thread tile 的行数
TN = thread tile 的列数
```

一个 thread 负责：

```text
TM x TN 个 C 输出
```

它的寄存器累加器是：

```cpp
float acc[TM][TN] = {0.0f};
```

在某个 K 段中，每次 `k`：

```cpp
float regA[TM];
float regB[TN];
```

`regA` 是一个小列向量：

```text
regA[0]
regA[1]
...
regA[TM-1]
```

`regB` 是一个小行向量：

```text
regB[0] regB[1] ... regB[TN-1]
```

然后做外积：

```cpp
for (int i = 0; i < TM; ++i) {
  for (int j = 0; j < TN; ++j) {
    acc[i][j] += regA[i] * regB[j];
  }
}
```

这段代码你可以读成：

```text
regA 小列向量
乘
regB 小行向量
更新
acc 小矩阵
```

这就是 register tiling 的核心。

## 6. 它和 Shared Memory Tiling 怎么接起来

shared-memory tiling 先做 block 级复用：

```text
一个 block 负责一个 C block tile。
block 把 A/B 的当前 K 段搬进 shared memory。
```

shared memory 里有：

```cpp
As[BM][BK]
Bs[BK][BN]
```

这里：

```text
BM = block tile 的行数
BN = block tile 的列数
BK = 每次处理的 K 段厚度
```

register tiling 接在后面：

```text
block tile 里的每个 thread，
不再只算一个 C 点，
而是算一个小的 thread tile。
```

所以层级是：

```text
Grid
  -> many block tiles of C
      -> one block tile
          -> many thread tiles
              -> one thread owns acc[TM][TN]
```

一张更短的图：

```text
C matrix
  -> C block tile       放到一个 block
      -> C thread tile  放到一个 thread 的寄存器 acc
```

## 7. 一个 Block Tile 如何切成 Thread Tile

假设一个 block 负责：

```text
BM x BN = 64 x 64 的 C block tile
```

假设每个 thread 负责：

```text
TM x TN = 4 x 4 的 C thread tile
```

那么这个 block tile 里有多少个 thread tile？

```text
行方向：BM / TM = 64 / 4 = 16 个 thread tile
列方向：BN / TN = 64 / 4 = 16 个 thread tile
```

总共：

```text
16 x 16 = 256 个 thread
```

也就是说：

```text
一个 block 用 256 个 thread。
每个 thread 算 4x4 = 16 个 C 输出。
整个 block 算 64x64 = 4096 个 C 输出。
```

检查：

```text
256 thread x 每 thread 16 输出 = 4096 输出
```

正好等于：

```text
64 x 64
```

## 8. Thread 怎么知道自己负责哪个 Thread Tile

一个 block 里有 256 个 thread。

先把一维 `threadIdx.x` 转成二维 thread tile 坐标。

如果：

```text
threadsPerRow = BN / TN = 64 / 4 = 16
```

那么：

```cpp
int threadTileCol = threadIdx.x % threadsPerRow;
int threadTileRow = threadIdx.x / threadsPerRow;
```

例如：

```text
threadIdx.x = 37
```

则：

```text
threadTileCol = 37 % 16 = 5
threadTileRow = 37 / 16 = 2
```

意思是：

```text
这个 thread 负责 block tile 中
第 2 个 thread tile 行
第 5 个 thread tile 列
```

因为每个 thread tile 是 `4x4`，所以它负责的 C 局部范围是：

```text
行：threadTileRow * TM + 0..3
列：threadTileCol * TN + 0..3
```

代入：

```text
行：2 * 4 + 0..3 = 8..11
列：5 * 4 + 0..3 = 20..23
```

所以这个 thread 的 `acc[4][4]` 对应：

```text
C block tile 内 rows 8..11, cols 20..23
```

## 9. 再加上 Block 在 Grid 中的位置

上面只是 block tile 内的局部位置。

如果：

```text
blockIdx.x = bx
blockIdx.y = by
```

那么这个 block 负责的全局 C 起点是：

```text
blockRowBase = by * BM
blockColBase = bx * BN
```

一个 thread tile 的全局起点是：

```cpp
int cRowBase = blockIdx.y * BM + threadTileRow * TM;
int cColBase = blockIdx.x * BN + threadTileCol * TN;
```

然后：

```text
acc[i][j] 对应 C[cRowBase + i][cColBase + j]
```

这句话很重要。

register tiling 不是脱离 grid/block 的魔法，它只是把一个 thread 负责的输出从：

```text
1 个 C 元素
```

变成：

```text
TM x TN 个 C 元素
```

## 10. 从 Shared Memory 读 regA 和 regB

假设 block 已经把当前 K 段加载到：

```text
As[BM][BK]
Bs[BK][BN]
```

对于一个 thread，它的 `acc[TM][TN]` 对应 C 的一小块。

在某个 `dotIdx` 上，也就是当前 K 段内部的一个 k：

它需要 A 的一个小列：

```cpp
for (int i = 0; i < TM; ++i) {
  regA[i] = As[threadTileRow * TM + i][dotIdx];
}
```

也需要 B 的一个小行：

```cpp
for (int j = 0; j < TN; ++j) {
  regB[j] = Bs[dotIdx][threadTileCol * TN + j];
}
```

然后做外积：

```cpp
for (int i = 0; i < TM; ++i) {
  for (int j = 0; j < TN; ++j) {
    acc[i][j] += regA[i] * regB[j];
  }
}
```

这三段连起来就是：

```text
从 shared 取 A 小列到 register
从 shared 取 B 小行到 register
在 register 里做外积更新 acc 小矩阵
```

## 11. 一个 K 段内的完整 Register Tile 计算

假设当前 shared 里已经有一个 K 段：

```text
As[BM][BK]
Bs[BK][BN]
```

一个 thread 的计算是：

```cpp
float acc[TM][TN] = {0.0f};

for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
  float regA[TM];
  float regB[TN];

  for (int i = 0; i < TM; ++i) {
    regA[i] = As[threadTileRow * TM + i][dotIdx];
  }

  for (int j = 0; j < TN; ++j) {
    regB[j] = Bs[dotIdx][threadTileCol * TN + j];
  }

  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      acc[i][j] += regA[i] * regB[j];
    }
  }
}
```

这段里最重要的是最后两层循环。

它不是随便嵌套的。

它是在做：

```text
regA[TM x 1] 外积 regB[1 x TN]
得到 TM x TN 的贡献
累加到 acc[TM x TN]
```

## 12. 加上外层 K 分段

真实 GEMM 里 K 可能很长，所以 block 仍然沿 K 分段。

完整结构是：

```cpp
float acc[TM][TN] = {0.0f};

for (int bk = 0; bk < K; bk += BK) {
  // 1. block 合作把 A/B 当前 K 段搬到 As/Bs
  load As[BM][BK];
  load Bs[BK][BN];
  __syncthreads();

  // 2. 每个 thread 用外积更新自己的 acc
  for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
    load regA[TM] from As;
    load regB[TN] from Bs;

    acc += outer_product(regA, regB);
  }

  __syncthreads();
}

// 3. 写回 acc 到 C
store acc[TM][TN] to C;
```

你可以把它读成：

```text
block 级：
  每轮搬一个 A/B 的 K 段到 shared。

thread 级：
  每个 thread 在这一段里做 BK 次小外积。

最终：
  所有 K 段的小外积累加起来，就是这个 thread 负责的 C 小矩形。
```

## 13. 为什么外积解释 register tiling 更自然

点积视角下：

```text
每个 C[row][col] 都是一条独立的点积。
```

如果一个 thread 算 `4x4` 个 C，你会觉得：

```text
它怎么突然同时算 16 条点积？
```

外积视角下：

```text
每个 k 都用 A 小列和 B 小行更新一个 C 小矩形。
```

这就非常自然：

```text
thread 手里有 acc[4][4]。
每个 k：
  取 regA[4]
  取 regB[4]
  做一次 4x4 外积
  更新 acc[4][4]
```

所以 register tiling 可以这样记：

```text
一个 thread 反复做小外积。
```

## 14. 为什么这样减少 shared memory 读取

假设 `TM=4, TN=4`。

每个 `dotIdx`：

```text
读 4 个 A 到 regA
读 4 个 B 到 regB
做 16 次 FMA
```

也就是：

```text
8 次 shared 读取
16 次计算
```

如果没有 register tiling，每个 thread 只算一个 C 输出：

```text
读 1 个 A
读 1 个 B
做 1 次 FMA
```

也就是：

```text
2 次 shared 读取
1 次计算
```

对比：

```text
普通 shared tiling:
  每 2 次 shared 读取，做 1 次 FMA。

register tiling 4x4:
  每 8 次 shared 读取，做 16 次 FMA。
```

这就是复用：

```text
regA[i] 被 TN 个输出复用。
regB[j] 被 TM 个输出复用。
```

## 15. 代价是什么

register tiling 不是免费午餐。

如果 `TM=4, TN=4`，每个 thread 至少需要：

```text
acc[4][4] = 16 个累加器
regA[4]   = 4 个临时
regB[4]   = 4 个临时
```

还要加上索引、指针等变量。

所以：

```text
每个 thread 用的寄存器变多。
```

寄存器用多了，可能导致：

```text
一个 SM 能同时驻留的 thread/warp 变少。
occupancy 降低。
```

但它仍然可能更快，因为：

```text
每个 thread 做了更多有用计算。
shared memory 读取更少。
指令级并行更多。
```

这就是 GEMM 调参的核心权衡：

```text
更大的 thread tile:
  复用更多，计算更密集
  但寄存器更多，occupancy 可能下降
```

## 16. 1D Register Tiling 是 2D 的简化版

如果 `TN=1`，那么 thread tile 是：

```text
TM x 1
```

这就是 1D register tiling。

例如 `TM=4, TN=1`：

```text
acc[0]
acc[1]
acc[2]
acc[3]
```

每个 `dotIdx`：

```text
regA = 4 个 A
regB = 1 个 B
```

外积：

```text
4x1 小外积
```

代码：

```cpp
float b = Bs[dotIdx][threadCol];

for (int i = 0; i < TM; ++i) {
  acc[i] += As[threadRow * TM + i][dotIdx] * b;
}
```

这就是你之前看到的：

```text
Btmp 读一次，被 TM 个输出复用。
```

所以：

```text
1D register tiling = TN=1 的外积。
2D register tiling = TM 和 TN 都大于 1 的外积。
```

## 17. 一份简化的 2D Register Tiling 骨架

下面不是完整可直接跑的 kernel，只是看结构。

```cpp
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 4;
constexpr int TN = 4;

__global__ void gemmRegisterTiledSkeleton(const float* A,
                                          const float* B,
                                          float* C,
                                          int M,
                                          int N,
                                          int K) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  constexpr int threadTilesPerRow = BN / TN;

  int threadTileCol = threadIdx.x % threadTilesPerRow;
  int threadTileRow = threadIdx.x / threadTilesPerRow;

  int cRowBase = blockIdx.y * BM + threadTileRow * TM;
  int cColBase = blockIdx.x * BN + threadTileCol * TN;

  float acc[TM][TN] = {0.0f};

  for (int bk = 0; bk < K; bk += BK) {
    // 这里省略 cooperative loading。
    // 真实代码要让 block 内所有 thread 一起加载 As/Bs。
    __syncthreads();

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float regA[TM];
      float regB[TN];

      for (int i = 0; i < TM; ++i) {
        regA[i] = As[threadTileRow * TM + i][dotIdx];
      }

      for (int j = 0; j < TN; ++j) {
        regB[j] = Bs[dotIdx][threadTileCol * TN + j];
      }

      for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += regA[i] * regB[j];
        }
      }
    }

    __syncthreads();
  }

  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      int row = cRowBase + i;
      int col = cColBase + j;
      if (row < M && col < N) {
        C[row * N + col] = acc[i][j];
      }
    }
  }
}
```

这个骨架里最应该看的不是加载部分，而是：

```text
threadTileRow / threadTileCol
cRowBase / cColBase
acc[TM][TN]
regA[TM]
regB[TN]
acc += regA outer regB
```

## 18. 和 Tensor Core 的关系

Tensor Core 也可以从外积或小矩阵乘的角度理解。

普通 CUDA Core register tiling：

```text
thread 自己维护 acc 小矩阵。
用普通 FMA 做小外积。
```

Tensor Core：

```text
warp 级别维护矩阵 fragment。
硬件一次做小矩阵乘加。
```

所以你现在学外积视角是很有价值的。

它会帮助你理解：

```text
C tile
thread tile
fragment
MMA
WMMA
CUTLASS 的层次化 tiling
```

但先不要急着看 Tensor Core。先把 register tiling 的外积吃透。

## 19. 用一句话总结

点积视角：

```text
一个 C 元素是一条点积。
```

外积视角：

```text
一个 k 会生成一张贡献小矩阵。
```

register tiling：

```text
一个 thread 手里有一个 acc 小矩阵。
每个 k 取 regA 小列和 regB 小行。
做一次外积，更新 acc。
```

最终：

```text
所有 k 的小外积累加起来，
就是这个 thread 负责的 C 小矩形。
```

## 20. 检查自己是否理解

你应该能回答：

```text
1. 外积为什么会得到一个矩阵？
2. GEMM 为什么可以写成很多次外积相加？
3. acc[TM][TN] 表示什么？
4. regA[TM] 和 regB[TN] 分别来自哪里？
5. 为什么 acc[i][j] += regA[i] * regB[j] 是外积？
6. 一个 block tile 怎么切成多个 thread tile？
7. threadTileRow/threadTileCol 是什么？
8. cRowBase/cColBase 是什么？
9. 1D register tiling 为什么可以看成 TN=1？
10. register tiling 为什么会增加寄存器压力？
```

最重要的三句话：

```text
Block 负责 C 的大 tile。
Thread 负责 C 的小 tile。
Thread 用外积不断更新自己的 acc 小矩阵。
```
