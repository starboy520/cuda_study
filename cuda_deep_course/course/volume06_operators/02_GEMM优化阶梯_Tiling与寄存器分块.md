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

### 2.2 为什么这样能提升算术强度

关键在**复用**：一个 `TILE×TILE` 的 A 子块被 load 进 shared 一次后，block 内
`TILE` 列的输出都用它——**一次 global 读，TILE 次复用**。算术强度因此提升约 TILE 倍：

```text
naive:        每个 A 元素从 global 读 1 次，用 1 次  → AI ≈ 0.25
tiling(TILE): 每个 A 元素从 global 读 1 次，用 TILE 次 → AI ≈ 0.25 × TILE

TILE=16 → AI ≈ 4 ；TILE=32 → AI ≈ 8 ... 离拐点 25 更近了
```

### 2.3 代码骨架

```cpp
#define TILE 16
__global__ void gemmTiled(const float* A, const float* B, float* C,
                          int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float sum = 0.0f;

  // 沿 K 维一段一段处理
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    // ① 协作加载：每个线程搬一个 A 元素、一个 B 元素进 shared
    int aCol = t * TILE + threadIdx.x;
    int bRow = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
    __syncthreads();                         // ② 等整块加载完

    // ③ 从 shared 做部分点积（复用！）
    for (int k = 0; k < TILE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();                         // ④ 等大家算完再 load 下一块
  }

  if (row < M && col < N) C[row * N + col] = sum;
}
```

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
