# Shared Memory 加载的两种写法（复习笔记）

> 场景：tiled GEMM 把 A/B 的 tile 从 global 搬进 shared memory。
> 同一件事有两种写法，Day 5 从「位置驱动」改成了「编号驱动（grid-stride）」。
> 一句话差别：**旧写法把「搬数据」和「算数据」绑在一起；新写法把它俩解耦。**

---

## 写法 A：二维 / 位置驱动（旧，教学版）

每个线程「搬的，正好是自己等会要算的那块」，加载下标里掺了 TM/TN。

```cuda
// sa[BM][BK]：每个线程 (tx,ty) 搬 TM*(BK/TM) 个元素
for (int i = 0; i < TM; i++) {
    for (int j = 0; j < BK/TM; j++) {
        int row    = TM * threadIdx.y + i;      // 连续 TM 行
        int column = j * TM + threadIdx.x;       // 跳步取列（步长 = blockDim.x）
        int globalRow    = blockIdx.y * BM + row;
        int globalColumn = step + column;
        sa[row][column] = (globalRow < M && globalColumn < K)
                          ? a[globalRow * K + globalColumn] : 0.0f;
    }
}
```

**隐含假设**（致命）：`blockDim.x == TM` 且 `blockDim.y == TN`。
- 现在 BM=BN=64、TM=TN=8 → blockDim=(8,8)=TM/TN，侥幸成立。
- 一旦改 TM/TN/BK/BM/BN，blockDim 不再等于 TM/TN → sa/sb **覆盖不全或越界 → 结果错**。
- 所以**不能扫参数**。

---

## 写法 B：一维 / 编号驱动（新，grid-stride，通用）

把 block 内线程拍平成一队，按一维编号轮流搬「一堆活」，加载下标只看 nthreads 和总元素数，**不出现 TM/TN**。

```cuda
int tid      = threadIdx.y * blockDim.x + threadIdx.x;  // 线程一维编号
int n_thread = blockDim.x * blockDim.y;                 // 队伍总人数

// sa[BM][BK]：搬 BM*BK 个元素
for (int i = tid; i < BM * BK; i += n_thread) {
    int row = i / BK;
    int col = i % BK;
    int gr  = blockIdx.y * BM + row;
    int gc  = step + col;
    sa[row][col] = (gr < M && gc < K) ? a[gr * K + gc] : 0.0f;
}

// sb[BK][BN]：搬 BK*BN 个元素
for (int i = tid; i < BK * BN; i += n_thread) {
    int row = i / BN;
    int col = i % BN;
    int gr  = step + row;
    int gc  = blockIdx.x * BN + col;
    sb[row][col] = (gr < K && gc < N) ? b[gr * N + gc] : 0.0f;
}
```

**无隐含假设**，任何 BM/BN/BK/TM/TN 都正确。

---

## 关键差别对照

| 维度 | 写法 A（位置驱动） | 写法 B（grid-stride） |
| --- | --- | --- |
| 加载是否含 TM/TN | 含（搬=算绑定） | 不含（搬≠算解耦） |
| 隐含假设 | blockDim.x==TM, blockDim.y==TN | 无 |
| 改 tiling 参数 | 直接崩 | 自动正确 |
| 能否扫参数调优 | 不能 | 能 ✓ |
| 循环结构 | 双层 for，次数跟参数挂钩 | 单层 grid-stride，只看 总数/线程数 |
| 思维模型 | 线程按自己 x/y 算地址 | 工人按编号 i 轮流搬箱子 |
| 工业级用哪种 | — | CUTLASS 等都用 B |

---

## 两个要点

### 1. grid-stride 的 stride 必须 = n_thread

```text
tid=0 : 0, 64, 128, ...
tid=1 : 1, 65, 129, ...
...
竖着看 = 0,1,2,3,... 连续整数，每个恰好一次 → 不重不漏。
stride < n_thread → 重复搬；stride > n_thread → 漏搬。
```

### 2. 一维拆二维：row=i/W, col=i%W（W 是该 shared 数组的列数）

- sa[BM][BK] → `row=i/BK, col=i%BK`
- sb[BK][BN] → `row=i/BN, col=i%BN`
- **相邻 tid → 相邻 col → global 连续地址 → coalesced**（合并访问，省带宽）。
  若映射反了（相邻 tid 跨行），就变跨行跳读、不合并、变慢。

---

## 反直觉但关键的一点

写法 B 里，**一个线程搬的数据，往往不是它自己要算的那块**。
能成立是因为：搬完有 `__syncthreads()`，此刻整块 sa/sb 已在 shared memory 就绪，
**谁搬的不重要，计算阶段每个线程各取所需即可**。
这就是「加载 / 计算解耦」的前提：shared memory 是 block 内共享的。

---

## 何时用哪种

```text
学习 / 讲清 tiling 直觉      → A 更直观（搬=算，好画图）
真正要调优、扫 tile 参数     → 必须 B（解耦、通用、可控 coalescing）
```
