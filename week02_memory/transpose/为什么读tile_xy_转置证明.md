# 为什么转置读的是 tile[tx][ty]（最难点彻底讲清）

> 源文件：`week02_memory/transpose/transpose.cu` 的 `transpose_shared`（版本 2b）
> 本文只攻一个点：**写入 `tile[ty][tx]`，读出却是 `tile[tx][ty]`，下标为什么互换？**

---

## 0. 先看这段代码

```cuda
__global__ void transpose_shared(const float* in, float* out, int width, int height) {
  __shared__ float tile[TILE][TILE + 1];

  // ── 阶段1：读 in → 写 tile ──
  const int col_in = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_in = blockIdx.y * blockDim.y + threadIdx.y;
  if (row_in < height && col_in < width) {
    tile[threadIdx.y][threadIdx.x] = in[row_in * width + col_in];   // 写 tile[ty][tx]
  }
  __syncthreads();

  // ── 阶段2：读 tile → 写 out ──
  const int col_out = blockIdx.y * blockDim.y + threadIdx.x;        // 注意用 blockIdx.y
  const int row_out = blockIdx.x * blockDim.x + threadIdx.y;        // 注意用 blockIdx.x
  if (row_out < width && col_out < height) {
    out[row_out * height + col_out] = tile[threadIdx.x][threadIdx.y]; // 读 tile[tx][ty]
  }
}
```

两个"反常"：
1. 阶段2 算 `col_out/row_out` 时，**block 坐标交换了**（col_out 用 blockIdx.**y**，row_out 用 blockIdx.**x**）。
2. 读 tile 用 `tile[tx][ty]`，和写入的 `tile[ty][tx]` **下标互换**。

本文证明：这两处交换合起来，正好实现转置 `out[r][c] = in[c][r]`，且读写都合并。

---

## 1. 符号约定

```text
block：(bx, by) = (blockIdx.x, blockIdx.y)
thread：(tx, ty) = (threadIdx.x, threadIdx.y)
TILE = 32（blockDim.x = blockDim.y = 32）
```

---

## 2. 关键事实：tile 里存了什么（阶段1）

阶段1 的写入：

```text
col_in = bx*32 + tx
row_in = by*32 + ty
tile[ty][tx] = in[row_in][col_in] = in[by*32+ty][bx*32+tx]
```

把它归纳成一条**通用规律**（对任意下标 a、b）：

```text
★ tile[a][b] = in[by*32 + a][bx*32 + b]      ……（式1）
```

> 记忆：tile 是 in 的一个 32×32 块的【原样副本】，第一维对应 in 的行、第二维对应 in 的列。
> tile 本身**没有转置**，只是把 in 的一块搬进来。

---

## 3. 代数证明：阶段2 实现了转置

### 第①步：阶段2 写到 out 的哪个位置

```text
col_out = by*32 + tx
row_out = bx*32 + ty
写位置 = out[row_out][col_out] = out[bx*32+ty][by*32+tx]   ……（式2）
```

### 第②步：阶段2 读出的值是什么（用式1）

读的是 `tile[tx][ty]`，把 a=tx、b=ty 代入式1：

```text
tile[tx][ty] = in[by*32 + tx][bx*32 + ty]                  ……（式3）
```

### 第③步：合并式2、式3，看是否=转置

这条语句实际做的是：

```text
out[bx*32+ty][by*32+tx] = in[by*32+tx][bx*32+ty]
   └──── 写位置(式2)───┘   └──── 读的值(式3)───┘
```

令 out 的行列为：

```text
r = bx*32 + ty   （out 的行）
c = by*32 + tx   （out 的列）
```

代入右边的 in：

```text
in[by*32+tx][bx*32+ty] = in[c][r]   （因为 c=by*32+tx，r=bx*32+ty）
```

于是：

```text
out[r][c] = in[c][r]   ✓✓✓   正好是转置定义！  证毕 ∎
```

---

## 3.5 用具体数字完整走一遍（代数太抽象就看这个）

为了让上面的代数落地，我们用 **TILE=4** 的小例子，追踪一个具体线程，看数据怎么流动。

设 block (bx=1, by=0)，看线程 (tx=3, ty=1)。in 是 8×8 矩阵，行优先。

### 阶段1：这个线程读 in、写 tile

```text
col_in = bx*4 + tx = 1*4 + 3 = 7
row_in = by*4 + ty = 0*4 + 1 = 1
读 in[1][7]，写到 tile[ty][tx] = tile[1][3]
→ tile[1][3] = in[1][7]
```

### 阶段2：这个线程读 tile、写 out

```text
col_out = by*4 + tx = 0*4 + 3 = 3
row_out = bx*4 + ty = 1*4 + 1 = 5
读 tile[tx][ty] = tile[3][1]，写到 out[row_out][col_out] = out[5][3]
→ out[5][3] = tile[3][1]
```

### 关键：tile[3][1] 是谁写进去的？

注意阶段2 这个线程读的是 **tile[3][1]**，但它阶段1 写的是 **tile[1][3]**——
**不是同一个格子！** tile[3][1] 是【另一个线程】在阶段1 写的。

谁写了 tile[3][1]？用式1 反推（tile[a][b]=in[by*4+a][bx*4+b]，a=3,b=1）：

```text
tile[3][1] = in[by*4+3][bx*4+1] = in[0*4+3][1*4+1] = in[3][5]
（也就是阶段1 里 ty=3、tx=1 那个线程写的：它读 in[3][5]）
```

### 拼起来验证

```text
out[5][3] = tile[3][1] = in[3][5]
→ out[5][3] = in[3][5]
对照转置定义 out[r][c]=in[c][r]：r=5, c=3 → in[c][r]=in[3][5] ✓ 完全正确！
```

### 这个例子说明的核心

```text
线程 (tx=3,ty=1)：
  阶段1 写 tile[1][3]（存 in[1][7]）
  阶段2 读 tile[3][1]（取 in[3][5]，是别人写的）
→ 同一线程，写和读碰的是 tile 的【对角对称】两格：[1][3] ↔ [3][1]
→ "转置"就藏在这一对对角格子的互换里
→ 而且 out[5][3]=in[3][5] 正是转置，证明映射对了
```

> 这就是为什么读 `tile[tx][ty]`：线程要写的 out 位置，对应的源数据被【对角线另一侧
> 的线程】存在了 tile[tx][ty]。读自己写的格子（tile[ty][tx]）反而是错的。

---

## 4. 反证：如果读 tile[ty][tx]（不互换）会怎样

把读出的值换成 `tile[ty][tx]`，用式1（a=ty、b=tx）：

```text
tile[ty][tx] = in[by*32+ty][bx*32+tx]
语句变成：out[bx*32+ty][by*32+tx] = in[by*32+ty][bx*32+tx]
令 r=bx*32+ty, c=by*32+tx：
  右边 = in[by*32+ty][bx*32+tx]
       ≠ in[c][r]      （c 含 tx、r 含 ty，但这里 in 第一维含 ty、第二维含 tx → 对不上）
→ 不是转置，结果错！
```

**结论：必须 `tile[tx][ty]` 互换下标，才能让读到的值 = in[c][r]。**

---

## 5. 一张表把三个式子串起来

| 量 | 表达式 |
|---|---|
| tile[a][b] 存什么（式1） | `in[by*32+a][bx*32+b]` |
| 写到 out 的位置（式2） | `out[bx*32+ty][by*32+tx]` |
| 读的 tile[tx][ty] 值（式3） | `in[by*32+tx][bx*32+ty]` |
| 合并 | `out[bx*32+ty][by*32+tx] = in[by*32+tx][bx*32+ty]` |
| 令 r=bx*32+ty, c=by*32+tx | `out[r][c] = in[c][r]` ✓ |

---

## 6. 直观理解：数据在 tile 内"对角翻转"

```text
写入：线程(tx,ty) 把数据放进 tile[ty][tx]
读出：线程(tx,ty) 取的却是 tile[tx][ty]（对角对称的格子）

→ 同一个线程，写入和读出碰的是 tile 里【两个不同的对角格子】
→ "转置"就发生在这一步：读对角对称的格子 = 行列互换 = 转置
```

### 用 4×4 tile 画出"对角对称"

阶段1 写入后，tile 的内容（每格标"哪个线程写的 [ty,tx]"，TILE=4）：

```text
            tx=0      tx=1      tx=2      tx=3
   ty=0   [0,0]     [0,1]     [0,2]     [0,3]
   ty=1   [1,0]     [1,1]     [1,2]    ▲[1,3]
   ty=2   [2,0]     [2,1]     [2,2]     [2,3]
   ty=3   [3,0]    ▼[3,1]     [3,2]     [3,3]
                    主对角线方向：[i][j] 与 [j][i] 关于左上-右下对角线对称
```

```text
线程 (tx=3,ty=1)：
  写入 ▲ tile[1][3]
  读出 ▼ tile[3][1]   ← [1][3] 和 [3][1] 是对角对称的两格
→ 它读的不是自己写的格子，而是对角线另一侧的格子
→ 全部线程一起这么做 = 整个 tile 沿主对角线翻转 = 块内转置
```

口诀：**写 tile[y][x]（原样存），读 tile[x][y]（互换取）= 块内转置。**

### 为什么"块内转置 + block 坐标交换"= 整体转置

```text
单个 32×32 块：靠 tile[x][y] 互换，块【内部】转置
块的位置：    靠 block 坐标交换（col_out 用 by、row_out 用 bx），
             让"in 的 (by,bx) 块"落到"out 的 (bx,by) 块"
→ 块内转置 + 块位置转置 = 整个矩阵转置（两个层次都翻）
```

> 这正是另一份文档《矩阵转置两层转置图解.md》讲的"两层转置"：
> ① 块与块之间的位置要转置（block 坐标交换）
> ② 每个块内部也要转置（tile 下标交换）
> 缺一层都不对。


---

## 7. 为什么这样读写都合并（性能）

合并 = 同一个 warp（tx 连续 0~31、ty 固定）访问的全局地址连续。

```text
读 in：in[(by*32+ty)*width + (bx*32+tx)]
  tx 变化 → 地址变化系数 = 1 → 连续 ✓ 合并

写 out：out[(bx*32+ty)*height + (by*32+tx)]
  tx 变化 → 地址变化系数 = 1（col_out=by*32+tx，tx 在最低位）→ 连续 ✓ 合并
```

对比 simple 版（`out[col*height+row]`，col=bx*32+tx）：

```text
写 out：tx 变化 → 地址变化 = tx*height（跨 height 跳）→ 不合并 ✗
```

→ 2b 版通过"block 坐标交换"，让 tx 在写 out 时对应连续的**列**方向，所以写也合并。
这就是 2b 比 simple 快的根本原因：**读、写两端都合并**。

---

## 8. +1 padding 的作用

```cuda
__shared__ float tile[TILE][TILE + 1];   // 注意 +1
```

读 `tile[tx][ty]` 时，同一 warp（tx=0~31、ty 固定）按**列**访问 shared：
`tile[0][ty], tile[1][ty], ..., tile[31][ty]`。

```text
若 tile[32][32]：这 32 个地址相差 32（一行的宽度）→ 全落在同一个 bank → 32 路 bank conflict
若 tile[32][33]：行宽变 33，相邻列访问错开 1 个 bank → 落到不同 bank → 消除 conflict
```

→ `+1` padding 用一列的浪费，换来消除"按列读 shared"的 bank conflict。

---

## 9. 三处交换缺一不可（总结）

代码里有三处"交换"，合起来才等于转置：

```text
① block 坐标交换：col_out 用 blockIdx.y、row_out 用 blockIdx.x
   → 写位置 = out[bx*32+ty][by*32+tx]
② tile 下标交换：读 tile[tx][ty]（不是 tile[ty][tx]）
   → 读的值 = in[by*32+tx][bx*32+ty]
③ ①②合起来 → out[r][c] = in[c][r] = 转置
```

去掉任何一个交换，等式就不成立 → 结果错。

---

## 10. 通用方法：以后任何"映射对不对"都能这样验

```text
1. 写出每个数组下标的【代数表达式】（用 bx/by/tx/ty）
2. 把"写入阶段"的规律（式1：tile[a][b]=...）代入"读出语句"
3. 化简，对照定义（转置是 out[r][c]=in[c][r]）
→ 不用追踪具体线程，纯代入就能证明对错
```

> 一句话：写 `tile[ty][tx]` 原样存 in，读 `tile[tx][ty]` 互换取——把式1（tile 存了什么）
> 代进写回语句，化简得 `out[r][c]=in[c][r]`，即转置。下标互换正是"转置"在 shared 内发生的地方。
