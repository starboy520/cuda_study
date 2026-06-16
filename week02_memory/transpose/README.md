# Matrix Transpose（统一 x/y 约定版）

**详细原理说明：** [转置优化详解.md](./转置优化详解.md)（4×4 例子 + 读写分工 + 三版本对比）

**入门图解：** [矩阵转置两层转置图解.md](./矩阵转置两层转置图解.md)（重点解释 block 位置转置 + block 内部转置）

与项目 `mat_mul_naive` 保持一致：

```text
x 方向 → col（列）
y 方向 → row（行）

global_col = blockIdx.x * blockDim.x + threadIdx.x
global_row = blockIdx.y * blockDim.y + threadIdx.y

行主序：matrix[row * width + col]
```

## 编译运行

```bash
cd week02_memory/transpose
make && ./transpose          # 1024x1024
make && ./transpose 512      # 512x512
make && ./transpose 1024 768 # 1024x768 非方阵
```

## 两个 kernel

| 版本 | 说明 |
|------|------|
| `transpose_naive` | 只用 global memory，易理解 |
| `transpose_shared` | shared memory tile + padding，Week 2 优化版 |

## 与原 column-major 示例的区别

旧示例常用 `INDX(row,col,ld) = col*ld+row`（列主序）且 **x→row, y→col**。

本目录全部改为 **行主序 + x→col, y→row**，与 Week 1 `mat_mul` 一致。
