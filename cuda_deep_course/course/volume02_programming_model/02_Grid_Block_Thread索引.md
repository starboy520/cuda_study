# 02 Grid、Block、Thread 索引

## 1. 索引的核心任务

每个 thread 必须知道：

> 我负责整个数据中的哪个元素？

答案由两层位置组成：

```text
Block 在 Grid 中的位置
Thread 在 Block 中的位置
```

## 2. 一维全局索引

```cpp
const int index =
    blockIdx.x * blockDim.x + threadIdx.x;
```

其中：

```text
blockIdx.x * blockDim.x = 当前 block 前面已有多少 thread
threadIdx.x             = 当前 thread 在 block 内的位置
```

### 数字例子

```text
blockDim.x = 256
blockIdx.x = 3
threadIdx.x = 10

index = 3 x 256 + 10 = 778
```

这个 thread 负责数组元素 `data[778]`。

### 2.1 反向推导

给定全局 index 和 block size：

```text
blockIdx = index / blockDim
threadIdx = index % blockDim
```

例如 `index=778, blockDim=256`：

```text
blockIdx=3
threadIdx=10
```

## 3. Block 数向上取整

```cpp
const int blocks =
    (count + threads - 1) / threads;
```

整数除法默认向下取整，通过加 `threads - 1` 实现向上取整。

例如：

```text
count = 257
threads = 256
blocks = (257 + 255) / 256 = 2
```

会启动 512 个 thread，多余 thread 由边界判断退出。

## 4. Grid-Stride Loop

当一个 thread 需要处理多个元素时，可用：

```cpp
for (int index = blockIdx.x * blockDim.x + threadIdx.x;
     index < count;
     index += blockDim.x * gridDim.x) {
  output[index] = input[index];
}
```

步长是整个 Grid 的 thread 总数。

优点：

- Grid 大小可以与 SM 数量配合。
- 一个 kernel 处理任意大数组。
- 便于复用 thread。

本章先理解公式，后续 reduction 和性能章节会实际使用。

### 4.1 为什么不重复处理

所有 thread 初始 index 不同，之后都增加相同的总 thread 数：

```text
thread 0: 0, T, 2T...
thread 1: 1, T+1, 2T+1...
```

这些序列覆盖不同余数类，因此不会重叠。

## 5. 二维索引约定

全书默认：

```text
x = column = 列
y = row    = 行
```

所以：

```cpp
const int col =
    blockIdx.x * blockDim.x + threadIdx.x;

const int row =
    blockIdx.y * blockDim.y + threadIdx.y;
```

行主序一维地址：

```cpp
const int index = row * width + col;
```

边界判断：

```cpp
if (row < height && col < width) {
  output[row * width + col] = ...;
}
```

### 5.1 为什么不是 `col * height + row`

那是列主序的典型形式。行主序中一行连续，所以先跳过 `row` 个完整行，每行
`width` 个元素。

## 6. 一个非方阵例子

输入矩阵：

```text
height = 7 行
width = 10 列
```

选择：

```cpp
dim3 block(4, 3);
```

表示：

```text
blockDim.x = 4 列 thread
blockDim.y = 3 行 thread
每个 block = 4 x 3 = 12 thread
```

Grid：

```cpp
dim3 grid(
    (width + block.x - 1) / block.x,
    (height + block.y - 1) / block.y);
```

数值：

```text
grid.x = ceil(10 / 4) = 3
grid.y = ceil(7 / 3) = 3
```

最后一列和最后一行 block 中会有越界 thread，所以仍需要边界判断。

## 7. Block 可以是 m x n

合法例子：

```cpp
dim3 blockA(16, 16);  // 256 threads
dim3 blockB(32, 8);   // 256 threads
dim3 blockC(8, 32);   // 256 threads
```

形状选择会影响：

- Warp 中 thread 的二维分布。
- Global memory 地址是否连续。
- Shared-memory tile 的设计。
- 边界浪费。
- 算法的数据方向。

线程总数相同，不代表性能相同。

## 8. Block 形状不等于 Tile 形状

例如高性能 transpose 常用：

```text
block = 32 x 8 threads
tile = 32 x 32 elements
```

每个 thread 循环处理 4 行：

```cpp
for (int offset = 0; offset < 32; offset += 8) {
  tile[threadIdx.y + offset][threadIdx.x] = ...;
}
```

Block 描述 thread，tile 描述数据。

## 9. Thread 的线性化顺序

这里要区分两类编号：

```text
block 内线性 thread ID：
  回答“这个 thread 在自己 block 中排第几个？”

矩阵线性下标：
  回答“这个 thread 负责的矩阵元素在内存中排第几个？”
```

它们都可能是一个整数，但意义不同。

二维 block 中：

```cpp
const int linearThread =
    threadIdx.y * blockDim.x + threadIdx.x;
```

硬件按这个线性顺序每 32 个 thread 组成一个 warp。

所以 `block=(32,8)` 中，一个 warp 通常覆盖固定 `y` 的连续 32 个 `x`。

公式中的：

```text
threadIdx.y * blockDim.x
```

表示跳过当前 thread 前面的完整行；每行有 `blockDim.x` 个 thread。然后加上
`threadIdx.x`，表示在当前行中向右走了多少格。

完整坐标图、三维公式和手算练习见卷一：

[一维、二维、三维 Block 与线性编号](../volume01_gpu_basics/02A_一维二维三维Block与线性编号.md)。

### 9.1 Warp 与二维边界

`block=(16,16)` 中 warp 横跨两行。矩阵最右边边界可能让一个 warp 中两段 lane
分别对应两行，理解线性化有助于判断分歧和地址。

## 10. 配套实验

构建并运行：

```bash
make -C labs/02_programming_model/index_mapping clean all
./labs/02_programming_model/index_mapping/index_mapping
```

输出包含：

- `blockIdx`
- `threadIdx`
- 全局 row/col
- block 内线性 thread ID
- warp ID 和 lane ID
- 矩阵的行主序线性下标

Device `printf` 的输出顺序不能作为线程执行顺序或同步保证。

## 11. 手工推演

给定：

```text
blockDim = (4, 3)
blockIdx = (2, 1)
threadIdx = (3, 2)
width = 10
```

计算：

```text
col = 2 x 4 + 3 = 11
row = 1 x 3 + 2 = 5
linear = 5 x 10 + 11 = 61
```

但 `width=10` 时合法列是 `0..9`，所以该 thread 的 `col=11` 越界，不能访问
矩阵。不要因为线性下标能算出来就跳过二维边界检查。

### 另一个有效 Thread

同样参数，若：

```text
blockIdx=(1,1)
threadIdx=(3,2)
```

则：

```text
col=7
row=5
index=57
```

在 `7x10` 矩阵中有效。

## 12. 故障注入

### 交换 row 和 col

将：

```cpp
index = row * width + col;
```

故意写成：

```cpp
index = col * width + row;
```

在一个填有 `row * 100 + col` 的非方阵中观察结果。

### 错误 Grid

故意使用：

```cpp
grid.x = width / block.x;
```

观察不能整除时末尾元素未处理。

### 删除边界检查

使用非整除尺寸运行 Compute Sanitizer，观察越界。

## 13. 面试问题

1. 一维全局索引如何推导？
2. 为什么 block 数要向上取整？
3. `block=(32,8)` 有多少 thread 和 warp？
4. 二维 block 为什么不必是正方形？
5. Block 形状和 tile 形状有什么区别？
6. 为什么 `x=col, y=row` 有利于行主序连续访问？
7. Device `printf` 顺序能否证明 thread 执行顺序？

## 14. 小结

```text
全局坐标 = block 坐标贡献 + thread 局部坐标。
x 默认对应列，y 默认对应行。
非整除输入必须向上取整 Grid 并做边界判断。
Block 可以是长方形，且不等于数据 tile。
```

## 15. 资料映射

- CUDA Programming Guide：Programming Model。
- CUDA Programming Guide：Intro to CUDA C++、Writing SIMT Kernels。
- PMPP：Multidimensional grids、data-to-thread mapping。
