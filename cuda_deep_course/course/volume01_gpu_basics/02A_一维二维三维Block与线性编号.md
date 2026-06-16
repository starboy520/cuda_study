# 02A 一维、二维、三维 Block 与线性编号

这一章只解决一个问题：

> `threadIdx` 已经有 `x、y、z` 坐标了，为什么还会出现 `linearThreadId`？

先不要背公式。我们从“一群 thread 如何编号”开始。

## 1. Block 是一组 Thread

启动 kernel 时，可以把一个 block 写成一维、二维或三维：

```cpp
dim3 block1D(8);
dim3 block2D(4, 3);
dim3 block3D(4, 3, 2);
```

它们分别表示：

```text
block1D: 8 x 1 x 1 = 8 个 thread
block2D: 4 x 3 x 1 = 12 个 thread
block3D: 4 x 3 x 2 = 24 个 thread
```

这里的维度主要是程序员组织问题的坐标方式：

```text
一维数组                  -> 常用一维 block
二维图像、矩阵            -> 常用二维 block
三维体素、三维网格        -> 可以使用三维 block
```

二维 block 不需要是正方形。下面都是合法形状，只要没有超过设备限制：

```cpp
dim3 blockA(16, 16);  // 256 个 thread
dim3 blockB(32, 8);   // 256 个 thread
dim3 blockC(8, 32);   // 256 个 thread
dim3 blockD(20, 10);  // 200 个 thread
```

## 2. 一维 Block 最简单

假设：

```cpp
dim3 block(8);
```

CUDA 实际保存的是：

```text
blockDim.x = 8
blockDim.y = 1
blockDim.z = 1
```

8 个 thread 的坐标是：

```text
threadIdx.x:  0  1  2  3  4  5  6  7
```

因为它本来就是一条直线，所以 block 内的一维编号就是：

```cpp
const int linearThreadId = threadIdx.x;
```

这里的 `linear` 是“排成一条线”的意思。

## 3. 二维 Block 是一张坐标表

假设：

```cpp
dim3 block(4, 3);
```

注意 `dim3 block(x, y)` 的顺序：

```text
blockDim.x = 4
blockDim.y = 3
```

因此它有 3 行，每行 4 个 thread，共 12 个 thread。

我们通常把 `x` 画成横向的列，把 `y` 画成纵向的行：

```text
                     x
             0       1       2       3
          +-------+-------+-------+-------+
y = 0     | (0,0) | (1,0) | (2,0) | (3,0) |
          +-------+-------+-------+-------+
y = 1     | (0,1) | (1,1) | (2,1) | (3,1) |
          +-------+-------+-------+-------+
y = 2     | (0,2) | (1,2) | (2,2) | (3,2) |
          +-------+-------+-------+-------+
```

坐标写作 `(x,y)`。例如 `(2,1)` 表示：

```text
threadIdx.x = 2
threadIdx.y = 1
```

它不是“第 2 行第 1 列”，而是：

```text
x = 第 2 列
y = 第 1 行
```

下标从 0 开始。

## 4. 为什么还需要一维编号

二维坐标很适合映射矩阵，但硬件把 block 内相邻的 thread 每 32 个组成一个
warp 时，需要知道：

```text
谁排第 0
谁排第 1
谁排第 2
...
```

也就是说，需要把二维坐标按照固定规则排成一条队伍。

CUDA 的顺序是：

```text
x 最先变化
x 走完一行以后，y 增加
```

给前面的 `block=(4,3)` 加上一维编号：

```text
                     x
              0         1         2         3
          +---------+---------+---------+---------+
y = 0     | (0,0) 0 | (1,0) 1 | (2,0) 2 | (3,0) 3 |
          +---------+---------+---------+---------+
y = 1     | (0,1) 4 | (1,1) 5 | (2,1) 6 | (3,1) 7 |
          +---------+---------+---------+---------+
y = 2     | (0,2) 8 | (1,2) 9 |(2,2) 10 |(3,2) 11 |
          +---------+---------+---------+---------+
```

所以排列顺序是：

```text
(0,0), (1,0), (2,0), (3,0),
(0,1), (1,1), (2,1), (3,1),
(0,2), (1,2), (2,2), (3,2)
```

右边的 `0..11` 就是 block 内的 `linearThreadId`。

## 5. 二维公式不是凭空出现的

公式是：

```cpp
const int linearThreadId =
    threadIdx.y * blockDim.x + threadIdx.x;
```

拆开理解：

```text
threadIdx.y * blockDim.x
```

表示当前 thread 前面已经有多少个完整行。

因为：

```text
前面有 threadIdx.y 行
每一行有 blockDim.x 个 thread
```

然后：

```text
+ threadIdx.x
```

表示它在当前这一行中又向右走了多少格。

### 例一：`threadIdx=(2,1)`

```text
blockDim.x = 4
threadIdx.y = 1
threadIdx.x = 2

前面完整的行：1 * 4 = 4 个 thread
当前行向右移动：2 个 thread
linearThreadId = 4 + 2 = 6
```

在表格中，`(2,1)` 的编号确实是 6。

### 例二：`threadIdx=(3,2)`

```text
前面完整的行：2 * 4 = 8
当前行向右移动：3
linearThreadId = 8 + 3 = 11
```

它是 block 中最后一个 thread。

## 6. Linear Thread ID 只在当前 Block 内有效

这是最重要的区别之一：

```text
linearThreadId 是 thread 在自己 block 内的编号
```

每个 block 都会重新从 0 编号。

假设 Grid 中有两个一维 block，每个 block 有 4 个 thread：

```text
block 0: local linear ID = 0, 1, 2, 3
block 1: local linear ID = 0, 1, 2, 3
```

如果要得到整个 Grid 中的一维位置，才计算：

```cpp
const int globalIndex =
    blockIdx.x * blockDim.x + threadIdx.x;
```

于是：

```text
block 0 的 globalIndex = 0, 1, 2, 3
block 1 的 globalIndex = 4, 5, 6, 7
```

不要混淆：

| 名称 | 表示什么 | 是否每个 Block 重新从 0 开始 |
|---|---|---|
| `threadIdx` | thread 在 block 内的坐标 | 是 |
| `linearThreadId` | thread 在 block 内排队后的编号 | 是 |
| `globalIndex` | thread 在整个一维 Grid 中负责的位置 | 否 |

`linearThreadId` 不是 CUDA 内置变量，只是我们自己起的变量名。你也可能在代码中
看到 `tid`、`localId`、`threadRank` 等名字。

## 7. Linear Thread ID 也不是矩阵地址

假设：

```text
矩阵 width = 100
blockDim = (16, 8)
blockIdx = (2, 3)
threadIdx = (4, 6)
```

这个 thread 在自己 block 内的线性编号：

```text
linearThreadId = 6 * 16 + 4 = 100
```

它负责的矩阵坐标：

```text
col = blockIdx.x * blockDim.x + threadIdx.x
    = 2 * 16 + 4
    = 36

row = blockIdx.y * blockDim.y + threadIdx.y
    = 3 * 8 + 6
    = 30
```

矩阵行主序地址：

```text
matrixIndex = row * width + col
            = 30 * 100 + 36
            = 3036
```

因此同一个 thread 可以同时有：

```text
block 内编号 linearThreadId = 100
负责的数据坐标 (row,col) = (30,36)
数据的一维地址 matrixIndex = 3036
```

三个数字回答三个不同问题，不能互换。

## 8. Linear Thread ID 如何形成 Warp

Block 内的 thread 先得到线性编号，然后每连续 32 个组成一个 warp：

```text
linear ID  0..31  -> warp 0
linear ID 32..63  -> warp 1
linear ID 64..95  -> warp 2
```

计算：

```cpp
const int warpId = linearThreadId / 32;
const int laneId = linearThreadId % 32;
```

例如 `linearThreadId=35`：

```text
warpId = 35 / 32 = 1
laneId = 35 % 32 = 3
```

意思是它在 block 内属于第 1 号 warp，是这个 warp 中第 3 号 thread。

### `block=(16,4)` 时 Warp 长什么样

每行只有 16 个 thread：

```text
y=0 -> linear ID  0..15
y=1 -> linear ID 16..31
y=2 -> linear ID 32..47
y=3 -> linear ID 48..63
```

所以：

```text
warp 0 = y=0 的整行 + y=1 的整行
warp 1 = y=2 的整行 + y=3 的整行
```

### `block=(32,2)` 时 Warp 长什么样

每行正好有 32 个 thread：

```text
warp 0 = y=0, x=0..31
warp 1 = y=1, x=0..31
```

这说明 block 总 thread 数相同，不代表 warp 在二维图上的形状相同。

## 9. 三维 Block 只是再增加“层”

假设：

```cpp
dim3 block(4, 3, 2);
```

可以把它看成：

```text
2 层
每层 3 行
每行 4 个 thread
```

排列顺序是：

```text
x 最先变化
然后 y 变化
最后 z 变化
```

三维公式：

```cpp
const int linearThreadId =
    threadIdx.z * blockDim.y * blockDim.x
  + threadIdx.y * blockDim.x
  + threadIdx.x;
```

仍然不要死背，分成三步：

```text
threadIdx.z * blockDim.y * blockDim.x
  = 前面完整的层包含多少 thread

threadIdx.y * blockDim.x
  = 当前层中，前面完整的行包含多少 thread

threadIdx.x
  = 当前行中向右走了多少格
```

例如：

```text
blockDim = (4,3,2)
threadIdx = (2,1,1)
```

计算：

```text
前面 1 个完整层：1 * 3 * 4 = 12
当前层前面 1 个完整行：1 * 4 = 4
当前行向右走 2 格：2

linearThreadId = 12 + 4 + 2 = 18
```

## 10. 为什么 `x` 最先变化

CUDA 规定 thread 的线性排列中 `x` 维最先变化。这样做也很适合常见的矩阵
访问约定：

```text
x -> 列
y -> 行
```

行主序矩阵中，相邻列在内存中也是相邻元素。因此让同一 warp 中相邻 lane
通常拥有相邻的 `x`，经常有利于形成连续的 global-memory 访问。

这不是说 `x` 永远必须表示列，而是全书采用这个最常见、最直觉的约定。

## 11. 配套实验

运行：

```bash
cd /home/qichengjie/workspace/cuda_study/cuda_deep_course
make -C labs/02_programming_model/index_mapping clean all
./labs/02_programming_model/index_mapping/index_mapping
```

先只找这四行：

```text
thread=31 -> linear=31 warp=0 lane=31
thread=32 -> linear=32 warp=1 lane=0

local=(row=1,col=15) -> thread_linear=31 warp=0 lane=31
local=(row=2,col=0)  -> thread_linear=32 warp=1 lane=0
```

Device `printf` 的显示顺序可能是乱的。你要比较的是每一行中的坐标和计算结果，
不是它们出现在屏幕上的先后顺序。

## 12. 手算练习

### 练习一

```text
blockDim = (8,4)
threadIdx = (5,2)
```

求 `linearThreadId`、`warpId` 和 `laneId`。

答案：

```text
linearThreadId = 2 * 8 + 5 = 21
warpId = 21 / 32 = 0
laneId = 21 % 32 = 21
```

### 练习二

```text
blockDim = (16,8)
threadIdx = (3,2)
```

答案：

```text
linearThreadId = 2 * 16 + 3 = 35
warpId = 1
laneId = 3
```

### 练习三

下面两个 thread 是否具有相同的 `linearThreadId`？

```text
blockIdx=(0,0), threadIdx=(3,2)
blockIdx=(5,7), threadIdx=(3,2)
blockDim=(16,8)
```

答案：相同，都是 35。因为 `linearThreadId` 只描述 block 内位置，不包含
`blockIdx`。

## 13. 本章只记住四句话

```text
1. 一维、二维、三维只是组织 thread 坐标的方式。
2. Block 内总 thread 数是 blockDim.x * blockDim.y * blockDim.z。
3. 排成一维时 x 最先变化，然后 y，最后 z。
4. linearThreadId 是 block 内编号，不是全局数据下标。
```
