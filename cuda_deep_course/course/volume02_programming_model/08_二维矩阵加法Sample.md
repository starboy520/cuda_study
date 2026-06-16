# 08 二维矩阵加法 Sample

## 1. 从数据形状出发

矩阵：

```text
height 行 x width 列
```

自然映射：

```text
x -> col
y -> row
```

## 2. Kernel

```cpp
const int col =
    blockIdx.x * blockDim.x + threadIdx.x;
const int row =
    blockIdx.y * blockDim.y + threadIdx.y;

if (row < height && col < width) {
  int index = row * width + col;
  c[index] = a[index] + b[index];
}
```

## 3. Grid

```cpp
dim3 block(16, 8);
dim3 grid(
    (width + block.x - 1) / block.x,
    (height + block.y - 1) / block.y);
```

Block 是 `16 列 x 8 行`，共 128 thread。

## 4. 数字推演

```text
width=37, height=23
block=(16,8)

grid.x=ceil(37/16)=3
grid.y=ceil(23/8)=3
```

总覆盖：

```text
48 列 x 24 行
```

多出的 thread 由二维边界判断排除。

## 5. 一个 Thread

```text
blockIdx=(2,1)
threadIdx=(4,6)

col=2*16+4=36
row=1*8+6=14
index=14*37+36=554
```

它处理最后一列中的一个元素。

若 `threadIdx.x=5`，则 `col=37` 越界。

## 6. Warp 形状

为了组成 warp，二维 thread 先按 `x` 最先变化的顺序排成一维：

```text
linear = y * blockDim.x + x
```

`block=(16,8)` 中一个 warp 横跨两行：

```text
warp 0: y=0,x=0..15 + y=1,x=0..15
```

`block=(32,8)` 中一个 warp 正好一行。形状影响访存和边界，但没有永久固定的
最佳形状。

注意这里的 `linear` 是 block 内 thread 编号，不是前面
`index = row * width + col` 的矩阵地址。前者用来判断 thread 属于哪个 warp，
后者用来访问矩阵数据。详细图解见：

[一维、二维、三维 Block 与线性编号](../volume01_gpu_basics/02A_一维二维三维Block与线性编号.md)。

## 7. Sample

```bash
make -C labs/02_programming_model/matrix_add_2d clean all
./labs/02_programming_model/matrix_add_2d/matrix_add_2d
./labs/02_programming_model/matrix_add_2d/matrix_add_2d 10 7
```

## 8. 故障实验

下面每个改动都对应一类真实 bug。重点不是"跑一下"，而是**先预测症状再验证**，
把错误和现象的因果对上：

1. **去掉 row 边界，只保留 col。** 当 `height` 不是 `block.y` 的整数倍时，
   grid 在纵向会多出几行 thread（§4 里 `grid.y` 覆盖到 24 行 > `height=23`）。
   这些 `row >= height` 的 thread 算出的 `index = row*width+col` 落到数组末尾
   之外，发生**越界写**。症状：偶发结果错误或 `compute-sanitizer` 报 invalid write，
   而且**只在非整除尺寸下暴露**——这正是下面第 4 条要警惕的"方阵掩盖"。
2. **将地址写成 `col * width + row`。** 这是行主序/列主序写反。在方阵
   （`width == height`）下它**可能侥幸只是转置了结果**而不越界，骗过你；一旦
   `width != height`，`col*width+row` 的取值范围就和真实矩阵不符，越界 + 算错。
3. **Grid 使用向下取整**（`width/block.x` 而非 `ceil`）。最右边和最下边不满
   一个 block 的元素**根本没有 thread 覆盖**，结果里这些位置是未初始化的垃圾值。
   症状：边缘行/列结果错，中间全对——典型的"缺了 ceil"指纹。
4. **使用 `block=(8,8)`、`(16,8)`、`(32,8)` 对比。** 结果都应正确（边界判断兜底），
   但 warp 形状不同（§6）：`(32,8)` 一个 warp 正好一行，访存最规整。这一条是为
   卷三的合并访问埋伏笔——形状不影响正确性，却影响性能。

> 务必用非整除尺寸（如 `10 7`）跑这些实验。方阵 + 整除尺寸会同时掩盖第 1、2、3
> 条的越界，让错误代码"看起来是对的"，这是初学者最容易踩的陷阱。

## 9. 面试题

- 为什么 `x` 通常对应列？
- 二维 block 如何线性化为 warp？
- 为什么边界需要同时检查 row 和 col？
- Block 为长方形是否合法？
