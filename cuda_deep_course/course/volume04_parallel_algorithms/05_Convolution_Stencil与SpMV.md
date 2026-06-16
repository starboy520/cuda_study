# 05 Convolution、Stencil 与 SpMV

## 1. 共同点

这些算法都不是简单“一输入对应一输出”：

- Convolution：输出依赖邻域和 filter。
- Stencil：网格点依赖周围点。
- SpMV：一行输出依赖稀疏非零元素。

核心问题是数据复用、边界和负载均衡。

## 2. 1D Convolution

```text
output[i] =
  sum(filter[k] * input[i + k - radius])
```

相邻 output 使用大量重叠 input。Naive 版本反复从 global memory 读取。

把"重叠"量化，就知道 shared tile 为什么值得做。设 filter 宽度
`F = 2*radius + 1`：每个 `output[i]` 要读 `F` 个 input；而相邻的
`output[i]` 和 `output[i+1]` 的输入窗口**重叠了 `F-1` 个元素**。于是 naive
版本里，**几乎每个 input 元素都被它附近的 `F` 个输出各读一遍**：

```text
naive global 读总量 ≈ N * F      （N 个输出，每个读 F 个输入）
理想（每元素读一次）≈ N
浪费倍数 ≈ F
```

`radius=3` 时 `F=7`，naive 把每个输入从 global 读了约 7 遍。Shared tile 的
做法是让一个 block **协作把"中心数据 + 左右 halo"读进 shared 一次**，块内所有
输出再从 shared 复用：

```text
中心数据 + 左右 halo
```

一个 block 协作加载后，多次复用，把那 `F` 倍的重复 global 读压成接近 1 倍，
剩下的复用全发生在快得多的片上 shared memory 里。`F` 越大（filter 越宽），
shared tile 的收益越明显。

## 3. 2D Convolution

Tile 需要四周 halo。加载策略包括：

- 每 thread 加载中心，再由部分 thread 加载 halo。
- 每 thread 循环加载多个元素。
- 直接让 cache 服务 halo。

Filter 小且只读，可考虑 constant memory。

## 4. Stencil

2D/3D stencil 常见于 PDE、流体和热传导。

时间迭代中通常使用 ping-pong buffer：

```text
input -> output
swap(input, output)
```

不能原地更新，否则邻居可能读到当前轮已更新值。

3D stencil 面临更大工作集和 halo，常使用平面缓存、register queue 或多级 tile。

## 5. 边界条件

常见：

- Zero padding。
- Clamp。
- Periodic。
- Mirror。
- 单独 boundary kernel。

边界策略是算法规格，不是随便加一个 `if`。

## 6. SpMV

CSR：

```text
rowOffsets
columnIndices
values
```

每行：

```text
y[row] = sum(values[j] * x[columnIndices[j]])
```

难点：

- 每行非零数量不同，负载不均衡。
- `x` 访问间接且不规则。
- 行很短时 thread 利用率低，行很长时单 thread 太慢。

## 7. SpMV 映射策略

- 每 thread 一行：简单，适合较均匀短行。
- 每 warp 一行：warp 协作归约，适合较长行。
- Merge/path 或 load-balanced 策略：处理高度不均匀矩阵。

不存在对所有稀疏矩阵最佳的固定 kernel。

## 8. 实践

### Convolution

实现 naive 与 shared 1D convolution，filter radius 为 3，比较：

- 正确性。
- Global load 数量直觉。
- 边界成本。

### Stencil

实现非方阵 2D 5-point stencil，使用 ping-pong buffer 迭代 100 次。

### SpMV

生成两种 CSR：

```text
每行固定 16 个非零
每行非零数高度不均匀
```

比较 thread-per-row 与 warp-per-row。

## 9. 资料映射

- PMPP：Convolution、Stencil、Sparse Matrix Computation。
- cuSPARSE 文档：工程基线。

