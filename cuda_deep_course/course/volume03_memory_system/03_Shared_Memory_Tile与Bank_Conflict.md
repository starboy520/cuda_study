# 03 Shared Memory、Tile 与 Bank Conflict

## 1. Shared Memory 的三种价值

1. **复用**：一次 global load 被多个计算使用。
2. **重排**：以合并方式读入，再按另一方向读取。
3. **协作**：block 内 thread 交换部分结果。

如果没有这三种价值之一，使用 shared memory 未必有益。

## 2. Tile

Tile 是算法处理的数据小块，不等同于 block 形状。

矩阵乘：

```text
block thread 加载 A/B 子块
-> 放入 shared memory
-> 每个 thread 多次复用
```

转置：

```text
按行合并读 input
-> 放入 tile
-> 交换 tile 下标
-> 按行合并写 output
```

## 3. 必须同步的原因

```cpp
tile[threadIdx.y][threadIdx.x] = input[index];
__syncthreads();
float value = tile[threadIdx.x][threadIdx.y];
```

写入和读取可能由不同 thread 完成。没有 barrier，读取者可能先到。

`__syncthreads()`：

- 等待 block 内参与的 thread 到达。
- 为相应 shared/global memory 操作提供 block 范围的顺序与可见性保证。

## 4. Barrier 必须一致到达

危险写法：

```cpp
if (valid) {
  __syncthreads();
}
```

若同一 block 只有部分 thread 满足 `valid`，可能死锁或产生未定义行为。

正确模式通常是：

```cpp
if (valid) {
  tile[...] = input[...];
}
__syncthreads();
if (outputValid) {
  output[...] = tile[...];
}
```

## 5. Bank

Shared memory 被划分成多个 bank。一个 warp 同时访问时：

- 不同 bank 可并行服务。
- 多个 lane 访问同一 bank 的不同地址会形成 bank conflict。
- 多个 lane 读取同一地址可使用广播语义。

对常见 4-byte 元素，可用第一层直觉：

```text
bank = wordIndex % 32
```

具体行为仍应以目标架构文档和 profiler 为准。

## 6. 为什么转置会冲突

声明：

```cpp
__shared__ float tile[32][32];
```

行主序地址：

```text
tile[row][col] -> row * 32 + col
```

一个 warp 固定 `col`、变化 `row` 读取列时：

```text
wordIndex = row * 32 + col
bank = col
```

32 个 lane 都落到同一 bank 的不同地址，形成严重冲突。

"严重"是可以量化的。shared memory 的硬件规则是：**同一 bank 上若有 N 个不同
地址被同时请求，硬件必须把它们拆成 N 次串行访问**（同地址广播不算冲突）。这里
32 条 lane 全压在同一个 bank 的 32 个不同地址上，于是本该一拍完成的访问被拆成
**32 拍**——这次 shared 读取直接慢 32 倍，这是 32-way bank conflict 的最坏情形。
无冲突时 32 条 lane 命中 32 个不同 bank，一拍并行完成。所以消除冲突的收益不是
"略快一点"，而是把一个 32 倍的串行惩罚降回 1 倍。

## 7. Padding

改成：

```cpp
__shared__ float tile[32][33];
```

列读取：

```text
wordIndex = row * 33 + col
bank = (row + col) % 32
```

相邻 row 映射到不同 bank，从而消除这类固定步长冲突。

多出的一列不保存逻辑矩阵数据，只改变物理行跨度。

## 8. 动态 Shared Memory

```cpp
extern __shared__ float buffer[];
kernel<<<grid, block, bytes>>>(...);
```

适合运行时决定 tile 或多个共享数组。常见做法：

```cpp
float* first = buffer;
float* second = buffer + firstCount;
```

需要自行处理大小、对齐和越界。

## 9. Shared Memory 与 Occupancy

一个 block 使用更多 shared memory，可能减少每个 SM 同时驻留的 block。

因此：

```text
更大 tile -> 可能更多复用
更大 tile -> 也可能降低并发
```

最终由性能数据决定。

## 10. 实践

在 transpose lab 中：

1. 将 `[32][33]` 改为 `[32][32]`。
2. 验证结果仍正确。
3. 用 Nsight Compute 比较 shared-memory bank conflict 指标和时间。
4. 恢复 padding。

## 11. 资料映射

- Best Practices Guide：Shared Memory and Memory Banks。
- Programming Guide：Shared Memory、Synchronization Primitives。

