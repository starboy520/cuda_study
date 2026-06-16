# 02 合并访问、对齐、AoS 与 SoA

## 1. 分析单位是 Warp

Global memory 访问不能只看单个 thread。要同时列出一个 warp 的地址。

连续访问：

```cpp
value = input[base + threadIdx.x];
```

若 `float` 为 4 bytes，32 个 lane 请求连续 128 bytes。对 Compute Capability
6.0 及更高设备，可把合并访问直觉理解为：

> 硬件发起足够数量的 32-byte 事务，覆盖 warp 请求的全部地址。

连续且对齐良好时，128 bytes 通常由 4 个 32-byte 区间覆盖。

## 2. 跨步访问

```cpp
value = input[threadIdx.x * stride];
```

当 `stride=32`，相邻 lane 相隔 128 bytes。每个 lane 可能落到不同事务区间，
实际搬运远多于程序需要的 128 bytes。

把它算成具体数字。一个 warp 32 条 lane，每条只要 4 bytes，程序总共需要
`32 * 4 = 128` bytes。但内存系统的最小搬运单位是 32-byte sector：

```text
连续访问(stride=1)：32 个 4B 落在 128B 内 = 4 个 sector，搬 128B
                    利用率 = 128 / 128 = 100%

跨步访问(stride=32)：每条 lane 的 4B 独占一个 32B sector
                    需要 32 个 sector，搬 32 * 32 = 1024B
                    利用率 = 128 / 1024 ≈ 12.5%
```

也就是说，跨步把有效带宽利用率从 100% 砸到约 1/8，硬件搬了 8 倍的数据，其中
7/8 是白搬的。stride 越大（直到超过 sector 跨度），浪费越接近这个上限。这就是
"合并访问"四个字背后真正的代价。

有效利用率可以粗略理解为：

```text
线程真正请求的数据
------------------
内存事务搬运的数据
```

Cache 可能改变 DRAM 实测，但不会让糟糕布局成为好布局。

## 3. 对齐

即使地址连续，起点跨越事务边界也可能需要额外事务。

`cudaMalloc` 返回的基础地址具有足够对齐，但数组切片或偏移可能破坏起点对齐：

```cpp
float* shifted = data + 1;
```

是否产生明显损失取决于 cache 复用和架构，必须测量。

## 4. 行主序二维访问

行主序：

```cpp
matrix[row * width + col]
```

若 warp 中：

```text
row 相同
col 连续
```

地址连续。

因此默认约定：

```text
threadIdx.x = col
threadIdx.y = row
```

通常有利于行方向访问。

## 5. AoS 与 SoA

Array of Structures：

```cpp
struct Particle {
  float x, y, z, mass;
};
Particle particles[count];
```

如果 kernel 只读取 `x`，相邻 thread 的 `x` 间隔 16 bytes。

Structure of Arrays：

```cpp
struct Particles {
  float* x;
  float* y;
  float* z;
  float* mass;
};
```

只读取 `x` 时地址连续。

选择不是绝对的：

- 若每个 thread 总是使用完整结构，AoS 可能方便且合理。
- 若 kernel 分阶段只使用部分字段，SoA 往往提高事务利用率。
- 可使用 AoSoA 在向量化、局部性和工程接口间折中。

## 6. 向量化访问

```cpp
float4 value = reinterpret_cast<const float4*>(input)[index];
```

可能减少指令数量并表达对齐访问，但必须保证：

- 地址满足类型对齐。
- 元素数量和尾部正确处理。
- 实际瓶颈值得优化。

向量化不会修复错误的 warp 数据布局。

## 7. Pitch

二维分配可使用：

```cpp
cudaMallocPitch(&pointer, &pitchBytes,
                width * sizeof(float), height);
```

每一行起点按 pitch 前进：

```cpp
char* rowAddress =
    reinterpret_cast<char*>(pointer) + row * pitchBytes;
```

Pitch 以 bytes 为单位，通常不等于逻辑 `width * sizeof(T)`。

## 8. 实验

```bash
make -C labs/03_memory_system/memory_access clean all
./labs/03_memory_system/memory_access/memory_access
```

实验比较：

```text
copyContiguous
copyStrided(stride=32)
```

输出的 `logical GB/s` 只按程序请求的 `float` 读写计数。跨步版本实际事务可能
搬运更多数据，因此不能把 logical GB/s 当作真实 DRAM 字节数。

使用 Nsight Compute：

```bash
ncu --set full \
  --kernel-name regex:copyContiguous \
  ./labs/03_memory_system/memory_access/memory_access

ncu --set full \
  --kernel-name regex:copyStrided \
  ./labs/03_memory_system/memory_access/memory_access
```

观察内存吞吐、sector/transaction 和 source counters 建议。

## 9. 面试推演

给出 32 个 lane 的地址：

```text
0, 4, 8, ..., 124 bytes
```

问：覆盖几个 32-byte 区间？

再给出：

```text
0, 128, 256, ..., 3968 bytes
```

问：为什么请求仍是 128 bytes，却可能产生大量事务？

## 10. 资料映射

- Best Practices Guide：Coalesced Access to Global Memory。
- Programming Guide：Global Memory 与多维分配。

