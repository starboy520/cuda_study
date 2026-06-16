# 01 CUDA 内存空间

## 1. 先问三个问题

看到一个变量时，先问：

1. 它由谁拥有？
2. 谁能读取或修改它？
3. 它活多久？

“快或慢”不是唯一维度，范围和生命周期同样决定正确性。

## 2. Register

寄存器主要保存 thread 私有的局部值：

```cpp
float sum = 0.0F;
int index = blockIdx.x * blockDim.x + threadIdx.x;
```

特点：

```text
共享范围：单个 thread
生命周期：thread 执行期间
分配方式：主要由编译器决定
```

寄存器不足时，编译器可能把值放到 local memory，这叫 register spilling。

## 3. Local Memory

名字容易误导：local memory 对 thread 是私有的，但物理存储通常位于 device
memory，并经过 cache。

可能进入 local memory 的情况：

- 寄存器压力太高。
- 无法静态索引的局部数组。
- 较大的 thread 私有对象。

不能仅看源代码判断是否发生 spill。可用：

```bash
nvcc -Xptxas=-v ...
```

查看寄存器和 spill 报告。

## 4. Shared Memory

```cpp
__shared__ float tile[32][33];
```

特点：

```text
共享范围：同一 block
生命周期：block
位置：片上
用途：复用、重排、协作
```

静态 shared memory 尺寸编译时确定；动态 shared memory 由 launch 指定：

```cpp
extern __shared__ float buffer[];
kernel<<<grid, block, bytes>>>(...);
```

## 5. Global Memory

`cudaMalloc` 分配的主要设备内存：

```cpp
float* data = nullptr;
CUDA_CHECK(cudaMalloc(&data, bytes));
```

特点：

```text
访问范围：device 上的 thread
生命周期：从分配到释放
容量大、延迟高、总带宽高
```

Global memory 性能高度依赖 warp 的地址模式和数据复用。

## 6. Constant Memory

```cpp
__constant__ float coefficients[64];
```

Host 通常通过：

```cpp
cudaMemcpyToSymbol(coefficients, source, bytes);
```

写入。Device 侧只读。

当一个 warp 的 thread 读取相同地址时，constant cache 可以高效广播；若每个
thread 读取大量不同地址，优势会下降。

适合小型、只读、许多 thread 重复使用的参数。

## 7. Texture 与 Read-Only 数据路径

Texture object 提供专门的读取语义、寻址方式和缓存局部性支持，常见于图像
和具有空间局部性的访问。

现代 GPU 普通 global load 也有 cache，因此不要因为“texture cache”这个名字
就默认 texture 一定更快。是否使用取决于：

- 寻址和采样功能。
- 访问局部性。
- 数据格式。
- 实测。

## 8. Host Memory

### Pageable memory

普通 `malloc`、`new`、`std::vector` 通常使用 pageable host memory。

### Pinned memory

```cpp
cudaMallocHost(&pointer, bytes);
```

页锁定内存适合高吞吐异步传输，但属于有限系统资源，不应无限分配。

### Mapped memory

某些系统可让 GPU 直接访问映射的 host memory。它避免显式复制，但访问仍跨
互连，不能当作 GPU 本地 DRAM。

## 9. Managed Memory

```cpp
cudaMallocManaged(&pointer, bytes);
```

Unified Memory 提供统一地址和系统管理的数据迁移机制。它改善可编程性，但
不代表数据无成本地同时存在于所有处理器旁边。

性能仍取决于：

- 数据当前驻留位置。
- page fault 和迁移。
- 访问模式。
- prefetch 和 advice。

## 10. 选择内存空间

```text
thread 私有小值        -> register
block 内协作与复用     -> shared
大规模输入输出         -> global
小型只读广播参数       -> constant
图像/特殊寻址读取      -> texture object
高吞吐异步 Host 传输   -> pinned host memory
简化复杂数据管理       -> managed memory，仍需优化驻留
```

## 11. 实践

1. 编译一个 kernel，加入较大局部数组，使用 `-Xptxas=-v` 观察 spill。
2. 查询设备的 shared memory、constant memory 和 L2 容量。
3. 解释为什么 local memory 不是“每个 thread 旁边的小型片上内存”。

## 12. 资料映射

- CUDA Programming Guide：Programming Model、Writing SIMT Kernels、Unified Memory。
- CUDA C++ Best Practices Guide：Memory Optimizations。

