# 04 Cache、Host 传输与 Unified Memory

## 1. Cache 解决什么问题

Cache 利用：

```text
时间局部性：刚访问的数据可能再次访问
空间局部性：附近地址可能很快访问
```

GPU 常见层次包括每个 SM 附近的 L1/共享资源，以及整个 GPU 共享的 L2。
具体组合和容量随架构变化。

## 2. 不要用 Cache 掩盖糟糕访问

跨步访问有时由于相邻 warp 重用 cache line，实测没有理论事务浪费那么严重。
这不代表布局合理：

- Cache 容量有限。
- 其他数据会竞争。
- 不同规模下结果可能改变。
- 写访问和读取行为不同。

先设计合并访问，再把 cache 当作额外收益。

## 3. 数据复用

重用方式：

- 同一 thread 在 register 中复用。
- block 内在 shared memory 中复用。
- 跨 block 依赖 L2 或重新读取。

优先使用范围最小、生命周期明确的存储，但要考虑资源压力。

## 4. Host-Device 传输

传输优化优先级通常是：

1. 减少传输次数。
2. 减少传输总量。
3. 合并小传输。
4. 使用 pinned memory 提升可异步传输能力。
5. 使用 stream 重叠传输与计算。

不要先优化一个 10 微秒 kernel，却忽略每次 5 毫秒的数据往返。

## 5. Pinned Memory

```cpp
float* host = nullptr;
CUDA_CHECK(cudaMallocHost(&host, bytes));
```

要理解 pinned 的价值，先得知道普通 `malloc`/`std::vector` 拿到的是
**pageable（可分页）内存**——操作系统可以随时把这些页换出到磁盘、或在物理内存
里挪位置。而 GPU 的 DMA 引擎是直接按**物理地址**搬数据的，它无法容忍页在传输
途中被 OS 移走。

于是当你对 pageable 内存做 `cudaMemcpy` 时，驱动其实是**偷偷多走一步**：

```text
pageable 传输：
  你的数据 --(CPU 拷贝)--> 驱动内部的 pinned 暂存区 --(DMA)--> GPU
            ^^^^^^^^^^^ 这次额外的 CPU 拷贝就是开销来源
```

`cudaMallocHost` 直接分配**页锁定（pinned）**内存，OS 保证它不被换出、不被移动，
DMA 可以**直接**从它搬到 GPU，省掉中间那次暂存拷贝：

```text
pinned 传输：
  你的数据 --(DMA)--> GPU
```

这也解释了为什么真正的 `cudaMemcpyAsync` **必须**用 pinned 内存才能和计算重叠：
异步传输靠 DMA 引擎独立工作，而 DMA 只能作用在不会被 OS 挪动的 pinned 页上。
pageable 的 "async" 拷贝会退化成同步行为。

优点：

- 避免运行时为 DMA 临时 staging（省掉上面那次 CPU 拷贝）。
- 是真正 `cudaMemcpyAsync` 与传输/计算重叠的前提条件。

代价：

- 占用不可分页系统内存，挤压 OS 可用内存。
- 过量使用影响整个系统稳定性。
- 分配成本较高，应**复用**缓冲区而非反复 `cudaMallocHost`/`cudaFreeHost`。

## 6. Mapped / Zero-Copy

GPU 直接访问映射 host memory，适合特定场景：

- 数据只访问一次。
- 集成 GPU 或共享物理内存系统。
- 避免显式复制比本地 DRAM 带宽更重要。

离散 GPU 通过 PCIe 频繁随机读取 mapped memory 通常不适合高带宽计算。

## 7. Unified Memory

Unified Memory 解决的是地址和数据管理问题，不是取消物理内存层次。

最简单写法：

```cpp
float* data = nullptr;
CUDA_CHECK(cudaMallocManaged(&data, bytes));
```

它的方便之处在于 CPU 和 GPU 用**同一个指针**，不必手写 `cudaMemcpy`。但数据
并没有魔法般同时存在于两边——底层靠**按需页迁移（on-demand page migration）**：
谁访问、页就迁到谁那边。这带来一个隐蔽的性能陷阱。看这段代码：

```cpp
for (int i = 0; i < iters; ++i) {
  cpuTouch(data);   // 页被迁回 CPU
  gpuKernel(data);  // 页又被迁到 GPU
}
```

每一轮循环，同一批页都要**跨 PCIe 来回搬两次**。如果工作集是 1 GB、PCIe 约
16 GB/s，一次单向迁移就要约 60 ms——CPU/GPU 反复争用时，这些 page fault 和迁移
开销会**远超 kernel 本身**，让"统一内存"看起来比手动 `cudaMemcpy` 还慢。

所以用 managed memory 做性能分析必须问：

- 页面当前驻留在哪里？（CPU 还是 GPU）
- CPU 和 GPU 是否在来回争用同一批页（ping-pong）？
- 是否在热路径上触发了按需迁移和 page fault？
- 能否用 `cudaMemPrefetchAsync` 在 kernel 前把页**预迁**到 GPU，消除首次 fault？
- 是否适合用 `cudaMemAdvise` 标注访问偏好（如只读、首选驻留位置）？

一句话：Unified Memory 提升的是**可编程性**，性能仍要靠 prefetch/advice 主动管理
驻留，否则迁移开销会吃掉收益。

## 8. L2 Persisting Access

某些 CUDA 功能允许为重复访问区域配置 L2 access policy window。它适合明确的
热点数据，但不是“把整个工作集锁进 L2”。

使用前必须：

- 确认设备支持。
- 确认热点范围和命中比例。
- 用 profiler 验证。

## 9. 实践

1. 对 vector add 分别测量 kernel-only 和 H2D+kernel+D2H。
2. 使用 `cudaMallocHost` 重写 Host 缓冲区，比较大块传输。
3. 使用 `cudaMallocManaged` 实现同一功能，分别测试无 prefetch 与 prefetch。
4. 用 Nsight Systems 观察页面迁移或 memcpy 时间线。

## 10. 资料映射

- Programming Guide：Unified and System Memory、L2 Cache Control。
- Best Practices Guide：Data Transfer Between Host and Device。

