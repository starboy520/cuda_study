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

### 6.1 它是什么：让 GPU"隔着 PCIe"直接读写 host 内存

前面 §5 的 pinned 内存是"放在 host、用 DMA 整块搬到 GPU 再算"。**Zero-copy（零拷贝）**
更进一步：把一块 pinned host 内存**映射**进 GPU 的地址空间，于是 kernel 可以**直接**用
一个指针访问它，**完全不发生显式 `cudaMemcpy`**——"zero copy"由此得名。

```text
普通路径：  host 数据 --cudaMemcpy--> device 显存 --> kernel 读显存
zero-copy： host 数据 <==== kernel 隔着 PCIe 直接读/写 ====   （没有 memcpy 这一步）
```

注意"没有拷贝"不等于"没有传输"：离散 GPU 上，kernel 每次访问这块内存，**数据仍要实时
走 PCIe**。只是把"一次性整块搬运"换成了"按需、细粒度地隔空访问"。

### 6.2 怎么写

两步：分配可映射的 pinned 内存，再拿到它在 device 侧的指针。

```cpp
// 1) 分配 pinned 且可映射的 host 内存
float* hostPtr = nullptr;
CUDA_CHECK(cudaHostAlloc(&hostPtr, bytes, cudaHostAllocMapped));

// 2) 取得同一块内存在 device 地址空间的指针
float* devPtr = nullptr;
CUDA_CHECK(cudaHostGetDevicePointer(&devPtr, hostPtr, 0));

// 3) kernel 用 devPtr 直接访问，无需 cudaMemcpy
kernel<<<grid, block>>>(devPtr);
CUDA_CHECK(cudaDeviceSynchronize());
// CPU 仍用 hostPtr 访问同一块数据
```

要点：

- 必须用 `cudaHostAlloc(..., cudaHostAllocMapped)`（或对已 pinned 的内存用
  `cudaHostRegister`），普通 `malloc` 不能映射。
- 在统一虚拟地址（UVA，64 位平台默认）下，`hostPtr` 和 `devPtr` 往往数值相同，但
  显式取一次 device 指针是最稳妥的写法。
- kernel 结束要正常同步；CPU 端读结果前确保 kernel 已完成（zero-copy 不豁免同步）。

### 6.3 适合用的场景

```text
✅ 数据只被访问一次（一次性流式读入，省掉"先 memcpy 整块"的成本和显存占用）
✅ 集成 GPU / 共享物理内存（host 和 device 本就是同一片内存，无 PCIe 往返）
✅ 极少量、零散的数据（如一个标志位、少数标量），整块 memcpy 不划算
✅ kernel 计算足以掩盖访问延迟，或访问可与计算重叠
```

### 6.4 不适合的场景（离散 GPU 的大坑）

```text
❌ 离散 GPU 上被反复 / 随机访问的大数组
   每次访问都隔着 PCIe，带宽只有显存的几十分之一（PCIe ~16 GB/s vs 显存 ~320 GB/s）
❌ 高带宽计算热点
   把热点数据放 zero-copy，等于让 SM 一直等 PCIe，吞吐崩塌
```

一句话判据：**数据要被多次复用，就别用 zero-copy，老老实实 memcpy 进显存**；只有
"访问一次就丢"或"集成 GPU 无往返"时，省掉拷贝才真正划算。

### 6.5 和另外两种 host 内存的关系

| 方式 | 数据放哪 | 怎么到 GPU | 典型用途 |
|---|---|---|---|
| pinned（§5）| host，page-locked | DMA 整块搬进显存再算 | 大块传输、异步重叠的基础 |
| **zero-copy / mapped** | host，映射进 GPU 地址空间 | kernel **隔 PCIe 直接访问**，无显式 memcpy | 一次性/零散数据、集成 GPU |
| Unified Memory（§7）| 自动迁移 | 按需**页迁移**到访问方 | 编程方便，靠 prefetch/advise 调性能 |

三者都建立在 pinned 之上，区别在"数据何时、以什么粒度跨过 PCIe"。zero-copy 是
"细粒度、按需、隔空访问"；pinned memcpy 是"粗粒度、一次性、搬过去再算"；Unified Memory
是"运行时替你决定迁移"。

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

