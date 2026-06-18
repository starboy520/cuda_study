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

### 5.0 先认识三个前置概念(否则会懵)

要听懂 pinned memory,先得知道三个底层概念。它们平时写代码用不到,但理解了 pinned 就通了。

**① 虚拟内存与分页(paging)**

你 `malloc` 拿到的指针是**虚拟地址**,不是数据在物理内存条上的真实位置。操作系统用"**分页**"
管理内存——把内存切成一页页(通常 4KB),并且**有权随时**:

```text
- 把暂时不用的页"换出"到磁盘(swap),腾出物理内存
- 在物理内存里"挪动"页的位置(整理碎片)
你的虚拟地址不变,但它背后的物理位置,OS 说了算、随时会变。
```

**② DMA(直接内存访问)**

GPU 从 host 内存搬数据,不靠 CPU 一字节字节地拷,而是用一个专门的硬件——**DMA 引擎**。它能
在 CPU 不参与的情况下,自己把数据从内存搬到 GPU。但 DMA 有个限制:

```text
DMA 直接按【物理地址】搬数据。
它要求:在搬运的整个过程中,那块内存的物理位置【不能变】。
```

**③ 矛盾来了**

把 ① 和 ② 放一起,矛盾就出现了:

```text
普通 malloc 的内存(pageable,可分页):OS 随时可能换出/挪动它
DMA:                                 要求物理位置在传输期间固定不动
→ DMA 不敢直接搬 pageable 内存,万一搬到一半 OS 把页挪走了,就搬错地方了!
```

**pinned memory 就是解决这个矛盾的**——下面就清楚了。

### 5.1 pinned memory 是什么

```cpp
float* host = nullptr;
CUDA_CHECK(cudaMallocHost(&host, bytes));   // 分配 pinned(页锁定)内存
// ... 用完后
CUDA_CHECK(cudaFreeHost(host));
```

**pinned(页锁定,page-locked)内存 = 被"钉死"在物理内存里、OS 保证不换出也不挪动的内存。**
既然它物理位置固定,DMA 就敢直接搬它了。

> 名字理解:"pin" 就是"用图钉钉住"——把这块内存钉在物理内存的固定位置,OS 不许动它。

### 5.2 为什么 pinned 更快:省掉一次暗中拷贝

知道了上面的矛盾,就能理解性能差异。当你对**普通 pageable 内存**做 `cudaMemcpy` 时,因为 DMA
不敢直接搬它,驱动其实**偷偷多走一步**:

```text
pageable 传输(慢):
  你的数据(pageable) --(CPU 先拷一份)--> 驱动内部的 pinned 暂存区 --(DMA)--> GPU
                      ^^^^^^^^^^^^^^^^ 这次额外的 CPU 拷贝,就是开销来源
  (驱动先把数据拷到一块它自己的 pinned 暂存区,再让 DMA 从那搬——绕了一道)
```

而你直接用 `cudaMallocHost` 分配的 pinned 内存,DMA 能**直接**从它搬,省掉中间那次暂存拷贝:

```text
pinned 传输(快):
  你的数据(pinned) --(DMA 直接搬)--> GPU
  (没有额外 CPU 拷贝)
```

所以 pinned 传输更快,本质是**省掉了 pageable 必须的那次"暗中 staging 拷贝"**。

### 5.3 更关键:pinned 是异步重叠的前提

pinned 不只是"快一点",它还是 `cudaMemcpyAsync` 真正异步的**必要条件**(卷七的重叠优化全靠它):

```text
cudaMemcpyAsync 想做的是:让 DMA 引擎独立搬数据,同时 CPU/GPU 去干别的 → 重叠
而 DMA 独立工作的前提是:内存物理位置固定 = pinned

所以:
  pinned + cudaMemcpyAsync  → DMA 真异步搬,能和计算重叠(卷七/02)
  pageable + cudaMemcpyAsync → DMA 没法直接搬,退化成【同步】行为,重叠失效!
```

> 这是个高频坑:用 `cudaMemcpyAsync` 但忘了用 pinned 内存,以为在重叠,其实退化成同步,白忙。

### 5.4 完整例子

```cpp
const size_t bytes = N * sizeof(float);
float* h_pinned = nullptr;
CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));     // pinned host 内存
// 填数据
for (int i = 0; i < N; ++i) h_pinned[i] = i;

float* d_data = nullptr;
CUDA_CHECK(cudaMalloc(&d_data, bytes));

cudaStream_t stream;
CUDA_CHECK(cudaStreamCreate(&stream));

// 因为是 pinned,这个 async 拷贝能真异步、和后续计算重叠
CUDA_CHECK(cudaMemcpyAsync(d_data, h_pinned, bytes,
                           cudaMemcpyHostToDevice, stream));
myKernel<<<grid, block, 0, stream>>>(d_data);      // 可与上面的传输重叠

CUDA_CHECK(cudaStreamSynchronize(stream));
CUDA_CHECK(cudaFreeHost(h_pinned));                // pinned 要用 cudaFreeHost 释放
CUDA_CHECK(cudaFree(d_data));
CUDA_CHECK(cudaStreamDestroy(stream));
```

注意:pinned 内存必须用 `cudaFreeHost` 释放,**不能用普通 `free`**。

### 5.5 优点与代价

```text
优点:
  ✓ 传输更快(省掉 pageable 的暗中 staging 拷贝)
  ✓ 是 cudaMemcpyAsync 真异步、传输与计算重叠的前提(卷七)

代价(不能无脑全用 pinned):
  ✗ 占用不可分页的物理内存,挤压 OS 和其他程序的可用内存
  ✗ 过量使用会拖慢甚至卡死整个系统(OS 能调度的内存变少)
  ✗ 分配/释放比 malloc 慢 → 要【复用】缓冲区,别在循环里反复 malloc/free
```

> 一句话总结 pinned:**用"把内存钉死在物理位置"换取"DMA 能直接搬"**,从而传输更快、且能异步
> 重叠。代价是占用宝贵的不可换出内存,所以只给真正需要高速传输的缓冲区用,并复用它。

### 5.6 进阶:`cudaHostAlloc` 的标志位

`cudaMallocHost` 是简化版。更灵活的是 `cudaHostAlloc`,可带标志:

```cpp
cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);   // 等价于 cudaMallocHost
cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);    // 同时可映射到 GPU 地址空间(zero-copy,见 §6)
cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable);  // 对所有 CUDA context 都是 pinned
```

初学用 `cudaMallocHost` 即可;需要 zero-copy(§6)时才用 `cudaHostAllocMapped`。

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

