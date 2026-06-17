# 01 CUDA 内存空间

## 0. 先建立大局观：内存是一栋"楼"，越上层越快越小

CUDA 不像写普通 C++ 那样只有"一块内存"。GPU 上有**好几种内存空间**，它们速度、容量、
谁能访问、活多久都不同。用一栋楼类比，从快到慢、从小到大：

```text
register   ≈ 你手里的便签    最快(~1拍)、最小、只有你自己(thread)能看
shared     ≈ 工位小白板      很快、小、同组(block)共享
L1/L2cache ≈ 楼层公共缓存    自动管理，不直接编程
global     ≈ 地下大档案室    很大、很慢(几百拍)、所有线程可见
constant   ≈ 公告栏          小、只读、大家读同一条时一次广播
host memory≈ 隔壁楼(CPU)     要跨 PCIe 搬过来
```

写 CUDA 的一大半功夫，就是**决定每个数据该放哪一层**——放对了又快又省，放错了（比如把
反复用的数据每次都从 global 重新读）就慢几十倍。本章把每种内存讲清楚，最后给一张"什么
数据放哪"的决策表。

## 0.1 术语速查表（先扫一眼，下面逐个讲）

| 内存空间 | 谁能访问 | 活多久 | 速度 | 典型用途 |
|---|---|---|---|---|
| **register** | 单个 thread | thread 执行期 | 最快 | 局部标量、循环变量 |
| **local memory** | 单个 thread | thread 执行期 | 慢（其实在 global）| 寄存器装不下的私有数据 |
| **shared memory** | 同一 block | block 生命周期 | 很快（片上）| 复用/重排/协作 |
| **global memory** | 所有 thread | 分配到释放 | 慢、带宽高 | 大规模输入输出 |
| **constant memory** | 所有 thread（只读）| 程序期 | 同址广播快 | 小型只读参数 |
| **texture / read-only** | 所有 thread（只读）| 程序期 | 带专用 cache | 图像/空间局部性访问 |
| **host memory** | CPU（GPU 经映射/传输）| 看分配方式 | 跨 PCIe | CPU 侧数据、传输缓冲 |

> 易混点：**local memory 不在"片上"**，它名字带"local"但物理上在慢速 global memory 里
> （§3 详解）。这是新手最容易误解的一个。

## 1. 看到一个变量，先问三个问题

学内存空间不是背参数，而是养成一个习惯——看到任何数据，先问：

1. **它由谁拥有？**（一个 thread？一个 block？还是全局？）
2. **谁能读写它？**（决定正确性：别的线程能不能看到我的写入）
3. **它活多久？**（出了作用域/block 还在不在）

```text
"快或慢"只是一个维度。范围(谁能访问)和生命周期(活多久)同样关键——
它们直接决定正确性：把 block 私有的数据当全局共享，就会读到垃圾值。
```

这三个问题的答案，恰好对应下面每种内存空间的核心区别。

## 2. Register（寄存器）：thread 手里的便签

kernel 里普通的局部变量，默认就放在寄存器里：

```cpp
float sum = 0.0F;                              // 在寄存器
int index = blockIdx.x * blockDim.x + threadIdx.x;  // 在寄存器
```

```text
谁拥有：  单个 thread（别的线程看不到我的寄存器）
活多久：  thread 执行期间
谁分配：  编译器自动决定，你管不着具体哪个寄存器
速度：    最快，约 1 个时钟周期
```

**为什么寄存器有数量限制很重要？** 每个 SM 的寄存器总数是固定的，由所有驻留线程**瓜分**。
一个线程用得越多，能同时驻留的线程就越少（occupancy 下降，见卷五）。这就引出下一节的坑：
寄存器不够用时会发生什么。

## 3. Local Memory（"假"的本地内存）：名字骗人的慢内存

这是全章**最反直觉**的一点：local memory 听起来像"每个 thread 旁边的小型片上内存"，
**但它根本不在片上**——它物理上位于慢速的 device global memory，只是逻辑上对 thread 私有。

```text
名字暗示：  thread 旁边的快速本地存储   ❌ 错觉
实际情况：  在 global memory 里（慢），只是 thread 私有、经过 cache
```

**什么时候变量会"掉进"local memory？**（这通常是性能问题的信号）

```text
1. register spilling：寄存器不够用，编译器把溢出的值塞进 local memory
2. 无法静态索引的局部数组：如 arr[runtime_index]，编译器没法放进固定寄存器
3. 太大的 thread 私有对象：寄存器装不下
```

**怎么知道发生了 spill？** 不能只看源码猜，要让编译器报告（回忆卷二第 06 章 §7）：

```bash
nvcc -Xptxas=-v ...      # 输出里看 "spill stores/loads"，非 0 就是发生了 spill
```

> 一句话：**local memory = 披着"本地"外衣的 global memory**。看到 spill 数字非 0，通常意味
> 着该减少每线程的寄存器压力（精简局部变量、避免大的局部数组），否则这些访问会拖慢 kernel。

## 4. Shared Memory（block 的小白板）

```cpp
__shared__ float tile[32][33];
```

```text
谁拥有：  同一个 block 的所有 thread 共享
活多久：  block 的生命周期（block 结束就没了）
位置：    片上，很快
用途：    复用、重排、协作（卷三第 03 章详解）
```

声明方式有两种——大小编译期已知用**静态**，运行时才知道用**动态**：

```cpp
__shared__ float tile[32][33];              // 静态：大小写死在代码里
// 或
extern __shared__ float buffer[];           // 动态：大小在 launch 时给
kernel<<<grid, block, bytes>>>(...);        // 第三个参数 = shared 字节数
```

> 本章只点到为止。shared memory 是内存系统的重头戏，bank conflict、padding、tile 都在
> 卷三第 03 章专门讲。

## 5. Global Memory（地下大档案室）

`cudaMalloc` 分配的就是 global memory，是 GPU 上最大、也是 host 数据搬过来的落脚点：

```cpp
float* data = nullptr;
CUDA_CHECK(cudaMalloc(&data, bytes));
```

```text
谁拥有：  device 上所有 thread 都能访问
活多久：  从 cudaMalloc 到 cudaFree
特点：    容量大、单次延迟高（几百拍）、但总带宽很高
```

**关键认知**：global memory 慢，但它的总带宽其实很高——前提是**访问模式对**。同一个 kernel，
合并访问能跑满带宽，跨步访问可能只用到 1/8（卷三第 02 章）。所以：

```text
global memory 的性能不取决于"它本身多快"，而取决于：
  1. warp 的地址模式（合并 vs 跨步，§02）
  2. 数据复用（能不能搬进 shared 少读几次，§03）
```

> 一句话：**global 是必经之地（数据总要先到这），优化的核心是"少读、合并读、读了就复用"**。

## 6. Constant Memory（公告栏）：只读 + 同址广播

```cpp
__constant__ float coefficients[64];        // device 端只读
```

host 端用专门的 API 写入（device 端不能改）：

```cpp
cudaMemcpyToSymbol(coefficients, source, bytes);
```

**它的杀手锏是"广播"**：当一个 warp 的 32 条 lane 都读**同一个地址**时，constant cache
一次广播给全部 lane，极快。但反过来——

```text
✅ 32 条 lane 读同一地址（如所有线程用同一个系数）-> 一次广播，最优
❌ 32 条 lane 读各不相同的地址                     -> 退化成多次访问，优势消失
```

所以它**只适合**：小型、只读、且**很多线程同时读同一个值**的参数（如卷积核系数、变换矩阵）。

> 对比 §5：global 适合"每个线程读不同数据"，constant 适合"所有线程读同一份小数据"。选错
> 场景（让 constant 承担散地址访问）反而更慢。

## 7. Texture / Read-Only 数据路径

```text
texture object 提供：专用的只读 cache、特殊寻址（如归一化坐标）、采样/插值
常见于：图像处理、有空间局部性（相邻线程访问相邻像素）的访问
```

**别被名字误导**：现代 GPU 的普通 global load 也带 cache，所以"texture cache"这个名字
**不代表它一定更快**。是否值得用，取决于：

- 是否需要它的**寻址/采样**功能（如双线性插值、边界处理）。
- 访问是否有**空间局部性**（相邻线程读相邻数据）。
- 数据格式是否匹配。
- **实测**——别凭名字假设。

> 初学阶段：知道有这条"只读专用路径"、适合图像类访问即可。一般数值 kernel 用普通 global
> （配合合并访问）就够了。

## 8. Host Memory（CPU 那边的内存）

### Pageable memory（默认）

```cpp
float* p = (float*)malloc(bytes);    // 普通 malloc / new / std::vector
```

操作系统可随时把它换出/移动。GPU 的 DMA 不能直接搬，传输时驱动要先偷偷拷到 pinned 暂存区，
有额外开销。

### Pinned memory（页锁定）

```cpp
cudaMallocHost(&pointer, bytes);     // 页锁定，OS 保证不换出
```

DMA 可直接搬，传输更快，且是**异步重叠的前提**。代价：占不可分页系统内存，是有限资源，
不能无限分配。

### Mapped memory（映射 / zero-copy）

GPU 直接访问映射的 host memory，**省掉显式 cudaMemcpy**。但每次访问仍跨 PCIe，**不能**
当 GPU 本地 DRAM 用——只适合一次性/零散访问或集成 GPU（卷三第 04 章 §6 详解）。

## 9. Managed Memory（统一内存）：方便，但不是免费

```cpp
cudaMallocManaged(&pointer, bytes);  // CPU 和 GPU 用同一个指针
```

它的好处是**可编程性**——不用手写 `cudaMemcpy`，CPU/GPU 共用一个指针。但数据**不会魔法般
同时存在两边**，底层靠**按需页迁移**：谁访问，页就迁到谁那边。

```text
所以"方便"不等于"快"。性能仍取决于：
  - 数据当前驻留在哪（CPU 还是 GPU）
  - 是否频繁 page fault / 迁移（CPU/GPU 反复争用同一批页会很慢）
  - 访问模式
  - 是否用 cudaMemPrefetchAsync 预迁、cudaMemAdvise 标注偏好
```

> 一句话：managed memory 省的是**写代码的心智**，不是**运行时的成本**。性能要靠 prefetch /
> advise 主动管理驻留（卷三第 04 章 §7 详解）。

## 10. 选择内存空间：一张决策表

把全章压成"什么数据放哪"：

```text
thread 私有小值（标量/循环变量）  -> register
block 内协作 / 复用 / 重排        -> shared memory
大规模输入输出                    -> global memory
小型只读、很多线程读同一个值      -> constant memory
图像 / 需要特殊寻址采样的只读读取 -> texture object
高吞吐异步 Host↔Device 传输       -> pinned host memory
想省掉手动 memcpy（可编程性优先） -> managed memory（仍需 prefetch 调性能）
```

决策的核心永远是 §1 那三个问题：**谁拥有、谁访问、活多久**，再叠加"访问模式"决定快慢。

## 11. 实践

1. 编译一个加了**较大局部数组**的 kernel，用 `-Xptxas=-v` 观察 **spill**（验证 §3：数组
   掉进 local memory）。
2. 查询设备的 shared memory / constant memory / L2 容量（用 deviceQuery 或
   `cudaGetDeviceProperties`）。
3. 用自己的话解释：**为什么 local memory 不是"每个 thread 旁边的小型片上内存"？**（§3）

## 12. 资料映射

- CUDA Programming Guide：Programming Model、Writing SIMT Kernels、Unified Memory。
- CUDA C++ Best Practices Guide：Memory Optimizations。

## 13. 面试题（附参考答案）

**Q1：CUDA 有哪些内存空间，怎么快速区分？**
register（thread 私有、最快）、local（thread 私有但其实在 global、慢）、shared（block 共享、
片上快）、global（全局可见、大而慢但带宽高）、constant（只读、同址广播）、texture（只读专用
cache）、host（CPU 侧，跨 PCIe）。区分靠三问：谁拥有、谁访问、活多久。

**Q2：local memory 在片上吗？**
不在。名字带"local"但物理位于 global memory，只是对 thread 私有、经过 cache。它通常因
register spilling、动态索引的局部数组或过大私有对象而产生，是性能警告信号。

**Q3：constant memory 什么时候快、什么时候慢？**
一个 warp 的所有 lane 读**同一地址**时一次广播，最快；读**各不相同的地址**时退化成多次访问，
优势消失。所以只适合"很多线程读同一份小只读数据"。

**Q4：global memory 慢，为什么还说它带宽高？**
单次访问延迟高（几百拍），但通过足够多并发线程 + 合并访问，总吞吐很高。性能不取决于它本身，
而取决于访问模式（合并 vs 跨步）和数据复用。

**Q5：managed memory（统一内存）会自动让程序变快吗？**
不会。它提升的是可编程性（省掉手动 memcpy），数据靠按需页迁移，CPU/GPU 反复争用会触发大量
迁移而变慢。性能要靠 prefetch / advise 主动管理驻留。

