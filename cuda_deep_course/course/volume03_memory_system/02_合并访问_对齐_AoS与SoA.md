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

### 5.1 同样的数据，两种摆放方式

假设有一堆粒子，每个粒子有 `x, y, z, mass` 四个字段。有两种在内存里摆放它们的方式。

**AoS（Array of Structures，结构体数组）**——把"一个粒子的所有字段"放在一起：

```cpp
struct Particle {
  float x, y, z, mass;
};
Particle particles[count];   // 内存里：x0 y0 z0 m0  x1 y1 z1 m1  x2 y2 z2 m2 ...
```

**SoA（Structure of Arrays，数组的结构体）**——把"所有粒子的同一字段"放在一起：

```cpp
struct Particles {
  float* x;     // x0 x1 x2 x3 ...  全部 x 连续
  float* y;     // y0 y1 y2 y3 ...  全部 y 连续
  float* z;
  float* mass;
};
```

逻辑上装的是同一批数据，区别只在**内存排布顺序**。但这个顺序，直接决定了 warp 访问时
能不能合并（§1）。

### 5.2 为什么它对合并访问是生死攸关的

关键场景：一个 kernel 只用到某**一个字段**（比如只更新 `x`）。看这时 warp 的访问模式。

**AoS：相邻线程的 `x` 隔着整个结构体**

```text
内存：  [x0 y0 z0 m0][x1 y1 z1 m1][x2 y2 z2 m2] ...
        ↑lane0       ↑lane1       ↑lane2
线程只读 x，但相邻 lane 的 x 间隔 16 bytes（一个 Particle 的大小）
-> 这正是 §2 的跨步访问！stride = 16 bytes
-> 每条 lane 拉来一个 32B sector，里面 y/z/mass 全是白搬的
-> 利用率约 4 / 16 = 25%，甚至更低
```

**SoA：相邻线程的 `x` 紧挨着**

```text
内存：  [x0 x1 x2 x3 ... x31][y0 y1 ...] ...
        ↑lane0,1,2... 全部连续
32 条 lane 读 x0..x31 = 连续 128 bytes -> 完美合并（§1）
-> 利用率 100%，没有一个字节白搬
```

一句话：**只用部分字段时，AoS 把访问变成跨步、浪费带宽；SoA 让它保持连续、合并**。
这就是 GPU 上 SoA 经常更快的根本原因。

### 5.3 但 AoS 不是一无是处——看访问模式

选择不是绝对的，取决于 kernel **怎么用这些字段**：

| 场景 | 更优布局 | 原因 |
|---|---|---|
| 每个线程一次用到**整个结构**（x,y,z,mass 都要）| AoS 也行 | 一个粒子的字段本就该一起加载，AoS 反而局部性好、接口方便 |
| kernel 分阶段，每阶段只用**部分字段** | **SoA** | 只读的字段连续、合并；不碰的字段不浪费带宽 |
| 大多数 GPU 数据并行 kernel | **SoA** | 典型模式就是"一个 kernel 处理一个维度"，SoA 几乎总是赢 |

经验法则：**写 GPU kernel 默认优先考虑 SoA**，除非你确认每个线程总是整体使用结构。

### 5.4 折中方案：AoSoA

有时纯 SoA 不方便（接口要传一大堆数组指针），纯 AoS 又不合并。**AoSoA（Array of
Structures of Arrays）** 是折中：按 warp 大小（或向量宽度）分块，块内用 SoA，块间用 AoS。

```text
AoSoA（每 32 个为一组，组内 SoA）：
  [x0..x31][y0..y31][z0..z31][m0..m31]  <- 第 0 组 32 个粒子
  [x32..x63][y32..y63]...                <- 第 1 组
组内：一个 warp 读 x0..x31 连续合并 ✅
组间：保持结构化分块，便于向量化和工程管理
```

它在**合并访问、缓存局部性、工程接口**之间取平衡，是高性能库（如某些粒子/物理引擎）
常用的布局。初学阶段先掌握"AoS vs SoA 的合并差异"即可，AoSoA 知道有这回事、需要时再用。

> 一句话总结：**布局要顺着 warp 的访问模式走**。只用部分字段 → SoA 让访问连续合并；
> 整体使用 → AoS 也可；两者难取舍 → AoSoA 分块折中。

## 6. 向量化访问

### 6.0 先认识 `float4`:它不是关键词,是"打包好的 4 个 float"

后面会突然用到 `float4`,先把它讲清楚,否则会懵。

**`float4` 不是 C++ 关键词**(不像 `int`、`for` 那种语言内置的),而是 **CUDA 头文件里预先定义
好的一个结构体**——把 4 个 `float` 打包在一起:

```cpp
// CUDA 内置类型,概念上等价于:
struct float4 {
    float x, y, z, w;     // 4 个 float 挨在一起,共 16 字节
};
```

CUDA 提供一整套这样的"向量类型",按元素个数和基础类型组合:

```text
float1 float2 float3 float4      // 1~4 个 float
int1   int2   int3   int4        // int 版
double1 double2                  // double 版
char4, uchar4 ...                // 还有 char/short/long 等版本
```

用 `.x .y .z .w` 访问每个分量(2 个的只有 `.x .y`,3 个的到 `.z`):

```cpp
float4 v;
v.x = 1.0f; v.y = 2.0f; v.z = 3.0f; v.w = 4.0f;   // 4 个分量
float2 p;
p.x = 10.0f; p.y = 20.0f;                          // 只有 2 个分量
```

> 关键认知:`float4` 只是"**把 4 个 float 摆在一起、当一个整体看待**"。它本身没什么魔法。
> 真正有用的是——**GPU 能用一条指令一次读写这"一整块 16 字节"**,这才是"向量化访问"的来源。
> 记住这个铺垫,下面就不懵了。

### 6.1 它想解决什么:用更少的指令搬同样多的数据

先看普通写法——每个线程读一个 `float`:

```cpp
float v = input[index];     // 一条 32-bit load 指令,搬 4 bytes
```

如果每个线程需要连续的 4 个 `float`,最直白的写法是循环读 4 次,发 **4 条** load 指令。向量化
访问的思路是:把这连续的 4 个 `float` 当成一个 **`float4`(16 字节)**,用 **一条** 指令一次读回来:

```cpp
// reinterpret_cast 是 C++ 的"强制把这块内存当另一种类型看"
// 这里:把 input(float 指针)重新看成 float4 指针,再取第 index 个 float4
float4 v = reinterpret_cast<const float4*>(input)[index];
// 一条 128-bit load 指令,一次搬 16 bytes,得到 v.x v.y v.z v.w 四个分量
```

这行代码拆开看(怕你卡在 `reinterpret_cast` 上):

```text
input                                   是 const float*    (指向一串 float)
reinterpret_cast<const float4*>(input)  把它"重新解释"成 const float4*  (每 16 字节算一个元素)
[index]                                 取第 index 个 float4,即 input 里第 index*4 ~ index*4+3 这 4 个 float
v.x = input[index*4 + 0]
v.y = input[index*4 + 1]   ← 一次性拿到这连续 4 个
v.z = input[index*4 + 2]
v.w = input[index*4 + 3]
```

收益有两层:

```text
1. 指令数变少:4 条 32-bit load -> 1 条 128-bit load
   减少指令发射压力,对 memory-bound kernel 尤其有用(指令也是一种资源)
2. 每条指令搬运更宽:单指令 128-bit,更容易把内存总线喂满
   一个 warp 用 float4 一次就请求 32 * 16 = 512 bytes,天然是大块对齐访问
```

类比:搬砖时一次搬 4 块(一条指令)比来回跑 4 趟(四条指令)更省力——只要你的手(寄存器/对齐)
撑得住一次抓 4 块。

### 6.2 常见向量化类型

```text
float2  = 8  bytes   int2 / uint2
float4  = 16 bytes   int4 / uint4
double2 = 16 bytes
```

`float4` 是最常用的,因为 16 bytes 正好是 GPU 偏好的访问粒度上界。

### 6.3 三个必须满足的前提（否则崩或变慢）

向量化不是无脑加速，它有硬性约束：

```text
① 地址对齐：float4 的地址必须是 16 的倍数
   cudaMalloc 返回的基址天然对齐，但 input + 1 这种偏移会破坏对齐 -> 非法/降速
② 尾部处理：元素总数不是 4 的倍数时，最后几个元素凑不满一个 float4
   要么单独用标量处理尾巴，要么保证规模是 4 的倍数
③ 值得做：只有当访存确实是瓶颈时才有意义
   compute-bound kernel 向量化收益很小，别为优化而优化
```

第 ① 条最容易踩坑。下面是错误示范：

```cpp
// ❌ 危险：input+1 的地址多半不是 16 字节对齐，float4 读会出错
float4 v = reinterpret_cast<const float4*>(input + 1)[index];
```

### 6.4 关键澄清：向量化不修复错误的布局

这是最重要的一句话：**向量化只是"把每个线程的多次访问压成一次"，它不会把一个本来
不合并的访问模式变合并。** 如果你的 warp 地址模式是跨步的（§2），用 `float4` 只会
让每条 lane 一次跨步搬 16 bytes，浪费照样存在，甚至更糟。

```text
先保证：warp 内 32 条 lane 地址连续、合并（§1）
再考虑：每条 lane 用 float4 一次多搬几个 -> 锦上添花
顺序不能反。布局错了，向量化救不了。
```

> 一句话总结向量化：**在已经合并的前提下，用更宽的指令减少指令数、喂满总线**。它是
> "优化的最后一公里"，不是"布局的修正药"。

### 6.5 完整例子:vector add 的普通版 vs 向量化版

把前面的概念拼成一个完整可对照的例子,你就彻底懂了。任务:`c[i] = a[i] + b[i]`。

**普通版**(每个线程算 1 个元素):

```cpp
__global__ void addNaive(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];          // 每线程 1 个 float
}
// 启动:需要 n 个线程
addNaive<<<(n + 255) / 256, 256>>>(a, b, c, n);
```

**向量化版**(每个线程用 `float4` 一次算 4 个元素):

```cpp
__global__ void addVec4(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;                          // 能凑成多少个完整的 float4
    if (i < n4) {
        // 把 a/b/c 都当成 float4 数组来读写
        float4 va = reinterpret_cast<const float4*>(a)[i];   // 一次读 a 的 4 个
        float4 vb = reinterpret_cast<const float4*>(b)[i];   // 一次读 b 的 4 个
        float4 vc;
        vc.x = va.x + vb.x;                  // 4 个分量分别相加
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        reinterpret_cast<float4*>(c)[i] = vc;                // 一次写 c 的 4 个
    }
}
// 启动:只需要 n/4 个线程(每个干 4 个元素的活)
addVec4<<<(n/4 + 255) / 256, 256>>>(a, b, c, n);
```

对照看两版的差别:

```text
普通版:  n 个线程,每个 1 条 load×2 + 1 条 store  → 访存指令多
向量化:  n/4 个线程,每个 1 条 128-bit load×2 + 1 条 store → 指令数砍到 1/4
        warp 一次请求更宽,更容易喂满内存总线
```

**别忘了尾部处理**(§6.3 的第②条):如果 `n` 不是 4 的倍数,`n/4` 会漏掉最后 `n%4` 个元素,要
单独补一个标量 kernel 或在 host 端处理:

```cpp
// 处理剩下的 n%4 个(向量化版没覆盖到的尾巴)
__global__ void addTail(const float* a, const float* b, float* c, int n) {
    int start = (n / 4) * 4;                 // 从第一个没被 float4 覆盖的元素开始
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

> 自己动手:把 `week01_basics/vec_add` 的 kernel 改成 `float4` 版,用大数组(如 1<<26)对比两版
> 耗时。vector add 是 memory-bound,向量化通常能看到提升——这是体会向量化最直接的实验。

## 7. Pitch

### 7.1 它想解决什么：让二维数组"每一行都对齐"

回忆 §3：合并访问不仅要地址连续，还要**每次访问的起点对齐**到事务边界。现在考虑一个
二维数组（比如图像），你想让每个 warp 处理一行。如果按逻辑紧凑存放：

```cpp
// 紧凑存放：每行 width 个 float 紧挨着
addr(row, 0) = base + row * width * sizeof(float);
```

问题来了：只要 `width * sizeof(float)` **不是对齐粒度（如 128 bytes）的整数倍**，那么
**除了第 0 行，后面每一行的起点都会错位**，落在事务边界中间：

```text
假设对齐粒度 = 128 bytes，width = 33 个 float = 132 bytes（不是 128 的倍数）
第 0 行起点：  0      -> 对齐 ✅
第 1 行起点：  132    -> 不对齐 ❌（落在 128 边界后 4 字节处）
第 2 行起点：  264    -> 不对齐 ❌
...每一行都错位，每行访问都多付一次跨边界事务
```

### 7.2 解决办法：给每行末尾偷偷加 padding

`cudaMallocPitch` 的做法是：**把每一行的实际存储宽度撑大到对齐粒度的整数倍**，多出来
的部分是不用的 padding。这个"撑大后的每行字节数"就叫 **pitch**：

```cpp
float* pointer = nullptr;
size_t pitchBytes = 0;
cudaMallocPitch(&pointer, &pitchBytes,
                width * sizeof(float),  // 你逻辑上想要的每行字节数
                height);                // 行数
// 返回的 pitchBytes >= width*sizeof(float)，且是对齐粒度的整数倍
```

图示（width=33，被撑到 pitch=256 bytes）：

```text
逻辑宽度 132B          padding
|<--- 33 个 float --->|<- 124B 浪费 ->|
行0: [================][##########]   起点 0    对齐 ✅
行1: [================][##########]   起点 256  对齐 ✅
行2: [================][##########]   起点 512  对齐 ✅
每行起点都是 pitch 的整数倍 -> 全部对齐
```

代价是浪费一点显存（padding），换来**每一行访问都对齐、都能合并**。

### 7.3 怎么用：必须按 pitch 计算行地址

关键点：分配后你**不能再用 `width` 算行偏移**，必须用 `pitchBytes`。而且 pitch 以
**字节**为单位，所以要先转成 `char*` 再加：

```cpp
// 取第 row 行的起始地址
char* rowAddr = reinterpret_cast<char*>(pointer) + row * pitchBytes;
float* row_f  = reinterpret_cast<float*>(rowAddr);
float  v      = row_f[col];     // 访问 (row, col)
```

常见错误：

```cpp
// ❌ 错：用 width 算偏移，pitch 的对齐就白做了，还会算错地址
float v = pointer[row * width + col];

// ✅ 对：用 pitchBytes 算偏移
float v = *reinterpret_cast<float*>(
            reinterpret_cast<char*>(pointer) + row * pitchBytes) + col;
```

配套的二维拷贝用 `cudaMemcpy2D`，它同时接收源和目的的 pitch，自动跳过 padding。

### 7.4 什么时候需要它

```text
✅ 二维数据、按行做合并访问（图像、矩阵、特征图）
✅ width 不是对齐粒度整数倍时收益最明显
❌ 一维数组用不上（cudaMalloc 的基址已对齐）
❌ width 本就是对齐粒度整数倍时，pitch == width*sizeof(T)，加不加都一样
```

> 一句话总结 pitch：**用每行末尾的 padding 换取"每一行起点都对齐"，让二维数组逐行访问
> 都能合并**。`cudaMallocPitch` 帮你算好 padding，代价是你之后必须用 `pitchBytes`（而非
> `width`）来寻址。

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

