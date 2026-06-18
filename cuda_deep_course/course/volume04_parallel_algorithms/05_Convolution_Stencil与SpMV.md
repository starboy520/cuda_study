# 05 Convolution、Stencil 与 SpMV

## 0. 本章主线：三个"邻域依赖"算法

前几章的 reduction/scan/histogram，每个输出基本只跟"自己那份数据"有关。本章三个算法不同——
**每个输出都依赖一片"邻居"**，这带来共同的新挑战：数据复用、边界处理、负载均衡。

```text
Convolution（卷积）  output[i] = 周围一窗口输入 × filter 的加权和   —— 邻居是固定窗口
Stencil（模板计算）  网格点新值 = 它和上下左右邻居的组合          —— 邻居是空间相邻格点
SpMV（稀疏矩阵×向量） y[行] = 该行非零元素 × 对应 x 分量            —— 邻居是稀疏、不规则的
```

用一个类比统一它们：**都像"每个人要参考周围几个人的意见做决定"**。卷积/stencil 的"周围"
是规整的（固定窗口/相邻格），SpMV 的"周围"是杂乱的（每行非零数量、位置都不同）。规整的
难点在**复用与边界**，不规整的难点在**负载均衡**。

## 0.1 术语速查表

| 术语 | 一句话定义 |
|---|---|
| **filter / kernel** | 卷积的权重窗口（注意：和 CUDA kernel 同名但不同义）|
| **radius** | 窗口半径，filter 宽度 `F = 2·radius + 1` |
| **halo（晕区）** | tile 边缘为了算边界输出，额外要加载的邻居数据 |
| **stencil** | 用固定邻居模板更新每个网格点的计算（如 5-point）|
| **ping-pong buffer** | 两块缓冲轮流当输入/输出，避免原地更新读到脏值 |
| **CSR** | 稀疏矩阵压缩格式：rowOffsets + columnIndices + values |
| **负载均衡** | 让每个线程/warp 干的活尽量均等，避免有的闲有的忙 |

## 1. 共同点：为什么它们比逐元素算法难

这三个算法都不是简单"一输入对应一输出"，而是**一输出依赖一片输入**：

- **Convolution**：输出依赖一个邻域窗口和 filter。
- **Stencil**：网格点依赖周围格点。
- **SpMV**：一行输出依赖该行所有稀疏非零元素。

由此产生三个贯穿本章的核心问题：

```text
1. 数据复用：相邻输出的输入窗口大量重叠，怎么不重复读？     -> shared tile
2. 边界：    窗口超出数组边缘怎么办？                       -> halo + 边界策略
3. 负载均衡：每个输出的工作量不等（尤其 SpMV）怎么分配？     -> 映射策略
```

## 2. 1D Convolution：复用是怎么省出来的

```text
output[i] = sum_k( filter[k] * input[i + k - radius] )
```

每个输出读一个宽度 `F = 2·radius + 1` 的窗口。关键观察：**相邻输出的窗口大量重叠**——
`output[i]` 和 `output[i+1]` 的输入窗口重叠了 `F-1` 个元素。于是 naive 版本里几乎每个
输入都被附近 `F` 个输出各读一遍：

```text
naive global 读总量 ≈ N * F      （N 个输出，每个读 F 个输入）
理想（每元素读一次）≈ N
浪费倍数 ≈ F
```

`radius=3` 时 `F=7`，naive 把每个输入从 global 读了约 **7 遍**。这正是 shared tile 的用武
之地——让一个 block **协作把"中心数据 + 左右 halo"读进 shared 一次**，块内所有输出再从
shared 复用：

```text
       ← halo →   ←──── 本 block 负责的中心输出 ────→   ← halo →
input: [r r r]    [c c c c c c c c c c c c c c c c]    [r r r]
       左边 radius 个    block 要算的输出对应区间       右边 radius 个
       （邻居，不输出）                                （邻居，不输出）
```

为什么要 halo？因为 block 边缘的输出，它的窗口会**伸到 block 外**。不把这些边缘邻居
（halo）也加载进来，边缘输出就算不全。加载一次后多次复用，把 `F` 倍的重复 global 读压成
接近 1 倍，剩下的复用全发生在快得多的 shared 上。

> 一句话：**filter 越宽（F 越大），naive 浪费越多、shared tile 收益越大**。卷积优化的核心
> 就是"把重叠窗口读一次、片上复用"。

## 3. 2D Convolution：halo 变成"一圈"

1D 的 halo 是左右两段；2D 的 halo 是**四周一圈**（上下左右 + 四角）。一个 `T×T` 的输出 tile，
要加载的 shared 是 `(T+2r)×(T+2r)`：

```text
       ┌─────────────────────────┐
       │  halo（上）              │
       │  ┌───────────────────┐  │
       │  │                   │  │
  halo │  │   T×T 中心输出     │  │ halo
 （左）│  │                   │  │（右）
       │  └───────────────────┘  │
       │  halo（下）              │
       └─────────────────────────┘
       整块大小 = (T+2r) × (T+2r)
```

难点在于**线程数（T×T）少于要加载的元素数（(T+2r)²）**，所以加载策略有几种：

```text
① 每 thread 先加载自己的中心，再由一部分 thread 额外加载 halo —— 常见，但 halo 加载分支多
② 每 thread 循环加载多个元素，直到整个 (T+2r)² 都填满      —— 代码统一，无特殊分支
③ 干脆不显式做 halo，靠 L1/L2 cache 服务边缘读取          —— 最简单，但复用不如 shared 可控
```

**filter 小且只读** → 适合放 **constant memory**（卷三第 01 章）：所有线程同时读同一个
filter 系数时，constant cache 一次广播，很高效。

> 取舍：2D 卷积的 shared tile 收益比 1D 更大（复用是二维的），但 halo 加载的边界代码也更
> 复杂。tile 太小则 halo 占比高（浪费），太大则 shared 不够、occupancy 降——又是一个要实测
> 的权衡。

## 4. Stencil：为什么不能原地更新

Stencil（模板计算）常见于 PDE、流体、热传导——每个网格点的新值由它和邻居的组合算出。最经典
的 2D **5-point stencil**：

```text
new[i][j] = f( old[i][j], old[i-1][j], old[i+1][j], old[i][j-1], old[i][j+1] )
                自己        上            下            左            右
```

时间迭代时**必须用两块 buffer（ping-pong），不能原地更新**：

```cpp
for (int t = 0; t < steps; ++t) {
    stencilKernel(input, output);   // 读 input，写 output
    swap(input, output);            // 交换，下一轮 output 变 input
}
```

**为什么不能原地（`input == output`）？** 因为如果一个线程把 `grid[i][j]` 原地改成新值，
它的邻居线程算 `grid[i+1][j]` 时本该读 `grid[i][j]` 的**旧值**，却读到了**刚被改过的新值**
——这是一种 race（卷四第 01 章），结果取决于线程执行顺序，错乱且不可复现。

```text
原地更新（错）：邻居可能读到"本轮已更新"的值 -> 数据污染
ping-pong（对）：本轮只读 old、只写 new，两者物理分离 -> 邻居永远读到一致的旧值
```

3D stencil 工作集和 halo 更大，常用平面缓存（plane caching）、register queue 或多级 tile，
但"读旧写新、不原地"的铁律不变。

> 一句话：**stencil 的正确性铁律是"读旧写新"**，靠 ping-pong buffer 保证邻居读到的是
> 一致的上一轮状态。

## 5. 边界条件：不是随便加个 `if`

窗口/邻居超出数组边缘时怎么处理？这是**算法规格的一部分**，不同选择会得到不同的正确结果：

| 策略 | 越界时怎么取值 | 典型场景 |
|---|---|---|
| **Zero padding** | 越界处当 0 | 图像卷积（边缘渐暗）|
| **Clamp（钳位）** | 取最近的边缘值 | 图像（边缘延伸）|
| **Periodic（周期）** | 绕回另一端 | 物理周期边界、FFT |
| **Mirror（镜像）** | 沿边界对称反射 | 图像（避免边缘伪影）|
| **单独 boundary kernel** | 边界用专门 kernel 处理 | 边界逻辑复杂时 |

> 关键认知：**边界策略是"业务要什么结果"决定的，不是实现细节**。同一份数据，zero padding
> 和 clamp 算出来的边缘值不同，且都"正确"——取决于你的算法定义。所以写之前要先明确选哪种，
> 而不是随手 `if (out of range) continue`。

## 6. SpMV：稀疏带来的负载不均衡

CSR（Compressed Sparse Row）用三个数组存稀疏矩阵——只存非零元素：

```text
rowOffsets[]     每行在 values/columnIndices 里的起始位置（长度 = 行数+1）
columnIndices[]  每个非零元素的列号
values[]         每个非零元素的值
```

每行的计算：

```text
y[row] = sum over j in [rowOffsets[row], rowOffsets[row+1]):
           values[j] * x[columnIndices[j]]
```

难点全部来自**稀疏不规则**：

```text
1. 每行非零数量不同        -> 负载不均衡：有的线程算 3 个、有的算 3000 个
2. x 的访问是间接的(x[columnIndices[j]]) -> 地址跳跃、不合并、cache 不友好
3. 行很短 -> 分给一个 warp 浪费；行很长 -> 一个 thread 算太慢
```

> 对比前面：卷积/stencil 的邻居是**规整**的（窗口大小固定），所以难点是复用和边界；SpMV 的
> 邻居是**杂乱**的（每行不同），所以难点变成**负载均衡 + 不规则访存**。

## 7. SpMV 映射策略：没有万能解

因为行长差异巨大，"一个线程/warp 处理多少"要看矩阵特征：

| 策略 | 怎么分 | 适合 | 问题 |
|---|---|---|---|
| **thread-per-row** | 每线程算一整行 | 行短且均匀 | 长行时该线程拖后腿；x 访问不合并 |
| **warp-per-row** | 一个 warp（32 lane）协作算一行，再 warp 归约 | 行较长 | 短行时 32 lane 大量闲置 |
| **merge/load-balanced** | 按非零总数均分给线程，不按行分 | 高度不均匀矩阵 | 实现复杂，要额外的行边界查找 |

> 一句话：**SpMV 不存在对所有稀疏矩阵都最优的 kernel**。选策略前要先看矩阵的"行长分布"——
> 均匀短行用 thread-per-row，长行用 warp-per-row，极度不均匀用 load-balanced。这也是为什么
> 工程上常直接用 cuSPARSE（它内部按矩阵特征选策略）。

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

## 10. 面试题（附参考答案）

**Q1：卷积为什么要用 shared memory tile？收益从哪来？**
相邻输出的输入窗口大量重叠，naive 把每个输入从 global 读约 `F`（filter 宽度）遍。shared tile
让一个 block 协作把"中心 + halo"读进 shared 一次，块内输出在片上复用，把 `F` 倍重复读压成
接近 1 倍。filter 越宽收益越大。

**Q2：halo（晕区）是什么，为什么需要它？**
halo 是 tile 边缘为了算边界输出而额外加载的邻居数据。因为 block 边缘输出的窗口会伸到 block
外，不加载这些邻居就算不全边界输出。1D 是左右两段，2D 是四周一圈。

**Q3：stencil 为什么不能原地更新？**
原地更新时，一个线程改了 `grid[i][j]`，邻居线程算 `grid[i+1][j]` 时本该读它的旧值却读到新值
（race），结果依赖执行顺序、不可复现。必须用 ping-pong buffer"读旧写新"。

**Q4：卷积/stencil 的边界策略有哪些？为什么说它是算法规格？**
zero padding、clamp、periodic、mirror、单独 boundary kernel。因为同一份数据不同策略算出的
边缘值不同且都"正确"，取决于业务定义——所以是规格，不是随手加 `if`。

**Q5：SpMV 为什么难写出通用最优 kernel？**
稀疏矩阵每行非零数量差异巨大，导致负载不均衡；`x` 的间接访问不合并。thread-per-row 适合
均匀短行、warp-per-row 适合长行、load-balanced 适合极不均匀——没有一种对所有矩阵都最优，要
按行长分布选。

**Q6：filter 适合放哪种内存？为什么？**
constant memory。因为 filter 小、只读、且所有线程同时读同一个系数——正好命中 constant cache
的"同址广播"，一次广播给整个 warp。

