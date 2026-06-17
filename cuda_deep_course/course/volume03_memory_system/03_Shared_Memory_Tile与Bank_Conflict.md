# 03 Shared Memory、Tile 与 Bank Conflict

## 0. 先建立大局观：shared memory 是"block 自己的小白板"

在钻术语前，先用一个类比锁住直觉。把内存层次想成办公场景：

```text
global memory ≈ 公司档案室     容量大、谁都能取，但跑一趟很慢（几百个时钟周期）
shared memory ≈ 工位上的小白板  只有同组(block)人能看，但伸手就到（几个时钟周期）
register      ≈ 你手里的便签    只有你自己(thread)能用，最快
```

一个 block 的线程共用一块 shared memory（小白板）。它的全部意义就是：**把要反复用、
或要换个方向读的数据，从慢的档案室搬到快的白板上，让一组人围着白板高效协作**。

但白板有个隐藏规则——它被切成 **32 条并行通道（bank）**，32 个人同时伸手，落在**不同
通道**才能一起拿到；都挤**同一条通道**就得排队。这就是本章后半段 bank conflict 的来源。
整章其实只讲两件事：**怎么用好白板（§1–§4）** 和 **怎么不让大家挤同一条通道（§5–§7）**。

## 0.1 术语速查表（先扫一眼，下面逐个讲）

| 术语 | 一句话定义 | 类比 |
|---|---|---|
| **shared memory** | block 内线程共享的片上高速内存 | 工位小白板 |
| **tile** | 算法每次处理的一小块数据 | 白板上当前这批数 |
| **`__syncthreads()`** | block 内 barrier，等所有线程到齐 | "大家都写完了吗？" |
| **bank** | shared memory 的 32 条并行访问通道 | 白板的 32 条取数通道 |
| **bank conflict** | 多个 lane 挤同一 bank 的不同地址，被迫串行 | 多人抢同一条通道排队 |
| **padding** | 多加一列改变行跨度，错开 bank | 把数据摆斜一点，避开撞车 |
| **broadcast** | 多个 lane 读同一地址，硬件一次广播 | 大家看白板同一个数 |

## 1. Shared Memory 的三种价值

用 shared memory 前先问：**我图它什么？** 只有下面三个理由之一成立才值得用，否则白白
占资源、还多一道 `__syncthreads()`：

1. **复用**：一次 global load 被多个计算反复使用。
   *例*：GEMM 里一个 tile 的数据，被一行/一列的多次乘加共享——读一次档案室，白板上用很多次。
2. **重排**：以合并方式读入（§02 的 coalescing），再换个方向读出。
   *例*：转置——按行合并读进白板，再按列取出，避免直接对 global 做不合并的跨步访问。
3. **协作**：block 内线程交换彼此的部分结果。
   *例*：reduction——每个线程算一部分，放白板上，大家一起做树形归约。

```text
没有这三个理由之一 -> 别用 shared memory
  纯逐元素、无复用、无需协作的 kernel（如 vector add），直接走 global 反而更简单更快
```

> 一句话：**shared memory 不是"更快的内存"这么简单，它是为"复用/重排/协作"准备的协作区**。
> 没有协作需求硬上它，只会增加复杂度和同步开销。

## 2. Tile：算法的"一小块"，别和 block 混为一谈

**tile = 算法每次搬上白板处理的一小块数据**。它是个**算法概念**，不等于 block 的线程形状
（虽然常常一一对应）。初学最容易把"32×32 的 tile"和"32×32 的 block"当成一回事——它们
经常同尺寸，但一个说的是**数据块**，一个说的是**线程组**。

矩阵乘（复用型 tile）：

```text
block 的线程协作把 A、B 的一个子块(tile) 搬进 shared
-> 子块留在白板上
-> 每个线程对它做多次乘加（复用！一次 global load 喂很多次计算）
```

转置（重排型 tile）：

```text
按行合并读 input  -> 写进 tile（白板）
-> 交换 tile 下标（转置发生在白板内部）
-> 按行合并写 output（两端 global 访问都合并，§02）
```

> 关键区分：**block 是"谁来干活"（线程），tile 是"这次干哪块"（数据）**。一个大矩阵会被
> 切成很多 tile，由 block 一块一块轮流处理。

## 3. 必须同步的原因

shared memory 一旦涉及"一个线程写、另一个线程读"，就必须用 `__syncthreads()` 隔开。
看这段转置的核心：

```cpp
tile[threadIdx.y][threadIdx.x] = input[index];  // 线程 A 写自己负责的格子
__syncthreads();                                 // 等全 block 都写完
float value = tile[threadIdx.x][threadIdx.y];    // 线程 A 现在读的是别人写的格子
```

**为什么非加不可？** 第三行 `tile[threadIdx.x][threadIdx.y]` 读的格子，多半是**别的线程**
在第一行写进去的（下标交换了）。GPU 上线程是分 warp 异步推进的，没有 barrier 的话，跑得
快的线程会在别人还没写完时就来读——读到**旧的/未初始化的值**，这就是 race。

`__syncthreads()` 在这里提供两个保证：

```text
1. 顺序：barrier 之后的读，一定发生在 barrier 之前所有写之后
2. 可见性：barrier 之前对 shared/global 的写，barrier 之后对全 block 可见
```

> 直觉：白板协作的铁律是"**都写完了，才能开始看别人写的**"。`__syncthreads()` 就是那句
> "大家都写完了吗？写完我们再继续"。

## 4. Barrier 必须被全 block 一致到达（否则死锁）

`__syncthreads()` 有个致命陷阱：它要求**block 里每个线程都执行到这一句**。如果把它放进
只有部分线程进得去的分支，就会出事：

```cpp
// ❌ 危险：只有 valid 的线程会到达 barrier
if (valid) {
  __syncthreads();   // 不满足 valid 的线程根本不会执行到这里
}
```

后果：到达 barrier 的线程**死等**那些永远不会来的线程 —— **死锁**，或未定义行为。

正确写法是：让**所有线程都无条件到达 barrier**，把条件判断放在 barrier 两侧的读写上，
而不是 barrier 本身：

```cpp
// ✅ 正确：barrier 在分支外，人人都到得了
if (valid) {
  tile[...] = input[...];     // 条件控制"要不要写"
}
__syncthreads();              // 但同步是全员无条件参与
if (outputValid) {
  output[...] = tile[...];    // 条件控制"要不要读/写回"
}
```

> 记忆点：**条件可以控制"读不读、写不写"，但绝不能控制"同不同步"**。`__syncthreads()`
> 必须在所有线程都走得到的位置（边界判断要写成这种"先全员同步、再各自决定读写"的模式）。

## 5. Bank：shared memory 的 32 条并行通道

为了让一个 warp 的 32 条 lane 能**同时**访问，shared memory 在硬件上被切成 **32 个 bank
（通道）**。连续的 4 字节 word 轮流分到这 32 个 bank 上：

```text
word 编号:  0   1   2   ...  31   32   33  ...  63
bank:       0   1   2   ...  31    0    1  ...  31    <- 每 32 个 word 一轮回
            └──────── 第一轮 ───────┘ └──── 第二轮 ────┘
```

映射规则（对常见的 4 字节元素，第一层直觉）：

```text
bank = wordIndex % 32
```

一个 warp 访问 shared 时，硬件按 lane 落在哪些 bank 来决定快慢：

```text
✅ 32 条 lane 落在 32 个不同 bank    -> 一拍全部并行完成（理想）
✅ 多条 lane 读同一个地址            -> broadcast，硬件一次广播，也不慢
❌ 多条 lane 落在同一 bank 的不同地址 -> bank conflict，被迫拆成多拍串行
```

注意区分最后两种：**同一地址 = 广播（不冲突）**，**同 bank 不同地址 = 冲突**。这是最容易
混的一点。

> 直觉：32 条通道好比超市 32 个收银台。32 个人各去一个台 → 同时结完（理想）；都看同一块
> 电子屏价格 → 一次广播（广播）；都挤同一个收银台买不同东西 → 排队（冲突）。
>
> 具体行为仍应以目标架构文档和 profiler 为准。

## 6. 为什么转置会触发最严重的冲突

声明一个常见的转置 tile：

```cpp
__shared__ float tile[32][32];
```

行主序下，`tile[row][col]` 的线性 word 编号是：

```text
wordIndex = row * 32 + col
bank      = (row * 32 + col) % 32 = col      // row*32 是 32 的倍数，对 bank 无影响
```

关键就在 `bank = col`。转置时有一步要**固定列、遍历行**地读一整列（一个 warp 的 32 条
lane 对应 32 个不同的 `row`，但 `col` 相同）：

```text
lane 0 读 tile[0][col]   -> bank = col
lane 1 读 tile[1][col]   -> bank = col
lane 2 读 tile[2][col]   -> bank = col
...
lane 31 读 tile[31][col] -> bank = col
全部 32 条 lane 落在同一个 bank(col)，但地址各不相同 -> 32-way 冲突
```

**"严重"是可以量化的**。硬件规则是：同一 bank 上有 N 个不同地址被同时请求，就拆成 **N 拍
串行**（同地址广播不算）。这里 32 条 lane 全压在同一 bank 的 32 个不同地址上，本该一拍完成
的访问被拆成 **32 拍**——这次 shared 读取直接慢 **32 倍**，是 32-way bank conflict 的最坏
情形。

```text
无冲突：  32 lane -> 32 个不同 bank -> 1 拍   ┐
32 路冲突：32 lane -> 同 1 个 bank  -> 32 拍 ┘  相差 32 倍
```

所以消除冲突的收益不是"略快一点"，而是**把一个 32 倍的串行惩罚降回 1 倍**。

## 7. Padding：加一列，把"撞同一通道"错开

改成每行多加一列（33 而非 32）：

```cpp
__shared__ float tile[32][33];   // 逻辑上还是 32 列，物理每行 33 个 word
```

再算一次"固定列、遍历行"时的 bank：

```text
wordIndex = row * 33 + col
bank      = (row * 33 + col) % 32 = (row * 33) % 32 + col ...
          = (row + col) % 32          // 因为 33 % 32 = 1，row*33 ≡ row (mod 32)
```

对比 §6 的 `bank = col`：加了 padding 后变成 `bank = (row + col) % 32`。现在固定 `col`、
让 `row` 从 0 走到 31，bank 也跟着从 `col` 走到 `col+31`（模 32）——**32 条 lane 正好散到
32 个不同 bank**，冲突归零。

```text
未 padding [32][32]：列访问 bank = col        -> 32 lane 全同 bank -> 32 路冲突
已 padding [32][33]：列访问 bank = (row+col)%32 -> 32 lane 散满 32 bank -> 0 冲突
```

代价极小：**多出的那一列不存任何逻辑数据，只是把每行的物理跨度从 32 改成 33**，让"列方向"
不再整除 32，从而错开 bank。每个 block 多花 `32 * 4 = 128` 字节 shared，换来 32 倍的访问
加速，几乎总是划算。

> 一句话：padding 的本质是**故意让行跨度不是 32 的整数倍**，使"同列不同行"的元素落进不同
> bank。这是消除固定步长 bank conflict 最常用、最便宜的手段。

## 8. 动态 Shared Memory

前面的 `tile[32][33]` 是**静态** shared memory——大小在编译期写死。如果 tile 大小要到
**运行时**才知道（比如随输入规模变化），就用动态 shared memory：

```cpp
extern __shared__ float buffer[];        // 大小不写在这里
kernel<<<grid, block, bytes>>>(...);     // 启动时第三个参数指定字节数
```

注意两点：

- 大小通过 `<<<grid, block, bytes>>>` 的**第三个参数**在 launch 时给出。
- `extern __shared__` 只能声明**一块**。要在同一块里放多个数组，得自己手动切分指针：

```cpp
extern __shared__ float buffer[];
float* first  = buffer;                  // 前 firstCount 个给 first
float* second = buffer + firstCount;     // 之后的给 second
// 大小、对齐、越界全部自己负责（编译器不帮你检查）
```

> 何时用：tile/数组大小依赖运行时参数。否则优先用静态 shared（写法简单、编译期能查越界）。

## 9. Shared Memory 与 Occupancy：tile 不是越大越好

shared memory 是**每个 SM 上的有限资源**。一个 block 用得越多，SM 上能**同时驻留的 block
就越少**（occupancy 下降，见卷五第 03 章）。于是 tile 大小是个**双向权衡**：

```text
更大 tile -> 每个数据复用次数更多（好：更少 global 访问）
更大 tile -> 每 block 吃更多 shared -> 同时驻留 block 变少（坏：并发/延迟隐藏变差）
```

举例：T4 每个 SM 的 shared memory 有限，若一个 block 用 48KB，可能只能驻留 1 个 block；
减到 24KB 也许能驻留 2 个，warp 更多、更能隐藏延迟。哪个更快**没有定式**——

> 经验法则：**tile 大小要扫参数 + 看 profiler 实测决定**，不要想当然"越大越快"。复用收益
> 和 occupancy 损失在某个点会反转，那个拐点只有测了才知道。

## 10. 实践

在 transpose lab 中亲手验证 bank conflict 的代价：

1. 把 `tile[32][33]` 改回 `tile[32][32]`（去掉 padding）。
2. 验证结果**仍然正确**——bank conflict 只影响**速度**，不影响**正确性**（这点很重要，
   它是性能 bug，不是功能 bug，不会报错，只会变慢）。
3. 用 Nsight Compute 对比两版的 **shared-memory bank conflict 指标**和**耗时**，应能看到
   `[32][32]` 版冲突激增、变慢。
4. 恢复 `[32][33]` padding。

```bash
ncu --set full --kernel-name regex:transposeShared \
  ./labs/03_memory_system/transpose/transpose 4096 4096
# 关注 shared 相关的 bank conflict / wavefronts 指标
```

> 自测问题：bank conflict 是正确性问题还是性能问题？（答：纯性能问题——结果照样对，只是
> 慢，所以它**不会被功能测试抓到**，只能靠 profiler 或对 bank 的理解发现。）

## 11. 资料映射

- Best Practices Guide：Shared Memory and Memory Banks。
- Programming Guide：Shared Memory、Synchronization Primitives。

## 12. 面试题（附参考答案）

**Q1：什么时候该用 shared memory？**
当且仅当有以下三种价值之一：**复用**（一次 global load 喂多次计算）、**重排**（合并读入再
换方向读出）、**协作**（block 内线程交换部分结果）。纯逐元素、无复用的 kernel（如 vector
add）用它反而增加复杂度和同步开销。

**Q2：`__syncthreads()` 为什么不能放在 `if` 分支里？**
它要求 block 内**每个线程都到达**。放进只有部分线程进得去的分支，到达的线程会死等永远不来
的线程 → 死锁。正确模式是"全员无条件同步，条件只控制 barrier 两侧的读写"。

**Q3：什么是 bank conflict？它影响正确性还是性能？**
shared memory 分 32 个 bank，一个 warp 的多条 lane 落在**同一 bank 的不同地址**时，硬件被迫
拆成多拍串行，这就是 bank conflict。它只影响**性能**（结果照样正确），是不会报错的性能 bug，
只能靠 profiler 或对 bank 的分析发现。

**Q4：`tile[32][32]` 转置为什么慢，`tile[32][33]` 为什么快？**
`[32][32]` 时按列访问的 `bank = col`，一个 warp 的 32 条 lane 全压在同一 bank → 32 路冲突，
慢 32 倍。加一列变 `[32][33]` 后 `bank = (row+col)%32`，32 条 lane 散到 32 个不同 bank →
零冲突。代价仅每 block 多 128 字节 shared。

**Q5：多条 lane 读同一个地址会冲突吗？**
不会。**同一地址**触发 **broadcast**（硬件一次广播），不算冲突。只有**同 bank 的不同地址**
才冲突。这是最容易混的点。

**Q6：tile 越大越好吗？**
不是。tile 越大复用越多，但每 block 占用更多 shared memory，导致每 SM 驻留的 block 变少
（occupancy 下降），削弱延迟隐藏。复用收益和并发损失存在拐点，必须扫参数 + profiler 实测。

