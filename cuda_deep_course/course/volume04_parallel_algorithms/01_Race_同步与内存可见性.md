# 01 Race、同步与内存可见性

> 本章是并行编程的"安全基石"。前面学的 reduction、transpose 都用了 `__syncthreads()`，
> 但没真正讲清"为什么必须同步"。这一章回答：多个线程同时读写同一块内存时，会出什么错，
> 以及 CUDA 提供哪些同步工具（barrier / fence / atomic / warp 同步）各管什么。

## 0.1 术语速查表（先扫一眼，下面逐个讲）

| 术语 | 一句话定义 | 让线程等待？ |
|---|---|---|
| **data race** | 多线程并发访问同一地址、至少一个写、且无同步 → 结果不确定 | — |
| **`__syncthreads()`** | block 内 barrier：全员到齐 + 内存可见 | 是 |
| **`__syncwarp(mask)`** | 一个 warp 内 lane 的同步 | 是 |
| **`__threadfence*()`** | fence：只保证本线程写入的顺序/可见范围 | **否** |
| **atomic** | 单地址读-改-写不可分割 | 否 |
| **happens-before** | "A 的写一定先于 B 的读可见"的顺序保证 | — |
| **kernel 边界** | 跨 block 唯一可移植的全局同步墙 | （Host 等）|

> 一句话先记住区别：**barrier 管"大家到齐 + 可见"，fence 只管"我的写按序可见"，atomic 管
> "一个地址不被打断"**。它们解决不同问题，下面逐一展开。

## 1. 什么是 Data Race

当多个执行者并发访问同一位置，至少一个是写，并且缺少正确同步时，就可能
发生 data race。

拆解这个定义的三个条件（缺一不构成 race）：

```text
① 多个线程访问【同一内存位置】
② 至少有一个是【写】       （全是读不会有 race）
③ 之间【没有同步】保证先后顺序
→ 三者同时满足 = data race = 结果不确定
```

错误例子：

```cpp
__shared__ int value;
if (threadIdx.x == 0) {
  value = 42;          // 线程0 写
}
// 缺少同步
if (threadIdx.x == 1) {
  output[0] = value;   // 线程1 读 —— 可能在线程0写之前就读了！
}
```

Thread 1 可能在 thread 0 写入前读取。**race 的可怕之处在于"有时对、有时错"**：
线程 0 碰巧先跑完，结果就对；线程 1 抢先，就读到未初始化的垃圾值。这种 bug 难复现、
难调试，所以要从原理上杜绝，而不是靠运气。

### 1.1 三种最常见的 race 场景

上面是"写后读"漏同步。实战中 race 主要有三类,认得出它们才能防:

**① Read-Modify-Write race(最经典)**:多个线程对同一地址"读-改-写"

```cpp
// ❌ 多个线程同时 counter++
counter++;   // 实际是三步:读 counter → 加 1 → 写回
```

`counter++` 不是一条原子指令,而是"读→改→写"三步。两个线程可能都读到 5、各自加成 6、都写回
6——本该是 7,**丢了一次更新**。这正是 atomic(第 9 节速查、卷四第 02 章)要解决的:让这三步
不可分割。

**② 写后读 race(producer-consumer)**:一个线程写、另一个读,没保证先后(就是 §1 的例子)。
解法:block 内用 `__syncthreads()`,跨线程发布数据用 fence + flag(第 5 节)。

**③ 共享中间结果的 race(reduction/scan 里最常见)**:

```cpp
// ❌ shared 树形归约漏了同步
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];   // 读别人上一轮写的值
    // 缺 __syncthreads()！
}
```

`sdata[tid] += sdata[tid + s]` 读的是**别的线程**上一轮写进 sdata 的值。没有 barrier,快的
线程会读到还没更新的旧数据。这是你写 reduction(卷四/03)时最容易漏的同步。

> 识别 race 的口诀:**只要"一个线程读/写的数据,是另一个线程写的",中间就必须有同步**。
> 三类场景对应三种解法:读改写→atomic,写后读→barrier 或 fence+flag,共享中间结果→barrier。

## 2. Barrier 的两个作用

```cpp
__syncthreads();
```

第一层理解：

1. block 内 thread 到达同一执行阶段。
2. barrier 前相关内存操作对 barrier 后参与 thread 可见。

仅仅"等待时间"不是全部，内存顺序同样重要。

第 2 点常被忽略，但它才是 race 的深层根源。即使你"感觉"thread 0 先写、thread 1
后读，没有 barrier 也可能读到旧值——原因有两层：

- **编译器**可能为了优化重排指令，或把写入暂存在寄存器里**迟迟不落回** shared
  memory；它并不知道别的 thread 在等这个值。
- **硬件**层面，不同 thread 的内存写入到达 shared/global 的**顺序和时机并不保证**，
  一个 thread 已经写了，不等于另一个 thread 立刻能看见。

`__syncthreads()` 同时解决"等待"和"可见性"两件事：它既让所有 thread 对齐到同一点，
又强制 barrier 之前的内存写入在 barrier 之后**对全 block 可见**（建立 happens-before
关系）。所以它不是单纯的"等一下"，而是一道**内存顺序的栅栏**。这也是为什么
fence（第 5 节）不能替代它——fence 只管顺序，不管"大家都到齐"。

### 2.1 `__syncthreads()` 的三个带返回值变体

除了基本的 `__syncthreads()`,还有三个"同步 + 顺便统计"的变体,在某些算法里很省事:

```cpp
int  __syncthreads_count(int predicate);  // 同步 + 返回"predicate 非 0 的线程数"
int  __syncthreads_and(int predicate);    // 同步 + 返回"是否所有线程 predicate 都非 0"
int  __syncthreads_or(int predicate);     // 同步 + 返回"是否至少一个线程 predicate 非 0"
```

它们先做一次普通 barrier,再额外对整个 block 做一次归约统计。用途举例:

```text
__syncthreads_count:统计 block 内满足条件的线程数(如 histogram、计数)
__syncthreads_and:  判断"是否全 block 都满足"(如全部收敛才退出循环)
__syncthreads_or:   判断"是否还有线程要继续"(如还有活就再迭代一轮)
```

> 这三个变体和基础版一样,**也受 divergent barrier 约束**(第 3 节)——必须全 block 都到达。
> 它们只是"同步顺便统计",省掉一次额外的 block 级归约。

### 2.2 `volatile` 和 CUDA 内存模型(进阶但要知道)

有时你会看到 shared/global 指针被标 `volatile`,它和同步有关:

```cpp
volatile int* sdata = ...;   // 告诉编译器:每次都真读/真写内存,别缓存到寄存器
```

**`volatile` 解决的是"编译器优化导致的可见性问题"**:编译器可能把 `sdata[tid]` 缓存进寄存器、
不每次都真的读写 shared,于是别的线程的更新看不到。`volatile` 强制每次都访问真实内存。

```text
volatile 管:编译器层面"每次真读真写"(不缓存到寄存器)
__syncthreads 管:全 block 到齐 + 跨线程可见 + 硬件层面顺序
→ volatile 不能替代 __syncthreads!它只解决编译器缓存,不解决"大家到齐"和硬件顺序
```

历史上 warp 内 reduction 曾用 `volatile` 省 `__syncthreads`(老架构锁步),但 Volta+ 独立线程
调度后(§4.2)这样不安全了,现在应该用 `__syncwarp` 或 shuffle 原语。

> 一句话:`volatile` 是个"弱"工具,只管编译器别缓存。**现代代码优先用 `__syncthreads`/
> `__syncwarp`/atomic/shuffle 这些有明确语义的原语,别依赖 `volatile` 做同步。**

## 3. Divergent Barrier

危险：

```cpp
if (threadIdx.x < 16) {
  __syncthreads();
}
```

一个 block 中部分 thread 到达、部分不达到，行为不正确。

为什么这会坏？因为 `__syncthreads()` 的语义是"**等本 block 的所有 thread 都到达
这一点**"。它就像班车规定"全员到齐才发车"。上面的代码里，`threadIdx.x >= 16`
的 thread 根本不会走进这个 `if`，永远不会到达这班车——于是已经上车的前 16 个
thread **无限等待**那些永不出现的同伴。结果是死锁或未定义行为（具体取决于架构对
"到达"的实现）。

关键判据：**barrier 的"到达与否"必须对整个 block 一致**。可接受的条件是那些
全 block 取值相同的量，例如 `blockIdx`：

```cpp
if (blockIdx.x == 0) {
  __syncthreads();
}
```

因为同一 block 的所有 thread 看到相同 `blockIdx.x`，要么**全部**进入、要么
**全部**跳过，barrier 两侧的参与集合一致，不会有人被落下。反之，任何依赖
`threadIdx`、数据值、循环次数因 thread 而异的条件，都不能用来包裹 barrier。

## 4. Warp 同步

```cpp
__syncwarp(mask);
```

用于指定 mask 内的 warp lane 同步和相应内存顺序。

### 4.1 它和 `__syncthreads()` 的区别

```text
__syncthreads()  → 同步【整个 block】的所有线程（粗，跨多个 warp）
__syncwarp(mask) → 只同步【一个 warp 内】mask 指定的 lane（细，单 warp）
```

当你的协作只发生在一个 warp 内（32 线程）时，用 `__syncwarp` 比 `__syncthreads`
更便宜——不必让整个 block 都停下来等。

### 4.2 为什么不能省略它（独立线程调度）

```cpp
// ❌ 危险：依赖"同一 warp 天然 lockstep"
sdata[tid] = x;       // 每个线程往 shared 写自己的值
// 没有任何同步
y = sdata[tid ^ 1];   // 读"邻居"那条 lane 刚写的值
```

先看这段在干什么。`tid ^ 1` 是把最低位翻转——lane 0 读 lane 1、lane 1 读 lane 0、lane 2 读
lane 3……即**每个线程要读另一个线程刚写进 shared 的值**。这是跨线程读写,本该需要同步保证
"对方写完了我才读"。

**但很多人会想**:这 32 个线程在同一个 warp 里,warp 不是同步执行的吗?不加同步它们也会步调
一致,我读的时候对方肯定写完了吧?——**这个想法在老 GPU 上碰巧成立,在新 GPU 上会出错。**

关键在架构的转折:

```text
老架构(Volta 之前):
  同一 warp 的 32 条 lane 严格【锁步(lockstep)】——永远执行同一条指令、步调完全一致。
  → 所有 lane 一定都执行完"写 sdata",才会一起执行"读 sdata"
  → 不加同步也【碰巧】对(读的时候对方保证写完了)

新架构(Volta 及以后,你的 T4/A100 都是):
  引入【独立线程调度(Independent Thread Scheduling)】——同 warp 的 lane 可以走到不同位置,
  不再保证步调一致。
  → 可能 lane 0 已经在"读 sdata"了,而 lane 1 还没执行到"写 sdata"
  → lane 0 读到旧的/垃圾值 → 出错!
```

**为什么这个坑特别坑**:

```text
1. 间歇性出错:有时对有时错(取决于这次调度顺序),难复现
2. 只在新架构暴露:老卡测试通过,换新卡就崩
3. 看代码看不出问题:逻辑上"先写后读"明明是对的
```

正是这三点,让它成为新手最容易踩、最难复现的坑之一。

正确做法是显式 `__syncwarp`（或用卷四第 02 章的 shuffle 原语，它们自带 `_sync`）：

```cpp
sdata[tid] = x;
__syncwarp();          // 显式:确保 warp 内所有 lane 都写完了
y = sdata[tid ^ 1];    // 现在读邻居的值,保证对方已写完
```

> **顺带解答"NVIDIA 为什么要改成独立线程调度,自找麻烦?"** 因为老的锁步模型有个限制:同 warp
> 的线程若想各走各的分支(更细粒度的并行,如 warp 内的锁、生产者-消费者),锁步会死锁。独立线程
> 调度让 warp 内线程能真正独立推进、功能更强,代价就是你不能再假设它们锁步、必须显式同步。
> 这是"更灵活"换来的"必须更严谨"。

> 教训：**永远不要依赖"warp 天然同步"来省略同步原语。**

## 5. Memory Fence（内存栅栏）

Fence 主要约束调用 thread 的内存操作顺序与可见范围，不负责让其他 thread
都到达某个阶段。

常见三个范围（范围越大越慢）：

```text
__threadfence_block   保证对【本 block】可见
__threadfence         保证对【整个 device 所有线程】可见
__threadfence_system  保证对【CPU + 所有 GPU】可见
```

### 5.1 Fence ≠ Barrier（最容易混的点）

```text
barrier (__syncthreads)：让一群线程"都到齐"这个点（等待 + 可见性）
fence   (__threadfence) ：只保证"我这个线程的内存写入"按顺序、对指定范围可见
                          —— 它【不让任何线程等待】
```

fence 解决的是"我写的数据，别人能不能看到、按什么顺序看到"，不解决"大家到齐"。

### 5.2 为什么 fence 单独用没意义——必须配 flag

fence 不让别人等，那别的线程怎么知道"数据已经准备好了"？答案是配一个 **flag**
（标志位）+ atomic。经典的 **producer-consumer（生产者-消费者）** 协议：

```cpp
// Producer（生产者线程）：
data[0] = 42;                 // ① 写数据
__threadfence();              // ② 栅栏：保证 data 的写入【先于】flag 可见
atomicExch(&flag, 1);         // ③ 发布信号：flag 置 1

// Consumer（消费者线程）：
while (atomicAdd(&flag, 0) == 0) { }  // ④ 自旋等 flag 变 1
__threadfence();                       // ⑤ 栅栏：保证读 flag 之后才读 data
int v = data[0];                       // ⑥ 现在保证读到 42，不是旧值
```

关键在第 ② 步的 fence：**如果没有它**，编译器/硬件可能把"写 flag"重排到"写 data"
之前，于是 consumer 看到 `flag==1` 却读到 `data` 的旧值——数据还没写进去信号就发了。
fence 强制了"data 先于 flag"的顺序，这才让 flag 成为可靠的"数据就绪"信号。

> 一句话：**fence 管顺序，flag 管通知，两者配合才能跨线程安全传递数据。**
> 单独一个 fence 不通知任何人，单独一个 flag 不保证数据已写完。

### 5.3 三种 fence 范围怎么选

`__threadfence_block` / `__threadfence` / `__threadfence_system` 的区别是**可见范围**,
范围越大越慢。怎么选取决于"谁要看到这次写入":

| fence | 保证可见范围 | 用在哪 | 成本 |
|---|---|---|---|
| `__threadfence_block` | 同 **block** 内线程 | block 内跨 warp 传数据(但通常 `__syncthreads` 更直接)| 最低 |
| `__threadfence` | 整个 **device** 所有线程 | 跨 block 的 producer-consumer、global flag | 中 |
| `__threadfence_system` | **CPU + 所有 GPU** | 和 host(mapped 内存)或其他 GPU 通信 | 最高 |

```text
选择原则:用【刚好够】的范围,别用过大的。
  只在 block 内传数据 → block 级(或干脆 __syncthreads)
  跨 block 传(如多 block 协调) → device 级
  要让 CPU 看到(zero-copy/mapped 内存) → system 级
范围越大,要等的内存子系统越多,越慢。
```

> 注意区分 fence 范围和 atomic 范围:fence 管"我的写对多大范围可见",atomic 管"对一个地址的
> 读改写不被打断"。两者常配合(第 5.2 节的 flag 用 atomic、data 用 fence)。

## 6. Block 之间如何同步

普通 kernel 中，不同 block 无法使用 `__syncthreads()`。

**为什么不能？** 因为 block 的调度是不确定的——一个 SM 同时只驻留一部分 block，
其余在排队。如果让 block 0 等 block 5000，而 block 5000 可能还没被调度上 SM，
就会**永久死锁**。所以 CUDA 干脆不提供普通 kernel 的"全 grid barrier"。

常见替代方案：

```text
① 拆成多个 kernel       —— kernel 结束 = 天然全局同步墙（最常用）
② atomic 协调           —— 用全局 atomic 做计数器/队列
③ cooperative launch    —— grid.sync()，但要求所有 block 同时驻留（有限制）
④ 重新设计算法          —— 避免全局同步的需求
```

Reduction 就是用方案①——多阶段 kernel，每次 launch 之间就是同步墙（你 week2 写过）。

## 7. Cooperative Groups

Cooperative Groups 提供显式线程组抽象，让同步范围更清晰：

```cpp
namespace cg = cooperative_groups;
cg::thread_block block = cg::this_thread_block();
block.sync();                              // 等价于 __syncthreads()，但更明确

cg::thread_block_tile<32> warp =
    cg::tiled_partition<32>(block);        // 显式拿到一个 warp 组
warp.sync();                               // 等价于 __syncwarp()
```

优势是把"我在和谁同步"写得明明白白（block 级？warp 级？），也能创建任意大小的
tiled partition。但它**不会绕过硬件限制**——grid 级同步仍需 cooperative launch
且所有 block 同时驻留。本周了解概念即可，深入留到后面。

### 7.1 Grid 级同步:`grid.sync()`

Cooperative Groups 提供了普通 kernel 没有的**全 grid barrier**——让所有 block 在 kernel
内部同步一次(不用拆成多个 kernel):

```cpp
namespace cg = cooperative_groups;
__global__ void k(...) {
    cg::grid_group grid = cg::this_grid();
    // ... 第一阶段计算 ...
    grid.sync();                 // ✅ 全 grid 所有 block 在这里到齐!
    // ... 第二阶段(能安全读第一阶段所有 block 的结果)...
}
```

但它有**硬性前提**(否则死锁):**所有 block 必须同时驻留在 SM 上**。所以:

```text
- 必须用 cudaLaunchCooperativeKernel 启动(不是普通 <<<>>>)
- grid 大小不能超过"所有 block 能同时驻留"的上限
  (用 cudaOccupancyMaxActiveBlocksPerMultiprocessor × SM 数 算)
- 因为这个限制,它不能处理任意大的 grid
```

```text
对比"拆多 kernel"(第 6 节方案①):
  拆多 kernel:任意 grid 大小都行,最通用,但每次 launch 有开销
  grid.sync(): 省去多次 launch,但 grid 受限于"能同时驻留"——只适合中小规模
```

> 何时用 grid.sync():迭代算法(每轮要全局同步,如某些 stencil/图算法),grid 不大、且想避免反复
> launch 时。否则优先用"拆多 kernel"(更通用)。初学了解即可。

## 8. 用工具抓 race：Compute Sanitizer

光靠脑子判断有没有 race 很容易漏，用工具自动检测：

```bash
compute-sanitizer --tool racecheck ./program   # 抓 shared memory 的竞争
compute-sanitizer --tool memcheck  ./program   # 抓越界、非法访问
compute-sanitizer --tool synccheck ./program   # 抓 divergent barrier 等同步误用
```

- `racecheck`：专门查 shared memory hazard（漏 `__syncthreads()` 之类）。
- `memcheck`：查 global memory 越界、错误地址。
- `synccheck`：查 barrier 用错（如第 3 节的 divergent barrier）。

> 重要：**工具没报错 ≠ 代码一定对**。racecheck 只能发现"这次运行实际发生的"竞争，
> 没覆盖到的执行路径它看不到。所以工具是辅助，理解同步语义才是根本。

## 9. 同步工具速查（全章总结）

| 工具 | 作用 | 范围 | 让线程等待？ |
|------|------|------|-------------|
| `__syncthreads()` | 到齐 + 内存可见 | 整个 block | 是 |
| `__syncwarp(mask)` | 到齐 + 内存可见 | 一个 warp 内 | 是 |
| `__threadfence*()` | 只保证内存顺序/可见 | block/device/system | **否** |
| `atomic*()` | 单地址读改写不可分割 | 任意线程间 | 否 |
| kernel 边界 | 全局同步墙 | 整个 grid | （Host 等） |
| cooperative groups | 显式分组同步 | 可配置 | 视组而定 |

选择口诀：

```text
block 内大家到齐      → __syncthreads()
warp 内协作          → __syncwarp / shuffle
安全更新一个地址      → atomic
跨线程发布数据        → fence + flag
跨 block 阶段同步     → 多个 kernel（kernel 边界）
```

## 10. 实践

1. 写一个 block 内 producer/consumer 示例（第 1 节那种），先删除 `__syncthreads()`，
   用 racecheck 抓出竞争；再加回 barrier 验证消失。
2. 把 barrier 放进 `if (threadIdx.x < 16)` 这种 thread-dependent 分支，观察死锁/
   未定义行为（用 synccheck），再改成全 block 一致的条件。
3. 实现第 5.2 节的 fence + flag 协议，故意删掉 fence，思考为什么可能读到旧值。
4. 解释为何 fence 不能替代"所有 thread 等待"，atomic 不能替代 barrier。

## 11. 资料映射

- CUDA Programming Guide：Synchronization Primitives、Memory Fence Functions、Memory Model。
- CUDA C++ Best Practices Guide：Thread Synchronization。
- Compute Sanitizer：Racecheck、Synccheck。
- 配套：[卷四第 02 章 Atomic 与 Warp 级原语](02_Atomic与Warp级原语.md)、[卷四第 03 章 Reduction](03_Reduction从错误到优化.md)。

## 12. 面试题（附参考答案）

**Q1：data race 的构成条件是什么？**
三个同时满足：① 多线程访问**同一内存位置**；② **至少一个是写**；③ 之间**没有同步**保证先后。
缺一不构成 race。它的危险在于"有时对有时错"，难复现。

**Q2：`__syncthreads()` 只是"让线程等一下"吗？**
不止。它有两个作用：**到齐**（block 内所有线程对齐到同一点）+ **内存可见性**（barrier 前的写
对 barrier 后全 block 可见，建立 happens-before）。第二点才是消除 race 的根本——没有它，编译器
重排或写入迟迟不落回 shared，会让"看似先写后读"仍读到旧值。

**Q3：为什么不能把 `__syncthreads()` 放进 `if (threadIdx.x < 16)`？**
它要求 block 内**每个线程都到达**。部分线程进不去分支就永远到不了 barrier，已到达的线程无限
等待 → 死锁/未定义行为。包裹 barrier 的条件必须对全 block 一致（如 `blockIdx`），不能依赖
`threadIdx` 或数据值。

**Q4：fence 和 barrier 有什么区别？**
barrier（`__syncthreads`）让一群线程**到齐 + 可见**，会等待；fence（`__threadfence`）只保证
**调用线程自己**的写入按顺序、对指定范围可见，**不让任何线程等待**。fence 单独用没意义，要配
flag + atomic 才能跨线程通知"数据就绪"。

**Q5：为什么不能依赖"同一 warp 天然锁步"省略同步？**
Volta+ 的**独立线程调度**让同 warp 的 lane 可能走到不同位置，不再保证锁步。依赖锁步的代码会
在新架构上**间歇性出错**且极难复现。要显式 `__syncwarp()` 或用自带 `_sync` 的 shuffle 原语。

**Q6：不同 block 之间怎么全局同步？**
普通 kernel 没有全 grid barrier（block 调度不确定，强行等会死锁）。最常用、最可移植的做法是
**拆成多个 kernel**——kernel 边界就是天然全局同步墙。其他：全局 atomic、cooperative launch
（`grid.sync()`，要求所有 block 同时驻留）。

**Q7：racecheck 没报错，是不是就一定没有 race？**
不是。racecheck 只能发现**这次运行实际发生**的竞争，没覆盖到的执行路径/调度顺序它看不到。
工具是辅助，真正可靠的是从同步语义上保证正确。

**Q8:race 主要有哪几类?各怎么解决?**
三类:① read-modify-write(如 `counter++`)→ 用 **atomic**;② 写后读(producer-consumer)→ block
内 **barrier**、跨 block **fence+flag**;③ 共享中间结果(reduction/scan)→ 每轮加 **barrier**。
口诀:一个线程读/写的数据是另一个线程写的,中间就必须有同步。

**Q9:`volatile` 能替代 `__syncthreads()` 吗?**
不能。`volatile` 只解决**编译器层面**"每次真读真写、别缓存到寄存器",不解决"全 block 到齐"和
硬件层面的内存顺序。现代代码(Volta+)应该用 `__syncthreads`/`__syncwarp`/shuffle 等有明确语义
的原语,别依赖 `volatile` 做同步。

**Q10:`__threadfence` 的三种范围怎么选?**
按"谁要看到这次写入"选最小够用的:block 内→`__threadfence_block`;跨 block(device 内)→
`__threadfence`;要让 CPU/其他 GPU 看到(mapped 内存)→`__threadfence_system`。范围越大越慢。

**Q11:grid.sync() 和拆多 kernel 做全局同步,怎么选?**
grid.sync()(cooperative launch)省去多次 launch,但要求所有 block 同时驻留,grid 受限、只适合
中小规模;拆多 kernel 任意 grid 都行、最通用,代价是每次 launch 有开销。迭代算法且 grid 不大时
用 grid.sync(),否则优先拆多 kernel。

