# 01 Race、同步与内存可见性

> 本章是并行编程的"安全基石"。前面学的 reduction、transpose 都用了 `__syncthreads()`，
> 但没真正讲清"为什么必须同步"。这一章回答：多个线程同时读写同一块内存时，会出什么错，
> 以及 CUDA 提供哪些同步工具（barrier / fence / atomic / warp 同步）各管什么。

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
sdata[tid] = x;
// 没有任何同步
y = sdata[tid ^ 1];   // 读同 warp 另一条 lane 刚写的值
```

老架构（Volta 之前）同一 warp 的 lane 严格锁步，上面**碰巧**能工作。但 **Volta+
的独立线程调度**让 lane 可能走到不同位置，不再保证锁步——这段代码会**间歇性出错**，
而且只在新架构上暴露，极难排查。

正确做法是显式 `__syncwarp`（或用 §卷四02 的 shuffle 原语，它们自带 `_sync`）：

```cpp
sdata[tid] = x;
__syncwarp();          // 确保同 warp 写入对彼此可见
y = sdata[tid ^ 1];
```

> 教训：**永远不要依赖"warp 天然同步"来省略同步原语。** 这是新手最容易踩、且最难
> 复现的坑之一。

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

