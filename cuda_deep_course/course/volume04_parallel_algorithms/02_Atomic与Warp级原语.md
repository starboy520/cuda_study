# 02 Atomic 与 Warp 级原语

> 本章分两大块：前半讲 **atomic**（多线程安全更新同一地址），后半讲 **warp 级原语**
> （让一个 warp 的 32 个线程直接协作，不经过 shared memory）。两者都是"线程协作"的
> 工具，但粒度不同：atomic 面向任意线程间，warp 原语面向同一 warp 内的 32 条 lane。

## 1. Atomic 解决什么问题

错误：

```cpp
counter[0] += 1;
```

它包含读取、加一、写回。多个 thread 并发执行会丢失更新。

这行代码看起来是"一条语句"，但硬件执行时是**三步**：

```text
① 读 counter[0] 到寄存器
② 寄存器 +1
③ 写回 counter[0]
```

两个线程同时跑这三步，就可能"撞车"：

```text
counter[0] 初始 = 5
线程A 读到 5 ┐
线程B 读到 5 ┘ 两个都读到旧值 5
线程A 写回 6
线程B 写回 6   ← B 用自己手里的 5+1，覆盖了 A 的结果
最终 = 6（错！应该是 7，丢了一次更新）
```

这就是 **race（竞争）**：结果取决于谁先谁后，且几乎总是偏小。

正确：

```cpp
atomicAdd(counter, 1);
```

Atomic（原子操作）保证这个 read-modify-write **不可被其他竞争更新打断**——硬件
让"读-改-写"作为一个**不可分割的整体**完成，其他线程要么看到操作前、要么看到操作后，
绝不会插进中间。

### 1.1 常用的 atomic 函数

CUDA 提供一组 atomic，都遵循"原子地读-改-写一个地址"：

```cpp
atomicAdd(addr, val)        // 原子加（最常用）
atomicSub(addr, val)        // 原子减
atomicMax(addr, val)        // 原子取最大
atomicMin(addr, val)        // 原子取最小
atomicExch(addr, val)       // 原子交换（写入 val，返回旧值）
atomicCAS(addr, cmp, val)   // compare-and-swap：若 *addr==cmp 则写 val（所有原子的基础）
atomicAnd / atomicOr / atomicXor  // 原子位运算
```

- **它们都返回操作前的旧值**——这个旧值很有用（比如用 `atomicAdd` 拿到"我是第几个到的"，做队列分配）。
- `atomicCAS` 是最底层的原语，其他原子都能用它拼出来；需要自定义原子操作时用它。

## 2. Atomic 不等于整个算法同步

Atomic 可以安全更新一个位置，但不自动提供：

- 所有 thread 到达（那是 `__syncthreads()` 的事）。
- 多个普通内存位置的完整事务（atomic 只保护**单个**地址，不能原子地同时改两个地址）。
- 算法阶段 barrier。

同步协议仍要单独设计。一句话区分：

```text
atomic        = 保护"一个地址"的读改写不被打断
__syncthreads = 保证"一个 block 的所有线程"到齐 + 内存可见
两者解决不同问题，不能互相替代。
```

## 3. 竞争

一百万 thread 对同一个 counter 做 atomic，操作会高度串行化。

为什么？因为 atomic 对一个地址的保证是"**同一时刻只有一个 read-modify-write
能进行**"。所以无论你启动多少线程，对**同一个地址**的百万次 atomic 最终都得
**排成一队**逐个完成——并行度坍缩到 1，这块的吞吐和单核没区别。这叫 atomic
contention（原子竞争），是热点计数器最常见的性能杀手。

解法是**分层聚合（privatization）**：把"大家抢一个地址"拆成"先各自在便宜的地方
攒，再少量汇总"。以 1,000,000 个 thread、256 thread/block 为例：

```text
朴素：     1,000,000 次 global atomic 全压在 1 个地址  -> 串行 100 万次

分层：
  每 thread 先在寄存器局部累加              -> 0 次原子争用
  每 block 在 shared memory 上聚合          -> 竞争范围缩到 block 内(更快的片上原子)
  每 block 只发 1 次 global atomic          -> global atomic 从 100 万降到 ~3907 次(block 数)
```

global atomic 的争用从百万级降到几千级，**降了两到三个数量级**。代价只是多写
几行聚合代码。这条"局部攒、少量汇总"的思路贯穿 reduction（第 03 章）和
histogram（第 04 章）——它们本质都是带聚合算子的分层归约。

## 4. Warp Lane 与 Active Mask

从这里进入**warp 级原语**——一组让同一个 warp 的 32 条 lane **直接协作**的指令。
先把基础概念铺清楚。

### 4.1 Lane 是什么

一个 warp 有 32 个线程，**lane = 线程在 warp 内的编号（0–31）**，就像座位号。

```cpp
int lane = threadIdx.x & 31;   // 等价于 threadIdx.x % 32，取低 5 位更快
```

> 注意：`threadIdx.x & 31` 只在一维 block 且按 32 对齐时直接等于 lane。二维 block
> 要先算线性 id 再 `% 32`（见卷一 02A）。

### 4.2 为什么 warp 原语都带 `_sync` 后缀和 mask（关键！）

你会发现所有 warp 原语都长这样：`__shfl_down_sync(mask, ...)`、`__ballot_sync(mask, ...)`。
这个 `_sync` 后缀和第一个参数 `mask` **不是可选装饰，是强制的**，背后有重要原因：

**根因：Volta（2017）之后的"独立线程调度"（Independent Thread Scheduling）。**

```text
老架构（Volta 之前）：
  一个 warp 的 32 条 lane 永远"锁步"执行同一条指令 → 天然同步
  → 老的 __shfl（无 _sync）默认 32 条 lane 都在，能直接交换

Volta 及之后：
  允许同一 warp 的 lane 走到不同位置（独立调度，便于实现更灵活的分支）
  → lane 不再保证锁步！调用 shuffle 时，可能有些 lane 还没到、或已退出
  → 必须显式告诉硬件"哪些 lane 现在参与这次协作" → 这就是 mask
```

所以 **mask 是一个 32 位的位图，第 i 位为 1 表示 lane i 参与本次操作**：

```text
mask = 0xffffffff  → 二进制 32 个 1 → 32 条 lane 全部参与（最常见）
mask = 0x0000000f  → 低 4 位是 1   → 只有 lane 0,1,2,3 参与
```

`_sync` 的含义：这条指令会**先让 mask 里的 lane 同步到此**，再交换数据。这保证
参与的 lane 数据都准备好了，不会读到没算完的值。

> 核心规则：**mask 必须如实反映"此刻哪些 lane 真的会执行到这条指令"**。如果你
> 声明 lane 5 参与（mask 第 5 位为 1），但 lane 5 实际走了别的分支没到，行为未定义。

### 4.3 `__activemask()`：查询当前哪些 lane 活着

```cpp
unsigned mask = __activemask();
```

它返回"此刻这条指令处，本 warp 中哪些 lane 正在一起执行"的位图。常用于你不确定
哪些 lane 存活时（比如在分支里）。但更安全的做法是用 `__ballot_sync` 显式构造 mask
（见第 7 节），而不是依赖 `__activemask` 的运行时结果。

## 5. Vote（投票）

Vote 让一个 warp 的 32 条 lane **就某个条件"投票"**，瞬间得到全 warp 的统计。

```cpp
__all_sync(mask, predicate)     // 所有参与 lane 的 predicate 都为真？→ 返回同一个 bool
__any_sync(mask, predicate)     // 存在任一 lane 的 predicate 为真？→ 返回同一个 bool
__ballot_sync(mask, predicate)  // 把每条 lane 的 predicate 收成一个 32 位 mask
```

具体例子，假设一个 warp 里每条 lane 有个值 `x`，问"有几条 lane 的 x > 0"：

```cpp
unsigned bits = __ballot_sync(0xffffffffU, x > 0);
// bits 的第 i 位 = 1 表示 lane i 的 x>0
int count = __popc(bits);   // __popc 数 1 的个数 = 满足条件的 lane 数
```

```text
假设 8 条 lane 的 x：  3  -1  5  0  2  -4  7  1
predicate (x>0)：      1   0  1  0  1   0  1  1
__ballot_sync 结果：   二进制 ...10110101
__popc(结果) = 5      → 有 5 条 lane 满足
```

用途：
- `__all` / `__any`：快速判断"全体一致"或"存在"，常用于提前退出、分支决策。
- `__ballot`：把条件收成位图，是 **stream compaction（流压缩）**、warp 聚合 atomic 的基础。

> 这些都是**一条指令**完成全 warp 统计，比用 shared memory + 循环快得多。

## 6. Shuffle（lane 间交换寄存器）

Shuffle 是 warp 原语里最重要的——它让 lane 之间**直接读取彼此寄存器的值**，
完全不经过 shared memory：

```cpp
__shfl_sync(mask, value, srcLane)    // 读 srcLane 那条 lane 的 value（广播常用）
__shfl_down_sync(mask, value, delta) // 读"我下面第 delta 条"lane 的 value
__shfl_up_sync(mask, value, delta)   // 读"我上面第 delta 条"lane 的 value
__shfl_xor_sync(mask, value, mask2)  // 读"lane 号 XOR mask2"那条的 value（蝶形交换）
```

### 6.1 为什么 shuffle 比 shared 快

```text
shared 方式交换数据：  写 shared → __syncthreads() → 读 shared   （三步，碰内存+同步）
shuffle 方式：         一条指令，lane 间寄存器直接传                （免内存、免 barrier）
```

### 6.2 用 shuffle 做 warp 归约（最经典用法）

把一个 warp 的 32 个值求和，5 步搞定（log2(32)=5）：

```cpp
for (int offset = 16; offset > 0; offset /= 2) {
  value += __shfl_down_sync(0xffffffffU, value, offset);
}
// 结束后 lane 0 的 value = 整个 warp 32 个值之和
```

### 6.3 手工推演（8 条 lane 看清楚每一步）

用 8 条 lane（真实是 32，原理一样）演示 `__shfl_down_sync` 归约，初始每条 lane 持有自己的值：

```text
初始：  lane0=a lane1=b lane2=c lane3=d lane4=e lane5=f lane6=g lane7=h

offset=4：每条 lane 加上"下面第 4 条"的值
  lane0 += lane4 → a+e
  lane1 += lane5 → b+f
  lane2 += lane6 → c+g
  lane3 += lane7 → d+h
  （lane4-7 的值之后不再用）

offset=2：每条 lane 加上"下面第 2 条"
  lane0 += lane2 → (a+e)+(c+g)
  lane1 += lane3 → (b+f)+(d+h)

offset=1：每条 lane 加上"下面第 1 条"
  lane0 += lane1 → a+b+c+d+e+f+g+h   ← 全部之和落在 lane 0 ✅
```

每一步把"有效值"减半（offset 16→8→4→2→1），最后 **lane 0 持有总和**。这正是
卷四第 03 章 reduction 用 shuffle 收尾的原理。

> 为什么是 `__shfl_down`（往下加）而不是 up？因为我们要把结果汇聚到 lane 0，
> 让每条 lane 把"下方"的值加上来，汇聚方向朝 lane 0。

## 7. Partial Warp（部分 lane 有效时的陷阱）

前面例子都假设 32 条 lane 全部参与（mask=0xffffffff）。但实际中常有"数组尾部不足
32 个元素"的情况，此时**不能盲目用 0xffffffff**——因为有些 lane 越界、不该参与。

错误做法：

```cpp
// ❌ 危险：越界 lane 提前 return，但 mask 还写全 1
if (index >= count) return;
value += __shfl_down_sync(0xffffffffU, value, 16);  // 声明 32 lane 参与，但有些已退出 → 未定义
```

正确做法——先用 `__ballot_sync` 算出"哪些 lane 真的有效"，再用这个 mask：

```cpp
unsigned mask = __ballot_sync(0xffffffffU, index < count);  // 有效 lane 的位图
if (index < count) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(mask, value, offset);  // 用真实 mask
  }
}
```

两个要点：
- **mask 要如实反映存活 lane**（第 4.2 节的核心规则）。
- 读取的 source lane 也必须在参与集合里，否则 shuffle 结果未定义。

## 8. Match 与 Warp Aggregation（进阶，了解即可）

`__match_any_sync(mask, key)` 找出 warp 内**拥有相同 key 的 lane 组**，返回一个 mask
标出"和我 key 相同的 lane"。典型用途是 **warp 聚合 atomic**：

```text
场景：histogram，warp 内多条 lane 想给同一个 bin 加 1
朴素：每条 lane 各发一次 atomicAdd → 同 bin 的争用
聚合：用 match 把"要更新同一 bin"的 lane 分组
      → 每组选一个 leader，leader 一次 atomicAdd(组内人数)
      → atomic 次数大幅减少
```

这是把第 3 节"分层聚合"思想下沉到 **warp 级**。更现代的 libcu++（`cuda::atomic`、
cooperative groups）提供更安全易读的封装。**学 intrinsic 是为了理解底层机制**，
工程中优先用库，关注可读性和 API 演进。

## 9. 怎么选：atomic vs shared vs warp 原语

三种"线程协作"工具，按场景选：

| 工具 | 适用 | 代价 |
|------|------|------|
| **atomic（global）** | 少量、分散的更新；不规则写入 | 热点地址争用严重 |
| **shared memory** | block 内复用、协作、树形归约 | 要 `__syncthreads()`，占 shared |
| **warp 原语（shuffle/vote）** | 同一 warp 内 32 线程协作 | 仅限 warp 内（≤32）|

实战常**组合使用**（如 reduction）：

```text
warp 内    → shuffle 归约（最快，免 barrier）
block 内    → 各 warp 结果写 shared，再归约
跨 block    → 每 block 一次 global atomic 或多阶段 kernel
```

这正是"分层聚合"的完整形态：**warp 级 → block 级 → grid 级**，每一层用最便宜的工具。

## 10. 实践

1. 用 `atomicAdd` 实现全局计数，再改成"每 block shared counter + 每 block 一次 global atomic"，对比耗时。
2. 用 `__ballot_sync` + `__popc` 统计一个 warp 内 predicate 为真的 lane 数。
3. 手工推演 8 个 lane 的 shuffle-down reduction（对照第 6.3 节验证）。
4. 实现一个**单 warp 归约**（32 线程，纯 shuffle），和 shared 版对比。
5. 处理非 32 整除的尾部：用 `__ballot_sync` 构造 mask 做 partial warp 归约。

## 11. 资料映射

- CUDA Programming Guide：Atomic Functions、Warp Vote/Match/Shuffle Functions。
- CUDA C++ Best Practices Guide：Atomics、Warp-level Primitives。
- PMPP：Histogram、Reduction 与线程协作。
- 配套：[卷四第 03 章 Reduction](03_Reduction从错误到优化.md)（shuffle 收尾的完整应用）。

