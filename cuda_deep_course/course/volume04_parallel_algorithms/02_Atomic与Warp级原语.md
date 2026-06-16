# 02 Atomic 与 Warp 级原语

## 1. Atomic 解决什么问题

错误：

```cpp
counter[0] += 1;
```

它包含读取、加一、写回。多个 thread 并发执行会丢失更新。

正确：

```cpp
atomicAdd(counter, 1);
```

Atomic 保证该 read-modify-write 操作不可被其他竞争更新打断。

## 2. Atomic 不等于整个算法同步

Atomic 可以安全更新一个位置，但不自动提供：

- 所有 thread 到达。
- 多个普通内存位置的完整事务。
- 算法阶段 barrier。

同步协议仍要单独设计。

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

Warp 内 lane 编号：

```cpp
int lane = threadIdx.x & 31;
```

参与当前控制流路径的 lane 集合可由：

```cpp
unsigned mask = __activemask();
```

得到。

同步 warp intrinsic 的 mask 表示哪些未退出 thread 参与。参与者必须使用一致
且合法的 mask。

## 5. Vote

```cpp
__all_sync(mask, predicate)
__any_sync(mask, predicate)
__ballot_sync(mask, predicate)
```

用途：

- 判断所有 lane 是否满足条件。
- 判断是否存在满足条件的 lane。
- 将 predicate 转成 bit mask。
- 实现 compact、warp-aggregated atomic 等。

## 6. Shuffle

```cpp
__shfl_sync(mask, value, sourceLane)
__shfl_down_sync(mask, value, delta)
__shfl_up_sync(mask, value, delta)
__shfl_xor_sync(mask, value, laneMask)
```

Shuffle 让 warp lane 交换寄存器值，无需经过 shared memory。

归约：

```cpp
for (int offset = 16; offset > 0; offset /= 2) {
  value += __shfl_down_sync(mask, value, offset);
}
```

最后 lane 0 得到 warp 总和。

## 7. Partial Warp

若只有部分 lane 有效，不能盲目使用 `0xffffffff` 并让无效 lane 提前退出。

常见模式：

```cpp
unsigned mask = __ballot_sync(0xffffffffU, index < count);
if (index < count) {
  // 使用 mask 进行后续 shuffle
}
```

还要保证读取的 source lane 属于参与集合，并理解 shuffle 对非参与 lane 的结果
限制。

## 8. Match 与 Warp Aggregation

`__match_any_sync` 可找出 warp 内拥有相同 key 的 lane 组。可以让每组 leader
执行一次 atomic，再广播结果，减少竞争。

更现代的 libcu++ 也提供更安全的 warp group 操作。学习 intrinsic 有助于理解
底层，但工程中应关注可读性和 API 演进。

## 9. 实践

1. 用 `atomicAdd` 实现全局计数。
2. 改成每 block shared counter，最后每 block 一次 global atomic。
3. 用 ballot 统计一个 warp 内 predicate 为真的 lane 数。
4. 手工推演 8 个 lane 的 shuffle-down reduction。

## 10. 资料映射

- CUDA Programming Guide：Atomic Functions、Warp Vote/Match/Shuffle。
- PMPP：Histogram、Reduction 与线程协作。

