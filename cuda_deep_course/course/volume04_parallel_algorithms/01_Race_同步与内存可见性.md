# 01 Race、同步与内存可见性

## 1. 什么是 Data Race

当多个执行者并发访问同一位置，至少一个是写，并且缺少正确同步时，就可能
发生 data race。

错误例子：

```cpp
__shared__ int value;
if (threadIdx.x == 0) {
  value = 42;
}
// 缺少同步
if (threadIdx.x == 1) {
  output[0] = value;
}
```

Thread 1 可能在 thread 0 写入前读取。

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

不要依赖“同一 warp 天然 lockstep”来省略必要同步。独立线程调度架构上，
错误假设更容易暴露。

## 5. Memory Fence

Fence 主要约束调用 thread 的内存操作顺序与可见范围，不负责让其他 thread
都到达某个阶段。

常见概念：

```text
__threadfence_block  block 范围
__threadfence        device 范围
__threadfence_system system 范围
```

Fence 不是 barrier。其他 thread 如何知道数据已经准备好，通常还需要 atomic
flag 或其他同步协议。

## 6. Block 之间如何同步

普通 kernel 中，不同 block 无法使用 `__syncthreads()`。

常见方案：

- 拆成多个 kernel，kernel 边界形成全局阶段。
- 使用 atomic 协调特定计数或队列。
- 使用 cooperative launch 和 grid group，前提是满足驻留和启动条件。
- 重新设计算法，避免全局同步。

Reduction 通常采用多阶段 kernel，而不是在一个普通 kernel 中等待所有 block。

## 7. Cooperative Groups

Cooperative Groups 提供显式线程组抽象：

```cpp
namespace cg = cooperative_groups;
cg::thread_block block = cg::this_thread_block();
block.sync();
```

优势是同步对象和范围更明确，也可创建 tiled partition。它不会绕过硬件和
launch 限制。

## 8. Racecheck

```bash
compute-sanitizer --tool racecheck ./program
```

适合检查 shared-memory hazards。Global memory 越界主要使用 memcheck。

工具没有报告不等于算法协议一定正确；仍需理解同步范围。

## 9. 实践

1. 写一个 block 内 producer/consumer 示例，先删除 barrier，再用 racecheck。
2. 将 barrier 放进 thread-dependent 分支，观察行为并恢复。
3. 解释为何 fence 不能替代“所有 thread 等待”。

## 10. 资料映射

- CUDA Programming Guide：Synchronization Primitives、Memory Model。
- Compute Sanitizer：Racecheck。

