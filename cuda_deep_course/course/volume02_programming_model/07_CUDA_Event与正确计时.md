# 07 CUDA Event 与正确计时

## 1. 最常见错误

```cpp
auto start = now();
kernel<<<...>>>();
auto stop = now();
```

这通常只测到 Host 提交 launch 的时间，因为 kernel 异步执行。

回到卷二第 05 章的传送带模型就明白了：`kernel<<<...>>>()` 只是把任务**放上
传送带**就立刻返回，GPU 可能还没开始算。所以 `stop - start` 量的是"Host 把
任务塞进队列花了多久"——通常只有几微秒，**和 kernel 真正算了多久毫无关系**。
你可能因此得到一个快得离谱的假数字，误以为 kernel 极快。

要测对，核心只有一条原则：**计时区间必须包含一个"等 GPU 干完"的等待点**。
下面三节就是给这个等待点的三种放法。

```text
错误：  start ── launch(立刻返回) ── stop     量到的是"提交耗时"
正确：  start ── launch ── 等GPU算完 ── stop   量到的才是"执行耗时"
```

## 2. CPU Timer + Synchronize

```cpp
auto start = now();
kernel<<<...>>>();
CUDA_CHECK(cudaDeviceSynchronize());
auto stop = now();
```

测到 Host 观察的 launch + 等待时间，适合端到端某段。

## 3. CUDA Event

```cpp
cudaEventRecord(start);
kernel<<<...>>>();
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

Event 插入 stream 时间线，适合 GPU interval。

它和"CPU timer + synchronize"的本质区别在于**时间戳由谁打**。CPU timer 在
Host 这一端读时钟，量的是 Host 视角；而 `cudaEventRecord` 是往传送带上插一个
**标记**，时间戳由 **GPU 在执行到该标记时**亲自盖上。于是 `start` 和 `stop`
两个标记夹住的，正好是 GPU 端这段工作的真实墙钟时间，不掺杂 Host 的 launch
开销或调度抖动。

注意顺序：`cudaEventRecord(stop)` 之后必须 `cudaEventSynchronize(stop)`——因为
record 同样是异步的（只是把标记入队），Host 必须等 GPU 真正执行到 `stop` 标记、
时间戳落定，`cudaEventElapsedTime` 读出来的差值才有意义。`event_timing.cu` 正是
用这一套量出 kernel 的毫秒级时间，与 launch-only 的微秒数字形成鲜明对比。

## 4. Event 的 Stream

```cpp
cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>();
cudaEventRecord(stop, stream);
```

Start、work、stop 应处于你想测的依赖关系中。

## 5. Warmup

首次运行可能包含 context、module、JIT、cache 和频率状态变化。稳定测量：

```text
warmup
-> 多次 measured iteration
-> 中位数或总时间/次数
```

首次延迟如果是业务目标，则单独报告。

## 6. Sample

[`event_timing.cu`](../../labs/02_programming_model/event_timing/event_timing.cu)

```bash
make -C labs/02_programming_model/event_timing clean all
./labs/02_programming_model/event_timing/event_timing
```

输出对比：

```text
CPU launch-only 微秒
CUDA Event kernel 毫秒
CPU + synchronize 毫秒
```

Launch-only 数字明显不能代表 kernel 时间。

## 7. 测什么

明确区分：

- Kernel-only。
- Kernel sequence。
- H2D + kernel + D2H。
- First request。
- Steady-state request。
- Throughput under concurrency。

“程序耗时”没有唯一含义。

## 8. 短 Kernel

单个几微秒 kernel 的计时噪声大，可循环多次：

```cpp
record(start);
for (...) kernel<<<...>>>();
record(stop);
```

总时间除以次数。注意循环是否改变 cache 和数据依赖。

## 9. 练习

1. 将 measured iteration 改为 1、10、100。
2. 比较 first-run 与 warmed-run。
3. 将 memcpy 放入 event 区间，解释变化。

## 10. 面试题

- 为什么普通 CPU timer 可能只测到 launch？
- Event 是否自动同步整个 Device？
- Warmup 解决什么问题？
- Kernel-only 与端到端时间哪个更重要？

