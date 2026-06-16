# 04 Scan 与 Histogram

## 1. Scan

输入：

```text
[a, b, c, d]
```

Inclusive scan：

```text
[a, a+b, a+b+c, a+b+c+d]
```

Exclusive scan：

```text
[0, a, a+b, a+b+c]
```

Scan 是 compact、排序、队列分配和许多并行算法的基础。

## 2. Hillis-Steele

每轮距离翻倍：

```text
offset = 1, 2, 4, ...
```

深度 `O(log n)`，但总工作量 `O(n log n)`。

优点：容易理解。
缺点：不是 work-efficient。

具体看 8 个元素一轮轮怎么算（每个 `+` 是一次加法）：

```text
初始:   a  b  c  d  e  f  g  h
off=1:  a a+b b+c c+d d+e e+f f+g g+h     7 次加法
off=2:  各元素再加上其左侧第 2 个         6 次加法
off=4:  各元素再加上其左侧第 4 个         4 次加法
        每个位置现在是它及左侧全部之和（inclusive scan）
```

每一轮几乎所有 `n` 个位置都做一次加法，共 `log2(n)` 轮，所以总工作量是
`n * log n`，而不是串行 scan 的 `n`。

## 3. Blelloch Scan

两阶段：

1. Up-sweep：构造归约树。
2. Down-sweep：传播前缀。

总工作量 `O(n)`，深度 `O(log n)`。

Exclusive scan 通常先将树根置零，再执行 down-sweep。

为什么要费两阶段的劲把工作量从 `n log n` 压到 `n`？因为 GPU scan 通常是
**访存受限**的，"工作量"几乎正比于内存操作次数。给个数字：`n = 1,000,000` 时
`log2(n) ≈ 20`，于是

```text
Hillis-Steele:  约 n * log n = 2000 万 次加法/访存
Blelloch:       约 2n        =  200 万 次（up-sweep + down-sweep 各约 n）
```

相差约 **10 倍工作量**。深度都是 `O(log n)`（关键路径一样短），但 Blelloch
搬的数据少一个数量级，所以在大数组上更快。Hillis-Steele 胜在简单，适合小
tile 或 warp 级 scan；大规模 work-efficient 实现选 Blelloch。这正是"深度相同
不代表性能相同"的典型例子——总工作量同样要算。

## 4. 多 Block Scan

大数组需要：

1. 每 block 扫描自己的 tile。
2. 输出每个 block 的总和。
3. 扫描 block sums。
4. 将 block offset 加回各 tile。

这是典型的分层并行算法。

## 5. Histogram

每个输入值更新一个 bin：

```cpp
atomicAdd(&histogram[value], 1);
```

若分布集中，许多 thread 竞争少数 bin。

## 6. Privatization

为每个 block 创建 shared histogram：

```text
global input
-> block-private shared bins
-> block 内 atomic
-> 合并到 global bins
```

Global atomic 数量从“每个元素一次”降为“每 block 每 bin 至多一次”。

但 shared histogram 也有约束：

- Bin 数是否放得下。
- Shared atomic 竞争。
- 初始化和合并成本。
- 数据分布。

## 7. Warp Aggregation

若一个 warp 多个 lane 更新相同 bin，可使用 match/ballot 分组，让 leader 执行
一次 atomic。

这对高度重复 key 有益，对随机 key 可能增加额外指令。

## 8. 实践

### Scan

对 8 个元素手工完成 Blelloch up-sweep/down-sweep。再实现单 block exclusive
scan，测试：

```text
1, 7, 31, 32, 257 个元素
```

### Histogram

实现：

1. Global atomic。
2. Block-private shared histogram。

测试两种数据：

```text
均匀分布
90% 元素集中在一个 bin
```

比较性能并解释竞争差异。

## 9. 不要重复造工业轮子

学习算法时手写；工程中优先评估：

- CUB scan/reduce/histogram primitive。
- Thrust scan。

手写版本的价值是理解和定制，而不是默认比成熟库快。

## 10. 资料映射

- PMPP：Prefix Sum、Histogram。
- CUB DeviceScan、DeviceHistogram。

