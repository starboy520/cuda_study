# 04 Scan 与 Histogram

## 0. 本章两条主线

本章讲两个看似无关、其实都靠"分层并行"解决的经典问题：

```text
Scan（前缀和）   把 [a,b,c,d] 变成 [a, a+b, a+b+c, ...] —— 每个位置 = 它及左边之和
Histogram（直方图）统计每个值出现多少次 —— 本质是"很多线程往少数计数器加"
```

它们的共同难点都是**协作**：scan 里每个结果依赖前面所有元素，histogram 里很多线程
争同一个计数器。解法也同源——**先在小范围（block/warp）算好，再分层汇总**，这正是第
03 章 reduction、第 02 章 atomic 反复出现的套路。

> 学习顺序建议（对应 Week3 Day4-5）：先吃透**单 block Hillis-Steele scan**（§2），
> 再扩到**多 block scan**（§4），最后看 **histogram + privatization**（§5-§7）。

## 0.1 术语速查表

| 术语 | 一句话定义 |
|---|---|
| **inclusive scan** | 每个位置 = 它自己 + 左边所有，`[a, a+b, a+b+c]` |
| **exclusive scan** | 每个位置 = 左边所有（不含自己），`[0, a, a+b]` |
| **Hillis-Steele** | 每轮距离翻倍的 scan，简单但工作量 `O(n log n)` |
| **Blelloch** | up-sweep + down-sweep 两阶段，工作量 `O(n)`，更省 |
| **work-efficient** | 总工作量和串行一样是 `O(n)`，没有多做无用功 |
| **privatization** | 每 block 先在 shared 私有副本上聚合，再合并到 global |
| **warp aggregation** | 一个 warp 内先合并对同一 bin 的更新，再发一次 atomic |

## 1. Scan：定义与两种形态

输入：

```text
[a, b, c, d]
```

**Inclusive scan**（含自己）：

```text
[a, a+b, a+b+c, a+b+c+d]
```

**Exclusive scan**（不含自己，前面补一个单位元 0）：

```text
[0, a, a+b, a+b+c]
```

两者可**互相转换**，这点 Week3 Day4 会用到：

```text
inclusive -> exclusive：整体右移一位，最左补 0
exclusive -> inclusive：每个位置再加上原数组对应元素
例：原数组 [3,1,2,4]
  inclusive = [3,4,6,10]
  exclusive = [0,3,4,6]   （inclusive 右移补 0）
```

为什么需要 exclusive？因为很多算法（尤其 **stream compaction 流压缩**）要的是"我前面有
几个元素"——这正好是 exclusive scan 的输出，即"我的写入位置"。

> Scan 是 compact、排序、资源/队列分配等大量并行算法的基础，用途极广。

## 2. Hillis-Steele：最易懂的并行 scan

每一轮，每个位置都加上"距离它 `offset` 的左邻"，`offset` 每轮翻倍：

```text
offset = 1, 2, 4, ...   直到 >= n
```

深度 `O(log n)`（只需 log 轮），但总工作量 `O(n log n)`（每轮几乎 n 个位置都做加法）。

### 2.1 逐轮手推（8 元素）

```text
初始:   a    b    c    d    e    f    g    h
off=1:  a   a+b  b+c  c+d  d+e  e+f  f+g  g+h       7 次加法（每个位置加左边第 1 个）
off=2:  a  a+b a+b+c .. 各位置再加左边第 2 个        6 次加法
off=4:  各位置再加上左边第 4 个                       4 次加法
结果:   每个位置 = 它及左侧全部之和（inclusive scan）
```

每轮约 `n` 次加法、共 `log2(n)` 轮，所以总工作量 `n * log n`，比串行 scan 的 `n` 多。

### 2.2 单 block kernel（Week3 Day4 要手写的就是这个）

```cpp
// 单 block inclusive scan，n <= blockDim.x，结果原地写回 data
__global__ void scanHillisSteele(float* data, int n) {
    extern __shared__ float tmp[];
    int t = threadIdx.x;
    tmp[t] = (t < n) ? data[t] : 0.0f;
    __syncthreads();

    // 每轮 offset 翻倍
    for (int offset = 1; offset < n; offset <<= 1) {
        float add = (t >= offset) ? tmp[t - offset] : 0.0f;  // 先读，避免读写竞争
        __syncthreads();                                     // 确保大家都读完旧值
        tmp[t] += add;                                       // 再写
        __syncthreads();                                     // 确保大家都写完，下轮再读
    }
    if (t < n) data[t] = tmp[t];
}
```

**三个关键点，每个都是坑**：

1. **为什么要两个 `__syncthreads()`？** 因为 `tmp[t] += tmp[t-offset]` 读的是**别的线程**
   写的值。必须"先全员读完旧值 → 再全员写新值 → 再进入下一轮"，否则快的线程会读到本轮
   刚被改过的数据（race）。
2. **为什么先存 `add` 再加？** 同一个 `tmp[t-offset]` 可能正被它的拥有者改写。先把要加的
   值读进寄存器、`__syncthreads()` 之后再写回，避免"边读边写"。
3. **`t >= offset` 边界**：最左边 `offset` 个位置没有"左邻"，跳过加法（加 0）。

> 一句话：Hillis-Steele 用"每轮翻倍 + 两道 barrier"换来简单。它**不是 work-efficient**
> （多做了 log 倍加法），但单 block / warp 级 scan 用它最省心。

## 3. Blelloch Scan（work-efficient）

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

## 4. 多 Block Scan：单 block 装不下时怎么办

§2 的 kernel 只能处理 `n <= blockDim.x`（一个 block 内）。可数组有上百万元素、跨很多
block 时，麻烦来了：**block 之间不能用 `__syncthreads()` 同步**（它只在 block 内有效），
而每个元素的前缀又依赖**前面所有 block** 的总和。解法是经典的**三趟分层 scan**：

```text
第1趟  每个 block 内部各自 scan 自己的 tile（用 §2 的方法）
       同时记下每个 block 的"总和"（tile 最后一个 inclusive 值）-> 存进 blockSums[]
              block0        block1        block2
       tile:  [3 1 2 4]     [2 5 1 1]     [4 2 ...]
       局部scan:[3 4 6 10]   [2 7 8 9]     [4 6 ...]
       blockSums: 10           9            ...

第2趟  对 blockSums[] 再做一次 scan（通常是 exclusive），得到每个 block 的"起始偏移"
       blockSums:  [10, 9, ...]
       exclusive:  [0, 10, 19, ...]   <- block_i 之前所有 block 的总和

第3趟  把每个 block 的偏移加回它 tile 里的每个元素
       block1 每个元素 += 10：[2 7 8 9] -> [12 17 18 19]
       于是跨 block 的全局前缀和就对上了
```

为什么这样就对？因为"全局前缀和 = block 内局部前缀和 + 我这个 block 前面所有 block 的
总和"。第 1 趟算前半项，第 2 趟算后半项（对 block 总和再 scan），第 3 趟把两者相加。

```text
全局 scan[i] = 局部 scan(i 在本 block 内) + 本 block 之前所有 block 的总和
                └── 第1趟 ──┘              └────── 第2趟得到、第3趟加回 ──────┘
```

这是**分层并行的典型范式**：大问题切块 → 块内解决 → 块间再解决一层 → 合并。reduction
的多阶段（第 03 章）、histogram 的 privatization（§6）本质都是它。

> Week3 Day5 的多 block scan 就是实现这三趟。注意第 2 趟若 block 数也超过一个 block，
> 还要递归再分层——初学先保证"block 数 ≤ 一个 block 能 scan 的量"即可。

## 5. Histogram

直方图就是"统计每个值落在哪个 bin、各有多少个"。最直接的写法是每个线程对它那个值的
bin 做一次 atomic：

```cpp
// naive：全局 atomic 版
__global__ void histGlobal(const int* in, int* hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&hist[in[i]], 1);   // 直接怼 global 的 bin
}
```

问题在于**竞争**（回忆第 02 章）：如果数据分布集中（大量值落进少数几个 bin），成千上万
线程就**抢同一个 global 地址**，atomic 被串行化，慢得惊人。分布越集中越惨。

## 6. Privatization：把竞争关进 block

思路和第 02 章分层聚合、第 03 章 reduction 完全一致——**先在便宜的 shared 上各自攒，
再少量汇总到 global**：

```cpp
// privatization：每 block 先在 shared 建私有直方图
__global__ void histShared(const int* in, int* hist, int n, int bins) {
    extern __shared__ int local[];                       // bins 个计数器
    for (int b = threadIdx.x; b < bins; b += blockDim.x) // 1) 清零私有直方图
        local[b] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&local[in[i]], 1);              // 2) 在 shared 上 atomic（快得多）
    __syncthreads();

    for (int b = threadIdx.x; b < bins; b += blockDim.x) // 3) 合并回 global
        atomicAdd(&hist[b], local[b]);                   //    每 block 每 bin 只回写一次
}
```

效果量化：

```text
global atomic 次数：  从"每个元素一次"(n 次)  ->  "每 block 每 bin 至多一次"(blocks × bins)
shared atomic：       竞争范围缩到 block 内，且 shared atomic 比 global 快很多
```

但 privatization 不是免费的，有约束：

- **bin 数要放得进 shared**（bin 太多放不下，得分块或退回 global）。
- shared atomic 仍有竞争（只是范围小）。
- 多了清零和合并两步成本。
- 收益大小取决于**数据分布**：越集中，naive 越惨、privatization 救得越多。

> 一句话：histogram 的 privatization 和 reduction、atomic 聚合是**同一招**——"局部攒、
> 少量汇总"。Week3 Day5 要对比的就是 `histGlobal` vs `histShared` 在集中分布下的差距。

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

## 11. 面试题（附参考答案）

**Q1：inclusive 和 exclusive scan 有什么区别？怎么互转？**
inclusive 每个位置含自己（`[a, a+b, a+b+c]`），exclusive 不含自己、前面补单位元 0
（`[0, a, a+b]`）。inclusive 右移一位补 0 得 exclusive；exclusive 每位再加原数组对应元素
得 inclusive。

**Q2：Hillis-Steele 和 Blelloch 哪个快，为什么？**
深度都是 `O(log n)`，但 Hillis-Steele 工作量 `O(n log n)`、Blelloch `O(n)`。scan 多为
访存受限，工作量≈访存次数，所以大数组上 work-efficient 的 Blelloch 更快（n=100万 时约
10 倍工作量差）。Hillis-Steele 胜在简单，适合小 tile / warp 级。

**Q3：单 block scan 里为什么每轮要两次 `__syncthreads()`？**
因为 `tmp[t] += tmp[t-offset]` 读的是别的线程写的值。必须"全员读完旧值 → 全员写新值 →
再进下一轮"，两道 barrier 分别保证读写不交叉、和轮次之间不串。少一道就会 race。

**Q4：数组太大、跨多个 block 时 scan 怎么做？**
三趟分层：① 每 block 内部 scan 并记录 block 总和；② 对 block 总和再 scan 得每 block 偏移；
③ 把偏移加回各 block 元素。因为 `__syncthreads()` 不能跨 block，只能用这种分层 + 多次
launch 的方式跨 block 同步。

**Q5：histogram 为什么会慢，privatization 怎么救？**
数据集中时大量线程对同一个 global bin 做 atomic，被串行化。privatization 让每 block 先在
shared 私有直方图上 atomic（范围小、shared 快），最后每 bin 只向 global 回写一次，把 global
atomic 从"每元素一次"降到"每 block 每 bin 一次"。

**Q6：scan 和 reduction、histogram 有什么共同点？**
都是**分层并行**：大问题切块 → 块内（block/warp）先解决 → 块间再汇总一层 → 合并。reduction
是"压成一个值"，scan 是"算前缀"，histogram 是"分桶计数"，但都靠"局部先算、少量汇总"避免
全局竞争或长依赖。

