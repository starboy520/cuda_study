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

> 表里的词现在看不懂没关系，下面每个都会用生活例子讲一遍，看完再回来对照。

## 0.2 先用生活例子秒懂这两件事（不懂代码也能看）

### Scan = 算"running total（跑动累计）"

想象你有一个**存钱罐记账本**，每天记下今天存了多少：

```text
每天存入：   3    1    2    4
累计余额：   3    4    6   10     <- 这就是 scan！
            ↑    ↑    ↑    ↑
          第1天 第2天 第3天 第4天
          =3   =3+1 =3+1+2 =3+1+2+4
```

**Scan 就是"每个位置 = 它自己 + 它前面所有的和"**——和你银行 App 里看到的"余额"列
一模一样：每一行的余额 = 这笔 + 之前所有笔。就这么简单。

```text
普通求和（reduction）：只要最后那个总数 10
Scan（前缀和）：    要【每一步】的累计 [3, 4, 6, 10]，一个都不能少
```

> 一句话区别：**reduction 给你"一个总和"，scan 给你"沿途每一站的累计"。**

### Histogram = "分拣 + 计数"

想象你在快递站，面前一堆包裹，每个上面写着目的地城市编号（0~9）。你要数
**每个城市各有多少个包裹**：

```text
包裹城市号： 3 1 3 7 3 1 ...
你做的事：   城市3 的格子 +1，城市1 的格子 +1，城市3 再 +1 ...
最后得到：   bin[0]=0  bin[1]=2  bin[3]=3  bin[7]=1 ...   <- 这就是 histogram
            └每个"格子"(bin)记录一个城市出现了几次┘
```

**Histogram（直方图）就是"给每个可能的值准备一个计数格子(bin)，扫一遍数据、
对应格子各加 1"**。统计成绩分布、像素亮度分布、词频，全是这个。

> 难点预告：成千上万个线程同时往"格子"加 1，如果很多线程抢**同一个格子**，就会
> 打架（要排队），这就是后面 §5-§6 要解决的"竞争"。先有个印象即可。

## 1. Scan：定义与两种形态

输入：

```text
[a, b, c, d]
```

**Inclusive scan**（含自己）：

```text
[a, a+b, a+b+c, a+b+c+d]
```

**Exclusive scan**（不含自己，前面补一个 0）：

```text
[0, a, a+b, a+b+c]
```

> 名词解释：上面说的"**单位元**"就是"加了等于没加的那个数"——对加法来说就是 **0**
> （任何数 +0 不变）。所以 exclusive scan 最左边补的就是 0。如果运算换成乘法，
> 单位元就是 1（任何数 ×1 不变）。本章只用加法，记住"单位元 = 0"就够了。

**这两个有什么用、为什么要分两种？**

```text
inclusive（含自己）：常用于"到目前为止的累计总额"，比如账本余额
exclusive（不含自己）：常用于"我前面有几个"="我该往哪个位置写"
```

举个 exclusive 的实际用途——**排座位**。10 个人按组分队，你想知道"我是全场第几个"：

```text
各组人数：   [3,  2,  4,  1]
exclusive：  [0,  3,  5,  9]   <- 第2组的人从第3号位开始坐，第3组从第5号位...
            └每组的"起始座位号"= 我前面所有组的总人数┘
```

这正是后面 §4 多 block scan、以及"流压缩"会用到的：**exclusive scan 的结果
直接就是"我该写到输出数组的第几格"。**

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

> 名词解释：**stream compaction（流压缩）** = "把一堆数据里【符合条件】的挑出来，
> 紧凑地排到一个新数组里"。比如从 `[5,0,3,0,0,8]` 里挑出非 0 的，得到 `[5,3,8]`。
> 难点是"挑出来的每个数该放新数组第几格"——用 exclusive scan 一算就知道。这是 GPU
> 上极常用的操作（过滤、去重、稀疏数据打包都靠它），下面 §1.1 会展开。

> Scan 是 compact（流压缩）、排序、资源/队列分配等大量并行算法的基础，用途极广。
> 你现在不需要会写这些应用，只要知道"scan 是块很基础的积木"就行。

## 1.1 Scan 到底能干嘛：流压缩（看一个完整例子）

很多人学 scan 时最大的困惑是"算前缀和有什么用？"。这里用最经典的应用
**stream compaction（流压缩）** 一次讲清，你就明白 scan 为什么这么重要。

**任务**：从 `[5, 0, 3, 0, 0, 8]` 里把非 0 的挑出来，紧凑排成 `[5, 3, 8]`。

GPU 上每个线程负责一个元素，难点是：**线程 2（值 3）怎么知道自己该写到输出的第几格？**
答案——**先做一次 exclusive scan**：

```text
原数组:        5   0   3   0   0   8
标记(非0为1):  1   0   1   0   0   1     ← 每个线程：我有效吗？有效填1
exclusive scan:0   1   1   2   2   2     ← 这就是"我前面有几个有效元素"=我的写入位置!
                                          
线程0(值5,标记1): 写到 output[0]   ← scan 值=0
线程2(值3,标记1): 写到 output[1]   ← scan 值=1
线程5(值8,标记1): 写到 output[2]   ← scan 值=2
结果: output = [5, 3, 8] ✓
```

```text
看懂了吗？exclusive scan 的输出，正好就是每个有效元素"该坐第几个座位"。
没有 scan，线程之间根本不知道彼此，没法紧凑排列。
→ 这就是 scan 的威力：让海量线程【无需互相通信】就能算出各自的位置。
```

排序（基数排序）、稀疏矩阵压缩、粒子系统的"删除死亡粒子"等等，本质都是这一招。
**所以 scan 是并行编程里和 reduction 并列的两大基础积木之一。**

## 2. Hillis-Steele：最易懂的并行 scan

每一轮，每个位置都加上"距离它 `offset` 的左邻"，`offset` 每轮翻倍：

```text
offset = 1, 2, 4, ...   直到 >= n
```

深度 `O(log n)`（只需 log 轮），但总工作量 `O(n log n)`（每轮几乎 n 个位置都做加法）。

### 2.1 逐轮手推（先看真实数字，再看字母）

**用真实数字走一遍**（8 个数：`[3,1,2,4,1,5,2,3]`，目标 inclusive scan）：

```text
初始:    3   1   2   4   1   5   2   3
off=1:  每个位置 += 左边第 1 个（最左 1 个没有左邻，不变）
         3   4   3   6   5   6   7   5
              ↑3+1 ↑1+2 ↑2+4 ↑4+1 ↑1+5 ↑5+2 ↑2+3

off=2:  每个位置 += 左边第 2 个（最左 2 个没有，不变）
         3   4   6  10   8  12  12  11
                  ↑3+3 ↑4+6 ↑3+5 ↑6+6 ↑5+7 ↑6+5

off=4:  每个位置 += 左边第 4 个（最左 4 个没有，不变）
         3   4   6  10  11  16  18  21
                          ↑8+3 ↑12+4 ↑12+6 ↑11+10
结果:   每个位置 = 它及左侧全部之和 ✓
        验证最后一个: 3+1+2+4+1+5+2+3 = 21 ✓
```

**为什么"翻倍"就能覆盖到最左边？** 这是最妙的地方：

```text
off=1: 每个位置吸收了"自己 + 左1个"   → 覆盖范围 2 个
off=2: 每个位置再吸收"左2个那一坨"     → 覆盖范围 4 个（2+2）
off=4: 再吸收"左4个那一坨"             → 覆盖范围 8 个（4+4）
→ 每轮覆盖范围翻倍：2 → 4 → 8 → ...，log2(n) 轮就覆盖全部 n 个！
```

**抽象版（字母）对照**——理解上面数字后，这个就秒懂：

```text
初始:   a    b    c    d    e    f    g    h
off=1:  a   a+b  b+c  c+d  d+e  e+f  f+g  g+h       7 次加法（每个位置加左边第 1 个）
off=2:  a  a+b a+b+c .. 各位置再加左边第 2 个        6 次加法
off=4:  各位置再加上左边第 4 个                       4 次加法
结果:   每个位置 = 它及左侧全部之和（inclusive scan）
```

每轮约 `n` 次加法、共 `log2(n)` 轮，所以总工作量 `n * log n`，比串行 scan 的 `n` 多。

> 回答你的预习问题：**"工作量 O(n log n) 比串行 O(n) 还多，为什么 GPU 上反而快？"**
> 因为 GPU 有海量线程，每一轮的 n 个加法是**同时**做的（并行），所以墙钟时间只看
> "轮数"= log n 轮 ≈ 20 步；而 CPU 串行要老老实实做 n 次 ≈ 100 万步。
> **GPU 用"多做点总工作量"换"步数极少"**，这正是并行算法的核心权衡。

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
**访存受限**的（名词解释：**访存受限 / memory-bound** = 瓶颈在“读写内存”而不是
“算术”，跟你 Day3 reduction 同理），“工作量”几乎正比于内存操作次数。给个数字：
`n = 1,000,000` 时 `log2(n) ≈ 20`，于是

```text
Hillis-Steele:  约 n * log n = 2000 万 次加法/访存
Blelloch:       约 2n        =  200 万 次（up-sweep + down-sweep 各约 n）
```

相差约 **10 倍工作量**。深度都是 `O(log n)`（关键路径一样短），但 Blelloch
搬的数据少一个数量级，所以在大数组上更快。Hillis-Steele 胜在简单，适合小
tile 或 warp 级 scan；大规模 work-efficient 实现选 Blelloch。这正是"深度相同
不代表性能相同"的典型例子——总工作量同样要算。

> 本周（Day4-5）你只写 Hillis-Steele 就够了，Blelloch 先理解思想、**不要求手写**。
> 知道"有个更省工作量的版本叫 Blelloch，代价是代码更绕"即可。

## 4. 多 Block Scan：单 block 装不下时怎么办

§2 的 kernel 只能处理 `n <= blockDim.x`（一个 block 内）。可数组有上百万元素、跨很多
block 时，麻烦来了：**block 之间不能用 `__syncthreads()` 同步**（它只在 block 内有效），
而每个元素的前缀又依赖**前面所有 block** 的总和。解法是经典的**三趟分层 scan**：

> 名词解释：**tile（瓦片/小块）** = 把大数组切成一段一段，每段交给一个 block 处理，
> "一个 block 负责的那一小段"就叫一个 tile。
>
> 用生活例子：想象 100 万本书要按顺序编号，但一个人（block）一次只搬得动 1024 本。
> 于是把书堆**切成约 1000 摞**，每摞 1024 本，一摞交给一个人。**每一摞就是一个 tile。**
>
> ```text
> 大数组（100 万个数）：[............................................]
> 切成 tile（每块 1024）：[tile0][tile1][tile2]......[tile976]
>                         ↓      ↓      ↓             ↓
>                       block0 block1 block2 ...   block976   ← 一个 block 管一个 tile
> ```
>
> 为什么要切？因为一个 block 最多 1024 线程，shared memory 也只有几十 KB，**装不下
> 100 万个数**。所以必须切成 block 装得下的小块（tile），分给很多 block 并行处理。
> 下面图里每个 block 下面的 `[3 1 2 4]` 就是一个被简化成 4 个元素的 tile（实际是 1024 个）。
>
> > 这个"切块"思想你其实见过：Day3 reduction 的 grid-stride、矩阵乘法的分块，
> > 本质都是"大问题装不下 → 切成小块分给 block"。tile 只是给这个小块起了个名字。

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

> 用生活类比理解"竞争"：`atomicAdd(&hist[v], 1)` 就像"往一个格子里写数，写的时候得
> 先把别人锁在门外"。如果数据**均匀**（大家加的是不同格子），互不打扰，很快；如果数据
> **集中**（90% 都加同一个格子），就变成"一万人排队，一次只能进一个往同一个格子加 1"——
> 彻底串行，慢到爆。**这就是为什么 §8 要专门用"集中分布"的数据来暴露这个问题。**

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

