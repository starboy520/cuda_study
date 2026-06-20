# Week 3 学习笔记

> 主题：并行模式与同步 —— Atomic、Warp 原语、Scan、Histogram、Stream

---

## Day 1：Atomic 与竞争

### 核心概念

- **race（竞争）**：`counter++` 是"读→改→写"三步，多线程会丢更新；`atomicAdd` 让三步不可分割。
- **atomic contention（原子竞争）**：百万线程抢同一地址 → 被串行化，并行度坍缩到 1，是热点计数器的性能杀手。
- **分层聚合（privatization）**：把"大家抢一个全局地址"拆成"先各自在便宜的地方攒，再少量汇总"——
  寄存器/shared 局部攒 → 每 block 一次 global atomic。global atomic 次数从"每元素一次"降到"每 block 一次"。

### 作业：atomic_sum

代码：`week3_parallel/atomic_sum/atomic_sum.cu`。实现两版求和并对比：
- `global_atomic_add`：每个线程直接 `atomicAdd(sum, in[idx])` —— 全员争一个地址。
- `shared_atomic_add`：每 block 先在 shared `value` 上 atomic 聚合，block 末尾只发 1 次 global atomic。

shared 版结构（分层聚合标准写法）：
```text
shared value = 0  →  __syncthreads()  →  局部 atomicAdd(&value, ...)
              →  __syncthreads()  →  线程0 atomicAdd(sum, value)
```

结果：两版都 PASS（与 CPU 对照），结构正确。

### 实测结果（N = 1<<24，T4，nvcc -O3 -arch=sm_75）

```text
PASS: sum = 140737479966720
global atomicAdd:  19.83 ms
shared atomicAdd:  22.19 ms   ← 反而更慢!
```

**反直觉发现:shared 分层聚合这里反而更慢。** 原因分析:
- 数据 `in[idx]=idx`，每个线程加的值**各不相同、分散**，global atomic 的竞争没那么致命
  （现代 GPU 硬件 atomic 有优化，分散写入能并行吸收一部分）。
- shared 版多了固定开销：shared 清零 + 两次 `__syncthreads` + 局部 atomic + 汇总。
- **竞争不严重时，这些固定开销 > 省下的竞争 → 分层聚合反而亏。**

**教训（比看到预期结果更有价值）**：
- 分层聚合不是"永远更快"，只在**竞争极端严重**时才划算。
- 优化要用数据验证，不能想当然——这正印证教材"先测量再下结论"。
- 想看到分层聚合碾压 global，要制造"大量线程抢极少数地址"的场景
  —— Day5 histogram 的"90% 集中在一个 bin"就是为暴露这个而设计的。

### 加第三版：完整三层聚合（寄存器 → shared → global）

加了 `register_shared_atomic_add`：每线程先用 grid-stride 把多个元素攒进**寄存器 local**，
再每线程只发 1 次 shared atomic。三版对比：

```text
global atomicAdd:              16.65 ms   基线
shared atomicAdd:              12.52 ms   1.3x
reg+shared(grid-stride):        1.68 ms   ~10x ⭐
```

**为什么 reg+shared 快 ~10 倍**：
- 原 shared 版：每个元素发 1 次 shared atomic → 1600万次 atomic。
- reg+shared：每线程先在**寄存器**攒（0 次 atomic），每线程只发 **1 次** shared atomic
  → atomic 总数从 1600万 砍到几千。
- grid-stride 让每线程处理多个元素，固定开销被摊薄。

**完整三层聚合**（每多一层，atomic 少一个数量级）：
```text
① 寄存器层：grid-stride 多元素攒进 local     → 0 次 atomic（最快）
② shared 层：每线程 1 次 shared atomic        → 片上，快
③ global 层：每 block 1 次 global atomic      → 最少全局争用
```
这就是 reduction 优化（卷四/03）的核心。

> 另注：本次 shared 版（12.5ms）比 global（16.6ms）快，和上一次（shared 更慢）相反
> —— 说明小数据单次测量噪声大、会波动，**要多跑取中位数**（Week5 强调）。

### 重点：grid-stride 模式（务必理解）

**普通模式 vs grid-stride**：
```text
普通："一人一个" —— 线程数必须 = 元素数，grid 写死
grid-stride："一人多个" —— 线程数随便设，每人循环领取，步长 = 总线程数
```

模板：
```cpp
int stride = gridDim.x * blockDim.x;                       // 总线程数 = 步长
for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride) {
    // 处理 in[i]
}
```

**两大好处**：
1. grid 大小灵活：不管 n 多大都能跑对（grid 开小也行，每线程多干活）。
2. 适合分层聚合：每线程先在寄存器攒多个元素，大幅减少 atomic/同步次数。

### 关键反直觉：为什么 grid-stride 比"分段连续"快（合并访问的本质）

我一开始想：跳着访问（in[0], in[1000]...）会不会比"每线程处理连续一段"慢？
**答案相反——grid-stride 才快。** 因为 GPU 的"连续"是横向的：

```text
GPU 看的是：同一拍，一个 warp 的 32 条 lane 访问的地址连不连续（空间连续）
不是 CPU 那种：单个线程连续访问（时间连续）
```

对比同一拍（时刻1）：
```text
grid-stride：  线程0读in[0] 线程1读in[1] ... 线程31读in[31]  地址 0~31 连续 ✓ 合并
分段连续：     线程0读in[0] 线程1读in[1000] ... 线程31读in[31000]  跨步 ✗ 不合并(慢32倍)
```

```text
CPU 优化：让【一个线程】的访问时间上连续（利用 cache line）
GPU 优化：让【一个 warp 的 32 线程】的访问空间上连续（合并访问）
→ "分段连续"是 CPU 思维，在 GPU 上会让 warp 内跨步，反而慢
```

> 一句话：**GPU 的"连续"看一个 warp 的 32 个线程，不看单个线程。** grid-stride 每一拍
> warp 内都连续（合并），所以"跳着访问"反而最快。这是 GPU vs CPU 最反直觉的区别之一。

### 这次 review 的收获（待改进点）

| 问题 | 说明 | 何时补 |
|---|---|---|
| 缺 warmup + 多次取中位数 | N=65536 太小，首次 kernel 含 context/JIT 开销，污染计时 | Week5 性能工程 |
| 数据规模太小 | N=65536 看不出分层聚合优势，差异被噪声淹没；应放大到 1<<24 | 可重测 |
| int 累加会溢出 | 放大 N 后 `sum += i` 溢出，应用 `unsigned long long` 或数据填 1 | 重测时注意 |
| 计时顺序 | `cudaGetLastError` 宜放在 record stop 后；用 `cudaEventSynchronize(stop)` 替 `cudaDeviceSynchronize` | 小问题 |

> Day1 核心目标（写对两版 + 理解分层聚合）已达成。计时严谨性和数据规模问题留到 Week5 深入。

### 自测（口头答出即掌握）

- [x] atomic 为什么保证正确但可能慢？→ 同一地址的 atomic 被串行化（contention）。
- [x] 怎么缓解？→ 分层聚合：寄存器/shared 局部攒，每 block 一次 global atomic。
- [x] shared atomic 为什么比 global 快？→ 片上 SRAM，延迟低一两个数量级；把争用关进 block 内。

---

## Day 2：Warp 级原语（shuffle / vote）

### 核心概念

Warp 内 32 条 lane 可以**不经过 shared memory** 直接交换数据/做统计，这是 warp 原语。两大类:

| 类别 | 代表 | 每条 lane 拿到的值 | 用途 |
|---|---|---|---|
| **shuffle**（洗牌） | `__shfl_down_sync` | **各不相同**（点对点搬数据） | warp 归约、广播、转置 |
| **vote/ballot**（投票） | `__ballot_sync` + `__popc` | **全 warp 一致**（同一个统计量） | 计数、判断、分支聚合 |

> 关键区别:shuffle 是"每条 lane 搬到不同的数"，ballot 是"全 warp 算出同一个数"。

### 作业 1：warp_reduce（shuffle 归约）

代码:`week3_parallel/warp_reduce/warp_reduce.cu`。单 warp 用 5 次 `__shfl_down_sync`(偏移 16/8/4/2/1)把 32 个数归约到 lane0。

核心:
```cpp
int lane = threadIdx.x & 31;
for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
}
if (lane == 0) printf("lane 0 value: %d\n", value);  // 32 个数之和
```

**蝶式归约图解**(8 lane 演示,初值 = lane 号):
```text
offset=4:  lane0+=lane4, lane1+=lane5, lane2+=lane6, lane3+=lane7
offset=2:  lane0+=lane2, lane1+=lane3
offset=1:  lane0+=lane1  →  lane0 = 0+1+...+7 = 28
```
32 lane 同理,多 offset=16、8 两步,共 5 步(log₂32=5)。结果 lane0 = 0+1+...+31 = **496** ✓

**反直觉发现(用 64 元素跑暴露的)**:main 改成 `<<<1,64>>>` 跑 64 个数,打印了**两行**:
```text
lane 0 value: 496    ← warp0 (lane 0~31) 的和 = 0+...+31
lane 0 value: 1520   ← warp1 (lane 32~63) 的和 = 32+...+63
```
原因:`lane == 0` 匹配**每个 warp 的第一条 lane**,64 线程 = 2 个 warp = 打印 2 次。
**这正说明单 warp 归约只是"积木"**——它只能搞定一个 warp 内的 32 个数,跨 warp(>32)
需要第二级归约(Day3 的 block 归约:每个 warp 先归约 → shared 暂存 → 第一个 warp 再归约一次)。

### 作业 2：ballot_sync（warp 投票 / 计数）

代码:`week3_parallel/ballot_sync/ballot_sync.cu`。数一个 32 元素数组里有几个正数。

核心:
```cpp
unsigned int bits = __ballot_sync(0xffffffffU, value > 0);  // 收集"谁>0"成 32 位位图
int count = __popc(bits);                                   // 数位图里有几个 1
if (lane == 0) printf("positive count: %d\n", count);       // = 17
```

**原理**:`__ballot_sync` 把全 warp 每条 lane 的布尔条件(`value>0`)收成一个 32 位整数
(lane i 的条件对应第 i 位),`__popc` 数其中 1 的个数 = 满足条件的 lane 数。结果 = **17** ✓

**关键理解**:`bits` 这个值**32 条 lane 拿到的完全一样**(投票是全 warp 共享的统计量),
所以每条 lane 算出的 count 都是 17。`if(lane==0)` 只是为了不打印 32 遍,**不是只有 lane0 才算对**。
这正是 vote 和 shuffle 的本质区别。

### 踩坑：IDE 误加的内部头文件

ballot_sync.cu 第一次编译报 `fatal error: __clang_cuda_builtin_vars.h: No such file`。
原因:编辑器(clangd)自动补全塞了 `#include <__clang_cuda_builtin_vars.h>`——这是 IDE 内部头文件,
nvcc 编译不了。**看到 `__clang_xxx` 开头的 include 直接删**,`threadIdx`/`blockDim` 这些 nvcc 自带。

### 自测（口头答出即掌握）

- [x] shuffle 和 ballot 的本质区别? → shuffle 每条 lane 拿到不同的数;ballot 全 warp 拿到同一个统计量。
- [x] `__shfl_down_sync` 归约为什么是 5 步? → log₂32 = 5,每步把"有效数据"的跨度减半(蝶式)。
- [x] 64 个数为什么打印两行? → 64 线程 = 2 warp,`lane==0` 匹配每个 warp 首 lane;单 warp 归约管不了跨 warp。
- [x] `__ballot_sync` + `__popc` 怎么计数? → ballot 把条件收成位图,popc 数 1 的个数 = 满足条件的 lane 数。
- [x] 为什么单 warp 归约只是"积木"? → 只能归约 ≤32 个;>32 要第二级(Day3 block 归约)。

---

## Day 3：Reduction 进阶（shuffle 收尾）

### 预习要点（v3「每线程加2个」+ 满血版 grid-stride）

参考：`cuda_deep_course/.../volume04_parallel_algorithms/03_Reduction从错误到优化.md`

**v2 的浪费**：每线程只搬 1 个数进 shared，树循环第一轮 `offset=blockDim/2` 时
**立刻一半线程退场**——加载阶段全员在岗，第一轮却浪费掉一半。

**v3 优化**：写进 shared 前，先在寄存器里把**两个 global 元素加起来**。
```cpp
const int start = blockIdx.x * blockDim.x * 2 + tid;  // ★ *2
float sum = 0.0F;
if (start < count)              sum += input[start];
if (start + blockDim.x < count) sum += input[start + blockDim.x];
values[tid] = sum;             // 进 shared 已是"两个之和"
```

**`*2` 为什么**：每个 block 现在管 512 个数（不是 256），所以下个 block 的起点要按 512 跳。
```text
v2: 起点 = blockIdx*blockDim       (跳256)  block管256个
v3: 起点 = blockIdx*blockDim*2     (跳512)  block管512个
→ blockDim*2 定位 block 起点，+tid 定位 block 内线程
```

**两个关键细节**：
- 第二个取 `start+blockDim` 不是 `start+1`：保证 warp 内两次加载都连续（合并访问）。
- 两个地址**分别 if 判边界**：输入长度未必是 512 的倍数，少判一个会越界读。

**满血版 = grid-stride 攒任意多个**（v3 的极致）：
```cpp
int gid = blockIdx.x*blockDim.x + tid;
int stride = gridDim.x*blockDim.x;        // 总线程数 = 步长
float sum = 0.0F;
for (int i = gid; i < count; i += stride) sum += input[i];  // 寄存器攒多个
values[tid] = sum;
```
步长=总线程数 → 每圈 warp 内仍连续（合并访问）。**这就是 Day1 atomic_sum
reg+shared 版 ~10x 的同款套路**：能在寄存器先攒，就别让数据裸奔到下一层。

```text
            每线程搬   atomic/同步   访存
v2 单加载    1         每元素进树    一半线程第一轮退场
v3 加两个    2         省第一轮      加载即计算
满血grid     任意多    ~每block一次  带宽吃满
```

> 一条主线：寄存器 → shared → global，每多压一层，下一层 atomic/同步少一个数量级。
> reduction 优化 = atomic 优化 = 同一个套路。

### 作业（已完成）✅

代码：`week03_parallel/reduction_sum_full/reduction_sum_full.cu`（8 个版本同台对比）。
block 级两段归约：warp reduce（shuffle）→ shared[warp_id]=partial → __syncthreads()
→ 第一个 warp 再归约 partials。（block 级用第二级归约，**不是** atomicAdd；
atomicAdd 只用于跨 block 的 grid 级。）自己手写的 `reduction_my` 与 `reduction_stride` 等价。

### 实测结果（N=16M=1<<24，全填 1，T4，nvcc -O3 -arch=sm_75）

```text
v1 global atomic         19.72 ms   基线：全员抢一个地址，串行化
v2 shared atomic         17.56 ms   1.1x  分层聚合（竞争不极端，提升有限）
v3 shared tree            1.35 ms   ~15x  树形归约，无 atomic
stride grid+shuffle       0.44 ms   单趟（到 1024 个部分和）
stride two-pass (GPU)     0.44 ms   ~45x  两趟全 GPU，出最终单值
my two-pass (GPU)         0.44 ms   ⭐ 自己手写，与 stride 完全一致
cub DeviceReduce::Sum     0.44 ms   ⭐ 工业库标杆 —— 追平！
shuffle_only(1线程1元素)   0.71 ms   纯 shuffle 反例：反而慢 1.6x
```

### 三个硬核结论（面试素材）

**① 追平 CUB —— 因为撞到带宽天花板**
```text
我手写的 grid-stride 版 = CUB = 0.44 ms，完全持平。
为什么？reduction 是 memory-bound，瓶颈在"读 N 个数"的访存。
当带宽吃满，CUB 的向量化加载也没法再快 —— 物理上限面前人人平等。
```

**② 91% 峰值带宽 —— 量化证明已榨干**
```text
数据量 = 16M × 8B = 128 MB，只读一遍
理论最快 = 128MB / 320GB/s ≈ 0.40 ms
实测 0.44 ms → 达到理论带宽的 ~91%！剩 9% 是启动等固定开销。
```

**③ 纯 shuffle 反而慢（0.71 vs 0.44）—— 证明 grid-stride 的价值**
```text
shuffle_only：一线程一元素 → 16M 线程 + 50万次 atomicAdd
grid-stride： 26万线程各攒 64 个 → atomic 降到几千次
→ 慢的根源不是 shuffle，是"聚合粒度"：没有寄存器层先压扁数据。
→ 这就是工业界都用 grid-stride、不用"一线程一元素"的原因。
```

### 两个关键概念（今天彻底搞懂）

**两个"32"别混淆**：
```text
WARP_SIZE(32) = warp 有 32 条 lane      ← warp"宽度"，决定 shuffle 5 步(log2 32)
MAX_WARPS(32) = block 最多 32 个 warp   ← warp"数量"=1024/32，决定 value[] 大小
数值撞了，含义不同。代码用命名常量区分，自解释。
```

**两趟全 GPU 归约（工业库标准结构）**：
```text
趟1: N(1600万) → gridStride(1024) 个部分和   数据量大，耗时主体
趟2: gridStride(1024) → 1 个最终值           <<<1, block>>> 收尾，几乎免费
第二趟只多 0.005 ms —— 归约是金字塔，越往上数据越少。
关键：同一个 grid-stride kernel 复用两次（grid 灵活，任意规模都能跑）。
全程不回 CPU → 结果可直接喂下一个 kernel（softmax 分母、loss 等）。
```

### warp shuffle 的两个坑（务必记住）

**坑1：尾部不足 32 时，mask 不能盲目用 0xffffffff**
```text
__shfl_*_sync 的 mask 是"点名表"，点名的 lane 必须全部到齐执行该指令。
错误：if(idx>=n) return; + 0xffffffff → 越界 lane 提前退出，点名表撒谎 → 未定义行为
正确：local = (idx<n) ? input[idx] : 0;（不 return，只填 0）→ 32 条全到齐，加 0 无害
口诀：warp 原语面前，宁可空转填 0，也别提前 return。
```

**坑2：shuffle 归约为什么不用 if(tid<offset)？**
```text
shared 树：线程写【公共格子】，多算会污染别人(race) → 必须 if(tid<offset) 让一半退场
shuffle： 每条 lane 只改【自己的寄存器】，多算只污染自己(最后只取 lane0) → 不用判断
→ 数据在私有寄存器 vs 公共 shared，是要不要边界判断的根本原因。
```

> Day3 超额完成：计划只要"加 shuffle 收尾"，实际做了 8 版对比 + 追平 CUB + 91% 带宽分析
> + 纯 shuffle 反例。reduction 这个主题已吃透（能默写生产级 kernel，理解每个设计选择的为什么）。

---

## Day 4-5：Scan + Histogram

### 预习要点（Scan 基础）

参考：`cuda_deep_course/.../volume04_parallel_algorithms/04_Scan与Histogram.md` + GPU Gems 3 Ch.39。

```text
Scan(前缀和) = 每个位置 = 它自己 + 左边所有  （像账本的"余额"列）
  inclusive: [3,1,2,4] → [3,4,6,10]   含自己
  exclusive: [3,1,2,4] → [0,3,4,6]    不含自己(右移补0)，= "我前面有几个"= 写入位置
区别 reduction：reduction 给一个总和，scan 给沿途每一站的累计。
```

**Hillis-Steele 算法**：每轮 `tmp[i] += tmp[i-offset]`，offset = 1,2,4...翻倍，log₂n 轮。
```cpp
for (int offset = 1; offset < n; offset <<= 1) {
    float add = (t >= offset) ? tmp[t-offset] : 0;  // 先读旧值
    __syncthreads();                                // 全员读完
    tmp[t] += add;                                  // 再写
    __syncthreads();                                // 全员写完，下轮再读
}
```
两个 `__syncthreads()` 缺一不可：读的是别人写的值，必须"全读旧→全写新→下轮"，否则 race。

### Hillis-Steele 正确性证明（两个视角）

**① 归纳法（标准）**：不变量 = "第 k 轮后 tmp[i] 覆盖左边 2^k 个元素的和"。
```text
基础: k=0，tmp[i]=a[i]，覆盖 1 个 ✓
归纳: 第 k+1 轮 tmp[i] += tmp[i-2^k]，把【两段相邻的 2^k 区间】严丝合缝拼成 2^(k+1)
      关键：第二项右端=i-2^k，第一项左端=i-2^k+1，差1接上 → 不重不漏 ✓
终止: log₂n 轮后覆盖 [0,i] 全部 = inclusive scan ∎
灵魂：两段等长、首尾相接的区间拼成双倍长区间。
```

**② 二进制视角（更优雅，一句话点破）**：
```text
任意距离 d 都有【唯一】的二进制分解(如 11=8+2+1)，
而 offset 恰好遍历所有 2 的幂(1,2,4,8...)。
→ a[j] 沿距离 d 的二进制"1"位逐跳到 i，路径唯一 → 不重不漏。
顺带解释：为什么 log₂n 轮(二进制最多 log₂n 位)、为什么 offset 翻倍(二进制每位权重翻倍)。
```

> 证明"用旧值"这个前提，正是代码两个 `__syncthreads()` 的来源——数学和代码咬合。

### 已完成的 scan 变体（代码：week03_parallel/scan/scan.cu，全 PASS）

```text
1. scanHillisSteele          单block shared inclusive，N≤1024     末值8128
2. scanHillisSteeleExclusive 右移补0(exclusive)                   末值8001
3. 单warp __shfl_up_sync      纯寄存器，仅32个                     末值496
4. scanBlockTwoStage         block级两阶段，128元素跨4warp ✓       末值8128 ⭐
```

### block 级两阶段 scan（warp shuffle + shared，突破 32→1024）

把"单 warp scan(32)"升级成"block scan(1024)"，四步（和 Day3 reduction 两阶段同源）：
```text
① warp 内 __shfl_up_sync inclusive scan（每个 warp 各自 scan 自己的 32 个）
② 每个 warp 的总和(lane31 的值)存进 shared warpSum[wid]
③ 第一个 warp 对 warpSum 做 scan → 每个 warp 的"前面所有 warp 的和"= 偏移
④ 每个元素 += warpSum[wid-1]（它所在 warp 之前所有 warp 的总和）
对照 reduction：reduction 是 warp归约→shared→第一warp再归约；
scan 多了第④步"加回偏移"，因为 scan 要每个位置都对，不只一个总和。
```

### ⚠️ 这一版踩的 6 个坑（并行编程核心难点，务必记住）

```text
1. __syncthreads() 不能放在 if(部分线程) 分支里 → 死锁
   (它要求 block 内所有线程都到达；部分线程不进分支就永久卡死)
   → 第③步改用 warp shuffle(warp 内 lock-step 天然同步，不需 syncthreads)

2. shuffle 不能边读边写 shared：__shfl_up_sync(..., warpSum[lane], ...) + warpSum[lane]+=
   → 上一轮写、这一轮读同一 shared 地址 → race
   → 先读进【寄存器 s】，在 s 上 shuffle，最后再写回 shared

3. warp 总和在 lane==31，不是 lane==0
   (inclusive scan 后，最后一个 lane 才是全部 32 个的和)

4. 边界用【实际】warp 数 num_warps=blockDim/32，不是【上限】MAX_WARPS=32
   (MAX_WARPS 只用于 __shared__ 数组声明；读取/循环边界要用实际 num_warps，
    否则 lane 读到未初始化的 warpSum[垃圾]，被 scan 进去)

5. 跨 warp 读 shared 前必须 __syncthreads()
   (warp0 在改 warpSum，warp1/2/3 不等它改完就读 → race)

6. 中间不要重复写回 data（只在最后加完偏移后写一次）
```

### 关键认知

```text
shuffle 管 warp 内(32)，shared 管 warp 间 —— 分层 scan
你写的 scanBlockTwoStage = CUB BlockScan 的核心思想(warp scan→shared→scan→加回)
单 warp scan 只是积木；block scan 把 32 个 warp 的积木拼起来 → 1024
```

### grid 级多 block scan（三趟分层，突破 1024 → 任意大小）

block scan 只能 ≤1024（一个 block）。跨多 block 时 **`__syncthreads()` 不能跨 block**，
靠 **kernel 边界**做全局同步。三趟（GPU Gems 3 §39.2.4 经典算法）：
```text
趟1 scanThreeStage<<<grid, block>>>(data, blockSum)
    每个 block scan 自己的 tile（复用 block 两阶段）+ 输出 block 总和 → blockSum[blockIdx]
趟2 scanThreeStage<<<1, grid>>>(blockSum, nullptr)
    对 blockSum 这些"块总和"再 inclusive scan（复用同一 kernel！grid≤1024 即可）
趟3 addBlockSum<<<grid, block>>>(data, blockSum)
    每个元素 += blockSum[blockIdx-1]（它前面所有 block 的总和），block0 加 0
```
实测：M=2048，8 个 block，数据 i%7 → **PASS（末值 6138）** ✓

**核心认知：跨 block 同步只能靠 kernel 边界**
```text
__syncthreads() 管 block 内；kernel 结束（再启动下一个）= block 间唯一的全局屏障
趟3 要的偏移来自趟2，趟2 要等趟1【所有 block】跑完 → 必须分成 3 个 kernel
(高级的 single-pass / decoupled look-back 能一个 kernel 搞定，但极复杂，CUB 用)
```

**这一版又踩 2 个坑：**
```text
7. 从 block 版改造时漏了写回 data[idx]=val
   → 只算了 block 总和，但 data 还是原始值 → 趟3 给原始值加偏移，全错
   → 教训：改造代码时原有核心逻辑（写回）不能丢
8. 趟2 自我覆盖：scanThreeStage(blockSum, ..., blockSum) 让输入和"块总和输出"同一块
   → kernel 结尾 blockSum[0]=val 污染了 scan 结果
   → 解法：blockSum 加 nullptr 保护（if(blockSum && ...) 才写），趟2 传 nullptr
```

### scan 完整阶梯（全部 PASS，week03_parallel/scan/scan.cu）

```text
32        单 warp __shfl_up_sync                              496
  ↓
1024      block 两阶段（warp shuffle + shared）               8128
  ↓
任意大小   grid 三趟（block scan → scan 总和 → 加回偏移）       6138(M=2048,8block)
分层主线：warp 原语(32) → shared(block内) → kernel 边界(block间)
= 你亲手实现了 CUB DeviceScan 的完整结构
```

### 1M 实测 + 三趟的上限（重要）

```text
1M(2^20) 元素，grid=1024 × block=1024（正好顶格）：PASS
  T4:   0.105 ms   (320 GB/s)
  A100: 0.028 ms   (~1555 GB/s)
  加速比 = 0.105/0.028 = 3.75x
  比 CPU 串行(1~2ms)快 10~20x；约 60% 带宽利用
  (scan 要多次读写 data，带宽利用不如 reduction 的 91% 极致)
```

**T4 vs A100：为什么是 3.75x，不是理论 5x？**
```text
理论带宽比 1555/320 ≈ 4.9x，实测只有 3.75x。
原因(memory-bound 小数据的典型现象)：
  ① 数据小(4MB)：A100 还没"热身"就跑完，固定开销占比大
  ② 三次 kernel launch：每次几μs，0.028ms=28μs 里就占了好几μs，不随带宽缩短
  ③ 趟2 单block scan 1024 个：数据极小，A100 的 108 SM 大部分闲着
→ 想榨满 A100 → 加大数据量让计算盖过启动开销；或跑 reduction(数据大、kernel少，更接近5x)
```

**关键区分：1024 是【硬件】限制，1M 是【算法】限制**
```text
单 block ≤ 1024 线程：所有 NVIDIA 卡通用硬限制(T4/A100/H100 都一样)
  → A100 强在 SM 更多(40→108)、带宽更大(5x)、shared 更大(64→164KB)，【不是单block更大】
  → 换卡能解决"速度"，解决不了"单block容量"
三趟版上限 1M：是你的算法设计(趟2 单block)决定的
  → 换卡也是 1M(A100 算得快但装不下更多)，突破靠递归
→ 工程判断：分清"瓶颈是硬件(换卡有用) 还是 算法(改代码才行)"
```

**三趟版的上限 = 1024 block × 1024 元素 = 100 万**：
```text
瓶颈在趟2：用【1 个 block】scan 所有 blockSum → grid 必须 ≤ 1024
超过 100 万怎么办？→ 递归：对 blockSum 再做一次完整三趟
  层数 vs 容量：1层(block)=1024，2层(现在)=1024²=1M，3层=1024³=10亿
工程：1M 以内用这版；更大直接用 CUB DeviceScan(内部自动多级递归)
```

> 面试金句：手写 grid 三趟 scan，100 万元素 0.1ms、约 60% 带宽；瓶颈是 scan 固有的
> 多次 data 读写，CUB 用 single-pass(decoupled look-back)减少读写能更快。

### 作业（scan 已完成）✅，histogram 待做

histogram(global atomic vs shared privatization)，测均匀 vs 90% 集中分布。（待补）

---

## Day 6：Stream / Event 重叠

（待填）

---

## Day 7：复盘 + 性能表

（待填）
