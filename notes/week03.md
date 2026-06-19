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

（待填）

---

## Day 4-5：Scan + Histogram

（待填）

---

## Day 6：Stream / Event 重叠

（待填）

---

## Day 7：复盘 + 性能表

（待填）
