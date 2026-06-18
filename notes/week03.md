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

（待填）

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
