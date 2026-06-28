# GPU 占用率与资源关系详解（用我们的 GEMM 数据讲）

> 目的：把 **寄存器 / occupancy / warp / shared memory / block** 这几个总是搞混的概念，
> 用我们 Day 5 在 A100 上亲手测的数字串成一条逻辑。
> 硬件以 **A100 (sm_80)** 为准。

---

## 0. 一张全景图

```text
GPU
 └─ 很多个 SM（A100 有 108 个）         ← 真正干活的核心单元
      └─ 同时驻留多个 block
           └─ 每个 block 有多个 warp
                └─ 每个 warp = 32 个 thread（锁步执行）
                     └─ thread = 最小执行单位，有自己的寄存器
```

记住一句话：**所有资源的争夺都发生在「一个 SM 内部」**。occupancy 讲的就是「一个 SM 被塞得多满」。

---

## 1. 基本单位

| 单位 | 是什么 | 关键点 |
| --- | --- | --- |
| thread | 最小执行单位 | 每个 thread 独占一组寄存器 |
| **warp** | 32 个 thread 捆在一起 | **GPU 的调度单位**；32 个 thread 锁步执行同一条指令（SIMT） |
| block | 一组 thread（你设的 blockDim） | 整个 block 调度到**同一个 SM**；block 内线程共享 shared memory |
| SM | 流多处理器 | 资源池所在地：寄存器、shared memory、warp 槽 |

为什么调度单位是 warp 而不是 thread？因为硬件一次发射一条指令给 32 个 thread 一起做，省调度开销。所以**occupancy 用 warp 数衡量，不用 thread 数**。

---

## 2. 一个 SM 上的四种「资源池」(A100)

这是所有计算的基础，背下来：

| 资源池 | A100 每个 SM 的上限 | 说明 |
| --- | --- | --- |
| 寄存器 | **65536 个**（32-bit） | 所有驻留线程共享这一池 |
| shared memory | **最多 164 KB**（可配置，默认每 block ≤ 48KB，超了要 opt-in） | block 内共享的高速 scratchpad |
| warp 槽 | **64 个 warp**（= 2048 thread） | 一个 SM 最多同时驻留 64 个 warp |
| block 槽 | **32 个 block** | 一个 SM 最多同时驻留 32 个 block |

> 单线程寄存器上限是 **255**。

---

## 3. 一个 block 会「消耗」多少资源

调度器要往 SM 上塞 block，每塞一个，从四个池里各扣一份：

```text
寄存器消耗   = 每block线程数 × 每线程寄存器数
shared 消耗  = 每block的 shared memory 字节数
warp 消耗    = 每block线程数 / 32
block 消耗   = 1 个 block 槽
```

**例（我们的 8x8 配置）**：
```text
blockDim = (BN/TN, BM/TM) = (8, 8) = 64 线程 = 2 warp
每线程寄存器 = 122
→ 一个 block 吃掉：寄存器 64×122 = 7808 个，warp 2 个，shared sa+sb 那几 KB
```

---

## 4. occupancy 到底是什么

```text
occupancy = 一个 SM 上实际活跃的 warp 数 / 64（warp 上限）
```

- **理论 occupancy (theoretical)**：纯按资源池算出来的「最多能塞几个 warp」。
- **实测 occupancy (achieved)**：ncu 跑出来的真实平均值，通常 ≤ 理论（受 grid 大小、尾效应、负载不均影响）。

occupancy 高 = SM 上待命的 warp 多 = **有更多 warp 可以在某个 warp 卡住（等访存）时顶上来**，这就是「藏延迟」。

---

## 5. 谁是「限制者」= 取四个限制的最小值

能塞几个 block，由**最紧的那个资源**决定：

```text
每SM能驻留的block数 = min(
    寄存器限制 = 65536 / (线程数 × 寄存器/线程),
    shared限制 = SM的shared / 每block的shared,
    warp限制   = 64 / (线程数/32),
    block限制  = 32
)
```

然后：
```text
理论 occupancy = (能驻留block数 × 每block warp数) / 64
```

**哪个 min 最小，就说「occupancy 被 XX 限制」。** 在我们的 GEMM 里，限制者是**寄存器**。

---

## 6. 用我们的数据走一遍（核心）

### 8x8 基线（122 寄存器，64 线程/block，2 warp）

```text
寄存器限制 = 65536 / (64 × 122) = 8.39 → 8 个 block
warp限制   = 64 / 2 = 32 个 block
→ 取 min = 8 个 block（被寄存器卡住）
理论 occupancy = 8 × 2 / 64 = 25%
（实际 ncu 因分配粒度/尾效应，achieved ≈ 14.5%）
```

### 8x4 赢家（78 寄存器，128 线程/block，4 warp）

```text
寄存器限制 = 65536 / (128 × 78) = 6.56 → 6 个 block
warp限制   = 64 / 4 = 16 个 block
→ 取 min = 6 个 block
理论 occupancy = 6 × 4 / 64 = 37.5%
（achieved ≈ 30.1%，确实比 8x8 高一倍）
```

### 为什么 8x4 的 occupancy 翻倍？

```text
TN 8→4 → reg_c[TM][TN] 从 [8][8]=64 个累加器 降到 [8][4]=32 个
→ 每线程寄存器 122→78
→ 同样的 65536 寄存器池，能养活更多 warp
→ occupancy 14.5%→30.1%
→ 更多 warp 藏延迟 → SM throughput 56%→66% → 快 18%
```

### 边界：为什么 4x4 128x128 启动失败

```text
blockDim = (128/4, 128/4) = (32,32) = 1024 线程
每线程 72 寄存器
一个 block 要 1024 × 72 = 73728 个寄存器
> 65536（整个 SM 的池子）
→ 连一个 block 都塞不下 → "too many resources requested for launch"
约束：线程数 × 寄存器/线程 ≤ 65536
```

---

## 7. occupancy 不是越高越好（重要）

我们的数据自己证明了这点：

```text
4x4（72 reg, occupancy 更高）= 9981 GFLOPS
8x4（78 reg, occupancy 略低）= 11382 GFLOPS  ← 反而更快
```

为什么？两个原因：

1. **藏延迟够了就行**：occupancy 的唯一作用是「有足够 warp 顶上来藏访存延迟」。一旦够了，再高没收益。
2. **单线程复用 / ILP**：TM/TN 越大，一个线程算的输出越多，寄存器里的数据复用越多、指令级并行越高。4x4 砍太狠，每线程只算 16 个输出，复用不足，丢的比 occupancy 赚的多。

**结论：occupancy 和单线程复用是一对拉扯，要找甜点（本例是 8x4）。**

---

## 8. 各资源之间的「拉扯」关系图

```text
                    每线程寄存器 ↑
                   /            \
       TM/TN ↑ ───              ── occupancy ↓
       (单线程复用↑, ILP↑)          (能驻留的warp少)
                   \            /
                    ↓          ↓
                 快(复用)   慢(藏延迟弱)
                      \      /
                      净效果 = 甜点

  shared memory/block ↑  →  能驻留block↓  →  occupancy↓
  blockDim ↑（大block）  →  block数少，可能喂不饱SM
```

一句话：**寄存器、shared、warp 三个池子谁先用光，谁就是 occupancy 的瓶颈；而 occupancy 又要和单线程复用平衡。**

---

## 9. 常见面试问答（速查）

**Q: occupancy 是什么？**
> 一个 SM 上活跃 warp 数占最大可驻留 warp 数（A100 是 64）的比例。衡量 SM 被塞得多满，决定藏延迟的能力。

**Q: 寄存器和 occupancy 什么关系？**
> 每线程寄存器越多，同样的寄存器池（65536）能养活的 warp 越少，occupancy 越低。我实测 TN 减半让寄存器 122→78，occupancy 14.5%→30%。

**Q: 为什么 100% occupancy 不一定最快？**
> occupancy 只负责藏延迟，够了就行。提 occupancy 往往要砍寄存器/单线程工作量，会牺牲数据复用和 ILP。我实测 4x4 occupancy 更高反而比 8x4 慢，因为复用不够。

**Q: shared memory 怎么影响 occupancy？**
> 每个 block 用的 shared 越多，一个 SM 能放的 block 越少，occupancy 越低。它和寄存器是并列的两个限制，取最紧的。

**Q: warp 为什么是 32？**
> 硬件以 warp 为单位调度和发射指令，32 个 thread 锁步执行同一条指令（SIMT），所以并行度和 occupancy 都按 warp 算。

---

## 10. A100 关键数字速记

```text
每个 SM：
  寄存器     65536 个（单线程 ≤255）
  shared     最多 164 KB（默认每block≤48KB，超了 opt-in）
  warp 槽    64 个（= 2048 线程）
  block 槽   32 个
全卡：108 个 SM
FP32 峰值 ~19.5 TFLOPS，带宽 ~1935 GB/s（80GB PCIe）

核心公式：
  occupancy = 活跃warp / 64
  能驻留block = min(寄存器/ (线程×reg), shared限制, 64/(线程/32), 32)
  约束：线程数 × 寄存器/线程 ≤ 65536
```

> 对照 T4（sm_75）：每 SM 寄存器同样 65536，但只有 **32 warp/SM**、64KB shared、40 个 SM。所以同一个 kernel 在 T4 和 A100 上的 occupancy 上限、瓶颈都可能不同——这也是我们迁移后瓶颈画像变化的原因。
