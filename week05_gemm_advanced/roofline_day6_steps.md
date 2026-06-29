# Day 6 工作表：访存账 + Roofline（自己一步步填）

> 目标：用数据移动量解释「naive → shared → 2D」的瓶颈变化，建 A100 Roofline，判断各版本 memory-bound 还是 compute-bound。
> 用法：每个 Step 有【方法】和【你来算】。先自己推，算完把数字填进空里，再叫我 review。
> 最终产出：`roofline.md`（把本表的结论整理成正式文档）。

---

## 固定前提（A100, FP32, 方阵 M=N=K=2048）

```text
FP32 峰值算力  P  = 19.5 TFLOPS = 19.5e12 FLOP/s
显存带宽       B  = 1935 GB/s   = 1935e9 byte/s
一个 float     = 4 字节
FLOP（三个版本都一样）= 2*M*N*K
tile 参数：shared/2D 用 BM=BN=64
```

---

## 方法论（先看懂这套，再动手填）

### 1. Roofline 是什么

一句话：**把「算力屋顶」和「带宽斜坡」画在一张图上，看你的 kernel 撞到哪个**。

```text
性能(GFLOP/s)
  ^
P |        ________________  ← 算力屋顶（compute roof）= 峰值算力
  |       /
  |      /  ← 带宽斜线 = AI × 带宽
  |     /
  |    /
  +---+------------------------> 算术强度 AI (FLOP/byte)
      AI*（拐点）
  左边：memory-bound（撞带宽斜线）
  右边：compute-bound（撞算力屋顶）
```

可达性能 = `min(P, AI × B)`。
- AI 小 → 受带宽限制（`AI×B` 那条斜线在下面）→ **memory-bound**。
- AI 大 → 受算力限制（`P` 那条平线封顶）→ **compute-bound**。

### 2. 三个核心量

| 量 | 含义 | 怎么算 |
| --- | --- | --- |
| **FLOP** | 总浮点运算数 | GEMM = `2*M*N*K`（乘+加各算 1） |
| **bytes** | 从 **global memory（DRAM）** 实际搬运的字节数 | 数「读了多少元素 × 4」，**要扣掉复用** |
| **AI** | 算术强度 = 每搬 1 字节能算几次 FLOP | `AI = FLOP / bytes` |

> **关键：bytes 指的是 DRAM 流量**，不是 shared/L1。优化（tiling）的本质就是「同样的 FLOP，靠复用把 DRAM bytes 降下来 → AI 升高 → 从带宽斜线往算力屋顶爬」。

### 3. 怎么估一个 kernel 的 global bytes（最难、最常考）

通用方法：**数每个数组从 global 被读了几遍**。
```text
完全不复用（naive）：每次用都从 global 读 → bytes 巨大 → AI 低
有复用（tiling）：    一个元素读进 shared/寄存器后被用 N 次
                     → global 读取次数 ÷N → bytes 降 → AI 升
所以核心是问：「这个数组的每个元素，从 DRAM 被读了几次？」
```

对 GEMM：
```text
A 的一个元素被多少输出用到 → 决定它从 global 读几遍
B 同理
tile 越大，复用越多，global 读取越少。
```

### 4. 判断 bound 的标准动作

```text
1. 算 FLOP（固定）
2. 算 bytes（核心，扣复用）
3. AI = FLOP / bytes
4. 拐点 AI* = P / B
5. AI < AI* → memory-bound；AI > AI* → compute-bound
6. 再对照实测 GFLOPS，看离屋顶/斜线多远 → 找下一步优化方向
```

### 5. Roofline 的盲区（本次重点之一）

```text
Roofline 只看 DRAM 层。
如果两个版本 DRAM bytes 一样（如 shared vs 2D），
Roofline 分不出它们 —— 但它们实测速度可能差很多，
因为瓶颈在 shared/L1/occupancy，不在 DRAM。
→ 这时要用「shared 层访存账」或 ncu（occupancy/throughput）补充。
```

记住这套，下面每个 Step 就是把它套到 naive/shared/2D 上。

---

## Step 0：先把 FLOP 算出来（热身）

【方法】GEMM 每个输出元素做 K 次「乘+加」= 2K FLOP，共 M*N 个输出。

【你来算】
```text
FLOP = 2 * M * N * K = 2 * 2048^3 = _17179869184_____ FLOP
```

---

## Step 1：naive 版本的 global 访存量（最简单，建立感觉）

【方法】naive：每个线程算一个 C[i][j]，在 K 循环里**每一步都从 global 读** A 的一个元素和 B 的一个元素。
```text
每个输出 C[i][j] 要读：A 的第 i 行（K 个）+ B 的第 j 列（K 个）= 2K 个元素
一共 M*N 个输出
→ 总读取元素数 = M*N*2K
→ bytes = M*N*2K * 4
（写 C 的 M*N 远小于读，先忽略）
```

【你来算】
```text
naive bytes = 2*M*N*K * 4 = ___68719476736___ 字节
naive AI    = FLOP / bytes = (2*M*N*K) / (2*M*N*K*4) = ___1/4___ FLOP/byte
```
> 提示：这个 AI 是个很小的常数，和 M/N/K 无关。算出来你会明白 naive 为什么是重度 memory-bound。

---

## Step 2：shared tiling 版本的 global 访存量

【方法】shared tiling：把 A/B 的 tile 搬进 shared memory 复用。
```text
A 的每个元素：被一个 block 搬进 shared 后，复用给该 block 负责的 BN 列
  → A 从 global 总共读 M*K*(N/BN) 个元素
B 的每个元素：复用给 BM 行
  → B 从 global 总共读 K*N*(M/BM) 个元素
总元素 = M*K*(N/BN) + K*N*(M/BM)
```

【你来算】（BM=BN=64）
```text
A 读取元素 = M*K*N/BN = ___16777216___
B 读取元素 = M*N*K/BM = _16777216_____
总元素     = ___16777216*2___
shared bytes = 总元素 * 4 = ___16777216*2*4___ 字节
shared AI    = FLOP / bytes = __16____ FLOP/byte
```
> 提示：和 naive 比，bytes 应该小了约 __64__ 倍（提示：和 BM/BN 有关）。

---

 

【方法】注意这里有个反直觉点：**2D register tiling 在 global/DRAM 层面，读的 tile 和 shared 版本一模一样**（每个 block 还是把 A/B 的 tile 从 global 搬一次进 shared）。register tiling 省的是 **shared memory 的读取次数**，不是 global 的。

【你来想清楚再填】
```text
2D 的 global bytes 和 shared 版本相比：______（相同 / 更少 / 更多）？
2D 的 global AI：______（和 shared 一样 / 不一样）？
```
> 提示：如果你算下来 2D 的 global AI 和 shared 一样，那是对的。
> 那么问题来了：**既然 global AI 没变，2D 为什么比 shared 快？**（这一问写进结论）

---

## Step 4：建 A100 的 Roofline（拐点）

【方法】Roofline 两条线：
```text
水平屋顶 = P（算力峰值）
带宽斜线 = AI * B
真实可达 = min(P, AI*B)
拐点 AI* = P / B  ← 左边 memory-bound，右边 compute-bound
```

【你来算】
```text
拐点 AI* = P / B = 19.5e12 / 1935e9 = __10____ FLOP/byte
```

---

## Step 5：把三个版本标到 Roofline 上判断

【你来填】（用 Step 1-3 的 AI 和 Step 4 的拐点比较）

| 版本 | AI (FLOP/byte) | < 拐点? | bound 判断 | 实测 GFLOPS(2048) |
| --- | --- | --- | --- | --- |
| naive | ___1/4___ | ___10___ | __memory____ | （没测，可选） |
| shared | ___16___ | ___10___ | __compute____ | ~4194 |
| 2D+pad | ____16__ | ___10___ | ____compute__ | ~11313（8x8）/ 11382（8x4） |

【你来判断】
```text
naive：AI 远 < 拐点 → 重度 ____-bound
shared/2D：AI ____ 拐点 → 理论上 ____-bound（global 层面）
```

---

## Step 6：解释「Roofline 没说出来的事」（最重要的结论）

【你来想 + 写】回答这几个问题，就是 Day 6 的精华：

```text
1. naive → shared：AI 为什么暴涨？（提示：global 复用）
2. shared vs 2D：global AI 一样，为什么 2D 还更快？
   （提示：瓶颈不在 DRAM。回看 Day4/5 ncu：DRAM 只有 1.5%。
    2D 省的是 shared/L1 的访问量，不是 DRAM。
    DRAM-roofline 看不出 2D 的好处，要用 shared-memory 层面的账或 occupancy 解释。）
3. 在 A100 上，shared/2D 已经是 compute/occupancy-bound，不是 DRAM-bound。
   那继续提速该往哪个方向？（提示：Day5 的 occupancy / vectorized load）
```

---

## Step 7：整理成 roofline.md

把上面填好的：
```text
- FLOP/bytes/AI 三版本表
- Roofline 拐点 + 各版本落点
- 三个结论问答（尤其第 2 条：2D 的好处为什么 DRAM-roofline 看不出来）
- 一段面试口述：「我怎么用 Roofline 判断 GEMM 的瓶颈」
```
整理进 `week05_gemm_advanced/roofline.md`。

---

## 验收标准

```text
[ ] 三个版本的 AI 都算对（naive 很小、shared/2D 相同且较大）
[ ] 拐点 AI* 算对
[ ] 能说清「naive 为何 memory-bound、shared 为何跳到 compute 侧」
[ ] 能说清「2D 的收益为什么不在 DRAM-roofline 上体现」（这是高级点）
[ ] roofline.md 成文 + 一段口述
```

> 先做 Step 0-1，算完把 naive 的 bytes 和 AI 贴给我，我确认你方法对了再往下。
