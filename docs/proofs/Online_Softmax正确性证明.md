# Online Softmax 正确性证明

> 目标：严格证明 online（流式 / 分块）softmax 与标准 softmax 在实数运算下**完全等价**。
> 复习顺序：先从标准 Attention 看清“分母 + 向量分子”，再证明标量流式更新、分块合并以及 FlashAttention 中 $O_{\text{acc}}$ 的重缩放。
> 记号：下标从 0 开始；以下先讨论实数运算，浮点舍入放在第 9 节单独说明。

---

## 1. 复习入口：从标准 Attention 到稳定分子、分母

### 1.1 语境与 shape

本文只讨论 **single batch、single head、forward-only** 的 Attention。省略 batch 和 head 下标后，各张量的 shape 为：

| 张量 | Shape | 含义 |
| --- | --- | --- |
| $Q$ | $[N_q, D_h]$ | query；$D_h$ 是 $Q/K$ 的 feature 维度 |
| $K$ | $[N_k, D_h]$ | key；与 $Q$ 在 $D_h$ 维做点积 |
| $V$ | $[N_k, D_v]$ | value；$D_v$ 是 value feature 维度 |
| $S$ | $[N_q, N_k]$ | Attention logits |
| $P$ | $[N_q, N_k]$ | 对 $S$ 的每个 query 行做 softmax 后的概率 |
| $O$ | $[N_q, D_v]$ | 最终 Attention 输出 |

这里严格区分两个 feature 维度：$D_h$ 只属于 $Q/K$，$D_v$ 只属于 $V/O$。

### 1.2 标准 Attention 公式

对 query 行 $i$ 和 key/value 位置 $j$，标准 scaled dot-product Attention 为：

$$
S_{ij} = \frac{Q_i K_j^\top}{\sqrt{D_h}},
$$

$$
P_{ij} = \frac{e^{S_{ij}}}{\displaystyle\sum_{t=0}^{N_k-1} e^{S_{it}}},
$$

$$
O_i = \sum_{j=0}^{N_k-1} P_{ij} V_j.
$$

其中 $Q_i,K_j \in \mathbb{R}^{D_h}$，$V_j,O_i \in \mathbb{R}^{D_v}$。

### 1.3 固定一行：输出就是“向量分子 / 标量分母”

固定任意 query 行 $i$，令

$$
x_j := S_{ij}, \qquad 0 \le j < N_k.
$$

把 $P_{ij}$ 代回输出公式：

$$
\begin{aligned}
O_i
&= \sum_{j=0}^{N_k-1}
\frac{e^{x_j}}{\sum_{t=0}^{N_k-1} e^{x_t}} V_j \\
&= \frac{\displaystyle\sum_{j=0}^{N_k-1} e^{x_j} V_j}
{\displaystyle\sum_{j=0}^{N_k-1} e^{x_j}}.
\end{aligned}
$$

因此，Attention 的一行输出本质上是一个加权平均：分母累加标量权重，分子累加“标量权重 $\times$ 向量 $V_j$”。

### 1.4 减去最大值：稳定表示不改变最终输出

令这一行的最大 logit、稳定分母和未归一化向量分子分别为：

$$
m = \max_{0 \le j < N_k} x_j,
$$

$$
\ell = \sum_{j=0}^{N_k-1} e^{x_j-m},
\qquad
O_{\text{acc}} = \sum_{j=0}^{N_k-1} e^{x_j-m} V_j.
$$

完整输出矩阵始终记为 $O$；固定当前 query 行时，其第 $i$ 行记为 $O_i$，并有：

$$
O_i = \frac{O_{\text{acc}}}{\ell}.
$$

这是因为减去 $m$ 会让向量分子和标量分母同时乘上同一个公共因子 $e^{-m}$，相除时该因子抵消：

$$
\frac{\sum_j e^{x_j-m}V_j}{\sum_j e^{x_j-m}}
= \frac{e^{-m}\sum_j e^{x_j}V_j}{e^{-m}\sum_j e^{x_j}}
= \frac{\sum_j e^{x_j}V_j}{\sum_j e^{x_j}}.
$$

当在线更新得到更大的基准 $m_{\text{new}}$ 时，旧状态需要乘换基准因子

$$
\alpha = e^{m_{\text{old}}-m_{\text{new}}}.
$$

后文将严格证明：$\ell$ 与 $O_{\text{acc}}$ 必须用同一个 $m$，并在换基准时同时乘同一个 $\alpha$。

| 符号 | 类型 / Shape | 本文中的唯一含义 |
| --- | --- | --- |
| $m$ | 标量 | 当前已处理 logits 的指数基准，即它们的最大值 |
| $\ell$ | 标量 | 相对 $m$ 的稳定权重之和，作为最终除法的分母 |
| $\alpha$ | 标量 | 从旧基准换到新基准的因子 $e^{m_{\text{old}}-m_{\text{new}}}$ |
| $O_{\text{acc}}$ | $[D_v]$ 向量 | 相对 $m$ 的未归一化向量分子，绝不表示最终输出 |
| $O_i$ | $[D_v]$ 向量 | 第 $i$ 行的最终输出，$O_i=O_{\text{acc}}/\ell$ |

> **一句话记忆**：$\ell$ 累加权重，$O_{\text{acc}}$ 累加“权重 $\times V$”；二者是同一个加权平均的分母和向量分子。

### 1.5 后续证明使用的集合状态

我们的目标是不一次性看到全部 $x_j$，而是**流式**或**分块**地计算，仍得到相同的 $(m,\ell,O_{\text{acc}})$，从而恢复相同的第 $i$ 行最终输出 $O_i$；逐行计算后得到完整输出矩阵 $O$。为简化后续证明，记 $N:=N_k$。先看 softmax 的标量部分：对任意非空索引集合 $S \subseteq \{0,\dots,N-1\}$，定义它的一对状态：

$$
m(S) = \max_{j \in S} x_j, \qquad
\ell(S) = \sum_{j \in S} e^{x_j - m(S)}.
$$

注意 $\ell(S)$ 是**相对于本集合的最大值 $m(S)$** 定义的。标准 softmax 的标量状态就是 $S = \{0,\dots,N-1\}$ 时的 $(m(S), \ell(S))$。

---

## 2. 核心引理：换基准（rebasing）恒等式

**引理 1（换基准）**：对任意集合 $S$ 和任意实数 $m' \ge m(S)$，有

$$
\sum_{j \in S} e^{x_j - m'} = \ell(S)\, e^{m(S) - m'}.
$$

**证明**：

$$
\sum_{j \in S} e^{x_j - m'}
= \sum_{j \in S} e^{(x_j - m(S)) + (m(S) - m')}
= e^{m(S) - m'} \sum_{j \in S} e^{x_j - m(S)}
= \ell(S)\, e^{m(S) - m'}. \qquad \blacksquare
$$

直觉：$\ell(S)$ 里每一项都以 $m(S)$ 为基准；要换成以更大的 $m'$ 为基准，只需整体乘公共因子 $e^{m(S)-m'}$。这就是"新 max 出现时旧 $\ell$ 要乘 $e^{m_{\text{old}}-m_{\text{new}}}$"的数学根源。

---

## 3. 分块合并的正确性

**定理 1（合并公式）**：设 $A, B$ 是两个不相交的非空索引集合，$S = A \cup B$。则

$$
m(S) = \max\big(m(A), m(B)\big),
$$
$$
\ell(S) = \ell(A)\, e^{m(A) - m(S)} + \ell(B)\, e^{m(B) - m(S)}.
$$

**证明**：最大值部分显然，因为 $A, B$ 覆盖 $S$：

$$
m(S) = \max_{j \in S} x_j = \max\Big(\max_{j\in A} x_j,\ \max_{j\in B} x_j\Big) = \max(m(A), m(B)).
$$

对 $\ell(S)$，因 $A \cap B = \varnothing$ 且 $A \cup B = S$：

$$
\ell(S) = \sum_{j \in S} e^{x_j - m(S)}
= \sum_{j \in A} e^{x_j - m(S)} + \sum_{j \in B} e^{x_j - m(S)}.
$$

对两项分别用引理 1（因 $m(S) \ge m(A)$ 且 $m(S) \ge m(B)$）：

$$
\ell(S) = \ell(A)\, e^{m(A) - m(S)} + \ell(B)\, e^{m(B) - m(S)}. \qquad \blacksquare
$$

这正是代码里的 `combine`：

```text
m = max(m_a, m_b)
l = l_a * exp(m_a - m) + l_b * exp(m_b - m)
```

---

## 4. 合并运算的结合律与交换律

定义二元运算 $\oplus$ 作用在状态对上：$(m_A, \ell_A) \oplus (m_B, \ell_B) = (m_S, \ell_S)$，由定理 1 给出。

**定理 2（$\oplus$ 满足交换律与结合律）**。

**交换律**：显然，因为定理 1 的公式关于 $A, B$ 对称（$\max$ 与加法都对称）。

**结合律**：设 $A, B, C$ 两两不相交。要证

$$
\big((m_A,\ell_A) \oplus (m_B,\ell_B)\big) \oplus (m_C,\ell_C)
= (m_A,\ell_A) \oplus \big((m_B,\ell_B) \oplus (m_C,\ell_C)\big).
$$

**证明**：由定理 1，两边的第一分量都等于 $\max(m_A, m_B, m_C) =: M$。

对第二分量，反复应用引理 1，两边都化简为**同一个表达式**：

$$
\ell_A e^{m_A - M} + \ell_B e^{m_B - M} + \ell_C e^{m_C - M}.
$$

具体地，左边：先合并 $A,B$ 得 $\big(m_{AB}, \ell_{AB}\big)$，其中 $\ell_{AB} = \ell_A e^{m_A - m_{AB}} + \ell_B e^{m_B - m_{AB}}$；再与 $C$ 合并，$M = \max(m_{AB}, m_C)$：

$$
\ell_{ABC} = \ell_{AB}\, e^{m_{AB} - M} + \ell_C\, e^{m_C - M}
= \big(\ell_A e^{m_A - m_{AB}} + \ell_B e^{m_B - m_{AB}}\big) e^{m_{AB} - M} + \ell_C e^{m_C - M}.
$$

利用 $e^{m_A - m_{AB}} e^{m_{AB} - M} = e^{m_A - M}$（指数相加），得

$$
\ell_{ABC} = \ell_A e^{m_A - M} + \ell_B e^{m_B - M} + \ell_C e^{m_C - M}.
$$

右边同理化简到相同表达式。故结合律成立。$\blacksquare$

**推论**：由于 $\oplus$ 满足交换律与结合律，把 $\{0,\dots,N-1\}$ **任意分块、按任意顺序**两两合并，最终得到的 $(m, \ell)$ 都相同，且等于整体的 $(m(S), \ell(S))$。这保证了：

- 并行归约（warp/block reduction）用什么合并顺序都对；
- 分成多少个 tile、tile 多大都不影响最终结果。

---

## 5. 标量流式更新是分块合并的特例

流式处理时，把"当前累积集合"记为 $S_t = \{x_0, \dots, x_{t-1}\}$，来一个新元素 $x_t$ 相当于合并单元素集合 $\{x_t\}$（其状态为 $(m, \ell) = (x_t, 1)$，因为 $e^{x_t - x_t} = 1$）。

代入定理 1（$A = S_t$，$B = \{x_t\}$）：

$$
m_{\text{new}} = \max(m_{\text{old}}, x_t),
$$
$$
\ell_{\text{new}} = \ell_{\text{old}}\, e^{m_{\text{old}} - m_{\text{new}}} + 1 \cdot e^{x_t - m_{\text{new}}}.
$$

这正是代码里的 `update`。

**定理 3（流式扫描正确性，归纳法）**：从 $S_1 = \{x_0\}$ 开始逐个合并到 $S_N$，得到的 $(m, \ell)$ 等于标准 softmax 的 $(m(S), \ell(S))$。

**证明**：对 $t$ 归纳。

- **基础** $t = 1$：$S_1 = \{x_0\}$，$m = x_0 = m(S_1)$，$\ell = e^{x_0 - x_0} = 1 = \ell(S_1)$。成立。
- **归纳步**：假设扫描到 $S_t$ 时状态正确为 $(m(S_t), \ell(S_t))$。合并 $\{x_t\}$ 后，由定理 1（其正确性已独立证明）得到的正是 $(m(S_{t+1}), \ell(S_{t+1}))$。

故对所有 $t$ 成立，特别地 $t = N$ 时得到整体正确的 $(m, \ell)$。$\blacksquare$

**初值/空状态**：约定空集状态为 $(m, \ell) = (-\infty, 0)$。合并任意集合 $A$：

$$
m = \max(-\infty, m_A) = m_A, \qquad
\ell = 0 \cdot e^{-\infty - m_A} + \ell_A\, e^{m_A - m_A} = \ell_A,
$$

其中约定 $0 \cdot e^{-\infty} = 0$（代码中必须显式判空，避免计算 $-\infty - (-\infty)$ 产生 NaN）。所以 $(-\infty, 0)$ 是 $\oplus$ 的**单位元**，流式扫描可安全地从它起步。

---

## 6. 从 $(m,\ell)$ 恢复概率

得到全局 $(m, \ell)$ 后，对任意 $i$：

$$
p_i = \frac{e^{x_i - m}}{\ell}.
$$

由定理 2/3，无论怎样分块流式得到的 $(m, \ell)$ 都与标准值相同，故 $p_i$ 与标准 softmax 完全一致。

**注意**：流式扫描一遍只能得到 $(m, \ell)$，不能在扫描途中就写出最终 $p_i$——因为最终分母 $\ell$ 直到看完所有元素才确定。要输出所有 $p_i$，需再遍历一次，或保存未归一化值最后统一除。这一点在 FlashAttention 中通过"输出累加器"巧妙绕过（见第 7 节）。

---

## 7. FlashAttention：带未归一化分子 $O_{\text{acc}}$ 的完整证明

Attention 中我们真正想要的不是概率 $p_j$ 本身，而是加权输出

$$
O_i = \sum_{j=0}^{N-1} p_j\, V_j = \frac{1}{\ell}\sum_{j=0}^{N-1} e^{x_j - m}\, V_j,
$$

其中 $x_j = (QK^\top)_{ij}/\sqrt{D_h}$ 是第 $j$ 个 logit，$V_j \in \mathbb{R}^{D_v}$ 是第 $j$ 个 value 向量。

定义**未归一化输出**（相对基准 $m(S)$）：

$$
O_{\text{acc}}(S) = \sum_{j \in S} e^{x_j - m(S)}\, V_j.
$$

则最终 $O_i = O_{\text{acc}}(S)/\ell(S)$，$S = \{0,\dots,N-1\}$。

**定理 4（输出累加器合并公式）**：对不相交非空 $A, B$，$S = A \cup B$，$m(S) = \max(m(A), m(B))$，有

$$
O_{\text{acc}}(S) = e^{m(A) - m(S)}\, O_{\text{acc}}(A) + e^{m(B) - m(S)}\, O_{\text{acc}}(B).
$$

**证明**：与定理 1 完全平行。因 $A \cap B = \varnothing$、$A \cup B = S$：

$$
O_{\text{acc}}(S) = \sum_{j \in S} e^{x_j - m(S)} V_j
= \sum_{j \in A} e^{x_j - m(S)} V_j + \sum_{j \in B} e^{x_j - m(S)} V_j.
$$

对第一项，逐项乘 $e^{m(A)-m(A)}=1$ 后换基准：

$$
\sum_{j \in A} e^{x_j - m(S)} V_j
= \sum_{j \in A} e^{(x_j - m(A)) + (m(A) - m(S))} V_j
= e^{m(A) - m(S)} \sum_{j \in A} e^{x_j - m(A)} V_j
= e^{m(A) - m(S)}\, O_{\text{acc}}(A).
$$

$B$ 同理。相加即得。$\blacksquare$

这正是 FlashAttention 里"旧输出也要乘 $\alpha = e^{m_{\text{old}}-m_{\text{new}}}$"的来源：$\ell$ 和 $O_{\text{acc}}$ 使用**同一个基准 $m$**，换基准时必须**一起**乘同一个 $\alpha$，否则分子分母尺度不一致，结果错误。

**流式版本**（来一个 $(x_t, V_t)$，即合并 $\{t\}$，其 $O_{\text{acc}} = e^{0}V_t = V_t$）：

$$
m_{\text{new}} = \max(m_{\text{old}}, x_t), \quad
\alpha = e^{m_{\text{old}} - m_{\text{new}}},
$$
$$
\ell_{\text{new}} = \alpha\,\ell_{\text{old}} + e^{x_t - m_{\text{new}}},
$$
$$
O_{\text{acc,new}} = \alpha\, O_{\text{acc,old}} + e^{x_t - m_{\text{new}}}\, V_t.
$$

遍历所有 $j$ 后：$O_i = O_{\text{acc}}/\ell$。

**定理 5（FlashAttention 精确性）**：上述流式（或任意分块）计算得到的 $O_i$ 与标准 attention 固定第 $i$ 行的 $O_i = \operatorname{softmax}(x) V$ 在实数运算下相等。

**证明**：由定理 4 与"$O_{\text{acc}}$ 的合并"同样满足交换律、结合律（证明与定理 2 逐字平行，把 $\ell$ 换成向量 $O_{\text{acc}}$，标量乘保持线性即可），故任意分块顺序得到的 $(m, \ell, O_{\text{acc}})$ 都等于整体值。于是

$$
O_i = \frac{O_{\text{acc}}(S)}{\ell(S)} = \frac{\sum_j e^{x_j - m} V_j}{\sum_j e^{x_j - m}} = \sum_j p_j V_j,
$$

即标准 attention 的第 $i$ 行输出。$\blacksquare$

**推论**：FlashAttention 计算的是**精确**的 attention（exact），不是近似。它与标准实现的唯一差别是运算顺序（进而在浮点下有末位差异），而非数学定义。它也**没有**降低主导的 $O(N^2 D_h)$ 计算量——省的是 HBM IO 和 $O(N^2)$ 中间存储，不是 FLOP。

---

## 8. 因果掩码（causal mask）下依然成立

causal 情形对某些 $j$ 令 $x_j = -\infty$（未来位置）。这些项满足

$$
e^{-\infty - m} = 0,
$$

对 $\ell$ 和 $O_{\text{acc}}$ 的贡献都是 $0$，等价于把它们排除出集合 $S$。因此上述所有定理对"有效集合 $S' = \{j : x_j > -\infty\}$"照常成立。

**边界情形**：若某个局部 tile 内所有 $x_j = -\infty$（整块被 mask），则该 tile 的 $m(\text{tile}) = -\infty$、$\ell = 0$、$O_{\text{acc}} = 0$，它是单位元，对合并无影响。代码必须显式处理，避免 $e^{-\infty-(-\infty)}$ 型 NaN。只要全局至少有一个有效位置（causal self-attention 中每行至少有自身 $j=i$ 可见），最终 $\ell > 0$，$O_i$ 有定义。

---

## 9. 浮点数下的说明

以上都在**实数**下证明为精确相等。浮点运算中：

- 加法不满足严格结合律，故不同归约/分块顺序可能产生**末位级差异**；
- 减 $\max$ 的稳定化保证每个 $e^{x_j - m} \in (0, 1]$，不会上溢；$m$ 是真实最大值时至少有一项等于 $e^0 = 1$，分母 $\ell \ge 1 > 0$，不会下溢为零除。

因此工程上应以**合理容差**（如相对/绝对误差 $< 10^{-4} \sim 10^{-6}$，视精度而定）判定与标准实现"相等"，而不是要求逐 bit 相同。这与"数学上精确"并不矛盾——差异纯粹来自浮点求和顺序。

---

## 10. 结论一览

| 命题 | 内容 |
|---|---|
| 引理 1 | 换基准：$\sum_{j\in S} e^{x_j-m'} = \ell(S)e^{m(S)-m'}$ |
| 定理 1 | $\ell$ 的分块合并公式 |
| 定理 2 | 合并运算 $\oplus$ 满足交换律 + 结合律 → 任意分块顺序结果一致 |
| 定理 3 | 标量流式扫描（归纳）等于标准 $(m,\ell)$ |
| 定理 4 | 输出累加器 $O_{\text{acc}}$ 的合并公式（旧输出乘 $\alpha$） |
| 定理 5 | FlashAttention 输出 = 标准 attention 输出（exact） |
| 第 8 节 | causal mask 下 $-\infty$ 项贡献 0，结论不变 |
| 第 9 节 | 浮点下精确相等 → 末位差异，用容差判定 |

**一句话**：online softmax（及 FlashAttention）的正确性 = 一个换基准恒等式（引理 1）+ 合并运算的结合律（定理 2/4）。所有“新 max 出现要重缩放旧 $\ell$ 和旧 $O_{\text{acc}}$”的操作，本质都是“把旧基准 $m_{\text{old}}$ 换成新基准 $m_{\text{new}}$，统一乘 $e^{m_{\text{old}}-m_{\text{new}}}$”。
