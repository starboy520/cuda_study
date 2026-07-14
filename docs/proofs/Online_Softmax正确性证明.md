# Online Softmax 正确性证明

> 目标：严格证明 online（流式 / 分块）softmax 与标准 softmax 在实数运算下**完全等价**。
> 复习顺序：先从标准 Attention 看清“分母 + 向量分子”，再证明完整状态 $(m,\ell,O_{\mathrm{acc}})$ 的换基准、更新、分块合并与最终等价性。
> 记号：下标从 0 开始；以下先讨论实数运算，浮点舍入放在第 9 节单独说明。

---

## 1. 复习入口：从标准 Attention 到完整在线状态

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
O_{\mathrm{acc}} = \sum_{j=0}^{N_k-1} e^{x_j-m} V_j.
$$

完整输出矩阵始终记为 $O$；固定当前 query 行时，其第 $i$ 行记为 $O_i$，并有：

$$
O_i = \frac{O_{\mathrm{acc}}}{\ell}.
$$

这是因为减去 $m$ 会让向量分子和标量分母同时乘上同一个公共因子 $e^{-m}$，相除时该因子抵消：

$$
\frac{\sum_j e^{x_j-m}V_j}{\sum_j e^{x_j-m}}
= \frac{e^{-m}\sum_j e^{x_j}V_j}{e^{-m}\sum_j e^{x_j}}
= \frac{\sum_j e^{x_j}V_j}{\sum_j e^{x_j}}.
$$

| 符号 | 类型 / Shape | 本文中的唯一含义 |
| --- | --- | --- |
| $m$ | 标量 | 当前已处理 logits 的指数基准，即它们的最大值 |
| $\ell$ | 标量 | 相对 $m$ 的稳定权重之和，作为最终除法的分母 |
| $\alpha$ | 标量 | 从旧基准换到新基准的因子 $e^{m_{\mathrm{old}}-m_{\mathrm{new}}}$ |
| $O_{\mathrm{acc}}$ | $[D_v]$ 向量 | 相对 $m$ 的未归一化向量分子，绝不表示最终输出 |
| $O_i$ | $[D_v]$ 向量 | 第 $i$ 行的最终输出，$O_i=O_{\mathrm{acc}}/\ell$ |

> **一句话记忆**：$\ell$ 累加权重，$O_{\mathrm{acc}}$ 累加“权重 $\times V$”；二者是同一个加权平均的分母和向量分子。

### 1.5 Tile 结束时必须保持的完整状态不变量

为简化记号，以下令 $N:=N_k$。对任意已处理的**非空**索引集合
$S\subseteq\{0,\dots,N-1\}$，定义完整 Attention 状态：

$$
m(S)=\max_{j\in S}x_j,
$$

$$
\ell(S)=\sum_{j\in S}e^{x_j-m(S)},
$$

$$
O_{\mathrm{acc}}(S)=\sum_{j\in S}e^{x_j-m(S)}V_j
\in\mathbb{R}^{D_v}.
$$

每处理完一个元素或一个 K/V tile，running state 都必须恰好等于当前已处理集合的
$(m(S),\ell(S),O_{\mathrm{acc}}(S))$。这就是全文要维护并证明的 **tile 结束不变量**。
特别地，$\ell(S)$ 与 $O_{\mathrm{acc}}(S)$ 中的每个指数权重使用同一个基准 $m(S)$；它们不能分别使用不同的 max。

---

## 2. 核心引理：标量和向量同时换基准

**引理 1（完整状态换基准）**：对任意非空集合 $S$ 和任意实数
$m'\ge m(S)$，有

$$
\sum_{j\in S}e^{x_j-m'}
=e^{m(S)-m'}\ell(S),
$$

以及

$$
\sum_{j\in S}e^{x_j-m'}V_j
=e^{m(S)-m'}O_{\mathrm{acc}}(S).
$$

**证明**：对标量指数和，提取与 $j$ 无关的公共因子：

$$
\begin{aligned}
\sum_{j\in S}e^{x_j-m'}
&=\sum_{j\in S}e^{(x_j-m(S))+(m(S)-m')} \\
&=e^{m(S)-m'}\sum_{j\in S}e^{x_j-m(S)} \\
&=e^{m(S)-m'}\ell(S).
\end{aligned}
$$

对向量分子，同一个标量公共因子可以提出向量和：

$$
\begin{aligned}
\sum_{j\in S}e^{x_j-m'}V_j
&=\sum_{j\in S}e^{(x_j-m(S))+(m(S)-m')}V_j \\
&=e^{m(S)-m'}\sum_{j\in S}e^{x_j-m(S)}V_j \\
&=e^{m(S)-m'}O_{\mathrm{acc}}(S).
\end{aligned}
\qquad\blacksquare
$$

当新数据把基准从 $m_{\mathrm{old}}$ 提高到 $m_{\mathrm{new}}$ 时，定义

$$
\alpha=e^{m_{\mathrm{old}}-m_{\mathrm{new}}}.
$$

引理表明：旧 $\ell$ 和旧 $O_{\mathrm{acc}}$ 都由同一批指数权重构成，因而必须一起乘同一个 $\alpha$。如果只缩放其中一个，分母与向量分子就不再处于同一指数尺度，二者相除也不再是原来的加权平均。

---

## 3. 元素级与 tile 级更新

### 3.1 元素级更新

设旧状态对应非空集合 $S_{\mathrm{old}}$，新到达一对 $(x_t,V_t)$，且
$t\notin S_{\mathrm{old}}$。令

$$
m_{\mathrm{new}}=\max(m_{\mathrm{old}},x_t),
\qquad
\alpha=e^{m_{\mathrm{old}}-m_{\mathrm{new}}}.
$$

由引理 1 把旧状态换到新基准，再加入新元素，得到完整更新：

$$
\ell_{\mathrm{new}}
=\alpha\ell_{\mathrm{old}}+e^{x_t-m_{\mathrm{new}}},
$$

$$
O_{\mathrm{acc,new}}
=\alpha O_{\mathrm{acc,old}}
+e^{x_t-m_{\mathrm{new}}}V_t.
$$

因此更新后的三元组正是
$(m(S_{\mathrm{old}}\cup\{t\}),\ell(S_{\mathrm{old}}\cup\{t\}),
O_{\mathrm{acc}}(S_{\mathrm{old}}\cup\{t\}))$。

### 3.2 Tile $B$ 更新

设 $B$ 是与已处理集合 $S_{\mathrm{old}}$ 不相交的非空新 tile。先求该 tile 的最大值：

$$
m_{\mathrm{block}}=\max_{j\in B}x_j.
$$

再统一选择合并后的新基准，并定义旧状态换基准因子与当前 tile 权重：

$$
m_{\mathrm{new}}=\max(m_{\mathrm{old}},m_{\mathrm{block}}),
\qquad
\alpha=e^{m_{\mathrm{old}}-m_{\mathrm{new}}},
$$

$$
w_j=e^{x_j-m_{\mathrm{new}}},\qquad j\in B.
$$

完整 tile 更新为：

$$
\ell_{\mathrm{new}}
=\alpha\ell_{\mathrm{old}}+\sum_{j\in B}w_j,
$$

$$
O_{\mathrm{acc,new}}
=\alpha O_{\mathrm{acc,old}}+\sum_{j\in B}w_jV_j.
$$

注意这里当前 tile 的 $w_j$ 已直接使用 $m_{\mathrm{new}}$，所以新贡献不需要再缩放；旧贡献原来以 $m_{\mathrm{old}}$ 为基准，必须通过 $\alpha$ 换到同一个 $m_{\mathrm{new}}$。

---

## 4. 两个 tile 的完整数字手算

固定一条 query 行，取

$$
x=[2,1,4,3],
$$

$$
V_0=[1,0],\quad V_1=[0,1],\quad
V_2=[2,0],\quad V_3=[0,2].
$$

将 key/value 分成 tile $A=\{0,1\}$ 与 tile $B=\{2,3\}$。

### 4.1 处理 tile $A$

tile $A$ 的最大值为

$$
m_A=\max(2,1)=2.
$$

相对 $m_A=2$ 的权重是 $1$ 与 $e^{-1}$，因此

$$
\ell_A=1+e^{-1}\approx1.3678794412,
$$

$$
\begin{aligned}
O_{\mathrm{acc},A}
&=1\cdot V_0+e^{-1}V_1 \\
&=[1,e^{-1}] \\
&\approx[1,0.3678794412].
\end{aligned}
$$

所以 tile $A$ 结束后的完整状态是

$$
(m,\ell,O_{\mathrm{acc}})
=\left(2,1+e^{-1},[1,e^{-1}]\right).
$$

### 4.2 处理 tile $B$：旧状态先换基准

tile $B$ 带来更大的最大值：

$$
m_{\mathrm{block}}=\max(4,3)=4,
\qquad
m_{\mathrm{new}}=4.
$$

旧状态从基准 $2$ 换到基准 $4$ 的因子为

$$
\alpha=e^{2-4}=e^{-2}\approx0.1353352832.
$$

因此旧分母换基准后为

$$
\alpha\ell_A
=e^{-2}(1+e^{-1})
=e^{-2}+e^{-3}
\approx0.1851223516,
$$

旧向量分子换基准后为

$$
\begin{aligned}
\alpha O_{\mathrm{acc},A}
&=e^{-2}[1,e^{-1}] \\
&=[e^{-2},e^{-3}] \\
&\approx[0.1353352832,0.0497870684].
\end{aligned}
$$

tile $B$ 相对新基准 $4$ 的权重为

$$
w_2=e^{4-4}=1,
\qquad
w_3=e^{3-4}=e^{-1}.
$$

它对分母的新贡献是

$$
w_2+w_3=1+e^{-1}\approx1.3678794412,
$$

对向量分子的新贡献是

$$
\begin{aligned}
w_2V_2+w_3V_3
&=1[2,0]+e^{-1}[0,2] \\
&=[2,2e^{-1}] \\
&\approx[2,0.7357588823].
\end{aligned}
$$

合并旧、新贡献：

$$
\begin{aligned}
\ell_{\mathrm{new}}
&=e^{-2}(1+e^{-1})+(1+e^{-1}) \\
&=e^{-2}+e^{-3}+1+e^{-1} \\
&\approx1.5530017928,
\end{aligned}
$$

$$
\begin{aligned}
O_{\mathrm{acc,new}}
&=e^{-2}[1,e^{-1}]+[2,2e^{-1}] \\
&=[2+e^{-2},2e^{-1}+e^{-3}] \\
&\approx[2.1353352832,0.7855459507].
\end{aligned}
$$

最终输出只在所有 tile 处理完后归一化：

$$
\begin{aligned}
O_i
&=\frac{O_{\mathrm{acc,new}}}{\ell_{\mathrm{new}}} \\
&=\frac{[2+e^{-2},2e^{-1}+e^{-3}]}
{1+e^{-1}+e^{-2}+e^{-3}} \\
&\approx[1.3749728385,0.5058242395].
\end{aligned}
$$

### 4.3 用一次性标准 Attention 反查

一次性计算全局 stable Attention 时，全局最大值同样是 $m=4$。四个权重依次为

$$
[e^{2-4},e^{1-4},e^{4-4},e^{3-4}]
=[e^{-2},e^{-3},1,e^{-1}].
$$

标准稳定分母为

$$
e^{-2}+e^{-3}+1+e^{-1}=\ell_{\mathrm{new}},
$$

标准稳定向量分子为

$$
\begin{aligned}
e^{-2}V_0+e^{-3}V_1+V_2+e^{-1}V_3
&=[e^{-2},0]+[0,e^{-3}]+[2,0]+[0,2e^{-1}] \\
&=[2+e^{-2},2e^{-1}+e^{-3}] \\
&=O_{\mathrm{acc,new}}.
\end{aligned}
$$

分子、分母分别完全一致，因此两个 tile 的在线结果与一次性标准 Attention 完全一致。这个例子也直接显示：若 tile $B$ 到来时只缩放旧 $\ell$ 而不缩放旧 $O_{\mathrm{acc}}$，反查的向量分子将无法相等。

---

## 5. 完整状态的分块合并定理

**定理 1（完整状态合并公式）**：设 $A,B$ 是两个不相交的非空索引集合，
$S=A\cup B$。简记

$$
(m_A,\ell_A,O_{\mathrm{acc},A})
:=(m(A),\ell(A),O_{\mathrm{acc}}(A)),
$$

$B$ 的状态同理。令

$$
m=\max(m_A,m_B).
$$

则 $S$ 的完整状态为

$$
m(S)=m,
$$

$$
\ell(S)=e^{m_A-m}\ell_A+e^{m_B-m}\ell_B,
$$

$$
O_{\mathrm{acc}}(S)
=e^{m_A-m}O_{\mathrm{acc},A}
+e^{m_B-m}O_{\mathrm{acc},B}.
$$

**证明**：最大值部分由并集定义直接得到：

$$
m(S)=\max_{j\in A\cup B}x_j
=\max\left(\max_{j\in A}x_j,\max_{j\in B}x_j\right)
=\max(m_A,m_B)=m.
$$

由于 $A\cap B=\varnothing$，标量分母可拆成两个集合的和；再对两项分别应用引理 1：

$$
\begin{aligned}
\ell(S)
&=\sum_{j\in A}e^{x_j-m}+\sum_{j\in B}e^{x_j-m} \\
&=e^{m_A-m}\ell_A+e^{m_B-m}\ell_B.
\end{aligned}
$$

向量分子同样按不相交并集拆分并换到共同基准 $m$：

$$
\begin{aligned}
O_{\mathrm{acc}}(S)
&=\sum_{j\in A}e^{x_j-m}V_j
+\sum_{j\in B}e^{x_j-m}V_j \\
&=e^{m_A-m}O_{\mathrm{acc},A}
+e^{m_B-m}O_{\mathrm{acc},B}.
\end{aligned}
\qquad\blacksquare
$$

第 3 节的 tile 更新是这个定理的一种等价写法：它不先构造 tile 自身以 $m_B$ 为基准的局部状态，而是直接以最终共同基准 $m$ 计算每个 $w_j=e^{x_j-m}$。

---

## 6. 完整状态合并的交换律与结合律

对由互不相交集合产生的完整状态定义二元运算 $\oplus$：按定理 1 换到共同最大值后，分别相加 $\ell$ 与 $O_{\mathrm{acc}}$。

**定理 2（$\oplus$ 满足交换律与结合律）**。

**交换律证明**：定理 1 的三条公式关于 $A,B$ 对称；$\max$、标量加法和向量加法都满足交换律，所以

$$
\operatorname{state}(A)\oplus\operatorname{state}(B)
=\operatorname{state}(B)\oplus\operatorname{state}(A).
$$

**结合律证明**：设 $A,B,C$ 两两不相交，并令

$$
M=\max(m_A,m_B,m_C).
$$

无论先合并哪两个集合，最终最大值都是 $M$。先合并 $A,B$，记
$m_{AB}=\max(m_A,m_B)$。其分母再与 $C$ 合并后为

$$
\begin{aligned}
\ell_{(AB)C}
&=\left(\ell_Ae^{m_A-m_{AB}}
+\ell_Be^{m_B-m_{AB}}\right)e^{m_{AB}-M}
+\ell_Ce^{m_C-M} \\
&=\ell_Ae^{m_A-M}+\ell_Be^{m_B-M}+\ell_Ce^{m_C-M}.
\end{aligned}
$$

先合并 $B,C$ 也得到同一个表达式：

$$
\ell_{A(BC)}
=\ell_Ae^{m_A-M}+\ell_Be^{m_B-M}+\ell_Ce^{m_C-M}.
$$

向量分子利用标量乘法对向量加法的分配律，同样有

$$
\begin{aligned}
O_{\mathrm{acc},(AB)C}
&=O_{\mathrm{acc},A}e^{m_A-M}
+O_{\mathrm{acc},B}e^{m_B-M}
+O_{\mathrm{acc},C}e^{m_C-M} \\
&=O_{\mathrm{acc},A(BC)}.
\end{aligned}
$$

三部分都相同，故

$$
(\operatorname{state}(A)\oplus\operatorname{state}(B))
\oplus\operatorname{state}(C)
=\operatorname{state}(A)\oplus
(\operatorname{state}(B)\oplus\operatorname{state}(C)).
\qquad\blacksquare
$$

**推论**：把所有 key/value 位置任意分块、按任意顺序两两合并，在实数运算下最终都会得到同一个完整状态
$(m(S),\ell(S),O_{\mathrm{acc}}(S))$。因此元素扫描、K/V tile 流式处理以及并行树形归约虽然合并顺序不同，但数学结果相同。

---

## 7. 流式归纳证明与空状态

### 7.1 元素级流式扫描的归纳证明

令 $S_t=\{0,\dots,t-1\}$，即已经处理前 $t$ 个 key/value 位置。

**定理 3（完整状态流式扫描正确性）**：从第一个元素开始反复应用第 3.1 节的元素级更新，处理完 $S_t$ 后，running state 恰好为

$$
(m(S_t),\ell(S_t),O_{\mathrm{acc}}(S_t)).
$$

**证明**：对 $t$ 归纳。

- **基础情形 $t=1$**：单元素集合 $S_1=\{0\}$ 的状态为

  $$
  m(S_1)=x_0,\qquad
  \ell(S_1)=e^{x_0-x_0}=1,\qquad
  O_{\mathrm{acc}}(S_1)=V_0.
  $$

- **归纳步骤**：假设处理完 $S_t$ 后完整状态不变量成立。新元素 $t$ 的单元素状态为 $(x_t,1,V_t)$。由定理 1 合并 $S_t$ 与 $\{t\}$，得到的恰好是 $S_{t+1}=S_t\cup\{t\}$ 的完整状态；该合并展开后正是第 3.1 节的更新公式。

因此命题对所有 $t\ge1$ 成立。特别地，$t=N$ 时得到整条 query 行的完整稳定状态。$\blacksquare$

结合定理 2，同一个归纳结论也适用于逐 tile 扫描：只要每个 tile 结束时保持第 1.5 节的不变量，处理完所有 tile 后就是全局状态。

### 7.2 空状态与单位元

定义尚未处理任何有效位置时的完整空状态为

$$
(m,\ell,O_{\mathrm{acc}})=(-\infty,0,\mathbf 0),
$$

其中 $\mathbf 0\in\mathbb R^{D_v}$。它表示没有最大值、分母没有权重、向量分子没有贡献。与任意非空集合 $A$ 合并后应保持 $A$ 的状态不变，所以它是 $\oplus$ 的单位元。

$$
\operatorname{state}(\varnothing)\oplus\operatorname{state}(A)
=\operatorname{state}(A)
=\operatorname{state}(A)\oplus\operatorname{state}(\varnothing).
$$

这里的 $-\infty$ 是“尚无最大值”的哨兵。数学语义上，空侧对 $\ell$ 和 $O_{\mathrm{acc}}$ 的贡献都是零；但不能把该哨兵不加判断地代入非空状态的通用合并公式。空状态与空状态相遇时，直接计算
$m=\max(-\infty,-\infty)$ 后再计算 $e^{-\infty-(-\infty)}$，会先出现未定义的 $-\infty-(-\infty)$。工程实现必须显式判空：首个有效元素或首个非空 tile 直接建立其状态；没有有效元素时继续保留空状态。

---

## 8. 最终结果等于标准 Attention

令全局有效索引集合为 $S=\{0,\dots,N-1\}$。由定理 2 和定理 3，无论采用元素级流式、tile 级流式还是任意归约顺序，最终完整状态都等于定义中的
$(m(S),\ell(S),O_{\mathrm{acc}}(S))$。

若需要显式 softmax 概率，则

$$
p_j=\frac{e^{x_j-m(S)}}{\ell(S)}
$$

与标准 softmax 概率相同。Attention 不必把所有 $p_j$ 写回中间矩阵，而是在扫描时累计向量分子。最终固定 query 行的输出为

$$
\begin{aligned}
O_i
&=\frac{O_{\mathrm{acc}}(S)}{\ell(S)} \\
&=\frac{\sum_{j\in S}e^{x_j-m(S)}V_j}
{\sum_{j\in S}e^{x_j-m(S)}} \\
&=\frac{e^{-m(S)}\sum_{j\in S}e^{x_j}V_j}
{e^{-m(S)}\sum_{j\in S}e^{x_j}} \\
&=\sum_{j\in S}\frac{e^{x_j}}{\sum_{t\in S}e^{x_t}}V_j \\
&=\operatorname{softmax}(x)V.
\end{aligned}
$$

因此，online / tiled Attention 在**实数运算**下与标准 Attention 的固定行输出 exact；逐行应用后得到的完整矩阵 $O$ 也 exact。这里的 exact 是数学算法等价，不是近似 Attention。

这种重排不减少 dense Attention 的主导 FLOP：$QK^\top$ 与概率乘 $V$ 的二次规模计算仍然存在，典型复杂度仍为
$O(N_qN_k(D_h+D_v))$，当 $N_q=N_k=N$ 且 $D_h\approx D_v$ 时即通常所说的 $O(N^2D_h)$。它主要减少 HBM IO 和 $O(N_qN_k)$ 的中间 score/probability 存储。

---

## 9. 因果掩码、局部空 tile 与浮点数

### 9.1 因果掩码下依然成立

causal 情形对未来位置令 $x_j=-\infty$。对任意有限的有效基准 $m$，这些项满足

$$
e^{-\infty-m}=0,
$$

所以它们对 $\ell$ 与 $O_{\mathrm{acc}}$ 的贡献都为零，等价于从集合中排除这些位置。前述定义、换基准引理、合并定理和最终等价性都可直接作用在有效集合

$$
S'=\{j:x_j>-\infty\}
$$

上。

### 9.2 局部全 mask tile

某个局部 tile 可能全部被 mask，此时没有新的有效元素。它在语义上就是空状态
$(-\infty,0,\mathbf0)$，合并后旧 $(m,\ell,O_{\mathrm{acc}})$ 必须保持不变。

实现时应显式采用：

- $\alpha=1$；
- 当前 tile 的全部权重为 $0$；
- $m$、$\ell$ 与 $O_{\mathrm{acc}}$ 保持原值。

即统一执行安全语义

$$
(m_{\mathrm{new}},\ell_{\mathrm{new}},O_{\mathrm{acc,new}})
=(m_{\mathrm{old}},\ell_{\mathrm{old}},O_{\mathrm{acc,old}}).
$$

不能令 $\alpha=0$：由于该 tile 的权重全为 $0$，这样会得到
$\ell_{\mathrm{new}}=0\cdot\ell_{\mathrm{old}}+0=0$ 和
$O_{\mathrm{acc,new}}=0\cdot O_{\mathrm{acc,old}}+\mathbf0=\mathbf0$，错误删除此前所有有效 tile 的状态。$\alpha=1$ 才表达“没有新数据，不改变旧状态”。

这套规则既覆盖旧状态已经非空的情况，也覆盖当前状态仍为空的情况，并避免计算
$e^{-\infty-(-\infty)}$ 产生 NaN。只要整条 query 行至少有一个有效位置（标准 causal self-attention 中自身位置可见），最终 $\ell>0$，$O_i$ 有定义。

### 9.3 浮点数下的说明

以上证明都在实数下成立。浮点运算中：

- 由于舍入和加法的非结合性，不同归约树、tile 大小或处理顺序会产生数值差异；在低精度、长序列或 $O_{\mathrm{acc}}$ 分量存在抵消时，误差可能累积，并不一定只体现为末位差异；
- 对有限的有效 logit 减去真实最大值后，实数精确值 $e^{x_j-m}\in(0,1]$，因此不会上溢；
- 浮点计算中，很小的非最大项可能下溢为 $0$。但假设至少存在一个有限有效项且 max 计算正确，真实最大项对应的 $e^0=1$ 可被浮点精确表示，因此 $\ell$ 至少包含该 $1$，不会因所有项下溢而变为 $0$。这不改变实数算法的 exact 结论。

工程验证应根据 dtype、shape 与误差分布确定合理的绝对/相对误差容差，而不要求与另一种求和顺序 bitwise 相同。

---

## 10. 推广到 M1：$B_r=4,B_c=16$ 的逐行状态

前述证明固定了一条 query 行；推广到多 query tile 时，只需对 CTA 内每个 query 行分别应用同一证明。M1 取 $B_r=4,B_c=16$，它改变的是数据复用范围，不改变“每个 query 行独立做一次 softmax”的数学定义。

### 10.1 Shape 与符号

设 CTA 内局部 query 行为 $r\in\{0,1,2,3\}$，当前 K/V tile 内局部列为
$c\in\{0,\dots,15\}$。对 tail tile，只在各自有效范围内取值。

| 量 | 一般 shape | M1 shape | 含义 |
| --- | --- | --- | --- |
| $\mathrm{scores}$ | $[B_r,B_c]$ | $[4,16]$ | 四条 query 对当前 16 条 key 的 logits |
| $\mathrm{weights}$ | $[B_r,B_c]$ | $[4,16]$ | 相对各行新基准的未归一化权重 |
| $m,\ell,\alpha$ | $[B_r]$ | $[4]$ | 每条 query 行各自的三个标量 |
| $O_{\mathrm{acc}}$ | $[B_r,D_v]$ | $[4,D_v]$ | 每行独立的未归一化向量分子 |
| $K_{\mathrm{tile}}$ | $[B_c,D_h]$ | $[16,D_h]$ | CTA 内四行共享读取的 key tile |
| $V_{\mathrm{tile}}$ | $[B_c,D_v]$ | $[16,D_v]$ | CTA 内四行共享读取的 value tile |

M1 的自注意力教学实现通常令 $D_v=D_h=D$，此时
$O_{\mathrm{acc}}$ 的 shape 可写成 $[4,D]$；一般 Attention 中仍应区分 $D_h$ 与 $D_v$。

### 10.2 每一行独立应用更新公式

对局部行 $r$，若当前 tile 至少有一个对该行有效的 key，定义

$$
m_{\mathrm{block}}[r]
=\max_{c\,\text{对行 }r\text{ 有效}}\mathrm{scores}[r,c],
$$

$$
m_{\mathrm{new}}[r]
=\max\bigl(m_{\mathrm{old}}[r],m_{\mathrm{block}}[r]\bigr),
\qquad
\alpha[r]
=e^{m_{\mathrm{old}}[r]-m_{\mathrm{new}}[r]},
$$

$$
w[r,c]
=
\begin{cases}
e^{\mathrm{scores}[r,c]-m_{\mathrm{new}}[r]},
& c\text{ 对行 }r\text{ 有效},\\
0,& c\text{ 被 mask}.
\end{cases}
$$

于是标量分母逐行更新为

$$
\ell_{\mathrm{new}}[r]
=\alpha[r]\ell_{\mathrm{old}}[r]
+\sum_c w[r,c],
$$

向量分子的每个 feature 分量逐行更新为

$$
O_{\mathrm{acc,new}}[r,d]
=\alpha[r]O_{\mathrm{acc,old}}[r,d]
+\sum_c w[r,c]V_{\mathrm{tile}}[c,d],
\qquad 0\le d<D_v.
$$

旧状态为空而当前 tile 非空时，按第 7.2 节直接由该 tile 建立本行状态，不计算包含空哨兵的差值。当前 tile 对行 $r$ 全 mask 时，则不使用上述非空公式，而是按第 9.2 节令
$\alpha[r]=1$、$w[r,c]=0$，并完整保留该行的 $m[r]$、$\ell[r]$ 与
$O_{\mathrm{acc}}[r,:]$。

所有 K/V tile 完成后，完整输出矩阵 $O$ 中对应的固定行才做最终归一化：

$$
O_{q_0+r,d}=\frac{O_{\mathrm{acc}}[r,d]}{\ell[r]}.
$$

### 10.3 CTA 内共享的是 K/V，独立的是行状态

同一 CTA 可以只加载一份 $K_{\mathrm{tile}}$ 和 $V_{\mathrm{tile}}$，供四条 query 行共同读取；但
$m[r]$、$\ell[r]$、$\alpha[r]$ 与 $O_{\mathrm{acc}}[r,:]$ 必须按 query 行独立。原因是四行产生不同的 score，causal mask 的有效区域也可能不同。

因此，同一个 K/V tile 到来时，四行的 $m_{\mathrm{block}}[r]$ 与
$m_{\mathrm{new}}[r]$ 可以不同，进而四个 $\alpha[r]$ 也可以不同：某行可能出现新 max 而有 $0<\alpha[r]<1$，另一行可能保持旧 max 而有 $\alpha[r]=1$，还有一行可能局部全 mask 并通过安全分支取 $\alpha[r]=1$。$B_r=4$ 不是把四条 query 合并为一个 softmax，而是在一个 CTA 中并排维护四个彼此独立的 online softmax 状态。

> **一句话记忆**：K/V tile 可以共享，softmax 状态不能跨 query 行共享。

## 11. 常见错误与闭卷自测

### 11.1 常见错误表

| 错误 | 后果 | 修正 |
| --- | --- | --- |
| 把 $O_{\mathrm{acc}}$ 当成最终输出 | 输出仍带未归一化权重总量，数值随 $\ell$ 改变 | 所有 tile 完成后计算 $O_i=O_{\mathrm{acc}}/\ell$ |
| 新 max 出现时只缩放 $\ell$ | 分母换到新指数基准，向量分子仍停在旧基准，二者不再对应同一加权平均 | 旧 $\ell$ 与旧 $O_{\mathrm{acc}}$ 共同乘同一个 $\alpha$ |
| 每个 tile 单独 softmax 后再平均 | 各 tile 的权重总量和 max 尺度被丢弃；等权平均通常不等于全局 softmax | 合并未归一化完整状态，只在最后归一化一次 |
| 四条 query 共用一组 $m/\ell/\alpha$ | 不同行的 score、mask 和指数基准互相污染，四行不再是独立 softmax | 使用 shape 为 $[B_r]$ 的逐行标量状态和 $[B_r,D_v]$ 的逐行向量状态 |
| 局部全 mask 时令 $\alpha=0$ | 当前 tile 没有新贡献，却把旧 $\ell/O_{\mathrm{acc}}$ 清零 | 令 $\alpha=1$、weights 全为 $0$，完整状态不变 |
| 空状态直接套通用公式 | 可能计算 $-\infty-(-\infty)$，产生 NaN | 显式判空；首个非空 tile 直接建立状态，空 tile 保持原状态 |

### 11.2 闭卷自测

先遮住后面的答题要点，尝试口述：

1. 为什么 $\ell$ 是标量，而 $O_{\mathrm{acc}}$ 是长度为 $D_v$ 的向量？
2. 为什么最终输出是 $O_{\mathrm{acc}}/\ell$，而不是直接使用 $O_{\mathrm{acc}}$？
3. 新 max 出现时，为什么旧 $\ell$ 与旧 $O_{\mathrm{acc}}$ 必须共同乘 $\alpha$？
4. $B_r=4$ 时，CTA 内哪些数据可以共享，哪些状态必须逐 query 行独立？为什么同一 K/V tile 的四个 $\alpha$ 可以不同？
5. 局部全 mask 时，$\alpha$、weights 与完整状态分别应是什么？为什么不能令 $\alpha=0$？
6. 分块顺序改变时，实数证明与浮点结果分别能保证什么？

**答题要点**：

1. $\ell$ 累加每个 key 的标量指数权重；$O_{\mathrm{acc}}$ 对每个 value feature 累加“权重 $\times V$”，所以有 $D_v$ 个分量。
2. $O_{\mathrm{acc}}$ 是未归一化向量分子，除以同一基准下的标量分母 $\ell$ 后才是加权平均 $O_i$。
3. 二者由同一批指数权重构成；基准从 $m_{\mathrm{old}}$ 改为 $m_{\mathrm{new}}$ 时，所有旧贡献都共同获得因子 $e^{m_{\mathrm{old}}-m_{\mathrm{new}}}$。
4. $K_{\mathrm{tile}}/V_{\mathrm{tile}}$ 可共享；$m/\ell/\alpha/O_{\mathrm{acc}}$ 按行独立。不同行的 score、mask 与新 max 不同，所以换基准因子也可不同。
5. $\alpha=1$、weights 全为 $0$，且 $m/\ell/O_{\mathrm{acc}}$ 完全不变；$\alpha=0$ 会删除旧状态。
6. 实数下交换律、结合律保证任意合法分块与合并顺序 exact 等价；浮点下舍入和加法非结合性只允许在合理容差内比较，不能据此要求 bitwise 相同。

## 12. 结论一览与最终口述

| 命题 | 内容 |
| --- | --- |
| 完整状态不变量 | 每个 tile 结束后保存同一基准下的 $(m,\ell,O_{\mathrm{acc}})$ |
| 引理 1 | 换基准时，$\ell$ 与 $O_{\mathrm{acc}}$ 同乘 $e^{m_{\mathrm{old}}-m_{\mathrm{new}}}$ |
| 元素 / tile 更新 | 旧分母和旧向量分子先共同换基准，再加入新贡献 |
| 定理 1 | 两个分块的完整状态换到共同最大值后可直接相加 |
| 定理 2 | 完整状态合并满足交换律与结合律，任意分块顺序结果一致 |
| 定理 3 | 流式归纳保证每一步都保持完整状态不变量 |
| 最终等价性 | 实数下 $O_i=O_{\mathrm{acc}}/\ell=\operatorname{softmax}(x)V$ |
| 边界 | causal mask 和局部空 tile 等价于零贡献，需显式避免无穷减无穷 |
| $B_r=4$ 推广 | K/V tile 在 CTA 内共享，四条 query 的完整状态逐行独立 |
| 浮点 | 数学上 exact，有限精度下仅因运算顺序与舍入产生差异 |

### 12.1 最终口述

> Online Attention 保存同一指数基准下的最大值、标量分母和向量分子；新 max 出现时，旧 $\ell$ 与旧 $O_{\mathrm{acc}}$ 共同乘 $\alpha$ 换基准；所有 tile 完成后计算 $O_i=O_{\mathrm{acc}}/\ell$，因此在实数运算下与标准 Attention 数学等价。
