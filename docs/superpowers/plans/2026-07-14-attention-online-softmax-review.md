# Attention 与 Online Softmax 复习文档扩充实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 Online Softmax 证明扩充为从标准 Attention 公式、`O_acc` 直觉到严格正确性证明和 `Br=4` 应用的自包含复习文档。

**Architecture:** 只重写一份现有证明文档，采用“Attention 加权平均直觉 → Stable/Online 状态 → 数字手算 → 分块证明 → causal 与浮点边界 → M1 多行推广”的阅读顺序。保留原证明的数学内容，统一使用未归一化向量分子 `O_acc`，不加入完整 CUDA kernel。

**Tech Stack:** Markdown、KaTeX 数学公式、仓库 Markdown 链接检查脚本、Git。

---

## 文件结构

- 修改：`docs/proofs/Online_Softmax正确性证明.md`——唯一的 Attention/Online Softmax 完整证明与复习正文。
- 参考：`docs/courses/attention/M1_Br4_QueryTiled_FP32_SIMT_Attention完整学习资料.md`——对齐 `Br=4`、`Bc=16`、逐行状态与 all-mask 语义。
- 参考：`docs/superpowers/specs/2026-07-14-attention-online-softmax-review-design.md`——本次扩充的范围和验收标准。

### Task 1：重建 Attention 与 `O_acc` 复习入口

**Files:**

- Modify: `docs/proofs/Online_Softmax正确性证明.md:1-305`

- [ ] **Step 1：写入文档目标、符号和 shape**

在开头明确单 batch、单 head、forward-only 语境，并固定：

```text
Q [N_q,D_h]
K [N_k,D_h]
V [N_k,D_v]
S [N_q,N_k]
P [N_q,N_k]
O [N_q,D_v]
```

写入标准 Attention 公式：

$$
S_{ij}=\frac{Q_iK_j^\top}{\sqrt{D_h}},\qquad
P_{ij}=\frac{e^{S_{ij}}}{\sum_t e^{S_{it}}},\qquad
O_i=\sum_j P_{ij}V_j.
$$

- [ ] **Step 2：把单行输出改写为分子除以分母**

对固定 query 行 `i`，令 `x_j=S_ij`，推导：

$$
O_i=
\frac{\sum_j e^{x_j}V_j}
     {\sum_j e^{x_j}}.
$$

随后引入稳定基准 `m=max_j x_j`：

$$
\ell=\sum_j e^{x_j-m},\qquad
O_{\mathrm{acc}}=\sum_j e^{x_j-m}V_j,\qquad
O_i=\frac{O_{\mathrm{acc}}}{\ell}.
$$

- [ ] **Step 3：增加状态对照表和一句话记忆**

表格必须区分：

| 状态 | shape | 含义 |
| --- | --- | --- |
| `m` | 标量 | 已处理 score 的最大值，也是指数基准 |
| `l` | 标量 | 相对 `m` 的未归一化权重之和，即分母 |
| `alpha` | 标量 | 从旧基准换到新基准的缩放因子 |
| `O_acc` | `[D_v]` | 未归一化权重乘 `V` 后的向量和，即分子 |
| `O` | `[D_v]` | 最终归一化输出 `O_acc/l` |

一句话固定为：`l` 累加权重，`O_acc` 累加“权重 × V”；二者是同一个加权平均的分母和向量分子。

- [ ] **Step 4：检查复习入口的符号一致性**

检查这一部分中 `D_h` 只表示 Q/K feature，`D_v` 只表示 V/output feature；不得用 `O` 指代未归一化累加器。

### Task 2：统一 Online 更新、手算与严格证明

**Files:**

- Modify: `docs/proofs/Online_Softmax正确性证明.md`

- [ ] **Step 1：定义集合状态不变量**

对任意已处理的非空索引集合 `S` 写出：

$$
m(S)=\max_{j\in S}x_j,
$$

$$
\ell(S)=\sum_{j\in S}e^{x_j-m(S)},
$$

$$
O_{\mathrm{acc}}(S)=
\sum_{j\in S}e^{x_j-m(S)}V_j.
$$

说明这是每轮 tile 结束后必须保持的不变量。

- [ ] **Step 2：证明标量和向量换基准**

对任意 `m' >= m(S)`，分别证明：

$$
\sum_{j\in S}e^{x_j-m'}
=e^{m(S)-m'}\ell(S),
$$

$$
\sum_{j\in S}e^{x_j-m'}V_j
=e^{m(S)-m'}O_{\mathrm{acc}}(S).
$$

由此定义：

$$
\alpha=e^{m_{\mathrm{old}}-m_{\mathrm{new}}}.
$$

明确指出 `l` 和 `O_acc` 都由同一批指数权重构成，因此必须一起乘 `alpha`。

- [ ] **Step 3：写出元素级和 tile 级更新公式**

元素级更新：

$$
m_{\mathrm{new}}=\max(m_{\mathrm{old}},x_t),
$$

$$
\ell_{\mathrm{new}}
=\alpha\ell_{\mathrm{old}}+e^{x_t-m_{\mathrm{new}}},
$$

$$
O_{\mathrm{acc,new}}
=\alpha O_{\mathrm{acc,old}}
+e^{x_t-m_{\mathrm{new}}}V_t.
$$

tile 级更新：

$$
m_{\mathrm{new}}=\max(m_{\mathrm{old}},m_{\mathrm{block}}),
$$

$$
w_j=e^{x_j-m_{\mathrm{new}}},
$$

$$
\ell_{\mathrm{new}}
=\alpha\ell_{\mathrm{old}}+\sum_{j\in B}w_j,
$$

$$
O_{\mathrm{acc,new}}
=\alpha O_{\mathrm{acc,old}}+\sum_{j\in B}w_jV_j.
$$

- [ ] **Step 4：加入两个 tile 的完整数字手算**

沿用：

```text
scores = [2,1,4,3]
V0 = [1,0]
V1 = [0,1]
V2 = [2,0]
V3 = [0,2]
tile A = {0,1}
tile B = {2,3}
```

必须计算并解释：

```text
tile A:
m=2
l=1+e^-1
O_acc=[1,e^-1]

tile B:
m_new=4
alpha=e^(2-4)=e^-2
l_new=e^-2(1+e^-1)+(1+e^-1)
O_acc_new=e^-2[1,e^-1]+[2,2e^-1]

final:
O=O_acc_new/l_new
```

最后用一次性标准 Attention 的全局 `m=4` 反查，确认分子和分母一致。

- [ ] **Step 5：保留并统一分块合并证明**

保留原文的分块合并、交换律、结合律和流式归纳证明，但将状态统一为三元组：

```text
(m, l, O_acc)
```

合并不相交集合 `A/B` 时使用共同基准 `m=max(m_A,m_B)`：

$$
\ell=
e^{m_A-m}\ell_A+e^{m_B-m}\ell_B,
$$

$$
O_{\mathrm{acc}}=
e^{m_A-m}O_{\mathrm{acc},A}
+e^{m_B-m}O_{\mathrm{acc},B}.
$$

证明最终结果等于标准 `softmax(S)V`，并保留 exact Attention 不减少主导 FLOP 的结论。

### Task 3：补全边界、`Br=4` 推广与复习检查

**Files:**

- Modify: `docs/proofs/Online_Softmax正确性证明.md`

- [ ] **Step 1：写清空状态和 all-mask 语义**

空状态定义为：

```text
m = -∞
l = 0
O_acc = 0
```

局部 tile 对某行全 mask 时固定为：

```text
alpha = 1
weights = 0
m/l/O_acc 保持不变
```

说明必须显式分支，不能计算 `-∞-(-∞)`。

- [ ] **Step 2：推广到 `Br=4` M1 状态**

写入 shape：

```text
scores/weights [Br,Bc] = [4,16]
m/l/alpha      [Br]    = [4]
O_acc          [Br,D_v]
```

逐行更新：

$$
O_{\mathrm{acc,new}}[r,d]
=\alpha[r]O_{\mathrm{acc,old}}[r,d]
+\sum_c w[r,c]V[c,d].
$$

明确说明 K/V tile 在 CTA 内共享，但 `m/l/alpha/O_acc` 按 query 行独立；同一 tile 到来时四个 `alpha[r]` 可以不同。

- [ ] **Step 3：补充常见错误表**

至少包含：

| 错误 | 后果 |
| --- | --- |
| 把 `O_acc` 当作最终输出 | 忘记最后除以 `l` |
| 只给 `l` 乘 `alpha` | 分子与分母指数尺度不一致 |
| 每个 tile 单独 softmax 后平均 | 丢失跨 tile 的相对权重 |
| 四条 query 共用一个 `m/l/alpha` | 混合四个独立 softmax 行 |
| all-mask 时令 `alpha=0` | 错误删除旧状态 |
| 空状态直接计算 `-∞-(-∞)` | 产生 NaN |

- [ ] **Step 4：补充闭卷自测和最终口述**

自测必须覆盖：

1. 为什么 `l` 是标量而 `O_acc` 是向量？
2. 为什么最终为 `O_acc/l`？
3. 新最大值出现时为什么二者都乘 `alpha`？
4. `Br=4` 时哪些数据共享，哪些状态逐行独立？
5. 局部全 mask 为什么保持旧状态？

最终一句话口述应概括：Online Attention 保存相同指数基准下的最大值、标量分母和向量分子；新 max 出现时二者共同换基准，所有 tile 完成后做一次 `O_acc/l`，因此与标准 Attention 数学等价。

### Task 4：验证文档并提交

**Files:**

- Test: `docs/proofs/Online_Softmax正确性证明.md`

- [ ] **Step 1：运行仓库链接检查**

Run:

```bash
python3 scripts/check_markdown_links.py \
  docs/proofs/Online_Softmax正确性证明.md
```

Expected:

```text
broken relative Markdown targets: 0
```

- [ ] **Step 2：检查 Markdown 基础结构**

Run:

```bash
python3 -c 'from pathlib import Path; p=Path("docs/proofs/Online_Softmax正确性证明.md"); lines=p.read_text(encoding="utf-8").splitlines(); assert not any(line.rstrip()!=line for line in lines), "trailing whitespace"; levels=[len(line)-len(line.lstrip("#")) for line in lines if line.startswith("#")]; assert all(b<=a+1 for a,b in zip(levels,levels[1:])), "heading level jump"; assert sum(line.startswith("```") for line in lines)%2==0, "unbalanced code fence"; print("Markdown structure checks: PASS")'
```

Expected:

```text
Markdown structure checks: PASS
```

- [ ] **Step 3：检查编辑器诊断与 Git 空白错误**

使用编辑器诊断确认目标文件无错误，然后运行：

```bash
git diff --check -- \
  docs/proofs/Online_Softmax正确性证明.md
```

Expected: 无输出，退出码为 0。

- [ ] **Step 4：人工复核数学合同**

逐项确认：

```text
Attention shape 和下标一致
l 与 O_acc 始终使用同一个 m
alpha 的指数方向始终是 old-new
最终输出只在所有 tile 完成后做 O_acc/l
all-mask 不改变旧状态
Br=4 的状态按行独立
实数 exact 与浮点容差结论分开
```

- [ ] **Step 5：提交文档更新**

```bash
git add -- docs/proofs/Online_Softmax正确性证明.md
git commit -m "docs: expand attention online softmax proof"
```