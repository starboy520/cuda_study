# Attention 分子与分母小节改写 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将第 23 节改写为从 softmax 定义到分 tile 累加的完整手算教程。

**Architecture:** 仅替换第 23 节，以第 22 节已有的 scores/values 为输入，从普通加权和推导共同分母，再手算两个 tile 的未归一化局部分子与局部分母。结尾只提出直接指数计算的稳定性问题，把解法留给第 24 节。

**Tech Stack:** Markdown、Python 数值复算、Git diff 范围检查

---

### Task 1: 建立教学缺口检查

**Files:**
- Modify: `docs/Week4_Attention与FlashAttention完整学习资料.md:1209`

- [ ] **Step 1: 运行修改前断言**

提取第 23 节，并断言其中存在 softmax 代入推导、scalar/vector shape 解释、tile A/B 局部分子分母、合并数值和“加法可以分批，除法不能提前”的总结。预期断言失败，证明现有小节缺少这些教学点。

### Task 2: 重写第 23 节

**Files:**
- Modify: `docs/Week4_Attention与FlashAttention完整学习资料.md:1209`

- [ ] **Step 1: 从 softmax 加权和逐行推导**

写出 `p_j=exp(s_j)/Σ_k exp(s_k)`、`O=Σ_j p_jV_j`，再把共同分母提出，得到 `O=Σ_j exp(s_j)V_j / Σ_j exp(s_j)`。

- [ ] **Step 2: 解释 shape 和除法含义**

明确 denominator 是一个标量；每项 `exp(s_j)V_j` 与 `V_j` 同 shape，求和得到向量 numerator；向量除以标量表示每个分量分别相除。

- [ ] **Step 3: 手算两个 tile**

沿用：

```text
scores = [2,1,4,3]
V = [[1,0],[0,1],[2,0],[0,2]]
```

计算：

```text
den_A = e²+e¹
num_A = [e²,e¹]
den_B = e⁴+e³
num_B = [2e⁴,2e³]
den = den_A+den_B
num = num_A+num_B
O = num/den ≈ [1.3749728,0.5058242]
```

- [ ] **Step 4: 总结分块规则并过渡**

明确“求和可以分批完成；除法只能等全局分子和分母齐全后做”。随后说明该版本直接计算指数可能溢出，引向第 24 节的共同 max 基准。

### Task 3: 验证并提交

**Files:**
- Test: `docs/Week4_Attention与FlashAttention完整学习资料.md`

- [ ] **Step 1: 运行教学点断言**

重新提取第 23 节，确认所有缺失教学点存在；预期通过。

- [ ] **Step 2: 独立复算数字**

使用 Python 计算 tile A/B 的 numerator、denominator 及合并输出，断言结果与普通 Attention `[1.3749728385179774,0.5058242394599053]` 对齐。

- [ ] **Step 3: 检查文档结构与修改范围**

确认围栏成对、Day 3/Day 4 标题各一次、`git diff --check` 无输出，并人工确认 diff 只替换第 23 节。

- [ ] **Step 4: 提交**

```bash
git add -- 'docs/Week4_Attention与FlashAttention完整学习资料.md'
git commit -m 'docs: clarify Attention numerator and denominator accumulation'
```
