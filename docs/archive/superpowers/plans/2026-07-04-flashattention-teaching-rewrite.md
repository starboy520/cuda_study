# FlashAttention 教学章节重写 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Week 4 资料中的 FlashAttention 部分改写成适合只懂少量 CUDA、刚接触 Attention 的渐进式教程。

**Architecture:** 保留 Day 1、Day 2 与后续课程主体，只替换 Day 3，并微调 Day 4 的开头。Day 3 使用同一组一行四列的数字贯穿普通 Attention、错误分块、正确分块、online softmax 与 A100 内存映射，避免在不同例子间切换。

**Tech Stack:** Markdown、基础线性代数、CUDA/A100 内存层次、Shell 文档检查

---

### Task 1: 建立重写前的结构检查

**Files:**
- Modify: `docs/Week4_Attention与FlashAttention完整学习资料.md:1137`

- [ ] **Step 1: 记录当前文档结构基线**

运行一个 Python 脚本，确认 Day 3、Day 4 标题存在且代码围栏成对；预期输出 `baseline PASS`。

- [ ] **Step 2: 保存 Day 3 前后边界以便精确替换**

运行：

```bash
rg -n '^# Day 3：|^# Day 4：' 'docs/Week4_Attention与FlashAttention完整学习资料.md'
```

预期两个标题各出现一次，Day 3 位于 Day 4 之前。

### Task 2: 用单一数字例子重写 Day 3

**Files:**
- Modify: `docs/Week4_Attention与FlashAttention完整学习资料.md:1137`

- [ ] **Step 1: 写普通 Attention 的完整基准计算**

使用 scores `[2,1,4,3]` 和二维 value：

```text
V_0=[1,0], V_1=[0,1], V_2=[2,0], V_3=[0,2]
den = e^-2 + e^-3 + 1 + e^-1 ≈ 1.553002
O ≈ [1.374948, 0.505965]
```

- [ ] **Step 2: 演示错误分块方法**

将前两个和后两个 key 分别 softmax，说明每个 tile 的概率都被强制归一化为 1，错误根因是两个 tile 使用了两个互不相同的分母。

- [ ] **Step 3: 先推导不考虑稳定性的累计分子与分母**

从 `O = Σexp(s_j)V_j / Σexp(s_j)` 定义累计 `den_acc` 和向量 `num_acc`，说明 tile 只是在分批完成同一个加法。

- [ ] **Step 4: 加入 m/l/O_acc 的稳定表示**

逐一说明：

```text
m     = 已扫描 score 的最大值
l     = Σ exp(score-m)
O_acc = Σ exp(score-m)V
```

使用“旧权重换到新 max 的计量基准”推导 `alpha`、`l_new` 与 `O_acc_new`。

- [ ] **Step 5: 用表格逐 tile 算完同一示例**

表格包含 tile、`m_block`、`m_new`、`alpha`、`l`、`O_acc`。最终计算 `O_acc/l`，与普通 Attention 数值结果对齐。

- [ ] **Step 6: 从单行例子推广到矩阵 tile**

在读者理解状态后，再给出 `[Br,Bc]`、`[Br]`、`[Br,Dv]` 的矩阵公式和伪代码，明确每个 query 行都有独立状态。

### Task 3: 补齐 IO 直觉与 A100 映射

**Files:**
- Modify: `docs/Week4_Attention与FlashAttention完整学习资料.md`

- [ ] **Step 1: 对比朴素与分块数据生命周期**

写出 `S/P` 在朴素三 kernel 方案中落入 HBM 的路径，以及局部 `S_tile/P_tile` 只在片上短暂存在的路径。

- [ ] **Step 2: 添加 A100 变量落点表**

表格包含 Q/K/V、当前 tile、`m/l`、`O_acc`、最终 O，注明典型位置为 HBM、shared memory 或 registers，并说明工业实现会随调度变化。

- [ ] **Step 3: 分开解释三种复杂度**

明确区分计算交互仍为 `O(N²)`、显式 `S/P` 中间存储降低，以及 HBM IO 因 tiling 和中间矩阵不落盘而减少。

- [ ] **Step 4: 保留 exact、causal 与常见误解**

把原章节中的 exact 结论、三类 causal tile、全 mask 边界和五个误解放在直觉与推导之后。

### Task 4: 调整 Day 4 的学习入口

**Files:**
- Modify: `docs/Week4_Attention与FlashAttention完整学习资料.md`

- [ ] **Step 1: 添加数学变量到 CUDA 存储的回扣**

加入以下映射：

```text
Day 3 的一行 query → 教学 kernel 的一个 block
该行的 m/l        → shared 标量
该行的 O_acc[D]   → shared acc[D]
当前 K/V tile     → k_s/v_s
```

- [ ] **Step 2: 保留七个必须手写的空位**

不提供完整 CUDA 答案，只增强每个空位与 Day 3 数学步骤的对应说明。

### Task 5: 验证文档正确性并提交

**Files:**
- Test: `docs/Week4_Attention与FlashAttention完整学习资料.md`

- [ ] **Step 1: 验证数字示例**

使用 Python 按 scores 和 values 重新计算 `l` 与输出，断言结果分别为 `1.553001792775919`、`1.3749481276310194`、`0.5059651263693235`，预期输出 `numeric example PASS`。

- [ ] **Step 2: 验证章节、围栏和关键教学点**

确认 Day 3、Day 4 各出现一次、顺序正确、代码围栏成对，并检查“错误做法”、`O_acc`、HBM、A100、exact 和“七个空位”等关键内容存在；预期输出 `document structure PASS`。

- [ ] **Step 3: 检查修改范围和空白错误**

```bash
git diff --check
git diff --stat -- 'docs/Week4_Attention与FlashAttention完整学习资料.md'
```

预期 `git diff --check` 无输出，正文修改仅涉及目标文档。

- [ ] **Step 4: 提交正文修改**

```bash
git add -- 'docs/Week4_Attention与FlashAttention完整学习资料.md'
git commit -m 'docs: make FlashAttention tutorial more approachable'
```
