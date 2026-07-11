# Week5 Transformer Block 小白向扩写 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Week5 教材 2.4 扩写成没有 ML 基础也能沿 shape 和数据流读懂的 Transformer block 入门章节。

**Architecture:** 只替换现有 2.4，不改变后续章节编号。先用完整数据流建立整体认识，再逐步解释两个子层、两次归一化和两次残差，最后连接 prefill/decode 与本周 CUDA 任务。

**Tech Stack:** Markdown、Transformer decoder block、A100 推理语境

---

### Task 1: 扩写 Transformer block 数据流

**Files:**
- Modify: `docs/Week5增强版_LLM推理优化与decode.md:111-125`
- Reference: `docs/Week4_Attention与FlashAttention完整学习资料.md`

- [ ] **Step 1: 记录修改前基线**

运行：

```bash
sed -n '100,150p' docs/Week5增强版_LLM推理优化与decode.md
```

预期：2.4 只有简略数据流，2.5 紧接 Logits。

- [ ] **Step 2: 写整体结构与逐步解释**

将 2.4 扩展为以下组成：

1. `X[B,N,D]` 输入与同 shape 输出；
2. 第一条 Attention 子层：RMSNorm、Q/K/V、KV Cache、Attention 输出、残差；
3. 第二条 MLP 子层：RMSNorm、升维、激活/门控、降维、残差；
4. 说明残差相加要求主分支最终回到 `D`。

- [ ] **Step 3: 加入 shape 跟踪**

用 `B=1,N=3,D=4` 表格追踪 prefill，并用 `B=1,N=1,D=4` 对比 decode。明确 decode 只新增一个位置，但 Attention 读取历史 KV。

- [ ] **Step 4: 加入 CUDA 对照与验收**

将线性投影对应到 Day 2/3 GEMV/GEMM，将 RMSNorm、Residual、Activation 对应到 Day 4，将 KV 管理对应到 Day 7。加入常见误区和一段闭卷口述。

### Task 2: 文档验证与提交

**Files:**
- Test: `docs/Week5增强版_LLM推理优化与decode.md`

- [ ] **Step 1: 检查结构**

确认 2.4 后仍直接连接 2.5，Markdown 代码围栏成对，且存在 `[B,N,D]`、`[B,1,D]`、`KV Cache`、`residual`、`MLP` 等关键词。

- [ ] **Step 2: 检查链接**

确认 Week4 Attention 学习资料的本地链接目标存在。

- [ ] **Step 3: 检查修改范围**

运行：

```bash
git diff --check -- docs/Week5增强版_LLM推理优化与decode.md
git diff --stat -- docs/Week5增强版_LLM推理优化与decode.md
```

预期：只修改目标教材。

- [ ] **Step 4: 提交**

```bash
git add -- docs/Week5增强版_LLM推理优化与decode.md
git commit -m 'docs: expand Week5 Transformer block introduction'
```
