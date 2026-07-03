# DeepSeek Week4（Attention）Work Log

> 对应：docs/Week4_Attention与FlashAttention完整学习资料.md
> 硬件：A100 80GB PCIe（sm_80）
> 起点：CUDA/GEMM 熟，LLM/Attention 零基础

---

## Day 1（2026-07-03）：朴素 CUDA Attention

### 交付
- `week04_attention/naive_attention.cu` —— 手写三个 kernel，7 组 shape 全 PASS
  - `qk_scores`：scores[i,j]=Q第i行·K第j行 × rsqrtf(d)，二维 grid，含 causal（j>i→-INF）
  - `row_softmax`：一 block 一行，`blockReduceMaxF/SumF` + shared 广播 max/denom，三遍
  - `pv_output`：out[i,x]=Σ_j probs[i,j]·V[j,x]，正宗 GEMM（probs 行 × V 列）
- `compute-sanitizer --tool memcheck` 0 errors
- `week04_attention/fused_attention.cu` —— 融合版脚手架（三个 TODO 未填，属 Day4）

### 测试结果
```text
N=3   D=2  causal=0  max_abs=5.96e-08  PASS
N=8   D=8  causal=0/1                  PASS
N=37  D=24 causal=0/1（非整除）         PASS
N=128 D=64 causal=0/1                  PASS
ALL PASS（误差 ~1e-7，FP32 舍入级）
```

### 概念（从零掌握）
- token/sequence/N/batch：一句话切成 N 个 token 组成 sequence；X=[N,D] 是 token→embedding 后的矩阵
- 投影=GEMM：X第i行×整个Wq=Q第i行，每 token 独立；query 与 token 一对一（N token=N query）
- softmax 沿 key 维：对每个 query 的 **N 个分数**（不是 d 维）归一化，每行和为 1
- 除 sqrt(Dh)：点积累加 Dh 项，方差∝Dh；不除→logits 过大→softmax 饱和/接近 one-hot→梯度消失
- causal mask：只看过去和自己，未来 j>i 写 -∞（不是 0；exp(0)=1 仍有权重）
- 多头：每个 head 投影权重不同→投到不同子空间；实现=大 Wq 一次算完再切 H 段

### 自己悟出的洞察
- QK 是"行·行"（K 不用真转置，读 k[j*d+x] 即 K 第 j 行）；PV 是"行·列"（正宗 GEMM）
- attention 是"可融合"链（GEMM→softmax→GEMM），和孤立 GEMM 不同
- 融合=一个 block 一条 query，一条龙走完，scores 只在 shared[N] 不落 HBM
- naive 版真瓶颈是 K/V 无复用（每 query 重读整个 K/V），不是访存合并；
  coalescing 看 warp 瞬时地址而非单线程轨迹→映射到连续维就合并

### 术语
logits（softmax 前的原始分数）、one-hot（只一个位置=1）、scaling factor（sqrt(Dh)）、
softmax saturation（饱和）、vanishing gradient（梯度消失）、coalescing（合并访问）、fusion（算子融合）

### 闭卷口述（已过）
QK^T→[N,N]（N query×N key）；K 按行读=K^T 列，不用真转置；除 sqrt(Dh) 稳定量级；
softmax 沿 key 对 N 个分数做；causal 写 -∞ 才能 exp 后为 0。

---

## 下一步（TODO）
- [ ] `fused_attention.cu` 三个 TODO（融合版，Day4）
- [ ] Day2：online softmax 写成 CUDA（理论已懂：running m/l + 新 max 重缩放）
