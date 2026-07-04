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

## Day 2（2026-07-04）：Online Softmax

### 交付
- `docs/Online_Softmax正确性证明.md` —— 完整正确性证明（增量更新与一次性算等价）
- `week04_attention/online_softmax.cu` —— 自己手写 CUDA 单行版
  - running `m/l`：遍历时维护最大值 m 和相对 m 的指数和 l
  - 新 max 出现 → 旧 l 乘 `alpha = exp(m_old - m_new)` 再加本段贡献
  - block reduction 求 max/sum（warp shuffle + shared 广播）

### 概念（从零掌握）
- 空状态约定 `(-INF, 0)`：初始 m=-INF、l=0；第一段有效数据时 alpha 要给 0，
  否则 `-INF-(-INF)=NaN` 会污染结果
- `__shared__` 不能直接放自定义 `Pair`（含构造/非平凡类型有坑）→ 拆成两个标量
  `s_m`/`s_l` 更稳妥
- warp reduction 收尾：`num_warp = blockDim.x/32`，只有 `lane_id < num_warp` 的线程
  从 shared 读回各 warp 的部分结果，其余填单位元
- online softmax 正确性核心：分子 acc 和分母 l 乘同一个 alpha，相除时抵消 → 与一次性算等价

### 自己悟出的洞察
- 重缩放的本质是"换基准"：旧值都带 e^{-m_old}，m 一升级就集体过期，乘 alpha 折算到 m_new
- 增量式的意义：永不落 N×N，只维护常数个 running 状态

### 闭卷口述（已过）
新 max 出现为何重缩放：旧 l/acc 相对旧 m，m 变大后每项该更小，乘 exp(m_old-m_new)∈(0,1] 对齐；
分子分母同乘同因子，最终相除抵消。

---

## Day 3–4（2026-07-04）：FlashAttention 数据流 + 教学版 tiled

### 交付
- `week04_attention/tiled_attention.cu` —— 自己填 7 个 TODO，**ALL PASS**（误差 ~1e-7）
  - 一个 block 处理一条 query，Q 放 shared
  - K/V 按 `BC=16` 分块遍历，全程只维护 `m/l/acc` 三个 running 状态，永不存 N×N
  - 支持任意 N（非整除）、可选 causal

### 测试结果
```text
N=3    D=2   causal=0                  PASS
N=8    D=8   causal=0/1                PASS
N=37   D=24  causal=0/1（非整除）       PASS
N=128  D=64  causal=0/1                PASS
ALL PASS（误差 ~1e-7）
```

### 修掉的 3 个 bug（初版没跑过）
1. **TODO 0 没初始化 `acc/m/l`** → shared 垃圾值污染；补 `acc[i]=0`、`tid==0: m=-INF,l=0`
2. **TODO 2 无视 `causal` 开关**（无条件 mask）→ 非 causal 用例全错；改 `if (causal && key_id>query)`
3. **TODO 6 后缺 `__syncthreads()`** → 下一 tile 提前覆盖仍在读的 `v_s`（data race）

### 概念（从零掌握）
- fused vs tiled：fused 一次性把整行 K/V 读进来（scores 只在 shared[N]）；
  tiled 连 K/V 也分块，才是完整 FlashAttention 数据流
- 协作加载 K/V tile：`BC ≠ 线程数`，用 `linear = tid; linear < valid*d; += blockDim.x`
  展平搬运，`j=linear/d, x=linear%d`
- 一线程一 score：`tid<valid` 各算一个 key 的点积（这步线程利用率低，是教学版取舍）
- `scores` 原地改成 `exp(score-m_new)`：省一份缓冲，后面 PV 累加直接当权重用
- acc 更新：`acc[x] = alpha*acc[x] + Σ_j scores[j]*v_s[j][x]`，旧 acc 也要乘 alpha
- 输出累加按 feature 维并行（acc[x] 各维独立，零写冲突）；score 按 key 维并行 → 两步并行轴不同
- `isinf(m)` 判断 = "是不是第一个有效 tile"：是→无旧状态，alpha=0 直接覆盖；
  防 `-INF-(-INF)=NaN`
- causal mask：query 位置 i 只能看 key 位置 j≤i；j>i 写 -∞（softmax 后权重=0）；
  `causal=false` 时不 mask（双向 attention，如 BERT）

### 自己悟出的洞察
- 加载 Q 时 `d < blockDim.x`，只有前 d 个线程干活——一次性廉价加载，不是瓶颈；
  真正闲的是"一线程一 key"的 score 步
- 教学版故意串行归约 + 单 query，为清晰牺牲性能；工业版靠"多 query tile + warp reduction
  + Tensor Core + 双缓冲"补回来（Day5 起）

### 闭卷口述（已过）
一个 block 一条 query，Q 放 shared，以 16 个 key 为一块遍历 K/V；每块只在 shared 存 16 个 score
并更新 running m/l/acc，处理完覆盖，故不需 N×N；新 max 出现时 l 和 acc 同乘 alpha，最后 out=acc/l。
旧输出也要缩放，因为它相对旧 m，m 升级后必须换基准。

---

## 下一步（TODO）
- [x] Day2：online softmax 写成 CUDA（running m/l + 新 max 重缩放）
- [x] Day3–4：tiled_attention.cu 教学版，ALL PASS
- [ ] Day5：A100 优化二选一（FP16+Tensor Core 或 K/V 双缓冲 cp.async），精度+性能分开验证
- [ ] Day6：KV cache 与 MLA（字节账本、MHA/GQA/MLA shape）
- [ ] Day7：profiling（benchmark 表 + ncu 证据）
- [ ] `fused_attention.cu` 三个 TODO（如需补 Day1 融合版）
