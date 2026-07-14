# DeepSeek Week4（Attention）Work Log

> 对应：docs/courses/attention/Week4_Attention与FlashAttention完整学习资料.md
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
- `docs/proofs/Online_Softmax正确性证明.md` —— 完整正确性证明（增量更新与一次性算等价）
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

## Day 5（2026-07-05）：cp.async 双缓冲 + ncu 验证

### 交付
- `week04_attention/tiled_attention_pipelined.cu` —— 在 Day4 tiled 上加 cp.async 双缓冲，**ALL PASS**
  - `cuda::pipeline<thread_scope_block>` + `pipeline_shared_state<...,2>` 双缓冲
  - `k_s[2][BC][MAX_D]` / `v_s[2][BC][MAX_D]` 乒乓，`stage ^= 1`
  - 序幕预取 tile0 → 循环内「预取下一块 + 计算当前块」重叠
- `week04_attention/ncu_pipeline_notes.md` —— Day4 vs pipelined 的 ncu stall 对比

### 修掉的 bug（迭代记录）
1. `isinf(block_m && block_m<0)` 括号错位 → 传 bool 给 isinf → 选中 host-only constexpr 重载
   → 「__global__ 里调用 __host__ constexpr」编译报错。修：`isinf(block_m) && block_m<0`
2. `load_title_async` 三连错：row_base 多乘 d、列索引 `i%tileH`（应 tileW）、写地址 `+r`（应 +c）
3. m/l 未初始化（同 Day4 老坑）、dot 用 `=` 而非 `+=`
4. **双缓冲记账**：`last_valid = valid` 慢一拍，末块（N=37）用成上一块的行数 → 多算垃圾行。
   修：删 last_valid，计算用 `valid=min(BC,n-tile_step)`，预取用 `next=min(BC,n-(tile_step+valid))`
   诊断信号：N=37 causal=1 PASS 但 causal=0 FAIL → 垃圾 key 被 causal 恰好 mask 掉

### ncu 对比（N=128 D=64 causal=0，同一 launch）
| 指标 | Day4 | Pipelined | 变化 |
|------|------|-----------|------|
| long_scoreboard stall（等 global） | 6.40 | 0.03 | ↓99.5% |
| short_scoreboard stall（等 shared） | 0.60 | 0.59 | ≈不变 |
| warp latency / inst | 15.15 cyc | 6.46 cyc | ↓57% |
| SM throughput | 7.47% | 16.79% | ↑2.25× |

### 概念（从零掌握）
- double buffering = 2 个 buffer，一边算一边预取，`buf = t & 1` 乒乓
- `cuda::pipeline` 三段式：producer_acquire/commit（发起+提交异步拷贝）、consumer_wait/release（等+释放）
- pipeline 块作用域保证跨线程可见：多线程分工搬 K/V，consumer_wait 后都能读到
- buffer 覆盖安全：同 buffer 2 轮后才重用，producer_acquire 会等它的 consumer_release
- 预取行数要按「下一块起点」算，不是当前块（否则慢一拍）

### 自己悟出的洞察
- cp.async 藏的是 **global 访存延迟**（long_scoreboard），不动 shared 延迟（short_scoreboard）
  → 「一降一稳」的对照正好证明藏对了东西
- 小 N（128 block，0.13~0.30 wave）填不满 A100 → 墙钟没大加速，瓶颈转成占用率；
  但 stall 组成变化真实，证明机制有效。大加速要靠「多 query tile 填满 GPU」

### 闭卷口述（已过）
cp.async 预取下一块 K/V 到另一个 shared buffer，与当前块计算重叠 → 藏 global 访存延迟。
ncu 看 long_scoreboard 从每指令 6.4 cycle 降到 0.03，short_scoreboard 不变，证明藏的是 global
而非 shared 延迟；warp 平均等待 15→6.5 cycle，SM 吞吐翻倍。小 N 墙钟没大改是 grid 填不满 GPU。

---

## 下一步（TODO）
- [x] Day2：online softmax 写成 CUDA（running m/l + 新 max 重缩放）
- [x] Day3–4：tiled_attention.cu 教学版，ALL PASS
- [x] Day5：K/V 双缓冲 cp.async（tiled_attention_pipelined.cu）+ ncu stall 对比验证
- [ ] Day6：KV cache 与 MLA（字节账本、MHA/GQA/MLA shape）
- [ ] Day7：profiling（benchmark 表 + ncu 证据）
- [ ] 进阶：FP16 + Tensor Core（多 query tile + mma.sync），留作 CUDA 深度专项
- [ ] `fused_attention.cu` 三个 TODO（如需补 Day1 融合版）

---

## Advanced Prefill M1（2026-07-14）：Br=4 Query-tiled FP32 SIMT

### 交付

- 在 `gpu-kernel-engineering/projects/attention_prefill/` 新建独立作品项目。
- 自己实现 `Br=4, Bc=16` 的 Query-tiled FP32 SIMT Attention Kernel。
- 一个 CTA 处理最多 4 条连续 Query；同一份 K/V tile 被 4 行 Query 复用。
- 每行维护独立的 `m/l/alpha/O_acc`，支持 causal、Query tail、K/V tail 和 feature tail。
- 工程补齐 CPU double reference、launcher 合同测试、126-case shape 矩阵和 2 个特殊输入。

### 修掉的错误

1. V 搬运目标写错，导致 `v_s` 未被正确填充。
2. Score 任务用 feature 维 `D` 解码；正确任务平面是 `valid_query × valid_kv`。
3. 阶段 E 用打平任务编号访问 V；应使用解码后的 `cur_feature`。
4. Grid-stride 写回从 `tid` 解码；第二轮必须从循环变量解码。
5. causal 比较局部 Query/Key 坐标；跨 Query CTA 后必须比较全局坐标。

### 核心理解

- `Br=4` 共享的是 K/V tile，不共享每行 Online Softmax 状态。
- 阶段 E 是小型 GEMV：`weight[query, key] × V[key, feature]`，Key 是归约维，feature 是独立输出维。
- 当前采用 output-stationary：一个线程拥有一个 `(query, feature)`，独立遍历全部 Key，无需 atomic 或 Warp reduce。
- 全 mask tile 必须成为恒等更新：`alpha=1` 且 tile 权重为 0，避免 `-∞ × V` 污染输出。
- Sanitizer 只能证明地址、竞争和同步安全；合法地址上的错误索引仍要靠 CPU reference 与边界 shape 发现。

### 验证证据

- launcher 输入与 32 位设备索引合同测试通过。
- shape matrix：`N={1,3,4,5,15,16,17,31,33}`、`D={1,2,63,64,65,127,128}`、causal `{0,1}`。
- 126 个 shape case 加 `zero-qk`、`rising-logits` 两个特殊输入通过。
- 代表性多 CTA、多 K/V tile、双 tail、causal shape 通过 memcheck、racecheck、synccheck、initcheck。

### 下一步

1. 先完成 M1 canonical CUDA Event benchmark、ncu 和 SASS 证据，不立即改线程映射。
2. M1 完整收口后进入 M2 Warp-per-query：一个 Warp 固定一条 Query，lane 负责 feature，并探索寄存器 `O_acc`。
3. 简版路线图见 `gpu-kernel-engineering/projects/attention_prefill/ROADMAP.md`。
4. 当前教材继续看 `docs/courses/attention/M1_Br4_QueryTiled_FP32_SIMT_Attention完整学习资料.md` 第 57～64 节。

### 一句话记忆

> 线性索引先认清任务平面，causal 必须回到全局坐标；K/V tile 可以共享，但每条 Query 的 Online Softmax 状态必须独立。
