# M1：Br=4 Query-Tiled FP32 SIMT Attention 完整学习资料

> 本文只研究单 batch、单 head、FP32、forward-only 的教学版 Attention。
> 起点是已经正确实现 `Br=1, Bc=16` 的 one-CTA-per-query Online Tiled Attention。
> 本阶段把一个 CTA 负责的 query 行数从 1 增加到 4，让同一份 K/V tile 被 4 条连续 query 复用。

---

## 0. 先把一句话目标说清楚

原版本：

```text
一个 CTA 固定一条 query
→ 依次加载所有 K/V tile
→ 为这一行维护 m / l / O_acc
```

M1 版本：

```text
一个 CTA 固定四条连续 query
→ 每次只加载一份 K/V tile
→ 四条 query 分别使用它
→ 每一行仍有独立的 m / l / alpha / O_acc
```

一句话记忆：

> `Br=4` 只是在一个 CTA 中同时处理 4 条 query 行，以复用 K/V；它没有把 4 行合并成一个 softmax。

这里的 `Br=4`：

- 是 4 条连续 query 行；
- 不是 4 个 attention head；
- 不是 MQA；
- 不是 GQA；
- 也不是 4 个 warp 各负责一行。

## 1. 学习目标

完成本阶段后，你应当能够：

1. 从 `Br=1` 的逐行在线状态推广到 `Br=4` 的逐行独立状态；
2. 写出 `Q_tile[4,D] × K_tile[16,D]^T` 对应的 64 个 score；
3. 正确处理 query、key、feature 三种 tail 同时出现；
4. 正确处理 causal tile 中逐行不同的有效区域；
5. 设计一个简单、可验证的 128-thread SIMT 映射；
6. 解释 K/V 请求加载量为什么理论上接近减少到原来的四分之一；
7. correctness 完成后，初步使用 sanitizer、benchmark、Nsight Compute 和 SASS 分别验证不同问题；
8. 为下一阶段 M2 warp-per-query 保留正确、清晰的数学基线。

### 1.1 推荐阅读顺序

不要要求自己第一遍就理解全部 2000 多行。按三层推进：

1. **核心必读**：第 4～40 节。先理解二维 tile、逐行状态、tail、线程映射和同步。
2. **开始实现**：第 41～56 节。严格按 Checkpoint 写测试、实现和运行 sanitizer。
3. **正确后进阶**：第 57～65 节。再学习 benchmark、Nsight Compute、occupancy 和 SASS。

第一次阅读遇到以下术语可以先跳过：

- `occupancy`：一个 SM 同时驻留多少活跃 warp 的比例；
- `sector`：GPU Cache/内存事务统计中的传输单位；
- `carveout`：L1 与 Shared Memory 容量分配配置；
- `FFMA contraction`：编译器把乘法与加法合并为一条融合指令；
- `HMMA`：Tensor Core 矩阵乘机器指令；
- `LDGSTS`：Ampere global-to-shared 异步复制机器指令。

M1 的核心任务是先写对 `Br×Bc` 数据流；这些性能术语不是开始编码的前置门槛。

## 2. 前置知识

开始前应当已经理解：

- 标准 Attention：`softmax(QK^T / sqrt(D))V`；
- stable softmax 为什么要减 row max；
- online softmax 的 `m`、`l` 与 `alpha`；
- `O_acc` 是未归一化的向量分子；
- shared memory tile、row-major offset 和 `__syncthreads()`；
- `Br=1, Bc=16` 教学版为何不保存完整 `N×N` score/probability；
- CUDA 中 block 内线程不能在 barrier 前后随意提前返回。

若下面这句话还不能闭卷解释，应先复习 `Br=1`：

> 新 tile 带来更大的 row max 时，旧 `l` 和旧 `O_acc` 必须同时乘 `alpha=exp(m_old-m_new)`，因为二者都要换到以 `m_new` 为基准的指数尺度。

## 3. 本阶段明确不做什么

M1 故意不加入以下内容：

- Tensor Core、WMMA、MMA 或 HMMA；
- `cp.async`、双缓冲或多 stage pipeline；
- warp-per-query；
- FP16、BF16、TF32 或其他低精度路径；
- 多 batch、多 head、MQA、GQA；
- backward；
- dropout、bias、RoPE 融合；
- 工业级 shared-memory swizzle；
- 工业级寄存器 fragment 布局。

本阶段只使用 FP32 scalar FMA 和普通 cooperative load。先把 `Br=4` 的状态、索引、tail 与同步写对，再改变并行粒度。

---

# 第一部分：从 Br=1 走到 Br=4

## 4. Br=1 数据流复盘

`Br=1, Bc=16` 时，一个 CTA 固定 query `q`：

```text
Q[q,:] 只加载一次

for k0 = 0, 16, 32, ...:
    加载 K[k0:k0+16,:]
    加载 V[k0:k0+16,:]
    计算 16 个 score
    更新这一行的 m / l / alpha / O_acc

O[q,:] = O_acc / l
```

这一版的优点是状态很直观：

```text
m       1 个标量
l       1 个标量
alpha   1 个标量
O_acc   D 个 FP32
```

但它有一个明显的 K/V 复用限制。

## 5. Br=1 的 K/V 复用限制

假设 `N=4`，忽略 cache：

```text
CTA 0 处理 Q0：加载整份 K/V
CTA 1 处理 Q1：再次加载整份 K/V
CTA 2 处理 Q2：再次加载整份 K/V
CTA 3 处理 Q3：再次加载整份 K/V
```

虽然 4 条 query 都要读取同一份 K/V，但 K/V 只在各自 CTA 的 shared memory 内复用。不同 CTA 不能共享 shared memory，因此源代码层面的 K/V global load 请求会重复 4 次。

当 `Br=4` 时：

```text
CTA 0 同时处理 Q0、Q1、Q2、Q3
→ 每个 K/V tile 只合作加载一次
→ 4 条 query 都从同一个 shared K/V tile 读取
```

对非 causal、满 query tile 的理想请求量：

```text
Br=1 的 K/V float 请求数 ≈ 2 × N × N × D
Br=4 的 K/V float 请求数 ≈ 2 × ceil(N/4) × N × D
```

当 `N` 可被 4 整除时，第二式正好是第一式的四分之一。

> [!IMPORTANT]
> 这是“源代码请求加载量”的账本，不等于 DRAM 实际传输字节必然下降 4 倍。Br=1 的重复加载可能命中 L2；内存事务合并、cache、调度和 causal 跳块都会影响 Nsight Compute 看到的 DRAM bytes。

## 6. Br 与 Bc 各自控制什么

把局部 score tile 写成：

```text
S_tile shape = [Br, Bc]
```

其中：

- `Br`：一个 CTA 同时固定多少条 query 行；
- `Bc`：一次加载多少条 key/value 行；
- `D`：每个 query/key/value 向量的 feature 数。

本阶段固定：

```text
Br = 4
Bc = 16
blockDim.x = 128
1 <= D <= 128
```

### 6.1 一个 Br=2、Bc=3 的小图

```text
query tile: Q2, Q3
key tile:   K6, K7, K8

score tile:
             K6      K7      K8
Q2         S[2,6]  S[2,7]  S[2,8]
Q3         S[3,6]  S[3,7]  S[3,8]
```

这里有 6 个 score，但只有 2 组 softmax 状态：

```text
Q2 有自己的 m[0], l[0], O_acc[0,:]
Q3 有自己的 m[1], l[1], O_acc[1,:]
```

`K6/K7/K8` 被两行 query 共同读取，但两行的 score、max、分母与输出绝不能混合。

### 6.2 Br=4、Bc=16 的含义

一个满 tile 有：

```text
4 条 query × 16 条 key = 64 个 score
```

四行分别做长度为 16 的局部处理：

```text
row 0: scores[0, 0..15]
row 1: scores[1, 0..15]
row 2: scores[2, 0..15]
row 3: scores[3, 0..15]
```

一句话记忆：

> K/V tile 是共享的，online softmax 状态是逐行私有的。

---

# 第二部分：完整数字手算

## 7. 手算设置：Br=2、Bc=2、D=2

为了把“多 query 独立状态”看清楚，先用比目标更小的 tile：

```text
Br = 2
Bc = 2
D  = 2
N  = 4
```

缩放因子：

```text
scale = 1 / sqrt(D) = 1 / sqrt(2)
```

选择：

```text
Q0 = [sqrt(2), 0]
Q1 = [0, sqrt(2)]

K0 = [1,  0]
K1 = [0,  1]
K2 = [2,  0]
K3 = [0, -1]

V0 = [1, 0]
V1 = [0, 1]
V2 = [2, 1]
V3 = [1, 2]
```

这样缩放后的 score 恰好是简单整数：

```text
S[r,j] = dot(Qr, Kj) / sqrt(2)
```

完整 score 矩阵为：

```text
          K0   K1   K2   K3
Q0        1    0    2    0
Q1        0    1    0   -1
```

两个 K/V tile：

```text
tile A: K0,K1 与 V0,V1
tile B: K2,K3 与 V2,V3
```

这组数字故意让第二个 tile 到来时：

- Q0 看到新的更大 max，`alpha<1`；
- Q1 没有看到新的更大 max，`alpha=1`。

这就是多行独立状态最重要的直觉。

## 8. 初始化两行状态

每一行单独初始化：

```text
m[0] = -∞, l[0] = 0, O_acc[0] = [0,0]
m[1] = -∞, l[1] = 0, O_acc[1] = [0,0]
```

第一次遇到有效 score 时，没有旧贡献，因此约定：

```text
alpha = 0
```

IEEE 浮点下 `exp(-∞ - finite)=0`，但实现仍应显式识别空状态，使首次有效 tile 的语义清楚，并与“空状态遇到全 mask 会产生 `-∞-(-∞)`”的危险情况区分开。

## 9. 处理 tile A：Q0 这一行

Q0 对 K0、K1 的 score：

```text
scores = [1, 0]
m_block = 1
m_new = max(-∞, 1) = 1
alpha = 0
```

相对 `m_new=1` 的未归一化权重：

```text
w0 = exp(1-1) = 1
w1 = exp(0-1) = e^-1 ≈ 0.3678794412
```

更新分母：

```text
l[0] = 0 × 0 + 1 + e^-1
     ≈ 1.3678794412
```

更新向量分子：

```text
O_acc[0]
= 0 × [0,0] + 1×V0 + e^-1×V1
= [1, e^-1]
≈ [1.0000000000, 0.3678794412]
```

最终暂存：

```text
Q0 state after tile A:
m=1
l≈1.3678794412
O_acc≈[1.0000000000, 0.3678794412]
```

## 10. 处理 tile A：Q1 这一行

Q1 对 K0、K1 的 score：

```text
scores = [0, 1]
m_block = 1
m_new = 1
alpha = 0
```

权重：

```text
w0 = exp(0-1) = e^-1
w1 = exp(1-1) = 1
```

更新：

```text
l[1] = e^-1 + 1
     ≈ 1.3678794412

O_acc[1]
= e^-1×V0 + 1×V1
= [e^-1, 1]
≈ [0.3678794412, 1.0000000000]
```

注意：两行此时的 `m`、`l` 恰好相同，但 `O_acc` 已经不同。不能因此合并状态。

## 11. 处理 tile B：Q0 出现新 max

Q0 对 K2、K3 的 score：

```text
scores = [2, 0]
m_block = 2
m_old = 1
m_new = 2
alpha[0] = exp(1-2) = e^-1 ≈ 0.3678794412
```

新 tile 权重：

```text
w2 = exp(2-2) = 1
w3 = exp(0-2) = e^-2 ≈ 0.1353352832
```

旧分母换基准：

```text
alpha × l_old
= e^-1 × (1+e^-1)
= e^-1 + e^-2
≈ 0.5032147244
```

新分母：

```text
l[0]
= 0.5032147244 + 1 + 0.1353352832
≈ 1.6385500076
```

旧向量分子换基准：

```text
alpha × [1, e^-1]
= [e^-1, e^-2]
≈ [0.3678794412, 0.1353352832]
```

新 tile 的向量贡献：

```text
1×V2 + e^-2×V3
= [2,1] + [e^-2, 2e^-2]
≈ [2.1353352832, 1.2706705665]
```

合并：

```text
O_acc[0]
≈ [0.3678794412, 0.1353352832]
 + [2.1353352832, 1.2706705665]
≈ [2.5032147244, 1.4060058497]
```

最终状态：

```text
m[0] = 2
l[0] ≈ 1.6385500076
O_acc[0] ≈ [2.5032147244, 1.4060058497]
```

## 12. 处理 tile B：Q1 的 max 不变

Q1 对 K2、K3 的 score：

```text
scores = [0, -1]
m_block = 0
m_old = 1
m_new = 1
alpha[1] = exp(1-1) = 1
```

新 tile 权重必须相对本行自己的 `m_new=1`：

```text
w2 = exp(0-1)  = e^-1 ≈ 0.3678794412
w3 = exp(-1-1) = e^-2 ≈ 0.1353352832
```

更新分母：

```text
l[1]
= 1×(1+e^-1) + e^-1 + e^-2
= 1 + 2e^-1 + e^-2
≈ 1.8710941656
```

更新向量分子：

```text
O_acc[1]
= 1×[e^-1,1] + e^-1×V2 + e^-2×V3

= [e^-1,1]
  + [2e^-1,e^-1]
  + [e^-2,2e^-2]

≈ [0.3678794412, 1.0000000000]
  + [0.7357588823, 0.3678794412]
  + [0.1353352832, 0.2706705665]

≈ [1.2389736067, 1.6385500076]
```

最终状态：

```text
m[1] = 1
l[1] ≈ 1.8710941656
O_acc[1] ≈ [1.2389736067, 1.6385500076]
```

## 13. 最终归一化

两行分别做向量除以本行标量：

```text
O0 = O_acc[0] / l[0]
   ≈ [2.5032147244, 1.4060058497] / 1.6385500076
    ≈ [1.527701, 0.858079]

O1 = O_acc[1] / l[1]
   ≈ [1.2389736067, 1.6385500076] / 1.8710941656
    ≈ [0.662165, 0.875718]
```

本例最该记住的不是末位小数，而是：

| 行 | tile B 的 `m_old` | `m_block` | `m_new` | `alpha` |
| --- | ---: | ---: | ---: | ---: |
| Q0 | 1 | 2 | 2 | `e^-1` |
| Q1 | 1 | 0 | 1 | `1` |

同一个 K/V tile 到来时，不同行可以有不同 `alpha`。因此 `alpha` 必须是 `[Br]`，不能是整个 CTA 共用的一个标量。

## 14. 用标准 Attention 反查手算

标准 Attention 对 Q0 的完整 score 行是 `[1,0,2,0]`。减去全局 max 2：

```text
权重分子 = [e^-1, e^-2, 1, e^-2]
分母     = e^-1 + e^-2 + 1 + e^-2
         ≈ 1.6385500076
```

这正是 online 结果的 `l[0]`。

Q1 的完整 score 行是 `[0,1,0,-1]`。减去全局 max 1：

```text
权重分子 = [e^-1, 1, e^-1, e^-2]
分母     = 1 + 2e^-1 + e^-2
         ≈ 1.8710941656
```

这正是 `l[1]`。

分块改变了求和顺序，没有改变标准 Attention 的定义。

### 14.1 即时关卡：先停 5 分钟

不要继续向下读，先遮住上面的结果，在纸上重新填写：

| 行 | tile A 后的 `m/l/O_acc` | tile B 的 `m_block` | `m_new` | `alpha` | tile B 后的 `l/O_acc` |
| --- | --- | ---: | ---: | ---: | --- |
| Q0 | | | | | |
| Q1 | | | | | |

检查重点不是小数最后一位，而是：

1. Q0 与 Q1 是否使用各自的 `m_old`；
2. Q0 的旧 `l/O_acc` 是否乘了 `e^-1`；
3. Q1 的 `alpha=1` 是否仍保留旧状态；
4. 两行是否分别用自己的 `l` 做最终归一化。

四点都能独立推导后，再进入公式和索引章节。

---

# 第三部分：数学状态与 mask

## 15. 标准 Attention 公式

对单 batch、单 head、`Q/K/V` 都为 `[N,D]`：

```text
S[i,j] = sum_d Q[i,d] × K[j,d] / sqrt(D)
P[i,j] = exp(S[i,j]) / sum_t exp(S[i,t])
O[i,d] = sum_j P[i,j] × V[j,d]
```

causal 时：

```text
S[i,j] = -∞, if j > i
```

每个 query 行 `i` 沿 key 维 `j` 做一组独立 softmax。

## 16. Br=4 的逐行 Online Softmax 状态

对 CTA 内局部 query 行 `r=0..q_valid-1`：

```text
global_q = q0 + r
```

每一行独立保存：

```text
m[r]          已处理 key 的最大 score
l[r]          sum exp(score - m[r])
alpha[r]      本 tile 的旧状态换基准因子
O_acc[r,d]    sum exp(score - m[r]) × V[key,d]
```

当前 K/V tile 产生 `scores[r,c]`，其中 `c=0..k_valid-1`。逐行更新：

```text
m_block[r] = max_c scores[r,c]
m_new[r]   = max(m[r], m_block[r])
alpha[r]   = exp(m[r] - m_new[r])
w[r,c]     = exp(scores[r,c] - m_new[r])

l_new[r]
= alpha[r] × l[r] + sum_c w[r,c]

O_acc_new[r,d]
= alpha[r] × O_acc[r,d]
  + sum_c w[r,c] × V_tile[c,d]
```

最后：

```text
O[q0+r,d] = O_acc[r,d] / l[r]
```

## 17. 局部全 mask 行的统一安全语义

某个 query 行在当前 tile 可能没有任何合法 key。此时：

```text
m_block = -∞
```

当旧状态也为空时，若直接做：

```text
m_new = max(-∞, -∞)
alpha = exp(-∞ - -∞)
```

中间会出现 `NaN`。若旧状态已经非空，通用公式可得到 `alpha=1` 和零贡献；但显式 all-mask 分支能统一两种情况并避免空状态的非法运算。

本教程规定局部全 mask 行使用：

```text
alpha = 1
本 tile 的所有 weights = 0
m / l / O_acc 完全保持原值
```

一句话记忆：

> 没有新数据，就不要重新解释旧状态；旧状态乘 1，新贡献为 0。

这条规则既适用于已经积累过旧状态的行，也适用于当前仍为空的行。标准 causal self-attention 的整条全局行至少可以看到自身，但某个局部 K tile 仍可能对该行全 mask。

## 18. causal mask 的逐元素判定

局部坐标：

```text
r = local query row
c = local key row
```

全局坐标：

```text
query = q0 + r
key   = k0 + c
```

causal 合法条件：

```text
key <= query
```

因此：

```text
masked = causal && (k0 + c > q0 + r)
```

不能用局部条件 `c>r`，除非 Q tile 与 K tile 恰好有相同起点；对一般 tile，它是错误的。

## 19. 4×16 causal mask 几何

为了在一张图中同时展示“局部全 mask 行”和“部分合法行”，先看一个**脱离当前 launcher 对齐约束的假想窗口**：

```text
query 行：q = 14,15,16,17
key 列：  k = 16..31
```

只画前几列，`✓` 合法，`×` 被 mask：

```text
             k16  k17  k18  k19  ...  k31
q14           ×    ×    ×    ×   ...   ×   全 mask
q15           ×    ×    ×    ×   ...   ×   全 mask
q16           ✓    ×    ×    ×   ...   ×   部分合法
q17           ✓    ✓    ×    ×   ...   ×   部分合法
```

四行虽然共享同一个 K/V tile，但有效 key 数分别是：

```text
0, 0, 1, 2
```

所以四行的 `m_block`、`tile_l`、`alpha` 与 `O_acc` 更新行为都可能不同。

> [!NOTE]
> 固定 `q0=4×blockIdx.x`、`k0=16×tile_id` 时，`q0=14` 不可能由当前 launcher 产生。这个假想移位窗口只用于直观看见“不同 Query 行的 mask 状态可以不同”，不能当作当前 M1 的实际测试输入。实现仍应使用全局条件 `key<=query`，避免把当前 tile 对齐性质写死。

当前 launcher 实际可产生的对角线窗口例如：

```text
q0=12, k0=16 → query 12..15 均为局部全 mask
q0=16, k0=16 → query 16..19 分别有 1、2、3、4 个合法 key
```

因此 M1 的实际测试应至少覆盖这两个窗口。未来调整 `Br/Bc` 或 tile 对齐后，假想窗口中的“同 CTA 内有的行全 mask、有的行部分合法”才可能直接出现。

### 19.1 三类 tile 的快速判断

Q tile 覆盖有效 query：

```text
[q0, q0+q_valid)
```

K tile 覆盖：

```text
[k0, k0+k_valid)
```

causal 时可以概念上分为：

1. 全合法：最大 key `<=` 最小 query；
2. 全未来：最小 key `>` 最大 query；
3. 跨对角线：其余情况，需要逐元素 mask。

M1 第一版可以始终做逐元素判断，先保证正确。之后才考虑用 tile 级判断跳过不必要的分支或加载。

---

# 第四部分：shape、offset 与三种 tail

## 20. 全局张量 shape

本阶段只有单 batch、单 head：

```text
Q   [N,D]
K   [N,D]
V   [N,D]
O   [N,D]
```

全部是 row-major、连续 FP32。

全局 offset：

```cpp
int q_offset = query * D + d;
int k_offset = key   * D + d;
int v_offset = key   * D + d;
int o_offset = query * D + d;
```

全局物理 stride 是运行时 `D`，不是 `MAX_D`。

## 21. shared memory 的 D 与 MAX_D

如果使用静态 shared 数组：

```text
Q_s     [BR, MAX_D]
K_s     [BC, MAX_D]
V_s     [BC, MAX_D]
O_acc_s [BR, MAX_D]
```

那么 shared row-major offset 是：

```cpp
int q_s_offset = r * MAX_D + d;
int k_s_offset = c * MAX_D + d;
```

即使运行时 `D=65`，下一行在 shared 中仍跨过 `MAX_D=128` 个 float。

对比：

```text
全局 Q/K/V/O 的行 stride：D
静态 shared 二维数组的物理行 stride：MAX_D
```

这是最常见的错位来源之一。

如果改用紧凑 dynamic shared memory，并明确按运行时 `D` 划分，则物理 stride 才可以是 `D`。两种布局都可行，但同一段代码不能把它们混用。

## 22. q0、grid 与 q_valid

固定 `BR=4`：

```text
grid.x = ceil(N / 4)
q0 = blockIdx.x × 4
q_valid = min(4, N-q0)
```

例如 `N=5`：

```text
CTA 0: q0=0, q_valid=4 → Q0..Q3
CTA 1: q0=4, q_valid=1 → Q4
```

不能让 CTA 1 的无效局部行 `r=1,2,3` 读取或写入全局内存。

因为 grid 已按 `ceil(N/4)` 精确构造，`N>0` 时每个已启动 CTA 都至少有一条有效 query。不要在部分线程中依据 `r>=q_valid` 提前 `return`；所有 128 个线程仍需参与 block barrier。

## 23. k0 与 k_valid

固定 `BC=16`：

```text
for k0 = 0, 16, 32, ... < N
k_valid = min(16, N-k0)
```

例如 `N=17`：

```text
tile 0: k0=0,  k_valid=16 → K0..K15
tile 1: k0=16, k_valid=1  → K16
```

最后一个 tile 只能加载、计算和累加 `c<k_valid` 的 key/value。

## 24. feature tail

`D` 可以是任意 `1..128`，不要求是 32、64 或 128。所有 feature 循环都必须以 `d<D` 为边界。

128 线程合作处理线性任务时，常见形式是：

```cpp
for (int linear = threadIdx.x; linear < task_count; linear += blockDim.x) {
    // 由 linear 解码逻辑坐标
}
```

`D=65` 时，不能读取 static shared row 中 `d=65..127` 的未初始化 padding；这些位置只是物理间隔，不是有效 feature。

## 25. query、key、feature tail 同时出现

用 `N=17, D=65` 看最后一个 CTA 的最后一个 K tile：

```text
q0=16, q_valid=1
k0=16, k_valid=1
D=65
```

此时：


### 25.1 即时关卡：写出合法索引集合

对 `N=17,D=65` 的最后 CTA、最后 K tile，先不写代码，写出：

```text
合法局部 query r：
合法局部 key c：
合法 feature d：
有效 score (r,c) 集合：
有效输出 (r,d) 数量：
```

完成后核对：只有 `r=0`、`c=0` 有效；`d=0..64` 有效，所以当前 tile 只有 1 个 score、最终有效输出任务为 65 个。其余物理槽位不能被读取，也不能让对应线程提前退出 barrier。
- 只有局部 query 行 `r=0` 有效；
- 只有局部 key 行 `c=0` 有效；
- feature `d=0..64` 有效；
- `r=1..3`、`c=1..15`、`d=65..MAX_D-1` 都不能当作有效数据；
- 128 个线程仍然共同经过相同 barrier 序列。

这就是“三种 tail 同时发生”。只测 `N=16,D=64` 无法暴露这些错误。

## 26. 推荐的线性坐标解码

下面只是独立索引片段，不是 kernel 解答。

### 26.1 Q tile 任务

```cpp
int r = linear / D;
int d = linear - r * D;
int query = q0 + r;
```

任务总数：

```text
q_valid × D
```

### 26.2 K/V tile 任务

```cpp
int c = linear / D;
int d = linear - c * D;
int key = k0 + c;
```

任务总数：

```text
k_valid × D
```

### 26.3 score 任务

满 tile 最多 64 个：

```cpp
int r = score_task / k_valid;
int c = score_task - r * k_valid;
```

任务总数：

```text
q_valid × k_valid <= 64
```

也可以用固定 `BC` 做 shared score stride：

```cpp
int score_s_offset = r * BC + c;
```

此时任务解码与物理 shared stride是两回事，不要混淆。

### 26.4 输出 row-feature 任务

```cpp
int r = linear / D;
int d = linear - r * D;
```

任务总数：

```text
q_valid × D
```

---

# 第五部分：128-thread 教学映射

## 27. 为什么选择 128 线程

`Br=4, Bc=16` 最多只有 64 个 score，而 Q/K/V 与输出 feature 任务可能更多。128 线程是一个简单折中：

- 足够合作加载 Q、K、V；
- 可让前 64 个线程一线程计算一个 score；
- 可让前 4 个线程充当 row leader；
- 可线性遍历最多 `4×128=512` 个 row-feature 输出任务。

它不是性能最优结论，只是 M1 的透明映射。

## 28. 阶段 A：合作加载 Q tile

任务：

```text
q_valid × D 个 FP32
```

128 个线程以线性 stride 合作完成。有效 Q 在整个 K/V tile 循环中不变，因此每个 CTA 只需加载一次。

同时可以用同样的 row-feature 任务划分初始化：

```text
O_acc[r,d] = 0
```

每行指定一个状态管理线程，后文称为 **row leader**。它负责初始化：

```text
m[r] = -∞
l[r] = 0
```

初始化完成后需要 block barrier，确保所有线程看到完整 Q 与状态。

## 29. 阶段 B：合作加载当前 K/V tile

每轮 K/V tile 的加载任务数：

```text
K: k_valid × D
V: k_valid × D
```

可以分别做两个线性循环，也可以把 K/V 看成连续的两个逻辑区间再解码。M1 优先清楚，不追求把所有加载写成最短代码。

加载完成后 barrier，之后 score 线程和 output 线程才可以读取 shared K/V。

## 30. 阶段 C：最多 64 个一线程一 score 任务

满 tile 有：

```text
q_valid × k_valid <= 4×16 = 64
```

推荐教学映射：

```text
thread 0..score_count-1
每个线程独立负责一个 (r,c)
并串行遍历 d=0..D-1 做 dot product
```

每个 score 的数学内容：

```text
dot = sum_d Q_s[r,d] × K_s[c,d]
score = dot / sqrt(D)
```

causal mask 使用全局 `query=q0+r` 与 `key=k0+c`。

这个阶段使用普通 FP32 scalar multiply-add。编译器通常可以把 `dot += a*b` 生成 FP32 `FFMA`，但应由 SASS 检查确认，而不是只看源码猜测。

所有 score 写入 shared 后需要 barrier，row leader 才能安全读取整行。

## 31. 阶段 D：最多 4 个 row leader

推荐：

```text
thread 0 负责局部 row 0
thread 1 负责局部 row 1
thread 2 负责局部 row 2
thread 3 负责局部 row 3
```

只对 `r<q_valid` 的行执行有效工作。每个 leader 串行扫描本行最多 16 个 score，完成：

1. 找 `m_block[r]`；
2. 识别局部全 mask；
3. 计算 `m_new[r]` 与 `alpha[r]`；
4. 把 score 原地改写为未归一化 weight，或写入独立 weight 区；
5. 求 `tile_l[r]`；
6. 更新 `m[r]` 与 `l[r]`。

row leader 完成后需要 barrier，output 线程才可以读取 `alpha[r]`、weight 与新 `l[r]`。

> [!IMPORTANT]
> row leader 是一个线程，不是一个 warp。`thread 0..3` 分别管理四行状态，并不意味着 warp 0..3 分别管理四行。

## 32. 阶段 E：线性 row-feature 输出任务

任务总数：

```text
q_valid × D <= 4×128 = 512
```

128 线程线性 stride 处理 `(r,d)`。每个任务读取这一行的最多 16 个 weight 与 V：

```text
add = sum_c weight[r,c] × V_s[c,d]
O_acc[r,d] = alpha[r] × O_acc[r,d] + add
```

当前 V tile 被全部 row-feature 任务使用完后，需要 barrier，才能让下一轮加载覆盖 shared K/V 和 weight。

## 33. 这不是 warp-per-query

虽然恰好有：

```text
128 threads = 4 warps
Br = 4 rows
```

M1 仍然明确不是“一 warp 一 query”。原因：

- score 任务按整个 `q_valid×k_valid` 线性分给前 64 个线程；
- 一个 score 由一个线程串行遍历 D；
- row max/sum 由单个 row leader 串行完成；
- 输出任务按 `(r,d)` 全局线性分配，线程可能先后处理不同行；
- 没有为每条 query 固定一个 warp，也没有使用 warp shuffle 做行内协作。

不要因为数目都是 4 就把逻辑映射误读成 M2。

---

# 第六部分：同步与所有权

## 34. 一轮 K/V tile 的完整时间线

```text
所有线程合作加载 K/V
        ↓ barrier A
score 线程读取 Q/K，写 scores
        ↓ barrier B
row leaders 读取 scores，写 weights/m/l/alpha
        ↓ barrier C
row-feature 线程读取 weights/V，更新 O_acc
        ↓ barrier D
下一轮才可覆盖 K/V/scores/weights
```

四个 barrier 各保护不同的数据依赖：

| barrier | 防止什么 |
| --- | --- |
| A | score 线程读到尚未加载完的 K/V |
| B | row leader 读到尚未写完的 score |
| C | output 线程读到旧 weight、旧 alpha 或旧 l |
| D | 下一 tile 覆盖当前 V/weight，而 output 线程仍在使用 |

第一版宁可同步清楚，不要在没有证据时合并 barrier。

## 35. Q、K、V 与状态的所有权

| 数据 | 写者 | 读者 | 生命周期 |
| --- | --- | --- | --- |
| `Q_s` | cooperative load threads | score threads | 整个 CTA |
| `K_s` | cooperative load threads | score threads | 当前 K tile |
| `V_s` | cooperative load threads | row-feature threads | 当前 V tile |
| `scores/weights` | score threads，随后 row leaders | row leaders，随后 row-feature threads | 当前 tile |
| `m/l/alpha` | row leaders | row-feature threads与后续 row leader | 每行跨 tile |
| `O_acc` | row-feature task owner | 同一任务的后续 tile与最终 store | 每行跨 tile |

如果每个 `(r,d)` 始终由线性 stride 的同一线程更新，可以考虑让该任务的 accumulator 保持在线程局部寄存器；但 M1 若采用 shared `O_acc` 会更容易看清与复核。不要同时改变 Br、所有权和存储位置，否则错误难以定位。

## 36. 为什么 barrier 附近不能 partial return

错误思路：

```text
如果某线程当前没有有效 query/score/output 任务，就 return
```

若其他线程之后执行 `__syncthreads()`，已返回线程不再到达 barrier，行为未定义，常表现为挂起、随机结果或 sanitizer 报告。

正确思路：

```text
所有 128 个线程走相同控制流并到达所有 block barrier
只有实际 load/compute/store 用 if 或循环边界屏蔽
```

特别是最后 CTA：即使 `q_valid=1`，另外 3 行无效，仍是 128 个线程参与同步，不是只留下 32 个或 1 个线程。

## 37. barrier 前后的调试口诀

每看到一个 `__syncthreads()`，闭卷回答：

1. barrier 前谁在写？
2. barrier 后谁在读？
3. 是否所有线程都必然到达？
4. 下一轮覆盖前是否确认所有读者已结束？

答不清楚就不要删 barrier。

### 37.1 即时关卡：恢复 barrier 顺序

下面四个同步点被打乱了。请先写出正确顺序，并为每个同步点填写“前面谁写、后面谁读”：

```text
① row-feature 更新 O_acc 完成，下一 tile 才能覆盖 V
② cooperative K/V load 完成，score 线程才能读取
③ score 写完，row leader 才能扫描一整行
④ row leader 写完 weight/alpha，row-feature 线程才能读取
```

正确顺序是 `② → ③ → ④ → ①`。如果只能背顺序而说不出 producer/consumer，说明同步所有权还没有真正理解。

---

# 第七部分：Shared Memory 账本

## 38. 紧凑布局的元素数

假设按运行时 `D` 紧凑存储：

```text
Q_s      BR×D      = 4D
K_s      BC×D      = 16D
V_s      BC×D      = 16D
O_acc_s  BR×D      = 4D
weights  BR×BC     = 64
m/l/alpha           = 3×4 = 12
```

总 FP32 元素数：

```text
4D + 16D + 16D + 4D + 64 + 12
= 40D + 76
```

总字节数：

```text
(40D + 76) × 4 bytes
```

## 39. D=64 与 D=128 的紧凑算术

### 39.1 D=64

```text
40×64 + 76 = 2636 floats
2636×4 = 10544 bytes
约 10.30 KiB
```

### 39.2 D=128

```text
40×128 + 76 = 5196 floats
5196×4 = 20784 bytes
约 20.30 KiB
```

这只是核心布局。若另存 scores 与 weights、加入 padding、归约 scratch 或调试区，需要继续加账。

## 40. 静态 MAX_D=128 的 caveat

若声明静态二维数组，所有 D 相关区域都按 `MAX_D=128` 分配：

```text
Q_s      4×128
K_s      16×128
V_s      16×128
O_acc_s  4×128
```

那么运行时即使 `D=64`，物理分配仍接近 D=128 的账本：

```text
5196 floats = 20784 bytes ≈ 20.30 KiB
```

也就是说：

```text
运行时有效工作量随 D 变化
静态 shared 占用不随 D 变化
```

这可能让 D=64 失去本可获得的 occupancy 空间。

A100 `sm_80` 每个 SM 有较大的 shared-memory 容量，但真实 resident CTA 数还同时受：

- 每 CTA shared memory；
- 每线程寄存器；
- 每 CTA 线程数；
- 架构 resident blocks/warps 上限；
- shared-memory carveout 与函数属性配置；
- 编译后实际资源使用量。

因此不能只用 `164 KiB / 20.30 KiB` 就宣布 occupancy。应查看编译资源报告、kernel attributes 与 profiler 的实际 resident/occupancy 指标。

一句话记忆：

> `D` 决定有效数据，`MAX_D` 可能决定物理 shared stride 和静态容量。

---

# 第八部分：Test-first 实现路线

## 41. 先准备测试，再填核心 kernel

本阶段核心 CUDA kernel 必须由学习者亲自完成。本文只给接口、检查点与提示，不提供完整 kernel body。

允许的接口示例：

```cpp
__global__ void query_tiled_attention_fp32(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int n,
    int d,
    bool causal);
```

launch 合同：

```text
block = 128 threads
grid  = ceil(n / 4)
支持 1 <= d <= 128
```

测试代码应先能调用尚未完成的 kernel，并明确失败，而不是先写完 kernel 再临时打印几个值。

## 42. Checkpoint 0：冻结 CPU reference 与比较器

先确认 CPU reference 支持：

- 单 batch、单 head；
- FP32 输入与输出；
- scale 为 `1/sqrt(D)`；
- 可选 causal；
- 任意测试矩阵中的 `N/D`；
- 输出 finite 检查；
- 最大绝对误差与最大相对误差；
- 首个失败坐标 `(row,d)` 及 got/expected。

这一阶段不需要改 CUDA kernel。

验收：故意把 GPU 输出清零时，非零测试必须稳定报 FAIL。

## 43. Checkpoint 1：只验证 q0/q_valid 与最终 store 范围

目标：先证明 grid 和 query tail 不越界。

可以让临时占位逻辑只向有效输出写一个可预测哨兵，但不要计算 Attention。

重点检查：

```text
N=1,3,4,5
```

通过条件：

- 只写 `[0,N×D)`；
- `N=5` 的第二 CTA 只写 Q4；
- guard 区未改变；
- sanitizer 无越界。

## 44. Checkpoint 2：合作加载 Q 与 K/V

先不更新 online softmax。用小 shape 把选定 shared 值或中间校验和复制到独立 debug buffer，验证：

- Q 的 global stride 使用 `D`；
- static shared stride使用 `MAX_D`；
- 最后 K tile 只加载 `k_valid` 行；
- `D=63/65/127` 不读 feature padding。

验证后移除或条件编译 debug 输出，不要让 benchmark 包含它。

## 45. Checkpoint 3：生成 4×16 score tile

只实现 score，先不做 softmax/PV。与 CPU 的局部 score 对比：

```text
score[r,c]
= dot(Q[q0+r,:], K[k0+c,:]) / sqrt(D)
```

必测：

- 非 causal；
- causal 全合法；
- causal 部分合法；
- causal 局部全 mask；
- `q_valid<4`；
- `k_valid<16`；
- `D` 非整除。

mask 后的值应按测试合同表示为负无穷，或另设 valid 标记；不要让 masked score 参与 max/exp。

## 46. Checkpoint 4：只更新 m/l/alpha

暂不计算 V 加权和。为每行输出最终 `m/l` 到 debug buffer，与 CPU online 状态对比。

构造“后 tile 出现新 max”和“后 tile 不出现新 max”的两行，复现第 7～13 节手算。

通过条件：

- 两行 alpha 可以不同；
- 局部全 mask 行状态保持；
- 不出现 `-∞-(-∞)` 导致的 NaN；
- 最终 `l>0` 且 finite。

## 47. Checkpoint 5：加入 O_acc

在已有正确 `m/l/weight` 上加入：

```text
O_acc = alpha × O_acc + weight × V
```

先测 `D=1/2`，逐元素打印每个 tile 后的状态，再扩展到 D64/128。

最容易漏掉的项是：

```text
alpha × old O_acc
```

如果只缩放 `l` 而不缩放 `O_acc`，最终概率分母看似合理，输出仍会错误。

## 48. Checkpoint 6：最终归一化与完整矩阵

所有 K/V tile 完成后：

```text
out = O_acc / l
```

只对有效 `(r,d)` 写回。完成 126 个基础 case、特殊模式与 guard 检查后，再运行完整 sanitizer 专用矩阵和性能测量。

但不要把 sanitizer 全部推迟到最后：Checkpoint 1～5 每次新增全局访问、Shared Memory 复用或 barrier 后，都先运行一个最小 `memcheck/racecheck/synccheck` smoke case，再继续下一个阶段。这样能在同步错误刚出现时定位它。

## 49. 为什么要按检查点推进

如果一次写完后结果错误，可能同时来自：

- query grid；
- global/shared stride；
- score 映射；
- mask；
- all-mask；
- alpha；
- l；
- O_acc；
- barrier；
- final store。

按检查点推进相当于一次只改变一个变量。这样失败时能定位到最近新增的数据流，而不是盲猜整个 kernel。

---

# 第九部分：Correctness 矩阵

## 50. 126 个基础 case

使用笛卡尔积：

```text
N = {1,3,4,5,15,16,17,31,33}
D = {1,2,63,64,65,127,128}
causal = {false,true}
```

总数：

```text
9 × 7 × 2 = 126
```

这组矩阵覆盖：

- `N<Br`；
- `N=Br` 与 `N=Br+1`；
- `N<Bc`；
- `N=Bc` 与 `N=Bc+1`；
- 多个 K tile；
- `N` 不是 4 或 16 的倍数；
- `D` 在 64/128 边界两侧；
- causal 与 non-causal。

每个 case 至少报告：

```text
N, D, causal, max_abs, max_rel, finite, PASS/FAIL
```

## 51. 特殊输入模式

基础矩阵之外，加入以下模式：

1. **全零 Q/K**：所有合法 score 相同，输出应为合法 V 的均值；
2. **全常量 V**：无论权重如何，输出都应接近该常量向量；
3. **identity-like V**：小 D 下便于从输出反推权重；
4. **后 tile 才出现最大 score**：强制 `alpha<1`；
5. **前 tile 已有最大 score**：后 tile `alpha=1`；
6. **不同行 max 轨迹不同**：复现完整手算意图；
7. **大正/大负但仍有限的 logits**：缩放后的 score 必须保持 finite，用于检验 stable online 更新；QK 点积本身已经溢出为 Inf/NaN 属于另一类输入/累加溢出问题；
8. **交替符号与非对称 Q/K/V**：避免错误索引被对称数据掩盖；
9. **causal 对角线附近**：检查 `key<=query`；
10. **局部全 mask 行**：检查 `alpha=1, weights=0, state unchanged`；
11. **NaN 哨兵 padding**：在 debug 路径中显式把无效 Shared Memory 槽位或专门分配的 global guard/padding 写成 NaN，确保不会参与有效输出；不要期待 `initcheck` 自动发现普通 Shared Memory 未初始化读取；
12. **输出 guard zone**：输出前后放哨兵，检查越界写。

## 52. 必测的同时 tail 组合

至少单列这些 case：

| case | 目的 |
| --- | --- |
| `N=5,D=65` | 第二 CTA 只有 1 行，feature 跨 64 |
| `N=17,D=65` | query/key/feature 三 tail 同时出现 |
| `N=31,D=127` | 最后 K tile 15 行，feature 接近上限 |
| `N=33,D=128` | 第 3 个 K tile 1 行，最后 CTA 1 行 |
| `N=3,D=1` | 极小 q/key/feature |

`N=17,D=65,causal=true` 尤其重要：最后 CTA、最后 K tile、feature tail、causal 对角线同时存在。

## 53. 容差原则

FP32 scalar FMA 的 GPU 累加顺序与 CPU reference 可能不同，不能要求逐 bit 相等。

建议比较条件采用：

```text
abs_diff <= atol + rtol × abs(expected)
```

阈值应由实际误差分布决定，并固定记录。不要为了让错误实现通过而不断放宽容差。

除数值误差外还必须单独检查：

- 所有有效输出 finite；
- guard zone 未改变；
- causal mask 语义；
- debug 状态中 `l>0`；
- 无非法内存访问。

---

# 第十部分：Sanitizer 计划

## 54. 工具顺序

每个涉及内存或同步的 Checkpoint 先跑最小 smoke；完整 correctness 通过后，再按以下顺序跑代表性和完整 sanitizer 矩阵：

1. `compute-sanitizer --tool memcheck`：越界、misaligned、非法地址；
2. `compute-sanitizer --tool initcheck`：未初始化 global memory 使用；
3. `compute-sanitizer --tool racecheck`：shared-memory data hazard；
4. `compute-sanitizer --tool synccheck`：barrier 与同步使用错误。

先跑最小边界 case，再跑代表性组合，最后跑完整矩阵或 sanitizer 专用子集。

## 55. sanitizer 专用 shape

至少包含：

```text
N=1,  D=1,   causal=false/true
N=5,  D=65,  causal=false/true
N=17, D=65,  causal=false/true
N=31, D=127, causal=false/true
N=33, D=128, causal=false/true
```

理由：

- 极小 shape 暴露空任务与初始化问题；
- `N=5` 暴露 query tail；
- `N=17` 暴露 key tail和同时 tail；
- `D=65/127` 暴露 feature 与物理 stride 错误；
- causal 暴露条件分支与局部全 mask。

## 56. sanitizer 结果如何解读

| 报告 | 优先检查 |
| --- | --- |
| invalid global read | `q_valid/k_valid/d<D`、全局 stride是否为 D |
| invalid global write | 最后 CTA 的 output store guard |
| uninitialized global read | `initcheck`：Host/device global buffer 是否先写后读 |
| 疑似 Shared Memory 未初始化值 | 显式初始化、NaN poison、debug state 与 correctness 对拍；`racecheck` 只检查 hazard，不能证明所有 shared slot 已初始化 |
| shared hazard | barrier A/B/C/D 是否缺失或位置错误 |
| divergent barrier | 是否有线程提前 return，barrier 是否放在非一致分支 |

sanitizer 零错误不等于数值正确；数值正确也不等于没有 race。尤其 `initcheck` 主要检查未初始化 global memory，不能替代 Shared Memory 初始化验证。两类证据必须分开。

---

# 第十一部分：Benchmark、ncu 与 SASS 假设

## 57. benchmark 前先写假设

M1 相对 `Br=1,Bc=16` 的核心变化只有：

```text
CTA 数从 N 变为 ceil(N/4)
每个 CTA 的 query 行从 1 变为最多 4
每份 K/V tile 被最多 4 条 query 复用
```

应先提出可证伪假设，再测量：

1. K/V requested global load 大致下降；
2. DRAM bytes 下降幅度可能小于 4 倍；
3. score 与 PV 的数学 FLOP 基本不变；
4. CTA 数减少到约四分之一；
5. 每 CTA shared memory、barrier 工作和 O_acc 状态增加；
6. occupancy 可能改善、持平或下降，不能仅凭 CTA 数判断；
7. 对小 N，额外同步与串行 row leader 可能抵消 K/V 复用收益。

## 58. requested K/V loads 与 DRAM bytes 的区别

非 causal、忽略 tail：

```text
Br=1:
每个 query CTA 请求读 K 和 V 各 N×D
总请求 ≈ 2N²D floats

Br=4:
每个 query tile CTA 请求读 K 和 V 各 N×D
总请求 ≈ 2(N/4)ND
       = N²D/2 floats
```

相对比值：

```text
Br4 / Br1 ≈ 1/4
```

但 profiler 的 DRAM bytes 还受 L2 命中影响：

```text
global load request
→ 可能命中 L1/L2
→ 未命中部分才形成更下层流量
→ 最后才反映为 DRAM bytes
```

因此报告时至少区分：

- 源代码/理论 requested K/V 元素数；
- profiler 的 global/L1/L2 请求或 sector；
- profiler 的实际 DRAM read bytes；
- cache hit rate。

不要把其中一个指标代替全部。

## 59. FLOP 为什么不变

对每个有效 `(query,key)`：

- QK 点积仍做 D 个乘加；
- 对 V 的加权仍更新 D 个 feature；
- exact dense Attention 的合法 pair 数没有改变。

因此主导 FLOP 仍约为：

```text
QK: 2N²D
PV: 2N²D
合计: 4N²D
```

causal 的数学合法 pair 数约为 dense 的一半；只有实现同时跳过 masked QK dot、全未来 tile 和无效 PV 项时，实际执行的主导 FLOP 才接近减半。第一版若仍对零 weight 执行 PV FMA，实际指令数不会严格按合法 pair 数减半。比较 Br1 与 Br4 时必须保证两者采用相同的 causal 工作策略。

M1 的目标不是减少数学 FLOP，而是改变 K/V 数据复用与 CTA 组织。

## 60. CTA count 与 grid

```text
Br=1 grid.x = N
Br=4 grid.x = ceil(N/4)
```

例如：

| N | Br=1 CTA | Br=4 CTA |
| ---: | ---: | ---: |
| 16 | 16 | 4 |
| 17 | 17 | 5 |
| 31 | 31 | 8 |
| 33 | 33 | 9 |

CTA 减少可能降低调度开销，也可能在小 N 时降低可用并行 CTA 数。单个 CTA 工作变多，不自动等于更快。

## 61. barrier、SMEM、O_acc 与 occupancy 权衡

Br=4 的收益来自 K/V 复用，但代价包括：

- Q 与 O_acc 从一行扩大到四行；
- score tile 从 16 增加到 64；
- 每轮有四组 row state；
- row-feature 更新总量变成四行；
- CTA 生命周期更长；
- 同步点仍在每个 K/V tile 重复；
- 若 O_acc 放寄存器，寄存器压力可能上升；
- 若 O_acc 放 shared，shared 流量与容量上升。

可能出现三种结果：

1. K/V 流量下降主导，性能提高；
2. 同步与串行 row leader 抵消收益，性能持平；
3. 资源压力降低 occupancy，性能下降。

负结果也是有效证据。应记录 shape、GPU、编译选项、计时方法和 profiler 现象，而不是隐藏。

## 62. benchmark 设计

对 Br1 与 Br4 使用相同：

- A100 `sm_80`；
- FP32 输入/累加/输出；
- 单 batch、单 head；
- `Bc=16`；
- causal 设置；
- 输入数据；
- warmup 次数；
- CUDA event 计时区间；
- 重复次数与统计量；
- correctness 门槛。

建议覆盖：

```text
N = 128, 256, 512, 1024, 2048
D = 64, 128
causal = false, true
```

输出至少包含：

```text
GPU, N, D, dtype, causal, implementation,
median/min latency, requested-load theory, correctness error
```

不要把 profiler 下的耗时当作普通 wall-clock benchmark。

## 63. Nsight Compute 观察方向

具体 metric 名称可能随 Nsight Compute 版本变化，可从以下问题出发选择对应指标：

1. DRAM read bytes 是否下降？
2. L2 hit rate 是否改变？
3. global load sectors/request 是否合理？
4. shared load/store 是否明显增加？
5. barrier stall 是否上升？
6. achieved occupancy 与 active warps 是否改变？
7. registers per thread、shared memory per CTA 是多少？
8. FP32 pipeline 活跃度是否提高或降低？
9. kernel 总 CTA 数是否符合 `ceil(N/4)`？
10. 小 N 是否因 CTA 数不足而无法填满 GPU？

结论要与假设对应。例如：

```text
requested loads 理论下降 4×
但 DRAM bytes 只下降 1.6×
同时 barrier stall 上升
→ 说明 Br1 原先有 cache 复用，Br4 又付出了同步代价
```

这比只写“Br4 更快/更慢”更有价值。

## 64. SASS 预期

本阶段是 FP32 SIMT scalar FMA，因此 SASS 假设为：

- dot product 与 weighted V 累加中出现 FP32 `FFMA`；
- global/shared 访问出现相应 load/store 指令；
- barrier 附近出现同步指令；
- 不应出现代表 Tensor Core MMA 的 `HMMA`；
- 不使用 `cp.async`，因此不应出现 Ampere async global-to-shared 路径常见的 `LDGSTS`；
- `expf` 可能展开为特殊函数或近似相关指令序列，不要假设它是一条普通 FMA。

如果看到 `HMMA` 或 `LDGSTS`，先确认检查的是不是正确 binary、正确 kernel 和正确编译路径。

若没有看到预期 `FFMA`，检查：

- 编译优化是否开启；
- 反汇编是否对应目标 kernel；
- 代码是否被优化掉；
- 是否因为编译选项禁止 contraction；
- 累加表达式是否真的处于热点循环。

---

# 第十二部分：常见 bug 症状表

## 65. 症状、根因与第一检查点

| 症状 | 常见根因 | 第一检查点 |
| --- | --- | --- |
| 每 4 行输出完全相同 | 四行共用了同一 Q 或同一 row state | `q0+r` 与 `m/l/O_acc[r]` |
| 只有第 0 行正确 | score/weight/shared offset漏了 `r` | `r*BC+c` |
| D64 正确，D65 开始错 | feature tail 或 D/MAX_D stride 混用 | global stride D，shared stride MAX_D |
| N16 正确，N17 错 | `k_valid` 或最后 K tile load 越界 | `min(BC,N-k0)` |
| N4 正确，N5 错 | `q_valid` 或最后 CTA store 越界 | `min(BR,N-q0)` |
| non-causal 正确，causal 错 | 用 `c>r` 代替 `key>query` | 全局 q/k 下标 |
| causal 出现 NaN | 局部全 mask 做了 `-∞-(-∞)` | `alpha=1, weights=0` 分支 |
| 后 tile max 更大时输出错 | 只缩放 l，未缩放 O_acc | `alpha×old O_acc` |
| 某些行对、某些行偏差大 | CTA 共用一个 alpha 或 m | 状态是否 `[BR]` |
| 结果随机变化 | 缺 barrier 或 shared hazard | A/B/C/D 时间线 |
| kernel 偶发挂起 | barrier 前 partial return | 所有线程是否到达同步点 |
| sanitizer 报 invalid read | 无效 r/c/d 仍参与 load | q_valid/k_valid/D guard |
| 输出接近每 tile 局部均值 | 每个 tile 提前归一化 | 只在所有 tile 后除以 l |
| Br4 比 Br1 慢很多 | barrier/SMEM/occupancy/并行 CTA 代价 | profiler，不先猜结论 |
| SASS 出现 HMMA | 编译/检查了别的实现路径 | kernel symbol 与 binary |
| SASS 出现 LDGSTS | 实际路径包含 async copy 或检查错对象 | 编译路径与反汇编目标 |

---

# 第十三部分：三级提示

## 66. 提示 1：只看不变量

卡住时先只检查以下不变量，不看完整实现：

1. 一个 CTA 负责 `[q0, q0+q_valid)` 四行以内的 query；
2. 每个 K/V tile 只被 cooperative load 一次；
3. 每个有效 `(r,c)` 恰好产生一个 score；
4. 每个 `r` 有独立 `m/l/alpha/O_acc`；
5. masked weight 必须为 0；
6. 局部全 mask 时该行状态不变；
7. 所有 tile 结束前不做最终除法；
8. 所有线程到达所有 barrier。

若某条不变量无法从代码中明确指出，先修它。

> **停止点：**只使用本级提示检查至少 15 分钟。能定位问题就不要继续看提示 2。

## 67. 提示 2：定位数据流

仍然错误时，逐 tile 导出小 shape 的：

```text
scores[Br,Bc]
m_block[Br]
m_new[Br]
alpha[Br]
weights[Br,Bc]
l[Br]
O_acc[Br,D]
```

推荐用第 7 节 `Br=2,Bc=2,D=2` 数字作为首个 debug case。

定位顺序：

```text
score
→ mask
→ m_block/m_new
→ alpha/weights
→ l
→ O_acc
→ final normalize
```

第一个与手算不一致的状态，通常就是根因所在阶段。不要从最终输出倒猜所有环节。

> **停止点：**按上述顺序完成一次逐状态对拍。仍无法定位时，才查看提示 3。

## 68. 提示 3：局部公式与伪代码

如果仍卡住，只检查以下三个局部职责，不提供最外层 tile 循环、初始化、barrier 或输出写回位置。你需要自己决定它们如何连接。

```text
Score owner：
    给定有效 (r,c)，用全局 query/key 做 causal 判断；
    未 mask 时只计算这一项 dot/scale。

Row state owner：
    只读取本行有效 score；
    all-mask 时保留状态；否则更新本行 m/l/alpha/weight。

Row-feature owner：
    给定有效 (r,d)，读取本行 alpha/weight 和当前 V tile；
    更新这一项 O_acc[r,d]。
```

若三个局部职责都正确而 CUDA 仍错误，优先检查索引、所有权与 barrier。不要从本提示推导“完整答案”；最外层控制流仍由你自己设计。

---

# 第十四部分：练习与答案

## 69. 练习

1. `N=17` 时，Br1 与 Br4 分别启动多少个 CTA？
2. `N=33` 时最后一个 Br4 CTA 的 `q0/q_valid` 是多少？
3. `N=31` 时最后一个 Bc16 tile 的 `k0/k_valid` 是多少？
4. `D=65,MAX_D=128` 时，全局 Q 下一行 stride 和静态 shared Q 下一行 stride 分别是多少？
5. 满 `Br=4,Bc=16` tile 有多少 score？128 线程映射中多少线程可以没有 score 任务？
6. 某行 `m_old=3,l_old=2`，新 tile row max 为 5，`alpha` 和旧分母贡献 `alpha*l_old` 分别是多少？
7. 另一行 `m_old=3`，新 tile row max 为 2，`alpha` 是多少？
8. causal 下实际窗口 `q0=12,r=3,k0=16,c=0` 是否合法？
9. causal 下实际窗口 `q0=16,r=3,k0=16,c=1` 是否合法？
10. 局部全 mask 时为什么不能令 `alpha=0`？
11. 为什么 Br4 的 FLOP 不会因 K/V 复用而自然变成 Br1 的四分之一？
12. 为什么 DRAM bytes 不一定下降 4 倍？
13. 为什么 M1 的 4 个 warp 不能称为“一 warp 一 query”？
14. 静态 `MAX_D=128` 布局在运行时 D64 大约占多少核心 shared memory？
15. `N=17,D=65` 的最后 CTA、最后 K tile 中，有效 `(r,c)` score 数是多少？

## 70. 答案

1. Br1 为 17 个；Br4 为 `ceil(17/4)=5` 个。
2. `q0=32,q_valid=1`。
3. `k0=16,k_valid=15`。
4. 全局 stride 是 65 个 float；静态 shared stride 是 128 个 float。
5. 64 个 score；满 tile 时恰好 64 个线程没有 score 任务。tail tile 中空闲线程可能更多，最多 127 个。
6. `alpha=exp(3-5)=e^-2`，旧分母贡献为 `2e^-2`。
7. `m_new` 仍为 3，所以 `alpha=exp(0)=1`。
8. query 为 15，key 为 16，`16>15`，不合法。
9. query 为 19，key 为 17，合法。
10. `alpha=0` 会删除旧 `l/O_acc`；全 mask tile 没有新信息，旧状态必须乘 1 保留。
11. 每个合法 query-key pair 的 QK 点积和 V 加权仍要计算；复用减少的是 K/V 搬运请求，不是数学 pair 数。
12. 重复 load 可能命中 cache，DRAM 只看到 cache miss 后的流量；事务合并与调度也会影响结果。
13. score、row leader、output 都没有按固定 warp 绑定一行，也没有 warp 行内归约。
14. 按本文核心布局约为 `20784 bytes≈20.30 KiB`，不是紧凑 D64 的 10.30 KiB。
15. `q_valid=1,k_valid=1`，所以只有 1 个有效 score。

---

# 第十五部分：自检与完成标准

## 71. 闭卷自检

尝试不用文档回答：

1. Br4 为什么能复用 K/V？
2. 哪些状态共享，哪些状态逐行独立？
3. 为什么同一 tile 的四个 alpha 可以不同？
4. 局部全 mask 的安全更新是什么？
5. global stride D 与 shared stride MAX_D 有何区别？
6. q_valid、k_valid、D tail 如何同时出现？
7. 128 线程分别在哪些阶段扮演什么角色？
8. 为什么 M1 不是 warp-per-query？
9. 四个 barrier 各保护什么？
10. Br4 为什么可能更慢？
11. requested loads 与 DRAM bytes 有何区别？
12. SASS 中为什么预期 FFMA，而不预期 HMMA/LDGSTS？

若有三题以上说不清，先回到对应章节，再继续写 kernel。

## 72. 完成清单

### 数学

- [ ] 能写出 Standard Attention 公式。
- [ ] 能解释每行独立的 `m/l/alpha/O_acc`。
- [ ] 能复算 Br2/Bc2/D2 两 tile 示例。
- [ ] 能解释 Q0 与 Q1 的 alpha 为什么不同。
- [ ] 能解释局部全 mask 的状态保持规则。

### 索引与 tail

- [ ] `q0=blockIdx.x×4`。
- [ ] `q_valid=min(4,N-q0)`。
- [ ] `k_valid=min(16,N-k0)`。
- [ ] causal 使用全局 `key>query`。
- [ ] global row stride 使用 D。
- [ ] static shared row stride 使用 MAX_D。
- [ ] query/key/feature tail 可同时正确处理。

### 线程与同步

- [ ] 128 线程合作加载 Q/K/V。
- [ ] 最多 64 个一线程一 score 任务。
- [ ] 最多 4 个 row leader。
- [ ] 输出使用线性 row-feature 任务。
- [ ] 没有把 M1 写成 warp-per-query。
- [ ] 没有 barrier 前 partial return。
- [ ] 覆盖 K/V/weight 前有明确同步。

### 验证

- [ ] 126 个基础 case 全部纳入测试。
- [ ] 特殊输入模式与同时 tail 已覆盖。
- [ ] 输出 finite、误差与 guard zone 分开检查。
- [ ] memcheck/initcheck/racecheck/synccheck 有计划。
- [ ] benchmark 与 profiler 测量分开。
- [ ] 性能报告包含 GPU、shape、dtype、路径与方法。
- [ ] SASS 检查 FFMA，并确认无 HMMA/LDGSTS 路径。

---

# 第十六部分：公式速查表

## 73. 核心公式

### Standard Attention

```text
S[i,j] = dot(Q[i,:],K[j,:]) / sqrt(D)
P[i,j] = softmax_j(S[i,j])
O[i,:] = sum_j P[i,j]V[j,:]
```

### Tile shape

```text
Q_tile    [Br,D]
K_tile    [Bc,D]
V_tile    [Bc,D]
S/weights [Br,Bc]
m/l/alpha [Br]
O_acc     [Br,D]
```

### Online row update

```text
m_block = rowmax(scores)
m_new   = max(m_old,m_block)
alpha   = exp(m_old-m_new)
weight  = exp(score-m_new)
l_new   = alpha*l_old + sum(weight)
O_new   = alpha*O_old + sum(weight*V)
```

### 局部全 mask

```text
alpha = 1
weights = 0
m_new = m_old
l_new = l_old
O_new = O_old
```

### causal

```text
valid iff key <= query
query = q0+r
key   = k0+c
```

### tile 起点与有效数

```text
q0      = blockIdx.x × 4
q_valid = min(4,N-q0)
k_valid = min(16,N-k0)
grid.x  = ceil(N/4)
```

### offset

```text
global row-major: row×D + d
static shared:    row×MAX_D + d
score shared:     r×BC + c
```

### shared memory

```text
compact floats = 40D + 76
D64  = 10544 bytes ≈ 10.30 KiB
D128 = 20784 bytes ≈ 20.30 KiB
```

### requested K/V load 理论

```text
Br1 ≈ 2N²D floats
Br4 ≈ 2ceil(N/4)ND floats
满 tile 时 Br4/Br1 ≈ 1/4
```

---

# 第十七部分：与 M2 warp-per-query 的关系

## 74. M1 为 M2 保留了什么

M2 不会改变以下数学合同：

- 每条 query 有独立 `m/l/alpha/O_acc`；
- K/V tile 可以被多条 query 共享；
- causal 使用全局 query/key 坐标；
- all-mask 局部行保持状态；
- 最终仍是 `O_acc/l`；
- correctness 矩阵与 sanitizer 边界仍适用。

M2 主要改变线程如何合作计算一条 query：

```text
M1:
一个 score 由一个线程串行遍历 D
row leader 串行扫最多 16 个 score
row-feature 任务跨整个 CTA 线性分配

M2:
一个 warp 固定负责一条 query
warp lanes 合作 feature/dot/reduction
row max 与 row sum 可用 warp shuffle
O_acc 所有权重新设计
```

## 75. 为什么不能跳过 M1

若直接进入 warp-per-query，错误可能来自：

- Br4 多行状态；
- warp lane 映射；
- shuffle mask；
- 跨 lane reduction；
- register accumulator；
- tail warp 的 active mask；
- 原有 causal/all-mask 逻辑。

M1 把第一类问题单独解决。等 M1 的数学与边界稳定后，M2 才能把“改变并行方式”作为唯一主要变量。

一句话记忆：

> M1 学会让 4 行共享 K/V 但不共享 softmax；M2 再让一个 warp 高效协作一行。

## 76. 最终口述

> Br4 Query-Tiled FP32 SIMT Attention 用一个 128-thread CTA 处理最多 4 条连续 query。CTA 只合作加载一份 Bc16 的 K/V tile，四条 query 复用它，但每行独立维护 online softmax 的 m、l、alpha 和 O_acc。满 tile 最多 64 个 score，可先用一线程一 score；前 4 个线程作为 row leader；输出按 row-feature 线性分配。query、key、feature tail 分别由 q_valid、k_valid 和 d<D 控制，所有线程仍共同经过 barrier。它理论上把满 query tile 的 K/V requested loads 降到 Br1 的约四分之一，但 FLOP 基本不变，实际收益还要与 barrier、shared memory、O_acc 存储、occupancy 和 cache 行为一起测量。M1 是正确性与数据复用基线，不是 warp-per-query；下一阶段 M2 才改变为一个 warp 协作一条 query。
