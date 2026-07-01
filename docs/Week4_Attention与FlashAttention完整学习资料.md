# Week 4：Attention、FlashAttention 与 MLA——A100 自包含学习手册

> 对应：[DeepSeek CUDA 2 月冲刺计划](DeepSeek_CUDA_2月冲刺计划.md) Week 4。
> 主要实验平台：NVIDIA A100（Ampere，`sm_80`）。
> 起点：你已经写过 CUDA kernel、GEMM、归约、stable softmax、WMMA 和 `cp.async`，但本书**不假设你知道 Attention、Transformer 或 KV cache**。

---

## 0. 先说清楚：这一周到底学什么

Attention 可以先压缩成一句话：

> 对序列里的每个 token，计算它应该从其他 token 那里读取多少信息，再把这些信息加权汇总。

而 FlashAttention 不是另一种近似 Attention：

> 它计算相同的精确结果，但改变计算顺序，避免把巨大的中间矩阵反复写入和读出 HBM。

这一周的主线是：

```text
token / embedding / shape
        ↓
Q、K、V 与标准 Attention
        ↓
stable softmax → online softmax
        ↓
分块 Attention 的 m / l / O_acc 状态
        ↓
FlashAttention 为什么 IO-aware
        ↓
A100：FP16/BF16、Tensor Core、cp.async
        ↓
prefill / decode / KV cache / MQA / GQA / MLA / FlashMLA
```

### 0.1 你的旧知识会在哪里复用

| 你已经做过的内容 | 这周对应的新用途 |
|---|---|
| GEMM、矩阵 shape | `QK^T` 和 `PV` 都是矩阵乘 |
| max/sum reduction | 每一行 stable softmax |
| warp shuffle、block reduce | 行内最大值与指数和 |
| 分层归约 | 合并不同 tile 的 online softmax 状态 |
| shared-memory tiling | 分块装入 Q/K/V，减少 HBM 访问 |
| register accumulator | 保存当前 query 行的输出累加器 |
| WMMA / HMMA | 加速 `QK^T` 和 `PV` 的矩阵乘部分 |
| `cp.async` / pipeline | 计算当前 K/V tile 时预取下一 tile |
| Roofline / ncu | 判断瓶颈究竟在 HBM、shared、计算还是 occupancy |

所以你不是从 CUDA 零开始。真正从零开始的是“这些 CUDA 原语在 Attention 里分别代表什么”。

### 0.2 五种代码标记

- **【演示】**：完整提供，直接运行，用来建立直觉。
- **【必须手写】**：提供接口、host 测试框架和明确空位，核心由你完成。
- **【提示 1/2/3】**：卡住后逐级看；不要一次全展开。
- **【参考实现】**：独立尝试并通过验收后再看。
- **【挑战】**：只给需求和验收标准。

推荐纪律：

```text
先学概念和手算
→ 独立写 30～60 分钟
→ 记录实际报错或错误结果
→ 一次只看一级提示
→ correctness PASS 后再对照参考实现
→ 第二天关掉答案重写核心循环
```

### 0.3 七天路线与交付

| Day | 主题 | 必须手写 | correctness | 闭卷口述 |
|---|---|---|---|---|
| 1 | 标准 Attention | `QK^T`、row softmax、`PV` 三个 kernel | 对齐 CPU，支持非整除 shape | Q/K/V、缩放、softmax 维度 |
| 2 | online softmax | CPU streaming + CUDA 单行版本 | 大值无 NaN，行和约 1 | 新 max 出现为何重缩放 |
| 3 | FlashAttention | `m/l/O_acc` 纸上推演与伪代码 | 两 tile 手算对齐标准结果 | 为什么省 IO 而非主要 FLOP |
| 4 | tiled Attention | 教学版核心循环 | `N=3/8/37/128` 对齐 CPU | 为什么旧输出也要缩放 |
| 5 | A100 优化 | 选择一项：FP16 或 K/V 双缓冲 | 精度与性能分别验证 | Tensor Core、`cp.async` 各解决什么 |
| 6 | KV cache 与 MLA | cache 字节账本 | MHA/GQA/MLA shape 计算正确 | prefill/decode、MLA 省什么 |
| 7 | profiling | benchmark 表 + ncu 证据 | 无 profiler 与 profiler 分开测 | 用指标讲清瓶颈 |

---

# 预备章：从一句话到张量

## 1. token 不是“一个汉字”

模型不能直接接收字符串。文本先经过 tokenizer，被切成若干 **token**。token 可能是一个汉字、一个英文词的一部分、标点，甚至空格模式；它不等于自然语言里的“词”。

每个 token 会映射成一个整数 `token_id`，再通过 embedding table 查到一个向量：

```text
文本        → token ids     → embedding vectors
"我 爱 CUDA" → [17, 42, 91] → 3 个 D 维向量
```

**Embedding（嵌入向量）**：模型用一组浮点数表示一个 token 当前的特征。真实模型里维度 `D` 可能是几千；为了手算，本书固定一个微型例子：

```text
N = 3 个 token
D = 2 维

X = [[1, 0],
     [0, 1],
     [1, 1]]
```

这里：

- `N`：sequence length，序列长度；
- `D`：hidden dimension，隐藏维度；
- `X` 的 shape 是 `[N,D]=[3,2]`；
- `X[1] = [0,1]` 是第 1 个 token 的向量。

这些数字只是用来理解计算，并不声称 `[1,0]` 具有某种真实语言含义。

## 2. 从单序列到多头

常见 shape 逐步增加：

```text
[N, D]          一个序列
[B, N, D]       B 个序列组成 batch
[B, H, N, Dh]   把隐藏维拆成 H 个 attention head
```

符号约定：

| 符号 | 含义 | 示例 |
|---|---|---:|
| `B` | batch size，同时处理多少条序列 | 2 |
| `N` | sequence length，每条序列多少 token | 32 |
| `D` | model hidden dimension | 64 |
| `H` | query head 数量 | 4 |
| `Dh` | 每个 head 的维度 | 16 |

通常有：

```text
D = H × Dh
64 = 4 × 16
```

“多头”可以先理解为：同一个 token 同时在多个不同的投影子空间里寻找关系。每个 head 独立做 Attention，最后把结果拼接起来，再做输出投影。

## 3. shape 只是解释方式，数据最终仍是一维内存

你在 GEMM 已经学过行主序压平。二维 `[N,D]`：

```cpp
int idx2(int n, int d, int D) {
    return n * D + d;
}
```

四维 `[B,H,N,Dh]`，最右边的 `d` 连续：

```cpp
int idx4(int b, int h, int n, int d,
         int H, int N, int Dh) {
    return ((b * H + h) * N + n) * Dh + d;
}
```

例如 `B=2,H=4,N=32,Dh=16`，坐标 `[b=1,h=2,n=3,d=4]`：

```text
offset = ((1×4 + 2)×32 + 3)×16 + 4
       = (6×32 + 3)×16 + 4
       = 3124
```

`reshape` 有时只修改“如何解释同一段连续内存”的元数据，不一定搬数据；但 transpose 往往会改变 stride，若后续代码要求 contiguous，就可能触发真实拷贝。不要把“shape 变了”和“数据一定搬了”画等号。

## 4. 预备章自测

1. `Q` 的 shape 是 `[B,H,N,Dh]=[2,8,128,64]`，每个 batch 一共有多少 query 向量？
2. `K` 从 `[N,Dh]` 转置后是什么 shape？
3. `[B,H,N,Dh]=[2,4,32,16]` 一共有多少元素？
4. 对 query `i` 的所有 key 分数做 softmax，应该沿 query 维还是 key 维？

答案：

1. 每个 batch 有 `H×N=8×128=1024` 个 query 向量；整个 batch 有 2048 个。
2. `[Dh,N]`。
3. `2×4×32×16=4096`。
4. 沿 key 维。固定 query `i`，把它对 `j=0..N-1` 的分数归一化成一组权重。

---

# Day 1：标准 Attention——先知道自己到底在算什么

## 5. 为什么需要 Attention

设一句话里有多个 token。更新第 `i` 个 token 的表示时，我们希望它能读取与自己有关的上下文，但不同 token 的重要程度不同。

Attention 把这个过程拆成三种角色：

- **Query（Q，查询）**：我现在想寻找什么？
- **Key（K，键）**：我能用什么特征被别人匹配？
- **Value（V，值）**：如果别人关注我，我实际贡献什么信息？

数据库类比只能帮你入门，不能完全等同：在神经网络里，Q/K/V 都是训练出来的连续向量。

输入 `X` 经过三个不同的线性投影：

```text
Q = X Wq
K = X Wk
V = X Wv
```

同一个 `X` 变成三份不是浪费，而是让“用于匹配的特征”和“真正传递的内容”承担不同职责。

## 6. Scaled Dot-Product Attention 五步

对单 batch、单 head：

```text
Q: [N,Dh]
K: [N,Dh]
V: [N,Dv]    通常 Dv=Dh，但数学上可以不同
```

### 第一步：每个 query 与每个 key 做点积

```text
S_raw = Q K^T
[N,Dh] × [Dh,N] = [N,N]
```

元素含义：

```text
S_raw[i,j] = Σ_d Q[i,d] × K[j,d]
```

第 `i` 行：query `i` 对所有 key 的匹配分数。
第 `j` 列：所有 query 对 key `j` 的匹配分数。

这正是 GEMM。区别只在于矩阵在模型里有了语义。

### 第二步：除以 `sqrt(Dh)`

```text
S = QK^T / sqrt(Dh)
```

若 Q/K 各维近似零均值、方差约 1，`Dh` 个独立乘积求和的方差会随 `Dh` 增长。logit 绝对值过大时，softmax 容易接近 one-hot，梯度很小。除以 `sqrt(Dh)` 把量级拉回更稳定的范围。

注意：这是统计尺度解释，不是说每一组实际 Q/K 都严格独立同分布。

### 第三步：可选 mask

自回归语言模型训练或 prefill 时，第 `i` 个 token 不允许看到未来 `j>i`：

```text
S[i,j] = -∞,  if j > i
```

softmax 后 `exp(-∞)=0`，未来位置权重为 0。这叫 **causal mask（因果掩码）**。

### 第四步：每一行做 softmax

```text
P[i,j] = exp(S[i,j]-m_i) / Σ_t exp(S[i,t]-m_i)
m_i = max_j S[i,j]
```

`P` shape 仍是 `[N,N]`。每行和为 1。softmax 沿 key 轴 `j` 做，不是沿 query 轴 `i`。

### 第五步：用概率对 V 加权

```text
O = P V
[N,N] × [N,Dv] = [N,Dv]
```

```text
O[i,d] = Σ_j P[i,j] × V[j,d]
```

对 query `i` 来说，它从所有 value `j` 读取信息，读取量由 `P[i,j]` 决定。

完整公式：

```text
Attention(Q,K,V) = softmax(QK^T / sqrt(Dh)) V
```

## 7. 用 `N=3,Dh=2` 真正手算一遍

为避免投影矩阵干扰第一遍理解，先设：

```text
Q = K = V = X

X = [[1,0],
     [0,1],
     [1,1]]
```

点积：

```text
QK^T = [[1,0,1],
        [0,1,1],
        [1,1,2]]
```

因为 `Dh=2`，缩放因子 `1/sqrt(2)≈0.7071`：

```text
S ≈ [[0.7071,0,     0.7071],
     [0,     0.7071,0.7071],
     [0.7071,0.7071,1.4142]]
```

以第 0 行为例：

```text
m = 0.7071
exp(S-m) = [1, exp(-0.7071), 1]
         ≈ [1, 0.4931, 1]
sum      ≈ 2.4931
P[0]     ≈ [0.4011, 0.1978, 0.4011]
```

于是：

```text
O[0] = 0.4011×[1,0] + 0.1978×[0,1] + 0.4011×[1,1]
     ≈ [0.8022, 0.5989]
```

请你亲手算第 1、2 行。重点不是小数，而是确认：

```text
QK^T 决定“看多少”
softmax 把分数变成每行和为 1 的权重
PV 决定“读到什么”
```

## 8. 多头 Attention 的 shape

标准多头自注意力通常先投影，再 reshape：

```text
X: [B,N,D]
Q: [B,H,N,Dh]
K: [B,H,N,Dh]
V: [B,H,N,Dh]
```

每个 `(b,h)` 独立做：

```text
[N,Dh] × [Dh,N] → [N,N]
[N,N]  × [N,Dh] → [N,Dh]
```

所有 head 输出拼成 `[B,N,H×Dh]=[B,N,D]`，再乘输出投影 `Wo`。

不要把 head 和 CUDA warp 绑定。head 是模型数学维度；一个 head 可以由很多 block/warp 计算，一个 warp 也可能处理某个 head 的一小块。

## 9.【演示】完整 CPU Attention reference

下面程序实现单 head、FP32、可选 causal。它直接融合 score/softmax/PV，目的是作为权威正确性参考，不是模拟 GPU 的三 kernel 数据流。

```cpp attention_cpu.cpp
// FILE: attention_cpu.cpp
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

void attention_cpu(const float* q, const float* k, const float* v,
                   float* out, int n, int d, bool causal) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> scores(n);

    for (int i = 0; i < n; ++i) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < n; ++j) {
            if (causal && j > i) {
                scores[j] = -std::numeric_limits<float>::infinity();
                continue;
            }
            float dot = 0.0f;
            for (int x = 0; x < d; ++x) {
                dot += q[i * d + x] * k[j * d + x];
            }
            scores[j] = dot * scale;
            row_max = std::max(row_max, scores[j]);
        }

        float denom = 0.0f;
        for (int j = 0; j < n; ++j) {
            if (causal && j > i) {
                scores[j] = 0.0f;
            } else {
                scores[j] = std::exp(scores[j] - row_max);
                denom += scores[j];
            }
        }

        for (int x = 0; x < d; ++x) {
            float acc = 0.0f;
            for (int j = 0; j < n; ++j) {
                acc += (scores[j] / denom) * v[j * d + x];
            }
            out[i * d + x] = acc;
        }
    }
}

bool finite_and_close(const std::vector<float>& got,
                      const std::vector<float>& expected,
                      float atol = 1e-5f) {
    if (got.size() != expected.size()) return false;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]) || std::fabs(got[i] - expected[i]) > atol) {
            std::printf("FAIL i=%zu got=%g expected=%g\n",
                        i, got[i], expected[i]);
            return false;
        }
    }
    return true;
}

int main() {
    constexpr int N = 3;
    constexpr int D = 2;
    const std::vector<float> x = {1, 0, 0, 1, 1, 1};
    std::vector<float> out(N * D);
    attention_cpu(x.data(), x.data(), x.data(), out.data(), N, D, false);

    // 由同一公式高精度预计算后四舍五入到 FP32 容差范围。
    const std::vector<float> expected = {
        0.802224f, 0.598888f,
        0.598888f, 0.802224f,
        0.751745f, 0.751745f
    };

    for (int i = 0; i < N; ++i) {
        std::printf("O[%d] = [%.6f, %.6f]\n", i,
                    out[i * D], out[i * D + 1]);
    }
    const bool ok = finite_and_close(out, expected, 2e-5f);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

提取、编译、运行：

```bash
mkdir -p /tmp/week4_attention
awk '/^```cpp attention_cpu.cpp$/{p=1;next} /^```$/{if(p){exit}} p' \
  docs/Week4_Attention与FlashAttention完整学习资料.md \
  > /tmp/week4_attention/attention_cpu.cpp
g++ -O2 -std=c++17 /tmp/week4_attention/attention_cpu.cpp \
  -o /tmp/week4_attention/attention_cpu
/tmp/week4_attention/attention_cpu
```

## 10.【必须手写】朴素 CUDA 三阶段

先故意 materialize 完整 `scores/probs [N,N]`，这样你能看清标准实现的问题。

固定接口：

```cpp
__global__ void qk_scores(const float* q, const float* k, float* scores,
                          int n, int d, bool causal);

__global__ void row_softmax(float* scores, int n);

__global__ void pv_output(const float* probs, const float* v, float* out,
                          int n, int d);
```

### 10.1 Kernel 1：`QK^T`

推荐映射：二维 grid，每个线程计算一个 `scores[i,j]`。

```cpp
int j = blockIdx.x * blockDim.x + threadIdx.x;  // key
int i = blockIdx.y * blockDim.y + threadIdx.y;  // query
```

核心式：

```text
scores[i*n+j] = Σ_x q[i*d+x] * k[j*d+x] / sqrt(d)
```

注意 K 虽然数学写成 `K^T`，内存里不必真的先转置。读取 `k[j*d+x]` 就是在取 K 的第 `j` 行。

### 10.2 Kernel 2：row softmax

一个 block 处理一行，复用你现有的 max/sum block reduction：

```text
第一遍：row_max
第二遍：denom = Σ exp(score-row_max)
第三遍：score = exp(score-row_max)/denom
```

现有参考：[operator_practice/softmax/softmax.cu](../operator_practice/softmax/softmax.cu)。这次需要从“单个向量”扩成 `n` 行：`blockIdx.x` 选择行。

### 10.3 Kernel 3：`PV`

二维 grid，每个线程计算一个 `out[i,x]`：

```text
out[i*d+x] = Σ_j probs[i*n+j] * v[j*d+x]
```

### 10.4 三级提示

**提示 1：** 三个 kernel 的输出依次是 `[N,N]`、`[N,N]`、`[N,D]`。先把 shape 写在纸上。

**提示 2：** `qk_scores` 与 naive GEMM 一样，只是 B 的逻辑访问为 `K[j,x]`；`pv_output` 就是普通 `[N,N]×[N,D]`。

**提示 3：** causal 时只要在写 `scores[i,j]` 前判断 `j>i`，写 `-INFINITY`。不要让整行全 mask；标准 causal 每行至少允许 `j=i`。

### 10.5 必测 shape

```text
N=3,   D=2    打印中间矩阵
N=8,   D=8    小规模
N=37,  D=24   非 block 整除
N=128, D=64   常规 head_dim
```

验收：

- 与 `attention_cpu` 比最大绝对误差；
- 检查输出无 NaN/Inf；
- softmax 每行和在 `1±1e-4`；
- causal 模式中 `j>i` 的概率必须为 0；
- `compute-sanitizer --tool memcheck` 为 0 errors。

## 11. 标准 Attention 的计算与显存账本

单 batch、单 head，忽略低阶项：

```text
QK^T: 约 2N²Dh FLOP
PV:   约 2N²Dh FLOP
合计: 约 4N²Dh FLOP
```

若显式保存 `S` 和 `P`，每个都是 `N²` 元素。FP16 单个矩阵：

| `N` | `N²` 元素 | 单个 FP16 `S` 或 `P` | 两者合计 |
|---:|---:|---:|---:|
| 2,048 | 4,194,304 | 8 MiB | 16 MiB |
| 8,192 | 67,108,864 | 128 MiB | 256 MiB |
| 32,768 | 1,073,741,824 | 2 GiB | 4 GiB |

这还只是单 batch、单 head。若真的给每个 head materialize，中间量随 `B×H` 线性放大。实际框架可能复用 buffer 或做融合，不能机械地把两份都视为同时常驻；但 `N²` 中间读写的根本问题仍在。

## 12. Day 1 自测与口述

1. 为什么 `QK^T` 是 `[N,N]`？
2. K 为什么数学上转置，但代码不一定先生成转置矩阵？
3. 为什么除以 `sqrt(Dh)`？
4. softmax 沿哪个维度做？
5. causal mask 为何写成 `-∞` 而不是 0？

闭卷口述：

> 标准 Attention 先用 `QK^T` 计算每个 query 对每个 key 的匹配分数，除以 `sqrt(Dh)` 控制点积随维度增长的量级，再对每个 query 的 key 维做 stable softmax，得到每行和为 1 的权重，最后用这些权重对 V 做加权和。`QK^T` 和 `PV` 都是 GEMM；朴素实现的问题是会把 `[N,N]` 的 score/probability 中间矩阵写入 HBM，长序列时 IO 和存储都呈平方增长。

---

# Day 2：Online Softmax——分块之后怎样保持数值稳定

## 13. 先复盘你已经会的 stable softmax

你已经实现过三遍式版本：[operator_practice/softmax/softmax.cu](../operator_practice/softmax/softmax.cu)。对一行 `x[0..N-1]`：

```text
第一遍：m = max(x)
第二遍：l = Σ_j exp(x_j-m)
第三遍：p_j = exp(x_j-m)/l
```

减 max 的等价性：

```text
exp(x_i-m) / Σ_j exp(x_j-m)
= exp(x_i)exp(-m) / [exp(-m)Σ_j exp(x_j)]
= exp(x_i) / Σ_j exp(x_j)
```

它解决的是数值上溢。例如 FP32 中 `exp(1000)` 会溢出，但最大元素减 max 后是 `exp(0)=1`。

问题是：FlashAttention 不一次看到完整行。它先看到第一个 K tile 的 scores，随后才看到第二个。如果全局 max 出现在后面的 tile，前面依据旧 max 算出的指数和怎么办？

## 14. Online Softmax 保存什么状态

扫描到前 `t` 个元素时，保存：

```text
m_t = max(x_0,...,x_{t-1})
l_t = Σ_{j<t} exp(x_j-m_t)
```

看到新元素 `x`：

```text
m_new = max(m_old, x)
l_new = l_old × exp(m_old-m_new) + exp(x-m_new)
```

为什么旧 `l` 要乘缩放？因为旧和原本以 `m_old` 为参考：

```text
l_old = Σ_old exp(x_j-m_old)
```

要改写成以 `m_new` 为参考：

```text
Σ_old exp(x_j-m_new)
= Σ_old exp(x_j-m_old) × exp(m_old-m_new)
= l_old × exp(m_old-m_new)
```

这不是性能技巧，而是数学等价所必需的换基准。

## 15. 用 `[2,1,4,3]` 一步一步算

初始化：

```text
m = -∞
l = 0
```

| 新元素 `x` | `m_new` | 旧和缩放 | 新元素贡献 | `l_new` |
|---:|---:|---:|---:|---:|
| 2 | 2 | `0` | `exp(0)=1` | 1 |
| 1 | 2 | `1×exp(0)=1` | `exp(-1)` | `1+e^-1≈1.367879` |
| 4 | 4 | `1.367879×exp(-2)` | `1` | `≈1.185122` |
| 3 | 4 | `1.185122` | `exp(-1)` | `≈1.553002` |

最终 `m=4`：

```text
l = exp(2-4)+exp(1-4)+exp(4-4)+exp(3-4)
  = e^-2 + e^-3 + 1 + e^-1
  ≈ 1.553002
```

与一次性 stable softmax 完全是同一个分母。

最容易漏的步骤是读到 `4` 时：旧分母必须从“以 2 为基准”转换成“以 4 为基准”，所以乘 `exp(2-4)`。

## 16. 从单元素升级到 tile 合并

对块 A，已知：

```text
m_a = max(A)
l_a = Σ_{x∈A} exp(x-m_a)
```

对块 B，已知 `(m_b,l_b)`。合并：

```text
m = max(m_a,m_b)
l = exp(m_a-m)l_a + exp(m_b-m)l_b
```

这正是你做分层 reduction 时的新版本：以前每部分只携带 `sum`，现在每部分携带一对 `(max, shifted_exp_sum)`。

合并满足实数数学意义上的结合性，因为无论怎样分组，最终都是：

```text
max(A∪B),  Σ_{x∈A∪B} exp(x-max(A∪B))
```

但浮点加法不满足严格结合律，所以不同并行归约顺序可能产生末位差异。正确性应使用合理容差，而不是要求逐 bit 相等。

## 17.【演示】CPU streaming online softmax

这个程序第一遍 online 扫描得到 `(m,l)`，第二遍才写概率。它减少的是获得归一化统计量所需的全量预扫描方式；要输出所有概率，仍要再次访问输入或保存未归一化值。

```cpp online_softmax_cpu.cpp
// FILE: online_softmax_cpu.cpp
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

struct OnlineState {
    float m;
    float l;
};

OnlineState update(OnlineState old, float x) {
    const float m_new = std::max(old.m, x);
    const float old_scale = std::isinf(old.m)
        ? 0.0f : std::exp(old.m - m_new);
    const float l_new = old.l * old_scale + std::exp(x - m_new);
    return {m_new, l_new};
}

OnlineState merge(OnlineState a, OnlineState b) {
    const float m = std::max(a.m, b.m);
    const float a_term = std::isinf(a.m) ? 0.0f : a.l * std::exp(a.m - m);
    const float b_term = std::isinf(b.m) ? 0.0f : b.l * std::exp(b.m - m);
    return {m, a_term + b_term};
}

std::vector<float> online_softmax(const std::vector<float>& x) {
    OnlineState s{-std::numeric_limits<float>::infinity(), 0.0f};
    for (float v : x) s = update(s, v);
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::exp(x[i] - s.m) / s.l;
    }
    return out;
}

std::vector<float> stable_softmax(const std::vector<float>& x) {
    const float m = *std::max_element(x.begin(), x.end());
    float l = 0.0f;
    for (float v : x) l += std::exp(v - m);
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = std::exp(x[i] - m) / l;
    return out;
}

int main() {
    const std::vector<std::vector<float>> tests = {
        {2, 1, 4, 3},
        {-9, -3, -7, -4},
        {500, 800, 1200, 1499}
    };
    bool ok = true;
    for (const auto& x : tests) {
        const auto a = online_softmax(x);
        const auto b = stable_softmax(x);
        double row_sum = 0.0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) {
            row_sum += a[i];
            max_abs = std::max(max_abs, std::fabs(a[i] - b[i]));
            ok = ok && std::isfinite(a[i]);
        }
        ok = ok && std::fabs(row_sum - 1.0) < 1e-5 && max_abs < 1e-6f;
        std::printf("sum=%.8f max_abs=%g\n", row_sum, max_abs);
    }

    // 验证两个 tile 合并：[2,1] 与 [4,3]。
    OnlineState a{-std::numeric_limits<float>::infinity(), 0.0f};
    OnlineState b = a;
    for (float x : {2.0f, 1.0f}) a = update(a, x);
    for (float x : {4.0f, 3.0f}) b = update(b, x);
    const OnlineState c = merge(a, b);
    ok = ok && std::fabs(c.m - 4.0f) < 1e-6f
            && std::fabs(c.l - 1.5530018f) < 1e-5f;

    std::printf("merged m=%g l=%g %s\n", c.m, c.l, ok ? "PASS" : "FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

编译运行：

```bash
awk '/^```cpp online_softmax_cpu.cpp$/{p=1;next} /^```$/{if(p){exit}} p' \
  docs/Week4_Attention与FlashAttention完整学习资料.md \
  > /tmp/week4_attention/online_softmax_cpu.cpp
g++ -O2 -std=c++17 /tmp/week4_attention/online_softmax_cpu.cpp \
  -o /tmp/week4_attention/online_softmax_cpu
/tmp/week4_attention/online_softmax_cpu
```

## 18.【必须手写】CUDA online softmax

先实现“正确但不追求最优”的单行版本：一个 block 处理一行，一个线程可以 grid-stride 读多个元素。

难点不在 `expf`，而在**线程局部状态如何合并**。每个线程得到局部 `(m,l)`，warp/block reduction 的操作不再是单独 max 或 sum，而是：

```cpp
struct Pair { float m, l; };

__device__ Pair combine(Pair a, Pair b) {
    float m = fmaxf(a.m, b.m);
    float l = a.l * expf(a.m - m) + b.l * expf(b.m - m);
    return {m, l};
}
```

空状态要特殊处理：`(-∞,0)` 与任何有效状态合并后应得到有效状态，不能无保护计算 `-∞ - (-∞)`，否则产生 NaN。

**提示 1：** 线程先顺序扫描自己的 `i=threadIdx.x; i<N; i+=blockDim.x` 元素。

**提示 2：** warp shuffle 每轮同时交换 `m` 和 `l`，再调用 `combine`；warp leader 把 Pair 写 shared，第一 warp 再合并各 warp。

**提示 3：** 最终 `(m,l)` 由线程 0 写 shared；同步后所有线程第二遍写 `expf(x-m)/l`。

必测：

```text
普通值，N=1000
大值 500..1499，N=1000
非整除 N=1031
全负数
```

验收：无 NaN/Inf、行和误差 `<1e-4`、对齐 CPU stable softmax。

## 19. 一个重要边界：online 不等于“一遍就输出所有概率”

只靠 `(m,l)` 扫描一遍，结束前并不知道最终分母，所以不能在看到每个元素时就永久写出最终概率。你可以：

- 第二遍重读输入并归一化；
- 保存未归一化值，最后缩放；
- 在 Attention 中不显式输出 `P`，而是边处理 tile 边把 `P_tile V_tile` 累加进 `O_acc`。

第三条正是 FlashAttention 的入口。

## 20. Day 2 自测与口述

1. `(m,l)` 中的 `l` 是普通 `Σexp(x)` 吗？
2. 新元素小于旧 max 时，旧 `l` 是否需要缩放？
3. 两个 tile 的 `(m,l)` 怎样合并？
4. 为什么空状态要特殊处理？

闭卷口述：

> Online softmax 为已经扫描的部分保存 running max `m` 和相对该 max 的指数和 `l`。当新块带来更大 max 时，旧 `l` 的参考基准变了，因此必须乘 `exp(m_old-m_new)`，再加入新块贡献。tile 级合并也是同一个公式。它让我们不需要先 materialize 完整 score 行就能维护准确的 softmax 统计量，但若要显式输出所有概率，最后仍要归一化；FlashAttention 则直接把局部概率贡献累加到输出，避免保存完整 P。

---

# Day 3：FlashAttention——优化的是数据搬运顺序

## 21. 标准 Attention 的 IO 路径

标准三 kernel 教学实现：

```text
HBM: Q,K
   ↓ 读取
Kernel 1: S = QK^T / sqrt(Dh)
   ↓ 写回完整 S[N,N]
HBM: S
   ↓ 读取 + 写回
Kernel 2: P = softmax(S)
HBM: P[N,N]
   ↓ 再读取 P,V
Kernel 3: O = PV
   ↓ 写回 O
```

虽然 `S/P` 只被短暂使用，却在 HBM 中物化（materialize）。长序列时它们是平方规模。

FlashAttention 原始论文将问题明确为 GPU 内存层次之间的 IO 问题：通过 tiling，把工作放进片上 SRAM，减少 HBM 读写，同时保持 exact attention。[FlashAttention 论文](https://arxiv.org/abs/2205.14135)

## 22. 分块后的数据流

```text
HBM                       片上 SRAM / registers
 Q_i  ────────────────→   固定一个 Q tile
 K_j,V_j ─────────────→   逐块加载 K/V tile
                            ↓
                         S_ij = Q_i K_j^T
                            ↓
                         局部 softmax 统计
                            ↓
                         更新 m / l / O_acc
                            ↓
HBM  ←────────────────   最终只写 O_i（及必要统计量）
```

关键不是“GPU 再也不读 K/V”。当 SRAM 容不下整个序列时，算法仍会按 tile 读取数据；关键是**不把完整 `S` 和 `P` 写回 HBM再读回来**。

## 23. 三个 running state：`m`、`l`、`O_acc`

对 Q tile 中每一行 query，保存：

- `m`：目前看过的所有 score 的最大值；
- `l`：相对 `m` 的指数和；
- `O_acc[Dv]`：相对同一 `m` 的**未归一化输出分子**。

对新的 K/V tile：

```text
S_ij = Q_i K_j^T / sqrt(Dh)          shape [Br,Bc]
m_block = rowmax(S_ij)               shape [Br]
m_new = max(m_old,m_block)           shape [Br]
alpha = exp(m_old-m_new)             shape [Br]
P_ij = exp(S_ij-m_new)               shape [Br,Bc]，尚未除分母
l_new = alpha*l_old + rowsum(P_ij)   shape [Br]
O_acc_new = alpha*O_acc_old + P_ij V_j   shape [Br,Dv]
```

遍历所有 K/V tile 后：

```text
O_i = O_acc / l
```

这里全文统一用“未归一化 `O_acc`”表示法。某些论文伪代码或实现会保存已经除过旧 `l` 的 O，那时更新公式长相不同；不要把两套表示法拼在一起。

## 24. 为什么旧输出也必须乘 `alpha`

设一个 query，`Dv=2`。

第一块：

```text
scores_a = [2,1]
V_a = [[1,0],
       [0,1]]

m_old = 2
l_old = 1 + e^-1
O_acc_old = 1×[1,0] + e^-1×[0,1]
          = [1,e^-1]
```

第二块：

```text
scores_b = [4,3]
V_b = [[2,0],
       [0,2]]
```

新 max 是 4，因此旧贡献都要换成“相对 4”的尺度：

```text
alpha = exp(2-4)=e^-2

l_new = e^-2(1+e^-1) + (1+e^-1)

O_acc_new
= e^-2[1,e^-1] + 1×[2,0] + e^-1×[0,2]
```

如果只缩放 `l_old` 而不缩放 `O_acc_old`，分母认为旧块贡献已变小，分子却仍保留旧尺度，最终输出必然错误。

一句话：

> `l` 是 softmax 分母，`O_acc` 是同一指数权重下的向量分子；换 max 基准时二者必须一起换单位。

## 25. 为什么 FlashAttention 仍是 exact

它没有：

- 删除某些 key；
- 把 softmax 换成线性函数；
- 做低秩近似；
- 改变 `QK^T` 或 `PV` 的数学定义。

它只利用分块与 online softmax 重排运算。从实数数学看结果与标准 Attention 相同；浮点下由于求和顺序不同，末位可能不同，所以工程上用容差对齐，而不是逐 bit 对齐。

原始 FlashAttention forward 的主导矩阵乘 FLOP 仍约：

```text
QK^T: 2N²Dh
PV:   2N²Dv
若 Dv=Dh，总计约 4N²Dh
```

它的主要胜利是减少 HBM IO 和中间存储。不能说“复杂度从 `O(N²)` 变成 `O(N)`”；精确 dense attention 的 score 交互数量仍是平方级。

## 26. causal 分块的三种情况

Q tile 覆盖 query `[q0,q1)`，K tile 覆盖 key `[k0,k1)`：

1. `k0 >= q1`：整块都在未来，直接跳过；
2. `k1 <= q0+1`：对块中所有 query 都合法，可正常计算；
3. tile 跨越因果对角线：算局部 score 后，对 `j>i` 的元素写 `-∞`。

边界警告：如果某行在某个局部 tile 中全部被 mask，该 tile 的局部 row max 是 `-∞`。不能直接算 `-∞-(-∞)`；应让该块对 `l/O_acc` 的贡献为 0。标准 causal self-attention 的全局行至少有自身或过去 token 可见，但局部 tile 仍可能全 mask。

## 27. Forward 伪代码

```text
for each Q tile i:
    load Q_i
    m = -∞ for each query row
    l = 0
    O_acc = 0

    for each legal K/V tile j:
        load K_j, V_j
        S = Q_i K_j^T / sqrt(Dh)
        apply causal mask if needed

        m_block = rowmax(S)
        m_new = max(m, m_block)
        alpha = exp(m-m_new)
        P = exp(S-m_new)       // 未归一化

        l = alpha*l + rowsum(P)
        O_acc = alpha*O_acc + P V_j
        m = m_new

    O_i = O_acc / l
    store O_i
```

## 28. 五个常见误解

| 误解 | 修正 |
|---|---|
| FlashAttention 是近似算法 | dense FlashAttention 是 exact，分块稀疏扩展另说 |
| 把 `O(N²)` 计算变成 `O(N)` | dense score 交互仍为平方级，主要减少 IO/存储 |
| 只是把三个 kernel fusion | fusion 是表象，关键是 tiling + online softmax 让 `S/P` 不落 HBM |
| online softmax 只缩放分母 | `l` 和未归一化 `O_acc` 都要缩放 |
| 有 shared memory 就自动快 | tile、bank conflict、寄存器、occupancy、同步和并行度都要权衡 |

## 29. Day 3 自测与口述

1. `m/l/O_acc` 分别是什么 shape？
2. `P_ij=exp(S_ij-m_new)` 为什么暂时不除 `l_new`？
3. FlashAttention 为什么不保存完整 P？
4. 什么情况下一个 causal K tile 可以整块跳过？
5. 为什么结果 exact 但不保证逐 bit 相同？

闭卷口述：

> FlashAttention 是 IO-aware 的精确 Attention。标准实现会把 `N×N` 的 score 和 probability 写入 HBM，softmax 和 PV 又把它们读回来。FlashAttention 固定 Q tile，流式遍历 K/V tile，在片上计算局部 score，并为每个 query 行维护 running max `m`、相对 max 的指数和 `l`、以及同一尺度下的未归一化输出 `O_acc`。如果新 tile 提高最大值，旧 `l` 和旧 `O_acc` 都乘 `exp(m_old-m_new)`。最后只写归一化输出，因此减少 HBM IO；主导 dense FLOP 仍约 `4N²Dh`。

---

# Day 4：教学版 Tiled Attention——先把状态机写对

## 30. 本日实现边界

第一版故意选择：

```text
FP32 输入 / FP32 累加
单 batch、单 head
一个 block 负责一条 query 行
K/V 按 Bc=16 分块
D≤128
支持任意 N、任意 1≤D≤128
可选 causal
```

它不会快过工业 FlashAttention，甚至未必快过三个高度优化的库 kernel。它的任务只有一个：

> 把不保存完整 `S/P` 的在线 Attention 数据流写正确，并让你能看见每个状态放在哪里。

为什么先一个 block 一条 query，而不是立刻 `Br×Bc` 大 tile？因为后者会同时引入二维 warp 映射、跨 warp reduction、Tensor Core fragment、shared swizzle 和寄存器压力，容易让你在尚未掌握 `m/l/O_acc` 时被工程细节淹没。

## 31. 线程与内存映射

```text
grid.x = N                 每个 block 负责 query i=blockIdx.x
block.x = 128              协作加载和更新 feature
Bc = 16                    每次处理 16 个 K/V
```

shared memory：

```text
q_s[D]          当前 query
k_s[Bc][D]      当前 K tile
v_s[Bc][D]      当前 V tile
scores[Bc]      当前局部 score，随后原地改成 exp(score-m_new)
acc[D]          未归一化 O_acc
m, l, alpha     每行在线状态
```

每个 K/V tile 的阶段：

```text
全 block 协作加载 K/V
→ 前 valid 个线程各算一个 key 的 dot(Q,K_j)
→ 线程 0 对 Bc 个 score 做局部 max/sum（教学简化）
→ 全 block 每个线程负责若干输出 feature，更新 acc[d]
→ 同步后进入下一 tile
```

这里局部 max/sum 串行是有意的。Day 5 会说明怎样换成 warp/block reduction。不要把“教学版串行一小段”误解为 FlashAttention 必须这么做。

## 32.【必须手写】七个空位

先只看下面数据流，自己建立 `week04_attention/tiled_attention.cu`：

```cpp
for (int k0 = 0; k0 < n; k0 += BC) {
    // 1. 协作加载 K/V tile，越界位置不参与
    // 2. 每个有效 key 计算一个局部 score
    // 3. 当前 tile 求 row max
    // 4. m_new=max(m_old,m_block)，计算 alpha，更新 l
    // 5. acc[d] *= alpha
    // 6. acc[d] += Σ_j exp(score_j-m_new)*V_j[d]
}
// 7. out[d] = acc[d]/l
```

### 提示 1：数据依赖

- `scores` 必须在 K tile 完全加载后计算；
- `alpha` 必须在所有 score 完成后计算；
- `acc` 更新结束后才能覆盖 shared K/V；
- 所以每个 tile 至少需要清楚的 block barrier。

### 提示 2：边界

```cpp
int valid = min(BC, n-k0);
```

只遍历 `j<valid`。`D` 不整除 `blockDim.x` 时使用：

```cpp
for (int d=threadIdx.x; d<D; d+=blockDim.x)
```

### 提示 3：causal

全局 key 下标为 `key=k0+j`。若 `causal && key>query`，令 `scores[j]=-INFINITY`。局部 tile 可能全部被 mask，此时其贡献应为 0，不能让 `exp(-∞-(-∞))` 产生 NaN。

## 33.【参考实现】完整可运行教学版

请先独立尝试，再看此程序。参考实现把局部 reduction 留在单线程，只用于教学正确性；真正优化方向见 Day 5。

```cpp tiled_attention_reference.cu
// FILE: tiled_attention_reference.cu
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define CUDA_CHECK(call) do {                                                  \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
        std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,          \
                     cudaGetErrorString(e));                                   \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
} while (0)

constexpr int BC = 16;
constexpr int MAX_D = 128;

void attention_cpu(const float* q, const float* k, const float* v,
                   float* out, int n, int d, bool causal) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> scores(n);
    for (int i = 0; i < n; ++i) {
        float m = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < n; ++j) {
            if (causal && j > i) {
                scores[j] = -std::numeric_limits<float>::infinity();
                continue;
            }
            float dot = 0.0f;
            for (int x = 0; x < d; ++x) dot += q[i*d+x] * k[j*d+x];
            scores[j] = dot * scale;
            m = std::max(m, scores[j]);
        }
        float l = 0.0f;
        for (int j = 0; j < n; ++j) {
            if (causal && j > i) scores[j] = 0.0f;
            else { scores[j] = std::exp(scores[j]-m); l += scores[j]; }
        }
        for (int x = 0; x < d; ++x) {
            float acc = 0.0f;
            for (int j = 0; j < n; ++j) acc += scores[j] * v[j*d+x];
            out[i*d+x] = acc/l;
        }
    }
}

__global__ void tiled_attention(const float* q, const float* k, const float* v,
                                float* out, int n, int d, bool causal) {
    __shared__ float q_s[MAX_D];
    __shared__ float k_s[BC][MAX_D];
    __shared__ float v_s[BC][MAX_D];
    __shared__ float scores[BC];
    __shared__ float acc[MAX_D];
    __shared__ float m;
    __shared__ float l;
    __shared__ float alpha;

    const int tid = threadIdx.x;
    const int query = blockIdx.x;
    if (query >= n) return;

    for (int x = tid; x < d; x += blockDim.x) {
        q_s[x] = q[query*d+x];
        acc[x] = 0.0f;
    }
    if (tid == 0) {
        m = -CUDART_INF_F;
        l = 0.0f;
    }
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(d));
    for (int k0 = 0; k0 < n; k0 += BC) {
        const int valid = min(BC, n-k0);
        const int tile_elements = valid*d;

        // 1. 协作加载当前 K/V tile。
        for (int linear = tid; linear < tile_elements; linear += blockDim.x) {
            const int j = linear/d;
            const int x = linear-j*d;
            k_s[j][x] = k[(k0+j)*d+x];
            v_s[j][x] = v[(k0+j)*d+x];
        }
        __syncthreads();

        // 2. 一个线程计算当前 tile 中一个 key 的完整点积。
        if (tid < valid) {
            const int key = k0+tid;
            if (causal && key > query) {
                scores[tid] = -CUDART_INF_F;
            } else {
                float dot = 0.0f;
                for (int x = 0; x < d; ++x) dot += q_s[x] * k_s[tid][x];
                scores[tid] = dot*scale;
            }
        }
        __syncthreads();

        // 3/4. 教学简化：线程 0 求局部 max，更新 m/l，并把 score 原地改成权重分子。
        if (tid == 0) {
            float m_block = -CUDART_INF_F;
            for (int j = 0; j < valid; ++j) m_block = fmaxf(m_block, scores[j]);

            if (isinf(m_block) && m_block < 0.0f) {
                alpha = 1.0f;  // 整块被 mask，没有新贡献。
                for (int j = 0; j < valid; ++j) scores[j] = 0.0f;
            } else {
                const float m_new = fmaxf(m, m_block);
                alpha = isinf(m) ? 0.0f : expf(m-m_new);
                float tile_l = 0.0f;
                for (int j = 0; j < valid; ++j) {
                    const float w = (isinf(scores[j]) && scores[j] < 0.0f)
                        ? 0.0f : expf(scores[j]-m_new);
                    scores[j] = w;
                    tile_l += w;
                }
                l = alpha*l + tile_l;
                m = m_new;
            }
        }
        __syncthreads();

        // 5/6. 每个线程负责若干输出 feature。
        for (int x = tid; x < d; x += blockDim.x) {
            float add = 0.0f;
            for (int j = 0; j < valid; ++j) add += scores[j] * v_s[j][x];
            acc[x] = alpha*acc[x] + add;
        }
        __syncthreads();  // acc 使用完当前 v_s 后，才能覆盖下一 tile。
    }

    // 7. 最终归一化。
    for (int x = tid; x < d; x += blockDim.x) out[query*d+x] = acc[x]/l;
}

bool run_case(int n, int d, bool causal) {
    if (d < 1 || d > MAX_D) {
        std::fprintf(stderr, "D=%d unsupported; require 1<=D<=%d\n", d, MAX_D);
        return false;
    }
    std::vector<float> hq(n*d), hk(n*d), hv(n*d);
    for (int i = 0; i < n*d; ++i) {
        hq[i] = static_cast<float>((i*17)%23-11)/11.0f;
        hk[i] = static_cast<float>((i*13)%19-9)/9.0f;
        hv[i] = static_cast<float>((i*7)%29-14)/14.0f;
    }
    std::vector<float> ref(n*d), got(n*d);
    attention_cpu(hq.data(), hk.data(), hv.data(), ref.data(), n, d, causal);

    float *dq=nullptr, *dk=nullptr, *dv=nullptr, *dout=nullptr;
    const size_t bytes = static_cast<size_t>(n)*d*sizeof(float);
    CUDA_CHECK(cudaMalloc(&dq, bytes));
    CUDA_CHECK(cudaMalloc(&dk, bytes));
    CUDA_CHECK(cudaMalloc(&dv, bytes));
    CUDA_CHECK(cudaMalloc(&dout, bytes));
    CUDA_CHECK(cudaMemcpy(dq, hq.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, hk.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, hv.data(), bytes, cudaMemcpyHostToDevice));

    tiled_attention<<<n,128>>>(dq,dk,dv,dout,n,d,causal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(got.data(), dout, bytes, cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    bool finite = true;
    for (int i = 0; i < n*d; ++i) {
        finite = finite && std::isfinite(got[i]);
        const float diff = std::fabs(got[i]-ref[i]);
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, diff/std::max(std::fabs(ref[i]),1e-5f));
    }
    const bool ok = finite && (max_abs < 2e-4f || max_rel < 2e-4f);
    std::printf("N=%d D=%d causal=%d max_abs=%g max_rel=%g %s\n",
                n,d,causal,max_abs,max_rel,ok?"PASS":"FAIL");

    CUDA_CHECK(cudaFree(dq));
    CUDA_CHECK(cudaFree(dk));
    CUDA_CHECK(cudaFree(dv));
    CUDA_CHECK(cudaFree(dout));
    return ok;
}

int main() {
    bool ok = true;
    ok = run_case(3,2,false) && ok;
    ok = run_case(8,8,true) && ok;
    ok = run_case(37,24,false) && ok;
    ok = run_case(128,64,true) && ok;
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

提取、编译：

```bash
awk '/^```cpp tiled_attention_reference.cu$/{p=1;next} /^```$/{if(p){exit}} p' \
  docs/Week4_Attention与FlashAttention完整学习资料.md \
  > /tmp/week4_attention/tiled_attention_reference.cu
nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo \
  /tmp/week4_attention/tiled_attention_reference.cu \
  -o /tmp/week4_attention/tiled_attention_reference
/tmp/week4_attention/tiled_attention_reference
compute-sanitizer --tool memcheck /tmp/week4_attention/tiled_attention_reference
```

## 34. 读懂参考实现的七个关键点

1. **没有 `scores[N,N]`**：shared 中只有 `BC=16` 个局部分数。
2. **Q 只加载一次**：一个 block 固定一行 Q，遍历全部 K/V tile。
3. **K/V tile 被覆盖复用**：下一轮装入同一块 shared。
4. **`scores` 原地变权重**：算完 `m_new` 后改成 `exp(score-m_new)`。
5. **`acc` 与 `l` 同尺度**：两者都在 max 更新时乘 `alpha`。
6. **causal 全 mask tile 不制造 NaN**：权重置零，状态不变。
7. **非整除安全**：最后 tile 只处理 `valid=min(BC,n-k0)` 个 key。

## 35. 这份教学代码慢在哪里

- 一个 block 只处理一条 query，Q tile 行复用不足；
- 一个线程串行完成一个长度 D 的 dot；
- 线程 0 串行做局部 max/sum；
- `QK^T`、`PV` 未使用 Tensor Core；
- K/V 在不同 query block 间重复从 HBM 读取；
- acc 放 shared，而高性能实现会设计更细的 register fragment；
- barrier 多，warp 分工简单。

但它已经完成最重要的算法跨越：**不保存 `N²` 中间矩阵，同时得到与标准 Attention 对齐的结果。**

## 36.【挑战】逐步优化，不要一步登天

按顺序选至少两项：

1. 用 warp/block reduction 替代线程 0 的局部 max/sum；
2. 一个 block 处理 `Br>1` 条 query，提高 K/V tile 复用；
3. 加 FP16 输入、FP32 状态；
4. 对 K/V 加双缓冲；
5. 扫描 `Bc=8/16/32`；
6. 用 ncu 对比 DRAM、shared、occupancy、stall。

每次只改一件事，并保留 correctness 与 benchmark 前后数据。

## 37. Day 4 自测与口述

1. 为什么 block 内 `acc[D]` 能跨 K tile 保留？
2. 为什么覆盖 `k_s/v_s` 前要同步？
3. 最后一个 tile 为什么不能循环到固定 BC？
4. 教学代码没有 `S[N,N]`，是否意味着没有计算 score？

闭卷口述：

> 我的教学版一个 block 固定一条 query，把 Q 放 shared，然后以 16 个 key 为一块遍历 K/V。每块只在 shared 中保存 16 个 score，并更新 running `m/l/O_acc`，处理完就覆盖，因此不需要 `N×N` score/probability。新 max 出现时 `l` 和 `acc` 同时乘 alpha，最后 `out=acc/l`。这个版本为了透明度把局部归约串行化，性能不是目标；下一步才是多 query tile、warp reduction、Tensor Core 和双缓冲。

---

# Day 5：A100 上怎样把 Attention 做快

## 38. 先分清：Attention 里不是只有 GEMM

一轮 K/V tile 处理包含：

```text
QK^T 矩阵乘
→ causal mask
→ row max reduction
→ exp
→ row sum reduction
→ m/l/O_acc 重缩放
→ P_tile V_tile 矩阵乘
```

其中 `QK^T` 和 `PV` 适合 Tensor Core；max/sum、mask、指数和状态更新主要由 CUDA Core、特殊函数单元和线程协作完成。

因此：

> 会写 WMMA GEMM，不等于自动会写 FlashAttention；但你已经掌握了最重的两段矩阵乘原语。

回看已有实测：[Tensor Core Profile](../week06_tensorcore/tensor_core_profile.md)。你的教学 WMMA 已看到 HMMA 指令，但 Tensor pipe 活跃度低，说明“算得快”之后必须解决“怎样喂数据”。Attention 同样如此，只是流水中又插入 softmax。

## 39. A100 上各精度负责什么

推荐的理解模型：

| 数据/计算 | 常见精度 | 原因 |
|---|---|---|
| Q/K/V 存储 | FP16 或 BF16 | 减少 HBM/shared 流量，适配 Tensor Core |
| `QK^T` 乘法 | FP16/BF16 Tensor Core | 高吞吐 |
| score/row max | FP32 | 避免精度和范围问题放大 |
| `exp`、`m/l` | FP32 | softmax 对舍入和溢出敏感 |
| `P·V` | 低精度乘，FP32 累加 | 性能与累积精度折中 |
| 最终 O | 按模型需要转回 FP16/BF16 | 供下一层使用 |

A100 支持 FP16、BF16、TF32 Tensor Core。你的 Attention 学习路径优先 FP16/BF16 输入和 FP32 softmax 状态；TF32 更适合从 FP32 接口迁移 GEMM 的场景。

注意：FP32 累加无法恢复 Q/K/V 转成低精度时已经丢失的信息。验证低精度版本时应使用比纯 FP32 更宽的容差，并报告最大绝对误差和最大相对误差。

## 40. 从教学版到 Tensor Core：映射发生在哪里

教学版中：

```cpp
for key in K_tile:
    for d in Dh:
        score[key] += q[d]*k[key][d];
```

高性能版会把多条 query 组成 `Br×Dh` 的 Q tile，把多条 key 组成 `Bc×Dh` 的 K tile：

```text
S_tile[Br,Bc] = Q_tile[Br,Dh] × K_tile^T[Dh,Bc]
```

再把概率 tile 与 V tile 相乘：

```text
O_delta[Br,Dv] = P_tile[Br,Bc] × V_tile[Bc,Dv]
```

这两个矩阵乘可继续拆成 16 级 MMA fragment。工业实现一般使用更底层 MMA/CUTLASS/CuTe 风格的 warp tile，而不是简单地在每个逻辑 tile 外套一次 WMMA。

WMMA 的 fragment lane/register 映射不透明；softmax 又需要按 row 读取 scores。因此真正实现必须设计 score fragment 如何从 MMA accumulator 转换为可归约的行布局，这也是工业代码复杂的原因之一。

## 41. `cp.async` 在这里做什么

Ampere 的异步拷贝可以把 global 数据搬到 shared，并与当前 tile 的计算形成 pipeline。你已有完整基础：[异步拷贝与 pipeline 文档](异步拷贝_pipeline_cooperative_groups学习文档.md)。

双缓冲时间线：

```text
时段 0：加载 K/V tile 0
时段 1：计算 tile 0  | 异步加载 tile 1
时段 2：计算 tile 1  | 异步加载 tile 2
时段 3：计算 tile 2  | 异步加载 tile 3
```

两个 shared buffer ping-pong：

```text
buffer[cur] 供 QK^T / PV 使用
buffer[next] 接收下一块 K/V
```

但 `cp.async` 不会：

- 自动完成 softmax；
- 自动消除 bank conflict；
- 自动减少寄存器；
- 保证加载与计算真的完全重叠；
- 让 compute-bound kernel 必然变快。

你在 GEMM 双缓冲中已经遇到“结构正确但逐 float 异步拷贝指令过多，反而变慢”。Attention 仍要用实际 benchmark 与 ncu 验证，不能因使用了高级指令就先宣布胜利。

## 42. Tile 的资源账本

若片上保存 Q/K/V，并可选保存 score tile，粗略 shared memory：

```text
bytes ≈ sizeof(T) × [Br×Dh + Bc×Dh + Bc×Dv + optional(Br×Bc)]
```

若 K/V 做双缓冲：

```text
bytes ≈ sizeof(T) × [Br×Dh + 2×Bc×(Dh+Dv) + optional(Br×Bc)]
```

例：FP16，`Br=Bc=64,Dh=Dv=64`，保存一份 score：

```text
Q       64×64×2       = 8 KiB
K/V双缓 2×64×(64+64)×2 = 32 KiB
score   64×64×2       = 8 KiB
合计约 48 KiB
```

这只是 shared 账本，不含：

- accumulator fragment 的寄存器；
- 每行 `m/l`；
- `O_acc[Br,Dv]`；
- 对齐、padding、swizzle；
- pipeline state。

tile 变大通常增加复用，却也提高 shared/寄存器压力，降低可驻留 block/warp。和你 A100 GEMM 参数实验一样，存在甜点而不是“越大越好”。

## 43. 从教学版到优化版的阶梯

```text
Level 0  三 kernel naive：完整 S/P 落 HBM
Level 1  教学 tiled：一 query/block，m/l/O_acc 正确
Level 2  多 query/block：一个 K/V tile 被 Br 条 query 复用
Level 3  warp/block reduction：去掉线程 0 串行段
Level 4  FP16/BF16 输入 + FP32 softmax 状态
Level 5  MMA/Tensor Core 做 QK^T 与 PV
Level 6  cp.async + 双缓冲搬 K/V
Level 7  布局/swizzle/warp specialization/shape 专门化
```

每级必须遵守：

```text
先 correctness
→ 无 profiler 的真实性能
→ ncu 指标
→ 解释瓶颈是否按预期转移
```

## 44. A100 与 Hopper 不要混用

| 能力 | A100（SM80） | Hopper（SM90） |
|---|---|---|
| FP16/BF16/TF32 Tensor Core | 支持 | 支持并增强 |
| `cp.async` global→shared | 支持 | 支持 |
| TMA | 不支持 | 支持 |
| WGMMA | 不支持 | 支持 |

本周 A100 代码只使用 SM80 能力。看到现代 FlashAttention/FlashMLA 代码中的 TMA/WGMMA 时，先识别那是架构边界，不要强行用 `-arch=sm_80` 编译。

关于异步数据搬运和 Ampere 特性，以 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 与 [Ampere Tuning Guide](https://docs.nvidia.com/cuda/pdf/NVIDIA_Ampere_Tuning_Guide.pdf) 为准。

## 45.【挑战】Day 5 二选一

### A. FP16 输入

把教学版 Q/K/V 改为 `half`，dot 先用普通转换到 FP32 累加。先不要上 WMMA。

验收：

- 四组 shape PASS；
- 记录 FP32 与 FP16 输入误差；
- 比较 logical bytes；
- 解释为何性能不一定达到 Tensor Core 量级。

### B. K/V 双缓冲结构

用两份 K/V shared buffer，把加载和计算阶段写成 ping-pong；随后再换成 `cuda::memcpy_async` 或 pipeline API。

验收：

- sanitizer 0 errors；
- 对比同步加载和异步加载；
- ncu 检查 stall/SM busy；
- 若变慢，必须解释指令粒度、计算量或流水未重叠，而不是隐去结果。

## 46. Day 5 自测与口述

1. Attention 中哪些部分适合 Tensor Core，哪些不适合？
2. 为什么 FP32 `m/l` 仍有价值？
3. `cp.async` 解决什么、不解决什么？
4. tile 加大有哪些正反效果？
5. A100 为什么不能运行依赖 WGMMA/TMA 的 kernel？

---

# Day 6：从 KV Cache 到 MLA 与 FlashMLA

## 47. 先分清 prefill 和 decode

大模型生成分两类阶段：

### Prefill

一次输入整个 prompt，例如 2048 个 token：

```text
Q length = 2048
K/V length = 2048
```

有大量并行 query，矩阵乘规模大，更容易利用 GPU 计算吞吐。

### Decode

每一步只产生一个新 token：

```text
Q length = 1
K/V length = 历史长度 + 1
```

新 token 的 query 要读取全部历史 K/V。矩阵变“又矮又长”，并行度与数据复用较差，常受 KV cache 读取带宽约束。

## 48. 为什么需要 KV cache

生成到第 `t` 步时，历史 token 的 K/V 不会因为新 token 到来而改变。若每一步都重新计算所有历史 K/V，会反复做相同投影。

所以每层缓存历史：

```text
K_cache: [B,N,Hkv,Dh]
V_cache: [B,N,Hkv,Dv]
```

新一步只计算新 token 的 K/V，追加进 cache；query 与整个 cache 做 Attention。

普通 MHA 单层 KV cache 字节：

```text
bytes = B × N × Hkv × (Dh+Dv) × bytes_per_element
```

若 `Dh=Dv`：

```text
bytes = 2 × B × N × Hkv × Dh × bytes_per_element
```

整个模型还要乘层数 `L`。

例：`B=1,N=32768,Hkv=32,Dh=Dv=128,FP16,L=32`：

```text
单层 = 32768×32×(128+128)×2 bytes
     = 512 MiB
全模型 = 512 MiB×32 = 16 GiB
```

这解释了为什么长上下文推理极其关心 KV cache。

## 49. MHA、MQA、GQA 的区别

假设 query head `Hq=32`、`Dh=128`：

| 机制 | `Hq` | `Hkv` | Q head 怎样使用 KV | 相对 MHA cache |
|---|---:|---:|---|---:|
| MHA | 32 | 32 | 每个 Q head 有独立 K/V head | 1× |
| GQA | 32 | 8 | 每 4 个 Q head 共享一组 K/V | 1/4 |
| MQA | 32 | 1 | 所有 Q head 共享同一组 K/V | 1/32 |

它们主要改变 K/V head 数量，query head 数可以保持不变。cache 字节近似与 `Hkv` 成正比。

代价是共享更多 K/V 可能影响模型表达能力；具体质量取决于训练和模型设计，不能只看 cache 越小越好。

## 50. MLA 的核心动机

Multi-head Latent Attention（MLA）由 DeepSeek-V2 系统性提出，目标之一是进一步压缩推理时需要缓存的表示。[DeepSeek-V2 论文](https://arxiv.org/abs/2405.04434)

普通 MHA 缓存每个 token、每个 KV head 的完整 K/V。MLA 先把 hidden state 下投影成更紧凑的 latent：

```text
c_t^KV = W_DKV h_t          shape [d_c]
```

需要参与 Attention 时，再通过上投影产生与各 head 有关的内容部分：

```text
k_t^C = W_UK c_t^KV
v_t^C = W_UV c_t^KV
```

所以 cache 的主体从：

```text
每 token：Hkv×Dh 的 K + Hkv×Dv 的 V
```

变为更接近：

```text
每 token：一个 d_c 维 latent + 位置编码所需的小部分
```

这里的上标 C 表示内容部分，不是 CUDA C。

## 51. RoPE 为什么让事情复杂一点

RoPE（Rotary Position Embedding）把位置信息以旋转形式作用在 Q/K 上。若把所有 K 都完全吸收到一个与位置无关的 latent 投影里，位置相关部分不一定能直接复用同样的矩阵吸收技巧。

DeepSeek MLA 使用“解耦 RoPE”的思路，概念上把 key/query 分成：

```text
内容部分：可由 latent 投影得到
位置部分：单独应用 RoPE
```

因此推理 cache 不是“只有一个 latent，其他什么都不存”，而通常还需缓存位置相关 key 分量。准确 shape 必须按具体模型配置和实现阅读。

## 52. 矩阵吸收：不一定真的重建完整 K/V

若直接每步从 `c^KV` 重建所有 head 的 K/V，可能用额外计算换 cache 带宽。MLA 的推理实现还可以利用线性代数结合律，把某些上投影权重吸收到 query 或输出投影中。

直觉示意：

```text
q^T (W_UK c)
= (W_UK^T q)^T c
```

原本“先把 c 展开成完整 key，再和 q 点积”，可以改为“先把 q 投到 latent 空间，再和 c 点积”。

同理，value/output 侧也可重新结合。这样减少重建和读取完整 K/V，但会改变 kernel 的矩阵 shape 与数据流。

不要把矩阵吸收说成免费：它在存储、带宽、计算和 kernel 形状之间重新做权衡。

## 53. MLA cache 账本怎么做

普通 MHA 每 token、每层（忽略 batch）：

```text
MHA elements = Hkv×(Dh+Dv)
```

MLA 概念账本：

```text
MLA elements = d_c + d_rope_cache
```

压缩比近似：

```text
ratio = (d_c+d_rope_cache) / [Hkv×(Dh+Dv)]
```

例子只用于练公式：

```text
Hkv=32, Dh=Dv=128
d_c=512, d_rope_cache=64

MHA = 32×256 = 8192 elements/token/layer
MLA = 512+64 = 576 elements/token/layer
ratio = 576/8192 ≈ 7.03%
```

这组数字与具体模型是否完全一致无关；真实报告必须读取模型配置，不能把练习数字冒充 DeepSeek 某版本参数。

## 54. FlashMLA 是什么

必须分开：

- **MLA**：模型的 Attention 架构和表示方式；
- **FlashMLA**：针对 MLA 数据流优化的 Attention kernel/工程库。

截至本教材编写时，DeepSeek 官方 [FlashMLA 仓库](https://github.com/deepseek-ai/FlashMLA) 的现行 README 提供 dense/sparse、prefill/decode 等 kernel，并列出其支持矩阵。当前主要要求 SM90/SM100 以及相应 CUDA 版本。

这意味着：

> A100 是 SM80，适合学习 MLA 数学、写简化教学 kernel、分析 cache 和数据流；但不能假设能直接编译运行当前官方 FlashMLA 生产 kernel。

这不是你的环境配置错误，而是 GPU 架构要求。看到 TMA、WGMMA 或 SM90 特化路径时，只做源码阅读，不在 A100 上强行移植。

## 55. 阅读 FlashMLA README 的正确问题

不要从模板元编程第一行硬啃。先回答：

1. 这是 prefill 还是 decode kernel？
2. dense 还是 sparse？
3. 输入 Q 与 KV cache 的 shape/layout 是什么？
4. README 所说 MLA mode 映射成 MQA 还是 MHA 计算形式？
5. KV cache dtype 是 BF16 还是 FP8？
6. 支持的 compute capability 是什么？
7. benchmark 报告的是 TFLOPS、带宽、延迟还是吞吐？
8. split-K、tile scheduler 或 metadata 做什么？

先画数据流，再定位到对应源码目录；不要只因为文件名有 `flash` 就假定它与原始 FlashAttention v1 的调度完全相同。

## 56. Day 6 自测

1. prefill 与 decode 的 query length 通常有什么不同？
2. 为什么 decode 特别关心 KV cache 带宽？
3. `Hq=32,Hkv=8` 是 MHA、GQA 还是 MQA？cache 相对 `Hkv=32` 是多少？
4. MLA 缓存 latent 后，为何还可能保留 RoPE 相关分量？
5. MLA 和 FlashMLA 各是什么层次？
6. 为什么当前官方 FlashMLA 不能直接作为 A100 作业？

闭卷口述：

> Decode 每步只有一个新 query，却要读取全部历史 K/V，因此常受 cache 带宽限制。MHA 为每个 query head 缓存独立 K/V；GQA/MQA 通过共享 KV head 缩小 cache。MLA 更进一步，把每个 token 的 K/V 信息压缩成低维 latent，并为位置编码保留必要分量，还可通过矩阵吸收避免显式重建某些完整 K/V。FlashMLA 是实现这些数据流的高性能 kernel 库，不是 MLA 架构本身；当前官方实现主要面向 SM90/SM100，所以 A100 本周以原理、账本和教学实现为主。

---

# Day 7：Benchmark、Nsight Compute 与闭卷复盘

## 57. 正确测量比“跑出一个毫秒数”难

建议 benchmark 规则：

```text
warmup: 10 次
正式迭代: 至少 100 次
CUDA Event: 只包要比较的 kernel 序列
stop event: 必须 synchronize 后读取 elapsed time
分配、初始化、H2D/D2H: 与 kernel-only 分开报告
```

每次报告：

```text
GPU 精确型号（A100 40GB/80GB、PCIe/SXM）
CUDA / driver 版本
nvcc 参数（-O3 -arch=sm_80 -lineinfo）
B/H/N/Dh/Dv
dtype
causal 或 non-causal
平均/中位延迟
最大误差
```

你已经在 A100 GEMM 学过：ncu 会重放 kernel、插桩或锁频，因此 **ncu 下程序自己打印的 CUDA Event 时间不能当真实性能**。真实性能脱离 ncu 测；ncu 用来读指标。

## 58. 建议的结果表

| 版本 | B/H/N/Dh | dtype | causal | ms | 是否保存 S/P | max abs | reg/thread | occ | DRAM | shared | Tensor pipe |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| naive 3 kernel | 1/1/128/64 | FP32 | no | 实测 | 是 | 实测 | ncu | ncu | ncu | ncu | 0 |
| teaching tiled | 1/1/128/64 | FP32 | no | 实测 | 否 | 实测 | ncu | ncu | ncu | ncu | 0 |
| 你的优化版 | … | … | … | 实测 | 否 | 实测 | ncu | ncu | ncu | ncu | 若用 MMA则测 |

不要先填“应该更快”。小 N 下教学 tiled 可能因同步和串行归约更慢，但仍能证明不保存平方中间量。

## 59. ncu 命令

先找到真实 kernel 名和 launch：

```bash
ncu --set basic ./attention_bench
```

过滤并保存报告：

```bash
ncu --set full -s 1 -c 1 \
    --kernel-name regex:.*attention.* \
    -o attention_tiled \
    ./attention_bench
```

若 full 太慢，按问题选 section/metric。判读顺序：

```text
1. 确认 profile 的是目标 kernel、目标 shape
2. Speed of Light：compute / memory 谁高
3. DRAM 与 L2：HBM 流量是否按预期下降
4. L1/TEX/shared：瓶颈是否转移到片上
5. registers/thread、theoretical/achieved occupancy
6. warp stall reasons
7. 若使用 MMA：Tensor pipe 是否真正活跃
8. 回到无 ncu benchmark 验证 wall-clock
```

常用命令模板：

```bash
ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --kernel-name regex:.*attention.* \
    ./attention_bench
```

具体 metric 名随 Nsight Compute/GPU 版本变化，先用：

```bash
ncu --query-metrics | rg -i 'dram|l2|shared|occupancy|tensor|hmma'
```

## 60. 你希望看到什么证据

### naive → tiled

预期逻辑：不再写/读完整 `S/P`，HBM logical traffic 下降。但教学实现可能因 K/V 跨 query block 重读、并行度低而未充分兑现。

需要证据：

- 代码和内存分配中不再存在 `N²` buffer；
- profiler 中相应 global store/load 减少；
- 峰值显存或可运行 N 上限改善；
- 正确性仍对齐。

### tiled → 多 query tile

预期逻辑：同一 K/V shared tile 被多条 query 复用，HBM 读取进一步摊薄。

需要证据：

- DRAM bytes/request 变化；
- 性能随 `Br` 的扫描；
- shared/寄存器/occupancy 是否成为新瓶颈。

### CUDA Core → Tensor Core

需要三类证据：

- API/代码路径使用 MMA；
- SASS 出现 HMMA/MMA；
- Tensor pipe > 0 且实际时间有合理变化。

只看到 HMMA 不等于 Tensor Core 已吃饱；你已经在 Week 3 见过 5.71% pipe 的反例。

## 61. Prefill 与 decode 不应共用一句瓶颈结论

### Prefill

- Q/K/V 序列都长；
- 大矩阵乘占比高；
- 更容易使用 Tensor Core；
- FlashAttention 重点减少 `N²` 中间 IO。

### Decode

- query length 常为 1；
- 读取长 KV cache；
- 矩阵很瘦，并行度与复用不同；
- cache 布局、量化、分页和 batch 合并更关键。

同一个 kernel 在 `N=128` 与 `N=8192`，或 prefill 与 decode，瓶颈画像可以完全不同。结论必须带 shape。

## 62. 常见错误诊断表

| 症状 | 优先检查 |
|---|---|
| 输出全 NaN | 全 mask tile、`-∞-(-∞)`、分母 0 |
| 第一个 tile 对，第二个开始错 | `m/l/O_acc` 重缩放或 barrier |
| 只有非整除 N 错 | `valid`、越界加载、最后 tile |
| causal 第 0 行错 | 自身是否允许、全 mask 处理 |
| FP32 对，FP16 偏差大 | 输入范围、累加精度、容差、指数前是否转 FP32 |
| ncu 下时间巨大 | kernel replay/插桩，不要拿该时间做 benchmark |
| Tensor pipe 为 0 | 未走 MMA 路径、编译架构/数据类型不对 |
| occupancy 很低 | registers、shared、block size、grid 是否填满 SM |
| DRAM 不高但很慢 | shared/依赖延迟/同步/低并行度/指令吞吐 |

## 63. 最终闭卷自测

### 概念题

1. Q、K、V 为什么来自同一个 X 却使用不同投影？
2. `QK^T` 每个元素是什么意思？
3. 为什么 scaled attention 除以 `sqrt(Dh)`？
4. stable softmax 为什么减 max？
5. online softmax 保存哪两个统计量？
6. 新 max 出现时为什么旧 `l` 要缩放？
7. FlashAttention 为什么还要保存 `O_acc`？
8. 为什么 `O_acc` 与 `l` 必须一起缩放？
9. FlashAttention 是否减少 dense forward 的主导平方 FLOP？
10. causal tile 有哪三种位置关系？
11. Tensor Core 加速 Attention 的哪两段？
12. `cp.async` 为什么不保证加速？
13. prefill 与 decode 有什么差异？
14. MHA、GQA、MQA 的 `Hkv` 如何变化？
15. MLA 与 FlashMLA 是不是同一层概念？

### 计算题

1. `B=2,H=8,N=1024,Dh=64` 时 Q 有多少元素？
2. 单 head FP16 的 `S[N,N]` 在 `N=8192` 时多少 MiB？
3. `B=1,N=32768,Hkv=8,Dh=Dv=128,FP16,L=32` 的 KV cache 多少 GiB？
4. `m_old=2,l_old=1.5,m_block=4,l_block=1.2`，合并后的 `m/l` 是什么？
5. `Br=Bc=64,Dh=Dv=128`，FP16 Q/K/V 单缓冲且保存 FP16 score，shared 粗略多少 KiB？

### 代码诊断题

1. 只对 `l_old` 乘 alpha，不对 `O_acc` 乘，会出现什么逻辑错误？
2. 在部分线程提前 `return` 后执行 `__syncthreads()`，为什么危险？
3. 把 causal 被 mask 的 score 写 0，softmax 后为什么仍会关注未来？

## 64. 自测答案

### 计算题答案

1. `2×8×1024×64=1,048,576` 元素。
2. `8192²×2 bytes=128 MiB`。
3. `1×32768×8×(128+128)×2×32 = 4 GiB`。
4. `m=4`，`l=1.5e^(2-4)+1.2e^(4-4)≈1.4030`。
5. `Q=16 KiB`，K=16 KiB，V=16 KiB，score=8 KiB，合计约 56 KiB，不含 padding、状态和双缓冲。

### 代码诊断答案

1. 分母已换成新 max 的尺度，分子仍是旧尺度，旧 value 贡献被错误放大。
2. block barrier 要求 block 内所有未退出线程按一致控制流到达；部分退出可能导致死锁或未定义行为。
3. `exp(0)>0`，0 只是普通有限 logit；mask 应使对应指数贡献为 0，通常写 `-∞` 或在指数阶段显式置零。

## 65. 三段面试口述

### 3 分钟：Attention

> Attention 对每个 token 生成 Q、K、V。`QK^T` 计算每个 query 与所有 key 的匹配分数，除以 `sqrt(Dh)` 控制点积方差，再沿 key 维做 stable softmax，得到每行和为 1 的权重，最后 `PV` 对 value 加权。多头只是让多个投影子空间独立执行这一过程。标准三阶段的主要系统问题是显式产生 `N×N` score/probability，中间内存和 IO 随序列长度平方增长。

### 5 分钟：FlashAttention

> FlashAttention 是 IO-aware 的 exact attention。它固定 Q tile，流式遍历 K/V tile，在片上算局部 score，不把完整 S/P 写回 HBM。为保持与全局 softmax 等价，每个 query 行维护 running max m、相对该 max 的指数和 l、以及未归一化输出 O_acc。新 tile 提高 max 时，旧 l 和 O_acc 都乘 `exp(m_old-m_new)`，再加入新块贡献；结束后 `O=O_acc/l`。主导 dense FLOP 仍约 `4N²Dh`，收益主要来自减少 HBM IO。在 A100 上，可用 FP16/BF16 Tensor Core 做 QK 与 PV，用 FP32 做 softmax 状态，并用 cp.async 双缓冲 K/V；但 tile、寄存器、shared、同步和 occupancy 必须实测权衡。

### 2 分钟：MLA

> 自回归 decode 每步只有一个新 query，却要读取全部历史 K/V，所以 KV cache 带宽是关键。MHA 每个 query head 有独立 KV；GQA/MQA 通过共享 KV head 缩小 cache。MLA 把每个 token 的 KV 信息下投影到紧凑 latent，缓存 latent 和必要的 RoPE 位置分量，并可通过矩阵吸收避免显式重建部分完整 K/V，从而进一步降低 cache 与带宽。FlashMLA 是实现 MLA 数据流的高性能 kernel 库，不是 MLA 架构本身；当前官方版本主要要求 SM90/SM100，因此 A100 上应掌握原理、账本和教学实现，而不是声称直接运行其生产 kernel。

---

# 附录 A：一页公式速查

## 标准 Attention

```text
S = QK^T/sqrt(Dh)
P = softmax_rows(S)
O = PV
```

## stable softmax

```text
m = max_j x_j
l = Σ_j exp(x_j-m)
p_j = exp(x_j-m)/l
```

## online softmax 合并

```text
m = max(m_a,m_b)
l = exp(m_a-m)l_a + exp(m_b-m)l_b
```

## tiled Attention（未归一化 O_acc 表示）

```text
m_new = max(m_old,rowmax(S_tile))
alpha = exp(m_old-m_new)
P_tile = exp(S_tile-m_new)
l_new = alpha*l_old + rowsum(P_tile)
O_acc_new = alpha*O_acc_old + P_tile V_tile
O = O_acc/l
```

## KV cache

```text
bytes = B×N×Hkv×(Dh+Dv)×bytes_per_element×layers
```

# 附录 B：推荐阅读（按顺序）

1. 本仓库 [stable softmax 实现](../operator_practice/softmax/softmax.cu)。
2. [FlashAttention 原始论文](https://arxiv.org/abs/2205.14135)：优先看算法图、IO 动机和 forward 伪代码。
3. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)：异步数据拷贝与矩阵乘部分。
4. [DeepSeek-V2 论文](https://arxiv.org/abs/2405.04434)：MLA 章节与附录。
5. [DeepSeek FlashMLA 官方仓库](https://github.com/deepseek-ai/FlashMLA)：先读支持矩阵和接口，再看 kernel。

# 附录 C：本周完成清单

- [ ] 能从 `[B,H,N,Dh]` 解释每个维度。
- [ ] 手算 `N=3,D=2` Attention。
- [ ] 独立完成 naive 三 kernel。
- [ ] 独立写出 online `(m,l)` 更新和 tile 合并。
- [ ] 用数字解释 `O_acc` 为什么要重缩放。
- [ ] 跑通教学 tiled reference 并通过 sanitizer。
- [ ] 至少完成两个 Day 5 挑战项或一个挑战项加完整 ncu 报告。
- [ ] 算清一组 MHA/GQA/MLA cache 账本。
- [ ] 完成 benchmark 表，不混用 ncu 插桩时间。
- [ ] 闭卷讲完 Attention、FlashAttention、MLA 三段口述。
