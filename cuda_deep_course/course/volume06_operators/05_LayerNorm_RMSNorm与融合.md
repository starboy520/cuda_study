# 05 LayerNorm、RMSNorm 与融合

> Transformer 里除了 GEMM 和 Softmax，第三类高频算子是**归一化**——LayerNorm 和
> RMSNorm。它们计算简单，但因为是**访存受限**的，"融合"成了优化的关键词。本章讲清
> 它们的数学、归约结构，以及"kernel 融合"为什么能省一大半内存带宽。

## 1. LayerNorm 数学

对一个向量 `x = [x_0, ..., x_{n-1}]`（通常是一个 token 的特征向量），LayerNorm：

```text
μ = (1/n) Σ x_i                       // 均值
σ² = (1/n) Σ (x_i - μ)²               // 方差
y_i = (x_i - μ) / sqrt(σ² + ε) · γ_i + β_i   // 归一化 + 缩放平移
```

- `ε` 是防止除零的小常数（如 1e-5）。
- `γ`（scale）、`β`（shift）是可学习参数，逐元素。
- 直觉：**把每个特征向量拉到均值 0、方差 1，再用 γ/β 微调**。

## 2. RMSNorm：更简单的变体

RMSNorm（Root Mean Square Norm）省掉了减均值，只用均方根归一化：

```text
rms = sqrt((1/n) Σ x_i² + ε)
y_i = x_i / rms · γ_i
```

```text
对比：
LayerNorm：要算 均值 μ 和 方差 σ²（两个统计量）
RMSNorm：  只算 Σx_i²（一个统计量），不减均值、无 β
→ RMSNorm 计算更少、归约更少，现代大模型（LLaMA 等）常用它
```

## 3. 结构：归约 + elementwise（又是归约！）

LayerNorm 的核心也是**归约**：

```text
① Σ x_i      → 归约（求和算 μ）
② Σ (x_i-μ)² → 归约（求和算 σ²）
③ y_i = ...  → elementwise（归一化）
```

和 softmax 一样，归一化类算子的骨架都是"**几个归约 + 一个逐元素**"。所以卷四的
归约技术（树形 + warp shuffle）在这里继续复用。

### 3.1 一个技巧：一遍求 μ 和 σ²

方差公式可以变形，让均值和方差**一遍归约**算完：

```text
σ² = E[x²] - (E[x])²       // 方差 = 平方的均值 - 均值的平方

所以一遍扫描同时累加 Σx_i 和 Σx_i²：
  sum  = Σ x_i
  sqSum = Σ x_i²
  μ = sum / n
  σ² = sqSum / n - μ²
→ 只扫一遍 x，而不是"先求 μ 再扫一遍求方差"
```

> 注意：这个公式在数值上不如"两遍法"稳定（大数相减可能丢精度），但工程中常用，
> 必要时用 Welford 在线算法提精度。理解"一遍归约"的思路是重点。

## 4. GPU 实现：一个 block 一行

```cpp
__global__ void layerNorm(const float* x, const float* gamma, const float* beta,
                          float* y, int rows, int cols, float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  __shared__ float s_sum;
  __shared__ float s_sqSum;
  __shared__ float reduce[256];
  __shared__ float reduce2[256];

  // ① 一遍归约：同时累加 sum 和 sqSum
  float sum = 0.0f, sqSum = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    float v = x[row * cols + c];
    sum += v;
    sqSum += v * v;
  }
  reduce[tid] = sum; reduce2[tid] = sqSum;
  __syncthreads();
  for (int s = blockDim.x/2; s > 0; s /= 2) {
    if (tid < s) { reduce[tid] += reduce[tid+s]; reduce2[tid] += reduce2[tid+s]; }
    __syncthreads();
  }
  if (tid == 0) { s_sum = reduce[0]; s_sqSum = reduce2[0]; }
  __syncthreads();

  // ② 算统计量
  float mean = s_sum / cols;
  float var  = s_sqSum / cols - mean * mean;
  float invStd = rsqrtf(var + eps);     // 1/sqrt，硬件快速指令

  // ③ elementwise 归一化
  for (int c = tid; c < cols; c += blockDim.x) {
    float v = x[row * cols + c];
    y[row * cols + c] = (v - mean) * invStd * gamma[c] + beta[c];
  }
}
```

要点：
- 用 `rsqrtf`（1/sqrt 的硬件快速指令）算 `invStd`，比 `1.0f/sqrtf()` 快。
- 一遍归约同时算 sum 和 sqSum（两个归约数组并行)。
- grid-stride 处理长行。

## 5. Kernel 融合（Fusion）：本章的核心价值

### 5.1 为什么归一化算子要融合

LayerNorm、softmax 这类算子**算术强度极低**（每个元素就几次运算），是**纯访存受限**。
对访存受限算子，性能 ≈ 内存读写次数。如果每个操作都是独立 kernel：

```text
不融合（每步一个 kernel，每个 kernel 都要读写 global）：
  kernel1: 读 x，算 μ，写回           （读 x 一遍）
  kernel2: 读 x，读 μ，算 y，写 y      （又读 x 一遍 + 写 y）
  → x 被从 global 读了 2 遍，中间结果来回 global

融合（一个 kernel 干完）：
  读 x 一遍 → 在 shared/寄存器里算完 μ、σ²、y → 写 y 一遍
  → x 只读 1 遍，没有中间结果往返 global
```

### 5.2 融合省的是内存带宽

```text
访存受限算子的优化 = 减少 global 内存读写次数
融合把"多个 kernel 各读各写"合并成"读一次、算完、写一次"
→ 对带宽受限算子，融合常带来接近 N 倍的提升（N=融合前的 kernel 数）
```

这就是为什么真实框架（PyTorch 的 fused kernels、TensorRT、Triton）大量做算子融合
——尤其是 norm、激活、残差这些访存受限的小算子。

### 5.3 更进一步：融合相邻算子

实战中常把一连串操作融合进一个 kernel：

```text
典型融合：  Linear(GEMM) → 残差加 → LayerNorm
            或  bias 加 → GELU 激活 → dropout
把这些"小而碎、访存受限"的操作贴在大算子后面，复用已经在寄存器/shared 里的数据，
避免每步都来回 global memory。
```

> 一句话：**融合的本质是"让数据在快内存里多停留一会、多干几件事，少回 global"。**
> 这是访存受限场景最有效的优化，比单独优化每个 kernel 收益大得多。

## 6. Elementwise、Broadcast 与 Reduction 的融合视角

归纳一下三类基本操作，融合就是把它们串起来：

```text
elementwise：out[i] = f(in[i])         逐元素，如激活、加 bias
broadcast：  out[i][j] = a[i][j] + b[j]  一个维度广播，如加 bias 向量
reduction：  out = Σ in[i]              归约，如 sum/max（softmax/norm 的核心）

融合策略：
  - elementwise 链 → 直接串在一起，零额外访存
  - reduction + 后续 elementwise → 归约结果留 shared，紧接着算（softmax/norm）
  - broadcast → 把被广播的小数据放 shared/constant，复用
```

## 7. 本章小结

```text
LayerNorm：减均值/除标准差 + γ/β；RMSNorm 更简（只均方根，无均值/β）
结构：归约(求 μ、σ²) + elementwise(归一化) —— 复用卷四归约
技巧：σ²=E[x²]-E[x]² 一遍归约同时算 sum 和 sqSum；用 rsqrtf
核心——融合：归一化是访存受限，性能≈内存读写次数
  不融合：x 读多遍、中间结果往返 global
  融合：  读一遍 → 快内存算完 → 写一遍 → 省接近 N 倍带宽
  实战：GEMM→残差→LayerNorm、bias→激活 等串成一个 kernel
三类操作：elementwise / broadcast / reduction，融合把它们串起来减访存
```

## 8. 资料映射

- PMPP：Reduction、Parallel Patterns。
- CUDA C++ Best Practices Guide：Memory Optimizations（融合减访存）。
- Triton / Apex / TensorRT 文档：fused LayerNorm、fused kernels 的工程实现。
- 配套：[卷四第 03 章 Reduction](../volume04_parallel_algorithms/03_Reduction从错误到优化.md)、[卷六第 04 章 Softmax](04_Softmax数值稳定与多级归约.md)。
