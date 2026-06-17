# 04 Softmax 数值稳定与多级归约

> GEMM 之后，第二重要的算子是 Softmax——它是 attention 的核心、分类输出的归一化。
> Softmax 看似简单（指数 + 归一化），却藏着一个**致命的数值陷阱**和一个**归约结构**。
> 本章讲清"为什么要减最大值"和"怎么用卷四的归约高效实现"。

## 1. 数学定义

对一个向量 `x = [x_0, x_1, ..., x_{n-1}]`，softmax 把它变成概率分布：

```text
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

每个输出在 (0,1) 之间，且所有输出之和为 1。直觉：**放大差异 + 归一化成概率**。

## 2. 致命陷阱：直接算会溢出

### 2.1 问题

直接按定义算 `exp(x_i)`：

```text
如果 x_i = 1000：
  exp(1000) = 无穷大（float 最大约 3.4e38，exp(89) 就溢出了）
  → inf / inf = NaN → 整个结果烂掉
```

`exp` 增长极快，输入稍大就溢出 float 范围。真实模型里 logits 完全可能到几十上百，
**朴素 softmax 必然在某些输入上炸成 NaN**。

### 2.2 解法：减去最大值（数值稳定 softmax）

利用一个数学恒等式——**softmax 减去任意常数 c 结果不变**：

```text
softmax(x_i) = exp(x_i) / Σ exp(x_j)
             = exp(x_i - c) / Σ exp(x_j - c)   （分子分母同乘 exp(-c)，约掉）
```

取 `c = max(x)`，则每个 `x_i - max ≤ 0`，于是 `exp(x_i - max) ≤ 1`，**永不溢出**：

```text
稳定版 softmax：
  m = max(x)                        // ① 求最大值
  e_i = exp(x_i - m)                // ② 减最大值再指数（每个 ≤ 1，安全）
  s = Σ e_i                         // ③ 求和
  out_i = e_i / s                   // ④ 归一化
```

```text
对比 x = [1000, 1001, 1002]：
朴素：exp(1000)=inf → NaN ❌
稳定：m=1002, exp(-2),exp(-1),exp(0) = 0.135, 0.368, 1.0 → 正常 ✅
```

> **这是 softmax 第一课，也是面试必考**：为什么减最大值？答案——防 `exp` 溢出，
> 且因为 softmax 平移不变，结果不受影响。

配套实验 `labs/06_operators/softmax/` 实测了这一点（T4）：

```text
大值数据 [500,1500]：
  naive   每行和 = NaN  ❌  （exp(500+) 溢出成 inf → inf/inf）
  stable  每行和 = 1.0  ✅  （减最大值后 exp ≤ 1）
普通小值数据：两版都正常、结果一致（验证平移不变）
```

可以亲手跑一遍看朴素版怎么炸成 NaN：

```bash
make -C labs/06_operators/softmax clean all
./labs/06_operators/softmax/softmax
```

## 3. 结构：三次归约 + 一次 elementwise

看稳定版的四步，本质是**两个归约 + 两个逐元素**：

```text
① m = max(x)        → 归约（max）
② e_i = exp(x_i-m)  → elementwise
③ s = Σ e_i         → 归约（sum）
④ out_i = e_i / s   → elementwise
```

所以 softmax 复用的正是**卷四的归约技术**——只是算子从"加"换成"max"和"sum"。两个
归约都可以用树形归约 + warp shuffle 高效实现。

## 4. GPU 实现：按行 softmax

深度学习里 softmax 通常作用在矩阵的**每一行**（如 attention 的每个 query 对所有 key）。
常见映射：**一个 block 处理一行**。

### 4.1 一个 block 一行（行长 ≤ block 能处理）

```cpp
__global__ void softmaxRow(const float* x, float* out, int rows, int cols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  __shared__ float reduce[256];

  // ① 求本行最大值（block 内归约）
  float localMax = -INFINITY;
  for (int c = tid; c < cols; c += blockDim.x)
    localMax = fmaxf(localMax, x[row * cols + c]);
  reduce[tid] = localMax;
  __syncthreads();
  // 树形归约求 max（同卷四，算子换成 fmaxf）
  for (int s = blockDim.x/2; s > 0; s /= 2) {
    if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
    __syncthreads();
  }
  float m = reduce[0];
  __syncthreads();

  // ② 求 exp 之和（block 内归约）
  float localSum = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x)
    localSum += expf(x[row * cols + c] - m);
  reduce[tid] = localSum;
  __syncthreads();
  for (int s = blockDim.x/2; s > 0; s /= 2) {
    if (tid < s) reduce[tid] += reduce[tid + s];
    __syncthreads();
  }
  float sum = reduce[0];
  __syncthreads();

  // ③ 归一化写回（elementwise）
  for (int c = tid; c < cols; c += blockDim.x)
    out[row * cols + c] = expf(x[row * cols + c] - m) / sum;
}
```

### 4.2 grid-stride 处理长行

`for (c = tid; c < cols; c += blockDim.x)` 是 **grid-stride 循环**：当一行比 block
线程数长时，每个线程处理多个元素。这样固定 block 大小也能处理任意行长。

### 4.3 优化点

```text
- 用 warp shuffle 做归约收尾（卷四第 03 章），省 shared 往返。
- exp(x_i - m) 算了两次（②求和、③归一化）→ 可以缓存到 shared/寄存器复用。
- 长行可考虑多 block 协作 + 多阶段归约（同 reduction 多阶段）。
```

## 5. 进阶：Online Softmax（一遍扫描）

### 5.1 朴素版扫了 3 遍

上面的实现读了 x **三遍**（求 max、求 sum、归一化），访存是瓶颈。

### 5.2 Online softmax：max 和 sum 一遍算完

有个巧妙的技巧——**边扫描边维护"当前最大值"和"当前和"**，遇到更大的值时按比例
修正已累积的和：

```text
维护 (m, s)，初始 m=-inf, s=0
对每个新 x_i：
  m_new = max(m, x_i)
  s = s · exp(m - m_new) + exp(x_i - m_new)   // 把旧和按新最大值重新缩放
  m = m_new
一遍扫完，(m, s) 就是最终的最大值和指数和
```

这把"两遍归约"压成"一遍"，是 **FlashAttention** 的核心思想之一——它让 attention
能在不存储完整注意力矩阵的情况下、分块流式地算 softmax，大幅省显存。

> 本章了解 online softmax 的思想即可，完整 FlashAttention 是进阶专题。但理解"一遍
> 维护 (max, sum) 并动态缩放"这个技巧，是读懂现代 attention 优化的钥匙。

## 6. 数值正确性验证

```cpp
// CPU 参考也要用稳定版（减 max），否则参考值自己就 NaN
// 验证：每行输出之和应 ≈ 1
for (each row) assert(abs(rowSum - 1.0) < 1e-4);
// 和 CPU 稳定版逐元素比，相对误差 < 1e-5
```

## 7. 本章小结

```text
softmax(x)_i = exp(x_i) / Σ exp(x_j)
陷阱：直接 exp 会溢出成 NaN
解法：减最大值 —— exp(x_i - max)，利用 softmax 平移不变性，结果不变且不溢出
结构：max 归约 → exp(elementwise) → sum 归约 → 除(elementwise)
     归约复用卷四技术，算子换成 max/sum
实现：一个 block 一行 + grid-stride + warp shuffle 收尾
进阶：online softmax 一遍维护 (max,sum) 并动态缩放 → FlashAttention 基石
验证：每行和≈1，和 CPU 稳定版比相对误差
```

## 8. 资料映射

- PMPP：Reduction（softmax 的归约基础）。
- CUDA C++ Programming Guide：Warp Shuffle Functions。
- FlashAttention 论文（Dao et al.）：online softmax 与分块 attention。
- 配套：[卷四第 03 章 Reduction](../volume04_parallel_algorithms/03_Reduction从错误到优化.md)、[卷四第 06 章 数值正确性](../volume04_parallel_algorithms/06_数值正确性_复习与面试.md)。
