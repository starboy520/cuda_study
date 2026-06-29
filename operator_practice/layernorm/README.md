# LayerNorm（层归一化）

> 第二个练习算子。承接 reduce —— 核心就是**沿每行做两次归约**（求均值、求方差）。

## 数学定义

对输入 `x` 的每一行（长度 `D`）：

```text
μ   = (1/D) * Σ x[i]                 # 均值
σ²  = (1/D) * Σ (x[i] - μ)²          # 方差
y[i] = (x[i] - μ) / sqrt(σ² + eps) * gamma[i] + beta[i]
```

- `gamma[D]`、`beta[D]`：可学习的缩放/平移参数（每列一个）
- `eps`：防止除 0 的小常数（如 1e-5）

## 输入输出 shape

```text
x      : [rows, D]   输入
gamma  : [D]         缩放参数（所有行共享）
beta   : [D]         平移参数（所有行共享）
y      : [rows, D]   输出
```

## 实现方案（第一版：两遍归约）

```text
布局：一个 block 处理一行
1. block 内线程 grid-stride 读这一行 → 块内 reduce 求 Σx → μ
2. 再来一遍块内 reduce 求 Σ(x-μ)² → σ²
3. 每个线程把负责的元素套公式写回 y
```

归约部分直接复用 reduce 的「warp shuffle + 两级归约」套路。

> 暂时不用 Welford / 一遍法，先把两遍法跑通过校验。

## 完成标准

```text
[ ] CPU reference 校验通过（逐元素相对误差 < 1e-4，注意 float）
[ ] 支持 rows=4, D=1024 等规模
[ ] gamma 全 1、beta 全 0 时，输出每行均值≈0、方差≈1
[ ] benchmark：记录 time_ms
[ ] 一段口述：LayerNorm 为什么是两次 reduce
```

## 进阶（之后再做，别现在做）

```text
- 一遍法：同时归约 Σx 和 Σx²，用 σ² = E[x²] - μ²（省一遍读，数值略差）
- Welford：单遍数值稳定算法
- 向量化 load（float4）
- RMSNorm：连均值都不减，只用均方根
```
