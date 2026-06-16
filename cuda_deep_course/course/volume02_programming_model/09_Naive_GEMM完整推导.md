# 09 Naive GEMM 完整推导

## 1. 数学规格

```text
A: M 行 x K 列
B: K 行 x N 列
C: M 行 x N 列
```

元素：

```text
C[row][col] =
  sum(A[row][inner] * B[inner][col]),
  inner = 0..K-1
```

## 2. Thread 映射

一个 thread 负责一个 `C[row][col]`：

```cpp
const int col =
    blockIdx.x * blockDim.x + threadIdx.x;
const int row =
    blockIdx.y * blockDim.y + threadIdx.y;
```

## 3. 行主序地址

```text
A[row][inner] -> A[row * K + inner]
B[inner][col] -> B[inner * N + col]
C[row][col]   -> C[row * N + col]
```

注意每个矩阵乘自己的逻辑宽度。

## 4. Kernel

```cpp
if (row < M && col < N) {
  float sum = 0.0F;
  for (int inner = 0; inner < K; ++inner) {
    sum += A[row * K + inner]
         * B[inner * N + col];
  }
  C[row * N + col] = sum;
}
```

## 5. 具体元素

```text
M=2, K=3, N=2

A = [1 2 3
     4 5 6]

B = [7  8
     9 10
    11 12]
```

`C[1][0]`：

```text
4*7 + 5*9 + 6*11
= 28 + 45 + 66
= 139
```

负责它的 thread 只写这一个输出，但循环读取 A 的一行和 B 的一列。

## 6. 为什么 Naive 慢

相邻输出 thread：

- 对同一 `inner` 读取相同 `A[row][inner]`。
- 读取连续的 `B[inner][col]`。

同一个 A 值被一行中的多个 thread 重复读取；B 值也会被其他输出行重复读取。
Naive 依赖 cache，未显式构造 tile 复用。

把"重复读取"量化成数字，慢的原因就具体了。考虑计算一个 `C[row][col]`：

```text
循环 K 次，每次读 1 个 A 元素 + 1 个 B 元素 = 2K 次 global 读
完成 2K FLOP（K 次乘 + K 次加）
```

于是**每个输出**的算术强度（卷五第 02 章）大约是：

```text
AI = 2K FLOP / (2K * 4 bytes) = 0.25 FLOP/byte
```

这个值非常低——意味着 naive GEMM 几乎是**纯访存受限**：算术单元大部分时间在
等数据。更糟的是这些读取**高度重复**：整个 `C` 的一行（N 个输出）都要把 `A` 的
同一行读 N 遍；整个 `C` 的一列（M 个输出）都要把 `B` 的同一列读 M 遍。理论上
`A` 的每个元素只需读一次就够算完它参与的所有输出，naive 却读了约 N 次。

```text
naive 的 global 读总量 ≈ 2 * M * N * K 个元素
理想（每元素读一次）   ≈ M*K + K*N 个元素
浪费倍数 ≈ N 或 M 量级
```

Shared-memory tiling（卷三）的全部动机就是把这块重复读取**搬进片上 shared
memory 复用一次**，从而把 AI 抬高、让 GEMM 从访存受限往计算受限移动。理解这个
0.25 的数字，就理解了后面所有 GEMM 优化为什么存在。

## 7. 计算量

每个输出约：

```text
K 次乘法 + K 次加法
约 2K FLOP
```

全矩阵：

```text
约 2*M*N*K FLOP
```

GFLOPS：

```text
2*M*N*K / seconds / 1e9
```

## 8. Sample

```bash
make -C labs/02_programming_model/gemm_naive clean all
./labs/02_programming_model/gemm_naive/gemm_naive
./labs/02_programming_model/gemm_naive/gemm_naive 64 48 33
```

默认尺寸故意不是 16 的倍数。

## 9. CPU Reference

CPU 使用 `double sum`，GPU 使用 `float sum`，比较容差而不是严格相等。

不同累加顺序和 FMA 可能造成末位差异。

## 10. 后续优化路线

```text
Naive
-> Shared-memory tiling
-> Register tiling
-> Vectorized load
-> Double buffering
-> Tensor Core/WMMA
-> cuBLAS/CUTLASS
```

本章目标是建立正确 baseline，不是追上 cuBLAS。

## 11. 故障实验

1. 将 A 地址错误写成 `row * M + inner`。
2. 将 B 地址错误写成 `inner * K + col`。
3. 删除 row/col 边界。
4. 只测试方阵，观察为何会掩盖宽度错误。

### C++ 无符号下溢案例

如果循环变量是 `size_t`，下面代码有陷阱：

```cpp
float value = static_cast<float>((i % 11) - 5);
```

`i % 11` 仍是无符号数，小于 5 时减法不会得到负数，而会下溢成巨大正数。
正确做法：

```cpp
int value = static_cast<int>(i % 11) - 5;
```

CUDA 调试不能只盯着 kernel，Host 侧输入生成同样可能有 bug。

## 12. 面试题

- GEMM 的 M/N/K 分别是什么？
- 为什么 A/B/C 的行主序乘数不同？
- Naive GEMM 的主要数据复用缺口是什么？
- GFLOPS 如何计算？
