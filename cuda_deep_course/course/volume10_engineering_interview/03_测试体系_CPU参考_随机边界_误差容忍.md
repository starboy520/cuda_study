# 03 测试体系：CPU reference、随机与边界、误差容忍

## 0. 先建立大局观：GPU 代码为什么特别需要测试

CPU 代码错了通常会崩或给出明显错误。GPU 代码不同——它**经常"错得很安静"**：

```text
- race / 漏 __syncthreads()：结果时对时错，单次运行可能"碰巧对"
- 越界写：可能没崩，只是污染了别的数据
- 边界处理漏判：大部分元素对，只有最后几个错
- 浮点求和顺序变了：和 CPU 差在末位，看着像 bug 其实正常
```

所以 GPU 开发的铁律是（贯穿全教材）：**先正确，再性能**。而"正确"不能靠肉眼看几个数，必须
有**自动化测试体系**。本章给出这套体系的四根支柱：

```text
① CPU reference   —— 一个"绝对可信"的对照答案
② 随机 + 边界输入 —— 覆盖正常和刁钻情况
③ 误差容忍        —— 浮点不能用 == 比，要按容差
④ Sanitizer       —— 工具抓 race / 越界（卷四/01、卷五/05）
```

## 0.1 术语速查表

| 术语 | 一句话定义 |
|---|---|
| **CPU reference** | 同一算法的简单 CPU 实现，作为正确性基准 |
| **边界输入** | 刁钻规模：n=0/1、非整除、非方阵、极大素数 |
| **绝对容差** | `abs(a-b)` 能接受的上限 |
| **相对容差** | `abs(a-b)/abs(b)` 能接受的上限 |
| **混合容差** | `abs(a-b) <= absTol + relTol*abs(b)`，兼顾大小数 |
| **确定性** | 同样输入每次跑结果完全一致（含末位）|

## 1. 第一根支柱：CPU reference

测 GPU 结果对不对，得有个"标准答案"。最可靠的标准答案是**同一算法的简单 CPU 实现**——它
慢没关系，只要**显然正确**：

```cpp
// GPU 要测的是并行 reduction；CPU reference 就是最朴素的串行求和
double cpu_reduce(const float* x, int n) {
    double sum = 0.0;                 // 注意用 double 累加，见第 4 节
    for (int i = 0; i < n; ++i) sum += x[i];
    return sum;
}
```

测试流程固定为：

```text
1. 生成输入
2. CPU reference 算出 expected
3. GPU kernel 算出 actual
4. 按容差比较 expected 和 actual（第 3 节）
5. 不一致 -> 报告哪个位置、差多少
```

> 为什么 reference 要"简单到显然正确"？因为如果 reference 本身也很复杂，你就是在"用一个可能
> 有 bug 的东西验证另一个"。reference 越朴素越可信。

## 2. 第二根支柱：随机 + 边界输入

只用 `[1,2,3,4]` 这种规整输入测，会漏掉一大半 bug。要两类输入都覆盖：

**随机输入**——发现一般性错误：

```cpp
std::mt19937 rng(42);                 // 固定种子！保证可复现（第 5 节）
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
for (int i = 0; i < n; ++i) data[i] = dist(rng);
```

**边界输入**——发现"刁钻规模"的 bug，这是 GPU 代码最容易翻车的地方：

```text
n = 0        空输入，别崩
n = 1        只有一个元素，几乎所有线程都越界
n = 31       小于一个 warp
n = 32       正好一个 warp
n = 33       跨一个 warp 一点点
n = 257      跨一个 block 边界一点点
n = 1000003  大素数，确保所有 ceil 除法都不整除（暴露边界判断漏洞）
非方阵       如 1003 x 769，暴露 width/height 写反的 bug
```

> 经验：**bug 几乎都藏在边界**。一个在 `n=1024`（整除）上 PASS 的 kernel，很可能在 `n=1000`
> （非整除）上越界或算错——因为最后那个不满的 block/tile 没处理好。所以非整除规模是必测项。

## 3. 第三根支柱：误差容忍（浮点不能用 ==）

这是 GPU 数值测试最容易踩的坑（卷四/06 讲过原理）。**浮点求和不满足结合律**，GPU 并行归约
改变了求和顺序，结果和 CPU 串行**末位必然有微小差异**。用 `==` 比一定失败：

```cpp
if (gpu_result == cpu_result) ...     // ❌ 几乎永远 false，即使算法完全正确
```

正确做法是**混合容差**——同时给绝对和相对容差，兼顾大数和接近 0 的小数：

```cpp
bool close(double a, double b, double absTol = 1e-5, double relTol = 1e-4) {
    return std::abs(a - b) <= absTol + relTol * std::abs(b);
}
```

为什么要"绝对 + 相对"两个一起：

```text
只用绝对容差：    对很大的数太严（大数末位差也超绝对容差）
只用相对容差：    对接近 0 的数除法不稳定
混合：absTol 兜住接近 0 的情况，relTol 兜住大数 -> 两头都稳
```

容差该取多大，取决于（卷四/06）：数据规模、数值范围、运算次数、算法稳定性、数据类型。运算
越多、规模越大，累积误差越大，容差要相应放宽。

## 4. 提升 reference 精度：用 double 累加

一个微妙但重要的点：**CPU reference 本身也有浮点误差**。如果 reference 也用 `float` 累加大
数组，它自己就不准了，拿它当基准不可靠。所以 reference 用 **double 累加**：

```cpp
double sum = 0.0;                     // double，不是 float
for (int i = 0; i < n; ++i) sum += x[i];   // 累加误差远小于 float
```

这样 reference 足够准，再让 `float` 的 GPU 结果按容差去逼近它。这正是卷四 reduction lab
里 `expected` 用 double 的原因。

## 5. 第四根支柱与可复现：固定种子 + Sanitizer

**固定随机种子**：测试必须可复现——同一个 bug 每次都能用同样输入重现，否则没法调试：

```cpp
std::mt19937 rng(42);                 // 固定种子，每次跑输入完全一样
```

**配合 Compute Sanitizer**（卷四/01、卷五/05）抓肉眼看不到的错：

```bash
compute-sanitizer --tool memcheck  ./test    # 越界、非法地址
compute-sanitizer --tool racecheck ./test    # shared memory 竞争
compute-sanitizer --tool initcheck ./test    # 读未初始化的 device 内存
```

> 重要（卷四/01 强调过）：**sanitizer 没报错 ≠ 一定没 bug**。它只能发现"这次实际发生的"问题，
> 没覆盖的路径看不到。所以工具 + reference 对照 + 多规模测试要一起用。

## 6. 把它拼成一个测试函数

四根支柱合起来，一个完整的 kernel 测试长这样：

```cpp
bool test_reduce(int n) {
    std::vector<float> h(n);
    std::mt19937 rng(42);                                  // ① 固定种子
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : h) x = dist(rng);

    double expected = cpu_reduce(h.data(), n);             // ② double reference
    float actual = gpu_reduce(h.data(), n);                // 被测 GPU 实现

    bool ok = close(actual, expected);                     // ③ 容差比较
    if (!ok) {
        fprintf(stderr, "FAIL n=%d: gpu=%.6f cpu=%.6f diff=%.2e\n",
                n, actual, expected, std::abs(actual - expected));
    }
    return ok;
}

int main() {
    int sizes[] = {0, 1, 31, 32, 33, 257, 1024, 1000003}; // ④ 含边界规模
    bool all = true;
    for (int n : sizes) all &= test_reduce(n);
    printf(all ? "ALL PASS\n" : "SOME FAILED\n");
    return all ? 0 : 1;                                    // 返回码供 ctest 判断
}
```

注意 `return` 退出码——这让 CMake 的 `ctest`（第 01 章）能自动判定通过/失败。

## 7. 实践

1. 给你的 reduction 写完整测试：double reference + 8 种规模 + 混合容差，跑 `ctest`。
2. 故意把 kernel 的边界判断删掉，确认测试在**非整除规模**上 FAIL（验证边界测试有效）。
3. 把容差比较换成 `==`，观察即使算法正确也大量 FAIL，体会为什么浮点要用容差。
4. 用 `compute-sanitizer --tool racecheck` 跑一个故意漏 `__syncthreads()` 的版本，看它抓出竞争。

## 8. 面试题（附参考答案）

**Q1：为什么 GPU 代码特别需要自动化测试？**
GPU bug 常"安静"——race 时对时错、越界不崩、边界漏判只错几个元素、浮点末位差异像 bug 其实
正常。肉眼看几个数无法发现，必须靠 reference + 多规模 + 容差 + sanitizer 的体系。

**Q2：为什么浮点结果不能用 `==` 比较？怎么比？**
浮点加法不满足结合律，GPU 并行归约改变求和顺序，和 CPU 末位必有差异，`==` 几乎永远失败。要用
混合容差 `abs(a-b) <= absTol + relTol*abs(b)`。

**Q3：为什么 CPU reference 要用 double 累加？**
reference 是基准，必须比被测对象更准。float 累加大数组自身误差大，拿它当基准不可靠；double
累加误差小得多，才能让 float 的 GPU 结果去逼近。

**Q4：测试为什么要专门测非整除/非方阵规模？**
bug 几乎都藏在边界。整除规模下每个 block/tile 都满，掩盖了边界处理；非整除会暴露"最后一个
不满的 block 没处理好"导致的越界或算错。

**Q5：sanitizer 没报错能说明代码一定对吗？**
不能。它只发现这次运行实际发生的问题，没覆盖的执行路径/调度看不到。要和 reference 对照、多
规模测试一起用。

## 9. 资料映射

- CUDA C++ Best Practices Guide：Numerical Accuracy、Debugging。
- Compute Sanitizer Documentation。
- 配套：[卷四第 06 章 数值正确性](../volume04_parallel_algorithms/06_数值正确性_复习与面试.md)、[卷四第 03 章 Reduction](../volume04_parallel_algorithms/03_Reduction从错误到优化.md)。
