# G0～G5：A100 Tensor Core GEMM 完整学习实操

> 本文面向已经完成 FP32 CUDA Core GEMM 优化阶梯、准备进入 A100 Tensor Core 深水区的学习者。
> 学习方式不是“先读完再写”，而是每理解一个概念，立刻完成一个最小实验，并用 correctness、SASS 和资源证据验收。
> 本阶段只展开 G0～G5；Multi-Warp Block Tile、`cp.async`、swizzle、tail 与正式大矩阵结果留到后续决定。

相关资料：

- `cuda_study/docs/courses/cuda/Week3_TensorCore学习文档.md`：WMMA 与混合精度基础手册；
- `gpu-kernel-engineering/projects/gemm/README.md`：已冻结的 FP32 CUDA Core GEMM 作品；
- `gpu-kernel-engineering/projects/gemm/TENSOR_CORE_ROADMAP.md`：作品集当前执行进度与验收门槛；
- NVIDIA PTX ISA：`mma.sync` 与 `ldmatrix` 的权威寄存器映射。

---

## 0. 一句话目标

G0～G5 要完成的不是一个 production GEMM，而是一条可解释的学习链：

```text
G0  定义 FP16 输入、FP32 累加的数值语义
 ↓
G1  用 WMMA 理解一个 Warp 共同持有 fragment
 ↓
G2  隔离一条 mma.sync，理解 lane/register/矩阵映射
 ↓
G3  隔离 ldmatrix，理解 Shared Memory 到 fragment 的分发
 ↓
G4  组合成一个 Warp 的 16×8×K 最小手写 GEMM
 ↓
G5  一个 Warp 持有多组 MMA accumulator，扩展为 Warp Tile
```

一句话记忆：

> G1 先看到 Tensor Core 会算，G2 搞清楚寄存器怎样算，G3 搞清楚数据怎样进去，G4 把两者接起来，G5 再扩大一条 Warp 的工作量。

## 1. 学习所有权与工程边界

### 学习者亲手完成

- WMMA、`mma.sync`、`ldmatrix` 的核心 CUDA/PTX 代码；
- lane 到寄存器、fragment 与矩阵坐标的映射；
- Shared Memory 布局；
- accumulator 写回；
- Warp Tile 的 MMA 排布；
- 根据错误形态提出下一步假设。

### Assistant 可以完成

- 独立项目脚手架；
- FP16 输入生成和 CPU FP32 累加 reference；
- cuBLAS reference；
- runner、测试、sanitizer、SASS 与 ncu 脚本；
- review、失败复现和分级提示；
- 结果表与阶段记录。

### 明确不做

- 不复制 CUTLASS Kernel；
- 不把完整答案式 MMA/`ldmatrix` Kernel 直接填入学习者文件；
- 不用 WMMA 的成功掩盖对底层寄存器映射的不理解；
- 不把 FP16 Tensor Core 与 pedantic FP32 放进同一性能排名；
- G0～G5 不提前加入 Multi-Warp、`cp.async`、swizzle 和复杂 tail。

---

# 第一部分：统一数值与验证合同

## 2. 为什么必须先定义数值语义

当前 FP32 GEMM 的语义是：

$$
C_{ij}=\sum_k A^{fp32}_{ik}B^{fp32}_{kj}
$$

Tensor Core 第一条路线定义为：

```text
A/B 存储：FP16
乘法输入：FP16
累加器：FP32
输出：FP32
```

数学上应理解为：

$$
C_{ij}=\sum_k \operatorname{fp32}(A^{fp16}_{ik})
                 \operatorname{fp32}(B^{fp16}_{kj})
$$

关键区别：host 生成的 FP32 数在转成 FP16 时已经发生一次舍入。CPU reference 必须读取**实际 FP16 输入**，而不是原始 FP32 生成值。

## 3. 推荐 reference 流程

```text
Host 生成 FP32 随机数
→ 显式转成 FP16，成为真实输入
→ CPU 读取 FP16 并转回 FP32
→ CPU 用 FP32 累加
→ 与 GPU FP32 output 比较
```

大矩阵再增加：

```text
cuBLAS FP16 input + FP32 compute/output
```

CPU reference 负责小 Shape 定位错误，cuBLAS 负责大 Shape 验证与性能基线。

## 4. 为什么不能复用 FP32 容差结论

误差来源至少有两层：

1. FP32 → FP16 输入量化；
2. CPU、WMMA、MMA、cuBLAS 的累加顺序不同。

因此不能看到误差后直接“放宽到能过”。先测试：

| 输入 | 目的 |
| --- | --- |
| 全 0 | 初始化与无效 fragment |
| 全 1 | K 累加次数 |
| Identity | 布局与矩阵方向 |
| 小整数 | 手算与精确定位 |
| 固定随机 | 回归测试 |
| 正负抵消 | 累加顺序与数值稳定性 |

## 5. G0～G5 的统一设备布局

为减少变量，先固定：

```text
A logical shape：M×K
A physical layout：row-major
B logical shape：K×N
C/D logical shape：M×N
Output：row-major FP32
```

G1 WMMA 可以使用 API 支持的布局。

G2 单条 MMA 的 A/B 直接由寄存器准备，不讨论 global layout。

G3 只 dump `ldmatrix` 寄存器，不做 GEMM。

G4/G5 为隔离 `.row.col` MMA，第一版允许 runner 准备 B 的 col-major 实验存储。该预排布成本不进入 Kernel benchmark，也不能与普通 row-major cuBLAS 直接比较。后续若进入 G6，再设计 Shared Memory 中的 B 转置/布局。

---

# G0：数值语义、MMA Shape 与 Warp collective

## 6. G0 当日目标

不开新 Kernel，先闭卷建立以下模型：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

它描述：

| 矩阵 | Shape | 类型 |
| --- | --- | --- |
| A | 16×16 | FP16 |
| B | 16×8 | FP16 |
| C | 16×8 | FP32 |
| D | 16×8 | FP32 |

计算：

$$
D_{16\times8}=A_{16\times16}B_{16\times8}+C_{16\times8}
$$

## 7. FLOP 账本

一次 MMA 包含：

$$
16\times8\times16=2048
$$

次标量 FMA。若一乘一加计两个 FLOP：

$$
4096\ FLOP
$$

这 4096 FLOP 由整个 Warp 共同发起，不是每 Lane 各做一份完整矩阵乘。

## 8. Fragment 寄存器账本

对典型 `m16n8k16` FP16/FP32 路线：

| Fragment | 总逻辑元素 | 每 Lane 常见持有量 |
| --- | ---: | ---: |
| A | 256 FP16 | 8 FP16，通常打包进 4 个 32-bit reg |
| B | 128 FP16 | 4 FP16，通常打包进 2 个 32-bit reg |
| C/D | 128 FP32 | 4 FP32 reg |

这里的“每 Lane 持有四个输出”不代表该 Lane 独自算出四个输出。硬件 Tensor Core 使用全 Warp 提供的 fragment 完成矩阵运算。

## 9. Warp collective 安全规则

- 32 Lane 必须一致执行 MMA；
- 不能只有部分 Lane 进入 MMA 分支；
- 所有 Lane 的 shape、layout、type 必须一致；
- fragment 寄存器顺序必须匹配 PTX ISA；
- 不根据“看起来连续”猜 lane mapping；
- `__syncwarp()` 不能代替一致控制流。

## 10. G0 纸面实验

### 实验 A：全 1

```text
A 全 1，B 全 1，C 全 0
```

每个输出：

$$
D_{ij}=16
$$

### 实验 B：Identity

```text
A=I16
B 为可辨认编号矩阵
C=0
```

预期：

$$
D=B
$$

用于暴露：B 行列方向、寄存器顺序和写回位置。

### 实验 C：单行非零

A 只有第 r 行非零，则只有 D 的第 r 行非零。用于暴露 accumulator 行映射。

## 11. G0 当日交付

- 一张 `m16n8k16` Shape 图；
- A/B/C/D 类型与寄存器数量表；
- 全 1、Identity、单行非零的预期结果；
- 闭卷回答 4096 FLOP 从哪里来；
- 选定 G1～G5 固定数值合同。

## 12. G0 闭卷关卡

1. `m16n8k16` 中 M/N/K 分别对应哪两个矩阵维度？
2. 为什么 FP32 accumulator 不等于 pedantic FP32 GEMM？
3. 为什么 32 Lane 必须执行同一条 MMA？
4. 每 Lane 的 accumulator register 与整个输出矩阵是什么关系？
5. 为什么 G2 前不能只会调用 WMMA？

通过后再进入 G1。

---

# G1：WMMA 单 Warp、单 Tile

## 13. G1 为什么先用 WMMA

WMMA 隐藏 lane/register 映射，但能快速建立 fragment 生命周期：

```text
声明 fragment
→ 加载 A/B
→ 初始化 accumulator
→ mma_sync
→ 写回 output
```

G1 的目标不是性能，而是证明：一个 Warp 可以作为一个矩阵计算实体。

## 14. G1 Kernel 合同

```text
threads/block：32
grid：1 block
A：16×16 FP16
B：16×16 FP16
C：16×16 FP32
不处理 tail
不循环多个 tile
不做 Shared swizzle
```

WMMA 常用 tile 是 `16×16×16`，底层可能拆成多个机器 MMA。G1 不要求你立即解释底层拆分，但 SASS 要确认 Tensor Core 指令存在。

## 15. G1 实现顺序

### Step 1：只验证 fragment 声明与一致执行

所有 32 Lane 进入同一 Kernel 路径，不让 `lane_id==0` 独自调用 WMMA。

### Step 2：全 1

期望 256 个输出全部为 16。

### Step 3：Identity

A 为 Identity，B 为编号矩阵。检查输出位置与 B 一致。

### Step 4：固定随机

使用 FP16 输入 + CPU FP32 累加 reference。

### Step 5：SASS

验证目标函数中存在 `HMMA`。如果没有：

- 检查架构是否 `sm_80`；
- 检查输入 fragment 类型；
- 检查是否真的执行 `mma_sync`；
- 确认反汇编的是正确 binary/Kernel。

## 16. G1 Runner 最小输出

```text
experiment=wmma-single-tile
shape=16x16x16
input=ones|identity|random
status=PASS|FAIL
max_abs=...
max_rel=...
```

## 17. G1 常见错误

| 症状 | 优先检查 |
| --- | --- |
| 全 0 | accumulator/store 是否执行 |
| 全部 16 但位置乱 | output layout/leading dimension |
| Identity 转置 | B layout 或 store layout |
| 部分随机 | Warp 分支不一致、输入未初始化 |
| SASS 无 HMMA | 编译架构或路径错误 |

## 18. G1 当日完成门槛

- 全 1、Identity、随机 PASS；
- memcheck 0 errors；
- SASS 存在 `HMMA`；
- 能说明四类 fragment 的职责；
- 不把 WMMA 代码复制为最终手写 MMA 结论。

## 19. G1 三级提示

### 提示 1

检查每个 WMMA API 是否由整个 Warp 一致调用。

### 提示 2

若值正确但位置错误，只看 layout、leading dimension 与 store，不先怀疑 MMA 数学。

### 提示 3

用 Identity 输入区分 A/B 方向问题，用全 1 区分 K 累加次数问题。

---

# G2：隔离一条 `mma.sync.m16n8k16`

## 20. G2 的核心问题

WMMA 成功只证明高级 API 可用。G2 要回答：

> 一条 PTX `mma.sync` 的 A/B/C/D 操作数具体分布在哪些 Lane 的哪些寄存器？

## 21. G2 固定合同

```text
一个 Warp
只执行一条 mma.sync
A：16×16 FP16
B：16×8 FP16
C/D：16×8 FP32
A row，B col
不使用 ldmatrix
不循环 K
```

A/B packed register 先由 Lane 根据 PTX ISA 映射直接准备。这样 G2 不混入 Shared Memory 布局问题。

## 22. 为什么 G2 不先用 `ldmatrix`

如果结果错，同时存在：

- MMA register mapping；
- Shared address；
- `ldmatrix` 分发；
- B transpose；
- accumulator store mapping；

就无法定位。G2 只验证寄存器和 MMA，G3 再验证 `ldmatrix`。

## 23. G2 实验数据设计

### 全 1

最先验证 K=16 累加次数，所有输出为 16。

### Identity

用于验证 A/B 逻辑布局。

### 编号矩阵

建议 B 编码：

$$
B_{k,n}=100k+n
$$

Identity A 下，D 应保留 B 的结构，方便识别行列交换。

### 单行或单列非零

用于定位 accumulator 的 Lane/Register 写回位置。

## 24. Accumulator dump 策略

第一版不要直接“聪明地”写回矩阵。先把每个 Lane 的四个 accumulator 按：

```text
raw[lane][reg]
```

写到 Global Memory。Host 打印：

```text
lane 0: d0 d1 d2 d3
...
lane 31: d0 d1 d2 d3
```

然后单独写重建函数：

```text
raw lane/register → logical D[row,col]
```

这样能区分：

- MMA 算错；
- MMA 算对但写回映射错。

## 25. PTX Inline Assembly 合同

实现前逐字段核对官方 PTX ISA：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

需要确认：

- A register 参数数量与 packed type；
- B register 参数数量；
- C/D accumulator 参数数量；
- 输入/输出约束符；
- 最低目标架构；
- fragment table 的 lane mapping。

不要根据博客中的 Turing/Hopper 例子猜 A100 映射。

## 26. G2 测试顺序

```text
编译通过
→ 全 1 raw dump
→ 全 1 logical rebuild
→ Identity
→ 编号矩阵
→ 固定随机
→ memcheck
→ SASS HMMA
```

## 27. G2 常见错误形态

| 症状 | 优先检查 |
| --- | --- |
| 全部固定倍数错误 | packed FP16 顺序或重复 MMA |
| 8 列为周期错位 | B fragment mapping |
| 8 行为周期错位 | A fragment mapping |
| raw 值对、矩阵错 | accumulator rebuild |
| 只有部分 Lane 错 | operand register 顺序 |
| SASS 有 HMMA 但随机错 | 映射/packing，不是硬件路径 |

## 28. G2 当日完成门槛

- 单条 MMA 全 1、Identity、随机正确；
- 生成 raw Lane/Register dump；
- 生成逻辑 D 重建表；
- SASS 出现 `HMMA`；
- 能指出任一 D 元素对应哪个 Lane 的哪个 accumulator register。

## 29. G2 三级提示

### 提示 1

先问错误发生在 operand packing 还是 accumulator rebuild。

### 提示 2

全 1 对但 Identity 错，优先检查布局；raw dump 对但矩阵错，只检查写回映射。

### 提示 3

用官方 fragment table，分别为 A、B、D 写独立坐标转换，不把三个映射揉进一个公式。

---

# G3：隔离 `ldmatrix` 映射

## 30. G3 的核心问题

`mma.sync` 消费寄存器 fragment。真实 GEMM 需要从 Shared Memory 高效准备这些寄存器。

G3 要回答：

> `ldmatrix` 读取 Shared Memory 的哪些地址，并把每个 16-bit 元素送到哪些 Lane 的哪些 packed register？

## 31. G3 固定合同

```text
一个 Warp
Shared Memory 中写入可辨认坐标值
执行 ldmatrix
直接 dump 目标寄存器
不执行 mma.sync
```

依次做：

```text
m8n8.x1.shared.b16
m8n8.x2.shared.b16
m8n8.x4.shared.b16
m8n8.x2.trans.shared.b16
```

## 32. 输入编码

Shared 中的逻辑元素使用可辨认 FP16 值：

$$
value(row,col)=16row+col
$$

避免 `100row+col` 超出某些 FP16 精确整数范围时造成不必要干扰。第一版值控制在小整数范围。

## 33. 地址与对齐

第一版要求：

- 每个逻辑 8 元素 FP16 行起点至少 16B 对齐；
- Shared base 对齐明确；
- 传给 PTX 的是 Shared address space 地址；
- 需要时使用 CUDA 提供的 generic → shared 地址转换；
- 不在 G3 同时加入 padding/swizzle。

## 34. G3 dump 格式

```text
mode=x1|x2|x4|x2-trans
lane=0 reg=0 low=... high=...
...
```

每个 32-bit register 解包成两个 FP16，Host 显示它们对应的逻辑坐标。

## 35. G3 实现顺序

### Step 1：Shared 填充

由 32 Lane 合作写入坐标值，`__syncwarp()` 后再加载。

### Step 2：x1

先确认最小 8×8 分发。

### Step 3：x2/x4

观察目标寄存器数量和多个子矩阵顺序。

### Step 4：trans

对比普通与转置分发，不通过修改 Host expected 来“迁就”错误。

### Step 5：SASS

确认目标函数出现 `LDSM`。

## 36. G3 常见错误

| 症状 | 优先检查 |
| --- | --- |
| 所有 Lane 为 0 | Shared 地址空间转换 |
| 前半正确后半错 | x2/x4 子矩阵地址 |
| 成对值交换 | packed low/high 顺序 |
| 行周期错位 | Shared leading dimension |
| trans 结果像普通加载 | PTX variant 或 destination mapping |
| memcheck 通过但映射错 | 语义错误，靠坐标 expected 定位 |

## 37. G3 当日完成门槛

- x1/x2/x4/trans dump 全部与 expected mapping 一致；
- memcheck、racecheck、synccheck 通过；
- SASS 出现 `LDSM`；
- 能解释哪些 Lane 提供地址、每个寄存器装哪两个 FP16；
- 形成一张 `mode × lane × reg × logical coordinate` 表。

## 38. G3 三级提示

### 提示 1

先确认 Shared 地址是否合法，再确认分发顺序。

### 提示 2

若每行第一个元素正确但后续错，看 leading dimension 和子矩阵起点。

### 提示 3

把 `ldmatrix` 结果先视为 raw packed registers，单独解包，不直接接 MMA。

---

# G4：最小手写 MMA GEMM

## 39. G4 目标

把 G2 的 `mma.sync` 和 G3 的 `ldmatrix` 组合成最小 GEMM：

```text
Global A/B
→ Shared A/B
→ ldmatrix
→ mma.sync
→ accumulator registers
→ Global C
```

## 40. G4 Kernel 合同

```text
一个 Warp/Block
输出 Tile：16×8
K：16 的整数倍
A：row-major FP16
B 实验存储：col-major FP16
C：row-major FP32
普通 Global→Shared cooperative load
不做 cp.async
不做 swizzle
不处理 M/N tail
```

第一批问题：

```text
M=16,N=8,K=16
M=16,N=8,K=32
M=16,N=8,K=48
```

## 41. 为什么 B 先用实验 col-major 存储

PTX variant 为 `.row.col`。G4 的主要变量是组合 `ldmatrix + mma.sync`，不是研究 row-major B 的在线转置。

因此 runner 可以显式准备 col-major B 实验缓冲：

```text
B_col[n*K + k] = B_logical[k*N + n]
```

记录中必须写清“prepacked B，不计预排布时间”。G4 结果不与普通 row-major cuBLAS 做性能结论。

## 42. K 循环不变量

每轮 K 前进 16：

```text
for k0 = 0; k0 < K; k0 += 16
```

每轮：

1. cooperative load A/B tile；
2. Warp 同步；
3. `ldmatrix` 得到 A/B fragment；
4. 一条 `mma.sync` 更新同一组 accumulator；
5. 下一轮覆盖 Shared 前确认消费完成。

Accumulator 只在 K 循环前清零一次。

## 43. G4 手算

### K=16

全 1输出为 16。

### K=32

两轮全 1输出为 32。

### K=48

三轮全 1输出为 48。

这三个用例能精准暴露：

- accumulator 每轮被错误清零；
- K step 少算或多算；
- Shared tile 复用时序错误。

## 44. G4 correctness matrix

| Shape | 输入 | 目的 |
| --- | --- | --- |
| 16×8×16 | ones | 单 Tile |
| 16×8×32 | ones | 两轮累加 |
| 16×8×48 | ones | 三轮累加 |
| 16×8×16 | identity | 布局 |
| 16×8×32 | random | 完整 reference |
| 非法 Shape | 任意 | launcher 明确拒绝 |

## 45. G4 同步模型

一个 Warp/Block 时可使用 Warp 同步，但 Shared producer/consumer 的参与 Lane 必须一致。不要因为只有一个 Warp 就省略所有同步，也不要无意义加入 CTA barrier。

需要逐段说明：

```text
谁写 Shared？
谁读 Shared？
覆盖下一 Tile 前，谁保证上一 Tile 已消费？
```

## 46. G4 SASS 合同

目标函数应出现：

- `LDSM`；
- `HMMA`；
- 对应 Global/Shared load/store；
- 无意外 `LDL/STL`，或明确记录 spill。

## 47. G4 当日完成门槛

- K=16/32/48 全部 PASS；
- Identity 与随机输入 PASS；
- 非法 Shape 被 launcher 拒绝；
- 四类 sanitizer 通过；
- SASS 同时出现 `LDSM/HMMA`；
- 能解释每轮 K tile 的地址、fragment 与 accumulator 生命周期。

## 48. G4 三级提示

### 提示 1

K=16 对、K=32 错，先看 accumulator 是否被重置和 Shared 是否提前覆盖。

### 提示 2

全 1对、Identity 错，检查 B col-major 与 fragment mapping。

### 提示 3

将一轮 K 拆成 Global→Shared、Shared→Reg、MMA、Acc 四个边界，分别 dump 输入输出。

---

# G5：扩展为 Warp Tile

## 49. G5 为什么做

G4 一条 Warp 只持有 16×8 输出，数据复用与 ILP 很低。G5 让一个 Warp 持有多组 accumulator，提高 A/B fragment 复用和独立 MMA 数量。

G5 仍然只有一个 Warp，不进入 Multi-Warp Block Tile。

## 50. 首个候选 Warp Tile

建议第一候选：

```text
Warp Tile：32×32
Instruction Tile：16×8
K Step：16
```

一个 K step 的 MMA 数：

$$
\frac{32}{16}\times\frac{32}{8}=2\times4=8
$$

即 8 组 accumulator fragment。

## 51. 二维 accumulator 索引

定义：

```text
mma_m = 0..1
mma_n = 0..3
acc[mma_m][mma_n][4 registers per lane]
```

每组对应输出子块：

```text
row range = mma_m*16 .. mma_m*16+15
col range = mma_n*8  .. mma_n*8+7
```

写代码前先画 2×4 子块图。

## 52. Fragment 复用

在同一 K step：

- A fragment `A[mma_m]` 可服务四个不同 `mma_n`；
- B fragment `B[mma_n]` 可服务两个不同 `mma_m`。

因此理想循环：

```text
load 2 组 A fragment
load 4 组 B fragment
for mma_m:
    for mma_n:
        mma(A[mma_m], B[mma_n], acc[mma_m][mma_n])
```

不要为每条 MMA 重复加载同一 fragment。

## 53. G5 资源账本

仅 accumulator：

```text
8 groups × 4 FP32 regs/lane = 32 accumulator regs/lane
```

还要加：

- A/B fragment regs；
- 指针与循环变量；
- 地址计算临时量。

所以 G5 必须记录 ptxas registers/thread，并检查：

- 是否出现 `LDL/STL`；
- occupancy 上限；
- 更大 Warp Tile 的 ILP 是否值得寄存器代价。

## 54. G5 实现顺序

### Step 1：只扩 N 方向

先做 `16×32`：1×4 MMA groups。确认 B fragment 与 accumulator 列偏移。

### Step 2：再扩 M 方向

做 `32×32`：2×4 groups。确认 A fragment 与行偏移。

### Step 3：K=16

先验证单 K step 的八组输出位置。

### Step 4：K=32/48

验证所有 accumulator 跨 K step 保留。

### Step 5：资源与 SASS

记录 registers、spill、静态 HMMA/LDSM 和正常墙钟。

## 55. G5 correctness matrix

```text
16×32×16 ones/identity/random
32×32×16 ones/identity/random
32×32×32 ones/random
32×32×48 ones/random
```

输出错误按 16×8 子块观察，而不是只看最大误差：

- 某一列组错：B fragment 或 `mma_n`；
- 某一行组错：A fragment 或 `mma_m`；
- 子块正确但位置交换：store mapping；
- K=16 对、K>16 错：accumulator 生命周期。

## 56. G5 Benchmark 的正确定位

G5 可以比较候选 Warp Tile，但不能宣称 production 性能：

```text
G4 16×8 Warp Tile
G5 16×32 Warp Tile
G5 32×32 Warp Tile
```

记录：

- latency；
- effective TFLOPS；
- registers/thread；
- occupancy；
- `HMMA/LDSM`；
- spill；
- 每个 Warp Tile 的 MMA 数。

由于仍是单 Warp/Block，Grid 映射和大矩阵覆盖尚未进入 G6，结果只用于选择 G6 的候选 Warp Tile。

## 57. G5 当日完成门槛

- 16×32 与 32×32 正确；
- K=16/32/48 正确；
- 八组 accumulator 地址与写回可闭卷说明；
- 无未解释的 spill；
- 至少比较两个 Warp Tile 的资源与墙钟；
- 选出一个进入 G6 的候选，但不开始 G6。

## 58. G5 三级提示

### 提示 1

先画 2×4 MMA 子块，不在一维线性索引里猜位置。

### 提示 2

某一列子块全部错只查 B/mma_n；某一行子块全部错只查 A/mma_m。

### 提示 3

将 accumulator 数量换算为每 Lane 寄存器下界，再对照 ptxas 报告和 SASS `LDL/STL`。

---

# 第八部分：G0～G5 的工程组织

## 59. 推荐独立项目

作品集创建相邻项目：

```text
projects/gemm_tensorcore/
├── CMakeLists.txt
├── README.md
├── include/gemm_tensorcore/
├── kernels/
│   ├── wmma_single.cu
│   ├── mma_single.cu
│   ├── ldmatrix_dump.cu
│   ├── mma_gemm_warp.cu
│   └── mma_warp_tile.cu
├── runner/
├── tests/
├── scripts/
└── results/
```

G0 只有文档/手算，不创建空 Kernel。

G1～G5 每阶段新增独立文件，不覆盖上一阶段，使错误和证据可回溯。

## 60. Runner 设计

建议独立 `gemm_tensor_runner`，支持：

```text
--experiment wmma-single
--experiment mma-single
--experiment ldmatrix-dump
--experiment mma-warp-gemm
--experiment mma-warp-tile
--m --n --k
--input ones|identity|random|coordinate
--mode validate|benchmark
```

不要复用 FP32 `gemm_runner` 的 `float*` Kernel registry 强塞 `half*`。

## 61. 每阶段固定验证顺序

```text
最小手算输入
→ CPU FP16-quantized reference
→ 边界/非法 Shape
→ memcheck
→ racecheck
→ synccheck
→ initcheck
→ CUDA Event smoke
→ ncu 资源与瓶颈
→ SASS 指令证据
→ 当日 worklog
```

G1～G3 的 benchmark 只确认机制，不发布性能结论。

G4/G5 可以比较内部实验，但正式 cuBLAS/大矩阵结果留待后续。

## 62. 每日记录模板

```markdown
## Gx：标题

### 今天理解了什么

### 手算与数据所有权

### 自己实现了什么

### Correctness / Sanitizer

### SASS / 资源

### 失败与修复

### 一句话记忆

### 下一步唯一变量
```

---

# 第九部分：总自检与延期边界

## 63. G0～G5 总自检

### 数值

- [ ] reference 使用实际 FP16 输入值；
- [ ] FP32 accumulator 与 pedantic FP32 的区别说清；
- [ ] 容差有输入和 K 长度依据。

### Warp/MMA

- [ ] 能解释 `m16n8k16`；
- [ ] 能指出 A/B/D fragment 的 Lane/Register 映射；
- [ ] MMA 由全 Warp 一致执行；
- [ ] raw accumulator dump 可重建逻辑矩阵。

### ldmatrix

- [ ] x1/x2/x4/trans 映射有独立 dump；
- [ ] Shared 地址、对齐和 leading dimension 可说明；
- [ ] SASS 出现 `LDSM`。

### 最小 GEMM

- [ ] K=16/32/48 正确；
- [ ] accumulator 不在 K 循环内清零；
- [ ] SASS 同时出现 `LDSM/HMMA`；
- [ ] sanitizer 通过。

### Warp Tile

- [ ] 16×32 和 32×32 正确；
- [ ] 2×4 MMA 子块映射清楚；
- [ ] registers/thread 和 spill 有证据；
- [ ] 选出 G6 候选 Warp Tile。

## 64. G6～G8 暂时冻结

G0～G5 完成后再决定是否继续：

```text
G6 Multi-Warp Block Tile
G7 cp.async Multi-stage
G8 Swizzle/Tail/cuBLAS/正式发布
```

进入 G6 的条件：

- G2/G3 mapping 能闭卷说明；
- G4 最小 GEMM 正确；
- G5 Warp Tile 正确且资源可接受；
- 没有靠复制神秘常量绕过理解。

若目标是尽快回到 Attention，可在 G5 后进入：

```text
Attention M4 MMA Playground
→ M5 Tensor Core QK
→ M6 Tensor Core PV
```

## 65. 最终口述

完成 G0～G5 后，应能在三分钟内说明：

> 我的 Tensor Core 路线使用 FP16 输入和 FP32 accumulator，reference 基于量化后的 FP16 值。先用 WMMA 建立 Warp fragment 直觉，再隔离 `mma.sync.m16n8k16` 验证 lane/register mapping；随后独立 dump `ldmatrix` 的 x1/x2/x4/trans 分发。将两者组合成一个 Warp 的 `16×8×K` GEMM，验证 K=16/32/48 的 accumulator 生命周期。最后扩展为 `16×32` 和 `32×32` Warp Tile，通过多组 MMA 复用 A/B fragment，并用 registers、spill、occupancy、HMMA/LDSM 和墙钟证据选择下一阶段候选。

一句话记忆：

> 每天只增加一个变量：先证明算得对，再证明数据送得对，最后才扩大一个 Warp 的工作量。
