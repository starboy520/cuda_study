# Naive GEMM PTX 逐段对照注释

> 对照源码：`/home/qichengjie/workspace/gpu-kernel-engineering/projects/gemm/kernels/naive.cu`
> 对照产物：`/tmp/cuda_focus/week02/day01/naive.compute80.ptx`
> 工具链：CUDA 13.3，`compute_80` PTX
> 用法：先看每节的“源码目标”，再读 PTX，最后遮住“人话解释”复述。

## 1. 先抓住唯一的数学主线

Naive GEMM 的核心只有：

```cpp
int row = blockDim.y * blockIdx.y + threadIdx.y;
int column = blockDim.x * blockIdx.x + threadIdx.x;

if (row < m && column < n) {
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum += a[row * k + i] * b[i * n + column];
    }
    c[row * n + column] = sum;
}
```

读 PTX 时只追踪：

```text
线程坐标
→ row / column
→ 越界判断
→ A/B 基址
→ A/B 元素地址
→ global load
→ FMA 累加
→ K 尾部
→ C 地址与写回
```

## 2. 不要背编号：先建立寄存器角色表

寄存器编号是编译器临时分配的，重新编译后可能改变。先看前缀，再看角色。

| 前缀 | 类型 | 本文中的主要角色 |
| --- | --- | --- |
| `%p` | predicate | 边界、循环条件 |
| `%r` | 32-bit 整数/bit | M/N/K、row、column、循环计数 |
| `%rd` | 64-bit 整数/bit | 指针、字节地址、字节步长 |
| `%f` | FP32 | A/B 元素、accumulator |

> `%r17` 与 `%rd17` 属于不同寄存器集合，不是同一个寄存器。

### 2.1 全局角色速查

| 寄存器 | 人话角色 |
| --- | --- |
| `%r1` | `row` |
| `%r2` | `column` |
| `%r12` | `N` |
| `%r13` | `K` |
| `%r14` | `M` |
| `%rd2` | A 的 global base |
| `%rd1` | B 的 global base |
| `%rd17` | C 的 generic pointer，来自 kernel 参数 |
| `%rd3` | `row * K * 4`，A 当前行的固定字节偏移 |
| `%rd5` | `N * 4`，B 一行的字节跨度 |
| `%rd21` | 主循环当前 A 地址 |
| `%rd30` | 主循环当前 B 地址，循环末更新为下一组起点 |
| `%rd22`～`%rd24` | 展开后的后 3 个 B 地址 |
| `%f29` | 跨循环保存的 FP32 accumulator |
| `%rd29` | 最终 C 写回地址 |

## 3. PTX 文件头与 kernel 参数：第 9～27 行

### 3.1 PTX 原文

```ptx
 9  .version 9.3
10  .target sm_80
11  .address_size 64

15  .visible .entry _Z12naive_kernelPKfS0_Pfiii(
16      .param .u64 ..._param_0,
17      .param .u64 ..._param_1,
18      .param .u64 ..._param_2,
19      .param .u32 ..._param_3,
20      .param .u32 ..._param_4,
21      .param .u32 ..._param_5
22  )
23  {
24      .reg .pred %p<9>;
25      .reg .f32  %f<30>;
26      .reg .b32  %r<32>;
27      .reg .b64  %rd<34>;
```

### 3.2 人话解释

```text
param_0：A 指针，64-bit
param_1：B 指针，64-bit
param_2：C 指针，64-bit
param_3：M，32-bit
param_4：N，32-bit
param_5：K，32-bit
```

`float*` 参数本身是 64-bit 地址，所以进入 `%rd`；只有解引用后读出的 FP32 元素才进入 `%f`。

`.reg` 声明的是 PTX 虚拟寄存器集合，不等于最终物理 `registers/thread`。物理资源以 ptxas 日志为准。

## 4. 读取参数与转换 A/B 基址：第 31～39 行

### 4.1 PTX 原文

```ptx
31  ld.param.u64 %rd18, [param_0];  // A generic pointer
32  ld.param.u64 %rd19, [param_1];  // B generic pointer
33  ld.param.u64 %rd17, [param_2];  // C generic pointer
34  ld.param.u32 %r14,  [param_3];  // M
35  ld.param.u32 %r12,  [param_4];  // N
36  ld.param.u32 %r13,  [param_5];  // K

38  cvta.to.global.u64 %rd1, %rd19; // B global base
39  cvta.to.global.u64 %rd2, %rd18; // A global base
```

### 4.2 人话解释

这里完成两步：

1. `ld.param` 从 kernel parameter space 读取参数；
2. `cvta.to.global` 把 generic pointer 转为 global address。

注意编译器的编号顺序：

```text
%rd1 = B global base
%rd2 = A global base
```

不要根据数字猜 A/B，要跟踪数据来源。

C 暂时仍保存在 `%rd17`，直到写回前才在第 133 行转成 global address。

## 5. 线程坐标映射：第 40～48 行

### 5.1 Y 方向生成 row

```ptx
40  mov.u32    %r15, %ctaid.y;          // blockIdx.y
41  mov.u32    %r16, %ntid.y;           // blockDim.y
42  mov.u32    %r17, %tid.y;            // threadIdx.y
43  mad.lo.s32 %r1, %r16, %r15, %r17;  // row
```

等价公式：

$$
row=blockDim.y\times blockIdx.y+threadIdx.y
$$

结果保存在 `%r1`。

### 5.2 X 方向生成 column

```ptx
45  mov.u32    %r18, %ctaid.x;          // blockIdx.x
46  mov.u32    %r19, %ntid.x;           // blockDim.x
47  mov.u32    %r20, %tid.x;            // threadIdx.x
48  mad.lo.s32 %r2, %r19, %r18, %r20;  // column
```

等价公式：

$$
column=blockDim.x\times blockIdx.x+threadIdx.x
$$

结果保存在 `%r2`。

> `mad.lo.s32 d, a, b, c` 可以先读成人话：`d = a * b + c`，保留低 32 bit。

## 6. M/N 边界判断：第 50～53 行

```ptx
50  setp.ge.s32 %p1, %r1, %r14; // p1 = row >= M
51  setp.ge.s32 %p2, %r2, %r12; // p2 = column >= N
52  or.pred     %p3, %p1, %p2;  // p3 = p1 || p2
53  @%p3 bra    $L__BB0_9;       // 越界则跳到 ret
```

人话：

```cpp
if (row >= m || column >= n) {
    return;
}
```

`%p` 是 predicate 寄存器。`@%p3` 表示只有 `%p3` 为真时才执行该分支。

## 7. K 循环分组准备：第 55～75 行

编译器把 K 主循环展开 4 次，并把 `K % 4` 留给尾循环。

### 7.1 空 K 与余数

```ptx
56  setp.lt.s32 %p4, %r13, 1;       // K < 1 ?
57  mov.f32     %f29, 0f00000000;    // acc = 0.0f
58  @%p4 bra    $L__BB0_8;           // K < 1，直接写回 0

60  add.s32     %r22, %r13, -1;      // K - 1
61  and.b32     %r31, %r13, 3;       // remainder = K & 3 = K % 4
62  setp.lt.u32 %p5, %r22, 3;        // K < 4 ?
63  mov.f32     %f29, 0f00000000;    // acc = 0.0f
64  mov.u32     %r30, 0;             // i = 0
65  @%p5 bra    $L__BB0_5;           // K < 4，跳过四次展开主循环
```

这里的 `%r31` 是 32-bit 的 `K % 4`。后面出现的 `%rd31` 是 64-bit A 地址游标，两者不是同一个寄存器。项目 runner 要求维度为正整数，因此正常输入下 `K < 1` 只会覆盖非法输入防御；孤立阅读 kernel 时不能把 PTX 的有符号条件无条件缩写成 `K == 0`。

### 7.2 主循环次数与固定地址量

```ptx
67  sub.s32      %r29, %r13, %r31; // main_count = K - K%4
68  mul.lo.s32   %r24, %r13, %r1;  // row * K
69  mul.wide.s32 %rd3, %r24, 4;    // row * K * sizeof(float)
70  mul.wide.s32 %rd20, %r2, 4;    // column * sizeof(float)
71  add.s64      %rd30, %rd1, %rd20; // &B[0][column]
72  mul.wide.s32 %rd5, %r12, 4;    // B row stride = N * 4 bytes
73  mov.f32      %f29, 0f00000000;  // acc = 0
74  mov.u32      %r30, 0;           // i = 0
75  mov.u64      %rd31, %rd2;       // A cursor = A base
```

下列源码公式成立还依赖项目的输入前提：`M/N/K` 为正，且 `row*K+i`、`i*N+column`、`row*N+column` 可以由 signed 32-bit 线性索引表示。PTX 的 `mul.lo.s32` / `mad.lo.s32` 只保留低 32 bit；超出范围时不能继续按数学整数公式解释。

角色总结：

```text
%r29  = 主循环还剩多少个 K 元素，4 个一组递减
%r30  = 当前 i，4 个一组递增
%rd3  = A 固定 row 的字节偏移
%rd5  = B 每跨一行增加的字节数
%rd31 = A base + i*4 的游标
%rd30 = B base + (i*N+column)*4 的游标
```

## 8. 四次展开主循环：第 77～102 行

这是整份 PTX 最值得读懂的部分。

### 8.1 一轮展开的伪代码

```cpp
acc = fma(A[row][i],     B[i][column],     acc);
acc = fma(A[row][i + 1], B[i + 1][column], acc);
acc = fma(A[row][i + 2], B[i + 2][column], acc);
acc = fma(A[row][i + 3], B[i + 3][column], acc);
i += 4;
```

### 8.2 第 1 对 A/B

```ptx
79  add.s64       %rd21, %rd31, %rd3;   // &A[row][i]
80  ld.global.f32 %f12, [%rd30];         // b0 = B[i][column]
81  ld.global.f32 %f13, [%rd21];         // a0 = A[row][i]
82  fma.rn.f32    %f14, %f13, %f12, %f29; // acc1 = a0*b0 + acc
```

人话别名：

```text
A_addr0 = %rd21
B_addr0 = %rd30
a0 = %f13
b0 = %f12
acc_old = %f29
acc1 = %f14
```

### 8.3 第 2 对 A/B

```ptx
83  add.s64       %rd22, %rd30, %rd5;   // &B[i+1][column]
84  ld.global.f32 %f15, [%rd22];         // b1
85  ld.global.f32 %f16, [%rd21+4];       // a1 = A[row][i+1]
86  fma.rn.f32    %f17, %f16, %f15, %f14; // acc2
```

A 在同一行沿 K 连续前进，所以从 `%rd21` 直接 `+4`。

B 固定 column 跨到下一行，所以从 `%rd30` 增加 `%rd5 = N*4`。

### 8.4 第 3 对 A/B

```ptx
87  add.s64       %rd23, %rd22, %rd5;   // &B[i+2][column]
88  ld.global.f32 %f18, [%rd23];         // b2
89  ld.global.f32 %f19, [%rd21+8];       // a2
90  fma.rn.f32    %f20, %f19, %f18, %f17; // acc3
```

### 8.5 第 4 对 A/B，并准备下一组 B

```ptx
91  add.s64       %rd24, %rd23, %rd5;   // &B[i+3][column]
92  add.s64       %rd30, %rd24, %rd5;   // 提前准备 &B[i+4][column]
93  ld.global.f32 %f21, [%rd24];         // b3
94  ld.global.f32 %f22, [%rd21+12];      // a3
95  fma.rn.f32    %f29, %f22, %f21, %f20; // acc = acc4
```

第 92 行不做本轮第 5 次 load，只把 `%rd30` 更新为下一轮主循环的 B 起点。尾循环会在第 108～110 行重新计算 `%rd33`，不复用更新后的 `%rd30`。

### 8.6 更新循环游标

```ptx
97   add.s32      %r30, %r30, 4;     // i += 4
99   add.s64      %rd31, %rd31, 16;  // A cursor += 4 floats
100  add.s32      %r29, %r29, -4;    // main_count -= 4
101  setp.ne.s32  %p6, %r29, 0;
102  @%p6 bra     $L__BB0_4;
```

主循环一轮实际完成：

```text
4 次 A global load
4 次 B global load
4 次 FP32 FMA
```

## 9. 为什么 A 可以 `+4/+8/+12`，B 要反复加 `N*4`

单个线程固定一个 `row` 和一个 `column`。

A 的访问：

```text
A[row][i]
A[row][i+1]
A[row][i+2]
A[row][i+3]
```

row-major 下它们连续，因此：

```text
A_addr
A_addr + 4
A_addr + 8
A_addr + 12
```

B 的访问：

```text
B[i][column]
B[i+1][column]
B[i+2][column]
B[i+3][column]
```

固定 column、跨不同 row，因此每次增加一整行：

$$
B\_row\_stride=N\times4\text{ bytes}
$$

> 这是单个线程的 K 方向地址轨迹；不要据此直接判断整个 warp 的 coalescing。

## 10. K 尾循环：第 104～127 行

如果 `K % 4 != 0`，处理剩余 1～3 个元素。

### 10.1 判断是否有尾部

```ptx
104 $L__BB0_5:
105 setp.eq.s32 %p7, %r31, 0; // remainder == 0 ?
106 @%p7 bra    $L__BB0_8;    // 无尾部，直接写回
```

这里 `%r31` 是 32-bit remainder，不是主循环中的 64-bit `%rd31` A 游标。

### 10.2 计算尾循环初始地址

```ptx
108 mad.lo.s32   %r25, %r30, %r12, %r2; // i*N + column
109 mul.wide.s32 %rd25, %r25, 4;
110 add.s64      %rd33, %rd1, %rd25;     // &B[i][column]
111 mul.wide.s32 %rd11, %r12, 4;         // B row stride = N*4

112 mad.lo.s32   %r26, %r13, %r1, %r30; // row*K + i
113 mul.wide.s32 %rd26, %r26, 4;
114 add.s64      %rd32, %rd2, %rd26;     // &A[row][i]
```

### 10.3 每次处理一个尾元素

```ptx
116 $L__BB0_7:
119 ld.global.f32 %f23, [%rd33];        // B[i][column]
120 ld.global.f32 %f24, [%rd32];        // A[row][i]
121 fma.rn.f32    %f29, %f24, %f23, %f29;
123 add.s64       %rd33, %rd33, %rd11; // B 跨一行
124 add.s64       %rd32, %rd32, 4;     // A 前进一个 float
125 add.s32       %r31, %r31, -1;      // remainder--
126 setp.ne.s32   %p8, %r31, 0;
127 @%p8 bra      $L__BB0_7;
```

人话伪代码：

```cpp
for (; i < K; ++i) {
    acc = fma(A[row*K+i], B[i*N+column], acc);
}
```

## 11. C 地址与最终写回：第 129～137 行

```ptx
131 mad.lo.s32   %r27, %r1, %r12, %r2; // row*N + column
133 cvta.to.global.u64 %rd27, %rd17;    // C global base
135 mul.wide.s32 %rd28, %r27, 4;       // C index → byte offset
136 add.s64      %rd29, %rd27, %rd28;  // &C[row][column]
137 st.global.f32 [%rd29], %f29;        // C[row][column] = acc
```

人话：

```cpp
C[row * N + column] = sum;
```

这里：

```text
%rd29 = 写入地址
%f29  = 要写入的 FP32 accumulator
```

`st.global.f32 [address], value` 的方括号操作数是目的地址。

## 12. 一张完整数据流图

```text
param_0(A*) ──ld.param──> %rd18 ──cvta──> %rd2 = A base
param_1(B*) ──ld.param──> %rd19 ──cvta──> %rd1 = B base
param_2(C*) ──ld.param──> %rd17 ───────────────┐
M/N/K       ──ld.param──> %r14/%r12/%r13       │
                                                │
block/thread id                                 │
    ├──> %r1 = row                              │
    └──> %r2 = column                           │
             │                                  │
             ├── 越界 predicate ──> return      │
             │                                  │
             ├── A_addr = A + (row*K+i)*4       │
             ├── B_addr = B + (i*N+column)*4    │
             │                                  │
             ├── ld.global A/B                  │
             ├── fma.rn.f32 ──> %f29 acc        │
             ├── 主循环每轮展开 4 次            │
             └── K%4 尾循环                     │
                                                │
C base <──────────────cvta──────────────────────┘
C_addr = C + (row*N+column)*4
st.global C_addr, %f29
```

## 13. 这份 PTX 能证明什么

- 在线性索引不发生 signed 32-bit 回绕的项目输入前提下，线程到 `C[row,column]` 的映射；
- 对非负线程坐标和合法正维度，`row >= M || column >= N` 的线程通过 predicate 跳过计算和写回；
- A/B/C 参数经过 parameter space 与 global address 转换；
- 单线程沿 K 时，A 连续前进 4 字节，B 每次跨 `N*4` 字节；
- K 主循环被展开 4 次；
- 每组执行 4 对 global load 和 4 次 FP32 FMA；
- `%f29` 保存跨主循环和尾循环的 accumulator；
- `K % 4` 由独立尾循环处理；
- 最终结果写入 `C[row*N+column]`。

## 14. 这份 PTX 不能单独证明什么

- 不能证明 A100 最终逐条执行这些 PTX 文本；最终机器路径要看 SASS；
- 不能证明 global load 真正访问了 DRAM，也可能命中 cache；
- 不能只凭单线程地址轨迹证明 warp coalescing 效率；
- 不能证明 kernel 是 memory-bound 或 compute-bound；
- 不能证明寄存器、occupancy 或 wall-clock 性能好坏；
- 不能从 PTX 虚拟寄存器数量推出物理 registers/thread；
- 不能从出现 `fma` 推出已接近 FP32 峰值。
- 不能证明输入维度合法、A/B/C buffer 容量足够，或所有 signed 32-bit 线性索引都不会溢出；这些由 host 端接口和输入验证负责。

这些需要结合 ptxas、SASS、ncu 和正常 benchmark。

## 15. 遮住答案后的自测

1. `%r1`、`%r2` 分别是什么？
2. 为什么 `%rd1` 是 B base，而 `%rd2` 是 A base？
3. `%rd3` 和 `%rd5` 分别是哪种字节偏移？
4. 为什么 A 使用 `%rd21+4/+8/+12`？
5. 为什么 B 使用 `%rd22/%rd23/%rd24`，每次增加 `%rd5`？
6. 第 92 行为什么没有立即对应一次 load？
7. `%r31` 与 `%rd31` 为什么不是同一个寄存器？
8. `%f29` 在主循环、尾循环和写回中分别扮演什么角色？
9. 为什么还需要 `$L__BB0_7` 尾循环？
10. PTX 能证明数据流，却为什么不能直接证明性能瓶颈？

## 16. 三分钟口述模板

> Kernel 参数首先从 parameter space 读入。A、B、C 指针是 64-bit 地址，M、N、K 是 32-bit 整数。线程使用 `%ctaid`、`%ntid`、`%tid` 计算 row 和 column，越界时通过 predicate 直接跳到函数尾部。A 的地址是 `A_base+(row*K+i)*4`，固定行沿 K 连续访问；B 的地址是 `B_base+(i*N+column)*4`，固定列沿 K 每次跨 `N*4` 字节。编译器把 K 主循环展开四次，执行四对 global load 和四次 FP32 FMA，并用尾循环处理 `K%4`。最终 `%f29` 中的 accumulator 被写入 `C_base+(row*N+column)*4`。这些 PTX 能证明程序语义和地址数据流，但不能单独证明 A100 最终机器指令或性能瓶颈，后者还需要 SASS、ptxas、ncu 和 wall-clock benchmark。
