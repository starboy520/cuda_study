# cuBLAS 与 CUTLASS 面试速成：从会调用到会讲原理

> 适合你当前的位置：已经写过 CUDA GEMM，做过 shared/register tiling、`float4`、`cp.async`、WMMA 和 ncu，但不想把大量时间花在库 API 与 CUTLASS 模板体操上。
>
> 目标：2–3 小时内读懂并能修改 `cublasSgemm`；独立推导非方阵的 `m/n/k/lda/ldb/ldc`；说清 cuBLAS、cuBLASLt、CUTLASS 的边界；能阅读 CUTLASS GEMM 的关键配置；能结合自己的 A100 数据回答面试题。

---

## 0. 全局地图：三者不是互相替代

| 工具 | 本质 | 你提供什么 | 它替你做什么 | 适合场景 |
| --- | --- | --- | --- | --- |
| cuBLAS | NVIDIA 编译好的 BLAS 库 | 操作、shape、指针、stride | 在库内选择高性能 kernel | 标准 GEMM/GEMV |
| cuBLASLt | 更灵活的 matmul API | 上述信息 + layout、epilogue、workspace | 用 heuristic 选算法并支持融合 | 混合精度、灵活布局、bias 等融合 |
| CUTLASS | 开源 CUDA C++ 模板库 | dtype、layout、tile、pipeline、epilogue | 生成可修改的 kernel | 特殊布局、数据类型、融合和内核研究 |

```text
标准线性代数？ -> 先用 cuBLAS
需要灵活 layout、算法选择或 epilogue？ -> cuBLASLt
库无法表达，或必须修改 kernel 内部？ -> CUTLASS / 手写
```

> cuBLAS 是“调用高性能库”，CUTLASS 是“组装和生成高性能 kernel”。标准 GEMM 优先 cuBLAS；只有 layout、dtype、fusion 或算法内部需要定制时，才向 cuBLASLt/CUTLASS 下沉。

官方入口：[cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/)、[CUTLASS 3.x 设计](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x.html)、[CUTLASS GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html)。

---

## 1. GEMM 的数学语言

$$
C = \alpha AB + \beta C
$$

$$
A \in \mathbb{R}^{M\times K},\qquad
B \in \mathbb{R}^{K\times N},\qquad
C \in \mathbb{R}^{M\times N}
$$

- `M`：输出 C 的行数；
- `N`：输出 C 的列数；
- `K`：点积长度；
- 工作量近似为 `2*M*N*K` FLOPs。

单个元素为：

$$
C[i,j] = \alpha\sum_{k=0}^{K-1} A[i,k]B[k,j] + \beta C[i,j]
$$

cuBLAS 实际计算 $C=\alpha\,op(A)\,op(B)+\beta C$。所以 `m/n/k` 描述的是 **`op(A)`、`op(B)` 与 C 的逻辑 shape**，不能只看原指针的 shape。

---

## 2. 逐行读懂项目代码

对照：[week05_gemm_advanced/gemm_cublas.cu](../week05_gemm_advanced/gemm_cublas.cu)。

### 2.1 handle

```cpp
#include <cublas_v2.h>
cublasHandle_t handle;
cublasCreate(&handle);
```

`handle` 是 cuBLAS 上下文，携带 stream、pointer mode、math mode 等状态。不要每做一次 GEMM 就 create/destroy；工程代码应检查每个 cuBLAS API 的返回状态。

### 2.2 math mode 不等于存储类型

```cpp
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
```

它允许符合条件的 FP32 计算使用 TF32 Tensor Core 路径，不表示 `dA/dB` 在显存中换了 C++ 类型。要分清：

1. **storage type**：显存中的 FP16/BF16/FP32；
2. **multiply precision**：乘法器的有效精度；
3. **compute/accumulator type**：部分和的累加精度；
4. **output type**：最终写回类型。

常见 TF32 路径中，输入输出仍可存为 FP32，乘法用 TF32 有效精度，累加用 FP32。不能只比速度而不说明精度语义。

### 2.3 GEMM 调用

```cpp
float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dB, N,
            dA, K,
            &beta,
            dC, N);
```

目标明明是 `C[M,N]=A[M,K]*B[K,N]`，却传 `N,M,K` 并先传 `dB`，原因是 C/C++ 数组按 row-major 准备，传统 cuBLAS 按 column-major 解释。第 3 节完整推导。

### 2.4 event 与异步

```cpp
cudaEventRecord(start);
cublasSgemm(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

GPU 工作通常相对 host 异步。CPU 计时器只包 API，往往只测到入队；CUDA event 在 stream 时间线上记录节点，才适合测 GPU 时间。项目先 warmup 一次再计时是正确方向，但正式 benchmark 应重复多次。

### 2.5 现有示例的工程缺口

- 未检查 cuBLAS 返回值；
- 未将结果拷回与 `hRef` 比较；
- 未释放 host/device 内存和 event；
- 只计时一次；
- 开启 TF32 后仍只打印“cuBLAS”，容易被误当成严格 FP32；
- 未显式绑定 stream。

---

## 3. row-major、column-major 与 leading dimension

### 3.1 用非方阵推导

设 $A$ 是 `2×3`，$B$ 是 `3×4`，所以 `M=2,N=4,K=3`。

$$
A=\begin{bmatrix}
a_{00}&a_{01}&a_{02}\\
a_{10}&a_{11}&a_{12}
\end{bmatrix}
$$

row-major A 的线性内存：

```text
a00 a01 a02 | a10 a11 a12
```

若用 column-major 解释同一段内存，它是 `3×2`：

```text
第 0 列: a00 a01 a02
第 1 列: a10 a11 a12
```

即 $\operatorname{view}_{col}(A_{row})=A_{row}^{T}$，B 同理。

目标 $C_{row}=A_{row}B_{row}$，转置后：

$$
C_{row}^{T}=(A_{row}B_{row})^{T}=B_{row}^{T}A_{row}^{T}
$$

cuBLAS 看到的 `dB` 正是 $B^T$，`dA` 正是 $A^T$。让它算 $B^TA^T=C^T$，所得线性内存恰好对应 row-major C，不需要真的转置数据。

```text
B_col: 4 x 3
A_col: 3 x 2
C_col: 4 x 2

m = N = 4
n = M = 2
k = K = 3
```

### 3.2 leading dimension 到底是什么

**对 cuBLAS 的 column-major 视图，leading dimension 是从一个逻辑列起点走到下一逻辑列起点的物理元素跨度。**

紧密无转置矩阵中，它常等于行数；但不是总元素数，也不能机械等同于逻辑 shape 的某一维。

```cpp
// cuBLAS 看到 B_row^T: N x K，每列长度 N
dB, /* API 的 lda 位置 */ N
// 看到 A_row^T: K x M，每列长度 K
dA, /* API 的 ldb 位置 */ K
// 写 C_row^T: N x M，每列长度 N
dC, /* ldc */ N
```

交换技巧中第一个指针是 `dB`，所以 `dB,N` 的 N 占 API 的 `lda` 位置。不要混淆 C++ 变量名、数学 A/B 与 API 参数位置。

若逻辑 `3×4` column-major 矩阵为了对齐让每列占 5 个元素：

```text
x00 x10 x20 pad pad | x01 x11 x21 pad pad | ...
```

逻辑行数是 3，但 `ld=5`。子矩阵也必须保留母矩阵的物理列跨度。

### 3.3 五步推参法

1. 写 `C[M,N]=A[M,K]*B[K,N]`；
2. 标实际 layout；
3. 写 cuBLAS 对每块内存看到的矩阵；
4. 由 `op(A)*op(B)` shape 推 `m/n/k`；
5. 沿物理内存推 `lda/ldb/ldc`。

一定用 `M != N != K` 测试。方阵会让许多错误参数碰巧相同。

---

## 4. `Sgemm`、`GemmEx`、Batched 与 cuBLASLt

### 4.1 `cublasSgemm`

S 表示 single-precision real。它适合学习 GEMM 参数、做 FP32 基线和使用经典 BLAS。无需背几百个函数，命名规律见 [cuBLAS函数速查](./cuBLAS函数速查.md)。

### 4.2 `cublasGemmEx`

`GemmEx` 的价值是能拆开存储类型与计算类型，例如：

```text
A/B storage      : FP16
C/output storage : FP16 或 FP32
compute          : FP32 accumulate
algorithm        : 允许 Tensor Core
```

面试不要只说“FP16 GEMM”，而要讲清 A/B 怎样存、乘法精度、累加精度和 C 的存储类型。

### 4.3 strided-batched

若要计算一批独立 GEMM，且相邻 batch 的矩阵等距排列，可用 strided-batched API：

```text
A0 ---- strideA ---- A1 ---- strideA ---- A2
```

它让库一次看到整个 batch，避免 host 对每个小 GEMM 单独调用。

- leading dimension：同一矩阵中相邻逻辑列的跨度；
- batch stride：相邻两个矩阵起点的跨度。

### 4.4 cuBLASLt：描述问题，再选择算法

典型心智模型：

```text
1. 创建 matmul descriptor
   compute type / transpose / epilogue
2. 创建 A/B/C/D matrix layout
   dtype / rows / cols / leading dimension / order
3. 创建 preference，指定 workspace 上限
4. 用 heuristic 获取候选 algorithm
5. 调用 matmul
6. 销毁 descriptors
```

两个重点：

1. heuristic 在 shape、layout、硬件和 workspace 约束下选实现，不是数学正确性的一部分；
2. 更多 workspace 可能开放更多候选，但不保证一定更快，仍需 benchmark。

cuBLASLt 也常用于 epilogue 融合：

```text
未融合：
GEMM -> 写 C 到 global -> 启动 bias kernel -> 再读写

融合：
GEMM accumulator -> epilogue 加 bias -> 只写最终 D
```

收益不只少一次 launch，还减少中间结果的 global-memory 往返。

---

## 5. handle、stream、pointer mode 与同步

### 5.1 handle 与 stream

```cpp
cublasSetStream(handle, stream);
cublasSgemm(handle, ...);
```

调用被排入 handle 当前绑定的 stream：

- H2D、GEMM、D2H 可放同一 stream 保持顺序；
- 跨 stream 依赖可用 event；
- 滥用 `cudaDeviceSynchronize()` 会破坏并发；
- 计时 event 要记录在被测工作所在 stream。

### 5.2 pointer mode

默认 host pointer mode 下：

```cpp
float alpha = 1.0f, beta = 0.0f;
// &alpha 和 &beta 是 host 指针
```

若切为 device pointer mode，`alpha/beta` 必须指向 device memory。它适合标量本就在 GPU 上或完整异步流水。常见 bug 是改了 mode，却仍传另一侧指针。

### 5.3 API 返回不等于 GPU 已完成

- 参数检查或入队失败：可由 cuBLAS status 发现；
- GPU 执行期异步错误：可能到后续同步点才暴露。

因此应检查 cuBLAS status 与 CUDA API，并只在真正依赖结果的边界同步。

---

## 6. 可信的 GEMM benchmark

### 6.1 公平比较清单

- 同一 `M/N/K` 和尾块条件；
- 同一 `alpha/op/beta` 数学语义；
- 同一 storage、compute/accumulator、output 类型；
- 明确是否允许 TF32；
- 同一 layout、leading dimension、batch stride；
- warmup 后多次执行；
- event 位于正确 stream；
- 不把不对称的 malloc/memcpy 算入一边；
- 使用匹配精度路径的 atol/rtol；
- 记录 GPU、SM、CUDA/库版本与运行环境。

正确性常用：

$$
|x-y| \le \mathrm{atol}+\mathrm{rtol}|y|
$$

接近 0 靠 atol，大值靠 rtol。FP16、BF16、TF32 不能照搬严格 FP32 容差。用非方阵和非对称数据更容易发现 layout/lda 错误。

### 6.2 解读你的 A100 数据

来自 [GEMM 优化阶梯](../week05_gemm_advanced/gemm_optimization_ladder.md)：

| A100, M=N=K=2048 | GFLOPS | 语义 |
| --- | ---: | --- |
| 手写 `float4` + padding | 12,681 | FP32 CUDA-core 优化路径 |
| cuBLAS FP32 | 17,314 | 项目记录的 FP32 CUDA-core 对照 |
| cuBLAS TF32 | 88,301 | TF32 Tensor Core 路径 |

结论：

1. `12681/17314≈73%`：手写 FP32 kernel 已达到很有意义的同语义水平；
2. 剩余差距来自库对架构和 shape 的 kernel 选择、多级 tiling、swizzle、流水与调参；
3. 12,681 与 88,301 不是同精度代码比拼，巨大差距首先来自计算原语和精度取舍改变。

[Tensor Core profile](../week06_tensorcore/tensor_core_profile.md) 也说明：用上 Tensor Core 不等于已经喂饱 Tensor Core。

### 6.3 常见失败模式

| 现象 | 高概率原因 | 定位方式 |
| --- | --- | --- |
| 方阵对，非方阵错 | `m/n/k` 或 A/B 交换错 | 手推 `2×3 * 3×4` |
| 结果像转置 | row/column-major 混淆 | 打印有坐标规律的小矩阵 |
| 每隔几列开始错 | `lda/ldb/ldc` 错 | 画物理跨度和 padding |
| `beta!=0` 才错 | C 未初始化或 `ldc` 错 | 先 beta=0，再给已知 C |
| 数值接近但失败 | 容差不匹配 TF32/FP16 | 同时检查 atol/rtol |
| 库异常慢 | 小 shape、计时错或语义不同 | 分离 warmup、计时、搬运 |
| 时间极短且不稳 | CPU timer 只测到入队 | 用同 stream CUDA event |
| 多 stream 无并发 | 每次全局同步 | 只在依赖边界同步 |
| 大量小 GEMM 慢 | host 循环逐个 launch | batched/grouped 路径 |

---

## 7. CUTLASS：先看数据流，不做模板体操

接近库性能的 GEMM 要同时处理多级数据移动和 tiling、向量化与对齐、shared layout/swizzle、异步流水、Tensor Core、边界与架构差异。CUTLASS 将这些常用构件模板化、组合化。

它的代价是编译时间、模板报错和学习曲线。因此“标准 GEMM 优先 cuBLAS”是工程判断。

### 7.1 数据流

```text
Global A/B
    | cooperative vectorized/async copy
    v
Shared-memory tiles (layout/swizzle)
    | ldmatrix / tiled copy
    v
Registers
    | mma.sync / WGMMA / SIMT FMA
    v
Accumulator registers
    | epilogue: alpha/beta, bias, activation, convert...
    v
Global C/D
```

**mainloop**：沿 K 维分块，反复搬入 A/B tile 并 MMA 累加。

**epilogue**：累加完成到最终写回之间的阶段，基础公式为：

$$
D=\alpha\cdot Accumulator+\beta\cdot C
$$

还可融合 bias、activation、clamp、类型转换。

### 7.2 为什么要多 stage

```text
单 stage:
load tile 0 | compute tile 0 | load tile 1 | compute tile 1

多 stage:
copy:    tile 0 | tile 1 | tile 2 | tile 3
compute:          tile 0 | tile 1 | tile 2
```

目的是用计算遮蔽搬运。但 stage 不是越多越好：每个 stage 消耗 shared memory 等资源，可能降低 occupancy；若 kernel 已 compute-bound，多 stage 也不会增加计算峰值。

这与项目实验直接对应：逐 float `cp.async` 指令过多，且 kernel 已偏 compute/occupancy-bound，所以双缓冲未自动变快。

---

## 8. CUTLASS 的两套层级

硬件工作分解：

```text
Thread block / CTA tile
    └─ Warp tile
         └─ MMA instruction tile
```

它回答：计算工作如何分给 CTA、warp 和硬件指令？

CUTLASS 3.x 软件组件：

```text
Device
  └─ Kernel
       ├─ Collective mainloop
       ├─ Collective epilogue
       └─ Tiled MMA / Tiled Copy
            └─ Atom
```

- **Device**：host 可调用封装，处理 arguments、workspace、initialize/run；
- **Kernel**：组装 shape、mainloop、epilogue 和调度策略；
- **Collective**：CTA 级 mainloop 或 epilogue 协作算法；
- **Tiled MMA/Copy**：将底层操作铺展到更大 tile；
- **Atom**：基本操作及其 layout 描述。

它回答：代码怎样从 host 封装下沉到底层 MMA/copy？

两套层级不冲突：前者是“谁算哪块”，后者是“软件怎样组装”。

---

## 9. 阅读 CUTLASS 配置：先找八件事

下面是阅读用伪配置，不保证独立编译：

```cpp
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using ArchTag = cutlass::arch::Sm80;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using ThreadblockShape = GemmShape<128, 128, 32>;
using WarpShape         = GemmShape<64, 64, 32>;
using InstructionShape  = GemmShape<16, 8, 16>;

constexpr int AlignmentA = 8;
constexpr int AlignmentB = 8;
constexpr int Stages = 3;

using Epilogue =
    LinearCombination<ElementC, 4, ElementAccumulator, float>;
```

遇到真实长模板，先找：

1. **architecture**：`Sm80` 还是 `Sm90`？
2. **operator class**：SIMT CUDA core 还是 Tensor Core？
3. **dtype**：A/B/C/Accumulator 各是什么？
4. **layout**：A/B/C 如何存？
5. **alignment**：每次访问多少元素，指针和 stride 是否满足？
6. **tile shape**：CTA/warp/instruction 各算多大？
7. **pipeline**：多少 stage，采用何种搬运或调度？
8. **epilogue**：输出类型、向量写回与融合是什么？

再问：

- 对 `M/N/K`、stride、pointer alignment 有何要求？
- tile/stage 消耗多少 register/shared memory，对 occupancy 有何影响？
- 配置针对什么 shape？大方阵最优解未必适合 skinny GEMM。

当前能按此框架读配置并讲取舍，比背几十个模板参数更有面试价值。

---

## 10. CuTe：先懂它解决什么

CuTe 是 CUTLASS 3.x 的 layout 与 tensor 抽象基础。当前可理解为：

> 用可组合 layout 描述“逻辑坐标映射到哪个物理位置，以及这些位置如何分给 thread/warp”。

它统一表达 global/shared/register 的 shape 与 stride、线性内存和多维坐标的映射、thread/value 对 tile 的分工，以及 copy 与 MMA 所需的数据排列。

你已经遇到真实矛盾：`float4/cp.async` 喜欢连续对齐，朴素 padding 能减 bank conflict，却可能破坏向量化布局。工业 kernel 用更精细的 swizzle/layout 同时照顾 copy 与 MMA；CuTe 提供表达这些映射的语言。

当前不必深入 CuTe layout algebra。未来若要修改 CUTLASS 3.x mainloop、设计 swizzle 或写 Hopper kernel，再系统学习 shape/stride、composition、partition。

---

## 11. 从 A100 到 Hopper

### 11.1 Ampere A100（SM80）

- `cp.async`：global 到 shared 异步复制；
- `ldmatrix`：按 Tensor Core fragment 的排列从 shared 协作加载到寄存器；
- `mma.sync`：warp 级 MMA；
- 多 stage pipeline：搬下一个 K tile，同时计算当前 tile。

### 11.2 Hopper（SM90）

- **TMA**：用描述符发起更高维、更粗粒度的异步搬运，减少线程手算地址和发 copy 的负担；
- **WGMMA**：warp-group 级异步 MMA，不是简单放大 `mma.sync`；
- **thread block cluster / distributed shared memory**：多个 block 在 cluster 层协作；
- **warp specialization**：不同 warp/warp group 承担生产者（搬运）与消费者（MMA）角色。

```text
Ampere: 多线程协作发 cp.async + warp 做 mma.sync
Hopper: TMA 粗粒度搬运 + warp-group 做异步 WGMMA

共同目标：
降低地址生成和搬运指令开销，
扩大协作粒度，
让数据移动与 Tensor Core 计算更好重叠。
```

A100 不能运行 SM90 特性，但面试可说：“Hopper 改变了 mainloop 的搬运与 MMA 机制，global→shared→compute→epilogue 的数据流主线没变。”

深入材料：[CUDA 深水区：PTX、SASS、MMA、异步流水与 Hopper](./CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md)。

---

## 12. 面试高频题

### Q1：为什么 row-major 的 `C=A*B` 常交换 A/B？

> 传统 cuBLAS 按 column-major 解释。同一段 row-major A 内存被视为 $A^T$。利用 $(AB)^T=B^TA^T$，让 cuBLAS 按 `B_col*A_col` 算 $C^T$，其线性内存正对应 row-major C，所以传 `N,M,K` 并交换 dB/dA。

### Q2：`lda` 就是 A 的行数吗？

> 不能死记。它是 column-major 视图中相邻列起点的物理元素跨度。紧密无转置时常等于行数，但 padding、子矩阵和 transpose 视图要按物理存储重推。

### Q3：`Sgemm` 与 `GemmEx` 的核心区别？

> Sgemm 是经典 FP32 GEMM；GemmEx 能分别描述存储类型与 compute/accumulator 类型，用于混合精度和 Tensor Core 路径。

### Q4：TF32 会把 FP32 数组变成新显存类型吗？

> 常见 cuBLAS TF32 路径中，存储和输出仍可为 FP32；Tensor Core 乘法采用 TF32 有效精度并用 FP32 累加。它是精度换吞吐的计算路径。

### Q5：为什么 cuBLAS API 返回时计算可能未完成？

> 工作被异步排入 stream。API 返回通常只表示已入队；host 用结果或计时时要在正确 event/stream 依赖上等待，但不应滥用全局同步。

### Q6：cuBLASLt 比 `GemmEx` 多了什么思路？

> 它用 descriptor 描述 matmul 与矩阵 layout，结合 workspace preference 用 heuristic 选算法，并支持更灵活的 epilogue fusion。

### Q7：CUTLASS mainloop 和 epilogue 是什么？

> mainloop 沿 K 维搬入 A/B tile 并 MMA 累加，核心是多级 tiling 与搬算重叠；epilogue 在累加后做 alpha/beta、转换、bias/activation 并写回。

### Q8：两套 CUTLASS 层级是一回事吗？

> CTA/warp/instruction 描述硬件工作分解；Device/Kernel/Collective/Atom 描述 3.x 软件组装。一个回答谁算哪块，一个回答代码怎样组成。

### Q9：何时不应自己写 CUTLASS kernel？

> 标准 GEMM/GEMV 已被 cuBLAS/cuBLASLt 支持且性能达标时。自定义 CUTLASS 会增加编译、调参、架构迁移与正确性维护成本。

### Q10：你的手写 GEMM 与 cuBLAS 差距来自哪里？

> 我的 A100 FP32 `float4`+padding 版在 2048 方阵上是 12681 GFLOPS，项目记录的 cuBLAS FP32 是 17314，约 73%。差距来自架构/shape kernel 选择、多级 tiling、swizzle、搬运流水与调参。cuBLAS TF32 的 88301 换了 Tensor Core/TF32 语义，不能当成同精度代码差距。

---

## 13. 2–3 小时学习路线

### 第 1 段：30 分钟——读现有代码

1. 打开 [gemm_cublas.cu](../week05_gemm_advanced/gemm_cublas.cu)；
2. 指着参数说出 handle、op、`m/n/k`、标量、指针与 `ld*`；
3. 解释为何传 `N,M,K` 与 `dB,dA`；
4. 找出错误检查、资源释放、正确性与 benchmark 缺口。

达标：不看本文也能口述整段调用。

### 第 2 段：45 分钟——手推非方阵

1. 写 `A[2,3]*B[3,4]`；
2. 展开 row-major 线性内存；
3. 用 column-major 重新分组；
4. 推出 $B^TA^T=C^T$；
5. 推出 `m=4,n=2,k=3` 与三个 leading dimension；
6. 再用 `M=3,N=5,K=2` 独立做一遍。

达标：解释 leading dimension 是物理跨度，说明 padding 为何改变它。

### 第 3 段：45 分钟——选型与 CUTLASS

1. 口述 `Sgemm→GemmEx→cuBLASLt→CUTLASS` 的边界；
2. 默写 `global→shared→register/MMA→accumulator→epilogue→global`；
3. 解释 mainloop、epilogue、stage、tile、alignment；
4. 从第 9 节配置找出八类信息；
5. 说清两套层级的差异。

达标：看到 CUTLASS GEMM 配置，即使不会从零写，也知道它在配什么。

### 第 4 段：30 分钟——模拟面试

1. 遮住答案回答第 12 节；
2. 每题先用 20 秒给结论，再用 40–60 秒讲原理；
3. 最后用 2 分钟讲自己的 A100 GEMM 优化链与 cuBLAS 对比。

达标：回答中有“选择—原因—代价—证据”，而不是只堆 API 名。

---

## 14. 现在需要学 CUTLASS 代码吗？

结论：**需要会读关键配置，目前不需要从零写完整 CUTLASS kernel。**

当前优先级：

1. 真正理解 cuBLAS 参数和 row/column-major；
2. 能将自己的 GEMM 与 cuBLAS 公平对比；
3. 能读 architecture/dtype/layout/tile/stage/epilogue；
4. 知道 CuTe、TMA、WGMMA 解决什么；
5. 真遇到 cuBLAS/cuBLASLt 表达不了的需求，再投入 CUTLASS/CuTe 编程。

你已经手写过 GEMM 优化链，最值钱的下一步不是复制庞大模板，而是把现有经验与库的设计语言准确接上。

---

## 15. 一页速记

```text
cuBLAS
  编译好的高性能 BLAS；标准 GEMM 首选

row-major 适配
  C=A*B -> C^T=B^T*A^T
  交换 A/B，交换 M/N

leading dimension
  column-major 视图中相邻逻辑列起点的物理元素跨度
  不是总元素数；padding/子矩阵尤其重要

GemmEx
  storage type 与 compute/accumulator type 可分开

cuBLASLt
  descriptor + layout + preference/workspace + heuristic + matmul
  灵活布局、算法选择与 epilogue fusion

CUTLASS
  模板构件生成可定制 kernel
  mainloop + epilogue
  global -> shared -> registers/MMA -> accumulator -> store

两套层级
  CTA -> warp -> instruction             : 硬件工作分解
  Device -> Kernel -> Collective -> Atom : 3.x 软件组装

A100 / Hopper
  Ampere: cp.async + ldmatrix + mma.sync
  Hopper: TMA + WGMMA + cluster + warp specialization

选型
  标准问题 -> cuBLAS
  灵活 layout/fusion -> cuBLASLt
  要改 kernel 内部 -> CUTLASS / 手写
```
