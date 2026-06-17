# 卷六：核心算子开发

> 前五卷打通了"GPU 怎么算、内存怎么用、算法怎么并行、性能怎么测"。卷六把这些
> 能力收束到一个目标：**写出真实深度学习/HPC 系统里用得上的核心算子**——GEMM、
> 卷积、Softmax、LayerNorm，以及把它们融合起来。这是从"会写 kernel"到"会写
> 工业级算子"的关键一跃。

## 为什么这一卷是分水岭

```text
前五卷的练习（vector add、transpose、reduction）：
  教概念，但单个算子简单，优化空间有限

卷六的算子（GEMM、attention 相关）：
  - 是真实模型 90% 算力的所在（Transformer 的矩阵乘 + softmax + norm）
  - 优化阶梯长（naive → tiling → register → 向量化 → 双缓冲 → Tensor Core）
  - 每一步都能量化、都有明确动机
  - 直接对应 CUDA 岗位的核心考点
```

学完这一卷，你应该能：把一个算子从 naive 实现，一步步优化到接近库的性能，并能
说清每一步**为什么这么做、提升了多少、瓶颈转移到了哪**。

## 学习目标

- 推导 GEMM 的数学、内存布局和性能上限（Roofline 视角）。
- 掌握 GEMM 优化阶梯：shared-memory tiling → register tiling → 向量化 → 双缓冲。
- 理解 Tensor Core / WMMA / 混合精度的价值与约束。
- 实现数值稳定的 Softmax（减最大值 + 多级归约）。
- 实现 LayerNorm / RMSNorm，理解"融合"如何省内存带宽。
- 区分 elementwise / broadcast / reduction，掌握 fusion 思维。
- 知道 cuBLAS / cuDNN / CUTLASS 的边界——什么时候用库、什么时候手写。

## 章节

1. [GEMM 数学、布局与性能上限](01_GEMM数学_布局与性能上限.md)
2. [GEMM 优化阶梯：Tiling 与寄存器分块](02_GEMM优化阶梯_Tiling与寄存器分块.md)
3. [向量化、双缓冲与 Tensor Core](03_向量化_双缓冲与Tensor_Core.md)
4. [Softmax 数值稳定与多级归约](04_Softmax数值稳定与多级归约.md)
5. [LayerNorm、RMSNorm 与融合](05_LayerNorm_RMSNorm与融合.md)
6. [库的边界与算子工程化](06_库的边界与算子工程化.md)

## 这一卷怎么承接前面

```text
卷三 内存系统    → GEMM tiling 用 shared memory、coalescing、bank conflict
卷四 并行算法    → softmax/layernorm 的归约就是卷四 reduction 的变体
卷五 性能工程    → 每个算子都用 Roofline / Nsight 判断瓶颈和优化方向
卷二 GEMM naive  → 本卷从那个 naive 版出发，一路优化
```

所以卷六不是新知识的堆砌，而是**把前五卷的工具组合起来解决真实问题**。

## 前置回顾（开始前确认你掌握）

- [ ] 行主序地址：`A[row * width + col]`（术语约定）。
- [ ] Naive GEMM 与它的算术强度 ≈ 0.25 FLOP/byte（卷二第 09 章）。
- [ ] Shared memory tile、`__syncthreads()`、bank conflict + padding（卷三）。
- [ ] 树形归约、warp shuffle（卷四）。
- [ ] Roofline、有效带宽、GFLOPS（卷五）。

## 配套实验（建议边学边写）

```text
labs/06_operators/
├── gemm_tiled/        # naive vs shared-memory tiled GEMM，实测 GFLOPS
└── softmax/           # 朴素(会溢出 NaN) vs 数值稳定 softmax
```

构建并运行：

```bash
cd /home/qichengjie/workspace/cuda_study/cuda_deep_course

make -C labs/06_operators/gemm_tiled clean all
./labs/06_operators/gemm_tiled/gemm_tiled 2048 2048 2048

make -C labs/06_operators/softmax clean all
./labs/06_operators/softmax/softmax
```

参考实测（T4，仅供量级参考，每台机器不同）：

```text
gemm_tiled (2048×2048×2048)：
  naive   441 GFLOPS
  tiled   736 GFLOPS   ← shared tiling 数据复用，约 1.67x（TILE=16）

gemm_tiled (512×512×512)：两版均 PASS（和 CPU double 参考比，相对误差 < 1e-3）

softmax (大值数据 [500,1500])：
  naive   每行和 = NaN  ❌  （exp 溢出 → inf/inf）
  stable  每行和 = 1.0  ✅  （减最大值，exp ≤ 1 永不溢出）
```

> 这两个 lab 把卷六的核心结论变成了可亲眼验证的数字：tiling 的 GFLOPS 提升、
> softmax 减最大值的必要性。GEMM 还能继续往上爬（register tiling、向量化、Tensor
> Core，见第 02/03 章），把 GFLOPS 推得更高。

> 正文聚焦"原理 + 优化阶梯 + 数值稳定"。更深的优化（register tiling、LayerNorm 融合
> 等）可按需逐个落地为新 lab，像卷七那样实测每一步的加速比。

## 主要资料

- CUDA C++ Programming Guide：Shared Memory、Tile Kernels、Warp Matrix Functions (WMMA)。
- CUDA C++ Best Practices Guide：Memory Optimizations、Arithmetic Intensity。
- PMPP：Matrix Multiplication、Convolution。
- CUTLASS 文档与论文（层次化 GEMM 设计思想）。
- cuBLAS / cuDNN 文档（作为性能基线与边界参考）。
- FlashAttention 论文（softmax 融合与在线归约的现代范例，进阶）。

## 完成标准

- [ ] 能推导 GEMM 的算术强度，解释为什么 naive 是访存受限。
- [ ] 能实现 shared-memory tiled GEMM，并说明 tile 如何提升数据复用。
- [ ] 能实现 register tiling，解释"每线程算多个输出"为什么进一步提速。
- [ ] 能说清 Tensor Core 的价值与使用约束（数据类型、对齐、布局）。
- [ ] 能实现数值稳定的 softmax，解释"减最大值"防溢出的原理。
- [ ] 能实现 LayerNorm，解释融合归约如何省内存往返。
- [ ] 能判断一个算子该用库还是手写。
