# GEMM 优化阶梯总表（A100, FP32, M=N=K=2048）

> 把 naive → cuBLAS 的每一步收进一张表，能看着表讲完整条优化链。
> 平台：A100 80GB (sm_80)，FP32 峰值 19.5 TFLOPS，带宽 1935 GB/s，Roofline 拐点 AI*≈10。

## 优化阶梯总表

| 版本 | 解决什么 | 新增代价 | 主要瓶颈 | GFLOPS(2048) | AI | ncu / 证据 |
|------|---------|---------|---------|-------------|-----|-----------|
| naive | — | — | global 重复读 A/B | (未测) | 0.25 | 重度 memory-bound |
| shared tiling | tile 搬 shared，block 内复用 → 砍 DRAM 流量 64× | — | shared 读 / 指令吞吐 | ~4200 | 16 | DRAM 流量↓，越过拐点 |
| 2D register tiling | 寄存器外积复用 → 砍 shared 读取 | 寄存器多、occupancy↓ | occupancy | 11305 | 16 | reg~118，occ 14%→30% |
| float4 向量化 load | float4 读 global（宽事务、少指令） | 需 K%4==0 && N%4==0 | compute / occupancy | **12681** 🥇 | 16 | 加载指令↓，约峰值 65% |
| cp.async 双缓冲 | cp.async 异步预取，藏 global 延迟 | 逐 float 拷贝指令多 | compute | 7612 | 16 | 逐 float memcpy_async 太多 |
| Tensor Core (WMMA) | MMA 硬件，吞吐远超 CUDA core | 需降精度(FP16/TF32) | — | 待 Week3 | — | HMMA 指令 / Tensor pipe |
| **cuBLAS (FP32, CUDA core)** | 三级 tile + swizzle + 手调 | 闭源 | — | **17314** | — | 手写达其 **73%** |
| **cuBLAS (TF32, Tensor Core)** | 换 Tensor Core（TF32 MMA） | 精度降(10位尾数) | — | **88301** | — | 硬件代差 **~7×**，Week3 动机 |

> cuBLAS 实测（A100, 2048, cudaEvent 计时）：
> - 默认 FP32（CUDA core）：**17314 GFLOPS**，你的向量化版 12681 ≈ 其 **73%**
> - 开 `CUBLAS_TF32_TENSOR_OP_MATH`（TF32 Tensor Core）：**88301 GFLOPS**（≈88 TFLOPS，TF32峰值~156T 的 57%）
> - 关键：12681 vs 17314 是"CUDA core 同台"（差 27%）；vs 88301 是"CUDA core vs Tensor Core"硬件代差（7×），不是代码问题。
> - 这 7× 就是 Week3 学 Tensor Core / WMMA 的直接动机。

## 优化链叙事（把表串成故事）

```text
naive：每算一个 C 元素都从 global 读 A/B，无复用，AI=0.25 → 重度 memory-bound。
  ↓ shared tiling
把 A/B tile 搬进 shared，block 内线程复用 → DRAM 流量降 64×，AI=16，越过 A100 拐点(10)
进入 compute 侧。但每线程仍频繁读 shared。
  ↓ 2D register tiling
一个线程算 TM×TN，用 regM×regN 做外积，砍掉大量 shared 读取 → 2048 到 11305。
代价：寄存器变多、occupancy 从 14% 降到 30%(仍靠复用取胜)。
  ↓ float4 向量化 load
A/B tile 用 float4 读 global（16B 宽事务、加载指令÷4），shared 仍标量写 + padding 防冲突
→ 2048 到 12681，约峰值 65%（当前手写最优）。
  ↓ cp.async 双缓冲
用 cuda::pipeline + 两块 buffer 预取下一块、算当前块。结构正确，但逐 float 的 cp.async
指令太多 + kernel 已 compute-bound → 反而降到 7612。要更快需 float4 cp.async + swizzle。
  ↓ Tensor Core / cuBLAS
再往上是降精度上 Tensor Core（Week3），以及 cuBLAS 的三级 tile + warp specialization +
按 shape/架构选 kernel + 极致手调。
```

## 关键认知（面试高频）

```text
1. AI 只有 naive 是 0.25，tiling 之后都是 16（DRAM 流量相同）
   → Roofline 在 DRAM 层分不出 shared/register/向量化版，要用 ncu 看 shared/occupancy。
2. 优化是"瓶颈转移"：DRAM → shared → occupancy/compute，每步对症下药。
3. occupancy 不是越高越好：2D register occupancy 只有 30% 却最快（复用 > 并发）。
4. 向量化只加速 global 读；shared 防 bank conflict 靠 padding；两者别混用。
5. 双缓冲在 compute-bound kernel 上收益有限，要 float4 cp.async + swizzle 才值。
```

## 面试口述（3 分钟压轴）

> 我把 GEMM 从 naive 一步步优化到向量化版。naive 是重度 memory-bound，AI 只有 0.25；
> shared tiling 复用把 DRAM 流量降 64 倍，AI 到 16，越过 A100 的 Roofline 拐点进入 compute 侧；
> 2D register tiling 用寄存器外积进一步砍 shared 流量，2048 到 11305，虽然 occupancy 从 14%
> 降到 30% 但因为复用更充分反而更快；float4 向量化读 global 再提到 12681，约峰值 65%。
> 我还试了 cp.async 双缓冲，结构正确三档都 PASS，但因为逐 float 加载 + kernel 已 compute-bound，
> 反而慢到 7612——想让它更快得用 float4 cp.async + swizzle，那是 CUTLASS 的做法。
> 每一步我都用 ncu 验证瓶颈从 DRAM 转到 shared 再到 occupancy，知道下一步该往哪优化，
> 这就是我理解的 GEMM 优化方法论：写出来、测出来、用 profiler 解释、再决定下一步。

## 待补
```text
[ ] naive 实测 GFLOPS（有 naive 代码就顺手跑）
[ ] cuBLAS 对照 GFLOPS（加 cublasSgemm）
[ ] Tensor Core 版（Week3）
```
