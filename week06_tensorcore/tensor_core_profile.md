# Tensor Core Profile：WMMA FP16 GEMM 分析

> Week3 Day5 产出。用三重证据证明 WMMA 真的用了 Tensor Core，并用 ncu 定位瓶颈。
> 平台：A100 80GB (sm_80)，输入 FP16、累加 FP32。

## 1. benchmark（教学版 WMMA，每 warp 一个 16×16 tile，直读 global）

| M=N=K | time (ms) | GFLOPS | 正确性（抽样） |
|-------|-----------|--------|----------------|
| 256   | 0.0077 | 4363  | PASS (max_rel 4e-6) |
| 512   | 0.0228 | 11771 | PASS (max_rel 4e-6) |
| 1024  | 0.1303 | 16484 | PASS (max_rel 1e-5) |

对照（同平台，2048）：
```text
CUDA core 向量化手写 : 12681 GFLOPS
本 WMMA(1024)        : 16484 GFLOPS  ← 教学版已超 CUDA core 精调版
cuBLAS FP32(CUDA core): 17314
cuBLAS TF32(TensorCore): 88301
```

## 2. 证据一：SASS 有 HMMA 指令（用了 Tensor Core）

```bash
cuobjdump --dump-sass ./wmma | grep -i HMMA
```
```text
HMMA.16816.F32 R8, R16.reuse, R26, R8   ← Tensor Core 机器指令
...共 10 条 HMMA

解读：
  HMMA   = Half-precision Matrix Multiply-Accumulate（Tensor Core 专用）
  .16816 = 16×16×16 MMA 形状（对应 WMMA_M/N/K=16）
  .F32   = FP32 累加（对应 c_frag 是 float）
→ wmma::mma_sync 被编译器降成 HMMA → GPU 用 Tensor Core 执行
```

## 3. 证据二：ncu Tensor pipe 活跃度 > 0（确实在工作）

```bash
ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
    --kernel-name 'regex:wmma.*' ./wmma 1024
```
```text
sm__pipe_tensor_op_hmma_cycles_active = 5.71%
> 0 → Tensor Core 确实在算（与 SASS 互相印证）
但只有 5.71% → Tensor Core 大部分时间在等数据，没吃饱
```

## 4. 瓶颈分析：为什么 Tensor pipe 只有 5.71%

```text
教学版每个 warp 直接从 global 读 A/B tile，没有 shared 复用：
  - 相邻 warp 重复读同一块 A（同行）或 B（同列）
  - 例：A 的第 0 行块被 (N/16) 个 warp 各读一遍
→ Tensor Core 算得快，但数据喂不上来，卡在 global 访存
→ Tensor pipe 95% 时间在等 → 只有 5.71% 在真正算
瓶颈：不在计算（Tensor Core），在访存（global 带宽）。
```

## 5. 优化方向（理解即可，不手写）

```text
1. shared tiling：block 内协作把 A/B tile 搬进 shared，warp 从 shared 读 fragment
   → 砍重复 global 读（load_matrix_sync 源改为 shared 指针）
2. double buffering：cp.async 预取下一块，搬算重叠，藏 global 延迟
3. swizzle：shared 布局避免 bank conflict（兼容 float4 cp.async）
4. 多级 tiling：block→warp→instruction（CUTLASS 三级）
= 把数据喂饱 Tensor Core，pipe 利用率才能提上去 → 逼近 cuBLAS
工业上用 CUTLASS 模板生成，不手写。
```

## 6. 关键认知

```text
1. 有 Tensor Core ≠ 自动快：喂不饱照样慢（pipe 只 5.71%）
2. Tensor Core 优化的是"计算原语"（标量FMA→矩阵MMA），
   访存优化（shared/tiling/流水）照样要做
3. 证明用了 Tensor Core 要三重证据：API + SASS指令 + 性能，不能只看一个百分比
```

## 7. 面试口述

> 我怎么分析我的 WMMA GEMM？三重证据加瓶颈定位。第一，代码用了 wmma::mma_sync；第二，cuobjdump 反汇编看到 HMMA.16816.F32 指令，这是 Tensor Core 机器指令，.16816 对应 16×16×16 MMA、.F32 对应 FP32 累加；第三，性能到 16484 GFLOPS，Tensor Core 量级。但我进一步用 ncu 看 Tensor pipe 只有 5.71% 活跃——说明 Tensor Core 没吃饱。原因是教学版每个 warp 直读 global、无 shared 复用，相邻 warp 重复读 A/B，瓶颈卡在 global 访存。优化方向是 shared tiling + double buffering 把数据喂上去。结论：Tensor Core 不等于自动快，访存优化照样是关键，而且要用 SASS 指令加 pipe 利用率来证明和定位，不能只看一个数字。
