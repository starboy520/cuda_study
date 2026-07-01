# Week 3：Tensor Core 与混合精度

> 学习手册：docs/Week3_TensorCore学习文档.md
> 本周主线：手写 WMMA FP16 GEMM，验证正确性 + benchmark + ncu 证明用了 Tensor Core。

## 产出清单
```text
[ ] wmma_fp16_gemm.cu       独立写的 WMMA GEMM（核心）
[ ] mixed_precision_table.md 精度对照表（FP32/TF32/FP16/BF16/FP8）
[ ] tensor_core_profile.md   benchmark + ncu 证据 + cuBLAS 对照
[ ] fp8_scaling_notes.md     FP8 scaling / DeepGEMM 口述笔记
```

## Day 安排（对照手册第16节）
```text
Day1 混合精度账本 → mixed_precision_table.md
Day2 WMMA 心智模型（fragment/K循环数据流）
Day3 跑通 Demo → wmma_fp16_gemm.cu，256/512/1024 PASS + benchmark
Day4 关掉答案独立重写 + 正确性（全1→随机→非方阵）
Day5 cuBLAS 对照 + ncu（SASS 找 HMMA/MMA）→ tensor_core_profile.md
Day6 A100 TF32/BF16 对照 + FP8 scaling → fp8_scaling_notes.md
Day7 DeepGEMM + 复盘
```

## 硬件
```text
T4 (sm_75)  : FP16 WMMA
A100 (sm_80): FP16 + TF32 + BF16
编译：nvcc -O3 -std=c++17 -arch=sm_80 wmma_fp16_gemm.cu -o wmma
```

## 关键认知（已理解）
```text
- 一个 warp 算一个 16×16 tile，grid 铺满所有 tile
- tile_row/tile_col = C tile 坐标；A 取 tile_row 行块、B 取 tile_col 列块
- 只给 tile 起点 + 行宽(lda=K, ldb=N, ldc=N)，thread↔element 映射由 WMMA 隐藏
- 输入 FP16、累加 FP32
- 目的：正确驱动 Tensor Core，不是比 cuBLAS 快
```

## 进度
```text
[x] Day3 wmma_fp16_gemm.cu 跑通：256/512/1024 PASS
    256:4363  512:11771  1024:16484 GFLOPS（误差1e-5）
    → 教学版WMMA已超CUDA core向量化版(2048=12681)，Tensor Core威力
```

## 待改进想法（记录，暂不做）
```text
grid-stride warp 版：让 launch 配置和 M/N/K 解耦
  warp_stride = 总warp数 = gridDim.x*blockDim.x/32
  for (t = warp_id; t < total_tiles; t += warp_stride) { 处理 tile t }
  fragment 声明放循环外，每个 tile 开头 fill_fragment(c_frag,0) 清零
  grid 固定(如 SM数×2)，任意规模通用
  —— 就是 reduce 的 grid-stride，单位从"元素"升到"16×16 tile"
```

