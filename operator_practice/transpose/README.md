# Transpose（矩阵转置）

> 面试超高频。核心不是"复用"，而是"转置时保持读写都合并访问"，并顺带练 bank conflict。

## 为什么要 tile（和 GEMM 不同）
```text
naive 转置：读或写必有一边跨步不合并
  output[col*H+row] = input[row*W+col]
  读 input 合并 ✓，写 output 跨 H 跳 → 不合并 ✗

tile 方案：合并读进 shared，转置在 shared 内发生，再合并写回
  global 读 → shared（合并）
  转置 = 读 shared 时交换 x/y 下标
  shared → global 写（也合并）
→ 读写都合并，转置的"错位"由 shared 承担
```

## 三版优化阶梯
```text
v1 naive  : 写回跨步不合并 → 慢
v2 shared : tile[32][32]，写回按列读 shared → 32 路 bank conflict
v3 padded : tile[32][33]，bank=(x+y)%32 错位 → 消 bank conflict → 最快
```

## 关键点
```text
1. block = 32×32 线程，一个 block 处理一个 32×32 tile，一线程一元素
2. 转置：写回时 tile[threadIdx.x][threadIdx.y]（x/y 交换）
3. 输出坐标的 block 也要交换：row_out 用 blockIdx.x，col_out 用 blockIdx.y
4. padding [TILE+1] 消 bank conflict（同 GEMM 的 padding）
5. 边界：非方阵/非整除时 if 判断 row<height && col<width
```

## 完成标准
```text
[ ] CPU 参考校验通过
[ ] 三版都 PASS（naive/shared/padded）
[ ] 带宽递增：naive < shared < padded
[ ] 非方阵测试（1003×769）
[ ] 一段口述：为什么 tile + 为什么 padding
```

## 面试口述
```text
naive 转置读或写必有一边跨步不合并。用 shared tile：合并读进 shared，
转置靠读 shared 时交换 x/y 下标，再合并写回 global——读写都合并。
但 tile[32][32] 写回按列读 shared 会 32 路 bank conflict，
padding 成 [32][33] 让 bank=(x+y)%32 错开，消除冲突。
transpose 是纯 memory-bound，优化目标是逼近 DRAM 峰值带宽。
```
