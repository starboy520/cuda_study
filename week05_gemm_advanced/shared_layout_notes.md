# Shared Memory 布局与 Bank Conflict 笔记

> Week2 Day2 产出。从 GEMM 实战理解 bank conflict：为什么发生、发生在哪、怎么消、实测证据。
> 平台：A100 (sm_80)，shared memory 32 bank。

## 1. bank 模型（基础）

```text
shared memory 物理分成 32 个 bank，按 4 字节(一个 float)轮流分配：
  地址0→bank0, 地址1→bank1, ..., 地址31→bank31, 地址32→bank0(绕回)

核心公式：bank = (元素线性下标) % 32

一个 warp 的 32 个线程同时访问 shared：
  - 落在 32 个不同 bank → 一拍完成（无冲突）
  - 多个线程落同一 bank 的不同地址 → 串行化（bank conflict）
  - 多个线程读同一 bank 的同一地址 → 广播（不冲突）
```

## 2. 冲突根源：列宽=32 按列访问

```text
二维 shared sa[BM][BK]，BK=32：
  a[r][c] 的线性下标 = r*32 + c
  bank = (r*32 + c) % 32 = c   ← 只看列！

计算阶段每个线程读"同一列不同行"（列访问）：
  线程0 读 a[0][k] → bank k
  线程1 读 a[1][k] → bank k   ← 同 bank！
  线程2 读 a[2][k] → bank k   ← 同 bank！
  → 32 线程全落 bank k 的不同地址 → 32 路串行冲突

结论：列宽正好=32(=bank数) → 同列所有行映射到同一 bank。
```

## 3. GEMM 里冲突发生在哪

```text
阶段              访问模式            冲突类型
global→shared 写   相邻线程连续地址    global coalescing（不是 bank）
shared→reg 读 sa   同列不同行(列访问)  bank conflict ← 真凶
shared→reg 读 sb   同行不同列(行访问)  无冲突

关键：
  reg_a[i] = sa[ty*TM+i][k]  ← 固定列 k，遍历不同行 → 列访问 → 冲突
  reg_b[i] = sb[k][tx*TN+i]  ← 固定行 k，遍历不同列 → 行访问 → 无冲突
  bank conflict 在"计算阶段读 sa"，不在 load 阶段（load 是连续写，管 coalescing）。
```

## 4. padding 解法

```text
列宽改成 BK+1 = 33：
  a[r][c] 下标 = r*33 + c
  bank = (r*33 + c) % 32 = (r + c) % 32   ← 行 r 也参与了！

按列读 a[r][k]：
  线程0 读 a[0][k] → bank (0+k)%32 = k
  线程1 读 a[1][k] → bank (1+k)%32 = k+1   ← 错开！
  线程2 读 a[2][k] → bank (2+k)%32 = k+2
  → 32 线程落 32 个不同 bank → 无冲突

原理：r*33 让每行多偏移 1 个 bank，把"同列"错开。
```

## 5. 实测证据（A100, 2048）

| shared 布局 | 2048 GFLOPS | ncu shared load bank conflict |
|-------------|-------------|-------------------------------|
| 无 padding `[BK]` | 8995 | **134,288,803**（1.3 亿） |
| padding `[BK+1]` | **12681** | ~0 |

```text
去 padding 后 ncu 显示 shared load bank conflict 高达 1.3 亿，
2048 GEMM 从 12681 掉到 8995（-29%）。加回 padding 恢复。
证明：bank conflict 是真实且巨大的性能杀手。
```

## 6. bank conflict vs coalescing（别混）

```text
shared memory : bank conflict（同 bank 不同地址 → 串行），解法 padding/swizzle
global memory : uncoalesced（地址不连续 → 多次事务），解法 相邻线程连续/向量化/对齐

两者都是"warp 内 32 线程的访问模式"问题，但：
  发生内存不同：shared vs global
  概念不同：撞 bank vs 连不连续
  解法不同：padding vs coalescing
面试别说"global 有 bank conflict"——global 说 coalescing。
```

## 7. padding 的局限（进阶）

```text
padding 简单有效，但和 float4 cp.async 直写 shared 冲突：
  cp.async float4 要求目标连续 16B 对齐 → 加 padding 就不连续。
工业方案 swizzle：用地址重排（异或/位运算打乱映射）消冲突，
  不用 padding → 兼容 float4 cp.async → CUTLASS 的做法。
```

## 8. 面试口述

> shared memory 分 32 个 bank，同一 warp 的线程若访问同一 bank 的不同地址就会串行，这就是 bank conflict。在 GEMM 里，我把 A tile 存成 `sa[BM][BK]`，BK=32 时，计算阶段按列读 `sa[row][k]` 会让 32 个线程全落到同一个 bank（因为 bank = 下标 % 32 = 列号），造成 32 路冲突。解法是 padding 成 `sa[BM][BK+1]`，让 bank 变成 (行+列)%32，每行错开一个 bank。我实测过：去掉 padding 后 ncu 显示 shared load bank conflict 高达 1.3 亿次，2048 GEMM 从 12681 掉到 8995；加回 padding 就恢复。注意 bank conflict 是 shared memory 专属，global memory 对应的是 coalescing。
