# Week 4 学习笔记

> 主题：GEMM 优化阶梯 + cuBLAS 标尺 + Nsight Compute(ncu)入门
> 硬件：Tesla T4(sm_75)，FP32 峰值 ~8.1 TFLOPS，带宽 320 GB/s，Roofline 拐点 ≈25

---

## Day 1：naive GEMM + Roofline 定位

### GEMM 索引口诀（务必记住，矩阵索引不再乱）

```text
1. 矩阵全是【一维数组】，二维只是你脑子里的逻辑视图(行优先压平)
2. 压平公式：M[行][列] = m[行 × 该矩阵的列宽 + 列]
   A[M×K] 列宽=K，B[K×N] 列宽=N，C[M×N] 列宽=N
3. 线程的 (row, column) 就是它要算的 C[row][column]
4. GEMM 内层：C[row][col] = Σ_i A[row][i]·B[i][col]
   A 沿列走(i 当列号) → a[row*K + i]
   B 沿行走(i 当行号) → b[i*N + col]
```

**为什么容易乱**：把"矩阵的二维"和"线程的二维"混了。
```text
矩阵 = 一维数组，二维是逻辑，用 行×列宽+列 手动压平(假二维)
线程 = 真二维(row=threadIdx.y+..., column=threadIdx.x+...)，每个算一个 C[row][col]
→ 想清楚"线程负责哪个C → 读A哪行B哪列 → 各自压平"就不乱
易错点：B 的索引是 b[i*N+col]，不是 b[col*N+i](后者把 B 转置了)
```

### 作业：gemm_naive（代码 week04_gemm/gemm_naive/gemm_naive.cu）

每个线程算一个 C[i][j]，循环 K 累加。2D grid(block 16×16)。

### 实测基线（T4，nvcc -O3 -arch=sm_75）

```text
方阵 512³:        1.185 ms   226.5 GFLOPS   PASS
非方阵 384×640×256: 0.336 ms   374.9 GFLOPS   PASS
→ baseline = 226 GFLOPS @ 512³
```

### 为什么 naive 慢：memory-bound（Roofline 定位）

```text
T4 FP32 峰值 8100 GFLOPS，naive 只 226 → 利用率 2.8%！还有 35x 空间

AI(算术强度)推导：每个 C[i][j] 读 A 一行(K)+ B 一列(K)= 2K 次读 = 2K×4 字节
                  算 2K FLOP(K 乘 + K 加)
  AI = 2K / (2K×4) = 0.25 FLOP/byte
T4 拐点 ≈ 25 → AI=0.25 << 25 → 远在拐点左边 → memory-bound(带宽卡死)

人话：每个 A 元素被 N 个输出各读一遍(复用为零)，读得多算得少 → 带宽满、算力闲
```

### Day1 完成标准 ✅

```text
✅ naive GEMM 正确(方阵 + 非方阵都 PASS)
✅ baseline GFLOPS = 226(512³)
✅ 口述 memory-bound：复用为零，AI=0.25 << 拐点25，被带宽卡死
```

> 下一步(Day2)：shared-memory tiling，把 A/B 子块搬进 shared 读一次复用 TILE 次，
> AI 提升 ~TILE 倍 → 逃离带宽墙 → GFLOPS 应明显上升。

---

## Day 2：Shared-memory Tiling

### 核心思想：tiling = 贴瓷砖

```text
tile = 瓷砖/方块；tiling = 把大矩阵 C 切成 TILE×TILE 小块,一个 block 贴一块
关键收益：A/B 子块搬进 shared 一次,block 内复用 TILE 次
→ 一次 global 读 → TILE 次复用 → AI ≈ 0.25×TILE → 逃离带宽墙
```

### 三件套（写 kernel 必记）

```text
1. block = dim3(TILE, TILE)  ← 必须!As/Bs 是 TILE×TILE,用 ty/tx 当下标
2. 两个 __syncthreads()       ← load 后(等搬完)、算后(等算完再换段)
3. 边界填 0                   ← M/N/K 非 TILE 整数倍时,越界读填 0
```

### 索引（和 naive 一脉相承）

```text
row=blockIdx.y*TILE+ty  col=blockIdx.x*TILE+tx
第 t 段:全局 K 下标 = t*TILE + k
  As[ty][tx] = A[row][t*TILE+tx] = a[row*K + (t*TILE+tx)]   (A 的 K 在列→用 tx)
  Bs[ty][tx] = B[t*TILE+ty][col] = b[(t*TILE+ty)*N + col]   (B 的 K 在行→用 ty)
计算:sum += As[ty][k] * Bs[k][tx]   (sum 不清零,跨 t 持续累加)
```

### 两层循环怎么扫完 K（最该想透）

```text
外层 t = K 方向的一个 tile(一个"段/括号"),内层 k = 段内偏移
分段求和 = 整段求和(加法结合律):每段算部分和,sum 累加 = 完整点积
```

### 跨 block vs 块内（别混）

```text
块内沿 K 滑动 → 一个 block 把"自己那格"算完(shared + __syncthreads)
跨 block      → 每个 block 包一格,互不通信、完全独立并行拼出整张 C
→ GEMM 输出不重叠,无需 atomic/跨 block 同步
```

### 作业：gemm_tiled（代码 week04_gemm/gemm_tiled/gemm_tiled.cu）

独立写出，踩过两个坑：
```text
1. tileA/tileB 忘加 __shared__ → 退化成"错误的 naive"(每线程私有,只填1格)
2. bRow = t*TILE_SIZE * ty 手误(应为 +) → t≥1 时 B 取错行
另:host 的 dim3 block 要用 TILE_SIZE,别写死 16(否则改 TILE 就错)
```

### 实测（T4，nvcc -O3 -arch=sm_75）

```text
            512³                1024³               PASS
naive       1.185 ms / 226      —                   ✅
tiled T16   0.580 ms / 463      5.086 ms / 422      ✅  ← 比 naive 快 ~2x
tiled T32   —                   5.355 ms / 401      ✅  ← 反而略慢!

→ tiling 带来 ~2x 加速(226→463 @512³)
```

### TILE=16 vs 32 调参结论（面试谈资）

```text
实测 TILE=32(401) < TILE=16(422),更大 tile 反而略慢!
原因:TILE↑ → 复用↑ AI↑(好),但 block↑(32²=1024线程/8KB shared)
     → 一个 SM 容纳 block 数↓ → occupancy↓ → 延迟隐藏变差(坏)
→ 16 是甜点;约束 TILE×TILE≤1024 → TILE≤32
→ 真正质变靠 register tiling(Day3),不是单纯加大 TILE
```

### Day2 完成标准 ✅

```text
✅ tiled GEMM 正确(512/1024 PASS)
✅ 比 naive 快 ~2x(226→463)
✅ 口述 tiling 原理:shared 复用 TILE 次,AI 提升 ~TILE 倍
✅ 实测并解释 TILE=16 vs 32(occupancy 权衡)
```

---

## Day 3：Register Tiling

> ⏸️ **暂缓**（2026-06-22 决定）：register/1D/2D tiling 手写难度大，且对"应用/融合算子"
> 岗位非必需（用 cuBLAS/CUTLASS 即可）。已掌握 naive + tiled 的核心（tiling 思想、
> 2x 加速、Roofline memory-bound 分析）→ 算法层"会写即可"目标达成。
>
> **口述级结论（面试够用，不必手写）**：
> ```text
> register tiling = 一个线程算多个输出(TM×TN) → 把 A/B 读进寄存器复用
>   → 减少 SMEM 访问、提升算术强度 → 再快几倍
> 代价:寄存器用量↑ → occupancy↓,靠 ILP 弥补,要实测找平衡
> ```
> 详细教学见课程文档 §3.5（1D thread tiling 一步步）+ siboehm 教程
> https://siboehm.com/articles/22/CUDA-MMM —— 以后回来啃。
>
> **下一步转向**：算子层（softmax/layernorm 等），复用 week03 的 reduction 技能，
> 对"工程层 + 面试手写算子"性价比更高。

---

## Day 4：cuBLAS 当标尺

（待填）

---

## Day 5：ncu 入门（本周重点）

（待填）

---

## Day 6：Thrust / 算子封装

（待填）

---

## Day 7：复盘 + 性能表

（待填）
