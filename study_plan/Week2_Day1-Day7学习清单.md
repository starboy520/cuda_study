# Week 2 每日学习清单（Day1–Day7）

> **使用方式**：按 Day1 → Day7 顺序勾选。每 Day 对应 [Week2详细步骤.md](Week2详细步骤.md) 中的 Step。  
> **官方文档**：[CUDA Programming Guide v13.x](https://docs.nvidia.com/cuda/cuda-programming-guide/)  
> **详细 Step**：[Week2详细步骤.md](Week2详细步骤.md)

### v13.x 阅读顺序（Week 2 总览）

```text
Day1  2.3 Global Memory + Coalescing + Best Practices Memory 前半
Day2  2.3 Shared Memory + 5.4.4 __syncthreads
Day3  2.3 Matrix Transpose + 本地转置详解（naive / shared_simple）
Day4  2.3 Bank Conflict + transpose_shared + GB/s 三版对比
Day5  Harris Reduction 文章 + reduction 目录搭建
Day6  shared 树形归约 + 1M 多 block PASS
Day7  性能表定稿 + compute-sanitizer（选做）+ Week 2 复盘
```

**Week 2 跳过**：2.4 Tile Kernels、5.4.5 Atomics（Week 3）、5.4.6 Shuffle（Week 3）、Unified Memory

---

## 本周总览

- [ ] Day1（Step 01–02）：内存层次 + 合并访问 Coalescing
- [ ] Day2（Step 03）：Shared Memory + `__syncthreads`
- [ ] Day3（Step 04–05）：transpose naive / shared_simple
- [ ] Day4（Step 06）：transpose_shared + 带宽三版对比
- [ ] Day5（Step 07–08）：reduction 基线 + 单 block 树形归约
- [ ] Day6（Step 09）：1M 多 block reduction PASS
- [ ] Day7（Step 10–12）：性能表 + 复盘 + `notes/week02.md` 定稿
- [ ] `notes/week02.md` 每天都有：目标、实验、问题

### 你已有的进度（Day3–4 可加速）

| 项目 | 状态 | 建议 |
|------|------|------|
| `transpose.cu` 三版 PASS | ✅ | Day3–4 改为验收 + 补 GB/s 计时 |
| `my_transpose.cu` | ✅ | Day4 对照阅读 |
| `转置优化详解.md` | ✅ | Day3–4 必读 |
| `vec_add` → `reduce_sum` | 🔄 | Day2 读懂，Day5–6 迁出 |

---

## Day1：内存层次 + 合并访问

**对应 Step**：01–02  
**预计时间**：3–4 h

### 今天要学什么

- [ ] 六种内存：Register / Local / Shared / Global / Constant / Texture（Week 2 重点前四种）
- [ ] Global Memory：`cudaMalloc`，全 Grid 可见，高带宽高延迟
- [ ] **Coalescing**：同一 warp 32 线程访问连续地址 → 合并成少量事务
- [ ] 反例：stride 访问 → 不合并 → 带宽暴跌

### 需要看的资料

- [ ] Programming Guide **2.3** — 页内搜 `Global Memory`、`Shared Memory`
- [ ] [Programming_Model详解.md](../docs/Programming_Model详解.md) — Memory 相关小节
- [ ] [Best Practices Guide — Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) 前半（Coalescing 段）
- [ ] [GPU架构图资源.md](../docs/GPU架构图资源.md) — Memory Hierarchy 图
- [ ] [T4实战指南.md](../docs/T4实战指南.md) — 300 GB/s 带宽、48 KB shared/block

### 动手步骤

1. 打开 `week02_memory/transpose/transpose.cu`，**只读** `transpose_naive`（29–40 行）
2. 回答并写入 `notes/week02.md`：
   - 读 `in` 时，warp 内地址连续吗？
   - 写 `out` 时，相邻线程地址差多少？（提示：差 `height`）
3. 填 `notes/week02.md` **内存层次表**（Register / Shared / Global 三行即可起步）

```bash
# 可选：复习 Week1 设备参数
cd week01_basics/device_query && ./device_query
```

### 笔记模板（复制到 notes/week02.md）

```markdown
### Day1
**概念**：Global vs Shared；Coalescing = warp 连续地址
**transpose_naive**：读合并 ✓，写不合并 ✗
**问题**：
```

### 完成标准

- [ ] 能一句话解释 Coalescing
- [ ] 能解释 transpose naive 写回为何不合并
- [ ] `notes/week02.md` 有内存层次表

---

## Day2：Shared Memory + 同步

**对应 Step**：03  
**预计时间**：3–4 h

### 今天要学什么

- [ ] `__shared__`：Block 内共享，物理在 SM 上，比 Global 快得多
- [ ] Shared 容量：T4 每 block 最多 **48 KB**（`sharedMemPerBlock`）
- [ ] `__syncthreads()`：block 内 barrier，全员必须到达
- [ ] 树形归约 preview：`stride /= 2` 每轮减半

### 需要看的资料

- [ ] Programming Guide **2.3** — Shared Memory 小节
- [ ] Programming Guide **5.4.4** — Synchronization Primitives（`__syncthreads`）
- [ ] 本地：`week01_basics/vec_add/vec_add.cu` → `reduce_sum`（44–65 行）
- [ ] [Week2详细步骤.md](Week2详细步骤.md) Step 03

### 动手步骤

1. 阅读 `reduce_sum`，逐段注释：
   - `sdata[threadIdx.x] = input[...]` — load 阶段
   - `for (stride = blockDim.x/2; ...)` — 树形归约
   - `if (threadIdx.x == 0) blocksums[...]` — 写 block 结果
2. 编译运行 vec_add（确认 reduce 测试仍 PASS）：

```bash
cd week01_basics/vec_add && make && ./vec_add 1048576
```

3. 在纸上画 8 线程版：`8→4→2→1` 树形图

### 笔记模板

```markdown
### Day2
**概念**：__shared__ + __syncthreads；reduce 每轮 stride/=2
**问题**：
```

### 完成标准

- [ ] 能解释为何每轮归约后要 `__syncthreads()`
- [ ] 能区分 Shared 与 Global 的作用域
- [ ] 看懂 `reduce_sum` 整体流程（不必独立重写）

---

## Day3：transpose — naive 与 shared_simple

**对应 Step**：04–05  
**预计时间**：3 h（若三版已 PASS，可压缩到 2 h 复习）

### 今天要学什么

- [ ] 2D 映射复习：`x→col`，`y→row`，`matrix[row*width+col]`
- [ ] 转置下标：`out[col*height+row] = in[row*width+col]`
- [ ] Shared 中转：Global → tile → Global（shared_simple 同线程读/写）
- [ ] shared_simple：**读合并、写仍不合并**（与 naive 对比）

### 需要看的资料

- [ ] Programming Guide **2.3** — Matrix Transpose Example
- [ ] [转置优化详解.md](../week02_memory/transpose/转置优化详解.md) § 一–§ 四
- [ ] [transpose/README.md](../week02_memory/transpose/README.md)

### 动手步骤

```bash
cd week02_memory/transpose
make && ./transpose 512
make && ./transpose 1024 768    # 非方阵
```

- [ ] 确认 `transpose_naive` PASS
- [ ] 确认 `transpose_shared_simple` PASS
- [ ] 阅读 [矩阵转置两层转置图解.md](../week02_memory/transpose/矩阵转置两层转置图解.md)（若有）
- [ ] 在 `notes/week02.md` 用 4×4 例子手算一个元素转置下标

**若代码已全部 PASS**：本日重点改为 **读文档 + 手算**，不必重写 kernel。

### 笔记模板

```markdown
### Day3
**手算**：in[1][2] → out[2][1] 线性下标 = ?
**shared_simple 与 naive 差异**：
```

### 完成标准

- [ ] 两版 transpose PASS
- [ ] 能口述 shared 为何只是「中转」，shared_simple 写仍不合并

---

## Day4：transpose_shared 优化 + 带宽测试

**对应 Step**：06  
**预计时间**：3–4 h

### 今天要学什么

- [ ] 写回时 **x/y 对调**：让 warp 写 out 的同一行 → 写合并
- [ ] `tile[threadIdx.x][threadIdx.y]` 与坐标对调配套
- [ ] **Bank Conflict** + `tile[TILE][TILE+1]` padding
- [ ] 有效带宽：`2×W×H×4 / kernel_time / 1e9` GB/s

### 需要看的资料

- [ ] [转置优化详解.md](../week02_memory/transpose/转置优化详解.md) § 五–§ 七（全文精读）
- [ ] Programming Guide **2.3** — Bank Conflict
- [ ] 对照 `my_transpose.cu` 与 `transpose.cu` 的 `transpose_shared`

### 动手步骤

1. 验收三版 PASS：

```bash
cd week02_memory/transpose
./transpose 1024
./transpose 4096      # 测带宽用大方阵
```

2. **给 `transpose.cu` 加 kernel 计时**（参考 `mat_mul.cu` 的 cudaEvent，只包 kernel）
3. 填 `notes/week02.md` transpose 表：

| 版本 | 规模 | Kernel (ms) | GB/s |
|------|------|-------------|------|
| naive | 1024² | | |
| shared_simple | 1024² | | |
| shared | 1024² | | |
| shared | 4096² | | |

4. （可选）用自己的话在笔记里解释「写回为何用 `threadIdx.x` 填 col_out」

### 笔记模板

```markdown
### Day4
**三版带宽**：naive __ / simple __ / shared __ GB/s（1024²）
**bank conflict + padding**：
```

### 完成标准

- [ ] 三版 PASS
- [ ] 至少 1024² 三版有 kernel 时间和 GB/s
- [ ] shared 版带宽明显高于 naive（大方阵上应可见）

---

## Day5：reduction 入门 + 单 block 树形归约

**对应 Step**：07–08  
**预计时间**：3–4 h

### 今天要学什么

- [ ] Reduction = 把数组归成一个值（如 sum）
- [ ] `atomicAdd` 慢：全局争用，仅作对比
- [ ] Shared 树形归约：`stride` 从 N/2 减半到 1
- [ ] 边界：`global_id >= n` 的线程 load **0**

### 需要看的资料

- [ ] Mark Harris «[Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)» — Kernel 1–4
- [ ] Programming Guide **5.4.5** — `atomicAdd`（浏览，Week 3 深入）
- [ ] 本地 `vec_add.cu` → `reduce_sum` + `test_reduce_sum`
- [ ] [Week2详细步骤.md](Week2详细步骤.md) Step 07–08

### 动手步骤

1. 创建目录：

```bash
mkdir -p week02_memory/reduction
# 创建 reduction.cu + Makefile（可从 vec_add 的 reduce_sum 迁移）
```

2. 实现顺序：
   - [ ] CPU `sum_ref` 参考
   - [ ] （可选）atomic 慢速版对比
   - [ ] `reduce_shared` 单 block，`n <= 256`，blockDim=256

3. 验收：

```bash
cd week02_memory/reduction
nvcc -O3 -arch=sm_75 -std=c++14 -o reduction reduction.cu
./reduction 256
```

### 笔记模板

```markdown
### Day5
**目录**：week02_memory/reduction/
**单 block 256 PASS**：是/否
```

### 完成标准

- [ ] `reduction/` 目录可编译
- [ ] n=256 与 CPU sum 一致
- [ ] 能画 256→128→…→1 归约树

---

## Day6：多 block reduction — 1M PASS

**对应 Step**：09  
**预计时间**：3–4 h

### 今天要学什么

- [ ] 两阶段归约：每 block 出一个部分和 → 再汇总
- [ ] Phase 1：GPU `reduce_sum<<<num_blocks, 256>>>`
- [ ] Phase 2（Week 2）：CPU 对 `blocksums[]` 求和即可
- [ ] grid 计算：`blocks = (n + 255) / 256`

### 需要看的资料

- [ ] Harris 文章 — 多 pass reduction 思想
- [ ] `vec_add.cu` → `test_reduce_sum` 完整流程
- [ ] [Week2详细步骤.md](Week2详细步骤.md) Step 09

### 动手步骤

1. 扩展 `reduction.cu` 支持任意 n（1M）：

```bash
./reduction 1048576
./reduction 16777216    # 可选
```

2. 检查边界：最后一 block 无效线程 load 0
3. 记录 kernel 时间（cudaEvent，只包 Phase1 kernel）
4. 填 `notes/week02.md` reduction 表

### 笔记模板

```markdown
### Day6
**1M sum**：GPU=__ CPU=__ PASS
**Kernel ms**：
```

### 完成标准

- [ ] n=1M PASS，相对误差 < 1e-4
- [ ] 理解 Phase2 为何 Week 2 可在 CPU 做

---

## Day7：性能定稿 + 复盘

**对应 Step**：10–12  
**预计时间**：3 h

### 今天要学什么

- [ ] 汇总 transpose 带宽表 + reduction 时间表
- [ ] 写 **5 句**「T4 合并访问实测结论」
- [ ] Week 2 自测 6 题（见 [Week2详细步骤.md](Week2详细步骤.md) Step 12）
- [ ] 了解 Week 3：scan、stream、atomic/shuffle

### 需要看的资料

- [ ] [T4实战指南.md](../docs/T4实战指南.md) — 转置带宽预期
- [ ] [项目清单.md](../docs/项目清单.md) — P03、P04 验收
- [ ] [CUDA学习路线图.md](../docs/CUDA学习路线图.md) — Week 3 预览

### 动手步骤

1. 每项 benchmark **跑 3 次取中位数**，更新 `notes/week02.md`
2. （选做）sanitizer：

```bash
compute-sanitizer --tool memcheck ./transpose 512
compute-sanitizer --tool memcheck ./reduction 1048576
```

3. 目录自检：

```text
week02_memory/transpose/   ✅
week02_memory/reduction/   ✅
notes/week02.md            ✅ 定稿
```

4. 写本周总结（掌握 / 薄弱 / 下周重点）

### 自测题（闭卷）

1. Global vs Shared 区别？
2. Coalescing 是什么？
3. `__syncthreads()` 作用与坑？
4. Bank conflict 与 padding？
5. 树形归约 stride 为何 `/=2`？
6. 两阶段 reduction 流程？

### 完成标准

- [ ] `notes/week02.md` 性能表完整 + 5 句结论 + 本周总结
- [ ] transpose + reduction 均可 `make` PASS
- [ ] 自测 ≥ 4/6
- [ ] 知道 Week 3 读 **2.5 Stream** + **5.4.5/5.4.6**

---

## 每日时间建议（业余 3–4h/天）

| Day | 阅读 | 编码 | 笔记 |
|-----|------|------|------|
| 1 | 1.5h | 0.5h | 1h |
| 2 | 1h | 1h | 1h |
| 3 | 1.5h | 1h | 0.5h |
| 4 | 1h | 2h | 1h |
| 5 | 1h | 2h | 0.5h |
| 6 | 0.5h | 2.5h | 0.5h |
| 7 | 0.5h | 1h | 1.5h |

---

## 与 Week 1 未完成项的关系

| Week 1 遗留 | 建议 |
|-------------|------|
| Occupancy block 对比表未填 | Day7 前补 30 min，或并入 Day7 复盘 |
| Week 1 自测未做 | 不挡 Week 2；Day7 一并复习 |

---

**返回**：[Week2详细步骤.md](Week2详细步骤.md) · [CUDA学习路线图.md](../docs/CUDA学习路线图.md)
