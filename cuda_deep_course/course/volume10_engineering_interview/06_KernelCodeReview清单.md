# 06 Kernel Code Review 清单

## 0. 先建立大局观：review 不是挑刺，是系统化排雷

写完一个 kernel"看起来能跑"，离"可以合并进项目"还有距离。CUDA 的 bug 有固定的"高发区"——
正确性靠同步、性能靠访存、健壮性靠边界。**code review 清单**就是把这些高发区列成可逐条对照
的检查表，让你（和 reviewer）不靠记忆、不靠运气，系统化地排雷。

本章是前九卷的**收口**：把散落各卷的"坑"汇成一份能照着审的清单。每一条都标注它来自哪一卷，
方便回查原理。

```text
review 四个维度（按重要性）：
  ① 正确性  —— 错了再快也是 0 分
  ② 访存    —— GPU 性能的命门
  ③ 资源    —— occupancy、寄存器、shared
  ④ 工程    —— 错误检查、可读性、可测试
```

## 0.1 怎么用这份清单

```text
- 自审：写完 kernel 后逐条过一遍，可疑的就回对应卷复习
- 互审：review 别人代码时按维度提问
- 不必每条都满足：根据 kernel 类型取舍（如纯逐元素 kernel 不涉及 shared）
```

## 1. 正确性维度（错了一切归零）

```text
□ 边界判断：if (idx < n) 了吗？非整除/非方阵规模会越界吗？        (卷二/08, 卷十/03)
□ 索引计算：row*width+col 有没有写反 width/height？               (卷二/02)
□ 同步充分：跨线程读写 shared 之间有 __syncthreads() 吗？         (卷三/03, 卷四/01)
□ barrier 一致：__syncthreads() 在所有线程都到得了的位置吗？      (卷四/01)
   （没放进 if(threadIdx...) 这种分支里）
□ warp 同步：依赖 warp 内协作时用了 __syncwarp / _sync 原语吗？   (卷四/01,02)
   （没靠"warp 天然锁步"省略同步）
□ race：有没有多线程无保护地写同一地址？需要 atomic 吗？         (卷四/01,02)
□ 输出初始化：累加型输出（如 atomicAdd 目标）初始化为 0 了吗？
□ 数值：浮点比较用容差不是 ==？reference 用 double？             (卷四/06, 卷十/03)
```

> 重点：**边界和同步是两大高发区**。一个在整除规模 PASS 的 kernel，先怀疑非整除规模；一个
> "偶尔出错"的 kernel，先怀疑漏同步。

## 2. 访存维度（GPU 性能命门）

```text
□ 合并访问：一个 warp 的 32 条 lane 访问连续地址吗？             (卷三/02)
   （threadIdx.x 对应最内层连续维度）
□ 跨步陷阱：有没有 input[tid * stride] 这种跨步访问？            (卷三/02)
□ AoS/SoA：只用部分字段时用了 SoA 吗？（避免 AoS 跨步）          (卷三/02)
□ 数据复用：反复用的数据搬进 shared 了吗？还是每次重读 global？  (卷三/03, 卷六)
□ bank conflict：shared 的列访问加 padding 了吗？(如 [32][33])   (卷三/03)
□ 只读广播：所有线程读同一小数据，考虑 constant memory 了吗？    (卷三/01)
□ 向量化：合并已满足后，float4 能减少指令数吗？（锦上添花）      (卷三/02)
```

> 顺序很重要：**先保证合并访问，再谈 shared/向量化**。布局错了，向量化和 shared 都救不回来。

## 3. 资源与并发维度

```text
□ 寄存器：用 -Xptxas=-v 看每线程寄存器数，有 spill 吗？          (卷二/06, 卷五/03)
□ occupancy：block size 是 32 的倍数吗？资源是否限制了驻留？     (卷一, 卷五/03)
□ shared 用量：每 block 的 shared 是否过大、限制了驻留 block？   (卷三/03, 卷五/03)
□ divergence：warp 内有数据相关的 if/else 分支吗？能否消除？     (卷五/03)
□ 占满 GPU：grid 够大吗？grid-stride loop 处理超大输入了吗？     (卷二/02, 卷四/03)
□ 异步：传输和计算能重叠吗？用了 stream / pinned 吗？           (卷七)
```

## 4. 工程维度

```text
□ 错误检查：每个 CUDA API 都 CUDA_CHECK 了吗？                   (卷十/02)
□ kernel 错误：launch 后 cudaGetLastError + 同步后查 execution？ (卷二/05, 卷十/02)
□ 资源管理：用 RAII 管显存/stream，没有裸 malloc/free 漏？       (卷十/02)
□ 可测试：有 CPU reference + 多规模测试吗？                      (卷十/03)
□ 可读性：magic number 抽成常量？kernel 职责单一？
□ 注释：复杂索引/同步逻辑有解释"为什么"的注释吗？
□ 调试残留：没留 -G 编译、没留 printf 调试、没留 deviceSync 热路径？
```

## 5. 一个 review 实战示例

拿一段"看起来能跑"的转置 kernel 过清单：

```cpp
__global__ void transpose(float* out, const float* in, int w, int h) {
    __shared__ float tile[32][32];                    // ⚠️ 访存②: 缺 padding -> bank conflict
    int x = blockIdx.x*32 + threadIdx.x;
    int y = blockIdx.y*32 + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y*w + x];     // ⚠️ 正确性①: 没判 x<w && y<h -> 越界
    // ⚠️ 正确性①: 缺 __syncthreads()
    out[x*h + y] = tile[threadIdx.x][threadIdx.y];    // ⚠️ 同上越界 + 读未同步的 tile
}
```

review 发现的问题（对照清单）：

```text
① 正确性：缺边界判断（非整除规模越界）、缺 __syncthreads()（读写 tile 之间 race）
② 访存：  tile[32][32] 缺 padding -> 列访问 32 路 bank conflict
```

修正后：

```cpp
__global__ void transpose(float* out, const float* in, int w, int h) {
    __shared__ float tile[32][33];                    // ✅ padding 消 bank conflict
    int x = blockIdx.x*32 + threadIdx.x;
    int y = blockIdx.y*32 + threadIdx.y;
    if (x < w && y < h)                               // ✅ 边界判断
        tile[threadIdx.y][threadIdx.x] = in[y*w + x];
    __syncthreads();                                  // ✅ 同步
    int tx = blockIdx.y*32 + threadIdx.x;
    int ty = blockIdx.x*32 + threadIdx.y;
    if (tx < h && ty < w)                             // ✅ 转置后坐标的边界判断
        out[ty*h + tx] = tile[threadIdx.x][threadIdx.y];
}
```

## 6. 实践

1. 把你 Week2 的 transpose 和 Week3 的 reduction 各过一遍本清单，记录命中的问题。
2. 找一段你早期写的 kernel，按四个维度审，看能发现几个隐患。
3. 为你的项目定制一份精简清单（删掉不相关的条目），贴在 PR 模板里。

## 7. 面试题（附参考答案）

**Q1：review 一个 CUDA kernel，你先看什么？**
先看正确性（边界判断、同步是否充分、有无 race），因为错了再快也没用；再看访存（合并访问、
shared 复用、bank conflict），这是 GPU 性能命门；最后看资源（寄存器/occupancy/divergence）和
工程（错误检查、可测试）。

**Q2：一个 kernel "偶尔结果不对"，最先怀疑什么？**
同步问题——漏 `__syncthreads()` 或依赖 warp 锁步导致的 race。race 的特征就是"时对时错、依赖
调度"。用 racecheck 验证。

**Q3：一个 kernel 在 n=1024 对、n=1000 错，问题在哪？**
边界处理。整除规模每个 block/tile 都满，掩盖了边界；非整除时最后一个不满的 block 越界或算错。
检查 `if (idx < n)` 类判断。

**Q4：shared memory 的 kernel review 重点看什么？**
读写之间有没有 `__syncthreads()`（race）、barrier 是否在所有线程都到得了的位置（死锁）、二维
tile 有没有 padding（bank conflict）、shared 用量是否过大压低 occupancy。

## 8. 资料映射

- CUDA C++ Best Practices Guide：综合优化检查项。
- 配套：卷二~卷五全部章节（本清单是它们的收口）。
