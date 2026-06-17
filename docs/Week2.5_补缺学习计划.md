# Week2.5 补缺学习计划（进入 Week3 前）

> **目的**：在开始 Week3 前，补齐卷一、卷二的几个缺口，让基础完整。
> **预计用时**：半天到一天（约 4–6 小时）。不求新内容，只补"没系统学/半懂/没自测"的部分。
> **原则**：你缺的这几章我都深度扩写过，直接读正文即可，重点是**理解 + 自测**，不是从零学。

---

## 你的缺口诊断（来自清点）

```text
卷一：✅ 实质学完
  ⚠️ 01 CPU vs GPU（零散懂，没系统读）
  ❌ 06 复习面试题（没自测）

卷二：✅ 大部分学完，真缺口：
  ❌ 06 NVCC/PTX 编译流程   ← 最大空白，优先补
  ⚠️ 03 函数修饰符（半懂：__host__ __device__、为什么 void）
  ⚠️ 05 异步执行（半懂，但卷七已超前补过 stream）
  ❌ 10 复习面试题（没自测）
```

---

## 半天计划（4 个时段）

### 时段 1（约 90 分钟）⭐ 重点：卷二 06 NVCC/PTX 编译流程

这是你最大的真空白，且面试常考。好消息：**你天天在用编译命令，只是没理解背后发生了什么**。

**读**：[卷二第 06 章 NVCC、PTX 与编译流程](../cuda_deep_course/course/volume02_programming_model/06_NVCC_PTX与编译流程.md)

**要搞懂的 5 个问题**（读完能口述就算掌握）：
```text
1. .cu 文件怎么被拆成 Host 代码和 Device 代码两条工具链？
2. PTX 和 SASS 区别？为什么要有 PTX 这个"虚拟中间层"？
3. compute_75（虚拟架构）vs sm_75（真实架构）区别？
4. fat binary 是什么？为什么生产构建同时放 cubin + PTX？
5. -arch=sm_75 到底告诉编译器什么？
```

**动手**（把抽象的编译流程变具体）：
```bash
# 1. 看 PTX（虚拟指令）
cd week01_basics/vec_add   # 或任意一个 .cu
nvcc --ptx -arch=sm_75 vec_add.cu -o vec_add.ptx
sed -n '1,60p' vec_add.ptx     # 读读 PTX 长什么样

# 2. 看寄存器/spill（你 local memory demo 用过这个！现在理解它属于编译流程）
nvcc -Xptxas=-v -arch=sm_75 vec_add.cu -o /dev/null

# 3. 看真实机器码 SASS
nvcc -arch=sm_75 vec_add.cu -o vec_add
cuobjdump --dump-sass vec_add | head -40
```

**完成标准**：
- [ ] 能口述 5 个问题
- [ ] 看过自己 kernel 的 PTX 和 SASS，知道它们不是一回事
- [ ] 理解你之前用的 `-Xptxas=-v`、`-arch=sm_75` 在编译流程里的位置

---

### 时段 2（约 45 分钟）：卷二 03 函数修饰符（补"为什么"）

你会用 `__global__`/`__device__`，但几个"为什么"没搞清。

**读**：[卷二第 03 章 CUDA 函数修饰符与执行空间](../cuda_deep_course/course/volume02_programming_model/03_CUDA函数修饰符与执行空间.md)（重点 §1 分流图、§2.1 为什么 void、§2.2 三尖括号、§5 双编译）

**要搞懂**：
```text
1. 为什么 kernel 返回值必须是 void？（异步 + 多线程，没法返回）
2. <<<grid, block>>> 编译成了什么？（一串 runtime 调用）
3. __host__ __device__ 生成几份代码？（两份，走两条工具链）
4. 为什么双修饰函数里不能写 std::cout？
```

**完成标准**：
- [ ] 能解释"为什么 void"（不是死记，是理解异步的必然结果）
- [ ] 能说出 `__host__ __device__` 是两份独立机器码

---

### 时段 3（约 45 分钟）：卷二 05 异步执行（快速过，你已超前）

你卷七已经深入学过 stream/重叠，这章是它的基础，**快速对齐概念即可**。

**读**：[卷二第 05 章 异步执行、同步与错误模型](../cuda_deep_course/course/volume02_programming_model/05_异步执行_同步与错误模型.md)（重点 §1 传送带模型、§3 两类错误）

**要搞懂**：
```text
1. launch 是异步的"传送带"模型（你卷七学过，复习）
2. launch error（提交时抓）vs execution error（同步时抓）—— 何时暴露
3. 为什么 CUDA_CHECK 要"launch 后查一次 + 同步后再查一次"
```

**完成标准**：
- [ ] 能解释两类错误为什么在不同时刻暴露

---

### 时段 4（约 60 分钟）⭐ 自测：卷一 06 + 卷二 10 面试题

这是检验你前面是否真懂的关键。**先自己答，再看答案**，答错的回头补。

**做**：
- [卷一第 06 章 复习与面试题](../cuda_deep_course/course/volume01_gpu_basics/06_卷一复习与面试题.md)
- [卷二第 10 章 复习、练习答案与面试题](../cuda_deep_course/course/volume02_programming_model/10_卷二复习_练习答案与面试题.md)

**做法**：
```text
1. 盖住答案，自己先口头/纸上答每道题
2. 答不出或答错的 → 标记 → 回到对应章节重读
3. 重点检查这些高频点：
   - SM/warp/block/thread 关系
   - occupancy 是什么、受什么限制
   - 为什么 GPU 要大量线程（延迟隐藏）
   - 合并访问、内存层次
   - 两类 CUDA 错误
   - PTX vs SASS
```

**完成标准**：
- [ ] 两卷面试题都过一遍
- [ ] 答错的回头补了对应章节

---

## 可选（如果还有时间）：卷一 01 系统读

[卷一第 01 章 CPU 与 GPU 为什么不同](../cuda_deep_course/course/volume01_gpu_basics/01_CPU与GPU为什么不同.md)

你零散懂这些，但没系统读过"设计哲学"。15 分钟扫一遍，建立完整框架。重点：
```text
- CPU 低延迟 vs GPU 高吞吐的设计取舍
- 延迟 vs 吞吐（跑车 vs 货运列车的类比）
- GPU 加速的额外成本（H2D/launch/D2H 开销）
- 什么问题适合 GPU
```

---

## 一天计划（如果想更扎实）

把半天计划拉长，每个时段后在 `notes/` 记录，并加一个"动手巩固"：

```text
上午：时段 1（NVCC/PTX）+ 时段 2（函数修饰符）
下午：时段 3（异步）+ 时段 4（自测）+ 卷一 01
收尾：写一段"卷一卷二我现在完整掌握了什么"的总结，标记仍模糊的点
```

---

## 完成后的检查清单（进入 Week3 的门槛）

```text
[ ] 能口述 .cu → PTX → SASS 的编译流程，知道 -arch 的作用
[ ] 能解释 PTX（虚拟）vs SASS（真实）、fat binary
[ ] 能解释为什么 kernel 返回 void、__host__ __device__ 是两份代码
[ ] 能解释 launch error vs execution error 何时暴露
[ ] 卷一、卷二面试题自测过，答错的补了
[ ] （可选）系统读过 CPU vs GPU 设计哲学
```

全部勾上 → 卷一、卷二真正闭环，可以放心进 Week3。

---

## 附录：卷三/卷四散落主题补读清单（你可能没仔细读）

> **背景**：这些主题**正文里其实都写过**，但分散在卷三/卷四不同章节，容易翻漏或只是
> 扫过。这里集中列出"该读哪一节 + 自测问题"，按需补，不必一次读完。和上面的 Week2.5
> 主线（卷一/卷二）分开，时间充裕再看。

### A. 内存传输三件套（pinned / zero-copy / Unified）⭐ 面试高频

**读**：[卷三第 04 章 §4–7](../cuda_deep_course/course/volume03_memory_system/04_Cache_Host传输与Unified_Memory.md)

```text
1. 为什么 pinned 比 pageable 快？（省掉驱动内部那次 staging 拷贝）
2. 为什么 cudaMemcpyAsync 必须用 pinned 才能真重叠？（DMA 只能搬不被换出的页）
3. zero-copy/mapped 何时用、离散 GPU 为什么要慎用？（PCIe 随机读慢）
4. Unified Memory 的性能陷阱是什么？（按需页迁移，CPU/GPU ping-pong）
5. prefetch / advise 各解决什么？
```
> 注：§6 zero-copy 正文偏薄（只有定性几行），知道"是什么、何时用、离散卡慎用"即可。

**完成标准**：
- [ ] 能口述 pinned 为什么快、为什么是异步重叠的前提
- [ ] 能说出 Unified Memory 慢的原因和两个补救手段（prefetch/advise）

### B. 数据复用的三个层次 ⭐ 贯穿全书的主线

**读**：[卷三第 04 章 §3](../cuda_deep_course/course/volume03_memory_system/04_Cache_Host传输与Unified_Memory.md) + [卷六第 01 章 §见 AI 推导](../cuda_deep_course/course/volume06_operators/01_GEMM数学_布局与性能上限.md)

```text
1. 三层复用：register（线程内）/ shared（block 内）/ L2 或重读（跨 block）
2. 数据复用如何提升 arithmetic intensity，把 kernel 从带宽墙推向算力墙？
3. GEMM tiling 为什么能把 AI 从 0.25 抬上去？
```
**完成标准**：
- [ ] 能用一句话把"数据复用 → 提升 AI → 逃离带宽墙"串起来

### C. Histogram（直方图）⭐ 经典手写题

**读**：[卷四第 04 章 §5–7](../cuda_deep_course/course/volume04_parallel_algorithms/04_Scan与Histogram.md)

```text
1. naive global atomic 的问题？（分布集中时大量线程争少数 bin）
2. privatization（分层聚合）怎么降竞争？（block 私有 shared 直方图 → 合并）
3. warp aggregation 何时有益、何时反而更慢？（重复 key 有益，随机 key 增指令）
```
**完成标准**：
- [ ] 能画出 global → block-private shared → 合并 的三段式
- [ ] 能解释 privatization 把 global atomic 次数从"每元素一次"降到"每 block 每 bin 至多一次"

### D. Scan → Compact（流压缩）⭐ 面试常考的 scan 应用

**读**：[卷四第 04 章 §1–4](../cuda_deep_course/course/volume04_parallel_algorithms/04_Scan与Histogram.md)

```text
1. Hillis-Steele vs Blelloch：工作量差约 10 倍，深度同为 O(log n)，各自适用场景？
2. exclusive scan 的输出为什么正好是"每个保留元素的目标下标"？
3. stream compaction = 标记(0/1) → exclusive scan 求位置 → 按位置搬运
```
> 注：正文把 compact 当作 scan 的"应用"一句带过，没落成完整代码。理解"scan 求位置"
> 这一步即可，想动手可自己补一个单 block compact。

**完成标准**：
- [ ] 能解释为什么 exclusive scan 的结果就是 compact 的写入位置

### E. 只读内存路径（constant / texture）+ pitch（小补丁）

**读**：[卷三第 01 章 §6–7](../cuda_deep_course/course/volume03_memory_system/01_CUDA内存空间.md)、[卷三第 02 章 §7 Pitch](../cuda_deep_course/course/volume03_memory_system/02_合并访问_对齐_AoS与SoA.md)

```text
1. constant memory 何时高效？（一个 warp 读同一地址 → 广播；读散地址则退化）
2. texture/read-only 路径的价值在寻址与空间局部性，不要因"cache"名字默认更快
3. pitch 为什么常 ≠ width*sizeof(T)？（每行对齐，保证逐行合并访问）
```
**完成标准**：
- [ ] 能说出 constant memory 的"广播"前提
- [ ] 知道 `cudaMallocPitch` 为什么存在

---

### 这份附录的优先级建议

```text
先读（面试高频、收益大）： A 传输三件套、C Histogram、D Scan→Compact
顺带读（贯穿主线）：       B 数据复用
查漏即可（小点）：         E constant/texture/pitch

仍未写、属于后续计划（现在不用管）：
  - 卷六 im2col 卷积独立章、CUTLASS 思想章
  - 卷七剩余章（cudaMallocAsync、UM prefetch、cp.async/TMA）
  - 卷八/九/十（HPC 多 GPU、硬件架构、工程面试）
```

> 同样的纪律：这是**补读**，不是重学。每个主题能口述自测问题就过，别又无限下挖。

---

## 附录动手时段：A / C / D 三块的实操（按需做）

> 上面附录是"读 + 自测"，这里把 **A / C / D** 三块配上**可运行命令或最小手写任务**，
> 让概念落到代码上。每块约 30–45 分钟，做完打勾。
> 说明：仓库里**已有** reduction / transpose / unified_memory / stream 的实验；
> **没有** histogram / scan 的现成 lab，所以 C/D 用"自己写最小版 + CPU 验证"的方式
> （正好符合教材"先手写"的要求），不依赖不存在的命令。

### 动手 A：亲眼看到 pinned 比 pageable 快 ⭐

用仓库里现成的 demo，对比 pinned 与 pageable 的传输/重叠差异。

```bash
cd /home/qichengjie/workspace/cuda_study

# 1. Unified Memory：观察有无 prefetch 的差异（page migration 陷阱）
ls week02_memory/unified_memory_demo/
cat week02_memory/unified_memory_demo/unified_memory_demo.cu | sed -n '1,40p'
# 按其 README/Makefile 编译运行，关注 prefetch 前后耗时

# 2. Stream + pinned：观察传输与计算重叠
ls week02_memory/stream/
nvcc -arch=sm_75 week02_memory/stream/stream.cu -o /tmp/stream_demo && /tmp/stream_demo
```

如果想要"串行 vs pinned 重叠"的完整实测对比，卷七第 02 章配的 demo 更系统：

```bash
# 卷七的重叠实验（README 里有 99ms 串行 vs 34ms 重叠的实测）
sed -n '1,60p' cuda_deep_course/course/volume07_async_system/02_传输与计算重叠.md
```

**完成标准**：
- [ ] 跑过至少一个 demo，看到 pinned/重叠或 prefetch 带来的时间差
- [ ] 能解释这个时间差对应附录 A 的哪条原理

### 动手 C：手写一个最小 Histogram（global → privatization）⭐

没有现成 lab，自己写一个最小版，重点体会**两版的竞争差异**。建议放到
`week02_memory/histogram_demo/`（或任意目录）：

```cpp
// 目标：统计 N 个 [0, BINS) 的整数落到各 bin
// 版本 1：每个线程直接对 global 直方图 atomicAdd
__global__ void histGlobal(const int* in, int* hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&hist[in[i]], 1);
}

// 版本 2：privatization —— 每个 block 先在 shared 里统计，最后合并到 global
__global__ void histShared(const int* in, int* hist, int n, int bins) {
    extern __shared__ int local[];                 // bins 个
    for (int b = threadIdx.x; b < bins; b += blockDim.x) local[b] = 0;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&local[in[i]], 1);        // shared atomic，快得多
    __syncthreads();
    for (int b = threadIdx.x; b < bins; b += blockDim.x)
        atomicAdd(&hist[b], local[b]);             // 每 block 每 bin 只回写一次
}
```

```bash
# 编译运行（自己补 main：CPU reference + 两种输入分布）
nvcc -arch=sm_75 histogram_demo.cu -o /tmp/hist && /tmp/hist
```

**验证与观察**（呼应附录 C 的自测）：
```text
- 两种输入各测一次：均匀分布  vs  90% 集中在一个 bin
- 集中分布下：版本 1 慢很多（全员争一个 global bin），版本 2 受影响小
- 必须和 CPU 串行统计结果逐 bin 比对，确认两版都正确
```

**完成标准**：
- [ ] 两版都与 CPU 结果一致
- [ ] 在"集中分布"输入上观察到版本 2 明显更快，并能解释为什么

### 动手 D：手写单 block Scan，并用它做 Compact ⭐

同样没有现成 lab，写一个**单 block exclusive scan**，再用它实现 stream compaction。

```cpp
// 单 block exclusive scan（Hillis-Steele，简单版；理解为主，不追求 work-efficient）
__global__ void scanExclusive(const int* in, int* out, int n) {
    extern __shared__ int tmp[];
    int t = threadIdx.x;
    tmp[t] = (t > 0 && t < n) ? in[t - 1] : 0;     // 右移一位 = exclusive
    __syncthreads();
    for (int off = 1; off < n; off <<= 1) {
        int v = (t >= off) ? tmp[t - off] : 0;
        __syncthreads();
        tmp[t] += v;
        __syncthreads();
    }
    if (t < n) out[t] = tmp[t];
}
```

Compact 的三步（这是面试要点，务必能复述）：
```text
1. 标记：flag[i] = keep(in[i]) ? 1 : 0
2. 求位置：pos = exclusive_scan(flag)      <- pos[i] 就是第 i 个保留元素的目标下标
3. 搬运：if (flag[i]) out[pos[i]] = in[i]
```

```bash
# 先用小数组手工核对，再写 main 验证
# 例：in = [3,0,5,0,0,7,2,0]，keep = 非零
#   flag = [1,0,1,0,0,1,1,0]
#   scan = [0,1,1,2,2,2,3,4]   <- 保留元素依次写到下标 0,1,2,3
#   out  = [3,5,7,2]
nvcc -arch=sm_75 scan_compact_demo.cu -o /tmp/scan && /tmp/scan
```

**完成标准**：
- [ ] 手工核对过上面那个 8 元素例子，scan 结果与表一致
- [ ] 能解释"为什么 exclusive scan 的结果正好是 compact 的写入位置"
- [ ] 代码输出与 CPU reference 一致

---

### 做完 A/C/D 后你应该能

```text
[ ] 看到过 pinned/重叠/prefetch 的真实时间差，对应到原理
[ ] 手写过 histogram 两版，理解 privatization 为什么降竞争
[ ] 手写过 scan，并能用它解释 stream compaction 的三步
```

这三块是 scan/reduce/histogram 这一类**分层并行 + 原子聚合**算法的核心，面试手写题
高频。做完即可放心进 Week3 / 卷四后续。

---

## 重要提醒

```text
- 这是【补缺】不是【从头学】：你大部分内容懂，只补空白和半懂的，别推倒重来。
- 不要又"无限下挖"：遇到 NVCC 的高级选项、PTX 汇编细节，记下来跳过，不是现在的事。
- 自测最重要：能答出面试题 > 读了多少。答不出的才是真缺口。
- 半天够了：别拖成两三天。补缺是为了进 Week3，不是停在这里。
```
