# CUDA 面试八股 · 追问答案速查（续篇：第 8-15 章）

> 配套：[CUDA面试八股全集.md](CUDA面试八股全集.md)、[追问答案第1-7章](CUDA面试八股_追问答案.md)
> 本篇覆盖第 8-15 章：Tensor Core / cp.async / PTX-SASS / profiling / 调试 / 多GPU / Hopper / 手写题与模拟面试。

---

# 第 8 章：Tensor Core 与底层 MMA

## Q23 TensorCore/WMMA/MMA/HMMA 的追问

**1. FP16 input + FP32 accumulate 的误差来自哪里？**
- 误差主要来自 **FP16 输入本身的表示精度**（10 位尾数，动态范围有限，大值/小值易舍入）。而**累加用 FP32** 恰恰是为了减少误差——长链累加若也用 FP16 会迅速丢精度。所以现代 Tensor Core 都是"低精度乘 + 高精度累加"。误差来自输入量化，不来自累加。

**2. 为什么 Tensor Core kernel 仍可能 memory-bound？**
- Tensor Core 把"算"做得极快，但如果**喂数据跟不上**（global→shared→fragment 的搬运带宽不够、或 shared bank conflict、或 `ldmatrix` 供数不足），算力单元就闲着等数据。你 week06 看到 Tensor pipe 只 ~5.7% active，就是"算得快但喂不饱"→ 瓶颈在供数不在算。

## Q24 ldmatrix/fragment 的追问

**1. `ldmatrix` 是 global→shared 吗？**
- **不是**。`ldmatrix` 是 **shared→寄存器(fragment)**，把 shared 里的一块 tile 协作加载成符合 MMA lane/register 布局的 fragment。global→shared 是 `cp.async`/普通 load 的事。两者是流水的不同环节。

**2. 为什么不能把一个 MMA shape 的 lane map 套到另一个 shape？**
- 不同 MMA shape（如 m16n8k8 vs m16n8k16）规定了**不同的 lane→寄存器→矩阵元素映射**。哪个 lane 持有矩阵哪几个元素是硬件固定死的，且随 shape/dtype/layout 变。套错映射 → 读到错误元素 → 结果全错。必须查对应 shape 的官方 fragment 布局。

## Q25 CUTLASS 三级 tile 的追问

**1. 为什么 shared swizzle 要同时考虑 bank 与 `ldmatrix`？**
- swizzle（打乱 shared 存储布局）要同时满足两个约束：① 避免 warp 访问 shared 时的 **bank conflict**；② 满足 `ldmatrix` 要求的**特定 lane 访问 pattern**。两个约束互相牵制——不能只为 bank 优化而破坏 ldmatrix 需要的布局，得设计一个同时满足两者的 swizzle。

**2. epilogue 为什么也可能成为瓶颈？**
- epilogue（写回阶段：accumulator → 加 bias/激活/scale → 转精度 → 写 global）也有访存和计算。大 tile 的 accumulator 写回是大量 global store，若不做好合并/向量化，或融合的激活很重，epilogue 会占可观时间。工业 GEMM 会把 epilogue 也流水化/融合。

---

# 第 9 章：cp.async 与多级流水

## Q26 cp.async 的追问

**1. `wait_group 1` 的直觉是什么？**
- `cp.async.wait_group N` = "等到**最多只剩 N 个** group 还没完成"。`wait_group 1` 就是"保留 1 个 group 在飞、其余都等完"——用于双缓冲：当前 tile 算之前，确保它已搬完（wait 到只剩下一个正在预取的），同时让下一 tile 的预取继续在飞。`wait_group 0` = 全部等完。

**2. 尾部 tile 怎样 zero fill？**
- 当 K 不是 tile 的整数倍，最后一块不足。越界的部分不能读 global（会越界），要在 shared 里**填 0**：对越界位置跳过 cp.async 拷贝并显式写 0（或用 cp.async 的越界处理），保证矩阵乘时那些位置贡献 0，不污染结果。

## Q27 3-stage vs 2-stage 的追问

**1. producer 何时能覆盖旧 slot？**
- 环形 slot 生命周期 `FREE→COPYING→READY→CONSUMING→FREE`。producer 要覆盖一个 slot（重新往里搬新 tile），必须等它变回 **FREE**——即消费者已经**读完**那个 slot 的数据（CONSUMING 结束）。靠 barrier/mbarrier 或 wait_group 保证"不覆盖仍在被读的 slot"。

**2. 如何证明收益来自 overlap？**
- 不能只看时间变快。要用 ncu 看：① **eligible warp / issue 效率**是否提升（流水让调度器有活干）；② **long scoreboard stall 是否下降**（访存延迟被藏）；③ 对比 sync/2/3-stage 的 GFLOPS + shared + occupancy。你 week04 attention 就是用 long_scoreboard 6.4→0.03 证明 overlap 生效的。

---

# 第 10 章：PTX、SASS 与编译链

## Q28 PTX vs SASS 的追问

**1. 为什么发布包同时放 cubin 和 PTX？**
- **cubin**：针对特定架构（sm_80）编译好的机器码，能直接跑、启动快，但只对那个架构有效。**PTX**：虚拟 ISA，能被 driver **JIT** 编译到未来的新架构。同时放 = 已知架构用 cubin（快），未知/更新架构 fallback 到 PTX JIT（向前兼容）。这就是 fatbin。

**2. `compute_80` 与 `sm_80` 区别？**
- `compute_80` = **虚拟架构**，编译目标是 **PTX**（compute capability 8.0 的 PTX）。`sm_80` = **真实架构**，编译目标是 **SASS/cubin**（Ampere A100 机器码）。`-arch=compute_80 -code=sm_80,compute_80` 就是"生成 sm_80 cubin + 保留 PTX 备 JIT"。

## Q29 用汇编证明优化的追问

**1. 为什么 SASS 不能单独证明 bank conflict？**
- bank conflict 是**运行时**行为——取决于 warp 里 32 个 lane 的**实际地址**落在哪些 bank。SASS 只是静态指令（`LDS`/`STS`），看不出运行时各 lane 的地址分布。必须用 **ncu 的 bank conflict 指标**（运行时测量）才能证明，SASS 只能看有没有 shared 访问指令。

**2. 为什么 `-G` 构建不能用于性能汇编分析？**
- `-G`（device debug）会**关闭优化**、插入调试信息，生成的 SASS 和真实优化版完全不同（更多指令、无 unroll、寄存器分配不同）。拿它分析性能会得出错误结论。性能分析要用 `-O3 -lineinfo`（保留行号但不关优化）。

---

# 第 11 章：Profiling、Roofline 与 Stall

## Q30 nsys/ncu/Roofline 的追问

**1. ncu replay 时间为何不能代替正常 benchmark？**
- ncu 为了采集硬件计数器会**多次重放(replay)** kernel、插桩，严重拖慢执行（你见过 GEMV 从 0.05ms→14ms）。ncu 报的时间是插桩后的，**不是真实性能**。性能数字要用没插桩的正常运行（CUDA Event）测；ncu 只用来看**指标/瓶颈**，不看绝对时间。

**2. AI 高就必然 compute-bound 吗？**
- 不必然。AI 高只说明"每字节做的计算多"，落在 Roofline 计算受限区**的前提是其他资源没先成为墙**。可能 AI 高但卡在 **shared 带宽、bank conflict、指令吞吐、延迟、occupancy 不足**——这些 Roofline 不建模。要 ncu 看 SOL 各项才能定论。

## Q31 warp 状态/scoreboard 的追问

**1. ILP 与 TLP 怎样分别隐藏延迟？**
- **TLP（线程级并行）**：靠**更多 ready warp**——一个 warp 等待时调度别的 warp，用数量盖延迟。**ILP（指令级并行）**：靠**同一 warp 内多条独立指令**——发起多个独立 load / 多个 accumulator，让一个 warp 自己就有活干，不必切换。两者都能藏延迟，可组合。

**2. 为什么 occupancy 高仍可能 issue 不足？**
- occupancy 高 = 驻留 warp 多，但如果这些 warp **全都在 stall**（比如一起等同一类 global 访存），某一拍**没有 eligible warp**，调度器无指令可发 → issue 不足。occupancy 只保证"warp 多"，不保证"此刻有 warp 能发指令"。

## Q32 long scoreboard 的追问

**1. Short Scoreboard 常与哪些路径有关？**
- **Short scoreboard** 主要关联 **shared memory** 访问依赖（以及 MUFU/特殊函数单元等固定延迟操作）。区别于 **long scoreboard**（关联 L1TEX 路径：global/local/texture/surface 的较长延迟访存）。short = 等 shared/片上，long = 等 global/片外。

**2. 怎样形成可证伪的 stall 假设？**
- 别只说"访存慢"。要：① 定位是哪条 load（source counters / SASS 行号）；② 提出假设（如"这条 global load 未合并导致 long scoreboard"）；③ 给可验证的预测（"改合并后 long scoreboard 应下降、sector/请求数应减少"）；④ 改完用 ncu 复测证实或证伪。假设要能被数据推翻。

---

# 第 12 章：调试与工程化

## Q33 compute-sanitizer 四工具的追问

**1. racecheck 为什么主要关注 shared hazard？**
- racecheck 检测的是**同 block 内线程通过 shared memory 的数据竞争**（缺同步的读写冲突）。global 上的竞争更难静态界定（跨 block、跨 kernel），且通常靠算法协议保证；shared 的 race 有明确的"缺 `__syncthreads`"模式，是最常见也最可检测的并发 bug。

**2. 什么错误需要 cuda-gdb？**
- sanitizer 报"有非法访问/竞争"但**定位不到具体逻辑原因**时，或需要**单步、看变量、看线程状态、下断点**追踪控制流/数据流的复杂 bug（如特定线程走错分支、索引算错）。sanitizer 查"有没有错"，cuda-gdb 查"为什么错、在哪一步"。

## Q34 benchmark/回归测试的追问

**1. 浮点 kernel 为什么不能只用绝对误差？**
- 结果数值范围差异大：值很大时，正常的相对舍入误差换算成绝对误差也很大（会误判 FAIL）；值接近 0 时，绝对误差小但相对可能很大。应主要用**相对误差**（或绝对+相对结合），阈值按 fp32/fp16 累加特性设（你 GEMV K=4096 就放宽到 1e-3）。

**2. 如何避免编译器把 microbenchmark 优化掉？**
- 如果结果没被使用，编译器会把整个计算**删掉**（dead code elimination）。要：① 把结果写到 global / 用 volatile / `__syncthreads` 依赖它；② 让输入运行时才知道（不能编译期常量折叠）；③ 打印或校验部分结果。确保被测代码真的执行了。

## Q35 错误检查/RAII 的追问

**1. `cudaPeekAtLastError` 与 `cudaGetLastError` 的状态差异？**
- 两者都返回最近的错误，区别：**`cudaGetLastError`** 会**清除**错误状态（读完复位为 success）；**`cudaPeekAtLastError`** **只看不清**（保留错误状态）。kernel launch 后常用 `cudaGetLastError` 检查并清除，避免错误"粘"到下一次检查。

**2. 析构函数里如何处理 CUDA 错误？**
- 析构函数**不应抛异常**（C++ 规则，抛了可能 terminate）。所以 RAII 封装的析构里调 `cudaFree`/`cudaStreamDestroy` 时，错误应**记录日志/吞掉**，不 throw。资源释放尽力而为，错误在构造/使用阶段用返回码或异常处理。

---

# 第 13 章：多 GPU 基础

## Q36 PCIe/NVLink/P2P/NCCL 的追问

**1. P2P 不可用时可能走什么路径？**
- P2P（GPU 直接访问对端显存）不可用时，数据要**绕道 host**：GPU0 → host 内存 → GPU1（staging copy）。跨 PCIe switch、跨 NUMA、或硬件/驱动不支持 P2P 时就会这样，带宽和延迟都变差。要用设备属性 / `cudaDeviceCanAccessPeer` 先查。

**2. 为什么标称 NVLink 带宽不等于应用有效带宽？**
- 标称是**物理链路峰值**（双向、理想）。实际有效带宽受：协议开销、消息大小（小消息 latency-bound）、拓扑（是否经 NVSwitch）、争用、单双向、对齐等影响。应用能拿到的通常是峰值的一个折扣（用 nccl-tests 的 busbw 实测）。

## Q37 All-Reduce 分解的追问

**1. All-Gather 与 All-to-All 数据分布差异？**
- **All-Gather**：每个 rank 出一段，最后**所有 rank 都拿到全部段的拼接**（大家结果相同）。**All-to-All**：每个 rank 给**每个其他 rank 发不同的一段**（类似矩阵转置，各 rank 结果不同）。All-Gather 是"广播式收集"，All-to-All 是"个性化重分布"。

**2. 小消息为何可能 latency-bound？**
- 小消息时，数据传输本身很快，但每次通信的**固定开销**（启动延迟、协议握手、同步）占主导。这时增大带宽没用，瓶颈是延迟和消息数。优化靠**合并小消息**、减少通信次数，而非提高带宽。

---

# 第 14 章：Hopper

## Q38 TMA 的追问

**1. 为什么 TMA 适合 warp specialization？**
- TMA 由**少量线程**（甚至一个）用 descriptor + 坐标就能发起整块多维搬运，不需要每个线程算地址。这样可以让**少数 producer 线程/warp 专职发 TMA**，其余 consumer warp 专职算（WGMMA）。分工清晰，正是 warp specialization 的生产者-消费者模式。

**2. expected transaction bytes 有什么作用？**
- Hopper 的 mbarrier 支持"事务计数"：告诉 barrier **预期会到达多少字节**的异步拷贝。TMA 搬完的字节数累加到 barrier，等达到 expected bytes 时 barrier 才放行。这让消费者能**精确知道"数据全到齐了"**，而不是靠固定线程数计数，适配 TMA 的 bulk 异步语义。

## Q39 WGMMA/Cluster/DSM 的追问

**1. 远端 shared 所属 CTA 为什么不能提前退出？**
- DSM（分布式 shared）让 cluster 内一个 CTA 能访问**另一个 CTA 的 shared**。如果被访问的那个 CTA 提前退出、释放了 shared，访问方就读到无效内存。所以 cluster 内有生命周期保证：持有被远端访问 shared 的 CTA 必须**等到没人再访问**才能退出（cluster sync 协调）。

**2. producer/consumer warp-group 怎样分工？**
- **producer warp-group**：用 TMA 把 global 数据搬进 shared，发 mbarrier 通知。**consumer warp-group**：等 mbarrier，用 WGMMA 从 shared 算矩阵乘，算完通知 producer 可覆盖。两组通过 mbarrier + stage 环形缓冲协作，形成深流水——这是 Hopper GEMM/FlashAttention-3 的核心范式。

## Q40 persistent kernel/warp specialization 的追问

**1. 常驻 CTA 数如何选择？**
- 选"**恰好占满所有 SM 的可驻留上限**"——即 `SM 数 × 每 SM 能驻留的 CTA 数`（受 register/shared/threads 限制）。太多会超出无法全驻留（退化）；太少填不满 GPU。要用 occupancy API 算出每 SM 能驻留几个，再乘 SM 数。

**2. 动态任务队列怎样避免热点 atomic？**
- 全局一个 atomic 计数器发任务会成热点。缓解：① 每个 CTA 用 atomicAdd **批量领取**一批任务（减少 atomic 次数）；② 分片队列（每个 SM/CTA 一个本地队列 + work stealing）；③ warp-aggregated 领取。核心是降低对单一 atomic 地址的争用频率。

---

# 第 15 章：手写题与模拟面试（关键点速记）

## 15.2 高频手写题的"必须写对"要点

| 题目 | 一定不能错的点 | 加分 |
|------|--------------|------|
| **vector add** | grid-stride 循环、`if(i<n)` 边界、launch 后 `cudaGetLastError` | float4 + K%4 尾部 + 16B 对齐 |
| **reduction** | 寄存器局部累加 → warp shuffle(偏移16→1) → 跨 warp 走 shared → 精确 mask | 两阶段(第二 kernel/atomic)、说清浮点非结合 |
| **scan** | 分清 inclusive/exclusive、warp 内 shfl_up、warp sums 再扫、偏移加回 | 多 block 递归、非整除边界 |
| **transpose** | shared 中转(合并读+合并写)、`[32][33]` padding 避 bank conflict | 非整除 tile、报 GB/s |
| **histogram** | bin 边界、shared privatize + `__syncthreads` + 合并到 global | warp-private、分布相关性 |
| **tiled GEMM** | 行列索引、tile 协作 load、`__syncthreads`、K 尾部处理 | register tiling、算 AI |
| **WMMA 骨架** | fragment 声明、layout、leading dimension、`fill_fragment` 清零 | SASS 看 HMMA 证明 |

## 15.3 模拟面试答题框架

**通用套路（B/A/C 都适用）**：
> 先给**结论**（15 秒）→ 再给**条件和边界**（什么情况成立/不成立）→ 最后用**代码/profiler 证据**支撑。

**避免的话术**（会被扣分）：
- "永远/一定/默认都这样" 回答架构问题（scheduler 数、occupancy 阈值都随架构变）
- 只报时间不报资源代价（reg/shared/occupancy）
- 只测一个大方阵就说通用性能
- grep 到一条 float4/HMMA 就说 kernel 优化好了

**A 档(性能岗)高频链条**：
> 给个慢 kernel → 先 benchmark 确认问题 → Roofline 定位 compute/memory → nsys 看大时间块/launch → ncu 看 SOL/stall/occupancy → 提**可证伪假设** → 改 → 复测。这套"假设→profile→修复→复测"能讲顺，A 档就稳。

**你的项目证据库（面试直接引用）**：
- GEMV：合并访问 72→1360 GB/s（19×）、float4 只 +4% 因已 memory-bound（DRAM 72%）
- attention pipelined：long_scoreboard 6.4→0.03 证明 cp.async 藏延迟
- GEMM：bank conflict 4.8→1.5 way、register tiling 8×4 甜点
- reduction：追平 CUB ~91% 带宽

## 15.4 复习清单（自检）
- [ ] B 题 15 秒无硬伤结论
- [ ] A 题 1 分钟证据链 + 一个反例
- [ ] C 题分清"读过源码/架构理解" vs "有实测"
- [ ] 闭卷写出 reduction / transpose / tiled GEMM
- [ ] 讲完 A100 GEMM 的"假设→profile→修复→复测"
- [ ] 不用固定 scheduler 数 / occupancy 阈值等跨架构绝对话术

---

> 全 40 题追问 + 手写题 + 模拟面试要点整理完毕（分第1-7章、第8-15章两篇）。
> 复习建议：先合上答案自己答，卡住的地方标记出来重点记——卡住处就是你的薄弱点。
