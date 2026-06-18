# 06 代际演进：Turing → Ampere → Hopper → Blackwell

## 0. 先建立大局观：为什么要懂"代际"

面试常问"你了解哪些 GPU 架构""A100 和 H100 有什么区别"。更重要的是——**新架构的新特性决定
你能不能用上某些优化**（卷二/06 讲过：用 cp.async/TMA 必须对应架构）。本章梳理 Turing 到
Blackwell 的关键演进，抓**主线趋势**而非记参数。

```text
代际演进的两条主线：
  1. AI 算力暴涨：Tensor Core 一代比一代强（精度更多、吞吐更高）
  2. 数据移动优化：从同步搬运 -> 异步搬运(cp.async) -> 硬件搬运引擎(TMA)
看懂这两条主线，比记住每代多少 SM 更有用
```

> 重要：本章只讲**官方公开**的架构特性，不涉及未公开细节或传闻。

## 0.1 架构速查表

| 架构 | 代表卡 | CC | 标志性特性 |
|---|---|---|---|
| **Turing** | T4, RTX 20 | 7.5 | 首批通用 Tensor Core、INT8 推理 |
| **Ampere** | A100, RTX 30 | 8.0/8.6 | 第三代 Tensor Core、`cp.async` 异步拷贝、BF16 |
| **Hopper** | H100 | 9.0 | 第四代 Tensor Core、TMA、thread block cluster、FP8 |
| **Blackwell** | B100/B200 | 10.x | 更强 Tensor Core、FP4/FP6、更大 NVLink（以官方为准）|

## 1. Turing（你的 T4，CC 7.5）

```text
关键特性：
- 首批把 Tensor Core 带到通用/推理场景（Volta 是数据中心首发，Turing 普及）
- 支持 INT8/INT4 推理加速
- 独立线程调度（Volta 引入，Turing 延续）—— 这就是卷四"不能依赖 warp 锁步"的来源

定位：能效高、适合推理和密集部署（70W）
你的学习载体：本教材大量实验基于 T4，sm_75
```

> 关键认知：Turing 的**独立线程调度**是 warp 编程的分水岭——之前可以依赖 warp 锁步，之后必须
> 显式 `__syncwarp`/`_sync` 原语（卷四/01、02）。

## 2. Ampere（A100，CC 8.0）

```text
关键演进：
- 第三代 Tensor Core：支持 TF32、BF16，吞吐大增
- cp.async（异步拷贝）：让数据从 global 直接异步搬到 shared，不占寄存器、能和计算重叠
  -> 这是卷七/卷九"数据移动优化"主线的关键一步
- 更大的 L2、更多 SM、HBM2e 高带宽显存
- MIG（多实例 GPU）：把一块 A100 切成多个独立小 GPU

定位：训练和高性能计算主力
```

**`cp.async` 的意义**（卷二/06 提过它是 compute_80+ 才有）：传统上 global→shared 要先 load 到
寄存器再存进 shared（占寄存器、同步）。`cp.async` 让数据**直接异步**流入 shared，省寄存器、
能和计算重叠——GEMM 等算子的双缓冲流水靠它（卷六/03）。

## 3. Hopper（H100，CC 9.0）

```text
关键演进：
- 第四代 Tensor Core + Transformer Engine：支持 FP8，专为大模型优化
- TMA（Tensor Memory Accelerator）：硬件搬运引擎，自动做大块多维数据的 global<->shared 传输
  -> "数据移动优化"主线的又一跃：从 cp.async（线程发起）到 TMA（硬件引擎自动搬）
- Thread Block Cluster：让多个 block 协作、共享数据（突破"block 间不通信"的限制）
- 分布式共享内存（DSM）：cluster 内 block 能互相访问 shared
- 更强 NVLink

定位：大模型训练/推理旗舰
```

**TMA 的意义**：以前搬大块数据要写复杂的 tiled 加载代码（卷四/05 的 halo 加载）；TMA 是专门
的硬件单元，给它描述符就自动搬，省指令、更高效。但它是 Hopper 专属，T4 用不了。

## 4. Blackwell（B100/B200，最新）

```text
关键演进（以官方公开为准）：
- 更强的 Tensor Core，支持更低精度（FP4/FP6），进一步提升 AI 吞吐
- 更大的 NVLink 带宽，更强的多卡互连
- 针对超大模型（万亿参数）的系统级优化
（具体细节随官方发布更新，不使用未公开传闻）

定位：新一代大模型基础设施
```

## 5. 两条主线趋势（比记参数更重要）

把四代串起来，抓住趋势：

### 主线一：Tensor Core / 精度演进

```text
Turing(7.5)：  FP16/INT8 Tensor Core
Ampere(8.0)：  + TF32/BF16，吞吐大增
Hopper(9.0)：  + FP8，Transformer Engine
Blackwell：    + FP4/FP6
趋势：精度越做越低（FP16->FP8->FP4），用低精度换更高吞吐，专为 AI 优化
```

### 主线二：数据移动演进

```text
传统：     global -> 寄存器 -> shared（占寄存器、同步、手动）
Ampere：   cp.async：global -> shared 异步直达（省寄存器、可重叠）
Hopper：   TMA：硬件引擎自动搬大块多维数据（省指令、更高效）
趋势：数据搬运从"线程手动"走向"硬件自动"，越来越省事省资源
```

> 这两条主线是面试讲"架构演进"的最佳框架：不要背 SM 数量，讲**"AI 算力靠 Tensor Core 低精度
> 化暴涨，数据移动靠 cp.async/TMA 硬件化"**——展示你理解趋势而非死记。

## 6. 对写代码的实际影响

```text
- 你的 T4(7.5)：能用 Tensor Core(FP16)，但没有 cp.async/TMA/cluster
- 想用 cp.async：需要 compute_80+ 编译 + Ampere+ 硬件（卷二/06 §3.1）
- 想用 TMA/cluster：需要 Hopper(9.0)
- 跨架构代码：用 __CUDA_ARCH__ 分支（卷二/06 §5.1），新架构走新路径、老架构退回通用
- 部署兼容：fat binary 打包多架构 + PTX 兜底（卷二/06、卷十/05）
```

## 7. 实践

1. 查你的 T4 的 CC（7.5），列出它**有**和**没有**的特性（对照本章表）。
2. 画两条演进主线（Tensor Core 精度、数据移动）的时间轴。
3. 解释：为什么用 `cp.async` 的代码在 T4 上编译/运行会失败？（卷二/06）
4. 准备一段 1 分钟口述："讲讲你了解的 GPU 架构演进"——用两条主线而非堆参数。

## 8. 面试题（附参考答案）

**Q1：讲讲 GPU 架构的代际演进。**
抓两条主线：① Tensor Core/精度——Turing FP16/INT8 → Ampere TF32/BF16 → Hopper FP8 → Blackwell
FP4/FP6，越做越低精度换吞吐，专攻 AI；② 数据移动——传统手动搬 → Ampere cp.async 异步直达 →
Hopper TMA 硬件引擎自动搬，越来越省资源。

**Q2：A100（Ampere）相比 Turing 的关键进步？**
第三代 Tensor Core（TF32/BF16，吞吐大增）、`cp.async` 异步拷贝（global→shared 直达、省寄存器
可重叠）、更大 L2、HBM2e、MIG 多实例。

**Q3：Hopper 的 TMA 和 cluster 解决什么？**
TMA 是硬件搬运引擎，自动做大块多维数据的 global↔shared 传输，比 cp.async 更省指令。Thread
Block Cluster 让多个 block 协作、共享数据（DSM），突破"block 间不通信"的传统限制。

**Q4：为什么 AI 算力一代比一代暴涨？**
主要靠 Tensor Core：每代支持更低精度（FP16→FP8→FP4），低精度单位时间能做更多乘加，矩阵吞吐
指数增长。AI 算力主要来自 Tensor Core 而非 CUDA Core。

**Q5：你的 T4 能用 cp.async 吗？**
不能。cp.async 需要 Ampere（compute_80）及以上，T4 是 Turing（7.5）。用了会编译/运行失败。
跨架构要用 `__CUDA_ARCH__` 分支或退回通用实现。

## 9. 资料映射

- NVIDIA 各架构白皮书（Turing/Ampere/Hopper/Blackwell）。
- CUDA Programming Guide：Compute Capabilities。
- 配套：[卷二第 06 章 NVCC/PTX](../volume02_programming_model/06_NVCC_PTX与编译流程.md)、[卷九第 03 章 执行单元](03_执行单元_CUDACore_TensorCore_SFU_LSU.md)。
