# 卷九：GPU 硬件架构深入与代际演进

> 前八卷你一直站在"软件"视角写 CUDA。卷九把视角彻底翻到"硬件"——GPU 芯片内部到底长什么样、
> warp 怎么被调度、各种执行单元和存储层次的物理结构、为什么不同代架构性能不同。这一卷把
> 前面所有"为什么这样写更快"的答案落到硬件根源上。

## 学习目标

- 看懂 GPU 的逻辑层次：die → GPC → TPC → SM，知道 SM 内部有什么。
- 理解 warp scheduler 怎么发射指令、scoreboard 怎么追踪依赖、延迟怎么被隐藏。
- 区分 CUDA Core / Tensor Core / SFU / Load-Store Unit 各干什么。
- 把存储层次（寄存器/shared/L1/L2/显存）对应到物理硬件和带宽延迟。
- 理解功耗、频率、散热如何影响持续性能。
- 看懂 Turing → Ampere → Hopper → Blackwell 的关键演进。
- 会读规格表、Compute Capability、白皮书和 SASS。

## 为什么需要这一卷（一张图）

```text
前八卷：合并访问快、shared 比 global 快、warp divergence 慢、occupancy……
        ↑ 你知道"怎么做"和"为什么"（软件层面）

卷九：  这些"为什么"的硬件根源 ——
        为什么 warp 是 32？因为 scheduler 一拍发射一个 warp 的指令
        为什么要很多 warp？因为要填满执行单元的流水线延迟
        为什么 shared 快？因为它是 SM 片上的物理 SRAM
        -> 打通软件优化和硬件结构，理解才真正完整
```

## 章节

1. [GPU 逻辑层次：die、GPC、TPC、SM](01_GPU逻辑层次_die_GPC_TPC_SM.md)
2. [Warp 调度：scheduler、scoreboard 与延迟隐藏](02_Warp调度_scheduler_scoreboard与延迟隐藏.md)
3. [执行单元：CUDA Core、Tensor Core、SFU、LSU](03_执行单元_CUDACore_TensorCore_SFU_LSU.md)
4. [存储层次硬件：寄存器、shared、L1/L2、显存](04_存储层次硬件_寄存器_shared_L1L2_显存.md)
5. [功耗、频率、散热与持续性能](05_功耗_频率_散热与持续性能.md)
6. [代际演进：Turing → Ampere → Hopper → Blackwell](06_代际演进_Turing_Ampere_Hopper_Blackwell.md)
7. [读规格表、Compute Capability、白皮书与 SASS](07_读规格表_ComputeCapability_白皮书与SASS.md)

## 怎么学这一卷

```text
这一卷偏"理解"而非"动手"——大部分是建立硬件心智模型，少量 microbenchmark。
建议：在完成卷一~卷五后读，把前面学的软件优化逐条对应到这里的硬件结构。
重点：01-04（结构）打底，06（演进）面试高频，07（怎么读资料）实用。
05（功耗）和代际细节了解即可。
```

> 实事求是的提醒：本卷涉及的微架构细节，**只用 NVIDIA 官方公开资料**（白皮书、Programming
> Guide、规格表），不使用社区传闻或未公开的逆向细节。已公开的讲清楚，未公开的明确说"未公开"。
