# 卷八：HPC 与多 GPU

> 前七卷都在单 GPU 上"把一个问题算对、算快"。卷八把视野拉到两个新维度：**数值**（浮点到底
> 准不准、可不可复现）和**规模**（一块卡不够时，怎么用多块卡协作）。这是从"会写 kernel"到
> "能做科学计算和大规模系统"的跨越。

## 学习目标

- 理解 IEEE-754 浮点、FMA、舍入和误差传播，知道"结果差一点"何时正常、何时是 bug。
- 在精度和性能之间做有依据的权衡，必要时实现可复现归约。
- 知道 cuBLAS / cuFFT / cuSPARSE / cuSOLVER 各解决什么、何时用库而非手写。
- 理解 GPU 系统拓扑：PCIe、NVLink、NVSwitch、P2P 的带宽层级。
- 掌握多 GPU 的数据划分、负载均衡和同步基本模式。
- 用 NCCL 做集合通信（broadcast / all-reduce / all-gather）。
- 让计算与通信重叠，理解多 GPU 的 scaling 与扩展效率。

## 为什么需要这一卷（一张图）

```text
单 GPU 的两个天花板：
  ① 数值精度：算得快但算得对吗？大规模累加的误差会失控吗？  -> 数值稳定性（01-02）
  ② 单卡规模：数据放不下 / 算太慢，一块卡到顶了            -> 多 GPU（04-07）

卷八教你突破这两个天花板，同时知道工业库（03）能帮你省多少力。
```

## 章节

1. [浮点、FMA 与误差传播](01_浮点_FMA与误差传播.md) —— IEEE-754、舍入、为什么结果会差
2. [可复现归约与精度-性能权衡](02_可复现归约与精度性能权衡.md) —— bitwise 一致的代价
3. [数学库：cuBLAS / cuFFT / cuSPARSE / cuSOLVER](03_数学库_cuBLAS_cuFFT_cuSPARSE_cuSOLVER.md)
4. [GPU 拓扑：PCIe、NVLink 与 P2P](04_GPU拓扑_PCIe_NVLink与P2P.md)
5. [多 GPU 数据划分与同步](05_多GPU数据划分与同步.md)
6. [NCCL 集合通信](06_NCCL集合通信.md) —— broadcast / all-reduce / all-gather
7. [计算通信重叠与 Scaling](07_计算通信重叠与Scaling.md) —— strong/weak scaling、扩展效率

## 适合什么人重点学

```text
HPC / 科学计算岗：01-07 全部，尤其数值（01-02）和多 GPU（04-07）
深度学习训练岗：  04-07（分布式训练靠 NCCL all-reduce）+ 03（库）
通用 CUDA 岗：    了解为主，01/03/06 知道概念即可
```

> 硬件提示：多 GPU 章节（04-07）需要多卡环境才能完整实验。单卡（如 T4）只能学概念 + 跑单卡
> 部分。本卷以原理 + 可迁移代码为主，不依赖你一定有多卡。
