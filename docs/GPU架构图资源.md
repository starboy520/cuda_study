# GPU 架构图资源索引

> 老版 Programming Guide 里的经典配图大多还在，分散在 **新版指南**、**Legacy 指南** 和 **Archive 归档** 里。

---

## 一、你最可能记得的那张「整体架构图」

### 新版（CUDA 13.x，推荐）

**Figure 2：GPU + CPU 系统架构**（GPC、SM、显存、PCIe/NVLink）

- 文档页：[Programming Model — GPU Hardware Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#gpu-hardware-model)
- 图片直链：https://docs.nvidia.com/cuda/cuda-programming-guide/_images/gpu-cpu-system-diagram.png

内容：CPU + 系统内存 ↔ 互联 ↔ GPU（多个 GPC → 多个 SM → 寄存器/共享内存/L1）+ GPU 显存 + L2。

### 老版经典配图（CUDA 12.0 归档，仍可用）

| 图 | 内容 | 直链 |
|----|------|------|
| **Memory Hierarchy** | 线程/Block/Global/Constant/Texture 内存层次 | https://docs.nvidia.com/cuda/archive/12.0.0/cuda-c-programming-guide/_images/memory-hierarchy.png |
| **Heterogeneous Programming** | Host/Device 分工示意图 | https://docs.nvidia.com/cuda/archive/12.0.0/cuda-c-programming-guide/_images/heterogeneous-programming.png |
| **Grid of Thread Blocks** | Grid / Block 逻辑结构 | https://docs.nvidia.com/cuda/archive/12.0.0/cuda-c-programming-guide/_images/grid-of-thread-blocks.png |
| **GPU transistors** | CPU vs GPU 晶体管分配（入门） | https://docs.nvidia.com/cuda/archive/12.0.0/cuda-c-programming-guide/_images/gpu-devotes-more-transistors-to-data-processing.png |

归档目录首页：https://docs.nvidia.com/cuda/archive/12.0.0/cuda-c-programming-guide/index.html

---

## 二、新版 Programming Guide 全部配图（CUDA 13.x）

文档入口：https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html

| Figure | 内容 | 图片 |
|--------|------|------|
| Fig 2 | **GPU/CPU 整体架构** | [gpu-cpu-system-diagram.png](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/gpu-cpu-system-diagram.png) |
| Fig 3 | Grid of Thread Blocks | [grid-of-thread-blocks.png](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/grid-of-thread-blocks.png) |
| Fig 4 | Block 调度到 SM | [thread-block-scheduling.png](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/thread-block-scheduling.png) |
| Fig 5–6 | Thread Block Cluster（H100 等） | [grid-of-clusters.png](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/grid-of-clusters.png) |
| Fig 7 | Warp divergence | [active-warp-lanes.png](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/active-warp-lanes.png) |

基础篇更多图（合并访问、矩阵转置等）：  
https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html

---

## 三、极老版本里的「SM 内部硬件图」（Figure 3-1 Hardware Model）

CUDA 2.x 时代的 **Figure 3-1 Hardware Model**（SM 内 SP 核心、Shared Memory 框图）在现行在线文档里 **已不再保留**，但 PDF 归档里仍有文字描述。

- 老 PDF 示例：https://docs.nvidia.com/cuda/archive/2.0/Programming_Guide_2.0beta2.pdf（搜 "Figure 3-1" / "Hardware Model"）
- 另一镜像：https://arcb.csc.ncsu.edu/~mueller/cluster/nvidia/2.0/Programming_Guide_2.0beta2.pdf

**学习建议**：SM 内部细节看 **架构白皮书**（见下节）比看 15 年前的 Tesla 图更准确；对你 T4 看 **Turing 白皮书**。

---

## 四、芯片级架构图（T4 / A100 等，比 PG 更详细）

Programming Guide 画的是 **编程模型**；要看 **真实芯片结构**（GPC、SM、HBM、NVLink），读 Architecture Whitepaper：

| GPU | 白皮书 |
|-----|--------|
| **T4 (Turing)** | [NVIDIA Turing Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf) |
| A100 (Ampere) | [NVIDIA Ampere Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-ampere-architecture-whitepaper.pdf) |
| H100 (Hopper) | [NVIDIA Hopper Architecture](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-whitepaper-28413) |

T4 白皮书里有 **SM 内部结构**（CUDA Core、Tensor Core、Register File、L1/Shared Memory），比老 PG 的 Figure 3-1 更贴近你现在的卡。

---

## 五、Legacy 与 Archive 怎么用

| 入口 | 说明 |
|------|------|
| [CUDA C++ Programming Guide (Legacy)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | 旧版章节结构，含 Memory Hierarchy 等文字 |
| [CUDA Documentation Archive](https://docs.nvidia.com/cuda/archive/) | 按 CUDA 版本选，如 12.0、11.x、9.2 |
| [Programming Guide PDF (v13.3)](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) 页顶 **PDF** 按钮 | 离线阅读新版全文 |

**版本对应建议**：
- 学编程模型配图 → **新版** 13.x 或 **归档 12.0**
- 查老教程引用的 Figure 编号 → **Archive 12.0** 最接近
- 查 T4 硬件 → **Turing 白皮书**

---

## 六、对照：老图 → 现在去哪找

| 你记得的老图 | 现在位置 |
|--------------|----------|
| GPU 整体框图（CPU+GPU+内存） | 新版 **Figure 2** `gpu-cpu-system-diagram.png` |
| Grid / Thread Block 网格图 | 新版 **Figure 3** 或归档 `grid-of-thread-blocks.png` |
| Memory Hierarchy 金字塔/层次图 | 归档 `memory-hierarchy.png`（Legacy §5.3） |
| Host/Device 异构图 | 归档 `heterogeneous-programming.png` |
| SM 内部 SP 核心图（很老） | CUDA 2.0 PDF；**更推荐 Turing 白皮书** |

---

## 七、Week 1 推荐阅读顺序

1. 打开 [gpu-cpu-system-diagram.png](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/gpu-cpu-system-diagram.png) — 建立 GPU 全局观  
2. 打开 [grid-of-thread-blocks.png](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/grid-of-thread-blocks.png) — 对照 vec_add  
3. 打开 [memory-hierarchy.png](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-c-programming-guide/_images/memory-hierarchy.png) — Week 2 预习  
4. （可选）浏览 T4 Turing 白皮书 SM 章节  

---

**返回**：[Programming_Model详解.md](Programming_Model详解.md) · [CUDA基础概念.md](../notes/CUDA基础概念.md)
