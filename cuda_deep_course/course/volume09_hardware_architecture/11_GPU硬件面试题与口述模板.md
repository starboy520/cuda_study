# 11 GPU 硬件面试题与口述模板

这一章不是新知识，而是把卷九变成面试表达能力。

目标：

```text
1. 能用 1-2 分钟讲清一个硬件概念。
2. 能把硬件概念连接到 CUDA 优化。
3. 能把单卡硬件连接到 AI infra。
```

## 1. 回答硬件题的通用结构

推荐结构：

```text
定义：
  这个东西是什么？

为什么需要：
  它解决什么问题？

对写代码的影响：
  CUDA kernel / AI infra 里怎么用？

限制：
  它不是万能的，约束是什么？
```

例如回答 shared memory：

```text
Shared memory 是 SM 内的片上 SRAM，由 block 内线程共享。
它解决 global memory 远、慢、重复读的问题。
CUDA 里常用它做 tiling，把 A/B tile 搬进 shared 后复用。
限制是容量小、block 私有、要注意 bank conflict 和 __syncthreads。
```

## 2. 题型一：GPU 执行模型

### Q1：GPU 为什么适合并行计算？

参考回答：

```text
GPU 是吞吐优先架构，有大量 SM 和执行单元，适合同时执行大量相似任务。
CUDA 程序中 thread 被组织成 block 和 grid，硬件中 thread 以 warp 为调度单位。
GPU 通过大量驻留 warp 隐藏内存和流水线延迟。
但它不是所有任务都快，性能取决于并行度、访存模式、分支分歧和数据搬运。
```

### Q2：Block、Warp、SM 的关系？

```text
Block 是 CUDA 编程模型里的逻辑协作单元。
SM 是 GPU 上执行 block 的硬件单元。
一个 block 会被调度到一个 SM 上执行，block 内 thread 会被硬件分成 warp。
Warp 是调度和发射指令的基本单位，通常 32 个 thread。
```

### Q3：为什么需要很多 warp？

```text
访存和指令都有延迟。一个 warp 等数据时，scheduler 可以切换到其他 ready warp。
这就是延迟隐藏。occupancy 太低时，ready warp 不够，执行单元容易空转。
但 occupancy 不是越高越好，寄存器 tiling 可能降低 occupancy 却提高每线程效率。
```

## 3. 题型二：内存层次

### Q4：GPU 的内存层次是什么？

```text
从近到远是 register、shared/L1、L2、global memory。
register 在 SM 内，最快但最小；shared/L1 是 SM 内片上 SRAM；
L2 是全 GPU 共享 cache；global memory 是片外 GDDR/HBM，容量大但延迟高。
CUDA 优化本质是减少慢层访问、提高快层复用。
```

### Q5：为什么 shared memory 快？

```text
shared memory 是 SM 内片上 SRAM，距离执行单元近，延迟远低于片外显存。
它由程序员显式管理，适合 block 内数据复用，比如 GEMM tiling。
限制是容量小、block 私有、需要同步、要避免 bank conflict。
```

### Q6：寄存器压力为什么影响 occupancy？

```text
每个 SM 的寄存器文件大小固定。
每个 thread 用的寄存器越多，同一 SM 能驻留的 thread/warp 就越少。
这会降低 occupancy，可能让延迟隐藏变差。
但寄存器多也可能带来 register tiling 和 ILP，所以要实测权衡。
```

## 4. 题型三：Tensor Core 和低精度

### Q7：Tensor Core 为什么快？

```text
CUDA Core 做标量 FMA，Tensor Core 做小矩阵乘加，一条 MMA/WGMMA 完成很多乘加。
深度学习核心算子 GEMM/conv/attention 都是矩阵密集运算，所以 Tensor Core 是 AI 算力的主要来源。
它通常使用 FP16/BF16/TF32/FP8/FP4 等低精度格式，通过更小数据和专用矩阵硬件提高吞吐。
限制是数据类型、shape、layout 和数值精度管理。
```

### Q8：怎么确认用了 Tensor Core？

```text
可以看 Nsight Compute 的 Tensor Core pipe utilization，
也可以看 PTX/SASS 是否出现 mma/wgmma 指令。
如果普通 FP32 for-loop GEMM 只生成 FFMA，那就是 CUDA Core 路径。
工程上通常通过 cuBLAS/cuBLASLt/CUTLASS/TensorRT 间接使用 Tensor Core。
```

### Q9：FP16、BF16、FP8、FP4 有什么意义？

```text
低精度减少存储和带宽，同时硬件能提供更高矩阵吞吐。
FP16 吞吐高但指数范围较小；BF16 指数范围接近 FP32，训练更稳；
FP8/FP4 进一步提高吞吐和容量效率，但需要 scale、量化、混合精度累加和误差控制。
```

## 5. 题型四：架构演进

### Q10：A100 和 H100 的主要区别？

```text
A100 是 Ampere，重点包括第三代 Tensor Core、TF32/BF16、cp.async、MIG 和 HBM2e。
H100 是 Hopper，重点包括第四代 Tensor Core、FP8 Transformer Engine、TMA、Thread Block
Cluster、DSM、更强 NVLink。
一条主线是 Tensor Core 和低精度增强，另一条主线是数据搬运从 cp.async 走向 TMA 硬件化。
```

### Q11：Hopper 的 TMA 解决什么？

```text
TMA 是 Tensor Memory Accelerator，用硬件引擎搬运大块多维 global/shared tile。
相比 cp.async 由线程发起许多小 copy，TMA 更适合复杂 tile 和高性能 GEMM/attention pipeline，
能减少指令和索引开销。
```

### Q12：Thread Block Cluster / DSM 是什么？

```text
Cluster 是 block 之上的可选层级，cluster 内 block 可以协同调度和同步。
DSM 让 cluster 内 block 可以访问彼此的 shared memory。
它突破传统 block 间不能直接共享 shared 的限制，但不是整个 grid 的全局同步。
```

### Q13：Blackwell 该怎么讲？

```text
Blackwell 延续 CUDA 编程模型，但强化 Tensor Core、低精度、内存系统和 NVLink。
数据中心 B200/GB200 属于 CC 10.0，B300/GB300 属于 CC 10.3，RTX Blackwell 属于 12.x。
AI infra 上要关注 FP4/NVFP4、HBM、NVLink/NVL72、系统级推理/训练吞吐，而不是只背 SM 数。
```

## 6. 题型五：AI Infra 硬件

### Q14：NVLink 和 PCIe 区别？

```text
PCIe 是通用设备互连，CPU-GPU 和 GPU-GPU 都可走 PCIe，但带宽/延迟不如 NVLink。
NVLink 是 NVIDIA 面向 GPU 间通信的高速互连，更适合多 GPU collective、tensor parallel 和
pipeline parallel。大系统还会用 NVSwitch 组织多 GPU 高带宽互连。
```

### Q15：NCCL 做什么？

```text
NCCL 是 NVIDIA collective communication library，提供 all-reduce、all-gather、reduce-scatter
等 GPU collective。训练中的 data parallel、tensor parallel、FSDP/ZeRO 都依赖 collective。
NCCL 性能受 NVLink、PCIe、InfiniBand/RoCE 和 topology 影响。
```

### Q16：为什么多卡 scaling 不线性？

```text
因为通信、同步、load imbalance、batch size、topology、网络和功耗都会影响。
数据并行要梯度 all-reduce，tensor parallel 可能每层通信。
如果通信不能和计算重叠，或者 batch 太小喂不满 GPU，多卡效率就下降。
```

### Q17：MIG 和 MPS 区别？

```text
MIG 是硬件级资源切分和隔离，把一块 GPU 切成多个实例，适合多租户和推理服务。
MPS 是软件服务，让多个 CUDA 进程更高效共享一块 GPU，减少多进程调度开销。
MIG 重隔离，MPS 重共享利用率。
```

### Q18：生产环境 GPU 要监控什么？

```text
利用率、显存、温度、功耗、频率、ECC、Xid、NVLink/PCIe、throttling、进程、作业失败。
常用 nvidia-smi、NVML、DCGM。AI infra 既要性能，也要可靠性和可恢复性。
```

## 7. 题型六：性能诊断

### Q19：一个 kernel 慢，你怎么定位？

```text
先确认正确性和输入规模，再用 Nsight Systems 看端到端时间线和同步/拷贝/launch。
然后用 Nsight Compute 看 kernel 级指标：occupancy、SM 利用率、memory throughput、cache hit、
stall reason、Tensor Core/FP32 pipe。
结合 Roofline 判断 memory-bound 还是 compute-bound，再决定优化方向。
```

### Q20：Roofline 怎么用？

```text
Roofline 用算术强度和硬件峰值判断性能上限。
算术强度低时通常 memory-bound，要减少访存或提高复用；
算术强度高但没到算力峰值时，要看执行管线、occupancy、ILP、Tensor Core 利用率等。
```

### Q21：为什么高 occupancy 不一定快？

```text
occupancy 只是驻留 warp 比例，代表潜在延迟隐藏能力，不等于实际吞吐。
如果已经带宽受限或 ILP 足够，高 occupancy 不一定提升性能。
register tiling 可能降低 occupancy，但提高数据复用和每线程计算量，整体更快。
```

## 8. 3 个 2 分钟口述模板

### 模板一：CUDA 硬件执行

```text
CUDA 程序在软件上组织成 grid、block、thread；硬件上 block 被调度到 SM，thread 被分成 warp。
warp 是调度基本单位，scheduler 选择 ready warp 发射指令，scoreboard 跟踪依赖。
GPU 通过大量驻留 warp 隐藏访存和流水线延迟。
因此 block size、occupancy、register/shared 用量、分支分歧都会影响性能。
```

### 模板二：GEMM 为什么快

```text
GEMM 的优化核心是数据复用。
naive GEMM 每个 thread 独立读 A/B，算术强度低。
shared tiling 把 A/B tile 搬到 shared，让 block 内 thread 复用 global 数据。
register tiling 让 thread 维护 acc 小矩阵，复用 shared 数据。
更高阶实现使用 Tensor Core，通过 MMA/WGMMA 做小矩阵乘加，并用 cp.async/TMA 做数据搬运 pipeline。
```

### 模板三：AI Infra 硬件

```text
AI infra 需要同时看单卡和多卡。
单卡上，Tensor Core、HBM 容量/带宽、L2/shared/register 决定算子性能和模型容量。
多卡上，NVLink/NVSwitch/PCIe/InfiniBand 和 NCCL collective 决定训练/推理并行效率。
生产环境还要考虑 MIG/MPS、DCGM/NVML 监控、ECC/Xid、功耗散热和故障恢复。
```

## 9. 最后检查清单

你准备 CUDA / AI infra 面试前，至少能讲清：

```text
SM / warp / block / scheduler / scoreboard
register / shared / L1 / L2 / HBM
coalescing / bank conflict / occupancy / Roofline
CUDA Core / Tensor Core / SFU / LSU
MMA / WMMA / WGMMA / Tensor Core
cp.async / TMA / Thread Block Cluster / DSM
Turing / Ampere / Hopper / Blackwell
NVLink / NVSwitch / NCCL / HBM / MIG / DCGM
```

如果你能把这些词都连回：

```text
性能为什么变快或变慢
代码应该怎么写
系统应该怎么部署和监控
```

卷九就真正学透了。
