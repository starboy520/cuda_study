# 10 AI Infra 硬件：NVLink、NCCL、HBM、MIG、DCGM

CUDA kernel 面试主要看：

```text
SM / warp / memory / Tensor Core
```

AI infra 面试还会看：

```text
多 GPU 怎么连
显存够不够
通信带宽够不够
怎么隔离资源
怎么监控故障和性能
```

这一章把单卡硬件扩展到集群硬件。

## 1. AI Infra 为什么不能只懂单卡

大模型训练和推理经常是：

```text
多 GPU
多机
高带宽互连
大显存
高功耗
持续运行
```

瓶颈可能来自：

```text
GPU 计算
HBM 带宽
GPU-GPU 通信
CPU-GPU PCIe
网络通信
存储和数据加载
功耗/散热
```

所以 AI infra 面试常问：

```text
为什么多卡训练 scaling 不线性？
NVLink 和 PCIe 差在哪？
NCCL 做什么？
HBM 容量如何影响 batch size / KV cache？
MIG 适合什么场景？
怎么监控 GPU 健康？
```

## 2. PCIe、NVLink、NVSwitch 的层次

### PCIe

PCIe 是 CPU 和 GPU、设备之间的通用互连。

特点：

```text
通用
生态成熟
但 GPU-GPU 带宽和延迟通常不如 NVLink
```

常见影响：

```text
Host <-> Device copy
数据加载
没有 NVLink 的 GPU 间通信
```

### NVLink

NVLink 是 NVIDIA 的高速 GPU 互连。

它的意义：

```text
比 PCIe 更适合 GPU-GPU 通信
提升 all-reduce、all-gather、tensor parallel、pipeline parallel 的效率
```

### NVSwitch

NVSwitch 可以把多块 GPU 连接成高带宽交换结构。

直觉：

```text
NVLink 是高速路。
NVSwitch 是高速路交换枢纽。
```

在大系统中，比如 DGX / HGX / NVL72，NVSwitch 让多 GPU 之间有更强的互联能力。

## 3. NCCL 是什么

NCCL 是 NVIDIA Collective Communications Library。

它不是硬件，但它是 AI infra 面试里绕不开的软件层。

NCCL 做 collective 通信：

```text
all-reduce
all-gather
reduce-scatter
broadcast
send/recv
```

训练中常见：

```text
Data Parallel:
  梯度 all-reduce。

Tensor Parallel:
  中间激活 all-reduce / all-gather。

Pipeline Parallel:
  stage 之间 send/recv。

ZeRO / FSDP:
  参数和梯度 shard 的 all-gather / reduce-scatter。
```

NCCL 会根据 topology 选择通信路径：

```text
NVLink
PCIe
InfiniBand / RoCE
```

所以硬件 topology 会直接影响训练吞吐。

## 4. 为什么多卡 scaling 不线性

理想情况：

```text
8 张 GPU = 8 倍性能
```

现实经常不是，因为：

```text
1. 通信开销增加。
2. 数据并行要 all-reduce 梯度。
3. tensor parallel 每层都可能通信。
4. batch 不够大时 GPU 吃不满。
5. 网络或 NVLink 带宽成为瓶颈。
6. straggler、功耗、温度、调度也会影响。
```

面试回答：

```text
多卡性能取决于 compute/communication overlap、通信量、topology、batch size 和并行策略。
不是 GPU 数量翻倍，吞吐就一定翻倍。
```

## 5. HBM 容量和带宽

HBM 是高带宽显存。

AI infra 里要同时关心：

```text
容量
带宽
```

容量决定：

```text
模型参数能不能放下
optimizer state 能不能放下
activation 能不能放下
KV cache 能支持多长上下文、多大 batch
```

带宽决定：

```text
memory-bound kernel 的速度
decode 阶段 KV cache 读取速度
embedding / layernorm / softmax / elementwise 的性能
```

LLM 推理里：

```text
prefill:
  GEMM 多，Tensor Core 利用更关键。

decode:
  batch 小时，KV cache 读取和 memory bandwidth 更关键。
```

所以：

```text
同一块 GPU 的 TFLOPS 很高，不代表所有推理场景都快。
```

## 6. MIG：Multi-Instance GPU

MIG 可以把一块支持 MIG 的 GPU 切成多个隔离实例。

它解决：

```text
多租户隔离
小模型推理资源切分
提高资源利用率
避免一个任务独占整卡
```

它提供隔离的资源包括：

```text
计算资源
显存容量
缓存/内存带宽的一部分
故障隔离能力
```

适合：

```text
推理服务
小训练任务
多用户共享集群
稳定 QoS 场景
```

不适合：

```text
需要整卡大显存的大模型训练
需要最大 NVLink/NCCL 性能的训练任务
对跨实例通信要求高的任务
```

## 7. MPS：Multi-Process Service

MPS 不是 MIG。

MPS 是软件服务，让多个 CUDA 进程更高效地共享同一块 GPU。

区别：

```text
MIG:
  硬件级切分和隔离。

MPS:
  软件层帮助多个进程共享 GPU 执行资源。
```

面试里可以这样说：

```text
MIG 偏资源隔离和多租户。
MPS 偏减少多进程调度开销、提升共享 GPU 时的利用率。
```

## 8. DCGM / NVML / nvidia-smi：监控与运维

AI infra 不是跑一次 benchmark，而是让 GPU 集群长期稳定服务。

常见工具：

```text
nvidia-smi:
  命令行查看 GPU 状态。

NVML:
  NVIDIA Management Library，程序化读取 GPU 状态。

DCGM:
  Data Center GPU Manager，面向数据中心监控、诊断、健康检查。
```

需要监控：

```text
GPU utilization
memory utilization
HBM used/free
temperature
power draw
clock
ECC error
PCIe replay / link
NVLink health
Xid error
throttling reason
```

面试回答：

```text
AI infra 要关注性能和可靠性。
除了 kernel 性能，还要监控功耗、温度、ECC、Xid、NVLink、显存、利用率。
DCGM/NVML 是生产环境常用的 GPU 监控入口。
```

## 9. ECC、RAS 和故障

长时间训练/推理中，硬件错误不是理论问题。

常见词：

```text
ECC:
  显存错误检测/纠正。

RAS:
  Reliability, Availability, Serviceability。

Xid:
  NVIDIA driver 报告的 GPU 错误代码。
```

工程上要做：

```text
监控 ECC / Xid
失败自动重试
节点隔离
checkpoint
训练作业恢复
推理服务降级
```

AI infra 面试常看你是否知道：

```text
GPU 集群不是只追求峰值性能，还要追求稳定、可观测、可恢复。
```

## 10. GPUDirect RDMA 和跨机通信

多机训练中，GPU 之间常通过 InfiniBand 或 RoCE 通信。

如果数据必须走：

```text
GPU -> CPU memory -> NIC -> network
```

开销很大。

GPUDirect RDMA 的目标是：

```text
让网卡更直接地访问 GPU memory，减少 CPU 中转。
```

这对多机 NCCL 很重要。

面试可以说：

```text
单机内 GPU 通信主要看 NVLink/NVSwitch/PCIe。
跨机通信主要看 NIC、InfiniBand/RoCE、GPUDirect RDMA 和 NCCL topology。
```

## 11. AI Infra 面试口述模板

可以这样回答：

```text
单卡性能主要由 Tensor Core、CUDA Core、HBM 带宽和显存容量决定。
多卡性能还取决于 GPU 间互连，比如 NVLink/NVSwitch，以及跨机网络，比如 InfiniBand/RoCE。

训练中 NCCL 的 all-reduce、all-gather、reduce-scatter 会受到 topology 和带宽影响，
所以多卡 scaling 不会天然线性。推理中 HBM 容量和带宽会影响 KV cache、batch size 和 decode
性能。

在生产环境里，还要考虑 MIG/MPS 资源共享、DCGM/NVML 监控、ECC/Xid/RAS、功耗散热和故障恢复。
```

## 12. 常见面试题

**Q1：NVLink 和 PCIe 区别？**

PCIe 是通用设备互连，GPU-GPU 通信带宽和延迟通常不如 NVLink。NVLink 是 NVIDIA 面向 GPU 间高
带宽通信的互连，更适合多 GPU collective 和模型并行。

**Q2：NCCL 做什么？**

NCCL 做 GPU collective 通信，如 all-reduce、all-gather、reduce-scatter。它会利用 NVLink、
PCIe、InfiniBand/RoCE 等 topology，为训练和推理并行提供通信基础。

**Q3：HBM 容量和带宽分别影响什么？**

容量影响模型、activation、optimizer state、KV cache 和 batch/context 能不能放下；带宽影响
memory-bound kernel、KV cache 读取、decode 阶段性能。

**Q4：MIG 和 MPS 区别？**

MIG 是硬件级 GPU 资源切分和隔离，适合多租户和小推理服务；MPS 是软件服务，让多个 CUDA 进程
更高效共享同一块 GPU。

**Q5：生产环境 GPU 要监控什么？**

利用率、显存、温度、功耗、频率、ECC、Xid、NVLink、PCIe、throttling、作业失败率。常用
nvidia-smi、NVML、DCGM。

## 13. 实践

1. 用 `nvidia-smi topo -m` 查看当前机器 GPU/NIC topology。
2. 用 `nvidia-smi dmon` 观察功耗、温度、显存、利用率。
3. 如果有多卡，用 NCCL tests 跑 all-reduce 带宽。
4. 用 DCGM exporter 或 `dcgmi` 了解数据中心 GPU 监控指标。
5. 画一张图：单机 8 GPU 中 PCIe、NVLink、NVSwitch、NIC 的关系。

## 14. 资料映射

- NVIDIA NCCL 文档：collective communication。
- NVIDIA NVLink / NVSwitch / Blackwell 官方资料。
- NVIDIA DCGM / NVML 文档。
- NVIDIA MIG 用户指南。
- 配套：[卷五性能工程](../volume05_performance/README.md)、[卷八 HPC 与多 GPU](../volume08_hpc_multigpu/README.md)。
