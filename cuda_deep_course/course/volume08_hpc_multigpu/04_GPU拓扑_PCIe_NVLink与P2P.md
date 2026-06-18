# 04 GPU 拓扑：PCIe、NVLink 与 P2P

## 0. 先建立大局观：多 GPU 的瓶颈是"卡之间怎么连"

单 GPU 的瓶颈是显存带宽（320 GB/s 量级）。一旦上多 GPU，新瓶颈出现了——**卡和卡之间、卡和
CPU 之间的数据怎么传**。这条"互连"链路的带宽，往往比显存带宽低一个数量级，成为多卡系统的
关键约束。

```text
单 GPU： SM <-> 显存（快，~320 GB/s）
多 GPU： GPU0 <-> GPU1（互连，可能只有 ~16-600 GB/s，差异巨大）
        GPU <-> CPU（PCIe，~16 GB/s，更慢）

设计多卡系统前，必须知道"谁和谁怎么连、带宽多少"——这就是拓扑。
```

## 0.1 术语速查表

| 术语 | 一句话定义 | 带宽量级 |
|---|---|---|
| **PCIe** | CPU 和 GPU、GPU 间的通用互连 | ~16-64 GB/s |
| **NVLink** | NVIDIA GPU 间的高速专用互连 | ~300-900 GB/s |
| **NVSwitch** | 连接多个 GPU 的 NVLink 交换芯片 | 全互连高带宽 |
| **P2P (Peer-to-Peer)** | GPU 直接访问另一 GPU 显存，不经 CPU | 看底层链路 |
| **拓扑 (topology)** | 系统里 GPU/CPU 怎么连接的结构图 |  |
| **bandwidth** | 单位时间传输量 | GB/s |
| **latency** | 单次传输的延迟 |  |

## 1. 带宽层级：记住这个金字塔

多 GPU 系统里，数据访问速度分明显的层级（从快到慢）：

```text
最快  GPU 访问自己的显存          ~320 GB/s（T4）/ ~2000 GB/s（A100 HBM）
  ↓   GPU 经 NVLink 访问邻卡显存   ~300-900 GB/s（有 NVLink 时）
  ↓   GPU 经 PCIe 访问邻卡/CPU     ~16-64 GB/s
最慢  跨 NUMA / 跨节点网络          更慢
```

**核心设计原则**：让数据尽量待在快的层级，**减少跨慢链路的传输**。多 GPU 优化的本质就是
"少传、就近传、传的时候和计算重叠"。

## 2. PCIe：通用但慢的互连

PCIe 是 CPU 和 GPU、以及没有 NVLink 时 GPU 之间的标准连接：

```text
PCIe Gen3 x16：~16 GB/s（单向）
PCIe Gen4 x16：~32 GB/s
PCIe Gen5 x16：~64 GB/s

特点：通用（所有 GPU 都有）、但比显存带宽低一个数量级
```

回忆卷七：H2D/D2H 传输走 PCIe，所以**传输常常是端到端瓶颈**。这也是为什么要：

```text
- pinned memory：让 DMA 直接搬，跑满 PCIe（卷三/04）
- 多 stream 重叠：让 PCIe 传输和 SM 计算并行（卷七/02）
- 减少传输：数据尽量留在 GPU，别来回搬
```

## 3. NVLink：GPU 间的高速公路

NVLink 是 NVIDIA 专为 GPU 互连设计的高速链路，带宽**远高于 PCIe**：

```text
PCIe Gen3：     ~16 GB/s
NVLink：        ~300 GB/s 起（多代演进到 600-900 GB/s）
                高出 PCIe 一二十倍
```

意义：有 NVLink 的系统里，GPU 之间传数据**快得多**，多卡协作（如分布式训练的梯度同步）才
高效。没有 NVLink、只能走 PCIe 的系统，卡间通信会成为更严重的瓶颈。

> 注意：不是所有多 GPU 系统都有 NVLink。消费级/部分数据中心配置只有 PCIe 互连。设计前要
> **查实际拓扑**（第 5 节）。

## 4. NVSwitch：让"很多 GPU"全互连

两块 GPU 直连 NVLink 就够。但 8 块、16 块 GPU 怎么办？两两直连线太多。**NVSwitch** 是一个
交换芯片，让多个 GPU 通过它**全互连**，任意两卡都能高带宽通信：

```text
无 NVSwitch：GPU 只能和直连的邻居高速通信，远的要中转
有 NVSwitch：任意两 GPU 都能直接高带宽互通（如 DGX 系统的 8 卡全互连）
```

这是大规模训练集群（如 DGX）能高效做 all-reduce 的硬件基础。

## 5. P2P：GPU 直接访问邻卡显存

**Peer-to-Peer (P2P)** 让一个 GPU **直接读写另一个 GPU 的显存**，不用经过 CPU 中转：

```cpp
// 检查并开启 P2P
int canAccess;
cudaDeviceCanAccessPeer(&canAccess, 0, 1);   // GPU0 能否访问 GPU1
if (canAccess) {
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);          // 开启 GPU0 -> GPU1 的 P2P
}
// 之后可直接 cudaMemcpyPeer 或 kernel 里访问对方显存
cudaMemcpyPeer(dst_on_gpu1, 1, src_on_gpu0, 0, bytes);
```

没有 P2P 时，GPU0 → GPU1 要走 **GPU0 → CPU → GPU1** 两段 PCIe（慢且占 CPU）。有 P2P（尤其
配 NVLink）时直接传，快得多。

```text
无 P2P：GPU0 显存 -> CPU 内存 -> GPU1 显存   两跳，慢
有 P2P：GPU0 显存 -> GPU1 显存              一跳，走 NVLink/PCIe 直连
```

## 6. 怎么查系统拓扑

设计多卡程序前，先用工具看清"谁和谁怎么连"：

```bash
nvidia-smi topo -m       # 打印 GPU 间的连接矩阵（NVLink/PCIe/同 NUMA 等）
```

输出会标明每对 GPU 的连接类型：

```text
NV#  ：通过 N 条 NVLink 连接（快）
PIX/PXB/PHB ：通过 PCIe 不同层级连接（慢）
SYS ：跨 NUMA/socket（最慢）
```

> 实战意义：知道拓扑才能合理分配任务——把通信频繁的 GPU 对放在 NVLink 连接上，避免热点数据
> 走慢链路。这是多 GPU 性能调优的起点。

## 7. 实践（需多卡环境）

1. 用 `nvidia-smi topo -m` 查看（或在多卡机器上）GPU 拓扑，识别 NVLink vs PCIe 连接。
2. 写一个 `cudaMemcpyPeer` 在两卡间传数据，对比开/关 P2P 的带宽。
3. 估算：在只有 PCIe（16 GB/s）的双卡系统上传 1GB 数据要多久？换 NVLink（300 GB/s）呢？
4. （单卡也可）用 bandwidthTest 测你的 H2D/D2H PCIe 带宽。

## 8. 面试题（附参考答案）

**Q1：多 GPU 系统的带宽层级是怎样的？**
GPU 访问自己显存最快（百 GB/s~TB/s）、NVLink 访问邻卡次之（300-900 GB/s）、PCIe 访问邻卡/CPU
更慢（16-64 GB/s）。设计原则是数据尽量待在快层级、减少跨慢链路传输。

**Q2：NVLink 和 PCIe 的区别？**
PCIe 是通用互连（所有 GPU 都有）但带宽低（~16-64 GB/s）；NVLink 是 NVIDIA GPU 间专用高速链路
（300-900 GB/s），高出一二十倍。有无 NVLink 直接决定多卡通信效率。

**Q3：什么是 P2P，有什么好处？**
Peer-to-Peer 让 GPU 直接读写另一 GPU 的显存，不经 CPU 中转。没 P2P 时 GPU0→GPU1 要走两段
PCIe 经 CPU；有 P2P 直接一跳（走 NVLink/PCIe 直连），更快且不占 CPU。

**Q4：NVSwitch 解决什么问题？**
多 GPU（8/16 块）两两直连 NVLink 线太多。NVSwitch 是交换芯片，让多 GPU 通过它全互连，任意两
卡都能高带宽通信，是大规模训练集群的硬件基础。

**Q5：为什么多 GPU 程序要先查拓扑？**
不同 GPU 对之间连接类型不同（NVLink 快 / PCIe 慢 / 跨 NUMA 更慢）。知道拓扑才能把通信频繁的
任务放在快链路上，避免热点走慢路径。用 `nvidia-smi topo -m` 查。

## 9. 资料映射

- NVIDIA NVLink / NVSwitch 技术文档；`nvidia-smi topo` 说明。
- CUDA Programming Guide：Peer Device Memory Access。
- 配套：[卷七 异步系统](../volume07_async_system/README.md)、[卷三第 04 章 Host 传输](../volume03_memory_system/04_Cache_Host传输与Unified_Memory.md)。
