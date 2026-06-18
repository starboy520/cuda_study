# 06 NCCL 集合通信

## 0. 先建立大局观：多卡通信不要自己手搓

卷八/05 提到多卡要"汇总结果"。手动用 `cudaMemcpyPeer` 在卡间两两传数据，既麻烦又难做到最优
（要考虑拓扑、带宽、重叠）。**NCCL（NVIDIA Collective Communications Library）** 把这些"多卡
协同通信"封装成优化好的原语，是分布式训练的通信基石。

```text
手动多卡通信：要自己处理"谁传给谁、怎么走最优链路、怎么重叠" -> 复杂易错
NCCL：       提供 all-reduce / broadcast 等原语，自动按拓扑选最优算法 -> 一行搞定
```

> 深度学习的分布式训练（数据并行）几乎都靠 NCCL 的 all-reduce 同步梯度。这是训练岗的核心
> 知识点。

## 0.1 集合通信原语速查表

| 原语 | 做什么 | 典型用途 |
|---|---|---|
| **Broadcast** | 一卡的数据复制到所有卡 | 分发初始权重 |
| **Reduce** | 所有卡的数据归约（求和等）到一卡 | 汇总到主卡 |
| **All-Reduce** | 归约后结果分发回所有卡 | **梯度同步（最常用）** |
| **All-Gather** | 每卡贡献一块，所有卡拿到全部 | 收集分片数据 |
| **Reduce-Scatter** | 归约后每卡拿一部分结果 | all-reduce 的一半 |

> "All-" 前缀的意思是"结果所有卡都有"；没有则结果只在一卡。

## 1. 为什么叫"集合"通信

**集合通信（collective communication）= 一组卡共同参与的通信操作**，区别于"点对点"（一对
一传）。所有参与的卡一起调用同一个操作，NCCL 协调它们完成数据交换：

```text
点对点：GPU0 把数据发给 GPU1（两个参与者）
集合：  所有 GPU 一起做 all-reduce（全体参与，结果全体共享）
```

## 2. 最重要的原语：All-Reduce

**All-Reduce = Reduce（归约）+ Broadcast（分发）**：所有卡各有一份数据，归约（如求和）后，
**每块卡都拿到最终结果**：

```text
初始：GPU0=[1] GPU1=[2] GPU2=[3] GPU3=[4]
All-Reduce(sum)：
结果：GPU0=[10] GPU1=[10] GPU2=[10] GPU3=[10]   （都拿到 1+2+3+4=10）
```

**为什么它是分布式训练的核心**：数据并行训练中，每卡用自己那批数据算出**梯度**，需要把所有
卡的梯度**求和平均**再让每卡更新——这正是 all-reduce：

```text
每卡算出本地梯度 -> All-Reduce(sum) 求和 -> 每卡拿到全局梯度 -> 各自更新权重
                    （NCCL 在这一步，是训练通信瓶颈）
```

## 3. NCCL 怎么高效：Ring All-Reduce

NCCL 不是简单"所有卡发给一卡再分发"（那样一卡的链路会成瓶颈）。它用 **Ring（环形）算法**：把
卡组成一个环，数据分块沿环流动，每卡同时收发，**充分利用每条链路、不浪费带宽**：

```text
朴素：所有卡 -> 主卡求和 -> 主卡分发    主卡链路成瓶颈，其他链路闲置
Ring： 卡组成环，数据分块沿环传递        每条链路都在用，带宽利用最优
```

意义：你不用关心算法细节，**NCCL 自动根据拓扑（NVLink/PCIe）选最优策略**。这正是不该自己
手搓多卡通信的原因——NCCL 已经把这些做到极致。

## 4. NCCL 基本用法

概念流程（实际 API 略繁，理解骨架即可）：

```cpp
// 1. 初始化：为每块卡创建 NCCL communicator
ncclComm_t comms[numGpus];
ncclCommInitAll(comms, numGpus, devs);

// 2. 每块卡发起 all-reduce（求和）
for (int g = 0; g < numGpus; ++g) {
    cudaSetDevice(g);
    ncclAllReduce(sendbuf[g], recvbuf[g], count, ncclFloat, ncclSum,
                  comms[g], stream[g]);
}

// 3. 同步等通信完成
for (int g = 0; g < numGpus; ++g) {
    cudaSetDevice(g);
    cudaStreamSynchronize(stream[g]);
}

// 4. 清理
for (int g = 0; g < numGpus; ++g) ncclCommDestroy(comms[g]);
```

要点：

```text
- communicator：NCCL 的通信上下文，每卡一个，初始化时建立
- 操作在 stream 上：NCCL 通信是异步的，可和计算重叠（卷八/07）
- 所有参与卡都要调用同一个集合操作，否则死锁
```

## 5. 其他原语什么时候用

```text
Broadcast：训练开始，把主卡的初始权重复制到所有卡
All-Gather：每卡算了一部分（如模型并行的分片输出），需要拼成完整结果
Reduce-Scatter：all-reduce 的前半（归约后每卡只拿一部分），配 all-gather 可实现 all-reduce
Reduce：只需要主卡拿到汇总结果（如打印全局 loss）
```

## 6. NCCL 的性能关键：拓扑与重叠

```text
1. 拓扑感知：NCCL 自动用 NVLink（快）而非 PCIe（慢），但前提是硬件有 NVLink（卷八/04）
2. 通信-计算重叠：把 all-reduce 放在独立 stream，和反向传播的计算重叠
   -> 训练中"算下一层梯度的同时同步上一层梯度"，藏住通信延迟（卷八/07）
3. 消息大小：太小的通信开销占比高，常把多个小梯度打包（bucketing）成大消息
```

> 实战：分布式训练框架（PyTorch DDP、Horovod）底层都用 NCCL，并做了梯度 bucketing + 通信
> 计算重叠。理解 NCCL 帮你看懂这些框架的性能行为。

## 7. 实践（需多卡 + NCCL）

1. （多卡）用 NCCL all-reduce 对每卡一个数组求和，验证每卡都拿到全局和。
2. 画出 4 卡 ring all-reduce 的数据流动（手绘），理解为什么每条链路都在用。
3. 估算：每卡 100MB 梯度、4 卡、NVLink 300 GB/s，一次 all-reduce 大概多久？
4. 思考：为什么分布式训练要把小梯度打包成大消息再 all-reduce？

## 8. 面试题（附参考答案）

**Q1：什么是集合通信？举几个原语。**
一组卡共同参与的通信操作（区别于点对点）。常见：Broadcast（一卡分发到所有）、Reduce（归约到
一卡）、All-Reduce（归约后所有卡都有结果）、All-Gather（每卡贡献一块、所有卡拿全部）。

**Q2：All-Reduce 是什么？为什么是分布式训练的核心？**
所有卡各有数据，归约（如求和）后每卡都拿到结果。数据并行训练中每卡算本地梯度，用 all-reduce
求和得到全局梯度再各自更新——这是训练的通信核心，也是主要瓶颈。

**Q3：NCCL 为什么比自己手写多卡通信好？**
NCCL 用 ring 等优化算法充分利用每条链路、自动感知拓扑选 NVLink/PCIe、支持和计算重叠。手写
难做到这些，且易出错。

**Q4：Ring All-Reduce 为什么高效？**
把卡组成环，数据分块沿环流动，每卡同时收发，所有链路都在用、不浪费带宽。比"汇总到一卡再分发"
（主卡链路成瓶颈）高效得多。

**Q5：怎么让 NCCL 通信不拖慢训练？**
把通信放独立 stream 和计算重叠（边算下一层梯度边同步上一层）、把小梯度打包成大消息（bucketing）
减少通信次数、确保走 NVLink。

## 9. 资料映射

- NCCL 官方文档；Ring All-Reduce 相关资料。
- 配套：[卷八第 05 章 多 GPU 数据划分](05_多GPU数据划分与同步.md)、[卷八第 07 章 计算通信重叠](07_计算通信重叠与Scaling.md)。
