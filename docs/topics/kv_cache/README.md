# KV Cache 专题

本目录围绕 KV cache 的容量账、decode 数据流、分页管理和系统取舍组织材料。四份文档职责不同，建议按下表使用，而不是把它们当成四篇重复的入门介绍。

## 四份材料的职责

| 材料 | 主要职责 | 适合回答的问题 |
| --- | --- | --- |
| [大模型 KV Cache 系统学习指南](大模型KVCache系统学习指南.md) | 建立全局系统地图：KV cache 为什么存在、如何随请求增长、与推理调度和显存管理怎样关联 | KV cache 在 Transformer 推理链路中处于什么位置？系统瓶颈会怎样变化？ |
| [kv_cache_accounting](kv_cache_accounting.md) | 做定量账：按层数、KV heads、head dimension、序列长度、batch 和 dtype 计算容量与带宽 | 一个 token/请求/整层 KV cache 占多少字节？显存还能容纳多少请求？ |
| [decode_step_dataflow](decode_step_dataflow.md) | 追踪单步 decode 的数据流，连接 GEMV、attention、KV 读取、调度和 kernel 级性能分析 | decode 每一步读写什么？为什么低 batch 容易受 HBM 带宽限制？ |
| [PagedAttention 详解](PagedAttention详解.md) | 解释 block pool、logical block、physical block 和 block table，以及分页管理对碎片和共享的影响 | 为什么 KV cache 可以物理分散却逻辑连续？PagedAttention 解决了什么、没有解决什么？ |

## 适用边界

- 完整 dense prefill 的 attention 交互在序列长度方向具有二次规模，重点是理解全序列 token 两两交互带来的工作量增长；不要把它等同于所有 prefill 阶段开销都必然是二次的。
- 单步 decode 只有一个 query，需要对历史 KV 做线性读取；因此序列越长，单步 attention 的读取和计算量通常随历史长度线性增长，低 batch 时更容易暴露带宽瓶颈。
- PagedAttention 不改变同一 attention 工作负载的数学 FLOP，只改变 KV 的存储组织、物理块映射和寻址方式；实际性能仍可能受到间接寻址、block table 访问和调度管理开销影响。

## PagedAttention 两天学习顺序

当前学习主线见 [四周聚焦计划：AI Infra 与 CUDA 深水区](../../../study_plan/四周聚焦计划_AIInfra与CUDA深水区.md)；本专题只负责 KV Cache、decode 和 PagedAttention 相关材料的阅读顺序与边界。

### Day 1：先建立连续 KV cache 的容量和数据流基线

1. 先读 [大模型 KV Cache 系统学习指南](大模型KVCache系统学习指南.md)，定位 KV cache 在 prefill、decode 和请求生命周期中的作用。
2. 再读 [kv_cache_accounting](kv_cache_accounting.md)，用一个具体模型配置手算单层、单请求和 batch 级 KV cache 字节数，并记录 dtype、KV heads 和序列长度的影响。
3. 最后读 [decode_step_dataflow](decode_step_dataflow.md)，把上一阶段的字节账放回单步 decode 数据流，画出 KV 的读取位置和可能的带宽瓶颈。
4. 当天输出：一张容量表 + 一张 decode 数据流图；暂时不要跳到 block table，先能解释连续分配为什么会产生预留浪费、碎片和搬迁问题。

### Day 2：从连续分配过渡到分页映射

1. 按 [PagedAttention 详解](PagedAttention详解.md) 的顺序阅读：连续 KV 的问题 → 虚拟内存类比 → physical/logical block → block table → attention 访问路径。
2. 手算一个 `block_size=2` 或 `block_size=4` 的 block table，明确逻辑块连续、物理块可以分散，以及最后一个 block 的内部碎片。
3. 回看 [kv_cache_accounting](kv_cache_accounting.md) 和 [decode_step_dataflow](decode_step_dataflow.md)，说明分页改变的是存储管理与地址翻译，不是 attention 的 FLOP；同时列出 block table、间接寻址和调度管理的代价。
4. 当天输出：一份连续分配 vs PagedAttention 的对照表，以及一段能在面试中讲清“解决什么、没有解决什么”的口述答案。

> 一句话记忆：**Day 1 算清 KV cache 并追踪 decode 数据流；Day 2 用 block table 把逻辑连续映射到物理分散，并解释收益与代价。**
