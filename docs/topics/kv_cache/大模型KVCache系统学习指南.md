# 大模型 KV Cache 系统学习指南

> 目标：系统理解 LLM 推理里的 KV Cache。  
> 读完后你应该能回答：KV Cache 为什么存在、存什么、怎么估算显存、为什么 decode 阶段慢、PagedAttention 解决什么、MHA/MQA/GQA/MLA 如何减少 KV Cache、它和 CUDA 性能优化有什么关系。

---

## 0. 先给一个总直觉

大模型推理不是一次性算完。

生成式大模型通常是自回归的：

```text
给定 prompt。
模型生成第 1 个 token。
把第 1 个 token 接到输入后面。
模型生成第 2 个 token。
再接上。
继续生成第 3、4、5... 个 token。
```

每一步生成新 token 时，模型都要看前面所有 token。

如果没有 KV Cache，每一步都会反复重新计算历史 token 的 Key 和 Value。

KV Cache 的核心作用就是：

```text
历史 token 的 K/V 已经算过了。
不要每步重复算。
把它们缓存起来。
下一步 decode 直接读缓存。
```

一句话：

```text
KV Cache 用显存换计算。
```

它减少了重复计算，但带来了新的问题：

```text
显存占用很大。
decode 每步要读很多历史 KV。
长上下文和大 batch 会把 KV Cache 撑爆。
KV Cache 的布局会影响 attention kernel 的访存效率。
```

所以 KV Cache 是 LLM 推理系统的核心，不是一个小细节。

---

## 1. 先复习 Transformer Attention

### 1.0 用一个生活类比理解 attention

先别看公式。Attention（注意力）想解决的问题是：

```text
当模型读到当前这个词时，
它应该“关注”前面哪些词，关注多少？
```

举个例子，句子：

```text
小明把书放在桌子上，然后他离开了。
```

当模型处理 “他” 这个词时，它需要知道 “他” 指的是 “小明”。

attention 做的事就是：让 “他” 这个词，
去看前面所有词，给每个词打一个“相关性分数”，
然后按分数加权，把相关词的信息汇总过来。

```text
他 -> 小明: 分数高（很相关）
他 -> 书:   分数低
他 -> 桌子: 分数低
```

可以类比成查字典/搜索：

```text
Query  = 你要查的问题（“他”指谁？）
Key    = 每个词的“索引标签”（用来被匹配）
Value  = 每个词真正的“内容”（匹配上之后取出来的信息）
```

匹配过程：拿 Query 和每个词的 Key 比一比有多像（打分），
分数越高，就从那个词的 Value 里取越多信息。

记住这三件事，下面的公式就只是把这个过程数学化。

### 1.1 Attention 的输入

Transformer 每层里都有 attention。

给定输入 hidden states（可以理解为“每个 token 当前的向量表示”）：

```text
X: [batch, seq_len, hidden_size]

batch:       一次处理几条句子
seq_len:     句子里有几个 token
hidden_size: 每个 token 用多长的向量表示
```

通过三个不同的线性层（就是矩阵乘法 + 各自的权重）得到 Q、K、V：

```text
Q = X * Wq   # Query
K = X * Wk   # Key
V = X * Wv   # Value
```

也就是说：同一个 token，乘三套不同权重，
得到“它的提问”、“它的索引标签”、“它的内容”三个向量。

### 1.2 Attention 的核心公式

```text
scores = Q * K^T / sqrt(head_dim)
prob   = softmax(scores)
out    = prob * V
```

逐行解释：

```text
1) scores = Q * K^T
   让每个 token 的 Query 和所有 token 的 Key 做点积。
   点积越大 = 两个向量越“像” = 越相关。
   结果是一个 [seq_len, seq_len] 的分数矩阵：
   第 i 行第 j 列 = 第 i 个 token 对第 j 个 token 的关注分数。

2) / sqrt(head_dim)
   缩放。head_dim 越大，点积数值越大，
   除以 sqrt(head_dim) 防止数值过大导致 softmax 梯度消失。

3) softmax(scores)
   把每一行的分数变成“加起来等于 1 的概率”。
   分数高的变成大权重，分数低的变成小权重。

4) out = prob * V
   用这些权重，对所有 token 的 Value 做加权求和。
   得到当前 token “汇总了相关信息后”的新表示。
```

### 1.3 一个最小数字例子

假设只有 3 个 token，head_dim = 2（向量很短，方便手算）。

当前 token 的 Query：

```text
Q_cur = [1, 0]
```

三个历史 token 的 Key：

```text
K_1 = [1, 0]    # 和 Q 方向一致
K_2 = [0, 1]    # 和 Q 垂直
K_3 = [1, 0]    # 和 Q 方向一致
```

先算点积分数（先不除 sqrt 简化）：

```text
score_1 = 1*1 + 0*0 = 1
score_2 = 1*0 + 0*1 = 0
score_3 = 1*1 + 0*0 = 1
```

softmax 后（e^1≈2.718, e^0=1）：

```text
分母 = 2.718 + 1 + 2.718 = 6.436
w_1 = 2.718 / 6.436 ≈ 0.42
w_2 = 1     / 6.436 ≈ 0.16
w_3 = 2.718 / 6.436 ≈ 0.42
```

可见 token 1 和 token 3（和 Query 像）拿到更大权重，
token 2（不像）权重小。

最后 out = 0.42*V_1 + 0.16*V_2 + 0.42*V_3，
也就是主要把 token 1 和 token 3 的内容汇总过来。

### 1.4 Multi-Head（多头）是什么意思

上面只算了“一种关注方式”。实际模型会并行算很多套（多个 head）：

```text
head 1 可能关注“语法关系”
head 2 可能关注“指代关系”
head 3 可能关注“距离关系”
...
```

每个 head 有自己的一套 Wq/Wk/Wv，独立做一遍上面的 attention，
最后把所有 head 的输出拼接起来。

```text
num_heads:  有几套并行的注意力
head_dim:   每套用的向量维度
hidden_size ≈ num_heads * head_dim
```

这就是 Multi-Head Attention（MHA）。后面第 8 节讲的
MQA/GQA/MLA，本质都是在“head 怎么共享 K/V”上做文章。

### 1.5 直观总结

```text
Q:  当前 token 想问什么。
K:  每个历史 token 能被什么问题匹配到。
V:  每个历史 token 真正携带的信息内容。
```

attention 做的事是：

```text
当前 token 的 Q 和所有历史 token 的 K 做匹配（打分）。
softmax 把分数变权重。
对所有历史 token 的 V 加权求和。
```

对单个 query token：

```text
score_i = dot(Q_current, K_i)
out = sum_i softmax(score_i) * V_i
```

关键点来了：**decode 时每生成一个新 token，它都要访问所有历史 token 的 K 和 V。**

```text
所有历史 token 的 K
所有历史 token 的 V
```

历史 K/V 每步都要用，而且不会变，
所以与其每步重算，不如缓存起来 —— 这就是 KV Cache 重要的根源。

---

## 2. Prefill 与 Decode

LLM 推理通常分两段：

```text
prefill
decode
```

### 2.1 Prefill

prefill 是处理 prompt 的阶段。

例如用户输入：

```text
"请解释一下 CUDA shared memory"
```

这段 prompt 会被 tokenizer 切成很多 token。

假设：

```text
prompt_len = 1024
```

prefill 做的是：

```text
一次性处理这 1024 个 token。
算出最后一个位置的 logits。
同时为这 1024 个 token 生成并保存 K/V Cache。
```

prefill 的特点：

```text
token 多。
矩阵比较大。
GEMM/Attention 比较大。
更容易吃满 GPU。
```

### 2.2 Decode

decode 是逐 token 生成阶段。

生成第一个输出 token 后，下一步输入通常只新增 1 个 token。

每一步 decode 做：

```text
1. 对新 token 计算 Q/K/V。
2. 把新 token 的 K/V 写入 KV Cache。
3. 用新 token 的 Q 去 attend 所有历史 K/V。
4. 得到 logits。
5. sampling 得到下一个 token。
```

decode 的特点：

```text
每步只处理少量新 token。
batch 可能动态变化。
每步都要读历史 KV Cache。
容易 memory-bound。
小 kernel 多，launch overhead 明显。
```

这也是为什么推理系统经常说：

```text
prefill 主要看吞吐。
decode 主要看延迟和 KV Cache 读带宽。
```

---

## 3. 没有 KV Cache 会发生什么

假设已经有历史 token：

```text
t1, t2, t3, ..., tn
```

现在要生成第 `n+1` 个 token。

如果没有 KV Cache，模型需要重新对所有历史 token 计算：

```text
K1, K2, K3, ..., Kn
V1, V2, V3, ..., Vn
```

下一步生成第 `n+2` 个 token，又要重新计算：

```text
K1, K2, K3, ..., Kn, K(n+1)
V1, V2, V3, ..., Vn, V(n+1)
```

这样历史越长，重复计算越多。

KV Cache 做的是：

```text
第一次算出 Ki/Vi 后，把它们保存起来。
后面 decode 直接读。
```

所以 decode 每一步只需要为新 token 计算新的 K/V：

```text
K_new, V_new
```

然后 append 到 cache。

---

## 4. KV Cache 到底存什么

每一层 Transformer 都有自己的 attention。

所以每一层都需要保存历史 token 的 K 和 V。

KV Cache 不是只存一份，而是：

```text
每一层一份 K cache
每一层一份 V cache
```

常见形状可以写成：

```text
K cache:
  [num_layers, batch, num_kv_heads, seq_len, head_dim]

V cache:
  [num_layers, batch, num_kv_heads, seq_len, head_dim]
```

也有人写成：

```text
[num_layers, batch, seq_len, num_kv_heads, head_dim]
```

或者为了 kernel 访存更高效，框架会用更复杂的 block layout。

但逻辑上你先记住：

```text
每层
每个请求
每个历史 token
每个 KV head
每个 head_dim
都要存 K 和 V。
```

---

## 5. KV Cache 显存公式

最重要的公式：

```text
KV cache bytes =
num_layers
* batch_size
* seq_len
* num_kv_heads
* head_dim
* 2
* bytes_per_element
```

其中：

```text
num_layers:
  Transformer 层数。

batch_size:
  同时服务多少条请求。

seq_len:
  每条请求当前上下文长度。

num_kv_heads:
  K/V head 数量。

head_dim:
  每个 head 的维度。

2:
  K 和 V 两份。

bytes_per_element:
  FP16/BF16 是 2 bytes，FP8 是 1 byte。
```

注意这里是：

```text
num_kv_heads
```

不是一定等于 attention heads。

因为 MQA/GQA/MLA 会减少 KV head 数量。

---

## 6. 手算例子一：普通 MHA

假设模型：

```text
num_layers = 32
batch_size = 8
seq_len = 4096
num_kv_heads = 32
head_dim = 128
dtype = FP16
bytes_per_element = 2
```

KV Cache：

```text
bytes =
32 * 8 * 4096 * 32 * 128 * 2 * 2
```

一步步算：

```text
32 layers * 8 batch = 256
256 * 4096 = 1,048,576
1,048,576 * 32 heads = 33,554,432
33,554,432 * 128 = 4,294,967,296
4,294,967,296 * 2(K,V) = 8,589,934,592
8,589,934,592 * 2 bytes = 17,179,869,184 bytes
```

约：

```text
17.18 GB
```

这只是 KV Cache，不包括：

```text
模型权重
activation
workspace
CUDA context
allocator fragmentation
临时 buffer
```

所以 KV Cache 非常贵。

---

## 7. 手算例子二：GQA 减少 KV heads

假设模型其他参数不变，只把：

```text
num_kv_heads = 8
```

则：

```text
bytes =
32 * 8 * 4096 * 8 * 128 * 2 * 2
```

因为 KV heads 从 32 变 8，KV Cache 直接变成原来的：

```text
8 / 32 = 1/4
```

也就是约：

```text
17.18 GB / 4 = 4.29 GB
```

这就是 GQA/MQA/MLA 对推理很重要的原因：

```text
它们减少 KV Cache。
```

---

## 8. MHA、MQA、GQA、MLA

### 8.1 MHA

MHA 是 Multi-Head Attention。

通常：

```text
num_q_heads = num_kv_heads
```

例如：

```text
num_q_heads = 32
num_kv_heads = 32
```

每个 Q head 有自己的 K/V head。

优点：

```text
表达能力强。
```

缺点：

```text
KV Cache 大。
decode 读 KV 成本高。
```

### 8.2 MQA

MQA 是 Multi-Query Attention。

通常：

```text
num_q_heads 很多
num_kv_heads = 1
```

多个 Q head 共享同一组 K/V。

优点：

```text
KV Cache 极小。
decode 读 KV 成本低。
```

缺点：

```text
可能影响模型效果。
```

### 8.3 GQA

GQA 是 Grouped-Query Attention。

它在 MHA 和 MQA 之间折中。

例如：

```text
num_q_heads = 32
num_kv_heads = 8
```

每 4 个 Q heads 共享 1 个 KV head。

优点：

```text
KV Cache 比 MHA 小很多。
效果通常比 MQA 更稳。
```

所以很多现代 LLM 使用 GQA。

### 8.4 MLA

MLA 是 Multi-head Latent Attention。

简单理解：

```text
不是直接缓存完整 K/V。
而是缓存更压缩的 latent 表示。
需要时再恢复或参与 attention 计算。
```

它的目标是：

```text
进一步减少 KV Cache 显存。
```

对 DeepSeek 这类模型很重要。

初学阶段你先抓住：

```text
MHA:
  KV Cache 大。

MQA:
  KV Cache 最小，但表达能力可能损失。

GQA:
  折中，工程中很常见。

MLA:
  更进一步压缩 KV Cache，是 DeepSeek 相关重点。
```

---

## 9. 为什么 Decode 容易 Memory-Bound

decode 每步只有一个新 token。

对这个新 token，每层 attention 需要：

```text
读历史 K cache
计算 Q*K
softmax
读历史 V cache
加权求和
写新 K/V
```

其中读历史 K/V 的数据量随：

```text
seq_len
num_layers
num_kv_heads
head_dim
```

线性增长。

但每步新增 token 很少，所以大 GEMM 不够大，GPU 计算单元不一定吃满。

这就导致：

```text
decode 阶段经常不是 FLOPS 不够，
而是读 KV Cache 的带宽、访存布局、小 kernel launch 和调度开销限制性能。
```

用 Roofline 语言说：

```text
decode attention 的算术强度可能不高。
它很容易被 memory bandwidth 限制。
```

---

## 10. KV Cache 与 CUDA 访存

从 CUDA 角度看，KV Cache 是一个巨大的 global memory 数据结构。

attention kernel 要高效读取它，需要考虑：

```text
连续访问
对齐
coalescing
cache locality
page/block layout
batch 内请求长度不一致
```

如果每个 request 的 KV Cache 都是连续大数组，单个请求读起来比较简单。

但在线服务里，请求是动态的：

```text
请求 A 生成 50 token 后结束。
请求 B 还在生成。
请求 C 新加入。
请求 D 上下文很长。
```

如果为每个请求预留一整块连续最大长度 KV Cache，会浪费大量显存。

如果不断申请释放，又会造成：

```text
碎片
allocator 压力
复制成本
调度复杂
```

这就是 PagedAttention 出现的背景。

---

## 11. 连续 KV Cache

最简单的方式：

```text
每个请求分配一段连续 KV Cache。
```

例如：

```text
request A:
  token 0..4095 连续存放

request B:
  token 0..2047 连续存放
```

优点：

```text
实现简单。
kernel 访问逻辑简单。
连续内存更容易 coalescing。
```

缺点：

```text
在线服务里请求长度不同。
很难提前知道每个请求最终长度。
容易预留过多显存。
请求进出会产生碎片。
```

---

## 12. Paged KV Cache / PagedAttention

### 12.0 先理解“连续分配”为什么浪费

回顾第 11 节的问题：如果给每个请求预留一整块连续内存，
就必须按“最坏情况（最大长度）”预留。

举例：max_seq_len = 2048，但请求实际只生成了 40 个 token：

```text
[■■■ 已用 40 token ■■■][□□□□□□ 浪费 2008 token 的空间 □□□□□□]
```

成百上千个请求同时在线，浪费会非常惊人。
而且请求长度事先不知道，进进出出还会造成内存碎片。

### 12.1 PagedAttention 的核心思想

它借用了操作系统“虚拟内存分页”的思路：

```text
不要给每个请求分配一整块连续大内存。
把 KV Cache 切成固定大小的 block/page（比如每块 16 个 token）。
请求需要多少 token，就分配多少 block。
这些 block 在物理显存里不需要连续。
```

类比操作系统：

```text
OS 把内存切成固定大小的 page，
进程看到的是“连续的虚拟地址”，
但背后映射到不连续的物理 page。

PagedAttention 把 KV Cache 切成固定大小的 block，
请求看到的是“连续的 token 序列”，
但背后映射到不连续的物理 block。
```

### 12.2 一个具体的分配例子

假设 block size = 16 tokens。

请求 A 当前长度 40 tokens，需要：

```text
ceil(40 / 16) = 3 个 block
  block 容纳 token 0..15
  block 容纳 token 16..31
  block 容纳 token 32..39（还剩 7 个空位，下一步 decode 继续填）
```

系统有一个全局的“空闲 block 池”（free block pool），
分配时随便挑 3 个物理 block，比如挑到物理编号 7、3、9：

```text
逻辑 token 位置  ->  物理 block 编号
block 0 (token 0..15)   ->  物理 block 7
block 1 (token 16..31)  ->  物理 block 3
block 2 (token 32..39)  ->  物理 block 9
```

这张“逻辑到物理”的对照表就叫 **block table**：

```text
request A 的 block table = [7, 3, 9]
```

当请求 A 再生成一个新 token（第 40 个，从 0 数）时：

```text
第 40 个 token 落在 logical block 2（token 32..47）。
查 block table -> 物理 block 9。
写到物理 block 9 内部的第 (40 - 32) = 8 个槽位。
```

如果某个 logical block 写满了，就再从空闲池申请一个新物理 block，
追加到 block table 末尾即可。请求结束时，把这些物理 block 还回空闲池。

### 12.3 kernel 怎么读 paged KV

普通连续 KV：kernel 知道起始地址，直接顺着往下读。

paged KV：kernel 想读第 i 个历史 token 的 K/V，要多做一步地址翻译：

```text
1. logical_block = i / block_size      # 它在第几个逻辑块
2. offset        = i % block_size      # 块内第几个槽位
3. physical_block = block_table[logical_block]   # 查表得到物理块
4. 真实地址 = physical_block 起始地址 + offset * 每 token 字节数
```

也就是每次访问多了一层“查 block table”的间接寻址。

### 12.4 好处与代价

好处：

```text
减少显存浪费（不用按最大长度预留）。
支持请求动态增长（用多少分配多少）。
支持请求完成后回收 block 给别人用。
更适合 continuous batching（见第 13 节）。
```

代价：

```text
attention kernel 访问 KV 时多了一层地址映射（查 block table）。
物理内存不连续，访存模式更复杂，coalescing 更难保证。
kernel 需要针对 paged layout 专门优化。
```

所以 PagedAttention 是“推理系统”和“CUDA kernel”的交界点：

```text
系统层面：
  管理 KV block 的分配、回收、block table。

kernel 层面：
  按 block table 高效读取 K/V。
```

一句话总结面试版：

```text
PagedAttention 借鉴 OS 分页，把 KV Cache 切成固定大小 block，
用 block table 把逻辑 token 映射到不连续的物理 block，
从而按需分配、减少碎片和浪费、支持动态批处理，
代价是 kernel 多了一层地址间接寻址。
```

---

## 13. KV Cache 与 Continuous Batching

传统 static batching：

```text
凑一批请求。
一起跑。
等整批都结束。
再跑下一批。
```

问题：

```text
有的请求短，有的请求长。
短请求结束后还要等长请求。
GPU 利用率低。
延迟也不好。
```

continuous batching：

```text
每个 decode step 都可以加入新请求。
完成的请求立刻移除。
活跃请求动态组成 batch。
```

这对 KV Cache 管理提出要求：

```text
请求会不断进入和退出。
每个请求长度不同。
KV blocks 要能动态分配和回收。
```

所以：

```text
Paged KV Cache + Continuous Batching
```

经常一起出现。

---

## 14. KV Cache 与推理指标

推理服务常看：

```text
TTFT:
  Time To First Token，首 token 延迟。

TPOT:
  Time Per Output Token，每个输出 token 延迟。

tokens/s:
  每秒生成 token 数。

P99 latency:
  99% 请求的延迟上界。

GPU utilization:
  GPU 利用率。

memory utilization:
  显存使用率。
```

KV Cache 影响：

```text
TTFT:
  prefill 会创建 prompt 的 KV Cache。

TPOT:
  每步 decode 要读历史 KV Cache。

tokens/s:
  KV Cache 显存和带宽影响并发量。

P99:
  长上下文请求、显存碎片、cache 分配失败都会影响尾延迟。
```

所以做推理性能优化时，不能只看单个 kernel GFLOPS。

还要看：

```text
KV Cache 占了多少显存。
每步 decode 读了多少 KV。
请求调度是否让 GPU 空等。
batching 是否提高吞吐但牺牲延迟。
```

---

## 15. KV Cache Quantization

KV Cache 太大，所以可以量化。

常见思路：

```text
FP16/BF16 KV Cache:
  每个元素 2 bytes。

FP8 KV Cache:
  每个元素 1 byte。

INT8 KV Cache:
  每个元素 1 byte。
```

理论上，从 FP16 到 FP8：

```text
KV Cache 显存减半。
读带宽压力也可能减半。
```

但量化不是免费午餐。

需要考虑：

```text
scale 怎么存。
per-tensor / per-head / per-token scale。
反量化 dequant 的计算开销。
数值误差。
attention kernel 能否融合 dequant。
硬件是否对 FP8/INT8 友好。
```

如果 dequant 不能很好融合，可能出现：

```text
显存省了，
但多了 kernel 或多了计算，
最终不一定更快。
```

面试回答可以说：

```text
KV Cache quantization 主要减少显存占用和 memory bandwidth。
收益在长上下文和高并发场景更明显。
但要处理 scale、误差和 dequant 融合问题。
```

---

## 16. Sliding Window / Chunked Attention

有些模型不会让每个 token attend 全部历史。

可能只看最近一段：

```text
sliding window attention
```

例如只看最近 4096 tokens。

这样 KV Cache 逻辑上可以限制窗口：

```text
只保留最近 window_size 的 K/V。
```

优点：

```text
显存和带宽可控。
长文本 decode 更稳。
```

缺点：

```text
模型不能直接访问窗口外信息。
需要配合模型结构设计。
```

还有一些系统会做：

```text
chunked prefill
prefix cache
speculative decoding
```

这些都会和 KV Cache 管理交织在一起。

初学阶段先知道：

```text
KV Cache 不一定无限增长。
系统和模型可能用窗口、压缩、共享、分页、量化来控制成本。
```

---

## 17. Prefix Cache

很多请求可能共享相同前缀。

例如：

```text
系统提示词 system prompt
工具说明
长文档前缀
few-shot examples
```

如果每个请求都重新 prefill 这些相同前缀，会浪费。

prefix cache 的思想：

```text
相同 prefix 的 KV Cache 可以复用。
```

这样可以减少：

```text
prefill 时间
重复计算
GPU 成本
```

难点：

```text
如何判断 prefix 相同。
如何管理 cache 生命周期。
如何处理不同请求继续生成后的分叉。
如何避免 cache 占满显存。
```

prefix cache 更偏推理系统层，但它的底层资产仍然是 KV Cache。

---

## 18. KV Cache 的生命周期

一个请求的 KV Cache 生命周期：

```text
1. 请求进入系统。
2. prefill 处理 prompt，生成 prompt KV Cache。
3. decode 每步追加新 token 的 K/V。
4. 请求完成或取消。
5. 释放 KV blocks。
```

如果支持 PagedAttention：

```text
1. 从 free block pool 分配 blocks。
2. block table 记录逻辑位置到物理 block。
3. decode 需要更多空间时继续分配。
4. 请求结束后 blocks 回收到 pool。
```

工程上要注意：

```text
显存不足时怎么办？
请求是否排队？
是否抢占低优先级请求？
是否把 KV Cache offload 到 CPU？
是否限制最大上下文？
```

这些问题决定推理服务的稳定性。

---

## 19. KV Cache 与 CUDA Kernel 的关系

KV Cache 对 CUDA kernel 的影响主要有四个：

```text
1. 读带宽：
   decode attention 每步要读历史 K/V。

2. 数据布局：
   layout 决定 coalescing、cache locality、vectorized load。

3. 地址映射：
   paged KV 需要 block table，kernel 多一层 indirection。

4. batch 动态性：
   每个 request 长度不同，kernel 需要处理 ragged sequence。
```

如果你做 CUDA 算子优化，要关心：

```text
K/V 在内存里怎么排？
一个 warp 读哪些 token/head/dim？
访问是否连续？
是否能用 vectorized load？
是否能把 K/V tile 搬到 shared memory？
softmax 是否在线计算？
是否需要支持 page table？
```

这就是为什么 KV Cache 不只是推理系统概念，它也影响 attention kernel 设计。

---

## 20. 一个简化的 Decode Step

一个 decode step 可以粗略写成：

```text
输入：
  last_token
  history KV cache

每一层：
  1. RMSNorm
  2. QKV projection
  3. RoPE
  4. append new K/V to KV cache
  5. attention: Q attends to all cached K/V
  6. output projection
  7. FFN / MoE

最后：
  logits
  sampling
  next_token
```

其中和 KV Cache 最直接相关：

```text
QKV projection:
  产生新 token 的 K/V。

append:
  写入 KV Cache。

attention:
  读取历史 K/V。
```

如果用 nsys 看 timeline，你会看到很多 kernel：

```text
norm
linear
rope
attention
matmul
sampling
```

decode 的性能瓶颈可能来自：

```text
KV Cache 读带宽
小 batch GEMM/GEMV
kernel launch overhead
CPU scheduler
sampling
通信
```

---

## 21. KV Cache 常见优化方向

### 21.1 减少 KV Cache 大小

方法：

```text
GQA / MQA / MLA
KV Cache quantization
sliding window
prefix cache
KV compression
```

### 21.2 提高 KV Cache 读效率

方法：

```text
优化 layout
coalesced access
vectorized load
paged attention kernel
FlashAttention/FlashInfer 类 kernel
减少不必要的 layout transform
```

### 21.3 提高显存利用率

方法：

```text
Paged KV
block pool
连续批处理
合理 max_seq_len
显存水位控制
cache eviction
```

### 21.4 降低 decode 延迟

方法：

```text
CUDA Graph
kernel fusion
减少 CPU launch overhead
batch 调度
speculative decoding
```

---

## 22. 面试里怎么讲 KV Cache

如果面试官问：

```text
什么是 KV Cache？
```

可以这样答：

```text
KV Cache 是 LLM 自回归推理中缓存历史 token 的 Key 和 Value 的机制。
在 decode 阶段，每生成一个新 token，只需要计算这个新 token 的 Q/K/V。
历史 token 的 K/V 不需要重复计算，而是直接从 cache 读取。
它用显存换计算，显著降低 decode 重复计算成本。
但代价是显存占用随 layers、batch、seq_len、kv_heads、head_dim 线性增长，
并且 decode attention 每步需要读取历史 KV，所以容易受到显存带宽和 cache layout 影响。
```

如果面试官问：

```text
KV Cache 显存怎么算？
```

可以答：

```text
KV cache bytes =
num_layers * batch * seq_len * num_kv_heads * head_dim * 2 * bytes_per_elem。

其中 2 表示 K 和 V 两份。
如果是 FP16/BF16，bytes_per_elem 是 2。
GQA/MQA/MLA 会减少 num_kv_heads 或压缩 KV 表示，所以能显著减少 KV Cache。
```

如果面试官问：

```text
PagedAttention 解决什么？
```

可以答：

```text
在线推理中请求长度不同、动态进入退出。
如果每个请求分配连续最大长度 KV Cache，会浪费显存并产生碎片。
PagedAttention 把 KV Cache 切成固定大小 block，通过 block table 映射逻辑 token 到物理 block。
这样可以按需分配和回收 KV blocks，提高显存利用率，并支持 continuous batching。
代价是 attention kernel 需要处理非连续物理布局和额外地址映射。
```

如果面试官问：

```text
为什么 decode 阶段容易 memory-bound？
```

可以答：

```text
decode 每步通常只新增 1 个 token，计算规模小，不像 prefill 那样有大 GEMM。
但每层 attention 都需要读取所有历史 token 的 K/V Cache。
随着 seq_len 增加，读取 KV 的 bytes 线性增长。
所以 decode 的瓶颈经常不是算力峰值，而是 KV Cache 读带宽、访存布局、小 kernel launch 和调度开销。
```

---

## 23. 常见误区

### 误区一：KV Cache 是为了加速 prefill

不准确。

KV Cache 主要服务 decode。

prefill 会创建 prompt 的 KV Cache，但它本身通常还是要处理整个 prompt。

### 误区二：KV Cache 只影响显存

不对。

它还影响：

```text
decode 带宽
attention kernel layout
batching
P99 latency
最大并发
系统调度
```

### 误区三：KV Cache 越大越好

不对。

更大的 KV Cache 允许更长上下文和更高并发，但也会：

```text
占用显存
降低可服务 batch
增加 decode 读带宽
带来碎片和调度问题
```

### 误区四：量化 KV Cache 一定更快

不一定。

量化能减少 bytes，但可能增加：

```text
scale 读取
dequant 计算
额外 kernel
数值误差
```

只有 kernel 和硬件能很好支持时，才容易转化成实际加速。

---

## 24. 练习题

### 练习一：手算 KV Cache

模型：

```text
layers = 40
batch = 4
seq_len = 8192
kv_heads = 8
head_dim = 128
dtype = BF16
```

求 KV Cache 大小。

答案：

```text
bytes =
40 * 4 * 8192 * 8 * 128 * 2 * 2

40 * 4 = 160
160 * 8192 = 1,310,720
1,310,720 * 8 = 10,485,760
10,485,760 * 128 = 1,342,177,280
1,342,177,280 * 2 = 2,684,354,560
2,684,354,560 * 2 = 5,368,709,120 bytes

约 5.37 GB
```

### 练习二：MHA 改 GQA

如果：

```text
MHA kv_heads = 32
GQA kv_heads = 8
```

其他不变，KV Cache 变成原来的多少？

答案：

```text
8 / 32 = 1/4
```

### 练习三：FP16 KV 改 FP8 KV

如果 KV Cache 从 FP16 改成 FP8，理论显存变成多少？

答案：

```text
FP16 = 2 bytes
FP8 = 1 byte

理论上变成 1/2。
```

但要补充：

```text
还要考虑 scale 元数据、dequant 开销和精度损失。
```

### 练习四：为什么 Paged KV 更适合在线服务

请用自己的话解释：

```text
请求长度不同。
请求动态进入退出。
连续大块 KV Cache 会浪费显存。
Paged KV 可以按 block 分配和回收。
```

### 练习五：从 CUDA 角度看 KV Cache

请回答：

```text
Paged KV Cache 对 attention kernel 有什么影响？
```

参考答案：

```text
它让物理内存不再连续，kernel 需要根据 block table 找到实际 K/V block。
这增加地址计算和间接访问，也可能影响 coalescing 和 cache locality。
因此 paged attention kernel 需要专门优化访问模式。
```

---

## 25. 学习路线

建议按这个顺序学：

```text
1. Attention Q/K/V 基础。
2. Prefill 和 decode 区别。
3. KV Cache 存什么。
4. KV Cache 显存公式。
5. MHA/MQA/GQA/MLA。
6. Decode 为什么 memory-bound。
7. Continuous batching。
8. PagedAttention / Paged KV。
9. KV Cache quantization。
10. CUDA attention kernel 如何读取 KV Cache。
```

如果你做大模型推理性能工程师，必须掌握：

```text
KV Cache 显存手算
PagedAttention 思想
decode memory-bound 原因
GQA/MQA/MLA 对 KV Cache 的影响
KV Cache 和 attention kernel 访存的关系
```

如果你做 CUDA 算子优化工程师，重点补：

```text
KV layout
paged attention kernel
FlashAttention / FlashInfer 数据流
coalesced read K/V
online softmax
decode attention profiling
```

---

## 26. 本章小结

```text
KV Cache:
  缓存历史 token 的 K/V，避免 decode 阶段重复计算。

核心公式:
  layers * batch * seq_len * kv_heads * head_dim * 2 * bytes_per_elem

主要收益:
  减少重复计算，提高 decode 效率。

主要代价:
  占用大量显存，decode 读取 KV 容易 memory-bound。

减少 KV 的方法:
  MQA、GQA、MLA、量化、sliding window、prefix cache。

系统优化:
  PagedAttention、continuous batching、block pool。

CUDA 视角:
  KV Cache 是 attention kernel 的大规模 global memory 输入。
  layout、coalescing、paged address mapping 会直接影响性能。
```

最后记住一句话：

```text
KV Cache 是 LLM 推理从“模型计算”走向“系统工程”的分界线。
它不是单纯的缓存，而是决定显存、吞吐、延迟、batching 和 attention kernel 设计的核心结构。
```
