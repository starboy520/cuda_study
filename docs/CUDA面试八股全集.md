# CUDA 面试八股全集

> 系统性复习用。从硬件架构到 LLM Infra，覆盖面试高频概念。建议配合手写 kernel 一起记。
> 用法：先通读建立地图，再挑薄弱章节深挖，最后每条都能闭卷口述。

---

## 目录

1. GPU 架构基础
2. 执行模型（SIMT / warp / block / grid）
3. 内存模型与层次
4. 内存访问优化
5. 同步与一致性
6. Occupancy 与资源
7. 性能分析方法论
8. 归约 / 扫描 / 原子
9. GEMM 优化阶梯
10. Tensor Core 与混合精度
11. 异步与流
12. 多 GPU 与通信
13. Attention / Softmax / LLM 算子
14. LLM 推理系统
15. 高频快问快答

---

## 1. GPU 架构基础

### CPU vs GPU
```text
CPU：少核、大缓存、强分支预测、低延迟。为串行/低延迟设计。
GPU：众核、小缓存、海量线程、高吞吐。靠线程切换隐藏延迟。
核心思想：GPU 不减少延迟，而是用大量并行线程掩盖延迟（latency hiding）。
```

### SM（Streaming Multiprocessor）
```text
GPU 由若干 SM 组成，SM 是真正的计算单元。
每个 SM 含：CUDA core、Tensor Core、寄存器堆、shared memory/L1、warp scheduler、LSU、SFU。
block 被分配到某个 SM，整个生命周期不迁移。
```

### 关键单位
```text
GPC > TPC > SM > warp scheduler > CUDA core
一个 SM 通常 4 个 warp scheduler，每周期各发射一个 warp 的指令。
```

### 典型卡（要记牢）
```text
T4   sm_75  Turing   ~320 GB/s   FP16 Tensor Core
A100 sm_80  Ampere   ~1555 GB/s  TF32/FP16/BF16 TC, cp.async, 40/80GB
H100 sm_90  Hopper   ~3TB/s      FP8 TC, TMA, 分布式 shared memory
```

---

## 2. 执行模型（SIMT）

### 三级并行
```text
thread → warp(32) → block → grid
warp：32 线程，硬件调度最小单位，同一指令共同执行。
block：跑在同一 SM，可用 shared memory + __syncthreads 协作。
grid：所有 block，block 间不能直接同步（除 cooperative groups）。
```

### SIMT
```text
Single Instruction Multiple Thread：一个 warp 内 32 线程执行同一指令、不同数据。
和 SIMD 区别：SIMT 每线程有独立寄存器/PC，可独立分支（代价是分歧）。
```

### warp divergence（分支分歧）
```text
warp 内线程走不同分支 → 串行执行各分支，掩码屏蔽 → 变慢。
优化：按 warp 对齐分支、减少 if/else、用算术替代分支。
```

### 线程索引
```text
global_id = blockIdx.x * blockDim.x + threadIdx.x
grid-stride loop：for(i=gid; i<N; i+=blockDim.x*gridDim.x) —— 处理任意大 N。
```

---

## 3. 内存模型与层次

### 内存层次（从快到慢）
```text
寄存器     每线程私有，最快，~256/thread 上限
shared/L1  block 内共享，~100KB/SM，可编程缓存
L2         全 GPU 共享，几十 MB
global     DRAM/HBM，最大最慢，所有线程可见
constant   只读，64KB，有 cache，广播友好
texture    只读，空间局部性 cache
local      "本地"实为 global 的私有部分（寄存器溢出）
```

### 各内存特点
```text
寄存器：超快但有限，溢出→local（慢）。
shared：可编程 scratchpad，分 32 bank，注意 bank conflict。
constant：同一 warp 读同地址→单次广播；不同地址→串行。
global：合并访问决定带宽利用率。
```

### 寄存器溢出（register spilling）
```text
寄存器不够 → 编译器把变量放 local memory（实为 global）→ 慢。
ncu 看 spill loads/stores。tiling 太大易溢出。
```

---

## 4. 内存访问优化

### 合并访问（coalescing）
```text
warp 32 线程访问连续对齐地址 → 合并成少量内存事务 → 带宽最大化。
反例：跨步/随机访问 → 多事务 → 带宽暴跌。
经验：相邻线程读相邻地址。
```

### bank conflict
```text
shared 分 32 bank，warp 内多线程访问同 bank 不同地址 → 串行。
广播（同地址）不冲突；padding（+1）常用来错开。
```

### 向量化访问
```text
float4/int4 一次 16B，减指令、宽事务。要求 16B 对齐。
```

### AoS vs SoA
```text
SoA（结构体数组拆成多数组）更易合并访问，GPU 友好。
```

---

## 5. 同步与一致性

```text
__syncthreads()    block 内 barrier，所有线程到齐才继续。
__syncwarp()       warp 内同步。
__shfl_*_sync      warp 内寄存器交换，无需 shared。
atomicAdd 等       原子操作，避免竞争但有争用开销。
__threadfence()    内存可见性，保证写对其他线程可见。
volatile           防编译器优化掉读写。
cooperative groups grid 级同步。
```

### memory fence vs barrier
```text
barrier(__syncthreads)：等线程，且隐含 fence。
fence(__threadfence)：只保顺序可见性，不等线程。
```

---

## 6. Occupancy 与资源

```text
Occupancy = 活跃 warp / SM 最大 warp。
限制因素：寄存器/thread、shared/block、block 大小。
高 occupancy 利于隐藏延迟，但不等于高性能。
低 occupancy + 高 ILP/复用 也可能更快（register tiling）。
经验：60~70% 常够；先看瓶颈再调。
```

---

## 7. 性能分析方法论

### bound 判定
```text
memory-bound：带宽吃满、算力闲（reduce/elementwise/decode）。
compute-bound：算力吃满（大 GEMM、Tensor Core）。
判据：AI = FLOP/bytes，对比 Roofline 拐点。
```

### Roofline
```text
AI=FLOP/bytes；roof=min(峰值算力, AI×带宽)。
AI 小→memory-bound；AI 大→compute-bound。
```

### 工具
```text
nsys：时间线，kernel/拷贝/流重叠、launch 开销。
ncu ：单 kernel 微观，看 SM busy、DRAM/L2 吞吐、occupancy、寄存器、stall。
```

### GFLOPS / GB/s
```text
GFLOPS = 2*M*N*K / time / 1e9（GEMM）
GB/s   = bytes / time / 1e9
利用率 = 实测 / 峰值
```

---

## 8. 归约 / 扫描 / 原子

```text
reduce：tree → sequential addressing → first-add-during-load → warp shuffle → grid-stride。
warp shuffle：__shfl_down_sync 寄存器归约，无 shared 无 syncthreads。
两级归约：warp 内 + warp 间(shared)。
scan：Hillis-Steele(简单/多算) / Blelloch(高效/work-efficient)。
原子：global atomic 慢→shared 分层聚合→warp shuffle 收尾。
```

---

## 9. GEMM 优化阶梯（核心）

```text
naive：每元素从 global 读 A/B，AI 极低。
shared tiling：tile 搬 shared，提高复用。
1D register tiling：一 thread 多输出，寄存器复用（过渡）。
2D register tiling：thread 算 TM×TN，regM×regN 外积，主线。
vectorized load：float4 减指令。
double buffering：cp.async 计算/搬运重叠。
warp tiling：block/warp/instruction 三级 tile。
Tensor Core：MMA 指令，对标 cuBLAS/CUTLASS。
```

---

## 10. Tensor Core 与混合精度

```text
CUDA core：标量 FMA。Tensor Core：矩阵块 MMA（D=A*B+C）。
WMMA：fragment 加载 16x16x16，做 MMA。
精度：FP32 / TF32(A100默认) / FP16 / BF16(宽指数,训练) / FP8(Hopper,需scaling) / INT8。
FP8：E4M3/E5M2，要 scale/amax，省带宽但难训。
```

---

## 11. 异步与流

```text
stream：操作队列，同流串行、异流可并行。
异步拷贝 + kernel 重叠：计算与传输重叠。
pinned memory：页锁定，加快 H2D/D2H 且可异步。
event：计时与依赖。
cp.async(Ampere)：global→shared 不过寄存器，配 double buffering。
CUDA Graph：固化 launch 序列，省启动开销。
```

---

## 12. 多 GPU 与通信

```text
并行：data / tensor / pipeline / expert / context parallel。
NCCL：all-reduce / all-gather / reduce-scatter / all-to-all / broadcast。
all-reduce = reduce-scatter + all-gather。
通信重叠：边算边传，bucket、反向重叠。
MoE：all-to-all 做 dispatch/combine。
```

---

## 13. Attention / Softmax / LLM 算子

```text
softmax：减max稳定 → exp → 归一；两次归约。
online softmax：流式更新 m,s，单遍。
FlashAttention：分块+online softmax，不存完整 S=QK^T，IO-aware 省显存。
LayerNorm：两次归约(μ,σ²)；一遍法 σ²=E[x²]-μ²；融合 add+norm。
RMSNorm：去均值，x/rms*γ，LLM 常用。
KV cache：缓存历史 K/V，decode 不重算。
MLA/GQA/MQA：减少 KV cache。
RoPE：旋转位置编码。
```

---

## 14. LLM 推理系统

```text
prefill：并行算 prompt，compute-bound。
decode：逐 token，batch=1 时 memory-bound（GEMV）。
KV cache 显存 = 2*层*头*头维*序列*batch*精度。
paged attention：分页管理 KV，减碎片(vLLM)。
量化：INT8/INT4，per-tensor/channel，GPTQ/AWQ。
框架：vLLM / TensorRT-LLM / SGLang。
```

---

## 15. 高频快问快答

```text
Q warp 是什么？        32线程，SIMT 调度最小单位。
Q 分支分歧为何慢？      warp 内分支串行+掩码。
Q coalescing？         相邻线程相邻地址→合并事务。
Q bank conflict？      同 bank 不同地址串行，padding 解。
Q occupancy 100% 一定快？ 否，可能寄存器复用更值。
Q reduce 怎么优化？     shuffle+grid-stride+向量化。
Q GEMM naive 为何慢？  global 重复读，AI 低。
Q shared tiling 解决啥？ 提高 global 复用。
Q register tiling？     thread 多输出，寄存器外积复用。
Q Tensor Core？        矩阵 MMA 硬件。
Q FP8 为何难？          范围小，需 scaling/amax。
Q online softmax？      m,s 流式重缩放。
Q FlashAttention 省啥？ 不存完整 attention 矩阵，IO-aware。
Q decode 为何 mem-bound？ batch小、GEMV、KV 频繁读。
Q all-reduce vs reduce-scatter？ 前者全得和，后者各得一段。
Q memory-bound 怎么优化？ 减字节、合并、向量化、复用。
Q compute-bound 怎么优化？ 提复用、Tensor Core、降精度。
```

---

## 复习建议

```text
1. 每条概念都能闭卷口述。
2. 配手写：reduce/transpose/softmax/GEMM/WMMA 骨架要能默写。
3. 每个 kernel 会算 GFLOPS、GB/s、AI 并判 bound。
4. ncu/nsys 看哪些指标、怎么解释要练熟。
```
