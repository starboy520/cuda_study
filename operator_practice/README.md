# 算子练习集 (Operator Practice)

> 目标：把 AI Infra / CUDA 面试高频算子逐个手写、benchmark、profile、口述。
> 每个算子按统一流程：**写代码 → 验证正确性 → benchmark → ncu 分析 → 面试口述**。

## 统一完成标准（每个算子至少做到）

```text
代码：能编译运行的 .cu，含 CPU reference 校验
数据：512 / 1024 / 更大规模的 benchmark（time_ms + GB/s 或 GFLOPS）
分析：是 memory-bound 还是 compute-bound？瓶颈在哪？
口述：一段面试风格回答（这个 kernel 慢在哪、怎么优化、为什么变快）
```

## 算子路线图（按优先级 / 难度排序）

### 1. 归约 / 扫描类（Reduction & Scan）
| 算子 | 说明 | 关键技术 | 状态 |
|------|------|----------|------|
| reduce (sum) | 求和归约 | tree reduction、warp shuffle、向量化 load | ✅ |
| reduce (max/min) | 极值归约 | 同 sum，注意初始值 | ⬜ |
| prefix sum (scan) | 前缀和 | Hillis-Steele / Blelloch、block 级 + 全局合并 | ⬜ |
| argmax / argmin | 带索引归约 | 同时归约值和下标 | ✅ |
| top-k | 取前 k 大 | 局部堆 / bitonic / 多轮归约 | ⬜ |

### 2. 归一化类（Normalization）
| 算子 | 说明 | 关键技术 | 状态 |
|------|------|----------|------|
| LayerNorm | 逐行均值/方差归一化 | 两遍归约 (mean, var)、warp/block reduce | ✅ |
| RMSNorm | 仅用均方根 | 一遍平方和归约，LLM 常用 | ✅ |
| BatchNorm | 批维度归一化 | 跨 batch 归约、running stats | ⬜ |

### 3. 激活 / 逐元素类（Activation & Element-wise）
| 算子 | 说明 | 关键技术 | 状态 |
|------|------|----------|------|
| ReLU | max(0,x) | element-wise、float4 读写双向量化 | ✅ |
| GELU | 高斯误差线性单元 | element-wise、向量化 load (float4) | ⬜ |
| SiLU / Swish | x * sigmoid(x) | 同上 | ⬜ |
| fused element-wise | 多算子融合 | 减少 kernel launch 和访存 | ⬜ |

### 4. Softmax / Attention 扩展
| 算子 | 说明 | 关键技术 | 状态 |
|------|------|----------|------|
| softmax (warp/block) | 数值稳定 softmax | max 归约 + exp + sum 归约 | ⬜ |
| online softmax | 单遍 softmax | 流式更新 m / s | ⬜ |
| RoPE | 旋转位置编码 | 逐元素复数旋转 | ⬜ |
| FlashAttention-2 思路 | 分块 attention | online softmax + tiling | ⬜ |
| GQA / MQA | 分组/多查询注意力 | KV 头共享 | ⬜ |

### 5. 量化类（Quantization）
| 算子 | 说明 | 关键技术 | 状态 |
|------|------|----------|------|
| quantize / dequantize | FP↔INT 转换 | per-tensor / per-channel scale、amax | ⬜ |
| INT8 GEMM | 量化矩阵乘 | dp4a / Tensor Core INT8 | ⬜ |

### 6. MoE 类
| 算子 | 说明 | 关键技术 | 状态 |
|------|------|----------|------|
| top-k routing | 专家路由 | top-k 选择 + 负载统计 | ⬜ |
| dispatch / combine | token 分发/合并 | gather/scatter、all-to-all | ⬜ |
| grouped GEMM | 分组矩阵乘 | 每个专家一段 GEMM | ⬜ |

### 7. 通信类（Multi-GPU，理解为主）
| 算子 | 说明 | 关键技术 | 状态 |
|------|------|----------|------|
| all-reduce | 全归约 | ring / tree、NCCL | ⬜ |
| all-to-all | 全交换 | MoE dispatch 用 | ⬜ |

### 8. 采样类（Sampling）
| 算子 | 说明 | 关键技术 | 状态 |
|------|------|----------|------|
| softmax + sampling | 概率采样 | 累积分布 + 随机数 | ⬜ |
| top-p (nucleus) | 核采样 | 排序/阈值截断 | ⬜ |

## 目录约定

```text
operator_practice/
  README.md                 <- 本文件（路线图 + 进度）
  reduce/                   <- 今天从这里开始
    reduce_sum.cu
    benchmark.md
    ncu_notes.md
  layernorm/
  gelu/
  ...
```

> 进度标记：⬜ 未开始 / 🟡 进行中 / ✅ 完成
