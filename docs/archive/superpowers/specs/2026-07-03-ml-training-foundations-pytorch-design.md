# ML 训练基础与 PyTorch 入门教材设计

## 1. 目标与读者

将现有 `docs/ML基础_训练侧入门.md` 从训练侧名词概览扩写成一份可顺序自学的中文教材。

目标读者：

- C++/CUDA 基础较强；
- 已学过 GEMM、reduction、softmax、LayerNorm、Attention、Tensor Core 与基础 profiling；
- 没有系统机器学习基础；
- 希望真正理解训练原理，逐步学习 PyTorch，并最终能训练一个迷你语言模型；
- 同时需要覆盖 CUDA / AI Infra 岗位所需的训练侧知识。

教材完成后，读者应能：

1. 用数字手算一次 forward、loss、gradient 和参数更新；
2. 解释导数、偏导、梯度、链式法则和反向传播；
3. 读写基础 PyTorch Tensor、autograd、Module、optimizer 和 DataLoader；
4. 独立训练二维分类器和迷你语言模型；
5. 推导 Linear 等核心算子的 backward shape；
6. 估算训练参数、梯度、优化器状态和激活显存；
7. 解释混合精度、gradient accumulation、checkpointing、DDP、FSDP/ZeRO 等训练系统概念；
8. 把 PyTorch 操作映射到 GEMM、reduction、elementwise、通信与 CUDA profiler 指标。

## 2. 对现有文档的诊断

现有文档的优点：

- 主线正确；
- 语言简洁；
- 能把训练概念连接到 CUDA；
- 已覆盖 weights、loss、gradient descent、backprop、optimizer、训练循环及训练/推理区别。

现有文档的主要缺口：

- 未先建立“模型是带参数函数”的完整概念；
- 没有贯穿全文的可手算数字例子；
- logits、probability、label、cross entropy 之间跳跃较快；
- 导数、偏导、链式法则仅给结论，没有逐步演算；
- 缺少 Linear/ReLU/MLP 的具体 forward/backward；
- 缺少 batch、epoch、step、micro-batch、gradient accumulation 的清晰区分；
- 缺少 autograd、`nn.Module`、optimizer、Dataset/DataLoader 的 PyTorch 实践；
- 训练显存没有按字节展开；
- 混合精度、loss scaling、AdamW、多卡训练过于速查化；
- 缺少真正训练成功的小项目和系统性能分析闭环。

改写时保留现有直觉和 CUDA 类比，但不再允许关键概念只用一句话带过。

## 3. 总体教学结构

采用一份主文档、四卷递进结构：

```text
第一卷：完全零基础的训练原理
  函数 → 参数 → 预测 → loss → 导数 → gradient → backprop

第二卷：PyTorch 对应实现
  Tensor → autograd → Module → optimizer → Dataset/DataLoader

第三卷：迷你语言模型
  token → logits → cross entropy → causal model → Transformer → 训练/生成

第四卷：CUDA / AI Infra
  backward GEMM → 显存账本 → 混合精度 → profiling → 多卡训练
```

内容顺序必须遵守依赖关系。读者在理解手算 gradient 前，不引入 autograd；在理解分类 cross entropy 前，不进入语言模型；在跑通单卡训练前，不进入多卡并行。

## 4. 统一教学模板

每个核心主题尽量包含：

1. 为什么需要；
2. 日常直觉；
3. 严格定义；
4. 小数字手算；
5. shape 与 dtype；
6. 普通 Python 或 PyTorch 演示；
7. 必须手写练习；
8. 分级提示；
9. 参考实现；
10. 常见错误；
11. 自测题与答案；
12. CUDA/系统映射；
13. 面试口述。

代码标记沿用 Week 4 教材：

- `【演示】`：完整可运行；
- `【必须手写】`：提供接口、数据和验收；
- `【提示 1/2/3】`：逐级解锁；
- `【参考实现】`：练习完成后查看；
- `【挑战】`：独立扩展。

## 5. 第一卷：训练原理从零开始

### 5.1 模型是带参数的函数

从 `y=wx+b` 开始解释：

- 输入 `x`；
- 参数 `w/b`；
- 预测 `y_hat`；
- 目标 `y`；
- 参数与超参数的区别；
- 模型结构与参数数值的区别。

### 5.2 完整数字训练例子

固定例子：

```text
x=2, y=5
初始 w=1, b=0
y_hat=wx+b=2
loss=(y_hat-y)^2=9
```

逐步计算 `dL/dy_hat`、`dy_hat/dw`、`dy_hat/db`、`dL/dw`、`dL/db`，执行一次 SGD 更新并观察 loss 下降。

### 5.3 导数与链式法则

不假设微积分基础，依次解释：

- 斜率和局部变化率；
- 一元导数；
- 多变量偏导；
- gradient 向量；
- 链式法则；
- 计算图中的局部导数与上游梯度。

加入有限差分：

```text
dL/dw ≈ [L(w+eps)-L(w-eps)]/(2eps)
```

用于检查手算或实现的 gradient。

### 5.4 向量、矩阵与 batch

从 scalar Linear 递进到：

```text
y = xW+b
X[B,Din] × W[Din,Dout] + b[Dout] → Y[B,Dout]
```

解释 batch 为什么让大量样本变成 GEMM，以及 bias broadcasting 的含义。

### 5.5 多层网络与反向传播

用 `Linear→ReLU→Linear→loss` 建计算图，推导：

- ReLU forward/backward；
- MSE；
- `dX/dW/db` 的 shape；
- backward 为什么反向遍历；
- activation 为什么要保存或重算。

### 5.6 训练循环基础

解释 sample、batch、mini-batch、micro-batch、step、epoch、learning rate、shuffle、validation、overfitting。

提供普通 Python 微型训练循环，不依赖 PyTorch，让 `w/b` 从错误值逐步学习到正确值。

## 6. 第二卷：PyTorch 自然过渡

### 6.1 Tensor

从 CUDA 数组类比切入，覆盖：

- scalar/vector/matrix；
- shape、dtype、device、stride；
- CPU↔A100；
- reshape、transpose、contiguous；
- broadcast；
- matmul；
- FP32/FP16/BF16。

### 6.2 Autograd

使用第一卷同一个 `y=wx+b`：

- `requires_grad`；
- 计算图和 `grad_fn`；
- `loss.backward()`；
- leaf tensor；
- `.grad`；
- gradient accumulation；
- `zero_grad(set_to_none=True)`；
- `no_grad` 与 `detach`；
- in-place 操作和断图问题。

PyTorch gradient 必须与手算和有限差分三方对齐。

### 6.3 Module 与 Parameter

递进：

```text
Tensor 手写 Linear
→ 自定义 Linear Module
→ nn.Linear
→ 两层 MLP
```

解释 `Parameter`、`model.parameters()`、`forward`、`model(x)`、module hook 基本概念、`state_dict`、`train/eval`。

### 6.4 Loss 与 optimizer

覆盖：

- MSE；
- binary/class cross entropy；
- logits 与 probability；
- 为什么 `CrossEntropyLoss` 直接接 logits；
- 手动 SGD；
- `torch.optim.SGD`；
- momentum；
- Adam；
- AdamW 与 decoupled weight decay。

### 6.5 Dataset 与 DataLoader

覆盖：

- Dataset 接口；
- DataLoader；
- shuffle、batch、drop_last、sampler；
- worker、pinned memory、non_blocking transfer；
- micro-batch 和 gradient accumulation；
- CPU 数据管线瓶颈。

### 6.6 二维分类项目

模型：`Linear→ReLU→Linear`。必须展示：

- loss 下降；
- accuracy 上升；
- 参数发生变化；
- checkpoint 保存/恢复；
- CPU 与 A100 运行；
- profiler 中的 matmul/elementwise/optimizer 操作。

## 7. 第三卷：迷你语言模型

### 7.1 语言模型 loss

从小词表手算 logits、softmax、correct-token probability、negative log likelihood 和 cross entropy。

说明：

- logits shape；
- label shape；
- batch/sequence reduction；
- `ignore_index`；
- perplexity；
- `CrossEntropyLoss` 内部已经包含稳定 log-softmax。

### 7.2 文本数据集

覆盖：

- 字符级或极简 tokenizer；
- vocabulary 和 token id；
- context length；
- input/target 错一位；
- teacher forcing；
- causal mask；
- train/validation split。

### 7.3 模型递进

```text
Embedding + Linear
→ Embedding + MLP
→ 单头 causal Attention
→ Multi-Head Attention
→ Transformer Block
→ 两层迷你 GPT
```

每一级说明 shape、参数量、forward、需要保存的 activation、主要 CUDA 算子和验收方法。

### 7.4 完整训练与生成

必须包含：

- 可复现随机种子；
- 配置；
- 训练与 validation loss；
- checkpoint 保存和恢复；
- greedy 或 temperature sampling；
- 文本生成；
- 参数量与显存估算；
- 过拟合一个小 batch 的诊断测试。

项目目标是验证训练链路，而不是追求生成质量。

## 8. 第四卷：训练侧 CUDA / AI Infra

### 8.1 Backward 与 GEMM

对 `Y=XW` 推导：

```text
dX = dY W^T
dW = X^T dY
db = reduce_rows(dY)
```

逐项核对 shape、FLOP 和布局。继续介绍 ReLU、LayerNorm、softmax/cross-entropy、Attention backward 的组成。

### 8.2 训练显存账本

分别计算：

- 参数；
- 梯度；
- FP32 master weights；
- Adam `m/v`；
- activations；
- temporary workspace；
- communication buffers。

解释 activation checkpointing、gradient accumulation、ZeRO/FSDP、offload 分别省哪部分、付出什么代价。

### 8.3 混合精度训练

按顺序讲：

```text
FP32 baseline
→ FP16 overflow/underflow
→ loss scaling
→ BF16 动态范围
→ autocast
→ GradScaler
→ FP32 master weights
```

连接 A100 Tensor Core 和已有 WMMA 知识。

### 8.4 训练性能分析

工具分工：

- PyTorch profiler：operator、shape、CPU/GPU 时间；
- Nsight Systems：DataLoader、CPU launch、kernel、通信时间线；
- Nsight Compute：关键 kernel 的吞吐、访存、occupancy。

指标覆盖 step time、tokens/s、MFU、显存峰值、通信占比。强调训练性能不能只由单 kernel GFLOPS 判断。

### 8.5 多卡训练

递进：Data Parallel/DDP、Tensor Parallel、Pipeline Parallel、Sequence/Context Parallel、ZeRO/FSDP。

每种方式统一回答：切分对象、每卡保存内容、collective、通信时机、解决计算还是显存、主要瓶颈。

### 8.6 面试映射

各章加入概念题、shape 题、显存题、profiler 题、系统取舍题和 2–3 分钟口述模板。

## 9. 代码与练习产出

主教材继续使用：

```text
docs/ML基础_训练侧入门.md
```

建议学习者按教材创建：

```text
ml_training_basics/
├── 01_scalar_training.py
├── 02_autograd.py
├── 03_mlp_classifier.py
├── 04_dataloader.py
├── 05_mini_lm.py
├── 06_mini_gpt.py
├── profile_training.py
└── notes.md
```

教材可以完整提供演示和参考实现，但核心练习默认只提供骨架与提示，避免变成复制运行。

## 10. 验证策略

### 数学验证

- 手算结果与代码一致；
- autograd 与解析 gradient 一致；
- gradient 与有限差分在容差内一致；
- 所有 shape 和参数量可复算。

### 训练验证

- scalar regression loss 下降并学到目标参数；
- 分类器 loss 下降、accuracy 上升；
- 迷你语言模型能过拟合小 batch；
- validation loss 有记录；
- checkpoint 恢复后训练连续。

### 工程验证

- 所有完整 Python 示例可运行；
- 无 A100 时 CPU fallback 可运行；
- A100 路径明确 dtype/device；
- profiler 命令可执行；
- 不把 profiler 插桩时间当真实性能；
- 外部事实优先引用 PyTorch、NVIDIA 和原始论文。

## 11. 范围边界

本教材不追求：

- 完整微积分课程；
- 从零证明所有矩阵微分公式；
- 训练有实际能力的大语言模型；
- 完整实现 PyTorch autograd engine；
- 深入复刻 Megatron/FSDP/DeepSpeed 内部源码；
- 立即实现工业级训练 kernel。

数学深度以“能理解训练、读 PyTorch、推导 shape、分析 CUDA/系统性能”为准。

## 12. 完成标准

1. 零 ML 基础可从第一节顺序阅读，无未定义核心术语；
2. 至少有一个贯穿的手算训练例子；
3. 导数、链式法则、backprop 有数字演算；
4. PyTorch API 均能对应前面已理解的概念；
5. 分类器与迷你语言模型有完整可验证训练闭环；
6. backward GEMM shape 推导准确；
7. 训练显存按字节分项计算；
8. 混合精度和多卡概念说明收益、代价与边界；
9. 练习、提示、答案分层清晰；
10. 所有完整代码经过实际运行或明确标注未验证环境；
11. 同时满足真正理解原理、逐步学习 PyTorch、覆盖 CUDA/AI Infra 面试三个目标。
