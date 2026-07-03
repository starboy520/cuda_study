# ML 基础：训练侧入门（给 CUDA 工程师的补课）

> 面向对象：会 C++/CUDA，但几乎没碰过机器学习的你。
> 定位：Week4 Attention 文档专注**推理侧 CUDA 优化**，刻意跳过了"模型怎么训练出来的"。
> 这份文档补上那部分——**不需要为了写 kernel 先懂，但懂了能让你对整个系统有全局观。**
> 阅读建议：有空翻，一次看一两节即可，每节都能独立读。

---

## 0. 一句话总览

```text
训练 = 用大量数据，反复微调模型里的一堆数字(权重)，让它预测得越来越准。
推理 = 训练完，权重固定，拿来算结果(你 Week4 做的就是这个)。
```

类比你熟悉的：

```text
权重(weights)   ≈ 一大堆 float 常量(训练完就不变了)
训练            ≈ 一个超大的"自动调参"循环，目标是让误差最小
推理            ≈ 拿调好的常量做前向计算(GEMM/softmax/...)
```

---

## 1. 模型里到底存了什么：参数(weights)

一个神经网络 = **一堆矩阵和向量**（就是 Week4 里的 Wq/Wk/Wv/Wo、FFN 的两个 Linear 等）。

```text
这些矩阵里的每个 float 数字 = 一个"参数"(parameter / weight)
GPT-3 有 1750 亿个参数 → 就是 1750 亿个 float
DeepSeek-V3 有 6710 亿参数(MoE，激活约 370 亿)
```

关键认知：

```text
模型结构(有几层、每层多大) = 人设计的，固定
参数的具体数值            = 训练"学"出来的
所以"训练模型" = "找到一组好的参数数值"
推理时这些数值就是只读常量，你做的 GEMM 里的权重矩阵就是它们。
```

类比：

```text
结构   ≈ 你写好的 kernel 代码框架
参数   ≈ 喂给 kernel 的输入常量数组
训练   ≈ 花几百万美元的 GPU 算力，去搜出这个常量数组的最佳值
```

---

## 2. 训练要解决的问题：让"预测"接近"真实"

以语言模型为例，任务是**预测下一个 token**：

```text
输入： "中国的首都是"
模型输出： 一个 logits 向量 [vocab_size]，每个词表 token 一个分数
理想情况： "北京" 这个 token 分数最高
```

训练数据长这样（自监督，不用人工标注）：

```text
拿海量文本，每个位置天然就有"标准答案"——就是下一个真实 token
文本： "中国 的 首都 是 北京"
样本1: 输入"中国"          → 目标"的"
样本2: 输入"中国 的"        → 目标"首都"
样本3: 输入"中国 的 首都"    → 目标"是"
样本4: 输入"中国 的 首都 是" → 目标"北京"
→ 一句话能拆出很多训练样本，这就是为什么互联网文本能训出大模型。
```

---

## 3. 怎么衡量"预测得好不好"：损失函数(loss)

需要一个数字来量化"错得多离谱"，这个数字叫 **loss（损失）**。

```text
模型预测下一个 token 的概率分布(softmax 后)：
  北京 0.6,  上海 0.2,  南京 0.1,  其他 0.1
标准答案是"北京"。
loss 越小 = 给正确答案的概率越高 = 预测越好。
语言模型常用 cross-entropy(交叉熵) loss：
  loss = -log(给正确 token 的概率) = -log(0.6) ≈ 0.51
如果模型给"北京"只有 0.01：loss = -log(0.01) ≈ 4.6 (惩罚大)
```

一句话：

```text
loss = 一个标量，表示"这次预测有多差"。训练目标 = 让所有样本的平均 loss 最小。
```

类比：loss 就像你性能优化里的"耗时"——一个要往下压的单一数字。

---

## 4. 核心机制：梯度下降(gradient descent)

有了 loss，怎么调参数让 loss 变小？答案是**梯度下降**。

### 4.1 直觉：下山

```text
把 loss 想成一片山地的"海拔"，参数是你的"坐标"。
你想走到海拔最低点(loss 最小)。
梯度(gradient) = 当前位置最陡的上坡方向。
→ 往梯度的反方向走一小步，海拔就降一点。
反复走 → 走到谷底 → loss 最小 → 参数训练好了。
```

### 4.2 数学形式（很简单）

```text
对每个参数 w：
  gradient = ∂loss/∂w   (loss 对这个参数的偏导：w 变一点，loss 变多少)
  w_new = w - lr × gradient
lr(learning rate，学习率) = 步子大小，太大会震荡，太小学得慢(常见 1e-4 量级)
```

类比：

```text
gradient ≈ "把这个参数调大一点，loss 会升还是降，升降多快"
更新     ≈ 朝让 loss 下降的方向，把每个参数挪一小步
1750 亿个参数就同时各挪一小步——所以需要巨量矩阵运算(你的 GEMM 舞台)。
```

---

## 5. 梯度怎么算出来：反向传播(backpropagation)

模型有很多层，怎么算出"每个参数对最终 loss 的影响"？靠**反向传播**。

```text
前向(forward)：  输入 → 层1 → 层2 → ... → 输出 → loss   (你 Week4 做的方向)
反向(backward)： loss → ... → 层2 → 层1 → 每个参数的 gradient (反着来)
原理：链式求导法则(chain rule)，从 loss 一层层往回传"误差责任"。
```

关键认知（对做 CUDA 的人很重要）：

```text
1. backward 的计算量 ≈ forward 的 2 倍，也是一堆 GEMM(只是矩阵转置着乘)
2. backward 需要用到 forward 的中间结果 → 要么存下来(占显存)，要么重算
   → 这就是"激活值显存"和"梯度检查点(gradient checkpointing)"的由来
3. 训练显存 = 参数 + 梯度 + 优化器状态 + 激活值，比推理大得多
   → 这就是为什么训练要那么多卡，而推理相对省
```

类比：forward 是你算 `C=A×B`；backward 就是已知"C 的误差"，反推"A 的误差"和"B 的误差"，又是两个 GEMM。

---

## 6. 优化器(optimizer)：不止"减去梯度"

第 4 节的 `w = w - lr×grad` 是最朴素的 SGD。实际大模型用更聪明的**优化器**，主流是 **Adam / AdamW**。

```text
Adam 额外为每个参数维护两个"状态"：
  m: 梯度的滑动平均(动量，方向更稳)
  v: 梯度平方的滑动平均(自适应步长，陡的地方小步、平的地方大步)
所以每个参数训练时要存：参数本身 w + 梯度 g + Adam 的 m + v
→ 4 份数据！这就是"训练显存 ≈ 参数量×若干倍"的主因之一。
```

类比：比起"闷头往下走"，Adam 像带惯性 + 会根据地形自动调步幅的下山法。

---

## 7. 训练循环长什么样（伪代码）

把前面串起来，一个训练步骤(step)：

```python
for batch in data:                 # 每次取一批样本
    logits = model.forward(batch)  # 前向：一堆 GEMM/attention/softmax
    loss   = cross_entropy(logits, batch.targets)
    grads  = loss.backward()       # 反向：算每个参数的梯度
    optimizer.step(grads)          # 用梯度更新所有参数(Adam)
    optimizer.zero_grad()          # 清空梯度，进入下一步
# 重复几十万~几百万个 step，喂完上万亿 token
```

```text
一次 forward+backward+update = 一个 step
大模型训练 = 几十万到几百万个 step，跑几周~几个月，几千张 GPU
你在 Week4 优化的，是上面 forward 里 attention 那一段。
```

---

## 8. 训练 vs 推理：到底差在哪

| 维度 | 训练(training) | 推理(inference，你做的) |
|---|---|---|
| 参数 | 一直在变 | 固定只读 |
| 方向 | forward + backward | 只有 forward |
| 显存 | 参数+梯度+优化器状态+激活 | 参数 + KV cache |
| 精度 | 常用 BF16 混合精度 + FP32 主权重 | FP16/BF16/FP8/INT8 均可 |
| 目标 | loss 最小 | 延迟低、吞吐高 |
| batch | 尽量大(利用率高) | 受延迟约束，常小 batch |
| GPU 数 | 几千张 | 几张~几十张 |

一句话：

```text
训练是"造模型"(贵、慢、一次性)；推理是"用模型"(反复跑、要快)。
DeepSeek 这类岗位两侧都招人，但你目前主攻推理侧 CUDA 优化。
```

---

## 9. 几个你迟早会听到的名词（速查）

| 名词 | 一句话 |
|---|---|
| epoch | 把整个训练集完整过一遍 |
| batch size | 一次 forward 喂多少个样本 |
| learning rate | 梯度下降的步子大小 |
| overfitting 过拟合 | 模型背下了训练数据，但新数据上差(死记硬背) |
| regularization 正则化 | 防过拟合的手段(如 weight decay) |
| dropout | 训练时随机丢弃一部分神经元，防过拟合(推理时关闭) |
| pretrain 预训练 | 用海量通用文本训基础模型(第 2 节那种) |
| fine-tune 微调 | 在预训练模型上，用少量特定数据继续训 |
| SFT | 有监督微调，用"指令-回答"对教模型听话 |
| RLHF / DPO | 用人类偏好进一步对齐模型输出 |
| checkpoint | 训练中途保存的一份参数快照 |
| gradient checkpointing | 用重算换显存的技巧(第 5 节) |
| mixed precision 混合精度 | BF16 算得快，FP32 存主权重保精度 |

---

## 10. 训练里也全是你的 CUDA 老本行

给你吃颗定心丸——训练不是另一个世界，底层还是你会的东西：

```text
forward 的每一层        → GEMM、attention、softmax、layernorm(你都写过)
backward               → 转置着乘的 GEMM(还是矩阵乘)
optimizer.step         → 逐元素更新(element-wise kernel，最简单那种)
多卡训练               → 通信(all-reduce/all-gather)，Week8 多卡会学
省显存/提速的一切招数   → tiling、混合精度、cp.async、算子融合(你正在学)
```

所以：

```text
你不需要"先成为 ML 专家再做 CUDA"。
反过来——你懂 CUDA + 硬件，正是训练/推理框架(Megatron/vLLM/SGLang)最缺的人。
ML 概念够用即可，深度留给有兴趣时慢慢补。
```

---

## 11. 想再深入时的最小阅读清单

不急，按兴趣来：

```text
1. 3Blue1Brown 的神经网络/梯度下降/backprop 视频(直觉最好，中文字幕)
2. 《Attention Is All You Need》(2017) —— Transformer 原始论文，你已懂 attention 再看很轻松
3. Karpathy 的 "Let's build GPT" / nanoGPT —— 从零手写训练一个小 GPT(代码向，最适合你)
4. Adam 论文的直觉部分 + weight decay/AdamW
5. 混合精度训练(NVIDIA AMP 文档) —— 和你的精度知识直接接上
```

---

## 12. 一页纸总结

```text
参数     = 模型里一堆 float(训练学出来，推理只读)
loss     = 衡量预测有多差的一个标量
梯度下降 = 朝让 loss 下降的方向，把每个参数挪一小步
反向传播 = 用链式法则从 loss 反推每个参数的梯度(又是一堆 GEMM)
优化器   = 更聪明的挪步方式(Adam，带动量和自适应步长)
训练循环 = forward→loss→backward→update，重复几十万步
训练 vs 推理 = 造模型(贵/慢/双向) vs 用模型(快/单向，你主攻这个)
底层     = 依旧是 GEMM/softmax/norm/通信 —— 全是你的 CUDA 主场
```
