# ML 训练基础与 PyTorch 自学教材 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `docs/ML基础_训练侧入门.md` 扩写为一份从手算训练原理出发、逐步学习 PyTorch、训练迷你语言模型并连接 CUDA/AI Infra 的自包含教材。

**Architecture:** 保留单一主文档，按四卷递进，避免跨文档跳转；完整示例以内嵌命名代码围栏提供，执行时提取到 `/tmp/ml_training_guide/` 后验证。普通数学例子用系统 `python3`，PyTorch 示例固定用 `/home/qichengjie/workspace/vllm_demo/.venv/bin/python`；后者已验证为 PyTorch 2.11.0+cu130、A100 80GB PCIe、SM80、BF16 可用。

**Tech Stack:** Markdown、Python 3、PyTorch 2.11、CUDA 13.0、A100 SM80、torch.profiler、Nsight Systems/Compute、Git。

---

## 文件与验证边界

**修改：**

- `docs/ML基础_训练侧入门.md`：四卷教材正文、练习、答案和完整内嵌示例。

**引用但不修改：**

- `docs/Week4_Attention与FlashAttention完整学习资料.md`
- `docs/Week3_TensorCore学习文档.md`
- `operator_practice/layernorm/layernorm.cu`
- `operator_practice/softmax/softmax.cu`
- `week05_gemm_advanced/gemm_optimization_ladder.md`

**运行时临时文件：**

- `/tmp/ml_training_guide/scalar_training.py`
- `/tmp/ml_training_guide/autograd_demo.py`
- `/tmp/ml_training_guide/mlp_classifier.py`
- `/tmp/ml_training_guide/mini_lm.py`
- `/tmp/ml_training_guide/mini_gpt.py`
- `/tmp/ml_training_guide/profile_training.py`

这些临时文件不进入 Git。学习者最终按教材自行创建 `ml_training_basics/` 练习目录，教材实现阶段不提前替其创建作业答案文件。

---

### Task 1：重构文档骨架与使用说明

**Files:**

- Modify: `docs/ML基础_训练侧入门.md`

- [ ] **Step 1：保留并升级定位说明**

明确读者会 CUDA 但无 ML 基础；目标同时覆盖原理、PyTorch、迷你语言模型和训练侧 AI Infra。说明教材很长，应按卷学习，不要求一次读完。

- [ ] **Step 2：加入代码标记和学习纪律**

定义 `【演示】`、`【必须手写】`、`【提示 1/2/3】`、`【参考实现】`、`【挑战】`。规定先手算、再写代码、再看答案；每卷有完成标准。

- [ ] **Step 3：建立四卷目录**

标题顺序必须是：

```text
第一卷 训练原理
第二卷 PyTorch
第三卷 迷你语言模型
第四卷 CUDA / AI Infra
附录 公式、术语、自测答案、环境命令
```

- [ ] **Step 4：写旧知识映射表**

映射 GEMM→Linear/backward、reduction→loss/norm、Attention→语言模型、Tensor Core→混合精度、ncu/nsys→训练 profiling。

- [ ] **Step 5：验证骨架**

Run:

```bash
rg -n '^#{1,4} ' docs/ML基础_训练侧入门.md
```

Expected: 四卷与附录顺序完整，无标题层级倒跳。

- [ ] **Step 6：提交骨架**

```bash
git add docs/ML基础_训练侧入门.md
git commit -m "docs: restructure ML training foundations guide"
```

### Task 2：第一卷上半——模型、loss、导数与一次参数更新

**Files:**

- Modify: `docs/ML基础_训练侧入门.md`

- [ ] **Step 1：写“模型是带参数函数”**

使用 `y_hat=wx+b` 区分输入、预测、目标、参数、模型结构、超参数。明确 prediction 与 truth 不是同一个变量。

- [ ] **Step 2：完整手算固定例子**

固定 `x=2,y=5,w=1,b=0,L=(y_hat-y)^2`，算出：

```text
y_hat=2
L=9
dL/dy_hat=-6
dy_hat/dw=2
dy_hat/db=1
dL/dw=-12
dL/db=-6
```

用 `lr=0.1` 更新到 `w=2.2,b=0.6`，再次计算 `y_hat=5,L=0`。说明此例一步恰好到达只因数据和学习率特意选择，不代表一般训练一步收敛。

- [ ] **Step 3：从斜率解释导数/偏导/gradient**

每个词首次出现给中文、符号和数值解释。gradient 是所有参数偏导组成的向量，不是 loss 本身。

- [ ] **Step 4：写链式法则计算图**

按节点 `mul→add→subtract→square` 列出局部导数和上游梯度，展示反向乘法如何得到 `dw/db`。

- [ ] **Step 5：加入有限差分检查**

给出中心差分，并解释 epsilon 太大是截断误差、太小是浮点消减误差。

- [ ] **Step 6：写完整普通 Python 演示**

命名围栏 `python scalar_training.py`，程序必须：

- 定义 `loss(w,b,x,y)`；
- 定义解析 gradient；
- 定义中心差分 gradient；
- 对比误差 `<1e-5`；
- 训练 20 step 的多样本线性回归；
- 检查最终 loss 小于初始 loss 且打印 `PASS`。

- [ ] **Step 7：提取运行**

```bash
mkdir -p /tmp/ml_training_guide
awk '/^```python scalar_training.py$/{p=1;next} /^```$/{if(p){exit}} p' \
  docs/ML基础_训练侧入门.md > /tmp/ml_training_guide/scalar_training.py
python3 /tmp/ml_training_guide/scalar_training.py
```

Expected: 解析梯度与有限差分对齐，loss 下降，最后打印 `PASS`。

- [ ] **Step 8：提交第一卷基础**

```bash
git add docs/ML基础_训练侧入门.md
git commit -m "docs: teach gradients from a scalar model"
```

### Task 3：第一卷下半——矩阵 Linear、MLP、backprop 与训练术语

**Files:**

- Modify: `docs/ML基础_训练侧入门.md`

- [ ] **Step 1：从 scalar 扩展到矩阵 Linear**

固定 row-major 约定：`X[B,Din] W[Din,Dout] + b[Dout] → Y[B,Dout]`。用 `B=2,Din=3,Dout=2` 手算一个输出和 bias broadcast。

- [ ] **Step 2：推导 Linear backward shape**

写出：

```text
dX = dY W^T       [B,Dout]×[Dout,Din]→[B,Din]
dW = X^T dY       [Din,B]×[B,Dout]→[Din,Dout]
db = reduce_B(dY) [Dout]
```

解释为什么 backward 主要仍是 GEMM，以及 db 是 reduction。

- [ ] **Step 3：解释 ReLU 和两层 MLP**

用 `Linear→ReLU→Linear→MSE` 画计算图；ReLU backward 是上游梯度乘 `x>0` mask。

- [ ] **Step 4：解释 activation 保存与重算**

列出 backward 需要的 `X/W/pre-activation`，连接 gradient checkpointing 的动机，不提前深入框架实现。

- [ ] **Step 5：统一训练术语**

用一个 `1000 samples, batch=100, accumulation=4` 的例子区分 sample、batch、micro-batch、optimizer step、epoch、gradient accumulation、shuffle、validation。

- [ ] **Step 6：加入第一卷自测和答案**

至少包含 5 道手算、5 道 shape、3 道错误诊断；答案必须给过程。

- [ ] **Step 7：提交第一卷完成版**

```bash
git add docs/ML基础_训练侧入门.md
git commit -m "docs: derive matrix backprop and training terms"
```

### Task 4：第二卷上半——PyTorch Tensor 与 autograd

**Files:**

- Modify: `docs/ML基础_训练侧入门.md`

- [ ] **Step 1：写验证环境命令**

固定解释器：

```bash
PY=/home/qichengjie/workspace/vllm_demo/.venv/bin/python
$PY -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

写明当前验证环境 PyTorch 2.11.0+cu130、A100 80GB PCIe，代码仍提供 CPU fallback。

- [ ] **Step 2：从 CUDA 数组解释 Tensor**

覆盖 shape、dtype、device、stride、storage、view、reshape、transpose、contiguous、broadcast、matmul，并给可运行的小例子打印结果。

- [ ] **Step 3：用同一 scalar 例子解释 autograd**

完整程序命名 `python autograd_demo.py`，必须比较：手算 gradient、有限差分、`loss.backward()` 的 `w.grad/b.grad`。

- [ ] **Step 4：演示 gradient accumulation**

连续两次 backward 不清梯度，验证 `.grad` 加倍；再比较 `zero_grad(set_to_none=True)`、手工置零与 `grad=None` 的区别。

- [ ] **Step 5：解释计算图边界**

覆盖 leaf tensor、`grad_fn`、`no_grad`、`detach`、in-place 风险、为什么 loss 通常 reduction 成 scalar。

- [ ] **Step 6：运行 autograd 示例**

```bash
awk '/^```python autograd_demo.py$/{p=1;next} /^```$/{if(p){exit}} p' \
  docs/ML基础_训练侧入门.md > /tmp/ml_training_guide/autograd_demo.py
/home/qichengjie/workspace/vllm_demo/.venv/bin/python \
  /tmp/ml_training_guide/autograd_demo.py
```

Expected: 三种 gradient 在容差内一致，累积行为正确，CPU 与 CUDA（可用时）均打印 `PASS`。

- [ ] **Step 7：提交 Tensor/autograd**

```bash
git add docs/ML基础_训练侧入门.md
git commit -m "docs: bridge manual gradients to PyTorch autograd"
```

### Task 5：第二卷下半——Module、loss、optimizer 与 DataLoader

**Files:**

- Modify: `docs/ML基础_训练侧入门.md`

- [ ] **Step 1：递进实现 Linear**

依次展示 Tensor 公式、自定义 `nn.Module`、`nn.Linear`；说明 Parameter 注册、`model.parameters()`、`state_dict` 和 `model(x)`。

- [ ] **Step 2：解释 loss API**

比较 MSE、BCE-with-logits、cross entropy。用小词表数字说明 `CrossEntropyLoss` 接 logits，重复 softmax 会错误或数值更差。

- [ ] **Step 3：解释 optimizer 顺序**

逐行说明 `zero_grad→forward→loss→backward→step`；对照手动 SGD、SGD momentum、Adam、AdamW，并准确解释 decoupled weight decay。

- [ ] **Step 4：解释 Dataset/DataLoader**

覆盖 shuffle、batch、drop_last、sampler、num_workers、pin_memory、non_blocking，以及数据管线可能喂不饱 GPU。

- [ ] **Step 5：解释 gradient accumulation**

给出对 loss 除以 accumulation steps 的正确循环，说明 optimizer step/zero_grad 的位置。

- [ ] **Step 6：加入 API 错误诊断表**

至少包含忘记 zero_grad、提前 detach、标签 dtype 错、CrossEntropy 输入 shape 错、`train/eval` 误解、把 model 移到 CUDA 但数据留在 CPU。

- [ ] **Step 7：提交 PyTorch 训练组件**

```bash
git add docs/ML基础_训练侧入门.md
git commit -m "docs: teach PyTorch modules optimizers and data"
```

### Task 6：第二卷项目——二维分类器

**Files:**

- Modify: `docs/ML基础_训练侧入门.md`

- [ ] **Step 1：定义可验证数据集**

使用固定随机种子生成二维两类数据；train/validation 分离；模型为 `Linear(2,16)→ReLU→Linear(16,2)`。

- [ ] **Step 2：先写失败标准**

训练前检查 validation accuracy 不应被硬编码为 100%；最终验收固定为 loss 下降、validation accuracy `>0.90`、参数发生变化。

- [ ] **Step 3：提供必须手写骨架与提示**

学习者补 Dataset/DataLoader、forward、training step、validation；参考实现放在验收之后。

- [ ] **Step 4：写完整参考程序**

命名 `python mlp_classifier.py`，支持 `--device cpu|cuda`，保存 checkpoint 到临时目录，重新加载后验证 logits 一致。

- [ ] **Step 5：CPU 与 A100 运行**

```bash
awk '/^```python mlp_classifier.py$/{p=1;next} /^```$/{if(p){exit}} p' \
  docs/ML基础_训练侧入门.md > /tmp/ml_training_guide/mlp_classifier.py
PY=/home/qichengjie/workspace/vllm_demo/.venv/bin/python
$PY /tmp/ml_training_guide/mlp_classifier.py --device cpu
$PY /tmp/ml_training_guide/mlp_classifier.py --device cuda
```

Expected: 两种设备均 loss 下降、accuracy >0.90、checkpoint reload 对齐、打印 `PASS`。

- [ ] **Step 6：提交分类器项目**

```bash
git add docs/ML基础_训练侧入门.md
git commit -m "docs: add guided PyTorch classifier project"
```

### Task 7：第三卷上半——语言模型 loss 与文本数据

**Files:**

- Modify: `docs/ML基础_训练侧入门.md`

- [ ] **Step 1：手算小词表 cross entropy**

固定 4-token logits，算 stable softmax、正确 token probability、NLL、batch/sequence mean。解释 perplexity=`exp(mean loss)`。

- [ ] **Step 2：解释语言模型 shape**

统一：logits `[B,T,V]`，targets `[B,T]`，调用 cross entropy 时 reshape 到 `[B*T,V]` 与 `[B*T]`。

- [ ] **Step 3：构造极简字符数据集**

解释 vocabulary、encode/decode、context length、input/target 右移一位、teacher forcing、train/validation split。

- [ ] **Step 4：写最小 bigram/embedding LM**

命名 `python mini_lm.py`，先训练 `Embedding(V,V)` 或等价模型；必须能过拟合短文本片段，并生成 token。

- [ ] **Step 5：运行最小 LM**

```bash
awk '/^```python mini_lm.py$/{p=1;next} /^```$/{if(p){exit}} p' \
  docs/ML基础_训练侧入门.md > /tmp/ml_training_guide/mini_lm.py
/home/qichengjie/workspace/vllm_demo/.venv/bin/python \
  /tmp/ml_training_guide/mini_lm.py --device cuda
```

Expected: 最终 loss 小于初始 loss、生成长度正确、打印 `PASS`。

- [ ] **Step 6：提交语言模型基础**

```bash
git add docs/ML基础_训练侧入门.md
git commit -m "docs: teach language modeling loss and data"
```

### Task 8：第三卷下半——迷你 GPT 完整闭环

**Files:**

- Modify: `docs/ML基础_训练侧入门.md`

- [ ] **Step 1：按组件递进说明**

Embedding+position、causal self-attention、multi-head、MLP、residual、LayerNorm、Transformer block、LM head。每个组件给 `[B,T,C]` shape。

- [ ] **Step 2：连接 Week 4 Attention**

链接已有 Attention 教材，只复盘训练需要的 causal mask、dropout 和输出 shape；避免重复 2000 行内容。

- [ ] **Step 3：写可运行小模型**

命名 `python mini_gpt.py`；固定小配置，例如 `context=32, embed=64, heads=4, layers=2`；支持 CPU/CUDA、checkpoint、resume、generation。

- [ ] **Step 4：加入过拟合单 batch 测试**

训练固定 batch 直到 loss 显著下降，用于验证 forward/backward/optimizer 接线；再运行短文本训练。

- [ ] **Step 5：验证 checkpoint 连续性**

保存 model、optimizer、step、随机状态或明确可复现边界；加载后验证 step 和参数恢复，并能继续训练。

- [ ] **Step 6：运行迷你 GPT**

```bash
awk '/^```python mini_gpt.py$/{p=1;next} /^```$/{if(p){exit}} p' \
  docs/ML基础_训练侧入门.md > /tmp/ml_training_guide/mini_gpt.py
PY=/home/qichengjie/workspace/vllm_demo/.venv/bin/python
$PY /tmp/ml_training_guide/mini_gpt.py --device cpu --steps 20
$PY /tmp/ml_training_guide/mini_gpt.py --device cuda --steps 100
```

Expected: CPU smoke test 退出码 0；A100 loss 下降、checkpoint/reload 通过、生成文本、打印 `PASS`。

- [ ] **Step 7：提交迷你 GPT**

```bash
git add docs/ML基础_训练侧入门.md
git commit -m "docs: add mini GPT training project"
```

### Task 9：第四卷上半——训练显存与混合精度

**Files:**

- Modify: `docs/ML基础_训练侧入门.md`

- [ ] **Step 1：深化 backward 算子映射**

连接 Linear、ReLU、LayerNorm、softmax-cross-entropy、Attention backward 到 GEMM/reduction/elementwise；说明 forward 的 kernel 优化不能直接代表完整 training step。

- [ ] **Step 2：写训练显存字节账**

分别计算参数、梯度、FP32 master weights、Adam m/v、activation、workspace、communication buffer。至少给一个 1B 参数 BF16+AdamW 示例，并声明具体框架存储策略会改变常数。

- [ ] **Step 3：解释省显存技术**

gradient accumulation、activation checkpointing、ZeRO/FSDP、CPU/NVMe offload 分别说明省哪项、增加何种计算/通信/延迟。

- [ ] **Step 4：解释混合精度**

FP32 baseline→FP16 underflow/overflow→loss scaling→BF16→autocast→GradScaler→master weights。A100 路径优先 BF16，并说明 BF16 不代表无舍入误差。

- [ ] **Step 5：写 PyTorch AMP 演示**

在 A100 上比较 FP32/BF16 autocast 的 dtype、loss 与显存；FP16 示例展示 GradScaler API，但不承诺每个小模型都触发 overflow。

- [ ] **Step 6：提交显存/精度卷**

```bash
git add docs/ML基础_训练侧入门.md
git commit -m "docs: explain training memory and mixed precision"
```

### Task 10：第四卷下半——profiling、多卡与面试映射

**Files:**

- Modify: `docs/ML基础_训练侧入门.md`

- [ ] **Step 1：写训练性能指标**

解释 step time、samples/s、tokens/s、model FLOPs、MFU、峰值显存、通信占比；所有指标带 shape/batch/dtype/hardware。

- [ ] **Step 2：写三层工具分工**

PyTorch profiler 看 operator；nsys 看 CPU/GPU/通信时间线；ncu 深挖单 kernel。强调 profiler 插桩时间不作为真实性能。

- [ ] **Step 3：提供 profile 示例**

命名 `python profile_training.py`，使用 `torch.profiler` 记录 CPU/CUDA activity、shape、memory，打印 top operators；以分类器或迷你 GPT 的数个 step 为负载。

- [ ] **Step 4：解释多卡方式**

DDP、Tensor Parallel、Pipeline Parallel、Sequence/Context Parallel、ZeRO/FSDP 均使用统一表格：切什么、每卡存什么、collective、通信时机、收益、代价。

- [ ] **Step 5：加入面试题与口述**

至少 15 道概念题、8 道 shape/显存题、5 道 profiler/系统取舍题，附答案和 3 分钟训练流程口述。

- [ ] **Step 6：运行 profiler 示例**

```bash
awk '/^```python profile_training.py$/{p=1;next} /^```$/{if(p){exit}} p' \
  docs/ML基础_训练侧入门.md > /tmp/ml_training_guide/profile_training.py
/home/qichengjie/workspace/vllm_demo/.venv/bin/python \
  /tmp/ml_training_guide/profile_training.py
```

Expected: 打印包含 CPU/CUDA operator 的 profiler table，退出码 0。

- [ ] **Step 7：提交系统卷**

```bash
git add docs/ML基础_训练侧入门.md
git commit -m "docs: add training profiling and distributed systems"
```

### Task 11：全书代码与内容交付验证

**Files:**

- Modify: `docs/ML基础_训练侧入门.md`

- [ ] **Step 1：对照设计完成标准**

逐项对照 `docs/superpowers/specs/2026-07-03-ml-training-foundations-pytorch-design.md` 第 12 节，确认每项有对应章节和验证证据。

- [ ] **Step 2：扫描含糊占位**

```bash
rg -n 'T''BD|TO''DO|FIX''ME|待''补|以后''填写|未''完成' \
  docs/ML基础_训练侧入门.md
```

Expected: 无输出。学习者练习空位必须用明确的 `【必须手写】` 需求和验收描述。

- [ ] **Step 3：提取并语法检查全部命名程序**

重新提取 Task 2、4、6、7、8、10 的六个程序，运行：

```bash
python3 -m py_compile /tmp/ml_training_guide/scalar_training.py
/home/qichengjie/workspace/vllm_demo/.venv/bin/python -m py_compile \
  /tmp/ml_training_guide/autograd_demo.py \
  /tmp/ml_training_guide/mlp_classifier.py \
  /tmp/ml_training_guide/mini_lm.py \
  /tmp/ml_training_guide/mini_gpt.py \
  /tmp/ml_training_guide/profile_training.py
```

Expected: 退出码 0。

- [ ] **Step 4：运行完整验证矩阵**

```bash
python3 /tmp/ml_training_guide/scalar_training.py
PY=/home/qichengjie/workspace/vllm_demo/.venv/bin/python
$PY /tmp/ml_training_guide/autograd_demo.py
$PY /tmp/ml_training_guide/mlp_classifier.py --device cpu
$PY /tmp/ml_training_guide/mlp_classifier.py --device cuda
$PY /tmp/ml_training_guide/mini_lm.py --device cuda
$PY /tmp/ml_training_guide/mini_gpt.py --device cpu --steps 20
$PY /tmp/ml_training_guide/mini_gpt.py --device cuda --steps 100
$PY /tmp/ml_training_guide/profile_training.py
```

Expected: 所有程序打印 `PASS`，无 traceback；训练项目 loss 下降；分类 accuracy、checkpoint 和 generation 验收满足前述阈值。

- [ ] **Step 5：检查结构、围栏和本地链接**

```bash
test $(( $(rg -c '^```' docs/ML基础_训练侧入门.md) % 2 )) -eq 0
rg -n '^#{1,4} ' docs/ML基础_训练侧入门.md
git diff --check -- docs/ML基础_训练侧入门.md
```

Expected: 围栏成对、标题顺序正确、无 whitespace error。

- [ ] **Step 6：提交最终修订**

```bash
git add docs/ML基础_训练侧入门.md
git commit -m "docs: finish ML training and PyTorch guide"
```

---

## 执行检查点

1. Task 3 后：零 ML 基础读者能否手算 scalar 和 matrix gradient；
2. Task 6 后：PyTorch API 是否都能映射回第一卷概念，分类器是否真的收敛；
3. Task 8 后：迷你 GPT 是否通过小 batch 过拟合和 checkpoint 恢复；
4. Task 9 后：训练显存与混合精度是否给出准确边界而非固定倍数口号；
5. Task 11 后：所有成功声明均有实际命令输出，不能因文档代码“看起来合理”就视为通过。
