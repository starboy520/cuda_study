# ML 基础 · 第二卷精讲（零基础版）

> 配套原文：[ML基础_训练侧入门.md](ML基础_训练侧入门.md) 第二卷
> 前置：读完 [第一卷精讲](ML基础_第一卷_精讲.md)（懂了 forward/loss/backward/update 四步）。
> 目标：把 PyTorch 还原成"帮你自动做第一卷那四步"的工具。每个 API 都回答"它对应第一卷的哪一步、为什么需要它"。

---

## 0. 一句话看懂整个第二卷

第一卷你**手写**了训练四步：

```
forward（手写 ŷ=wx+b）→ loss（手写 (ŷ-y)²）→ backward（手推链式法则）→ update（手写 w-=lr*g）
```

**PyTorch 的全部意义：把这四步自动化。** 尤其是最烦的 **backward——你不用再手推链式法则，框架自动算梯度**。

| 第一卷你手做的 | PyTorch 帮你做的 | 用什么 |
|---------------|-----------------|--------|
| 用数组存数据、算 GEMM | Tensor（带自动求导的数组） | `torch.Tensor` |
| 手推链式法则算梯度 | **自动反向传播** | `autograd` / `loss.backward()` |
| 手动管理 w、b 一堆参数 | 自动组织参数 | `nn.Module` |
| 手写 `(ŷ-y)²` | 现成损失函数 | `nn.MSELoss` 等 |
| 手写 `w -= lr*g` | 现成优化器 | `torch.optim.SGD/Adam` |
| 手动分批喂数据 | 自动 shuffle/batch | `DataLoader` |

**读第二卷时,每遇到一个新 API,就问自己:"它自动化了第一卷的哪一步?"** 这样就不跳了。

> **一句话记忆**：PyTorch = 把第一卷的"forward→loss→backward→update"自动化的工具箱,核心是 autograd 帮你自动算梯度(不用再手推链式法则)。

---

## 1. Tensor：带"自动求导"的数组

### 它是什么

`Tensor` = **多维数组 + 一些元数据 + 自动求导能力**。你就把它理解成"能在 GPU 上跑、还能自动算梯度的 numpy 数组"。

四个元数据（都是你 CUDA 里的老朋友）：

| 元数据 | 含义 | CUDA 对应 |
|--------|------|-----------|
| `shape` | 每维大小,如 `(3,4)` | 你 kernel 里的 N、K |
| `dtype` | 元素类型,如 `float32` | `float`/`half` |
| `device` | 在哪,`cpu` 或 `cuda:0` | 主机 vs 设备内存 |
| `stride` | 某维下标 +1,内存偏移多少个元素 | **leading dimension / pitch** |

```python
import torch
x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print(x.shape, x.dtype, x.device, x.stride())  # (3,4) float32 cpu (4,1)
```

### stride 就是你手算的行主序偏移

`x[i,j]` 的线性偏移 = `i*stride(0) + j*stride(1)`。对 `(3,4)` 的行主序数组,`stride=(4,1)` → `x[i,j]` 在 `i*4+j`。**这和你 CUDA 里 `W[row*K+col]` 完全一样。** PyTorch 只是把这个偏移规则存进了元数据。

> **一句话记忆**：Tensor = GPU 数组 + 元数据(shape/dtype/device/stride) + 自动求导。stride 就是你熟的行主序偏移(leading dimension)。

### 1.1 view / reshape / transpose / contiguous（内存布局）

这几个是"改元数据 vs 真复制数据"的区别——**和你 CUDA 里"逻辑索引 vs 物理地址"是同一回事**：

| 操作 | 干什么 | 复制数据吗 |
|------|--------|-----------|
| `view` | 只改 shape 的元数据 | **不复制**(要求 stride 能表示新形状) |
| `reshape` | 优先 view,不行才复制 | **可能复制**(别依赖它共享内存) |
| `transpose(0,1)` | 交换两维的 shape 和 stride | **不复制**(结果常"不连续") |
| `contiguous()` | 按逻辑顺序真的复制一份连续内存 | **复制**(有成本) |

```python
x = torch.arange(12).reshape(3, 4)
y = x.transpose(0, 1)          # 只换了 stride,没复制 → 内存不连续
assert not y.is_contiguous()
z = y.contiguous().view(12)    # 先复制成连续,才能 view 成一维
```

**为什么你要在意**：你以后写**自定义 CUDA 算子接进 torch** 时,如果 kernel 假设"内存连续",但传进来的是 transpose 后的非连续 Tensor,就会读错。所以要么先 `contiguous()`,要么在 kernel 里正确处理 stride。这正是你 CUDA 背景的用武之地。

> **一句话记忆**：view/reshape 改元数据(可能不复制)、transpose 换 stride 导致"不连续"、contiguous 真复制成连续。对应你 CUDA 的"逻辑索引 vs 物理地址"。自定义算子要检查 stride。

### 1.2 broadcasting（广播）

两个不同 shape 的 Tensor 做逐元素运算时,PyTorch 自动"虚拟扩展"较小的那个,不真复制:

```python
x = torch.randn(32, 128)   # 一批 32 个样本,每个 128 维
b = torch.randn(128)       # 一个 128 维的偏置
y = x + b                  # b 被当成 (32,128),加到每一行
```

这就是第一卷 Linear 层 `Y=XW+b` 里"b 加到每一行"的机制。**实现上等价于"某一维 stride=0 的只读视图"**——不真复制,靠元数据骗过去。

**危险点**：无意中生成巨大中间量。比如 `(N,1,D) - (1,N,D)` 会广播成 `(N,N,D)`——N 大时直接爆显存。

> **一句话记忆**：广播 = 自动把小 Tensor"虚拟扩展"匹配大 Tensor(靠 stride=0,不真复制)。就是 `Y=XW+b` 里 b 加到每行的机制。小心广播出 `(N,N,D)` 这种巨中间量。

### 1.3 matmul（矩阵乘,你最熟的）

```python
a = torch.randn(8, 16, 32)
b = torch.randn(1, 32, 64)
c = a @ b                   # (8,16,64)
```

- 二维:`a @ b` 就是 GEMM,`(M,K)@(K,N)→(M,N)`
- 三维+：**最后两维是矩阵维,前面的维当 batch**(batched GEMM)
- ⚠️ `@` 是矩阵乘,`*` 是**逐元素乘**,别搞混

> **一句话记忆**：`@` = 矩阵乘(GEMM),多维时前面维是 batch;`*` = 逐元素乘。这就是你写的 GEMM,只是 torch 一行搞定。

---

## 2. autograd：框架自动帮你做反向传播（第二卷核心）

**这是 PyTorch 最重要、也是第二卷最该懂的东西。** 第一卷你手推链式法则算梯度,痛苦吧?autograd 就是**自动帮你做这件事**。

### 怎么用（三步）

```python
w = torch.tensor(2.0, requires_grad=True)   # ① 标记"这个参数要算梯度"
y = w * w + 3 * w                           # ② 正常 forward,torch 偷偷记录了每一步
y.backward()                                # ③ 自动反向,算出梯度
print(w.grad)                               # dy/dw = 2w+3 = 7 ✓
```

对比第一卷:你要手推 `dy/dw = 2w+3` 再代入。这里 `y.backward()` **自动**算好放进 `w.grad`。**你只写 forward,backward 白送。**

### 它是怎么做到的（理解原理）

1. **`requires_grad=True`**：告诉 torch "这个 Tensor 要算梯度,请记录它参与的运算"
2. **forward 时自动建图**：你每做一步运算(`*`、`+`...),torch 在背后**记下这一步和它的局部导数**——这就是第一卷说的"计算图"。每个中间结果都有个 `grad_fn` 指向"生成它的反向操作"。
3. **`backward()` 时自动跑链式法则**：从最终结果出发,沿着记录的图**从后往前,把局部导数逐段相乘**(正是第一卷的链式法则!),算出每个 `requires_grad` 参数的梯度,累积到它的 `.grad`。

**一句话:autograd = 自动记录计算图 + 自动跑链式法则。** 你第一卷手推的那套"齿轮传动连乘",torch 全自动化了。

### 名词澄清：雅可比、上游梯度（原文这里最跳）

原文提到"向量-雅可比积""上游梯度",对新手很跳。翻译成人话:

- **上游梯度** = "loss 对当前这个量的导数",就是第一卷链式法则里**从后面传回来的那个累乘结果**。比如 `∂L/∂e = 2e` 就是传给 e 的上游梯度。
- **雅可比** = "每个输出对每个输入的导数排成的表"(多输入多输出时的导数矩阵)。
- **向量-雅可比积(VJP)** = 反向传播的实际操作:**不真的构造那张大表,而是把"上游梯度"和"本层局部导数"相乘**,直接得到传给更前面的梯度。

**说白了,VJP 就是链式法则那步"上游梯度 × 本层局部导数"的正式名字。** 你第一卷已经懂这个动作了,只是没叫它 VJP。

> **一句话记忆**：autograd = forward 时自动记计算图,`backward()` 时自动跑链式法则算梯度(放进 `.grad`)。你只写 forward,梯度白送。"上游梯度 × 本层局部导数"就是反向传播的核心动作(正式名 VJP)。

### 2.1 梯度默认累积（重要坑）

`backward()` 是把新梯度**加到** `.grad` 上,不是覆盖(为了支持第一卷说的"梯度累积")。所以**每次迭代前要清零**:

```python
optimizer.zero_grad(set_to_none=True)   # 清零梯度,每步开头做
```

`set_to_none=True` 把 `.grad` 设成 `None`(而不是全 0),省一次清零的写带宽,还能区分"没梯度"和"梯度恰好是 0"。**这和你 CUDA 里"累加前先清零缓冲区"是同一个道理。**

### 2.2 no_grad / detach（什么时候不要梯度）

- **`with torch.no_grad():`**：这段里不记录计算图。**验证/推理时用**(不训练就不用算梯度,省显存省时间)。
- **`x.detach()`**：返回一个"共享数据但切断求导历史"的新 Tensor。想拿某个中间结果但不想让梯度传过它时用。

```python
with torch.no_grad():      # 推理:不建图
    pred = model(x)
```

> **一句话记忆**：`backward` 默认累加梯度 → 每步先 `zero_grad`。推理/验证用 `no_grad` 不建图(省资源);`detach` 切断某处的求导历史。

---

## 3. 三种梯度对齐（验证 autograd 没骗你）

原文这节用一个程序,同时算三种梯度并确认它们一致:

1. **手推**(第一卷方法)：`dL/dw = 2(wx-y)x`
2. **有限差分**(第一卷梯度检查)：`(L(w+eps)-L(w-eps))/(2eps)`
3. **autograd**：`loss.backward()` 后的 `w.grad`

三者对上 → 证明 autograd 算的梯度是对的。

**这就是你 CUDA 里"三方对拍"的精神**:手算 = CPU reference,autograd = GPU kernel,有限差分 = 独立验证。目的都是"用可靠的慢方法验证自动的快方法"。

实际工程验证自定义算子用 `torch.autograd.gradcheck`(帮你自动做有限差分对拍)。

> **一句话记忆**：手推 / 有限差分 / autograd 三种梯度对齐 = 验证 autograd 没算错。和你 CUDA 的 CPU-reference 对拍同一个精神。

---

## 4. nn.Module：帮你管理一堆参数

真实模型有几百个 `w`、`b`,手动一个个管理会疯。`nn.Module` 就是**帮你自动组织参数**的容器。

```python
from torch import nn

class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # nn.Parameter 声明的,自动注册为"要训练的参数"
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.T + self.bias   # 就是第一卷的 Y=XW+b
```

关键点:
- **`nn.Parameter`** 声明的张量,自动:① 注册为参数 ② 默认 `requires_grad=True` ③ 被 `model.parameters()` 收集(交给优化器)。你不用手动追踪每个 w/b。
- **`forward`** 定义前向。调用时写 `model(x)`,**别写 `model.forward(x)`**——因为 `model(x)` 还会触发 hooks、autocast 等框架功能。
- **`state_dict()`**：参数的字典,用来存/读 checkpoint(`torch.save(model.state_dict(), path)`)。
- **`model.train()` / `model.eval()`**：切换模式(影响 Dropout/BatchNorm 的行为),**不是开关梯度**。验证时通常 `model.eval()` + `torch.no_grad()` 一起用。

> **一句话记忆**：nn.Module = 参数容器。`nn.Parameter` 声明的自动注册+求导+被优化器收集,你不用手动管一堆 w/b。调用写 `model(x)`,存权重用 `state_dict`。

---

## 5. 损失函数：现成的 loss（含一个大坑：logits）

第一卷你手写 `(ŷ-y)²`。PyTorch 给了现成的损失函数:

| 损失 | 用于 | 输入 | target |
|------|------|------|--------|
| `MSELoss` | 回归 | 预测(浮点) | 同 shape 浮点 |
| `BCEWithLogitsLoss` | 二分类/多标签 | **logits** | `[0,1]` 浮点 |
| `CrossEntropyLoss` | 多分类(互斥) | **logits** `(N,C)` | 类别编号 `(N,)` long |

### 大坑：logits ≠ 概率

**logit** = 模型输出的**原始分数**(激活前的任意实数),**不是概率**。

- `CrossEntropyLoss` **内部已经包含 softmax**——所以你**不要**提前 `softmax`,直接喂 logits!(重复 softmax 是常见 bug)
- 想要概率才 `softmax(logits)`;想要预测类别用 `logits.argmax(dim=1)`

这呼应你 Week5 学的:`hidden → W_vocab → logits → softmax → 采样`。这里的 logits 就是那个"过 softmax 之前的分数"。

> **一句话记忆**：损失函数是现成的 `(ŷ-y)²`。大坑:`CrossEntropyLoss` 吃 **logits(原始分数)且自带 softmax**,别提前 softmax。logits≠概率,要概率才 softmax。

---

## 6. 优化器：现成的 update（SGD / Adam）

第一卷你手写 `w -= lr*g`。PyTorch 的优化器封装了这一步,还提供升级版:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# 或 torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

- **`model.parameters()`** 把所有参数交给优化器,`optimizer.step()` 就自动更新它们(用各自的 `.grad`)。
- **SGD**：`w -= lr*g`(你第一卷写的)。加 **momentum**(动量)= 给下山的球加惯性,冲过小坑、加速。
- **Adam / AdamW**：每个参数**自适应步长**(维护梯度的一阶矩、二阶矩),**大模型标配**。代价:要多存"优化器状态"(动量),所以**训练显存 = 参数 + 梯度 + 优化器状态**,是推理的好几倍。

> **一句话记忆**：优化器封装 `w -= lr*g`。SGD+momentum 加惯性;Adam/AdamW 每参数自适应步长(LLM 标配),但多存优化器状态 → 训练显存 = 参数+梯度+优化器状态(推理的好几倍)。

---

## 7. Dataset / DataLoader：喂数据的流水线

第一卷你手动 `for step`。真实训练要从磁盘读数据、shuffle、分 batch——`Dataset` + `DataLoader` 自动化这些:

- **`Dataset`**：定义"总共多少样本"和"怎么按下标取一个样本"。
- **`DataLoader`**：负责 **shuffle、组 batch、多进程加载、pin memory**(锁页内存,加速 H2D 拷贝——你 CUDA 里学过 pinned memory)。

```python
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(features, labels)
loader  = DataLoader(dataset, batch_size=64, shuffle=True)
for xb, yb in loader:      # 每次拿一个 batch
    ...
```

> **一句话记忆**：Dataset 定义"怎么取一个样本",DataLoader 自动 shuffle/组 batch/多进程/pin memory。pin memory 就是你 CUDA 的锁页内存,加速 H2D。

---

## 8. 把一切串起来：完整 PyTorch 训练循环

现在第一卷那四步,全用 PyTorch 写出来（这是所有训练代码的模板）：

```python
model = LinearModel(in_features, out_features).to("cuda")     # 模型(nn.Module)
loss_fn = nn.MSELoss()                                        # 损失(第2卷§5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)    # 优化器(第2卷§6)

for epoch in range(num_epochs):
    for xb, yb in loader:                    # DataLoader 喂数据(§7)
        xb, yb = xb.to("cuda"), yb.to("cuda")

        optimizer.zero_grad()                # ① 清零梯度(§2.1 的坑)
        pred = model(xb)                     # ② forward(第一卷第1步)
        loss = loss_fn(pred, yb)             # ③ loss(第一卷第2步)
        loss.backward()                      # ④ backward——autograd 自动算梯度(第一卷第3步)
        optimizer.step()                     # ⑤ update——优化器更新参数(第一卷第4步)
```

**对照第一卷的四步循环,一一对应**:
- `model(xb)` = forward
- `loss_fn(...)` = 算 loss
- `loss.backward()` = backward(**你不用手推链式法则了,这就是 PyTorch 的最大价值**)
- `optimizer.step()` = update
- 加一个 `zero_grad()` 是因为梯度默认累加

**记住这个模板 = 记住了 90% 的 PyTorch 训练代码。** 所有模型训练都是这五行的循环,只是 `model`、`loss_fn`、`optimizer` 换不同的。

> **一句话记忆**：训练循环 = `zero_grad → forward(model(x)) → loss → backward → step`。这就是第一卷四步的 PyTorch 版,backward 由 autograd 自动完成。背这五行 = 会写 PyTorch 训练。

---

## 9. 映射回你的 CUDA 世界

| PyTorch | 底层 CUDA 现实 |
|---------|---------------|
| `a @ b` | GEMM kernel(cuBLAS/自定义) |
| Tensor 的 stride | 你手算的 leading dimension |
| broadcasting | stride=0 的视图 |
| `loss.backward()` | 一串反向 GEMM/kernel(约前向 2× 算力) |
| ReLU / 激活 | 逐元素 kernel(memory-bound) |
| `pin_memory=True` | 锁页内存,加速 H2D |
| 优化器状态 | 额外显存(Adam 动量) |
| 自定义算子 | ⭐ 你写 CUDA kernel 用 `cpp_extension` 接进来 |

**最后一行是你的杀手锏**:PyTorch 允许你把自己写的 CUDA kernel 注册成算子,还能配 `autograd.Function` 手写 forward/backward。你的 CUDA 能力 + PyTorch 生态 = 差异化竞争力(第四卷会深入)。

---

## 10. 常见坑（第二卷易错点集合）

1. **忘记 `zero_grad`** → 梯度累加,训练乱掉。
2. **`CrossEntropyLoss` 前又 `softmax`** → 重复 softmax,结果错。
3. **`model.forward(x)` 而非 `model(x)`** → 跳过 hooks/autocast。
4. **`@` 和 `*` 搞混** → 矩阵乘 vs 逐元素乘。
5. **`model.eval()` 以为关了梯度** → 它只切 Dropout/BN 模式,关梯度要 `no_grad`。
6. **广播出巨中间量** → `(N,1,D)-(1,N,D)=(N,N,D)` 爆显存。
7. **自定义 CUDA 算子没检查 stride/contiguous** → 读到非连续内存出错。

---

## 11. 一页总结（背这个）

```
PyTorch = 把第一卷"forward→loss→backward→update"自动化

Tensor        = GPU 数组 + 元数据(shape/dtype/device/stride) + 自动求导
autograd      = forward 自动建图,backward() 自动跑链式法则算梯度(核心!)
nn.Module     = 参数容器,nn.Parameter 自动注册+求导+被优化器收集
损失函数       = 现成的 loss;坑:CrossEntropy 吃 logits 且自带 softmax
优化器         = 封装 w-=lr*g;Adam/AdamW 自适应步长(多存状态→显存涨)
DataLoader    = 自动 shuffle/batch/pin memory

训练循环模板(背这五行):
  optimizer.zero_grad()      # 清零梯度
  pred = model(xb)           # forward
  loss = loss_fn(pred, yb)   # loss
  loss.backward()            # backward(autograd 自动,不用手推链式法则!)
  optimizer.step()           # update

核心认知:PyTorch 最大价值 = 你只写 forward,backward 白送。
```

---

> 读法建议:先记住 §8 的训练循环模板(五行),再回头看每个 API 是模板里的哪一步。这样第二卷就从"一堆散概念"变成"围绕训练循环的工具"。
> 第三卷用这些搭一个 mini GPT(动手);第四卷讲这些怎么落到 GPU 算子/混合精度/显存/并行——那卷是你 CUDA 主场,重点看。
