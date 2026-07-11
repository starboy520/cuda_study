# ML 基础：从训练原理到 PyTorch、迷你 GPT 与 CUDA 系统

> 面向对象：会 C++/CUDA，但没有系统机器学习基础。
>
> 学习目标：先真正理解模型为什么能训练，再逐步学会 PyTorch，亲手训练一个迷你语言模型，最后把训练过程映射回 GEMM、显存、混合精度、profiling 和多卡通信。
>
> 验证环境：普通 Python 示例使用系统 Python；PyTorch 示例使用 \`/home/qichengjie/workspace/vllm_demo/.venv/bin/python\`，已在 PyTorch 2.11.0+cu130、NVIDIA A100 80GB PCIe（SM80）上验证。

---

## 0. 怎么使用这份教材

这不是一份需要一次读完的名词表，而是一条四阶段路线：

\`\`\`text
第一卷：手算训练原理，不依赖 PyTorch
第二卷：把手算逐项映射到 PyTorch
第三卷：从 token 和 cross entropy 走到迷你 GPT
第四卷：回到 CUDA / AI Infra，理解 backward、显存、混合精度、profiling 和多卡
\`\`\`

代码标记：

- **【演示】**：完整可运行，用于建立直觉。
- **【必须手写】**：先独立完成核心部分。
- **【提示 1/2/3】**：卡住后逐级看。
- **【参考实现】**：完成练习后对照。
- **【挑战】**：用于加深理解，不阻塞主线。

建议节奏：先手算，再运行演示，再关掉答案重写；每学完一卷，能回答该卷自测再继续。

你已有的 CUDA 知识会这样复用：

| CUDA 旧知识 | 训练中的位置 |
|---|---|
| GEMM | Linear forward、dX、dW、Attention、MLP |
| reduction | loss、db、norm、softmax |
| elementwise | 激活函数、optimizer update、mask |
| Tensor Core / BF16 | 混合精度训练 |
| CUDA Event / profiler | step time、operator/kernel 瓶颈 |
| stream / NCCL | 数据流水和多卡通信重叠 |

---

# 第一卷：从 CUDA 程序员到第一次模型训练

## 1. 这一卷要解决什么

你已经会 CUDA，说明你知道线程、数组、kernel、显存和并行归约。但机器学习会突然抛出一串新词：模型、参数、损失、梯度、反向传播、batch。它们并不神秘。本卷只做一件事：把这些词还原成你熟悉的数值计算。

读完并运行本卷程序后，你应该能够：

- 把模型看成一个“带可调参数的函数”；
- 手算一次前向计算、损失、梯度和 SGD 更新；
- 看懂导数、偏导、gradient、链式法则和计算图之间的关系；
- 用中心有限差分检查解析梯度；
- 写出矩阵 Linear 层的前向与反向形状；
- 理解 ReLU、两层 MLP，以及训练循环里的 sample、batch、step、epoch；
- 把这些操作对应到 CUDA kernel、GEMM 和 reduction。

本卷不要求概率论、线性代数课程或 PyTorch。完整程序只使用 Python 标准库。

---

## 2. 模型：一个带参数的函数

**模型（model）**就是一个函数：输入进去，输出出来。它与普通函数的特别之处在于，它含有可以通过数据调整的数值，这些数值叫作**参数（parameter）**。

最简单的线性模型是：

$$
\hat y = f(x;w,b)=wx+b
$$

- $x$：输入；
- $\hat y$（读作“y hat”）：模型的预测；
- $w$：权重，控制直线斜率；
- $b$：偏置，控制直线与纵轴的交点；
- 分号只是提醒我们：$x$ 是数据，$w,b$ 是要学习的参数。

训练不是让程序凭空“理解”数据，而是反复修改 $w,b$，让预测 $\hat y$ 更接近正确答案 $y$。

### 2.1 固定例子：一次完整训练更新

这一卷反复使用同一个例子：

$$
x=2,\quad y=5,\quad w=1,\quad b=0
$$

**前向传播（forward）**是从输入到预测、再到损失的计算过程。

第一步，预测：

$$
\hat y=wx+b=1\times2+0=2
$$

第二步，误差：

$$
e=\hat y-y=2-5=-3
$$

第三步，使用平方误差作为**损失（loss）**。损失是一个标量，用来衡量预测有多差：

$$
L=(\hat y-y)^2=e^2=(-3)^2=9
$$

损失越小越好。接下来要回答：怎样改 $w,b$ 才会让损失下降？答案由梯度给出。

---

## 3. 导数不是咒语：它就是局部斜率

### 3.1 从斜率到导数

直线的斜率是“输出变化量除以输入变化量”。曲线的斜率会随位置变化，因此我们在某个位置给输入一个很小的改变量 $h$：

$$
\frac{g(z+h)-g(z)}{h}
$$

当 $h$ 趋近于 0，这个局部斜率叫作 $g$ 对 $z$ 的**导数（derivative）**，写作 $dg/dz$。例如 $g(z)=z^2$，其导数是 $2z$。在 $z=-3$ 处，斜率是 $-6$：把 $z$ 稍微增大，$z^2$ 会减小。

导数的符号给方向，绝对值给敏感程度：

- 导数为正：输入稍增大，输出通常增大；
- 导数为负：输入稍增大，输出通常减小；
- 绝对值大：输出对输入敏感；
- 接近零：局部较平。

### 3.2 偏导与 gradient

当函数有多个输入，例如 $L(w,b)$，我们只改变一个变量、固定其他变量，得到**偏导数（partial derivative）**：

$$
\frac{\partial L}{\partial w},\qquad \frac{\partial L}{\partial b}
$$

把损失对所有参数的偏导按顺序收集起来，就得到**梯度（gradient）**：

$$
\nabla L=\left[\frac{\partial L}{\partial w},\frac{\partial L}{\partial b}\right]
$$

梯度指向损失局部上升最快的方向，所以训练要朝它的反方向走。

### 3.3 链式法则：沿计算路径乘局部斜率

我们的损失不是直接由 $w$ 算出，而是连续几步：

$$
w\longrightarrow \hat y=wx+b\longrightarrow e=\hat y-y\longrightarrow L=e^2
$$

这叫**计算图（computation graph）**：节点是中间数值，边表示依赖关系。每一步只需知道自己的局部导数：

$$
\frac{\partial L}{\partial e}=2e,
\quad \frac{\partial e}{\partial \hat y}=1,
\quad \frac{\partial \hat y}{\partial w}=x,
\quad \frac{\partial \hat y}{\partial b}=1
$$

**链式法则（chain rule）**说：一条路径上的总影响等于沿路局部导数相乘。因此：

$$
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial e}
 \frac{\partial e}{\partial \hat y}
 \frac{\partial \hat y}{\partial w}
=2ex
$$

$$
\frac{\partial L}{\partial b}
=\frac{\partial L}{\partial e}
 \frac{\partial e}{\partial \hat y}
 \frac{\partial \hat y}{\partial b}
=2e
$$

代入固定例子的 $e=-3,x=2$：

$$
\frac{\partial L}{\partial w}=2\times(-3)\times2=-12
$$

$$
\frac{\partial L}{\partial b}=2\times(-3)=-6
$$

所以梯度是 $[-12,-6]$。这段从最终损失反向计算每个参数影响的过程叫**反向传播（backward 或 backpropagation）**。反向传播不是另一套数学，它只是高效复用链式法则。

### 3.4 一次 SGD 更新

**SGD（stochastic gradient descent，随机梯度下降）**按照负梯度方向更新参数：

$$
w\leftarrow w-\eta\frac{\partial L}{\partial w},\qquad
b\leftarrow b-\eta\frac{\partial L}{\partial b}
$$

$\eta$ 是**学习率（learning rate）**，控制每步走多远。取 $\eta=0.1$：

$$
w\leftarrow1-0.1\times(-12)=2.2
$$

$$
b\leftarrow0-0.1\times(-6)=0.6
$$

更新后的预测为 $2.2\times2+0.6=5$，损失从 9 降为 0。这个例子恰好一步命中，并不表示一般训练也会如此；学习率太大还可能越过最低点并发散。

---

## 4. 用中心有限差分检查梯度

手推公式可能写错。**有限差分（finite difference）**通过轻微扰动参数来近似导数。更准确、误差通常更对称的**中心有限差分（central finite difference）**是：

$$
g'(z)\approx\frac{g(z+h)-g(z-h)}{2h}
$$

其中 $h$ 是很小的正数，例如 $10^{-6}$。检查 $w$ 时固定 $b$：

$$
\frac{\partial L}{\partial w}\approx
\frac{L(w+h,b)-L(w-h,b)}{2h}
$$

同理可检查 $b$。这叫**梯度检查（gradient check）**。它慢：每个参数至少多做两次 forward，所以只适合小例子和调试，不用于正常训练。$h$ 也不是越小越好；过小会受浮点舍入误差影响。

---

## 5. 从一个数扩展到矩阵 Linear

真实模型通常一次处理许多样本，每个样本也有多个特征。先约定形状：

- $B$：batch size，一批样本数；
- $I$：输入特征数（input features）；
- $O$：输出特征数（output features）。

**Linear 层（线性层，也常叫全连接层）**的矩阵形式为：

$$
Y=XW+b
$$

形状是：

| 张量 | 含义 | shape |
|---|---|---|
| $X$ | 一批输入 | $[B,I]$ |
| $W$ | 权重 | $[I,O]$ |
| $b$ | 每个输出通道一个偏置 | $[O]$ |
| $Y$ | 一批输出 | $[B,O]$ |

这里 $b$ 会被加到每一行，这种自动扩展叫**广播（broadcasting）**。矩阵乘法要求内部维度 $I$ 相同，输出保留外部维度 $B,O$。

反向传播时，上游给来 $dY=\partial L/\partial Y$，形状仍是 $[B,O]$。Linear 的三个结果是：

$$
dX=dY W^T \quad [B,O][O,I]\to[B,I]
$$

$$
dW=X^T dY \quad [I,B][B,O]\to[I,O]
$$

$$
db=\sum_{n=1}^{B}dY_n \quad [B,O]\to[O]
$$

注意 $db$ 沿 batch 维归约，因为同一个 $b$ 被一批中的所有样本共用。若 loss 定义为 batch 均值，上游 $dY$ 中还会带有 $1/B$。矩阵布局约定可能不同，但只要 forward 公式和 shape 前后一致，数学等价。

---

## 6. ReLU 与两层 MLP

如果连续堆叠 Linear 而不加别的操作，多个线性变换仍能合并成一个线性变换，表达能力没有本质增长。因此要插入**激活函数（activation function）**，也就是逐元素的非线性函数。

最常用的一个是 **ReLU（Rectified Linear Unit，修正线性单元）**：

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

其导数在 $z>0$ 时为 1，在 $z<0$ 时为 0。$z=0$ 处数学上不可导，软件通常约定梯度为 0。ReLU backward 就是把上游梯度乘以一个掩码 $z>0$。

**MLP（multilayer perceptron，多层感知机）**是由 Linear 和激活函数堆叠而成的网络。一个两层 MLP 可写为：

$$
Z_1=XW_1+b_1
$$

$$
H=\operatorname{ReLU}(Z_1)
$$

$$
Y=HW_2+b_2
$$

若隐藏特征数为 $H_d$，形状依次是：

- $X:[B,I]$，$W_1:[I,H_d]$，$b_1:[H_d]$；
- $Z_1,H:[B,H_d]$；
- $W_2:[H_d,O]$，$b_2:[O]$，$Y:[B,O]$。

backward 按 forward 的逆序执行：先对第二个 Linear 求导，再穿过 ReLU 掩码，最后对第一个 Linear 求导。每一步只接收上游梯度，乘自己的局部导数，再把梯度传给更早的节点。

---

## 7. 训练循环中的六个计量单位

这些词容易混淆，先给出精确定义。

**sample（样本）**是一条独立训练数据，例如一对 $(x,y)$。

**batch（批）**是一次共同计算 loss 和梯度的一组样本。**batch size** 是其中样本数。例如 1000 个样本，batch size 为 100，就有 10 个 batch。

**microbatch（微批）**是因显存放不下完整 batch，而实际一次送进设备的更小子批。四个 25 样本的 microbatch 可以共同构成一个有效 batch 100。

**gradient accumulation（梯度累积）**是在多个 microbatch 上依次 backward、把梯度相加，暂不更新参数；累积到指定次数后才更新。若要模拟完整 batch 的平均梯度，必须正确缩放：可以让每个 microbatch 的 loss 除以累积次数，也可以最后将梯度除以总样本数。最后一个 microbatch 大小不等时，应按样本数加权，不能盲目平均各 microbatch。

**step（优化器步）**通常指执行一次参数更新。只 forward/backward 一个 microbatch但尚未更新，不算一次 optimizer step。日志中的“step”偶尔也被项目用来表示 iteration，读代码时要确认定义。

**epoch（轮次）**指训练集中的每个样本大致被使用一次。假设 1000 个样本，有效 batch size 为 100，一轮约有 10 个 optimizer step。训练前通常打乱样本顺序，避免每轮都按同一顺序分组。

典型训练结构是：

1. 开始 epoch，打乱训练样本；
2. 取一个或多个 microbatch；
3. forward 得到预测与 loss；
4. backward 得到并累积梯度；
5. 达到有效 batch 后执行一次参数更新并清零梯度；
6. 重复，直到遍历数据；再进入下一 epoch。

“清零梯度”很关键：许多框架默认 backward 是把新梯度加到已有梯度上，而不是覆盖。

---

## 8. 完整程序：标量训练与梯度检查

下面的文件可以直接复制运行。它先复现固定例子的解析梯度并用中心有限差分检查；再用多条样本训练 $y=2x+1$，确认均方损失下降；全部满足后打印 `PASS`。

**【演示】完整程序 `scalar_training.py`**：先原样运行，观察解析梯度、有限差分与训练损失。

```python scalar_training.py
"""Pure-Python demonstration of gradient checking and linear regression."""


def predict(x, w, b):
    """Linear model: y_hat = w*x + b."""
    return w * x + b


def single_loss(x, y, w, b):
    """Squared error for one sample."""
    error = predict(x, w, b) - y
    return error * error


def analytic_gradients(x, y, w, b):
    """Return dL/dw and dL/db for L=(w*x+b-y)^2."""
    error = predict(x, w, b) - y
    dw = 2.0 * error * x
    db = 2.0 * error
    return dw, db


def central_difference(x, y, w, b, h=1e-6):
    """Numerically estimate dL/dw and dL/db."""
    dw = (
        single_loss(x, y, w + h, b)
        - single_loss(x, y, w - h, b)
    ) / (2.0 * h)
    db = (
        single_loss(x, y, w, b + h)
        - single_loss(x, y, w, b - h)
    ) / (2.0 * h)
    return dw, db


def mean_loss(samples, w, b):
    """Mean squared error over all samples."""
    total = 0.0
    for x, y in samples:
        total += single_loss(x, y, w, b)
    return total / len(samples)


def mean_gradients(samples, w, b):
    """Gradient of mean squared error over all samples."""
    total_dw = 0.0
    total_db = 0.0
    for x, y in samples:
        dw, db = analytic_gradients(x, y, w, b)
        total_dw += dw
        total_db += db
    count = len(samples)
    return total_dw / count, total_db / count


def main():
    # The fixed hand-worked example.
    x, y, w, b = 2.0, 5.0, 1.0, 0.0
    y_hat = predict(x, w, b)
    loss = single_loss(x, y, w, b)
    analytic_dw, analytic_db = analytic_gradients(x, y, w, b)
    numeric_dw, numeric_db = central_difference(x, y, w, b)

    print("fixed example:")
    print(f"  prediction={y_hat:.6f}, loss={loss:.6f}")
    print(f"  analytic gradient=({analytic_dw:.6f}, {analytic_db:.6f})")
    print(f"  numeric  gradient=({numeric_dw:.6f}, {numeric_db:.6f})")

    tolerance = 1e-5
    gradient_ok = (
        abs(analytic_dw - numeric_dw) < tolerance
        and abs(analytic_db - numeric_db) < tolerance
    )

    learning_rate = 0.1
    updated_w = w - learning_rate * analytic_dw
    updated_b = b - learning_rate * analytic_db
    updated_loss = single_loss(x, y, updated_w, updated_b)
    print(
        f"  one SGD update: w={updated_w:.6f}, b={updated_b:.6f}, "
        f"loss={updated_loss:.6f}"
    )

    # Diverse x values, all following y=2*x+1 exactly.
    samples = [
        (-3.0, -5.0),
        (-1.5, -2.0),
        (0.0, 1.0),
        (0.5, 2.0),
        (2.0, 5.0),
        (4.0, 9.0),
    ]
    w, b = 0.0, 0.0
    initial_loss = mean_loss(samples, w, b)
    for _ in range(300):
        dw, db = mean_gradients(samples, w, b)
        w -= 0.05 * dw
        b -= 0.05 * db
    final_loss = mean_loss(samples, w, b)

    print("multi-sample training:")
    print(f"  initial loss={initial_loss:.9f}")
    print(f"  final loss={final_loss:.9f}")
    print(f"  learned w={w:.6f}, b={b:.6f}")

    training_ok = final_loss < initial_loss and final_loss < 1e-10
    if gradient_ok and training_ok:
        print("PASS")
    else:
        raise AssertionError(
            f"verification failed: gradient_ok={gradient_ok}, "
            f"training_ok={training_ok}"
        )


if __name__ == "__main__":
    main()
```

**【必须手写】练习：不看程序，手写 `analytic_gradients` 与一次 SGD 更新。**

- **【提示 1】** 先写 `error = w*x+b-y`。
- **【提示 2】** 链式法则先求平方对 `error` 的导数，再乘 `error` 对参数的导数。
- **【提示 3】** 更新方向是梯度的反方向：参数减去 `learning_rate * gradient`。
- **【参考实现】** 完成后只对照上方程序中的 `analytic_gradients` 和训练循环。
- **【挑战】** 加入第三个参数 $c$，令模型变为 $wx^2+bx+c$，并同时做有限差分检查。

运行方法：

```bash
python3 scalar_training.py
```

观察重点不是 300 这个数字，而是三项事实：解析梯度与数值梯度非常接近；loss 从大变小；参数逼近 $w=2,b=1$。

---

## 9. 映射回 CUDA

机器学习框架没有绕开你熟悉的硬件工作，只是自动组织了大量 kernel。

| 训练概念 | 典型数值操作 | CUDA 视角 |
|---|---|---|
| 标量/逐元素 forward | 加、乘、max | elementwise kernel，每线程处理一个或多个元素 |
| Linear forward | $XW+b$ | GEMM（常由 cuBLAS 提供）加 bias kernel，或融合 epilogue |
| ReLU forward/backward | `max(0,z)`、掩码乘法 | 分支或谓词化的 elementwise kernel |
| batch loss | 每样本损失再求和/平均 | map 后 reduction |
| $dW=X^TdY$ | 矩阵乘法 | 另一轮 GEMM，不是“神秘反向指令” |
| $db=\sum_B dY$ | 沿 batch 维求和 | reduction kernel |
| SGD 更新 | `p -= lr * grad` | elementwise 参数更新 kernel |
| 计算图 | 记录操作与依赖 | host 侧调度，或编译/融合后形成更少 kernel |

shape 不只是数学记号，它决定线程如何映射数据、访存是否合并、是否需要转置、归约在哪个维度发生。训练比推理多出的主要工作，是保存 backward 所需的中间值（例如 ReLU 的输入或掩码），并计算参数梯度与输入梯度。因此训练通常需要更多显存。

梯度累积对应的底层动作也很直接：每个 microbatch 产生一份梯度，把它加到同一梯度缓冲区；累积完成后启动参数更新 kernel。多 GPU 训练还常用 all-reduce 汇总各设备梯度，但这属于后续内容。

性能上不要看到每个公式就机械写一个 kernel。框架或编译器可能把 bias、激活等操作融合，减少 global memory 往返和 launch 开销；大矩阵乘法则通常交给高性能库。先确保数学和 shape 正确，再谈融合、tiling 与占用率。

---

## 10. 常见误区

1. **预测与标签混为一谈。** $\hat y$ 是模型算出的，$y$ 是数据给出的正确答案。
2. **把梯度当作参数改变量。** 梯度是损失的局部斜率；真正改变量还要乘负学习率。
3. **忘记 loss 是和还是均值。** 两者梯度相差 batch size 倍，学习率表现也会不同。
4. **认为 backward 是 forward 的数值逆运算。** backward 不是恢复输入，而是用链式法则传播损失敏感度。
5. **有限差分通过就代表训练一定成功。** 它只支持“局部梯度实现正确”，数据、优化器和学习率仍可能有问题。
6. **gradient accumulation 后每个 microbatch 都更新。** 那会变成多个小 batch step，不再等价于一个大有效 batch。
7. **只检查元素数量，不检查轴含义。** 同样的元素数不表示 $[B,I]$ 与 $[I,B]$ 可以互换。

---

## 11. 自测题

先独立完成，再看答案。

1. 模型 $\hat y=wx+b$ 中，哪些是参数，哪个是输入？训练直接修改哪个量？
2. 对固定例 $x=2,y=5,w=1,b=0$，写出预测、误差和平方损失。
3. 为什么 $\partial L/\partial w=-12$ 时，SGD 会增大 $w$？
4. 若学习率改为 0.01，固定例一次更新后的 $w,b$ 是多少？
5. 链式法则为何是路径上的局部导数相乘？用 $w\to\hat y\to e\to L$ 写出式子。
6. 中心有限差分为何需要两次额外 forward？它适合日常完整训练吗？
7. $X:[32,128]$，$W:[128,64]$，$b:[64]$，求 $Y,dY,dX,dW,db$ 的 shape。
8. 为什么没有 ReLU 的两层 Linear 仍等价于一层 Linear？
9. 1024 个样本，microbatch size 32，累积 4 次更新，忽略最后不足批的情况：有效 batch size 和每 epoch 的 optimizer step 各是多少？
10. Linear backward 中，哪两个梯度主要由 GEMM 得到，哪个梯度需要沿 batch 维 reduction？

## 12. 自测答案

1. $w,b$ 是参数，$x$ 是输入。训练算法直接修改参数；数据通常不被 SGD 修改。
2. $\hat y=2$，$e=-3$，$L=9$。
3. 更新式是 $w\leftarrow w-\eta\partial L/\partial w$。减去负数等于加正数，所以 $w$ 增大。
4. $w=1-0.01(-12)=1.12$，$b=0-0.01(-6)=0.06$。
5. 每个中间变量的微小变化会按局部斜率缩放，连续缩放的总倍率相乘：$\partial L/\partial w=(\partial L/\partial e)(\partial e/\partial\hat y)(\partial\hat y/\partial w)=2e\cdot1\cdot x$。
6. 分别计算 $L(z+h)$ 和 $L(z-h)$，因此每个待查参数至少需要两次 forward。它太慢，只适合小规模调试。
7. $Y:[32,64]$，$dY:[32,64]$，$dX:[32,128]$，$dW:[128,64]$，$db:[64]$。
8. 仿射变换的复合仍是仿射变换：$(XW_1+b_1)W_2+b_2=X(W_1W_2)+(b_1W_2+b_2)$。ReLU 打破这种可合并性。
9. 有效 batch size 为 $32\times4=128$，每 epoch 为 $1024/128=8$ 个 optimizer step。
10. $dX=dYW^T$ 与 $dW=X^TdY$ 是 GEMM；$db=\sum_BdY$ 是沿 batch 维 reduction。

---

## 13. 小结

训练可以压缩为一个循环：带参数函数做 forward，loss 把预测质量压成标量，backward 用链式法则求梯度，优化器沿负梯度方向改参数。标量例子中的乘法、加法和求和，扩展到 batch 后就成为 elementwise kernel、GEMM 与 reduction。后续学习框架 API 时，始终追问四件事：张量 shape 是什么、forward 算什么、backward 需要什么、参数何时更新。这样 API 只是语法，不会遮住计算本身。

<!--
VALIDATION
Extraction command:
sed -n '/^```python scalar_training.py$/,/^```$/p' /tmp/ml_training_agents/volume1.md | sed '1d;$d' > /tmp/ml_training_agents/scalar_training.py

Run command:
python3 /tmp/ml_training_agents/scalar_training.py

Observed output (2026-07-04 UTC):
fixed example:
  prediction=2.000000, loss=9.000000
  analytic gradient=(-12.000000, -6.000000)
  numeric  gradient=(-12.000000, -6.000000)
  one SGD update: w=2.200000, b=0.600000, loss=0.000000
multi-sample training:
  initial loss=23.333333333
  final loss=0.000000000
  learned w=2.000000, b=1.000000
PASS
-->

---

# 第二卷：从 CUDA 程序员到 PyTorch 训练者

> 读者画像：会写 CUDA kernel，理解线程、显存、步长和矩阵乘法，但尚未系统学习机器学习。

这一卷只做一件事：把“训练神经网络”还原成你已经熟悉的计算过程。模型是带参数的函数，损失是一个标量，反向传播是自动生成的梯度程序，优化器则根据梯度更新参数。

## 1. Tensor：带元数据的多维数组

CUDA 中，一段显存本身不知道它是图片还是矩阵。PyTorch 的 `Tensor` 在一段 storage 上附加元数据：

- `shape`：每一维的元素数，例如 `(32, 784)`；
- `dtype`：每个元素的编码，例如 `float32`；
- `device`：数据和运算所在设备，例如 `cpu` 或 `cuda:0`；
- `stride`：每一维下标增加 1 时，storage 偏移增加多少个元素。

```python
import torch

x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print(x.shape, x.dtype, x.device, x.stride())  # (3,4), float32, ..., (4,1)
```

`x[i, j]` 的线性元素偏移是 `i*x.stride(0) + j*x.stride(1)`，与 CUDA 中手算 pitch/leading dimension 完全同源。`untyped_storage()` 是底层字节 storage；多个 Tensor 可以共享它，但拥有不同 shape、stride 和 storage offset。通常不要直接改 storage。

### 1.1 view、reshape、transpose、contiguous

`view` 只改元数据，不复制数据，因此要求当前 stride 能表示目标形状。`reshape` 优先返回 view，做不到时会悄悄复制；不能依赖它一定共享 storage。`transpose(0, 1)` 交换 shape 和 stride，通常不复制，于是结果常为 non-contiguous。`contiguous()` 在需要时分配并按逻辑次序复制。

```python
x = torch.arange(12).reshape(3, 4)
y = x.transpose(0, 1)
assert not y.is_contiguous()
z = y.contiguous().view(12)
```

这对应第一卷中“逻辑索引”和“物理地址”的区别。自定义 CUDA 扩展若假设连续内存，应先检查 stride，或明确调用 `contiguous()`；后者有复制成本。

### 1.2 broadcasting：虚拟扩展维度

形状从右向左对齐；对应维度相等、其中一个为 1、或某侧不存在时可以广播：

```python
x = torch.randn(32, 128)
b = torch.randn(128)
y = x + b                 # b 被逻辑地视为 (32,128)，一般不真实复制
```

广播可理解为某维 stride 为 0 的只读视图。危险点是无意生成巨大中间量，例如 `(N,1,D)-(1,N,D)` 得到 `(N,N,D)`。

### 1.3 matmul 与 batch 维

二维时 `a @ b` 是矩阵乘法 `(M,K)@(K,N)->(M,N)`；三维及以上时最后两维是矩阵维，前面的维按广播规则作为 batch。不要把 `*`（逐元素乘）误当矩阵乘。

```python
a = torch.randn(8, 16, 32)
b = torch.randn(1, 32, 64)
c = a @ b                 # (8,16,64)
```

## 2. autograd：动态构建反向计算图

当输入或参数 `requires_grad=True` 时，PyTorch 记录参与计算的算子。结果的 `grad_fn` 指向生成它的反向节点。用户创建且不是其他可求导运算结果的张量称为 leaf；模型的 `Parameter` 通常是 leaf，梯度最终累积到 leaf 的 `.grad`。

```python
w = torch.tensor(2.0, requires_grad=True)  # leaf
y = w * w + 3 * w                         # 非 leaf，有 grad_fn
y.backward()                              # 标量可省略上游梯度
print(w.grad)                             # dy/dw = 2w+3 = 7
```

`backward()` 实现向量—雅可比积。直白地说，**雅可比**就是“每个输出分别对每个输入求导”排成的表；**上游梯度**是最终 loss 对当前各输出的导数，也就是后续计算传回来的权重。反向传播不必真的构造那张可能很大的表，而是把上游梯度与它相乘，直接得到 loss 对输入的梯度。非标量输出必须传入同形状上游梯度，如 `y.backward(torch.ones_like(y))`，或先 `y.sum().backward()`。

### 2.1 梯度默认累积

连续两次 `backward()` 会把结果加到 `.grad`，这和 CUDA kernel 中对同一梯度缓冲区做累加相似。训练迭代通常先调用 `optimizer.zero_grad(set_to_none=True)`。`None` 可少一次清零写带宽，并能区分“没有梯度”和“梯度恰为零”。若想重复反传同一张图，需要 `retain_graph=True`，但训练循环通常应重新 forward，而不是长期保图。

### 2.2 no_grad、detach 与原地操作

- `with torch.no_grad():` 暂时不记录图，适合验证和手动更新；
- `x.detach()` 返回共享数据但切断历史的新 Tensor；若还要独立修改，使用 `x.detach().clone()`；
- 后缀 `_` 常表示 in-place，例如 `add_`。autograd 会保存反向所需值并检查版本号，随意原地修改可能报错，或对 requires-grad leaf 直接修改时报错。

不要用 `.data` 绕过检查。推理可用 `torch.inference_mode()` 获得更激进的开销优化，但其 Tensor 的限制也更强。

## 3. 三种梯度必须对齐

下面同时做符号手算、中心有限差分和 autograd。函数为 `L(w)=(wx-y)^2`，手算 `dL/dw=2(wx-y)x`。有限差分 `(L(w+eps)-L(w-eps))/(2eps)` 是数值近似；用 float64 和适当 eps 减少舍入误差。

**【演示】完整程序 `autograd_demo.py`**：运行后确认三种梯度逐项对齐。

```python autograd_demo.py
import torch

torch.manual_seed(0)
dtype = torch.float64
x = torch.tensor(3.0, dtype=dtype)
target = torch.tensor(7.0, dtype=dtype)
w = torch.tensor(2.0, dtype=dtype, requires_grad=True)

def loss_fn(weight):
    return (weight * x - target) ** 2

loss = loss_fn(w)
loss.backward()
autograd_grad = w.grad.item()
manual_grad = (2 * (w.detach() * x - target) * x).item()

eps = 1e-6
with torch.no_grad():
    finite_difference = ((loss_fn(w + eps) - loss_fn(w - eps)) / (2 * eps)).item()

print(f"loss={loss.item():.6f}")
print(f"manual={manual_grad:.9f}")
print(f"finite_difference={finite_difference:.9f}")
print(f"autograd={autograd_grad:.9f}")
assert abs(manual_grad - finite_difference) < 1e-6
assert abs(manual_grad - autograd_grad) < 1e-12
print("PASS")
```

有限差分仅适合检查少量参数：每个参数至少多两次 forward，成本远高于一次反向传播。实际工程可用 `torch.autograd.gradcheck` 检查自定义算子。

## 4. nn.Module：组织参数与子模块

`nn.Module` 是参数、buffer 和子模块的树。把 `nn.Parameter` 赋给模块属性后，它会自动注册、默认参与求导并由 `model.parameters()` 交给优化器。**buffer** 是用 `register_buffer` 注册的非训练状态，例如 BatchNorm 的运行均值：它不由优化器更新，但会随 `model.to(device)` 移动，并默认进入 `state_dict`。普通 Tensor 属性既不会出现在 `parameters()`，也不会自动随模块迁移或保存；临时量适合做普通 Tensor，需要持久化的非训练状态才用 buffer。定义 `forward`，调用时写 `model(x)`，不要直接写 `model.forward(x)`，因为 `__call__` 还负责 hooks、autocast 等框架行为。

```python
from torch import nn

class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.T + self.bias
```

`state_dict()` 是从名字到参数/buffer Tensor 的映射，是推荐的 checkpoint 内容。保存 `torch.save(model.state_dict(), path)`，加载到相同结构的实例后 `load_state_dict(...)`。

`model.train()` 与 `model.eval()` 切换行为模式，不是启用/禁用梯度。它们影响 Dropout 和 BatchNorm；验证时通常同时使用 `model.eval()` 和 `torch.no_grad()`。重新训练前再 `model.train()`。

## 5. 损失函数与 logits

logit 是激活前的任意实数分数，不是概率。稳定的损失函数会把激活和对数运算融合，避免 `log(0)` 与溢出。

- `MSELoss`：回归；预测和 target 通常同 shape、浮点 dtype，计算平方误差均值。
- `BCEWithLogitsLoss`：二分类或多标签；输入 logits 与浮点 target 同 shape，target 通常在 `[0,1]`。预测概率才用 `sigmoid(logits)`。
- `CrossEntropyLoss`：互斥多分类；输入 `(N,C)` logits，target 是 `(N,)` 的 `torch.long` 类别编号。它已包含 log-softmax，不要提前 softmax。预测用 `logits.argmax(dim=1)`。

常见混淆：二分类可以输出一个 logit 配 BCE，也可输出两个 logits 配 CE，但 target/shape 必须与所选定义一致。

## 6. 优化器：如何使用梯度

一次标准迭代是 `zero_grad -> forward -> loss -> backward -> step`。

### SGD 与 momentum

SGD 更新近似为 `w <- w - lr*g`。momentum 保存梯度的指数惯性，沿一致方向加速、对摆动方向平滑。学习率是首要超参数。

### Adam 与 AdamW

Adam 同时维护梯度一阶矩和平方梯度二阶矩，并做偏差校正，使各参数获得自适应步长。Adam 的 `weight_decay` 历史上可能作为 L2 项耦合进梯度；AdamW 将权重衰减与梯度更新解耦，训练现代网络时通常更符合预期。它们保存每参数状态，因此显存不只有参数与梯度，还包括优化器状态。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# 或 torch.optim.Adam(..., lr=1e-3)
# 或 torch.optim.AdamW(..., lr=1e-3, weight_decay=1e-2)
```

## 7. Dataset、DataLoader 与传输流水线

`Dataset` 定义样本总数和如何按索引取一个样本；`DataLoader` 负责 shuffle、batch、可选多进程和内存固定。默认 collate 会把一批同 shape Tensor 沿第 0 维堆叠。

```python
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(features, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True,
                    pin_memory=(device.type == "cuda"))
for x, y in loader:
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
```

页锁定（pinned）CPU 内存允许 CUDA DMA 更高效地异步读取。`non_blocking=True` 只有在来源等条件允许时才真正异步；要与计算重叠还需要合适 stream 和预取。小数据集 pinning/多 worker 反而可能更慢。

### 梯度累积模拟大 batch

显存放不下大 batch 时，把 `accum_steps` 个 micro-batch 的梯度相加再 `step()`。如果 loss 使用默认的 `reduction="mean"`，简单除以 `accum_steps` 只在每个 micro-batch 样本数相同且没有不足一组的尾批时严格正确。下面采用更安全的模板：先用 `reduction="sum"` 累积每个样本的损失梯度，到更新边界再按这一组实际样本总数缩放梯度。这样最后不足 `accum_steps` 的尾组也会更新且尺度正确。BatchNorm 仍看到 micro-batch，不能完全等价于真实大 batch。

```python
criterion_sum = torch.nn.CrossEntropyLoss(reduction="sum")
optimizer.zero_grad(set_to_none=True)
group_samples = 0
for i, (x, y) in enumerate(loader):
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    loss_sum = criterion_sum(model(x), y)
    loss_sum.backward()
    group_samples += y.numel()
    group_end = (i + 1) % accum_steps == 0 or i + 1 == len(loader)
    if group_end:
        # sum reduction 的梯度除以真实样本数，得到这一组的 mean 梯度。
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(group_samples)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        group_samples = 0
```

## 8. 完整分类器：训练、验证、保存与重载

这是一个可直接运行的二维三类问题。固定随机种子使教学结果可复现；GPU 算法仍可能因环境差异有微小浮动。命令行 `--device cpu` 或 `--device cuda` 显式选择设备。

**【参考实现】完整程序 `mlp_classifier.py`**：先完成下方练习，再运行它检查 loss、准确率和重载一致性。

```python mlp_classifier.py
import argparse
import io
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.net(x)  # logits，不做 softmax

def make_data(seed=1234):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[-2.0, -1.5], [2.0, -1.0], [0.0, 2.2]])
    xs, ys = [], []
    for label, center in enumerate(centers):
        xs.append(center + 0.55 * torch.randn(400, 2, generator=generator))
        ys.append(torch.full((400,), label, dtype=torch.long))
    x, y = torch.cat(xs), torch.cat(ys)
    order = torch.randperm(len(y), generator=generator)
    return x[order], y[order]

def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA, but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    seed_everything(7)

    x, y = make_data()
    train_ds = TensorDataset(x[:900], y[:900])
    val_ds = TensorDataset(x[900:], y[900:])
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              generator=torch.Generator().manual_seed(99),
                              pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=128, pin_memory=pin)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)
    losses = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running += loss.item() * batch_y.numel()
            count += batch_y.numel()
        epoch_loss = running / count
        losses.append(epoch_loss)
        print(f"epoch={epoch + 1:02d} loss={epoch_loss:.6f}")

    val_acc = accuracy(model, val_loader, device)
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    assert val_acc > 0.90, val_acc

    # BytesIO 让示例不遗留文件；真实项目通常传入 checkpoint 路径。
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    reloaded = MLP().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    reloaded.load_state_dict(state)
    reloaded.eval()
    model.eval()
    probe = x[900:916].to(device)
    with torch.no_grad():
        old_logits = model(probe)
        new_logits = reloaded(probe)
    assert torch.equal(old_logits, new_logits)
    print(f"device={device} val_accuracy={val_acc:.4f}")
    print("checkpoint reload logits identical")
    print("PASS")

if __name__ == "__main__":
    main()
```

## 9. 常见错误速查

1. **device 不一致**：CPU 输入喂给 CUDA 模型。统一 `.to(device)`，并确认新建 Tensor 的 device。
2. **dtype 不一致**：线性层通常是 float32，输入却是 float64；CE target 又必须是 long。打印 dtype，不要盲目转换全部数据。
3. **shape 错但广播没报错**：预测 `(N,1)` 与 target `(N,)` 做 MSE 会广播成 `(N,N)`。在 loss 前 assert shape。
4. **CE 前做 softmax**：数值稳定性和梯度都变差。直接传 logits。
5. **忘记 zero_grad**：梯度跨迭代累积，训练异常。
6. **验证时忘记 eval**：Dropout 随机、BatchNorm 更新统计；但 `eval()` 本身并不关闭梯度。
7. **把 tensor loss 长期存进列表**：保存着整张图导致显存增长。记录 `loss.item()` 或 `loss.detach()`。
8. **在 requires-grad leaf 上原地更新**：使用 optimizer；手写更新要放在 `no_grad` 中。
9. **transpose 后直接 view**：stride 不兼容。用 `reshape`，或明确 `contiguous().view(...)` 并接受复制。
10. **错误理解 detach**：它共享 storage；独立快照应 `detach().clone()`。
11. **DataLoader worker 过多**：启动、IPC 和内存复制可能主导耗时。用 profiler 测量。
12. **梯度爆炸/NaN**：先检查输入、loss 与学习率，再考虑梯度裁剪；不要只用 `nan_to_num` 掩盖根因。
13. **checkpoint 只存模型却想无缝续训**：精确续训还需 optimizer、scheduler、epoch 与 RNG 状态。optimizer 含 momentum/Adam 矩，scheduler 决定当前学习率进度，RNG 状态决定 dropout、shuffle 与采样；漏掉它们会让恢复后的下一步不再等价于中断前轨迹。
14. **用 accuracy 参与 backward**：argmax 不可导；训练优化可导 loss，accuracy 只是指标。

## 10. 从 CUDA 心智模型到训练心智模型

forward 是一次 kernel/DAG 调度，autograd 根据已记录的算子生成反向 DAG；Parameter 是需要持久化并更新的 device buffer；optimizer state 是额外持久 buffer；DataLoader 与 H2D 传输是输入流水线。性能分析仍遵循第一卷原则：先测量，再区分算力、带宽、launch、同步和输入瓶颈。不同之处在于训练还必须保证数学语义正确：shape、归约尺度、模式、梯度生命周期都可能让“能运行”的程序学不到东西。

## 11. 卷末自测

1. transpose 后的 Tensor 为什么可能不能 `view`？`reshape` 又可能付出什么代价？
2. 为什么连续两次 `backward()` 会改变 `.grad`？标准训练步如何避免误累积？
3. Parameter、buffer、普通 Tensor 属性在求导、设备迁移和 `state_dict` 上有何区别？
4. `CrossEntropyLoss` 为什么接收 logits，而不是先 softmax 的概率？
5. **【必须手写】** 写出一个 micro-batch 训练步，要求包含设备搬运、清梯度、forward、loss、backward 和 step。
   - **【提示 1】** 训练次序是 `zero_grad → model(x) → criterion → backward → step`。
   - **【提示 2】** 标签也要移动到模型所在设备。
   - **【提示 3】** CE 的标签应为 `long` 类别编号。

## 12. 自测答案

1. transpose 常只交换 stride，逻辑顺序不再是连续物理顺序；`view` 无法仅改元数据时会失败，`reshape` 可能分配并复制。
2. autograd 设计为把新梯度加进 leaf 的 `.grad`，以支持多路径和梯度累积；普通迭代在 backward 前调用 `optimizer.zero_grad(set_to_none=True)`。
3. Parameter 默认求导、会迁移、会保存；buffer 不由优化器更新但会迁移且默认保存；普通 Tensor 属性三者都不会自动获得。
4. 融合 log-softmax 与 NLL 可避免指数溢出和 `log(0)`，且接口能使用稳定公式。
5. **【参考实现】** `x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True); loss=criterion(model(x),y); loss.backward(); optimizer.step()`。

<!--
验证环境（2026-07-04）：
- 解释器：/home/qichengjie/workspace/vllm_demo/.venv/bin/python
- PyTorch：2.11.0+cu130；GPU：NVIDIA A100 80GB PCIe（SM80）
- 从本文命名围栏原样提取 autograd_demo.py：PASS；loss=1；manual=-6；finite_difference=-6；autograd=-6。
- 原样提取 mlp_classifier.py，运行 --device cpu：PASS；首轮/末轮 loss=0.265401/0.000487；val_accuracy=1.0000；checkpoint reload logits identical。
- 原样提取 mlp_classifier.py，运行 --device cuda：PASS；首轮/末轮 loss=0.265401/0.000487；val_accuracy=1.0000；checkpoint reload logits identical。
-->

---

# 第三卷：从一个字符到迷你 GPT

本卷假定你会写 CUDA kernel，却刚接触 PyTorch。目标不是背 API，而是把训练看成：张量布局、数值稳定、自动微分和优化循环的组合。需要更完整的推导与 FlashAttention 实现时可读 [`docs/courses/attention/Week4_Attention与FlashAttention完整学习资料.md`](../attention/Week4_Attention与FlashAttention完整学习资料.md)；下面先在本卷就地建立读懂程序所需的最小概念，不依赖不存在的前卷内容。

## 1. Attention 最小词典：Q、K、V、head 与残差

输入 `x` 的形状是 `[B,T,C]`：batch、序列位置、隐藏维。Attention 让每个位置按内容从允许看到的位置收集信息。三个线性投影产生：Query（Q，“我在找什么”）、Key（K，“我能被怎样匹配”）和 Value（V，“匹配后取走什么内容”）。`Q @ Kᵀ / sqrt(d)` 得到位置两两之间的分数；causal mask 禁止看未来；softmax 把分数变权重，再乘 V 得到加权信息。

一个 **head** 是一套较窄的 Q/K/V 子空间。多头把隐藏维 C 拆成 H 份，每头宽 `d=C/H`，让不同 head 学不同关系，最后拼回 C。**LayerNorm** 对每个 token 的隐藏维做归一化，缓和数值尺度漂移。**residual（残差连接）**把子层结果加回原输入，即 `x + sublayer(x)`，给信息和梯度提供短路径。**pre-norm** 表示先归一化再进子层：`x = x + attention(LN(x))`，随后 `x = x + MLP(LN(x))`；迷你 GPT 正采用此结构。

**【必须手写】练习：** 给定 `B=2,T=8,C=32,H=4`，写出拆头后 Q、注意力分数和拼头后输出的 shape。

- **【提示 1】** 每头宽度 `d=C/H`。
- **【提示 2】** 拆头 Q 为 `[B,H,T,d]`。
- **【提示 3】** `Q @ K.transpose(-2,-1)` 的最后两维是 `[T,T]`。
- **【参考实现】** Q 是 `[2,4,8,8]`，分数是 `[2,4,8,8]`，拼头输出回到 `[2,8,32]`。

## 2. logits、概率与损失

模型最后一层不直接输出概率，而输出任意实数 logits。设词表为 `['a','b','c']`，某位置 logits 为 `[2,1,0]`。直接 softmax 是

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

稳定实现先减最大值（不改变比值）：`[0,-1,-2]`，指数约为 `[1,.3679,.1353]`，和为 `1.5032`，概率约 `[.6652,.2447,.0900]`。若真值是 `b`，负对数似然 NLL 为 `-log(.2447)=1.4076`。交叉熵对单个 one-hot 标签正是这个 NLL；批量交叉熵是各 token NLL 的均值。工程实现使用 log-sum-exp：`-z_y + logsumexp(z)`，不要先算概率再 `log`。

困惑度 `perplexity = exp(mean NLL)`。上例约 `4.086`。它可理解为模型平均面对多少个“等可能选择”，越低越好；不同 tokenizer 的 token 粒度不同，困惑度不可直接横比。

语言模型 logits 常为 `[B,T,V]`，标签为 `[B,T]`。`cross_entropy` 期望类别维第二维，最清楚的做法是把所有位置当独立样本：

```python
loss = torch.nn.functional.cross_entropy(logits.reshape(B*T, V), targets.reshape(B*T))
```

`reshape` 会在必要时复制，`view` 要求内存连续；转置后尤其要留意 stride。

## 3. 数据：字符 tokenizer 与错一位预测

字符 tokenizer 是教学用的最小 tokenizer：收集排序后的字符集，建立 `stoi/itos`，encode 把字符串变整数，decode 反向映射。真实模型通常用 BPE/SentencePiece，原理仍是离散 token ID。

给序列 `hello`，训练片段 `x=hell`、`y=ello`，即 target 左移一位。Teacher forcing 表示训练位置 `t` 总能看到真实前缀，而非模型刚生成的错误 token；这允许所有位置并行算 loss。推理时没有真值，只能自回归逐个追加。

仅仅 shift 不足以阻止偷看未来。自注意力分数 `[T,T]` 的上三角必须填 `-inf`，softmax 后未来权重为 0，这就是 causal mask。PyTorch 的 `scaled_dot_product_attention(..., is_causal=True)` 或显式下三角 mask 都可实现。

## 4. 第一站：bigram / embedding 模型

最简单模型以当前字符直接查表得到“下一个字符”的 V 维 logits；参数矩阵 `[V,V]` 本质是 `Embedding(V,V)`。它不记更长上下文，却完整展示采样、反传、checkpoint 和生成。

**【演示】完整程序 `mini_lm.py`**：先运行最小语言模型，观察 token loss 和生成结果。

```python mini_lm.py
import argparse, math, os, tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7)
TEXT = ("hello cuda, hello pytorch!\n" * 80)
chars = sorted(set(TEXT)); stoi = {c:i for i,c in enumerate(chars)}; itos = dict(enumerate(chars))
data = torch.tensor([stoi[c] for c in TEXT], dtype=torch.long)

class BigramLM(nn.Module):
    def __init__(self, vocab):
        super().__init__(); self.table = nn.Embedding(vocab, vocab)
    def forward(self, x, y=None):
        logits = self.table(x)
        loss = None if y is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, x, n):
        for _ in range(n):
            logits, _ = self(x[:, -1:]); p = F.softmax(logits[:, -1], dim=-1)
            x = torch.cat((x, torch.multinomial(p, 1)), dim=1)
        return x

def batch(device, B=32, T=16):
    ix = torch.randint(len(data)-T-1, (B,))
    x = torch.stack([data[i:i+T] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+T+1] for i in ix]).to(device)
    return x, y

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=120); ap.add_argument('--device',default='auto'); a=ap.parse_args()
    device = torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device))
    m=BigramLM(len(chars)).to(device); opt=torch.optim.AdamW(m.parameters(),lr=.08)
    eval_x,eval_y=batch(device)
    @torch.no_grad()
    def eval_loss():
        return m(eval_x,eval_y)[1].item()
    initial=eval_loss()
    for _ in range(a.steps):
        x,y=batch(device); _,loss=m(x,y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    final=eval_loss(); assert final < initial, (initial,final)
    # 单 batch 过拟合：训练集足够小，loss 应明显下降
    ox,oy=batch(device,B=4,T=8); over0=m(ox,oy)[1].item()
    oo=torch.optim.AdamW(m.parameters(),lr=.05)
    for _ in range(80):
        _,l=m(ox,oy); oo.zero_grad(set_to_none=True); l.backward(); oo.step()
    over1=m(ox,oy)[1].item(); assert over1 < over0
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,'bigram.pt'); torch.save({'model':m.state_dict(),'stoi':stoi},p)
        m2=BigramLM(len(chars)).to(device); ck=torch.load(p,map_location=device,weights_only=True); m2.load_state_dict(ck['model'])
        assert torch.equal(m.table.weight,m2.table.weight)
        out=m2.generate(torch.tensor([[stoi['h']]],device=device),20)[0].tolist()
    print(f'PASS device={device} loss={initial:.3f}->{final:.3f} overfit={over0:.3f}->{over1:.3f} sample={"".join(itos[i] for i in out)!r}')
if __name__=='__main__': main()
```

## 5. 从 embedding 到迷你 GPT

GPT 将 token embedding `[B,T,C]` 与 position embedding `[T,C]` 相加。每个 block 是 pre-norm：`x += causal_attention(LN(x))`，再 `x += MLP(LN(x))`。多头只是把 C 拆成 H 份，每头宽 D=C/H；QKᵀ 得 `[B,H,T,T]`，缩放 `1/sqrt(D)` 后施加 causal mask，再乘 V。最后 LayerNorm 与线性词表头生成 `[B,T,V]`。

下面程序刻意小，可 CPU 20 步，也可 CUDA 短跑。20 步不保证随机 mini-batch 的末步小于首步，所以程序用固定评估 batch 比较训练前后；这是比“打印最后一步”更可靠的验收。

**【挑战】完整程序 `mini_gpt.py`**：先依据上节画出一个 Block 的数据流，再运行完整实现。

```python mini_gpt.py
import argparse, os, tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(11)
TEXT=("to be or not to be, cuda makes tensors fly.\n"*100)
chars=sorted(set(TEXT)); stoi={c:i for i,c in enumerate(chars)}; itos=dict(enumerate(chars))
raw=torch.tensor([stoi[c] for c in TEXT],dtype=torch.long)

class Attention(nn.Module):
    def __init__(self,c,h,drop=0.):
        super().__init__(); assert c%h==0; self.h=h; self.d=c//h
        self.qkv=nn.Linear(c,3*c); self.proj=nn.Linear(c,c); self.drop=drop
    def forward(self,x):
        B,T,C=x.shape; q,k,v=self.qkv(x).chunk(3,dim=-1)
        def split(z): return z.view(B,T,self.h,self.d).transpose(1,2)
        q,k,v=map(split,(q,k,v))
        a=F.scaled_dot_product_attention(q,k,v,dropout_p=self.drop if self.training else 0.,is_causal=True)
        return self.proj(a.transpose(1,2).contiguous().view(B,T,C))
class Block(nn.Module):
    def __init__(self,c,h):
        super().__init__(); self.n1=nn.LayerNorm(c); self.a=Attention(c,h); self.n2=nn.LayerNorm(c)
        self.m=nn.Sequential(nn.Linear(c,4*c),nn.GELU(),nn.Linear(4*c,c))
    def forward(self,x): return (lambda y:y+self.m(self.n2(y)))(x+self.a(self.n1(x)))
class GPT(nn.Module):
    def __init__(self,V,T=32,C=48,H=4,L=2):
        super().__init__(); self.T=T; self.tok=nn.Embedding(V,C); self.pos=nn.Embedding(T,C)
        self.blocks=nn.Sequential(*[Block(C,H) for _ in range(L)]); self.norm=nn.LayerNorm(C); self.head=nn.Linear(C,V,bias=False)
        self.head.weight=self.tok.weight
    def forward(self,x,y=None):
        B,T=x.shape; assert T<=self.T
        z=self.tok(x)+self.pos(torch.arange(T,device=x.device)); logits=self.head(self.norm(self.blocks(z)))
        loss=None if y is None else F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1))
        return logits,loss
    @torch.no_grad()
    def generate(self,x,n):
        for _ in range(n):
            logits,_=self(x[:,-self.T:]); p=F.softmax(logits[:,-1]/.8,dim=-1); x=torch.cat((x,torch.multinomial(p,1)),1)
        return x
def batch(dev,B=16,T=24):
    ix=torch.randint(len(raw)-T-1,(B,)); x=torch.stack([raw[i:i+T] for i in ix]).to(dev); y=torch.stack([raw[i+1:i+T+1] for i in ix]).to(dev); return x,y
def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=20); p.add_argument('--device',default='auto'); a=p.parse_args()
    dev=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device))
    m=GPT(len(chars)).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=3e-3); ex,ey=batch(dev); initial=m(ex,ey)[1].item()
    for _ in range(a.steps):
        x,y=batch(dev); _,loss=m(x,y); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
    final=m(ex,ey)[1].item(); assert final<initial,(initial,final)
    # 用单 batch 验证整个网络可拟合，常用于排除 mask/shift/反传接线错误
    ox,oy=batch(dev,B=2,T=12); over0=m(ox,oy)[1].item()
    # 继续使用同一个 optimizer，使其 Adam 矩始终与当前模型参数轨迹匹配。
    for group in opt.param_groups: group['lr']=5e-3
    for _ in range(40):
        _,l=m(ox,oy); opt.zero_grad(set_to_none=True); l.backward(); opt.step()
    over1=m(ox,oy)[1].item(); assert over1<over0*.7,(over0,over1)
    with tempfile.TemporaryDirectory() as d:
        path=os.path.join(d,'gpt.pt'); torch.save({'model':m.state_dict(),'opt':opt.state_dict(),'step':a.steps+40,
            'config':{'V':len(chars)},'cpu_rng':torch.random.get_rng_state()},path)
        m2=GPT(len(chars)).to(dev); opt2=torch.optim.AdamW(m2.parameters(),lr=1.)
        state=torch.load(path,map_location=dev,weights_only=True); m2.load_state_dict(state['model']); opt2.load_state_dict(state['opt'])
        # 真正走一次恢复后的训练步，验证 optimizer state 不只是“能反序列化”。
        _,resume_loss=m2(ox,oy); opt2.zero_grad(set_to_none=True); resume_loss.backward(); opt2.step()
        assert torch.isfinite(resume_loss) and state['step']==a.steps+40
        m2.eval()
        seed=torch.tensor([[stoi['t']]],device=dev); ids=m2.generate(seed,30)[0].tolist()
    print(f'PASS device={dev} loss={initial:.3f}->{final:.3f} overfit={over0:.3f}->{over1:.3f} sample={"".join(itos[i] for i in ids)!r}')
if __name__=='__main__': main()
```

checkpoint 应同时保存模型、与该模型当前参数轨迹匹配的优化器、step、随机数状态和配置；恢复训练还需恢复 scheduler。scheduler 决定当前及后续学习率，漏恢复会让下一步突然回到错误学习率；Python、CPU 和各 CUDA 设备的随机数状态决定 dropout、采样与数据打乱，漏恢复就不能从断点复现原训练轨迹。上例保存并加载模型与 optimizer 后实际继续一步，验证最小续训路径；它只保存 CPU RNG，完整分布式续训还应保存 Python、各 CUDA rank、采样器及 scheduler 状态。跨设备加载用 `map_location`。只推理时保存模型即可。生产代码还应区分 train/eval，后者关闭 dropout。

## 6. 卷末自测

1. Q、K、V 分别回答什么问题？注意力分数为何除以 `sqrt(d)`？
2. teacher forcing 与 causal mask 各解决什么问题？
3. 多头 Attention 中 `[B,T,C]` 如何变成 `[B,H,T,d]`，要求 C 满足什么条件？
4. pre-norm Block 的两条 residual 更新分别是什么？
5. 为什么断点续训除了模型权重，还要保存 optimizer、scheduler 和 RNG 状态？

## 7. 自测答案

1. Q 表示查询、K 表示可匹配特征、V 表示被聚合内容；缩放避免 d 较大时点积方差过大，使 softmax 过早饱和。
2. teacher forcing 用真实前缀构造并行训练目标；causal mask 从计算上禁止读取未来，两者不能互代。
3. 先令 `d=C/H`，reshape 后 transpose 得 `[B,H,T,d]`；C 必须能被 H 整除。
4. `x=x+attention(LN(x))`，再 `x=x+MLP(LN(x))`。
5. optimizer 保存动量等状态；scheduler 决定学习率进度；RNG 决定 dropout、采样和 shuffle。缺任何一项都不能等价续接。

# 第四卷：训练系统的算子、精度、显存与并行

## 1. backward 如何落到 GPU 算子

把上游梯度记作 `dY`：

- Linear `Y=XWᵀ+b`：`dX=dY W`、`dW=dYᵀX` 是 GEMM，`db=sum(dY)` 是 reduction。
- ReLU：`dX=dY*(X>0)`，逐元素 kernel，通常受内存带宽限制。
- LayerNorm：沿最后一维 reduction 求均值/方差；反传也需要多次 reduction 与逐元素归一化。融合 kernel 可少读写全局内存。
- softmax-cross-entropy：前向稳定 log-sum-exp；对 logits 的梯度是 `(p-one_hot(y))/N`。高效实现融合 max、sum、归一化和 NLL，避免落地完整中间概率。
- Attention：`S=QKᵀ/sqrt(d)`、`P=softmax(S+mask)`、`O=PV`。反传先 `dV=PᵀdO`、`dP=dO Vᵀ`，经过 softmax Jacobian 的向量积得到 `dS`，再 `dQ=dS K/sqrt(d)`、`dK=dSᵀQ/sqrt(d)`。主体是 batched GEMM，softmax/mask 是 reduction/逐元素。FlashAttention 通过分块在线 softmax 避免保存 `[T,T]`，不是改变数学结果。

因此 CUDA 视角可先问：它是 GEMM（算力密集）、reduction（同步/数值问题）、逐元素（带宽密集），还是通信。autograd 保存的张量正是 backward 公式所需输入；`detach` 或原地覆盖可能破坏这条链。

## 2. 每一字节去哪了

以参数数目 P 估算，普通 Adam 混合精度常见账本：FP16/BF16 参数 2P 字节；梯度 2P；FP32 master weight 4P；Adam 一阶矩 m 4P、二阶矩 v 4P，合计约 16P 字节。实现若同时保留 FP32 参数副本或梯度为 FP32，数字会变化，必须以实际框架为准。

### 2.1 用 10 亿参数实际算一遍

假设模型有 `P=1,000,000,000` 个参数，使用 BF16 参与 forward/backward，优化器状态为 FP32：

| 项目 | 每参数字节 | 10 亿参数十进制 GB | 是否随 batch/序列变化 |
|---|---:|---:|---|
| BF16 参数 | 2 | 2 GB | 否 |
| BF16 梯度（假设） | 2 | 2 GB | 否 |
| FP32 master weight | 4 | 4 GB | 否 |
| Adam 一阶矩 `m` | 4 | 4 GB | 否 |
| Adam 二阶矩 `v` | 4 | 4 GB | 否 |
| 合计 | 16 | 16 GB | 否 |

十进制 GB 与二进制 GiB 不同：`16,000,000,000 bytes ≈ 14.90 GiB`。硬件显存规格常写 GB，程序工具可能按 GiB 显示，做账时要写清口径。

这个 `16P` 只是一个教学配置，不是宇宙常数：

- 梯度可能是 FP32，于是再加 `2P` 字节；
- 某些训练方案没有独立 FP32 master weight；
- 8-bit optimizer 会压缩状态；
- fused optimizer 可能持有临时 buffer；
- 参数分片后，每卡只常驻部分状态，但计算前可能临时 all-gather。

因此面试时正确说法是“先列组成，再按实际 dtype/框架计算”，而不是死背“训练永远 16 bytes/parameter”。

### 2.2 Activation 为什么经常才是大头

参数状态主要随 `P` 变化；activation 通常随下面的量增长：

```text
micro_batch × sequence_length × hidden_size × layers
```

而朴素 Attention 若保存每层每头的概率，还会出现与 `sequence_length²` 有关的中间量。训练 backward 要用 forward 的部分结果，例如 Linear 输入、激活函数 mask、norm 统计量、Attention 的统计量。框架为了避免重算会保存它们。

要区分：

- **累计保存量**：所有层保存 activation 的总和；
- **峰值显存**：某一时刻同时存活的张量、workspace 和通信 buffer；
- **allocator reserved**：PyTorch 为以后复用而保留的内存。

三者不相等。仅把各 tensor 大小相加，不考虑生命周期，会高估或低估真实峰值。

### 2.3 四种省显存手段究竟省哪一栏

| 方法 | 主要减少 | 主要代价 | 不会自动减少 |
|---|---|---|---|
| gradient accumulation | 单次 micro-batch activation | 更多 forward/backward，step 变慢 | 参数、gradient、optimizer state |
| activation checkpointing | 保存的 activation | backward 重算 forward 片段 | 参数与 optimizer state |
| ZeRO/FSDP | 每卡参数/梯度/optimizer state（按阶段） | all-gather、reduce-scatter、调度复杂度 | 全局总参数量 |
| CPU/NVMe offload | GPU 常驻状态 | PCIe/NVMe 搬运延迟与带宽 | 系统总存储需求 |

gradient accumulation 的有效 global batch：

```text
global_batch = micro_batch_per_gpu × accumulation_steps × data_parallel_world_size
```

但它不完全等价于一次大 batch forward：BatchNorm 统计、dropout 随机性、数据顺序和浮点求和顺序可能不同。Transformer 常用 LayerNorm，因此 BatchNorm 这一差异常不是主问题，但概念上不能忽略。

此外还有 activation（随 `B*T*层数*隐藏维` 增长，Attention 朴素实现还随 `T²`）、临时 workspace（cuBLAS/cuDNN/编译器）、CUDA context、allocator 碎片、通信 bucket。`nvidia-smi` 是进程保留量，`torch.cuda.memory_allocated()` 是活张量，`memory_reserved()` 是缓存分配器持有量；三者不要混为一谈。

降低显存的旋钮：activation checkpointing 不保存部分激活、反向时重算，省显存但加计算；gradient accumulation 用多个 micro-batch 累加再 step，模拟大 batch，但吞吐和 BN 语义需留意；ZeRO/FSDP 分片 optimizer state、gradient、parameter（阶段越高分片越多）；CPU/NVMe offload 省 GPU 显存但付 PCIe/存储延迟。它们可以组合，但通信与重算可能让训练更慢。

## 3. 混合精度

FP16 指数范围窄，小梯度会下溢；loss scaling 先把 loss 乘 S，反传后梯度除 S。动态 scaler 发现 inf/nan 就跳过 step 并降低 S。BF16 与 FP32 指数位相同，范围大但尾数少，通常不需要 loss scaling，A100 原生支持 BF16/Tensor Core。权重更新常在 FP32 master weight 上做，避免小更新被低精度舍入。

推荐写法：

```python
use_amp = device.type == 'cuda'
amp_dtype = torch.bfloat16       # 改成 torch.float16 才启用 loss scaling
use_scaler = use_amp and amp_dtype == torch.float16
scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
    loss = model(x, y)[1]
scaler.scale(loss).backward()
scaler.unscale_(optimizer)       # clip 前必须 unscale
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer); scaler.update()
```

上例默认 BF16，因此 `use_scaler=False`；若明确把 `amp_dtype` 改成 FP16，才启用动态 loss scaling。某些 reduction、loss 和 optimizer 状态仍由框架保持 FP32。遇到 NaN 应先定位首个非有限值，而不是盲目改精度。

### 3.1 autocast 不是“把整个模型永久转成 BF16”

`torch.autocast` 在一个动态作用域里，根据算子策略选择输入/计算 dtype：适合低精度的 GEMM/卷积通常走低精度，部分对范围或精度敏感的运算会保留或提升精度。离开作用域后，模型参数本身不会因此永久改 dtype。

这与直接执行 `model.to(torch.bfloat16)` 不同。后者真的把参数存储改成 BF16，优化器与某些算子行为也会随之变化。

推荐把 loss 计算也放在 autocast 作用域内，让框架为相关算子选择策略；但 `backward()` 不需要包在 autocast 中，反向算子会沿用 forward 建立的 dtype 关系。

官方入口可查 [PyTorch AMP 文档](https://docs.pytorch.org/docs/stable/amp.html)。新代码优先使用 `torch.amp.GradScaler("cuda", ...)` 与 `torch.autocast(...)`，而不是旧的 `torch.cuda.amp.*` 别名。

### 3.2 FP16 与 BF16 的区别不是“一个快、一个准”

| 格式 | 指数范围 | 有效精度 | A100 Tensor Core | 典型训练风险 |
|---|---|---|---|---|
| FP32 | 大 | 高 | 可走 TF32 路径 | 显存和吞吐成本高 |
| FP16 | 小 | 比 BF16 更细 | 支持 | 小梯度下溢、大值上溢，需要 loss scaling |
| BF16 | 接近 FP32 | 比 FP16 粗 | 支持 | 舍入更粗，但范围通常更稳 |

BF16 通常不需要 loss scaling，是因为动态范围接近 FP32，不是因为它“不会出现 NaN”。学习率过大、错误除零、错误 mask、梯度爆炸仍能产生非有限值。

### 3.3 loss scaling 的顺序

FP16 训练的核心顺序：

```text
原 loss
→ 乘 scale，放大小梯度
→ backward
→ 检查 inf/nan
→ 梯度除 scale（unscale）
→ gradient clipping
→ optimizer step
→ 动态调整 scale
```

gradient clipping 必须在 unscale 之后，否则裁剪的是被人为放大的梯度，阈值没有原本含义。

## 4. 性能指标与剖析

step time 必须先 warmup，再用 CUDA event 或在墙钟两侧 `torch.cuda.synchronize()`；CUDA 默认异步。`tokens/s = global_batch * sequence_length / step_time`，若梯度累积还要乘 accumulation steps。MFU（Model FLOPs Utilization）=`模型每步理论 FLOPs / (step_time * GPU 峰值 FLOP/s * GPU数)`；它依赖 FLOPs 口径与所用精度峰值，应注明假设，不能把它当硬件利用率的绝对真相。

工具分层：PyTorch profiler 找到昂贵 op、shape、CPU/GPU 时间和内存；Nsight Systems (`nsys`) 看 CPU launch、GPU timeline、NCCL overlap 和空洞；Nsight Compute (`ncu`) 深挖单个 kernel 的 occupancy、Tensor Core、访存吞吐和 stall。先宏观后微观，避免一开始就对错误 kernel 做精细优化。

**【演示】完整程序 `profile_training.py`**：先无 trace 建立基线，再开启 trace 比较，不要把 profiler 开销当模型耗时。

```python profile_training.py
import argparse, time, torch
import torch.nn as nn

def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=8); p.add_argument('--device',default='auto'); p.add_argument('--trace',default='')
    a=p.parse_args(); dev=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device))
    torch.manual_seed(3); B,T,C,V=8,64,128,512
    model=nn.Sequential(nn.Embedding(V,C),nn.Linear(C,4*C),nn.GELU(),nn.Linear(4*C,V)).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3); x=torch.randint(V,(B,T),device=dev); y=torch.randint(V,(B,T),device=dev)
    # 把 CUDA lazy loading 与 optimizer state 初始化移出正式采集区间。
    for _ in range(3):
        warm_logits=model(x); warm_loss=torch.nn.functional.cross_entropy(warm_logits.reshape(-1,V),y.reshape(-1))
        opt.zero_grad(set_to_none=True); warm_loss.backward(); opt.step()
    if dev.type=='cuda': torch.cuda.synchronize()
    activities=[torch.profiler.ProfilerActivity.CPU]+([torch.profiler.ProfilerActivity.CUDA] if dev.type=='cuda' else [])
    prof=torch.profiler.profile(activities=activities,record_shapes=True,
                                profile_memory=True) if a.trace else None
    if prof: prof.__enter__()
    if dev.type=='cuda': torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(a.steps):
        logits=model(x); loss=torch.nn.functional.cross_entropy(logits.reshape(-1,V),y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if dev.type=='cuda': torch.cuda.synchronize()
    sec=time.perf_counter()-t0
    if prof:
        prof.__exit__(None,None,None)
        print(prof.key_averages().table(
            sort_by='self_cuda_time_total' if dev.type=='cuda' else 'self_cpu_time_total',
            row_limit=12))
        prof.export_chrome_trace(a.trace)
    mode='PROFILED' if a.trace else 'BASELINE'
    note=' includes_profiler_overhead=true' if a.trace else ''
    assert torch.isfinite(loss); print(f'PASS mode={mode} device={dev} steps={a.steps} step_ms={sec/a.steps*1e3:.2f} tokens_s={B*T*a.steps/sec:.0f} loss={loss.item():.3f}{note}')
if __name__=='__main__': main()
```

示例命令：`python profile_training.py --trace trace.json`；系统时间线可用 `nsys profile -o train python profile_training.py --device cuda`；kernel 指标可用 `ncu --set full python profile_training.py --device cuda --steps 1`。后两者开销很大，只跑少量 step。

本例在正式采集前做了 3 次 warmup，把 CUDA lazy loading 和 AdamW state 初始化移出采集区间。否则 profiler 可能把首次初始化成本归到某个 operator，得到“某项 CUDA 百分比超过合理范围”的误导表格。

PyTorch 2.11 在这种手工 profile cycle 用法下可能提示事件 cycle 会被清理；本例只查看当前采集区间并同时导出 trace。若设计带 wait/warmup/active/repeat 的长期 schedule，应按 [torch.profiler 官方文档](https://docs.pytorch.org/docs/stable/profiler) 配置 `schedule`、`on_trace_ready` 和是否 `acc_events`，不要为了消除警告盲目累计不同 cycle。

### 4.1 三种工具怎样接力

假设训练很慢，不要同时打开所有重型工具。推荐：

```text
先用无 profiler benchmark 确认 step time
→ PyTorch profiler 找慢 operator 和 shape
→ nsys 看 CPU launch、DataLoader、kernel 空洞、NCCL overlap
→ 只对关键 kernel 用 ncu
```

| 问题 | 首选工具 | 你要找的证据 |
|---|---|---|
| DataLoader 是否喂不饱 | nsys / CPU profiler | GPU 时间线空洞、CPU 取数过慢 |
| 哪个 PyTorch op 最贵 | torch.profiler | op self/total CPU/CUDA time、input shape |
| 通信是否与 backward 重叠 | nsys | NCCL 与计算 kernel 时间线重叠程度 |
| GEMM 是否用 Tensor Core | ncu / SASS | Tensor pipe、HMMA/MMA 指令 |
| kernel 为何 stall | ncu | DRAM/shared/occupancy/stall reason |
| 是否 OOM/碎片 | memory stats/snapshot | allocated、reserved、峰值、生命周期 |

### 4.2 tokens/s 与 MFU 回答不同问题

- `tokens/s`：业务吞吐，受模型、序列、batch、数据和硬件共同影响；
- MFU：把估算的模型 FLOP 除以硬件理论峰值与时间，试图衡量算力利用；
- kernel GFLOPS：某一个 kernel 的局部计算吞吐。

它们不能互相替代。一个 fused kernel 变快，不保证 step time 等比例下降；tokens/s 高也可能只是模型更小；MFU 计算若使用错误的 FLOP 公式或 FP32 峰值去除 BF16 Tensor Core 工作，结果没有意义。

## 5. 并行训练地图

先统一四个首次出现的术语。**rank** 是一个分布式进程在通信组内的编号，通常一张 GPU 对应一个进程；**world size** 是组内进程总数。**collective（集合通信）**是组内所有 rank 按约定共同参加的操作，例如 broadcast、all-gather、reduce-scatter。**all-reduce** 先把各 rank 的张量按元素求和或取最大值等归约，再把相同结果发回每个 rank；DDP 通常用它汇总梯度。若任一 rank 没有按相同顺序进入 collective，其他 rank 可能一直等待。

| 方式 | 主要切分 | 每卡是否有完整参数 | 典型通信 | 主要解决 | 主要代价 |
|---|---|---|---|---|---|
| DDP / Data Parallel | batch | 是 | gradient all-reduce | 扩吞吐 | 模型和状态必须单卡容纳 |
| Tensor Parallel | 单层 hidden/head | 否 | all-reduce/all-gather/reduce-scatter | 单层太宽、分计算 | 层内频繁通信，依赖高速互联 |
| Pipeline Parallel | layers | 否 | stage 间 send/recv activation/gradient | 模型太深 | bubble、micro-batch 调度 |
| Context/Sequence Parallel | sequence | 部分 | all-gather、reduce-scatter、all-to-all 或 ring | 长序列 activation/Attention | 通信模式复杂 |
| ZeRO/FSDP | DP 状态 | 否或计算时临时聚合 | parameter all-gather、gradient reduce-scatter | 参数/梯度/optimizer 显存 | 额外通信和预取调度 |

### 5.1 DDP：先理解 replicated model

DDP 中每个 rank 有完整模型，但读取不同 mini-batch。每卡独立 forward/backward；当某个 gradient bucket 准备好时，DDP 注册的 hook 发起 all-reduce，使所有 rank 最终得到一致梯度，再各自执行相同 optimizer step。

梯度通信可与更前面层的 backward 重叠，所以 bucket 大小、参数使用顺序和 unused parameter 会影响效率。DDP 解决吞吐扩展，不解决单卡放不下完整模型/optimizer state。官方 API 见 [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。

### 5.2 Tensor Parallel：切一层矩阵

以 `Y=XW` 为例，W 可以按输出列切：各卡计算部分 Y，下一算子若能继续消费分片就暂不聚合；也可以按输入行切，各卡产生部分和，再 all-reduce。实际 Transformer 会配对 column-parallel 与 row-parallel Linear，减少不必要通信。

TP 每层都会通信，因此更适合节点内 NVLink/NVSwitch。跨慢网络盲目加 TP，通信可能盖过 GEMM 收益。

### 5.3 Pipeline Parallel：切层并用 micro-batch 填流水

若 GPU0 放前几层、GPU1 放后几层，一个完整 batch 直接流过会让多数 stage 等待。把 batch 切成多个 micro-batch，可以让不同 stage 同时处理不同 micro-batch。

但流水开始和结束仍有空洞（bubble）。micro-batch 越多通常越能摊薄 bubble，却会改变 activation 生命周期、调度开销和有效 batch 设计。

### 5.4 FSDP / ZeRO：切数据并行状态

概念上的阶段：

```text
ZeRO-1：分片 optimizer state
ZeRO-2：再分片 gradient
ZeRO-3：再分片 parameter
```

FSDP FULL_SHARD 与 ZeRO-3 思路接近：模块计算前 all-gather 参数，计算后重新分片；gradient 用 reduce-scatter，optimizer 在本地 shard 上更新。它降低每卡常驻状态，却会产生参数通信和临时峰值。

wrap 粒度太大，all-gather 峰值高且 overlap 困难；太小，collective 太碎。prefetch 可以提升 overlap，但会增加同时存活参数，形成吞吐与峰值的权衡。以当前 [PyTorch FSDP 文档](https://docs.pytorch.org/docs/stable/fsdp.html) 为准，API 和推荐路径会随版本演进。

### 5.5 组合并行不是越多越好

实际常做 3D/4D 混合：节点内 TP，跨节点 DP/FSDP，层数再 PP，极长序列加 context parallel。选择顺序通常是：模型能否单卡放下、互联拓扑、目标 global batch、序列长度，再测吞吐而非凭名称决定。

一个实用决策顺序：

```text
模型和 optimizer 能否单卡放下？
├─ 能：先 DDP 扩吞吐
└─ 不能：状态过大先考虑 FSDP/ZeRO
         单层过宽考虑 TP
         层数过深考虑 PP
         序列 activation 过大考虑 context/sequence parallel

最后根据节点内/节点间带宽组合，并用 timeline 验证通信重叠。
```

## 6. 面试题与参考答案

**问：为什么 softmax 要减最大值？** 答：平移不改变概率，却让最大指数为 1，避免 `exp(大数)` 溢出。

**问：cross entropy 对 logits 的梯度？** 答：均值口径下为 `(softmax(z)-one_hot(y))/N`。

**问：teacher forcing 与 causal mask 各解决什么？** 答：前者让训练使用真实前缀并行预测，后者从计算图上禁止位置访问未来；二者不可互相替代。

**问：为什么 gradient accumulation 不能省参数/optimizer 显存？** 答：它只减 micro-batch 激活，参数、梯度槽和 Adam 状态仍在。

**问：checkpointing 的代价？** 答：反向重算前向片段，减少保存激活，增加计算和可能的 RNG/编译复杂度。

**问：BF16 为什么常比 FP16 稳？** 答：它与 FP32 具有相同指数位宽，动态范围大；代价是有效尾数较少。

**问：DDP backward 为什么可能很快？** 答：梯度 bucket 可在后续层仍反传时异步 all-reduce，实现通信计算重叠；bucket 大小和层顺序会影响效果。

**问：allocated、reserved、nvidia-smi 为什么不同？** 答：分别近似活张量、PyTorch allocator 缓存、进程/CUDA 上下文总体占用。

**问：tokens/s 高是否代表 MFU 高？** 答：未必；token 长度、模型大小和 FLOPs 口径不同。tokens/s 是业务吞吐，MFU 是对理论算力的归一估算。

**问：OOM 首先怎么查？** 答：确认是参数、optimizer、activation、workspace 还是碎片；记录 peak allocated/reserved，逐步改变 batch/sequence，必要时用 memory snapshot，再选择 checkpoint/FSDP/offload，而非盲调 allocator。

**问：单 batch 过拟合为何重要？** 答：极小数据仍降不下去，通常说明 shift、mask、loss、梯度、optimizer 或数据管线有错误；它是训练系统的“连通性测试”。

## 7. 卷末自测

1. **【必须手写】** 对 10 亿参数模型列一张显存账本，至少包含参数、梯度、FP32 master weight 和 Adam 两个矩。
   - **【提示 1】** 先写每项 dtype 对应的每参数字节数。
   - **【提示 2】** BF16 是 2 字节，FP32 是 4 字节。
   - **【提示 3】** 最后分别写十进制 GB 与 GiB，别混用单位。
2. FP16 loss scaling 中，为什么 clipping 必须发生在 unscale 之后？
3. rank、world size、collective、all-reduce 分别是什么？
4. DDP 为什么能隐藏一部分通信，又为什么不能解决完整模型单卡放不下？
5. profiler、nsys、ncu 应按什么顺序使用，各回答哪一层问题？

## 8. 自测答案

1. **【参考实现】** 教学假设下 BF16 参数 2 GB、BF16 梯度 2 GB、FP32 master weight 4 GB、Adam m/v 各 4 GB，共 16 GB，约 14.90 GiB；实际实现须按真实 dtype 和状态重算。
2. scale 后的梯度被人为放大；若先裁剪，阈值对应错误尺度，破坏预期更新。
3. rank 是组内进程编号，world size 是进程总数，collective 是全组共同参加的通信，all-reduce 是归约后把结果返回所有 rank。
4. 梯度 bucket 一准备好即可与后续 backward 重叠 all-reduce；但每个 DDP rank 仍持有完整参数和优化器状态。
5. 先用 profiler 找慢 op/shape，再用 nsys 看整机时间线与通信重叠，最后只对关键 kernel 用 ncu 看硬件指标。

**【挑战】** 修改 profiling 示例，将 batch 或隐藏维翻倍；先预测 step time、tokens/s 和峰值显存怎样变化，再用测量解释偏差。

<!--
验证环境：/home/qichengjie/workspace/vllm_demo/.venv/bin/python；PyTorch 2.11.0+cu130；NVIDIA A100 80GB PCIe，compute capability 8.0。
验证方式：从本文 Markdown 围栏逐段抽取并以原样代码执行。
CPU：mini_lm --steps 30 PASS，同一固定评估 batch loss 3.231→0.661，单 batch 0.666→0.367；mini_gpt --steps 20 PASS，固定评估 loss 31.191→2.706，单 batch 3.563→0.000；profile_training --steps 3 BASELINE PASS，约 144,412 tokens/s，PROFILED 输出明确标记含插桩开销。
CUDA：mini_lm --steps 10 PASS，同一固定评估 batch loss 3.231→1.911，单 batch 1.945→0.332；mini_gpt --steps 3 PASS，固定评估 loss 31.191→18.799，单 batch 16.980→0.000；profile_training --steps 3 BASELINE PASS，约 490,633 tokens/s，PROFILED 输出明确标记含插桩开销。
mini_lm 的权重 reload 与生成路径由 PASS 前断言及随后生成覆盖；mini_gpt 还加载与当前模型匹配的 optimizer state，并实际完成一个恢复训练步。示例不是逐 bit 精确续训测试；后者还需恢复 scheduler、所有 rank RNG 与数据采样位置。
-->

---



---

# 全书完成清单

- [ ] 能手算一次 scalar forward、loss、gradient 和 SGD 更新。
- [ ] 能解释导数、偏导、gradient、链式法则和 backprop 的关系。
- [ ] 能推导 Linear 的 dX、dW、db shape。
- [ ] 能用 autograd，并理解梯度累积、no_grad、detach。
- [ ] 能独立完成 PyTorch 分类器并保存/恢复 checkpoint。
- [ ] 能解释 logits、cross entropy、perplexity 和 teacher forcing。
- [ ] 能运行迷你语言模型和迷你 GPT，并完成单 batch 过拟合测试。
- [ ] 能分项估算参数、梯度、Adam 状态和 activation 显存。
- [ ] 能说明 FP16 loss scaling、BF16、autocast 的作用和边界。
- [ ] 能区分 PyTorch profiler、nsys、ncu 的职责。
- [ ] 能说明 DDP、TP、PP、Context Parallel、ZeRO/FSDP 各切分什么。

# 延伸阅读

1. [PyTorch Autograd 官方教程](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
2. [PyTorch nn.Module 官方文档](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
3. [PyTorch AMP 官方文档](https://pytorch.org/docs/stable/amp.html)
4. [DistributedDataParallel 官方文档](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
5. [FullyShardedDataParallel 官方文档](https://pytorch.org/docs/stable/fsdp.html)
6. [Week 4 Attention 与 FlashAttention 教材](../attention/Week4_Attention与FlashAttention完整学习资料.md)
