# ML 零基础记忆卡

> 用途：面试或学习前 5 分钟快速回忆；只记主线，推导、代码和系统细节统一回到同目录主教材。

## 1. 训练四步：forward → loss → backward → update

**一句话记忆：** 模型先用参数做预测，loss 衡量误差，backward 算每个参数该往哪边改，optimizer 最后更新参数。

**最小例子：** $\hat y=wx$，取 $w=1,x=2,y=5$，则 forward 得 $\hat y=2$，$L=(\hat y-y)^2=9$，backward 得 $\frac{\partial L}{\partial w}=2(wx-y)x=-12$，若 $lr=0.1$，SGD 更新为 $w\leftarrow1-0.1\times(-12)=2.2$。

**深入：** [训练模型、链式法则与一次 SGD 更新](./ML基础_训练侧入门.md#2-模型一个带参数的函数)

## 2. 显存四栏：参数 / 梯度 / 激活 / optimizer state

**一句话记忆：** 参数是模型本体，梯度告诉参数怎么改，激活是 forward 留给 backward 的中间结果，optimizer state 是动量等更新历史。

**最小例子：** 100 万个 FP32 参数约占 4 MB；参数 4 MB + 梯度 4 MB + Adam 的一阶/二阶状态 8 MB，合计约 16 MB，尚未计入随 batch、序列长度和层数增长的激活，也未计混合精度 master weights 等额外副本。

**深入：** [训练显存账本与 Activation](./ML基础_训练侧入门.md#2-每一字节去哪了)

## 3. FP16 / BF16：一个精度高，一个范围大

**一句话记忆：** FP16 尾数更多、1 附近更精细但范围小；BF16 指数与 FP32 相同、范围大但有效精度更低。

**最小例子：** 不计 subnormal，FP16 最大有限值为 $65504$、最小正规数约 $6.10\times10^{-5}$、1 附近间隔为 $2^{-10}\approx9.77\times10^{-4}$；BF16 最大值约 $3.39\times10^{38}$、最小正规数约 $1.18\times10^{-38}$、1 附近间隔为 $2^{-7}=7.8125\times10^{-3}$。因此 FP16 训练常配 loss scaling，BF16 通常更不易 overflow/underflow。

**深入：** [混合精度、FP16、BF16 与 loss scaling](./ML基础_训练侧入门.md#3-混合精度)

## 4. batch / epoch / learning rate

**一句话记忆：** batch 是一次拿多少样本，epoch 是全数据看一遍，learning rate 决定每次参数更新迈多大步。

**最小例子：** 1000 个样本、`batch_size=100`，每个 epoch 有 10 个 step；训练 3 个 epoch 共约 30 次更新，若 `lr=1e-3`，SGD 每次执行 $\theta\leftarrow\theta-10^{-3}g$。

**深入：** [训练循环中的 batch、step、epoch 与 learning rate](./ML基础_训练侧入门.md#7-训练循环中的六个计量单位)

## 5. 过拟合与验证集

**一句话记忆：** 训练集用于更新参数，验证集只用于检查泛化；train loss 继续下降而 val loss 回升，是典型过拟合信号。

**最小例子：** 第 5 个 epoch 的 `train_loss=0.20, val_loss=0.24`，第 20 个 epoch 变成 `train_loss=0.03, val_loss=0.40`；模型更会背训练数据，却在未参与更新的验证数据上变差，可考虑 early stopping、正则化、数据增强或减小模型。

**深入：** [训练、验证、保存与重载的完整分类器](./ML基础_训练侧入门.md#8-完整分类器训练验证保存与重载)

## 6. 训练侧 vs 推理侧的 CUDA 关注点

**一句话记忆：** 训练要为 backward 和 optimizer 付出算力与显存，推理只做 forward，但 prefill 重吞吐、decode 重单步延迟与内存带宽。

**最小例子：** 同一个 Linear，训练包含 forward GEMM、反向的 $dX/dW$ GEMM、梯度保存和参数更新；推理没有参数梯度与 optimizer state，但 LLM decode 每步只生成一个 token，要反复读取权重和 KV cache，常比大 batch 训练更偏 memory-bound。

**深入：** [backward 的 GPU 算子与训练系统视角](./ML基础_训练侧入门.md#1-backward-如何落到-gpu-算子)
