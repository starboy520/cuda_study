# Week5 Transformer Block 小白向扩写设计

## 目标

扩写 `docs/Week5增强版_LLM推理优化与decode.md` 的 2.4 节，使没有机器学习基础、但具备少量 CUDA 基础的读者能够理解一个 Transformer block 内部的数据流，并为后续 GEMV、融合算子和 KV Cache 学习建立上下文。

## 范围

只修改 2.4 `Transformer block`，保留 Day 1 的入门节奏。新增内容包括：

1. 先解释 block 的输入和输出为什么保持 `[B,N,D]`；
2. 逐步解释 RMSNorm、Attention、残差连接、第二次 RMSNorm、MLP 和第二次残差；
3. 使用 `B=1,N=3,D=4` 的微型 shape 例子贯穿数据流；
4. 对比 prefill 的 `[B,N,D]` 与 decode 的 `[B,1,D]`；
5. 标出 K/V 在 Attention 内生成、写入和读取的位置；
6. 解释残差为何要求主分支最终回到 `D`；
7. 解释 MLP 的“升维—非线性—降维”直觉；
8. 加入一份小白常见误区和一段闭卷口述。

## 深度边界

- 不在本节重新推导完整 Softmax Attention；
- 不展开多头拆分、RoPE、GQA 的数学细节；
- 不讲训练、反向传播或梯度；
- 不加入需要编译的新代码；
- 对 Attention 细节链接到现有 Week4 学习资料；
- 对 CUDA 映射只做预告：线性层对应 GEMM/GEMV，Norm/Residual/Activation 对应后续融合 kernel。

## 结构

2.4 节将按以下顺序组织：

1. 一句话直觉；
2. 完整 block 数据流图；
3. 六个步骤逐项解释；
4. 微型 shape 跟踪表；
5. prefill/decode 对比；
6. 与本周 CUDA 任务的对应关系；
7. 常见误区与口述验收。

## 验收标准

- 读者能说清 Attention 分支与 MLP 分支各自做什么；
- 能解释两次 residual add 的两个输入来自哪里；
- 能说明为什么 block 输入输出 shape 相同；
- 能指出 KV Cache 只属于 Attention 分支；
- 能从 `N=3` 切换到 `N=1` 理解 decode shape；
- 原有 Day 1 后续章节编号和主线保持不变。
