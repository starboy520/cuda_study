# FlashAttention 教学章节重写设计

## 目标

重写 `docs/Week4_Attention与FlashAttention完整学习资料.md` 的 Day 3，并调整 Day 4 的入口，使只掌握少量 CUDA、刚接触 Attention 的读者能够从普通 Attention 自然过渡到 FlashAttention 的状态更新和 A100 实现。

## 读者起点

- 知道 CUDA 中 global memory、shared memory、thread block 的基础概念；
- 刚刚学习 Q、K、V、softmax 和普通 Attention；
- 已阅读前一日的 stable softmax 与 online softmax，但尚不能把 `(m,l)` 自然推广到 Attention；
- 不预设读者理解 FlashAttention 论文中的矩阵符号和 IO complexity。

## 教学主线

章节围绕一句话展开：

> FlashAttention 不改变 Attention 的数学结果；它不保存完整的注意力矩阵，而是分块读取 K/V，并在读取过程中累计同一个加权平均。

内容按以下顺序组织：

1. 用一个 query、四个 key/value 完整计算普通 Attention；
2. 把四个 key/value 切成两个 tile，演示“每块各做 softmax 再相加”为何错误；
3. 暂时不考虑数值稳定，先引出分母累计量和向量分子累计量；
4. 加入 running max，逐一解释 `m`、`l`、`O_acc` 的含义和单位；
5. 用同一组数字逐 tile 更新，最终与普通 Attention 的结果对齐；
6. 把数学变量映射到 A100 的 HBM、shared memory 和 registers；
7. 对比朴素实现与 FlashAttention 的数据搬运，解释为何 FLOP 仍为平方级却可能更快；
8. 最后给出矩阵形式、forward 伪代码、causal 边界和常见误解。

## 表达策略

- 每次只引入一个新概念，先说用途，再给符号，最后给公式；
- 区分标量、向量和矩阵，并在首次出现时标注 shape；
- 把 `l` 称作“当前指数权重之和（分母）”，把 `O_acc` 称作“使用同一批指数权重形成的向量分子”；
- 使用“换计量基准”解释 `alpha=exp(m_old-m_new)`，避免只要求记公式；
- 保留专业术语，并在首次出现时给中文直觉解释；
- 明确教学实现与工业 FlashAttention 的边界，不暗示单行标量版本就是高性能实现。

## 修改范围

只修改以下内容：

- 重写 Day 3；
- 调整 Day 4 开头，使 CUDA 状态机显式回扣 Day 3 的变量；
- 必要时调整目录、章节编号和内部引用。

不修改：

- Day 1 的普通 Attention 实现；
- Day 2 的 online softmax 代码；
- 后续 MLA、性能分析与练习代码主体；
- 用户已有的 `week04_attention` 源码和其他工作区文件。

## 验收标准

- 在不知道 FlashAttention 的前提下，只读 Day 3 能解释 `m/l/O_acc`；
- 数字示例同时给出朴素结果和分块结果，二者在合理精度下相等；
- 能明确回答 FlashAttention 省掉了哪些 HBM 中间读写；
- 不把 exact、FLOP 复杂度、IO 复杂度混为一谈；
- Day 4 的每个 CUDA 状态都能对应到 Day 3 的数学变量；
- Markdown 代码围栏、标题层级、内部链接及数学定界符保持有效。
