# Week 6 每日详细安排(Day1–Day7)

> **主题**:核心算子开发——softmax、layernorm、kernel 融合(算子开发岗的核心竞争力)
> **为什么这周**:前五周你练的是通用 kernel 和 GEMM。Week6 进入"真实算子",学数值稳定、
> 多级归约、kernel 融合——这是大模型/推理算子工程师每天做的事。
> **硬件**:Tesla T4(sm_75)。
> **前置**:Week5 完成(会用 ncu 判读、量化优化)。
> **本周交付**:`week06_operators/` 下 softmax + layernorm 的数值稳定融合实现 +
> 和 PyTorch/CPU 对比 + `notes/week06.md`。

---

## 使用方式

- 按 Day1→Day7。每个算子都要:先正确(对比 reference)→ 再数值稳定 → 再融合优化。
- 每天在 `notes/week06.md` 记:目标 / 实验结果 / 遇到的问题。
- 阅读标注:📖 精读 · 👀 扫读 · ✍️ 必须动手 · ⏭️ 跳过。

---

## 本周总览

| Day | 主题 | 动手产出 | 关键概念 |
|---|---|---|---|
| 1 | 算子开发全景 + 复习归约 | warp/block 归约工具函数 | 多级归约 / 库的边界 |
| 2 | Softmax 数值稳定 | `softmax` 朴素 + 稳定版 | 减最大值防溢出 |
| 3 | Softmax 优化 | 一个 block 一行 + shuffle 收尾 | grid-stride / 融合 |
| 4 | LayerNorm | `layernorm` 实现 | mean/var 归约 / 融合 |
| 5 | RMSNorm + kernel 融合 | `rmsnorm` + 融合思维 | 减少 global 往返 |
| 6 | 和库/框架对比 | 对比 cuDNN/PyTorch | 何时手写 vs 调库 |
| 7 | 复盘 + 算子报告 | 算子性能 + 数值报告 | 数值稳定 + 性能 |

---

## 本周用得上的现成资料

| 主题 | 教材正文 |
|---|---|
| GEMM/算子数学 | `cuda_deep_course/course/volume06_operators/01_GEMM数学_布局与性能上限.md` |
| Softmax 数值稳定 | `.../volume06_operators/04_Softmax数值稳定与多级归约.md` |
| LayerNorm/RMSNorm | `.../volume06_operators/05_LayerNorm_RMSNorm与融合.md` |
| 库的边界 | `.../volume06_operators/06_库的边界与算子工程化.md` |
| warp 归约 | `.../volume04_parallel_algorithms/03_Reduction从错误到优化.md` |
| 浮点数值稳定 | `.../volume08_hpc_multigpu/01_浮点_FMA与误差传播.md` |

配套 lab:`cuda_deep_course/labs/06_operators/`(softmax、gemm_tiled)。

---

## Day 1:算子开发全景 + 复习归约

**学什么**
- 算子开发的核心:正确性(对比 reference)+ 数值稳定 + 性能(融合/复用)。
- 复习多级归约:warp shuffle 归约 + block 归约,这是 softmax/layernorm 的基础工具。
- 库的边界:标准算子用 cuBLAS/cuDNN,融合/特殊算子才手写。

**看什么** 📖 `volume06/06`(库边界)、`volume04/03` §5(warp shuffle)。

**动手** ✍️ 写两个可复用的归约工具函数
```cpp
__device__ float warpReduceSum(float v);   // warp 内归约(5 次 shfl_down)
__device__ float blockReduceSum(float v);  // block 内归约(warp 归约 + shared + 再归约)
```

**完成标准**
- [ ] warpReduceSum / blockReduceSum 正确(小数组验证)
- [ ] 能解释 block 归约为什么是"warp 归约 → shared → 再 warp 归约"两级

---

## Day 2:Softmax 数值稳定

**学什么**
- softmax 定义:`exp(x_i) / Σexp(x_j)`。
- 朴素实现的问题:x 较大时 `exp(x)` 溢出(inf)。
- 数值稳定:先减每行最大值 `exp(x_i - max) / Σexp(x_j - max)`(结果不变,不溢出)。

**看什么** 📖 `volume06/04`(数值稳定推导)、`volume08/01`(浮点溢出)。

**动手** ✍️ `week06_operators/softmax/softmax.cu`
1. 朴素版(不减最大值),用大值输入触发 inf,观察出错。
2. 稳定版(减最大值),验证不溢出。
3. 和 CPU reference 按容差对比。

**完成标准**
- [ ] 能复现朴素版的溢出
- [ ] 稳定版结果正确(含大值输入)
- [ ] 能解释为什么减最大值后结果不变

---

## Day 3:Softmax 优化

**学什么**
- 一个 block 处理一行,grid-stride 处理超长行。
- 两遍归约:第一遍求 max,第二遍求 Σexp,用 block 归约。
- 融合:max/sub/exp/sum/div 融成一个 kernel,数据只读写 global 一遍。

**看什么** 📖 `volume06/04`(多级归约 + 融合)。

**动手** ✍️
1. 用 Day1 的 blockReduce 重写 softmax(max 归约 + sum 归约 + warp shuffle 收尾)。
2. 确保是一个 kernel 完成全部(融合),不是三个 kernel。
3. 测性能,和 Day2 版本对比。

**完成标准**
- [ ] 融合版正确且更快
- [ ] 能说出融合省了什么(global 往返次数)

---

## Day 4:LayerNorm

**学什么**
- layernorm:`(x - mean) / sqrt(var + eps) * gamma + beta`,对每行归一化。
- 要算两个归约:mean(Σx/n)和 var(Σ(x-mean)²/n)。
- 数值:var 用稳定公式,避免灾难性消减(卷八/01)。

**看什么** 📖 `volume06/05`(layernorm 实现)。

**动手** ✍️ `week06_operators/layernorm/layernorm.cu`
1. 一个 block 一行,用 block 归约算 mean 和 var。
2. 应用 gamma/beta(可学习参数)。
3. 和 PyTorch/CPU reference 对比。

**完成标准**
- [ ] layernorm 正确(对比 reference)
- [ ] 能解释为什么 var 要用数值稳定的算法

---

## Day 5:RMSNorm + kernel 融合

**学什么**
- RMSNorm:layernorm 的简化(不减 mean,只用 RMS),大模型常用。
- kernel 融合思维:把 elementwise + 归约 + 缩放融成一个 kernel。
- 融合的收益:每个独立 kernel 都要读写一遍 global,融合后只读写一次。

**看什么** 📖 `volume06/05`(RMSNorm + 融合思维)。

**动手** ✍️
1. 实现 RMSNorm(`x / sqrt(mean(x²) + eps) * gamma`)。
2. 思考:你的 layernorm 里有几次 global 读写?能不能减少?
3. (可选)实现一个 fused bias+relu 或类似的小融合算子。

**完成标准**
- [ ] RMSNorm 正确
- [ ] 能解释 kernel 融合为什么减少访存

---

## Day 6:和库/框架对比

**学什么**
- cuDNN 有 softmax/layernorm 等算子,工业级优化。
- 何时手写:库没有的、需要和前后算子融合的、特殊形状/精度。
- 你的手写版到了库的百分之几?差距在哪?

**看什么** 📖 `volume06/06`(库的边界)。

**动手** ✍️
1. (有 PyTorch 的话)用 `torch.nn.functional.softmax/layer_norm` 跑同样数据,对比你的结果和性能。
2. 分析差距来源(向量化?更好的归约?Tensor Core?)。
3. 总结"什么时候该手写算子"。

**完成标准**
- [ ] 手写版和库结果一致(容差内)
- [ ] 知道自己到了库的百分之几,差距在哪
- [ ] 能回答"为什么大模型要手写融合算子"

---

## Day 7:复盘 + 算子报告

**动手**
1. 整理 `notes/week06.md`:
   - softmax/layernorm/rmsnorm 的正确性(对比 reference)+ 性能。
   - 每个算子的"数值稳定怎么做的、融合省了什么"。
   - 和库的对比。
2. 写一段 3 分钟口述:讲一个你实现的算子,数值稳定怎么处理、怎么优化、和库差多少。

**本周自测**
- [ ] softmax 为什么要减最大值?
- [ ] layernorm/rmsnorm 的区别?
- [ ] kernel 融合为什么能加速?
- [ ] block 归约怎么实现(两级)?
- [ ] 什么时候手写算子、什么时候调库?

---

## 本周交付清单

```text
week06_operators/
├── softmax/softmax.cu        (数值稳定 + 融合)
├── layernorm/layernorm.cu
└── rmsnorm/rmsnorm.cu

notes/week06.md  —— 三个算子的正确性 + 性能 + 数值稳定说明 + 库对比 + 讲解稿
```

> 这周的产出很适合作为 Week7 作品集的素材——一个数值稳定的融合算子是算子岗很有说服力的项目。

---

**返回**:[study_plan/README.md](README.md) · 上一步:[Week5 性能工程](Week5_性能工程与Nsight实战.md) · 下一步:[Week7 作品集](Week7_作品集项目.md)
