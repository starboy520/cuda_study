# Week 5 增强版：LLM 推理优化（decode 主场 + 8 个补充点全嵌入）

> 对应：[DeepSeek CUDA 2 月冲刺计划](DeepSeek_CUDA_2月冲刺计划.md) Week 5（LLM 推理优化）。
> 增强点：把原 Week5（KV cache / paged attention / 量化 / GEMV）与之前列的 8 个 CUDA 盲点
> **合并成一条主线**，绝大多数补充点作为 decode 优化的"副产品"顺手拿下，不额外开天。
> 硬件：A100 80GB PCIe（`sm_80`，108 SM，FP32 19.5 TFLOPS，HBM 带宽约 1935 GB/s）。
> 前置：Week4 Attention / FlashAttention 已完成（online softmax、tiled attention、KV cache 概念）。
> 本周核心认知：**decode 阶段是 memory-bound 的天下，优化逻辑和你练的 GEMM（compute-bound）相反。**

---

## 0. 这一周到底在练什么

```text
prefill（处理 prompt）= 大矩阵、compute-bound → 你 Week1~4 的 GEMM/Attention 直接用
decode（逐 token 生成）= M=1、memory-bound → 全新战场，本周主角
面试区分度：会写 GEMM 的人很多；能讲清 decode 为什么吃不满 GPU、怎么救，才是加分项。
```

本周产出（每天都要有代码/数据/口述，只看资料不算完成）：

```text
week05_inference/
  gemv.cu                  # M=1 矩阵向量乘（Day2）
  fused_rmsnorm.cu         # 算子融合（Day3）
  decode_graph.cu          # CUDA Graph 套 decode 循环（Day4）
  grid_reduce.cu           # cooperative groups 网格级归约（Day4）
  dequant_gemv.cu          # INT8 反量化 + GEMV（Day5）
notes/week05.md            # 每天：目标/数据/问题/口述
docs/
  kv_cache_accounting.md   # KV cache 显存账
  decode_step_dataflow.md  # 一次 decode step 的数据流与瓶颈
```

阅读标注：📖 精读 · 👀 扫读 · ✍️ 必须自己写 · ⏭️ 本周跳过。

---

## 1. 本周总览（一张表看完 8 个补充点怎么嵌入）

| Day | 主题 | 动手产出 | 嵌入的补充点 |
|-----|------|----------|-------------|
| 1 | 推理全景 + KV cache 显存账 | `kv_cache_accounting.md` | — |
| 2 | ⭐ **GEMV 手写**（本周核心） | `gemv.cu` + 带宽利用率 | ①GEMV ⑦occupancy 定量 ⑤nsys vs ncu |
| 3 | 算子融合 | `fused_rmsnorm.cu` | ②算子融合 ④compute-sanitizer |
| 4 | CUDA Graph + 网格级同步 | `decode_graph.cu` `grid_reduce.cu` | ③CUDA Graph ⑥cooperative groups ⑤nsys 时间线 |
| 5 | 量化推理 + Paged Attention | `dequant_gemv.cu` | （量化 GEMV，接 Day2） |
| 6 | 框架地图 + Hopper 概念 | 框架对比笔记 | ⑧Hopper（TMA/wgmma/cluster）|
| 7 | 复盘：decode 数据流 + 口述 | `decode_step_dataflow.md` | 全部串讲 |

```text
8 个补充点全部落位：①②③④⑤⑥⑦⑧
真正"额外"花时间的只有 Day3 算子融合和 Day6 Hopper 概念，其余都嵌在主线里。
```

---

## Day 1：推理全景 + KV cache 显存账（打地基，不写 kernel）

### 学什么
```text
1. prefill vs decode：
   prefill = 一次并行处理整个 prompt（N 个 token 一起）→ 大 GEMM，compute-bound
   decode  = 每次只生成 1 个新 token（M=1）→ GEMV，memory-bound
2. 为什么 decode 吃不满 GPU：
   每步只有 1 个 query，算得少、读得多（要读全部权重 + 全部历史 KV）
   → 算术强度(AI)极低 → 卡在 HBM 带宽，SM 大量空闲
3. latency vs throughput：单请求延迟 vs 批量吞吐，decode 靠 batching 提吞吐
```

### 动手（算账，不写代码）
```text
推导 KV cache 显存公式并算一个真实例子：
  KV cache 字节 = 2(K和V) × L层 × N(seq) × H_kv × Dh × dtype_bytes × batch
拿一个配置手算（如 L=32, H_kv=8, Dh=128, N=4096, batch=1, fp16）：
  = 2 × 32 × 4096 × 8 × 128 × 2 = 约 0.5 GB / 请求
再算 batch=32、N=8192 时多大 → 体会为什么长上下文/大 batch 显存爆炸
→ 这就是 MQA/GQA/MLA（Week4 学过）省 KV cache 的动机
```

### 完成标准
```text
[ ] notes/week05.md 写清 prefill/decode 区别 + 为什么 decode memory-bound
[ ] kv_cache_accounting.md 有公式 + 至少 2 组配置的显存数字
```

### 口述（面试风格）
```text
"decode 为什么难吃满 GPU？"
→ 每步 M=1，计算量小但要读全部权重和历史 KV，算术强度极低，
  瓶颈在 HBM 带宽而非算力，所以 SM 利用率低，只能靠 batching 摊薄。
```

---

## Day 2：⭐ GEMV 手写（本周核心，嵌入 ①⑤⑦）

### 学什么
```text
GEMV：y = A · x，A 是 [N,K] 矩阵，x 是 [K] 向量，y 是 [N] 向量（M=1 的 GEMM）
和 GEMM 的根本区别：
  GEMM：每个元素被复用 O(N) 次 → 靠 tiling 提高复用 → compute-bound
  GEMV：矩阵 A 每个元素只用 1 次 → 没有复用可榨 → 天生 memory-bound
优化目标也变了：
  GEMM 目标 = 逼近算力峰值(TFLOPS)
  GEMV 目标 = 逼近带宽峰值(GB/s)，让读 A 的带宽打满就赢了
```

### 动手 ✍️
```text
1. 写 baseline GEMV：每个线程/warp 负责 A 的一行，点积 x，输出 y[row]
2. 关键优化（都是"喂饱带宽"导向，不是"提高复用"）：
   - 合并访问：保证读 A 的相邻线程读相邻地址（行主序下按列切）
   - 向量化：float4/half2 读 A 和 x（你 Week2 练过）
   - warp 级归约：一个 warp 算一行的点积（你 Week3 的 warpReduceSum 直接复用）
   - x 放 shared/常量内存（x 被所有行复用，是唯一能复用的东西）
3. 测有效带宽：GB/s = 读写字节数 / 时间，对比 A100 峰值 ~1935 GB/s
```

### 嵌入 ⑦ occupancy 定量分析
```text
用 ncu 看 occupancy，并手算理论上限：
  理论 occupancy = 受限于 [寄存器/线程, shared/block, block 数] 的最小者
  A100 每 SM：65536 寄存器、最多 2048 线程、48KB(可配 164KB) shared
练习：算你的 GEMV kernel 每线程用多少寄存器 → 反推每 SM 能驻留多少 warp
结论对 GEMV：occupancy 高有助于隐藏访存延迟（memory-bound 尤其吃这个）
```

### 嵌入 ⑤ nsys vs ncu
```text
ncu（单 kernel 显微镜）：看 GEMV 的 SoL、Memory Throughput、occupancy、是否 memory-bound
nsys（时间线望远镜）：跑一个"多次 GEMV 串起来"的循环，看 kernel 之间有没有空隙
两者定位：ncu 回答"这个 kernel 好不好"，nsys 回答"整体流程哪里有气泡/能不能重叠"
```

### 完成标准
```text
[ ] gemv.cu PASS（对 CPU 参考）
[ ] 有效带宽记录，达到峰值合理比例（memory-bound kernel 目标 >70% 带宽）
[ ] ncu 证明 memory-bound（Memory SoL 高、Compute SoL 低）
[ ] 手算一次理论 occupancy
```

### 口述
```text
"为什么 decode 用 GEMV 而不是 GEMM 优化那套？"
→ GEMV 矩阵元素零复用，tiling 无用武之地，瓶颈是读矩阵的带宽；
  优化重点变成合并访问+向量化+打满带宽，目标是逼近 GB/s 峰值而非 TFLOPS。
```

---

## Day 3：算子融合（嵌入 ②④）

### 学什么
```text
未融合：norm kernel 写回 HBM → 激活 kernel 再读回来 → ... 每个算子一次 HBM 往返
融合：把 RMSNorm + 残差 + 激活 合进一个 kernel，中间结果留在寄存器/shared，不落 HBM
为什么对 LLM 关键：推理里有大量 elementwise/norm 小算子，全是 memory-bound，
  融合直接砍掉 HBM 往返次数 = 直接省带宽 = 直接提速
```

### 动手 ✍️
```text
1. 拿你 Week 练过的 rmsnorm + 一个激活(SiLU/GELU)
2. 先写"未融合"两个 kernel 版本，测 HBM 流量（ncu 的 dram__bytes）
3. 再写"融合"单 kernel 版本：一次读入、算完 norm 直接接激活、一次写出
4. 对比两版的 HBM 读写字节数和耗时 → 用数据证明融合省了多少往返
```

### 嵌入 ④ compute-sanitizer
```text
融合 kernel 容易出越界/竞争（多算子塞一个 kernel，索引更复杂）
用 compute-sanitizer 验证正确性：
  compute-sanitizer --tool memcheck ./fused_rmsnorm   # 查越界/非法访问
  compute-sanitizer --tool racecheck ./fused_rmsnorm  # 查 shared memory race
拿你之前的 race.cu 也跑一遍，体会它怎么定位数据竞争
```

### 完成标准
```text
[ ] fused_rmsnorm.cu PASS
[ ] ncu 数据：融合版 HBM 字节数明显低于未融合版
[ ] compute-sanitizer 无 error
```

### 口述
```text
"算子融合为什么能提速？"
→ 推理里 norm/激活/残差都是 memory-bound，未融合时每个算子都要往返 HBM；
  融合后中间结果留在片上，HBM 往返次数减少，直接省带宽，对 memory-bound 算子收益大。
```

---

## Day 4：CUDA Graph + 网格级同步（嵌入 ③⑥⑤）

### 学什么（③ CUDA Graph）
```text
问题：decode 每生成 1 token 要 launch 几百个小 kernel，每次 launch ~5μs，
     kernel 本身又小 → launch 开销占比大，GPU 在等 CPU（气泡）
CUDA Graph：把这串 kernel 录成一张图，之后一条 cudaGraphLaunch 重放，
           CPU 只发一次，GPU 连续跑完，消除 kernel 间气泡
适用前提：每步 kernel 序列/形状固定（decode 完美契合）
```

### 动手 ✍️
```text
1. 用 stream capture 把"几个 GEMV/融合 kernel 串成的伪 decode step"录成 graph：
   cudaStreamBeginCapture → 一串 kernel → cudaStreamEndCapture
   cudaGraphInstantiate → 循环里 cudaGraphLaunch
2. 对比"逐个 launch"vs"graph 重放"的耗时（尤其 kernel 多而小的时候）
```

### 嵌入 ⑤ nsys 看气泡
```text
用 nsys 抓两版时间线：
  逐个 launch 版：kernel 之间有 CPU 发起的空隙（气泡）
  graph 版：kernel 紧挨着，气泡消失
→ 这是"nsys 看整体流程/重叠"的最佳实战，和 Day2 的 ncu 形成互补
```

### 嵌入 ⑥ cooperative groups / grid sync
```text
学什么：普通 kernel 只能 block 内 __syncthreads()，跨 block 不能同步；
       cooperative groups 的 grid.sync() 能做全网格同步（需 cooperative launch）
动手：写一个 grid-level reduction（一个 kernel 内跨所有 block 归约完，不用二次 launch）
      对比你 Week3 的"两次 launch 归约"，体会 persistent kernel 思路
注意：cooperative launch 有 occupancy 限制（所有 block 要同时驻留），了解约束即可
```

### 完成标准
```text
[ ] decode_graph.cu：graph 版比逐个 launch 版快（记录数字）
[ ] nsys 时间线截图/描述，能指出气泡消失
[ ] grid_reduce.cu PASS，能说清 grid.sync() 和 __syncthreads() 区别
```

### 口述
```text
"decode 逐 token 慢，除了 kernel 本身还能怎么救？"
→ 用 CUDA Graph 把每步固定的 kernel 序列录成图重放，消除几百次 launch 的 CPU 开销和气泡；
  nsys 能直接看到气泡从有到无。
```

---

## Day 5：量化推理 + Paged Attention（接 Day2 的 GEMV）

### 学什么（量化）
```text
为什么量化：decode memory-bound，瓶颈是读权重的带宽；
          把权重从 fp16 压到 int8/int4 → 读的字节数减半/减到 1/4 → 直接提速
per-tensor vs per-channel：整个张量一个 scale vs 每列一个 scale（精度/开销权衡）
GPTQ/AWQ：只需知道是"更聪明地选 scale/保护重要权重"的后训练量化方法（概念）
```

### 动手 ✍️
```text
写 dequant_gemv.cu：权重存 INT8 + scale，kernel 里"边读边反量化"再做 GEMV
  y = (int8_weight × scale) · x
对比 fp16 GEMV：读权重字节数减半 → 有效带宽压力下降 → 应更快
用 ncu 确认 dram__bytes 降了
```

### 学什么（Paged Attention，概念为主）
```text
问题：KV cache 长度不定、请求来去，连续大块显存会碎片化、浪费
Paged Attention（vLLM 核心）：像操作系统虚拟内存，把 KV cache 切成固定大小 block，
  用 block table 映射逻辑位置→物理 block → 消除碎片、支持共享前缀
本周只需讲清"解决什么问题、类比 OS 分页"，不要求手写完整 kernel
```

### 完成标准
```text
[ ] dequant_gemv.cu PASS，ncu 显示读权重字节数下降
[ ] notes 写清 paged attention 解决的碎片问题（类比 OS 分页）
```

### 口述
```text
"量化为什么能加速 decode？" → decode 卡在读权重带宽，量化减少权重字节数，直接缓解带宽瓶颈。
"Paged Attention 解决什么？" → KV cache 显存碎片，用分页+block table 管理，提高显存利用率、支持前缀共享。
```

---

## Day 6：框架地图 + Hopper 概念（嵌入 ⑧）

### 学什么（框架地图，扫读）
```text
vLLM：Paged Attention + continuous batching，主打高吞吐服务
TensorRT-LLM：NVIDIA 官方，kernel 高度优化 + 图编译，主打极致性能
SGLang：结构化生成 + RadixAttention（前缀缓存），主打复杂调度
记住"各自解决什么"即可，不深入源码
```

### 嵌入 ⑧ Hopper 新特性（概念，不写代码）
```text
DeepSeek 用 H800(Hopper)，面试可能问，了解概念即可：
  TMA(Tensor Memory Accelerator)：硬件异步搬大块数据，比 cp.async 更强，
    地址计算交给硬件，SM 更专注计算（你 Week2 的 cp.async 的进化版）
  wgmma：warp-group 级的 Tensor Core 矩阵乘，比 Ampere 的 wmma 吞吐更高
    （你 Week3 写的 wmma 的 Hopper 升级版）
  Thread Block Cluster + 分布式 shared memory：多个 block 组成 cluster，
    能互相访问对方的 shared memory，扩大片上数据复用范围
一句话：A100(你的硬件) → Hopper 的每个新特性，都是你已学东西的加强版。
```

### 完成标准
```text
[ ] 一段话说清 vLLM/TRT-LLM/SGLang 各自定位
[ ] 能把 TMA/wgmma/cluster 各对应到你在 A100 上学过的哪个东西
```

### 口述
```text
"Hopper 比 Ampere 强在哪（和你会的东西对应）？"
→ TMA 是 cp.async 的硬件加强版，wgmma 是 wmma 的升级，cluster 让 shared memory 跨 block 复用；
  本质都是把我在 A100 上手写优化的那些点，用硬件做得更彻底。
```

---

## Day 7：复盘——画出一次 decode step 的数据流 + 口述

### 动手
```text
1. 画一次 decode step 的完整数据流（decode_step_dataflow.md）：
   输入 1 个 token → embedding → [每层: QKV投影(GEMV) → attention(读KV cache)
   → O投影(GEMV) → FFN(GEMV) → norm(融合)] × L → logits → 采样 → 新 token
   在每一步标注：这步是 GEMV 还是 attention？memory-bound 还是 compute-bound？
2. 整理本周 benchmark 表：GEMV 带宽、融合前后 HBM、graph 前后耗时、量化前后
3. 写 3 段面试口述：decode 瓶颈 / decode 优化手段全家桶 / prefill vs decode 差异
```

### 本周验收（闭卷能答）
```text
[ ] 为什么 decode 比 prefill 难吃满 GPU？
[ ] GEMV 和 GEMM 优化策略为什么相反？
[ ] 算子融合、CUDA Graph、量化 各自解决 decode 的什么瓶颈？
[ ] KV cache 显存怎么估？MQA/GQA/MLA/Paged Attention 各解决什么？
[ ] nsys 和 ncu 分别什么时候用？
[ ] Hopper 的 TMA/wgmma/cluster 对应你会的哪些 A100 技术？
```

---

## 附：8 个补充点落位速查

| 补充点 | 落在哪天 | 形式 |
|---|---|---|
| ① GEMV/decode kernel ★★★ | Day2 | 手写主任务 |
| ② 算子融合 ★★ | Day3 | 手写主任务 |
| ③ CUDA Graph ★★ | Day4 | 手写主任务 |
| ④ compute-sanitizer ★ | Day3 | 工具，验证融合 kernel |
| ⑤ nsys vs ncu ★★ | Day2+Day4 | 工具，穿插使用 |
| ⑥ cooperative groups ★ | Day4 | 手写 grid reduction |
| ⑦ occupancy 定量 ★★ | Day2 | 分析 GEMV 时手算 |
| ⑧ Hopper 特性 ★ | Day6 | 概念，对应已学技术 |

```text
真正"额外开天"的只有 Day3(融合) 和 Day6 半天(Hopper 概念)；
其余 6 个点都是 decode 优化主线的副产品，顺手拿下。
```
