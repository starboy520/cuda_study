# Week5（LLM 推理优化 / Decode）Work Log

> 对应：docs/Week5增强版_LLM推理优化与decode.md
> 硬件：A100 80GB PCIe（sm_80，HBM 峰值约 1935 GB/s）
> 主题：decode = memory-bound 的天下，优化逻辑和 GEMM（compute-bound）相反

---

## Day 1（2026-07-07）：Prefill / Decode / KV Cache（概念 + 算账）

### 概念（从零掌握）
- prefill/decode 输入 shape：prefill `[B,N,D]`（N=prompt 长度，大）；decode `[B,1,D]`（每步 1 个 token）
  - N 从大变 1 → 线性层从 GEMM 塌成 GEMV → decode memory-bound 的源头
- Transformer 一层数据流（Pre-Norm）：`H=X+Attn(RMSNorm(X))`，`Y=H+MLP(RMSNorm(H))`
  - attention = token 间通信；MLP = 每 token 独立加工（升维→激活→降维）
- KV cache 避免的是「历史 token 的 K/V 重复投影计算」（不是 attention 值；Q 用完即弃不缓存）
- KV 字节公式：`2×L×B×N×Hkv×Dh×dtype_bytes`
- MHA/GQA/MQA：只改 Hkv（32 / 8 / 1），cache ∝ Hkv；MLA 压成低维 latent
- 为什么不能无条件说 decode memory-bound：batching 把 M 从 1 拉到几十 → 权重复用↑ → 可能 compute-bound；
  还取决于 dtype/shape/kernel，要账本+profiler 判定

### 闭卷（已过）
decode 每步 1 个 query 却要读全部历史 KV，故受 cache 带宽限制；cache 用「存储+带宽」换「重复计算」。

---

## Day 2（2026-07-07）：GEMV —— 一线程/一 warp 一行

### 交付
- `week05_inference/gemv.cu` —— 自己写两个 kernel，4 组 shape 全 PASS，memcheck 0 errors
  - `gemv_thread`（v0）：一线程一行，`fmaf` 累加
  - `gemv_warp`（v1）：一 warp 一行，lane-stride 累加 + warp shuffle 归约
- CPU reference + 测试框架（对拍/CUDA Event 计时/GB·s/GFLOPS）由脚手架提供

### 性能对比（N=4096 K=4096，正常运行非 sanitizer）
| 版本 | 时间 | 带宽 | max_rel |
|------|------|------|---------|
| thread/row (v0) | 0.935 ms | 72 GB/s | 2.6e-4 |
| warp/row (v1) | 0.049 ms | 1360 GB/s | 6.8e-5 |
| 提升 | ~19× | ~19×（达峰值 ~70%） | 更准 |

### 洞察（自己悟出）
- v1 快 19×：一 warp 32 lane 读同一行连续 k → **W 合并访问**（v0 相邻线程 stride=K，访存分散）
- v1 更准：v0 顺序累加 4096 个 → 累加链长；v1 每 lane 只累加 128 个再树形归约 → 链短，fp32 误差小
  → **树形归约不只快，还更数值稳定**
- W 是真瓶颈（N×K 每元素读 1 次）；x 被 N 行复用但小、常驻 cache，放 shared 未必赢（省小头）
- `fmaf(a,b,c)=a*b+c`：一条硬件指令、单次舍入，又快又准，GEMV/GEMM 内层标配

### 踩坑 / 修的框架 bug
- 输入生成 `(size_t)%23 - 11` 无符号下溢 → 巨值污染（被正确 kernel 暴露）；修：先转 int 再减
- 容差 1e-4 对 K=4096 fp32 累加太严（误差 ~K*eps≈2.5e-4）→ 放宽到 1e-3
- sanitizer 下时间会慢几十倍（插桩），只看 0 errors，不看性能

### 闭卷（已过）
v0 一线程一行、地址 stride=K 不合并；v1 一 warp 一行、lane 读连续 k 合并 + shuffle 归约。
GEMV memory-bound，W 无复用是瓶颈，x 有复用但可靠 cache。

---

## 下一步（TODO）
- [x] Day1：prefill/decode/KV cache 概念 + 字节账
- [x] Day2：gemv.cu（thread/warp 一行），19× 带宽提升，memcheck 0 errors
- [ ] Day3：GEMV 性能深化（float4 向量化、occupancy、ncu/nsys 实测）
- [ ] Day4：RMSNorm/Residual/SiLU 融合 kernel + HBM 账本
- [ ] Day5：CUDA Graph + cooperative grid reduce
- [ ] Day6：INT8 weight-only 量化 GEMV
- [ ] Day7：Paged Attention / 框架 / Hopper + 完整 decode 图
