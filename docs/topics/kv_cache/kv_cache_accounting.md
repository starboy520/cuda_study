# KV Cache 显存账

> 对应：docs/courses/inference/Week5增强版_LLM推理优化与decode.md（Day1 §5、§7）
> 目的：会算「KV cache 占多少显存」，理解为什么长上下文推理极度关心它。

---

## 1. 公式

标准（按 KV heads 存储）单层、全序列的 KV cache 字节：

$$
\text{KV bytes} = 2 \times L \times B \times N \times H_{kv} \times D_h \times \text{dtype\_bytes}
$$

| 符号 | 含义 |
|------|------|
| `2` | K 和 V 两份 |
| `L` | Transformer 层数（每层独立一份 KV） |
| `B` | batch（同时处理的序列数） |
| `N` | 序列长度（已缓存的 token 数） |
| `Hkv` | **K/V head 数**（不一定等于 query head 数 Hq） |
| `Dh` | 每个 head 的维度 |
| `dtype_bytes` | FP16=2、BF16=2、FP8=1、FP32=4 |

> 不含 block metadata、padding、workspace、碎片。

---

## 2. 手算一（单请求）

```text
L=32, B=1, N=4096, Hkv=8, Dh=128, FP16=2 bytes

bytes = 2 × 32 × 1 × 4096 × 8 × 128 × 2
      = 536,870,912 bytes
      = 512 MiB
```

## 3. 手算二（batch + 长序列）

```text
同模型，B=16, N=8192：

512 MiB × 16(B) × 2(N 翻倍) = 16 GiB
```

一个请求 512 MiB，16 路并发 + 上下文翻倍就吃掉 **16 GiB**——还没算权重、activation、workspace、碎片。这就是长上下文 / 高并发推理显存吃紧的直接原因。

---

## 4. 每 token 每层的最小单元

```text
一个 token 单层：K 有 Hkv×Dh 个数，V 有 Hkv×Dh 个数
  = 2 × Hkv × Dh 个数
例：Hkv=32, Dh=128 → 2×32×128 = 8192 个数
  × FP16(2B) = 16 KiB / token / 层
```

再乘 N（token 数）× L（层数）就是全部。

---

## 5. MHA / GQA / MQA：只改 Hkv

query head 数 Hq 不变，变的是 **K/V head 数 Hkv**，cache ∝ Hkv：

| 机制 | Hkv（设 Hq=32） | 相对 MHA cache |
|------|----------------|---------------|
| MHA | 32（=Hq） | 1× |
| GQA | 8（几个 Q 头共享一组 KV） | 1/4 |
| MQA | 1（所有 Q 头共享） | 1/32 |

代价：共享越多，表达能力可能受损；不是越小越好，取决于训练与模型设计。

---

## 6. MLA（DeepSeek）：不能套 Hkv×Dh

MLA 把每 token 的 K/V 压成低维 latent `c^KV`（+ 位置相关分量）再缓存，cache 主体从
`Hkv×(Dh+Dv)` 变成约 `d_c + d_rope`：

```text
例（练公式，非某版真实参数）：
  MHA = Hkv×(Dh+Dv) = 32×256 = 8192 元素/token/层
  MLA = d_c + d_rope = 512 + 64 = 576 元素/token/层
  压缩比 ≈ 576/8192 ≈ 7%
```

MLA cache 不能机械套 `Hkv×Dh`，要按 latent + 位置分量的实际 shape 算。

---

## 7. Day1 填空练习（示例：LLaMA-7B 量级）

```text
模型：L=32, Hq=32, Hkv=32, Dh=128, dtype=FP16(2B)
配置A：B=1,  N=2048  → KV = 2×32×1×2048×32×128×2  = 1024 MiB = 1 GiB
配置B：B=8,  N=8192  → 1 GiB × 8 × 4               = 32 GiB
权重约 = 7B params × 2B(FP16) ≈ 14 GiB
剩余显存能放多少 cached tokens：
  A100 80GB − 14 GiB(权重) − 若干 workspace ≈ 60+ GiB
  每 token 每层 KV = 2×Hkv×Dh×2B = 2×32×128×2 = 16 KiB
  每 token（全 L 层）= 16 KiB × 32 = 512 KiB
  60 GiB / 512 KiB ≈ 12 万 tokens（粗算，不含碎片/对齐）
```
> 数字随配置变化，重点是掌握「先算 bytes 账、再判断能放多少」的方法。

---

## 8. 闭卷要点

1. **KV cache 避免的重复计算**：历史 token 的 K/V 投影（它们不随新 token 改变，可复用）；不是 attention 值（query 每步不同，要重算），也不缓存 Q。
2. **为什么省 FLOP 却增显存/带宽**：用「存 K/V + 每步读一遍」换掉「每步重算历史 K/V」——空间/带宽换计算。
3. **为什么用 Hkv**：cache 只存 K/V，正比于 K/V head 数；GQA/MQA 靠减 Hkv 省 cache。
4. **长上下文变慢主因**：每步 attention 要读整个 KV cache，读取量随 N 线性增长。
