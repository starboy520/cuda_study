# 05 部署兼容性：Compute Capability 与 PTX JIT

## 0. 先建立大局观："在我机器上能跑"≠"到处能跑"

你在 T4 上编译、测试、一切正常。交付到客户的 A100 上——**报错跑不了**。或者反过来，换到一台
更老的卡上直接崩。这类"换台机器就挂"的问题，根源是 **GPU 架构兼容性**。

本章回答一个工程上极其现实的问题：**怎么编译，才能让我的程序在目标部署环境（可能不止一种
GPU）上都能跑？**

这一卷是卷二/06（NVCC/PTX/fat binary）的**部署视角收口**——同样的知识，从"编译原理"转到
"上线后会不会跑不了"。

## 0.1 术语速查表

| 术语 | 一句话定义 |
|---|---|
| **Compute Capability (CC)** | GPU 的能力版本号，如 7.5（Turing/T4）、8.0（A100）|
| **sm_XX** | 真实架构，决定为哪代卡生成 SASS（机器码）|
| **compute_XX** | 虚拟架构，决定生成什么 PTX（中间码）|
| **cubin** | 某架构的现成机器码，启动快但只认那代卡 |
| **PTX JIT** | 运行时把 PTX 即时编译成当前卡的 SASS（前向兼容）|
| **fat binary** | 一个程序里打包多份 cubin + PTX |
| **`no kernel image`** | 找不到匹配当前 GPU 的代码时的经典报错 |

## 1. 问题的根源：SASS 绑定架构

回忆卷二/06：GPU 真正执行的是 **SASS**（机器码），而**每代架构的 SASS 不兼容**。所以一份只为
sm_75 编译的 cubin：

```text
在 T4（sm_75）上：直接用现成 SASS，完美
在 A100（sm_80）上：SASS 不匹配 -> 找不到可用的 kernel image -> 报错
在更老的卡上：    同样不匹配 -> 报错
```

这就是那个最经典的部署报错：

```text
no kernel image is available for execution on the device
```

它**几乎总是架构没匹配上**——你编译的架构和运行的 GPU 对不上。

## 2. 两条兼容路径：cubin（向后） vs PTX（向前）

理解兼容性，先分清两个方向（卷二/06 §3.1）：

```text
向后兼容（cubin）：为 sm_75 编的 cubin，能在 sm_75 上跑。
                  注意 cubin 一般只认那一代（不能跨代用别代的 cubin）。

向前兼容（PTX JIT）：compute_75 的 PTX，能在 sm_75 及更高（8.0/9.0）上
                    由驱动 JIT 成对应 SASS —— 这是覆盖"未来新卡"的唯一办法。
```

两条路径的权衡（卷二/06 三方案的部署版）：

| 打包内容 | 已知目标卡 | 未来新卡 | 启动速度 |
|---|---|---|---|
| 只 cubin（如 sm_75）| ✅ 匹配的能跑 | ❌ 报错 | 快（零 JIT）|
| 只 PTX（compute_75）| ✅ 能 JIT | ✅ 能 JIT | 慢（每次首启 JIT）|
| cubin + PTX（推荐）| ✅ 用现成 cubin | ✅ 回退 PTX JIT | 已知卡快、新卡兜底 |

## 3. 推荐做法：列出部署架构 + 最高架构留 PTX

生产构建的黄金法则（卷二/06 §6.2 + 卷十/01 CMake）：

```bash
# 假设部署环境有 T4(7.5) 和 A100(8.0)
nvcc -gencode arch=compute_75,code=sm_75 \    # T4 现成 cubin
     -gencode arch=compute_80,code=sm_80 \    # A100 现成 cubin
     -gencode arch=compute_80,code=compute_80 \  # 最高架构留 PTX，给未来卡 JIT 兜底
     app.cu
```

CMake 等价写法（第 01 章）：

```cmake
set_target_properties(app PROPERTIES CUDA_ARCHITECTURES "75;80;80-virtual")
```

原则：

```text
✅ 把【所有已知部署架构】各打一份 cubin -> 这些卡启动快、零 JIT
✅ 把【最高架构】额外留一份 PTX        -> 未来更新的卡能 JIT 兜底
❌ 别用"全历史架构"               -> 编译慢、体积大，多数用不上（卷二/06 §6.2）
```

## 4. JIT 的隐藏成本：首启延迟与缓存

只靠 PTX JIT 兜底要知道它的代价（卷二/06 §6.3）：

```text
首次启动：驱动把 PTX JIT 成 SASS，有明显延迟
之后：    JIT 结果缓存到磁盘（~/.nv/ComputeCache），同程序同卡再启动命中缓存、跳过 JIT
```

但在**容器/CI** 场景这个缓存常常失效：

```text
每个新容器 = 干净环境、没有缓存 -> 每次首启都重新 JIT -> 反复吃延迟
```

> 这正是生产偏向"直接打包目标卡 cubin"（而非只留 PTX）的核心原因之一：避免每个新容器实例
> 都付一次 JIT 成本。PTX 只作为"未知新卡"的兜底，不作为已知卡的主路径。

## 5. 运行时查 GPU 能力：别硬编码假设

代码里不要假设"一定是 sm_75"。运行时查实际设备能力（卷二/06 §5.1 提过 `__CUDA_ARCH__` 是
编译期、查不了运行时架构，要用这个 API）：

```cpp
cudaDeviceProp prop;
CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
printf("GPU: %s, Compute Capability %d.%d\n", prop.name, prop.major, prop.minor);

// 例：某特性要 8.0+，运行时判断走哪条路径
if (prop.major >= 8) {
    // 用需要 Ampere+ 的实现
} else {
    // 退回通用实现
}
```

`prop.major/minor` 就是计算能力（如 7/5 = 7.5）。这让程序能**适配实际硬件**，而不是编译时
写死一种假设。

## 6. 部署检查清单

上线前过一遍，避免"换台机器跑不了"：

```text
[ ] 列出所有目标部署 GPU 的 Compute Capability
[ ] 为每个目标 CC 打了 cubin（启动快）
[ ] 为最高 CC 额外留了 PTX（未来卡兜底）
[ ] 没有滥用"全架构"导致编译/体积爆炸
[ ] 容器场景评估了 JIT 缓存失效的首启成本
[ ] 运行时用 cudaGetDeviceProperties 适配，没硬编码架构假设
[ ] 用 cuobjdump --list-elf 验证产物里确实有目标架构（卷二/06）
```

## 7. 实践

1. 把一个 kernel 只编 `-arch=sm_80`，在 T4（sm_75）上运行，复现 `no kernel image` 报错。
2. 改成 `-gencode ...code=compute_75`（留 PTX），确认它能在 T4 上 JIT 跑起来。
3. 用 `cuobjdump --list-elf` 对比"只 cubin" vs "cubin+PTX"两种产物里打包了什么。
4. 写一段 `cudaGetDeviceProperties` 代码打印当前卡的 CC，并据此选择 block size。

## 8. 面试题（附参考答案）

**Q1：`no kernel image is available` 通常是什么原因？**
编译的架构和运行的 GPU 不匹配——程序里没有当前卡能用的 cubin，也没有可 JIT 的 PTX。解决：为
目标卡补对应 `-gencode`，或留一份 PTX 兜底。

**Q2：cubin 和 PTX 在部署兼容性上各负责什么方向？**
cubin 是某架构的现成机器码，负责"已知目标卡启动快"；PTX 经驱动 JIT 能在更高架构上跑，负责
"未来新卡前向兼容"。生产两者都打包。

**Q3：生产构建的架构怎么选？**
为所有已知部署架构各打 cubin（启动快），再为最高架构留一份 PTX（未来卡兜底）；不要用全历史
架构（编译慢、体积大）。

**Q4：只留 PTX 有什么代价？**
每次首启要 JIT，有延迟；虽有磁盘缓存，但容器/CI 每个新实例都是干净环境、缓存失效，反复付 JIT
成本。所以已知卡应直接打 cubin。

**Q5：运行时怎么知道当前 GPU 的能力？`__CUDA_ARCH__` 行吗？**
用 `cudaGetDeviceProperties` 读 `major/minor`。`__CUDA_ARCH__` 是编译期常量，查不了运行时实际
架构。

## 9. 资料映射

- CUDA C++ Best Practices Guide：Building for Maximum Compatibility。
- CUDA Programming Guide：Compute Capabilities、Versioning and Compatibility。
- 配套：[卷二第 06 章 NVCC、PTX 与编译流程](../volume02_programming_model/06_NVCC_PTX与编译流程.md)、[卷十第 01 章 CMake 与可复现构建](01_CMake与可复现构建.md)。
