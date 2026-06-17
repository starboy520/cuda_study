# 06 NVCC、PTX 与编译流程

## 0. 先建立大局观：一个类比

在钻进术语之前，先用一个你大概率熟悉的东西类比——**Java**。

```text
Java:   .java 源码 -> javac -> .class 字节码(跨平台) -> JVM 在运行时 JIT -> 本机机器码
CUDA:   .cu 源码  -> nvcc  -> PTX 中间码(跨GPU代)   -> 驱动在运行时 JIT -> SASS 本机机器码
```

两边几乎一一对应：

- `.cu` 里的 **device 代码** ≈ `.java`，是你写的源码。
- **PTX** ≈ Java 字节码：一份**不绑定具体硬件**的中间码，能在"未来的卡"上继续用。
- **SASS** ≈ JVM 跑出来的本机机器码：**只认这一代 GPU**，换代就不一样。
- **驱动 JIT** ≈ JVM 的即时编译：在程序运行时把中间码翻成真正的机器码。

记住这条主线，后面所有术语都是在给这条流水线的某个环节起名字。

## 0.1 术语速查表（先扫一眼，看不懂没关系，下面会逐个讲）

| 术语 | 一句话定义 | 类比 |
|---|---|---|
| **NVCC** | 编译总指挥，负责把 `.cu` 拆开分别交给两套编译器 | 项目经理 |
| **Host code** | `.cu` 里跑在 CPU 上的普通 C++ 代码 | 普通 C++ |
| **Device code** | `.cu` 里跑在 GPU 上的 kernel 代码（`__global__` 等）| GPU 代码 |
| **PTX** | 面向"虚拟 GPU"的中间汇编，跨架构代稳定 | Java 字节码 |
| **SASS** | 某一代真实 GPU 的机器指令（最终执行的东西）| x86 机器码 |
| **cubin** | 装着 SASS 的二进制文件（CUDA binary 的缩写）| `.exe`/`.o` |
| **ptxas** | 把 PTX 编译成 SASS/cubin 的汇编器 | 汇编器 as |
| **JIT** | 运行时把 PTX 即时翻译成 SASS | JVM 即时编译 |
| **fat binary** | 一个可执行文件里同时塞多份 cubin + PTX | 一包多种规格 |
| **compute_XX** | 虚拟架构编号（决定 PTX 能用哪些特性）| 字节码版本 |
| **sm_XX** | 真实架构编号（决定为哪代卡生成 SASS）| CPU 型号 |

## 1. `.cu` 文件包含两种世界

一个 `.cu` 文件里其实**混着两类代码**，它们最终跑在不同的处理器上：

```text
Host C++ code    -> 跑在 CPU 上（main、内存分配、循环、printf……）
Device CUDA code -> 跑在 GPU 上（__global__ kernel、__device__ 函数）
```

NVCC 做的第一件事就是把这两个世界**分开**：Host 部分交给系统自带的 C++ 编译器
（g++ / clang / MSVC），Device 部分交给 NVIDIA 自己的 GPU 工具链。所以 NVCC 本身
**不是**一个完整编译器，而是一个**编译驱动（driver）/总指挥**，负责调度这两套工具，
最后再把产物拼到一起。

## 2. 简化流程

把第 1 节的"分开"画成图：

```text
.cu
├─ Host 部分 -> g++/clang/MSVC -> Host object code（CPU 机器码）
└─ Device 部分 -> CUDA front-end（cicc）
                  -> PTX（中间码）
                  -> ptxas -> SASS/cubin（GPU 机器码）
                  -> 一起塞进 fat binary
最终由 nvcc 把 Host object + fat binary 链接成一个可执行文件
```

注意这里 Device 侧有**两段编译**：先把源码编成 PTX，再把 PTX 编成 SASS。
理解这"两段"是理解后面所有内容的关键——因为你可以选择**只保留前半段产物（PTX）**、
**只保留后半段产物（cubin）**，或者**两个都留**（见第 6 节 fat binary）。

真实流程比这张图更复杂（还有 link、device link 等阶段），但这张图足够解释日常用到的选项。

## 3. PTX：跨代稳定的"中间码"

PTX（Parallel Thread eXecution）是一种**虚拟指令集 / 中间表示**，它**不是**任何一代
GPU 真正执行的机器码。你可以单独把它导出来看：

```bash
nvcc --ptx file.cu -o file.ptx
```

打开 `file.ptx`，长得像汇编但相当可读，例如一段加法 kernel 可能是这样：

```text
.visible .entry add(...)
{
    .reg .f32   %f<4>;          // 声明用到的寄存器
    ld.param.u64  %rd1, [add_param_0];
    ...
    ld.global.f32 %f1, [%rd4];  // 从 global memory 读 a[i]
    ld.global.f32 %f2, [%rd6];  // 读 b[i]
    add.f32       %f3, %f1, %f2;// a[i] + b[i]
    st.global.f32 [%rd8], %f3;  // 写回 c[i]
    ret;
}
```

**为什么要有这个"虚拟"中间层，而不是直接编成 GPU 机器码？** 因为 NVIDIA 每一代
架构（Turing、Ampere、Hopper……）的**真实指令集 SASS 都不一样**，而且会变。
如果只编译成今天这块卡的 SASS，那这个二进制**换一代新卡就跑不了**。PTX 解决的正是
这个矛盾：它是一份**面向"虚拟 GPU"的稳定中间码**，编一次，就能在未来的卡上由驱动在
运行时即时（JIT）翻译成那代卡真正的 SASS。

```text
源码 .cu
  -> PTX（虚拟 ISA，跨代稳定）
       -> [运行时] Driver JIT
            -> SASS（这台机器这代卡的真实机器码）
```

代价是：这次 JIT 翻译要花时间，发生在程序**首次启动**时（见第 6 节 fat binary 的权衡）。

## 3.1 PTX 的前向兼容到底兼容到什么程度

"PTX 能在更新的卡上 JIT"这句话很容易被误解成"PTX 万能、一次编译永久通用"。实际有
三条硬规则，务必记住：

**规则一：只能向上（更高计算能力）JIT，不能向下。**
`compute_75` 的 PTX 能在 7.5、8.0、9.0 的卡上 JIT 运行，但**不能**在 7.0、6.1 这些更
低的卡上跑。

```text
compute_75 的 PTX
  ├─ 在 sm_75 / sm_80 / sm_90 ... 上：可以 JIT ✅（向上兼容）
  └─ 在 sm_70 / sm_61 ... 上：直接失败 ❌（不能向下）
```

**规则二：向上兼容 ≠ 能用上新硬件的新特性。**
`compute_75` 的 PTX 在 Hopper（9.0）上 JIT，跑出来的功能仍停留在"7.5 能表达的范围"。
Ampere/Hopper 新增的指令（如更强的 tensor core、异步拷贝 `cp.async`、TMA 等）在 7.5 的
PTX 里**根本没有对应写法**，自然用不上。想吃新特性，**必须用更高的 `compute_XX` 重新
编译**。

```text
想在 Hopper 上用它的新指令？
  用 compute_75 编：跑得起来，但只发挥出 7.5 的能力（新指令用不到）
  用 compute_90 编：才能生成 Hopper 专属的新指令
```

**规则三：PTX 版本还受工具链 / 驱动版本约束。**
高版本 CUDA 工具链产出的 PTX，老驱动可能不认识（"PTX ISA version not supported"）。
所以"前向兼容"是**硬件向前**，不代表**软件可以无限向后**——驱动太旧照样 JIT 失败。

> 一句话：PTX 给你的是"**未来新卡还能跑**"，不是"**自动用上新卡的全部本事**"，也不是
> "**任何驱动都能 JIT**"。

## 4. Cubin 与 SASS：最终真正执行的东西

这是你说"不知道是什么"的两个词，重点讲。

- **SASS**（Shader ASSembly）= **某一代真实 GPU 的机器指令**。它才是 GPU 硬件**真正
  执行**的东西，相当于 CPU 世界里的 x86/ARM 机器码。Turing 的 SASS 和 Ampere 的 SASS
  指令编码不同，所以 SASS **绑定具体架构**。
- **cubin**（CUDA binary）= **装着 SASS 的二进制文件**。SASS 是"指令内容"，cubin 是
  "装它的盒子/文件格式"（ELF 格式）。类比：SASS 像机器码字节，cubin 像 `.exe`/`.o` 文件。

三者关系一句话串起来：

```text
PTX  --ptxas 编译-->  SASS（指令） 打包进  cubin（文件）
中间码                真实机器码          二进制容器
```

**怎么把它们看出来？** 编译出可执行文件后：

```bash
cuobjdump --list-elf  executable   # 列出里面打包了哪些架构的 cubin
cuobjdump --dump-sass executable   # 反汇编出真正的 SASS 指令
```

`--dump-sass` 的输出大致长这样（比 PTX 更底层、更接近硬件）：

```text
/*0000*/  MOV R1, c[0x0][0x28] ;
/*0010*/  LDG.E R0, [R2] ;          // 从 global memory 装载
/*0020*/  LDG.E R5, [R4] ;
/*0030*/  FADD R0, R0, R5 ;         // 浮点加法
/*0040*/  STG.E [R6], R0 ;          // 写回
/*0050*/  EXIT ;
```

对比第 3 节的 PTX：PTX 里是 `add.f32`、寄存器写成 `%f1`（数量不限的虚拟寄存器）；
SASS 里变成了 `FADD`、用的是 `R0` 这种**真实物理寄存器**。这正体现了"虚拟 → 真实"
的最后一跳。也可用 `nvdisasm` 单独反汇编一个 cubin 文件。

## 5. `compute_XX` 与 `sm_XX`：两个编号到底差在哪

编译时你常要指定架构，会看到两种写法，初学者最容易混：

```text
compute_75 = 虚拟架构（virtual arch）-> 决定生成什么样的 PTX（能用哪些特性）
sm_75      = 真实架构（real arch）   -> 决定为哪代卡生成 SASS（Turing / T4）
```

记忆法：**compute_ 管 PTX（虚拟），sm_ 管 SASS（真实硬件）**。数字是**计算能力
（Compute Capability）**，`75` 表示 7.5，对应 Turing 架构（T4 就是这一代）。

最简单的用法：

```bash
nvcc -arch=sm_75 program.cu     # 为 T4 生成代码
```

`-arch=sm_75` 是个便捷写法，它实际同时隐含了 `compute_75`（先生成 7.5 的 PTX，
再编成 7.5 的 SASS）。要更精细地控制"打包哪些产物"，就要用第 6 节的 `-gencode`。

## 5.1 `__CUDA_ARCH__`：在 device 代码里按架构走不同分支

有时你想让同一份 kernel **在不同架构上走不同实现**（比如新卡用新指令、老卡退回通用
写法）。这要靠编译期宏 `__CUDA_ARCH__`：

```cpp
__global__ void k() {
#if __CUDA_ARCH__ >= 800
    // Ampere(8.0) 及以上：用新特性，比如异步拷贝 cp.async
#elif __CUDA_ARCH__ >= 750
    // Turing(7.5)：退回到通用写法
#else
    // 更老的架构
#endif
}
```

三个关键点，每一个都是坑：

1. **它的值等于"当前正在编译的目标架构 ×10"**。编 `sm_80` 时 `__CUDA_ARCH__ == 800`，
   编 `sm_75` 时是 `750`。注意是 **SASS 那段编译**在用它。
2. **它会被编译多遍**。回忆第 2 节"两段编译"——如果你 `-gencode` 给了多个架构，device
   代码会**对每个架构各编一次**，每次 `__CUDA_ARCH__` 取不同值，于是不同架构进不同分支。
3. **host 侧未定义**。在 host 代码（普通 C++ 部分）里 `__CUDA_ARCH__` 是**没有定义**的。
   所以下面这种写法很常见——用"是否定义"来区分 host/device 编译路径：

```cpp
__host__ __device__ int f() {
#ifdef __CUDA_ARCH__
    return 1;   // 这次是在为 GPU 编译（device 路径）
#else
    return 0;   // 这次是在为 CPU 编译（host 路径）
#endif
}
```

> 常见误区：想用 `__CUDA_ARCH__` 在**运行时**判断"这张卡是什么架构"——做不到，它是
> **编译期**常量。运行时要查架构得用 `cudaGetDeviceProperties`（读 `major`/`minor`）。

## 6. Fat Binary

一个程序可以包含：

- 多个具体 SM 的 cubin。
- 一份较新/兼容的 PTX 供未来 Driver JIT。

生产构建常用 `-gencode` 明确组合。只放 PTX 有启动 JIT 成本；只放旧 cubin
可能无法覆盖新设备需求。

把这两条权衡用一个具体场景串起来。假设你现在为 T4 构建（`sm_75`）：

```text
方案 A：只打包 sm_75 cubin
  在 T4 上：直接加载已编译好的 SASS，启动最快。
  换到一块未来的新架构 GPU：找不到匹配 cubin，且没有 PTX 可 JIT —— 直接报错跑不了。

方案 B：只打包 compute_75 PTX
  在任何 >= sm_75 的卡上都能跑：驱动首次启动时 JIT 成 SASS。
  代价：每次首启都要 JIT，有启动延迟（可被 JIT 缓存缓解）。

方案 C（生产常用）：sm_75 cubin + compute_75 PTX 一起打包
  T4 上用现成 cubin，零 JIT；
  新卡上回退到 PTX 走 JIT。 兼顾启动速度与前向兼容。
```

对应的 `-gencode` 写法：

```bash
nvcc -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_75,code=compute_75 \
     program.cu
```

第一行产出 `sm_75` cubin，第二行把 `compute_75` 的 PTX 也嵌进 fat binary——
这就是方案 C。

## 6.1 把 `arch=` 和 `code=` 彻底讲清楚（新手最容易写错）

`-gencode arch=compute_XX,code=YYY` 里有两个参数，分工是固定的：

```text
arch=compute_XX   "中间码该按哪代虚拟架构生成 PTX"  —— 必须是 compute_，不能是 sm_
code=...          "这次最终要产出什么、塞进 fat binary"
```

`code=` 可以填两种值，含义完全不同：

| `code=` 的写法 | 产出什么 | 进 fat binary 的是 |
|---|---|---|
| `code=sm_75` | 把 PTX 继续编成 7.5 的 SASS | **cubin（现成机器码）** |
| `code=compute_75` | 到 PTX 为止，不再往下编 | **PTX（留给运行时 JIT）** |

所以这两行的差别一目了然：

```bash
-gencode arch=compute_75,code=sm_75        # 产 cubin：T4 上零 JIT，但只认 7.5
-gencode arch=compute_75,code=compute_75   # 留 PTX：未来新卡能 JIT，启动有成本
```

记忆口诀：**`arch=` 永远是 `compute_`（PTX 从哪来）；`code=` 决定终点是 `sm_`（cubin）
还是 `compute_`（PTX）。** 想要"现成 + 兜底"两者都有，就把这两行都写上（= 第 6 节方案 C）。

三个常见简写和它们的等价展开：

```text
-arch=sm_75
   ≡ -gencode arch=compute_75,code=sm_75          # 注意：只产 cubin，不留 PTX！

-arch=compute_75 -code=sm_75,compute_75
   ≡ -gencode arch=compute_75,code=sm_75
     -gencode arch=compute_75,code=compute_75      # cubin + PTX 都有
```

> 易错点：很多人以为 `-arch=sm_75` 会顺便留一份 PTX——**不会**。它只产 `sm_75` cubin，
> 换新架构卡就因"没有 cubin 也没有 PTX"而报错。要前向兼容必须显式加上 `code=compute_75`。

## 6.2 编译时间、二进制体积与生产取舍

`-gencode` 列得越多，代价越直接：

- **编译时间**：device 代码会对**每个架构各编译一遍**（回忆 5.1），列 5 个架构 ≈ 编 5 遍，
  编译时间近似翻 5 倍。
- **二进制体积**：每份 cubin 都实打实占空间，fat binary 会随架构数膨胀。

所以生产构建的常见做法是**只列实际部署的架构**，再加一份最高架构的 PTX 兜底：

```bash
# 假设线上只有 T4(7.5) 和 A100(8.0)
nvcc -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_80,code=compute_80 \   # 最高架构留 PTX，给未来新卡兜底
     program.cu
```

不要无脑用 `-arch=all` / 把所有历史架构都列上——那会让 CI 编译变慢、产物变大，多数架构还根本用不到。

## 6.3 JIT 只慢一次：JIT 缓存

第 3 节说"PTX JIT 有首启延迟"，但它**通常只慢第一次**。驱动会把 JIT 出来的 SASS
**缓存到磁盘**，下次同一程序、同一张卡、同一份 PTX 直接命中缓存，跳过 JIT：

```text
默认缓存目录：
  Linux   ~/.nv/ComputeCache
  Windows %APPDATA%\NVIDIA\ComputeCache
```

几个能控制它的环境变量：

```bash
export CUDA_CACHE_DISABLE=1          # 完全禁用缓存（每次都 JIT，调试用）
export CUDA_CACHE_MAXSIZE=1073741824 # 缓存上限（字节），满了按 LRU 淘汰
export CUDA_CACHE_PATH=/path/to/dir  # 自定义缓存位置
export CUDA_FORCE_PTX_JIT=1          # 强制走 PTX JIT（即使有匹配 cubin），验证 PTX 路径用
```

实战含义：

- 容器/CI 里每次都是干净环境、没有缓存，所以"首启 JIT 慢"会在**每个新容器**重现——
  这也是生产偏向**直接打包 cubin（方案 A/C）** 而非只留 PTX 的原因之一。
- 驱动升级、PTX 变化、卡型变化都会让缓存失效、重新 JIT。

## 6.5 cubin 是怎么和 host binary 拼到一起的

前面一直说"打包进 fat binary、链接成一个可执行文件"，但 host 可执行文件是给 CPU 用的
（ELF/PE 格式），里面**不能直接放 GPU 机器码去执行**，那 cubin 到底怎么进去的？

核心机制一句话：**cubin 被转成一段 C 字节数组，编进 host 目标文件，最终作为 host
可执行文件里的一段普通数据被带着走，运行时再由 CUDA runtime 取出来交给驱动。**

### 本质：GPU 代码"伪装"成 host 的数据

类比：就像你用 `xxd -i` 把一张图片转成 `unsigned char img[] = {0x89, 0x50, ...}`
编进 C 程序——图片并不"执行"，只是作为数据被带着走。GPU 代码也是这样被"夹带"进
host 二进制的。

### 分步过程

```text
第1步  device 源码 ──nvcc──► PTX ──ptxas──► cubin（SASS，ELF）
                                  │
第2步  把多份 cubin + PTX 打包 ──fatbinary 工具──► fatbin（一个 blob）
                                  │
第3步  把 fatbin 包成一个 .c 文件：里面是 host 胶水代码 + 嵌入的字节数组
                                  │
第4步  host 编译器(g++) 编译这个 .c ──► host object（fatbin 成了它的数据段）
                                  │
第5步  和你其它 host object 一起 ──链接──► 最终可执行文件
```

关键是**第 3 步**：nvcc 会生成一个中间文件（用 `nvcc --keep` 可以看到，名字类似
`xxx.fatbin.c` / `xxx.cudafe1.stub.c`），里面大致是：

```c
// nvcc 自动生成的胶水代码（示意）
asm(".section .nv_fatbin ...");          // 声明一个专门的段
static const unsigned char __fatbin[] =  // cubin/PTX 变成字节数组
    { 0x50, 0xed, 0x00, /* ... */ };

// 程序启动时（main 之前）自动运行的注册函数
static void __cuda_register_all(void) {
    __cudaRegisterFatBinary(__fatbin);           // 把这块数据登记给 runtime
    __cudaRegisterFunction(/* ... */ "myKernel");// 登记每个 kernel 的名字
}
```

这段胶水和你的 host 代码一起被 g++ 编译，于是 fatbin 就**物理上躺在了 host 可执行
文件的一个段里**（段名常是 `.nv_fatbin` / `.nvFatBinSegment`）。

### 验证：它真的在 host 文件里

```bash
# 看 ELF 段表，会有 .nv_fatbin / .nvFatBinSegment
readelf -S ./my_program | grep -i nv

# 直接把里面打包的 GPU 代码反汇编出来
cuobjdump --dump-sass ./my_program
```

`cuobjdump` 能从一个**普通 host 可执行文件**里 dump 出 SASS，正说明 cubin 确实被嵌
在里面了。

### 运行时怎么被用起来

链接进去只是"带着"，真正生效是在运行时：

```text
程序启动
  → 启动前自动调用 __cudaRegisterFatBinary，把那块数据交给 CUDA runtime
  → 你第一次 myKernel<<<...>>>() 启动 kernel
  → runtime 在 fatbin 里挑一份匹配当前 GPU 架构的 cubin
        ├─ 找到对应 sm 的 cubin：直接用现成 SASS（快）
        └─ 没有但有 PTX：驱动 JIT 编译 PTX → SASS（首次慢，见第 6 节方案 B/C）
  → 把 SASS 加载到 GPU 执行
```

`<<<>>>` 语法其实被 nvcc 翻译成了一串 runtime 调用（`cudaLaunchKernel` 等），它靠
**第 3 步登记的 kernel 名字**找到 fatbin 里对应的 GPU 代码——这就是 host 端一个看似
"普通的函数调用"最终能跑到 GPU 上的原因。

### 一句话总结

cubin 先打包成 fatbin，再被 nvcc 转成"字节数组 + 注册代码"的 host 源文件，交给 g++
编进 host 目标文件，于是它作为**数据段**和 host 代码一起链接成一个可执行文件；运行时
由 CUDA runtime 取出、按当前 GPU 架构挑选并加载执行。

## 7. 资源报告：让 ptxas 把"成本"打印出来

加一个开关，ptxas 会在编译时报告每个 kernel 用了多少硬件资源：

```bash
nvcc -Xptxas=-v ...
```

`-Xptxas=-v` 的意思是"把 `-v`（verbose）这个参数转交给 ptxas"。典型输出：

```text
ptxas info: Used 24 registers, 4096 bytes smem, 360 bytes cmem[0]
```

逐项含义：

- **Registers（寄存器）**：每个线程用了多少寄存器。用得越多，能同时驻留的线程越少，
  直接影响 **occupancy（占用率）**。
- **Shared memory（smem）**：每个 block 用的共享内存字节数，同样限制能同时跑的 block 数。
- **Spill stores/loads（寄存器溢出）**：寄存器不够用时，编译器把变量临时塞回慢速
  local memory，这里会显示溢出的读写次数。**出现 spill 通常是性能警告信号。**
- **Constant memory（cmem）**：用到的常量内存字节数。

这些数字是后面做 **occupancy 分析**和 **local spill 排查**的直接输入，是性能调优的起点。

## 8. `-lineinfo` 与 `-G`：两种"调试信息"别搞混

```text
-lineinfo  只保留"源码行号映射"，几乎不影响优化，主要给 profiler（如 Nsight）用
-G         生成完整 Device 调试信息，会关掉大量优化，给 cuda-gdb 单步调试用
```

关键区别：`-lineinfo` **基本不拖慢**性能，能在 profiler 里把热点对应回源码行；而 `-G`
为了能逐行断点调试会**关闭优化**，性能可能差好几倍。

> 切记：**不要用 `-G` 编出来的程序去测性能**，那不代表 release 的真实速度。

## 9. Sample

```bash
cd labs/02_programming_model/compile_inspection
make clean all
make ptx
make resource
sed -n '1,120p' compile_inspection.ptx
cuobjdump --dump-sass compile_inspection | less
```

观察：

- PTX 中 kernel entry。
- `.reg` 声明。
- Load、multiply/add、store。
- Ptxas 寄存器报告。

编译器可能把 `x*y+x` 变成 FMA，具体取决于选项和目标。

## 10. Separate Compilation（分离编译）

### 10.1 先理解问题：默认情况下 device 代码不能跨文件

普通 C++ 里你习以为常的事：`a.cpp` 定义一个函数，`b.cpp` 调用它，链接器把它们连起来。
这叫**分离编译（separate compilation）**。

但 CUDA 的 **device 代码默认不支持这件事**。默认模式叫 **whole program compilation
（整程序编译）**：每个 `.cu` 文件里的 device 代码必须**自包含**，一个文件里的
`__device__` 函数**看不到**另一个文件里的定义。

```text
file_a.cu:  __device__ int helper(int x){ return x*2; }
file_b.cu:  __global__ void k(){ ... helper(5) ... }   // ❌ 链接失败：找不到 helper
```

报错通常是：

```text
ptxas fatal: Unresolved extern function 'helper'
```

### 10.2 为什么默认是这样——SASS 没有"链接"这一步

回忆前面学的编译流水线。普通 CPU 代码：

```text
a.cpp -> a.o (含未解析符号 helper)
b.cpp -> b.o
        ↓ 链接器(ld)解析符号、重定位
     可执行文件
```

CPU 目标文件（`.o`）里保留了**重定位信息**，链接器能把"调用 helper"这个空位填上真实地址。

而默认模式下，nvcc 把每个 `.cu` 的 device 代码**一口气编到底**（直接生成最终
SASS/cubin），中间**不保留重定位信息**。SASS 是"已经定死地址"的机器码，没有给链接器
留任何"待填的空"。所以跨文件符号无从解析——**因为根本没有 device 端的链接阶段**。

### 10.3 解决方案：`-rdc=true`（可重定位 device 代码）

打开这个开关：

```bash
nvcc -rdc=true -c file_a.cu -o file_a.o
nvcc -rdc=true -c file_b.cu -o file_b.o
nvcc -rdc=true file_a.o file_b.o -o app     # device link + host link
```

`rdc` = **R**elocatable **D**evice **C**ode。开启后发生两个变化：

1. **每个 `.cu` 的 device 代码停在"可重定位"的中间产物**（保留重定位信息，不直接编死），
   类似 CPU 的 `.o`。
2. **多出一个 device link 阶段**：由一个叫 `nvlink` 的设备链接器，把各文件的 device
   代码连起来、解析 `helper` 这类跨文件符号，再统一生成最终 cubin。

```text
file_a.cu --(rdc)--> a.o (device 部分可重定位)
file_b.cu --(rdc)--> b.o
                      ↓ nvlink：device 端链接，解析 helper
                      ↓ 生成最终 cubin
                      ↓ 再和 host 代码做普通链接
                    app
```

### 10.4 它启用了哪些原来做不到的事

`-rdc=true` 不只是"跨文件调函数"，它解锁了一类能力：

- **跨 `.cu` 调用 `__device__` 函数**（最常见）。
- **跨文件共享 `__device__` / `__constant__` 全局变量**。
- **device 端的外部链接**（`extern __device__`）。
- **dynamic parallelism（动态并行）**：kernel 里再启动 kernel（`<<<>>>` 套娃），
  **强制要求** `-rdc=true`，因为子 kernel 的调用本质也是跨编译单元的 device 符号解析。
- 让 **device LTO（`-dlto`）** 成为可能（更激进的跨文件 device 优化）。

### 10.5 代价：为什么不默认开

既然这么有用，为什么 NVIDIA 默认关掉它？因为有性能代价：

1. **跨文件的 device 函数难以内联**。whole program 模式下编译器能看到全部 device 代码，
   激进 inline；分离编译时 `helper` 在另一个编译单元，**inline 受阻**，可能多出真实函数
   调用开销（寄存器压栈、跳转）。
2. **优化范围变窄**。编译器看不到被调函数的内部，常量传播、死代码消除等跨函数优化打折。
3. **多一个 link 阶段**，构建略慢。

> 经验法则：**能不开就不开**。只有当你确实需要跨文件 device 代码、或要用 dynamic
> parallelism 时才开。开了之后若发现热点函数性能下降，优先考虑把它 `__forceinline__`
> 或干脆放回同一编译单元。`-dlto` 可以部分挽回被牺牲的跨文件优化。

### 10.6 和"普通分离编译"的关系

注意区分两个层面，一个 `.cu` 里有**两套**编译：

```text
Host 代码：  本来就支持分离编译（走系统 ld），不受 -rdc 影响
Device 代码：默认不支持，-rdc=true 才开启
```

所以 `-rdc=true` 影响的**只是 device 那一半**。host 端的多文件链接一直都正常。

### 10.7 最小可验证例子

如果想亲手验证，三个文件：

```cpp
// helper.cu
__device__ int twice(int x) { return x * 2; }
```

```cpp
// kernel.cu
extern __device__ int twice(int);      // 声明在别处定义
__global__ void k(int* out) { *out = twice(21); }
```

```cpp
// main.cu
#include <cstdio>
__global__ void k(int*);
int main(){ int *d,h; cudaMalloc(&d,4); k<<<1,1>>>(d);
    cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost); printf("%d\n",h); }
```

对比两种编译：

```bash
# 不加 rdc：链接失败
nvcc helper.cu kernel.cu main.cu -o app          # ❌ Unresolved extern function 'twice'

# 加 rdc：成功，输出 42
nvcc -rdc=true helper.cu kernel.cu main.cu -o app # ✅
./app   # 42
```

### 10.8 一句话总结

device 代码默认"整程序编译、一文件一编到底"，没有链接阶段，所以跨 `.cu` 调 device
函数会失败；`-rdc=true` 让 device 代码变成"可重定位中间产物"并引入 `nvlink` device
链接阶段，从而支持跨文件 device 函数、全局变量和 dynamic parallelism，代价是牺牲部分
内联与跨文件优化、构建变慢——**按需开启**。

## 11. 练习

1. 分别用 `-arch=sm_75` 和另一目标编译，用 `cuobjdump --list-elf` 比较产物里打包了哪些架构。
2. 删除 kernel 上的 `__forceinline__`，对比前后 PTX/SASS 的差异。
3. 在 kernel 里加一个较大的局部数组，用 `-Xptxas=-v` 观察 registers 与 spill 的变化。

## 12. 面试题（附参考答案）

**Q1：NVCC 是否完全替代 Host compiler？**
不。NVCC 是编译驱动，它把 Host 代码**转交**给系统的 g++/clang/MSVC 去编，自己只负责
device 部分和整体调度。没有 Host compiler，NVCC 无法独立完成编译。

**Q2：PTX 与 SASS 有什么区别？**
PTX 是面向"虚拟 GPU"的**中间码**，跨架构代稳定、用无限虚拟寄存器，不被硬件直接执行；
SASS 是**某一代真实 GPU 的机器码**，用物理寄存器，绑定架构，是硬件真正执行的指令。
关系：PTX 经 ptxas（或运行时 JIT）编成 SASS。

**Q3：为什么发布程序可能同时包含 cubin 和 PTX？**
cubin（SASS）在目标卡上启动快、零 JIT，但只认对应架构；PTX 能在更新的卡上通过驱动
JIT 运行，保证**前向兼容**。两者一起打进 fat binary，就能"已知卡用现成 cubin、未来
卡回退 PTX"，兼顾启动速度与兼容性（见第 6 节方案 C）。

**Q4：`-lineinfo` 与 `-G` 有何区别？**
`-lineinfo` 只加源码行映射、几乎不影响优化，给 profiler 用；`-G` 生成完整调试信息并
关闭优化，给单步调试用，**不能拿来测性能**。

