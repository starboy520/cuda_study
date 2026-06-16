# 06 NVCC、PTX 与编译流程

## 1. `.cu` 文件包含两种世界

```text
Host C++ code
Device CUDA code
```

NVCC 是编译驱动，协调 Host compiler 与 NVIDIA Device toolchain。

## 2. 简化流程

```text
.cu
├─ Host 部分 -> g++/clang/MSVC -> Host object code
└─ Device 部分 -> CUDA front-end
                  -> PTX 或 GPU machine code
                  -> 嵌入 fat binary
最终链接成可执行文件
```

真实流程更复杂，但这张图足够解释常见选项。

## 3. PTX

PTX 是虚拟指令集/中间表示，不是最终某一代 GPU 的机器码。

```bash
nvcc --ptx file.cu -o file.ptx
```

PTX 可由 Driver JIT 为实际 GPU 编译。

为什么要有这么一个"虚拟"中间层，而不是直接编成 GPU 机器码？因为 NVIDIA 每一代
架构（Turing、Ampere、Hopper……）的**真实指令集 SASS 都不一样**，而且是会变的。
如果只编译成今天这块卡的 SASS，那这个二进制**换一代新卡就跑不了**。PTX 解决的正是
这个矛盾：它是一份**面向"虚拟 GPU"的稳定中间码**，编一次就能在未来的卡上，由
驱动在运行时即时（JIT）翻译成那代卡真正的 SASS。

```text
源码 .cu
  -> PTX（虚拟 ISA，跨代稳定）
       -> [运行时] Driver JIT
            -> SASS（这台机器这代卡的真实机器码）
```

代价是这次 JIT 翻译要花时间，发生在程序**首次启动**时（见第 6 节的 fat binary）。

## 4. Cubin 与 SASS

Cubin 包含针对具体 SM 架构的 device binary；SASS 是实际 GPU 指令表示。

查看可执行文件中的代码：

```bash
cuobjdump --list-elf executable
cuobjdump --dump-sass executable
```

也可使用 `nvdisasm` 分析 cubin。

## 5. `compute_XX` 与 `sm_XX`

```text
compute_75 = virtual architecture / PTX capability
sm_75      = real architecture / Turing T4 machine target
```

示例：

```bash
nvcc -arch=sm_75 program.cu
```

为 T4 生成对应代码。

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

## 7. 资源报告

```bash
nvcc -Xptxas=-v ...
```

可以看到：

- Registers。
- Shared memory。
- Spill stores/loads。
- Constant memory。

这些数字是 occupancy 和 local spill 分析的输入。

## 8. `-lineinfo` 与 `-G`

```text
-lineinfo  保留源码行映射，通常用于 profiler
-G         生成 Device debug 信息，关闭部分优化，适合调试
```

不要用 `-G` 的性能代表 release 性能。

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

## 10. Separate Compilation

跨多个 `.cu` 文件调用 Device 函数可能需要 relocatable device code：

```bash
nvcc -rdc=true ...
```

它带来 device link 阶段。初学 sample 单文件即可，工程卷再展开。

## 11. 练习

1. 分别编译 `sm_75` 和另一目标，比较产物。
2. 删除 `__forceinline__`，比较 PTX/SASS。
3. 添加局部数组，观察 registers 与 spill。

## 12. 面试题

- NVCC 是否完全替代 Host compiler？
- PTX 与 SASS 有什么区别？
- 为什么发布程序可能同时包含 cubin 和 PTX？
- `-lineinfo` 与 `-G` 有何区别？

