# 01 CMake 与可复现构建

## 0. 先建立大局观：为什么不能一直 `nvcc xxx.cu`

前几卷你一直用 `nvcc vec_add.cu -o vec_add` 这种**手敲命令**编译。单文件学习够用，但工程化
后会立刻崩溃——想象一个真实项目：

```text
10 个 .cu + 20 个 .cpp，依赖 cuBLAS，要为 sm_75 和 sm_80 各编一份，
还要 Debug/Release 两种配置，团队 5 个人在不同机器上构建……
```

手敲 `nvcc` 在这里完全失控：参数记不全、改一个文件全量重编、换台机器路径全错。**CMake
就是来解决"用一份声明式配置，在任何机器上可复现地构建"这件事的**。

用一个类比：

```text
nvcc 手敲命令  ≈ 每次做菜临时想步骤，换个厨房就乱套
CMake          ≈ 一份标准化菜谱，任何厨房照着做都出一样的菜
```

## 0.1 术语速查表

| 术语 | 一句话定义 |
|---|---|
| **CMake** | 跨平台构建系统生成器：读 `CMakeLists.txt`，生成 Makefile/Ninja |
| **`CMakeLists.txt`** | 声明式构建配置文件（项目的"菜谱"）|
| **target** | 一个构建产物（可执行文件或库）|
| **`CMAKE_CUDA_ARCHITECTURES`** | 为哪些 GPU 架构编译（对应 `-arch`）|
| **out-of-source build** | 把构建产物放进独立 `build/` 目录，不污染源码 |
| **`find_package`** | 查找并链接外部依赖（如 CUDAToolkit、cuBLAS）|
| **可复现构建** | 同一份配置在任何机器上得到一致结果 |

## 1. 第一个 CUDA CMake 工程

现代 CMake（3.18+）把 CUDA 当**一等语言**，不用再手动调 `nvcc`。最小工程：

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)        # 3.18+ 才有成熟的 CUDA 原生支持
project(my_cuda LANGUAGES CXX CUDA)         # 声明用到 C++ 和 CUDA 两种语言

add_executable(vec_add vec_add.cu)          # 一个 target：从 vec_add.cu 编出可执行文件
set_target_properties(vec_add PROPERTIES
    CUDA_ARCHITECTURES "75")                # 为 sm_75（T4）编译
```

构建（**out-of-source**，产物全进 `build/`，不弄脏源码）：

```bash
cmake -S . -B build          # -S 源码目录, -B 构建目录：在 build/ 里生成 Makefile
cmake --build build          # 真正编译
./build/vec_add              # 运行
```

**为什么要 out-of-source？** 因为构建产物（`.o`、可执行文件、缓存）和源码混在一起会让
`git status` 一团乱、`clean` 困难。独立 `build/` 目录可以随时整个删掉重来，源码丝毫不动。

## 2. `LANGUAGES ... CUDA` 到底做了什么

`project(... LANGUAGES CXX CUDA)` 这一行让 CMake：

```text
1. 探测系统里的 nvcc 和 host 编译器（g++/clang），确认能配合工作
2. 把 .cu 文件自动识别为 CUDA 源码，用 nvcc 编译
3. 把 .cpp / .cc 用 host 编译器编译
4. 链接时自动处理 CUDA runtime（cudart）
```

也就是说，**你不再手动写 `-arch`、`-lcudart`**——CMake 根据 target 属性自动生成正确的
`nvcc` 命令行。回忆卷二/06：CMake 在背后帮你拼那串 `-gencode`。

## 3. 跨架构构建：`CUDA_ARCHITECTURES`

这是卷二/06 fat binary 的工程化落地。想同时支持 T4 和 A100、并给未来卡留 PTX：

```cmake
set_target_properties(vec_add PROPERTIES
    CUDA_ARCHITECTURES "75;80;80-virtual")
```

含义对照（回忆卷二/06 的 `-gencode`）：

```text
"75"          -> sm_75 cubin（T4 现成机器码）
"80"          -> sm_80 cubin（A100 现成机器码）
"80-virtual"  -> compute_80 PTX（给未来新卡 JIT 兜底）
```

CMake 把它翻译成对应的 `-gencode arch=compute_XX,code=...`。**好处**：架构列表写在一处，
不用在每条编译命令里重复，换部署目标只改这一行。

> 生产建议（呼应卷二/06 §6.2）：只列实际部署的架构 + 一份最高架构的 `-virtual` PTX 兜底，
> 别用"全架构"拖慢编译、撑大产物。

## 4. 链接外部库：以 cuBLAS 为例

Week4 要用 cuBLAS。CMake 用 `find_package` + `target_link_libraries` 干净地接入：

```cmake
find_package(CUDAToolkit REQUIRED)          # 找到 CUDA 工具包（含 cuBLAS 等）

add_executable(gemm gemm.cu)
set_target_properties(gemm PROPERTIES CUDA_ARCHITECTURES "75")
target_link_libraries(gemm PRIVATE CUDA::cublas)   # 链接 cuBLAS，不用手写 -lcublas
```

`CUDA::cublas` 是 CMake 提供的 **imported target**，它自动带上 cuBLAS 的头文件路径和库
路径。对比手敲的 `nvcc gemm.cu -lcublas -L/usr/local/cuda/lib64`——CMake 让依赖**可移植**：
换台机器只要 CUDA 装了，路径自动适配。

## 5. Debug / Release：别拿 Debug 测性能

回忆卷二/06 §8：`-G`（device debug）会关优化，性能不代表 release。CMake 用 build type 管理：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # 开优化，测性能用这个
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug     # 带调试信息，定位 bug 用
```

在 `CMakeLists.txt` 里可针对性加 flag：

```cmake
target_compile_options(vec_add PRIVATE
    $<$<CONFIG:Debug>:-G>                          # Debug 才加 -G（device 调试）
    $<$<CONFIG:Release>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)  # 给 profiler 留行号映射（卷五）
```

`$<...>` 是 CMake 的**生成器表达式**：`$<CONFIG:Debug>` 只在 Debug 配置生效，
`$<COMPILE_LANGUAGE:CUDA>` 只作用于 `.cu` 文件（不会误加到 `.cpp` 上）。

> 一句话纪律：**性能测试永远用 Release 构建**。Debug 的 `-G` 关了优化，数字没有参考价值
> （卷五/01 也强调过）。

## 6. 一个完整的工程模板

把上面拼成一个能直接用的模板：

```cmake
cmake_minimum_required(VERSION 3.18)
project(cuda_project LANGUAGES CXX CUDA)

# 全局默认：没指定就用 Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 全局默认架构（target 可覆盖）
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit REQUIRED)

# 一个库 target：把可复用的 kernel 编成库
add_library(ops STATIC src/gemm.cu src/reduce.cu)
target_include_directories(ops PUBLIC include)

# 主程序链接这个库
add_executable(app src/main.cpp)
target_link_libraries(app PRIVATE ops CUDA::cudart)

# 测试程序
enable_testing()
add_executable(test_gemm tests/test_gemm.cu)
target_link_libraries(test_gemm PRIVATE ops)
add_test(NAME gemm COMMAND test_gemm)        # 之后 ctest 一键跑测试（见第 03 章）
```

这个结构把"可复用算子（库）/ 主程序 / 测试"分开，是真实项目的雏形。

## 7. 可复现性：让"在我机器上能跑"不再是借口

可复现构建的核心是**消除隐式依赖**，把一切写进配置：

```text
✅ 固定 CMake 最低版本（cmake_minimum_required）
✅ 显式声明架构（CUDA_ARCHITECTURES），不靠 nvcc 默认值
✅ 用 find_package 找依赖，不硬编码 /usr/local/cuda 路径
✅ build type 显式指定，不依赖"上次 cmake 留下的缓存"
✅ 把 CMakeLists.txt 纳入版本控制，build/ 加进 .gitignore
```

> 反例：在脚本里硬编码 `nvcc -arch=sm_75 -I/home/me/cuda/include ...`——换个人、换台机器
> 立刻失败。CMake 的价值就是把这些"机器相关"的细节自动化、可移植化。

## 8. 实践

1. 把你 Week1 的 `vec_add.cu` 改成 CMake 工程，`cmake -S . -B build && cmake --build build` 跑通。
2. 加上 `CUDA_ARCHITECTURES "75;80-virtual"`，用 `cuobjdump --list-elf` 确认产物里有 sm_75
   cubin + compute_80 PTX（呼应卷二/06）。
3. 分别用 `-DCMAKE_BUILD_TYPE=Debug` 和 `Release` 构建，对比 kernel 耗时，体会为什么不能用
   Debug 测性能。
4. 把一个 kernel 抽成 `add_library`，让主程序和测试程序都链接它。

## 9. 面试题（附参考答案）

**Q1：为什么用 CMake 而不是直接 nvcc 或手写 Makefile？**
CMake 是声明式、跨平台、可复现的构建生成器：一份 `CMakeLists.txt` 能在不同机器/编译器上
生成正确构建，自动处理架构、依赖、配置；手敲 nvcc 不可维护，手写 Makefile 不跨平台且要手动
管理 CUDA 细节。

**Q2：`CUDA_ARCHITECTURES "75;80;80-virtual"` 各是什么？**
`75`/`80` 生成 sm_75/sm_80 的 cubin（现成 SASS）；`80-virtual` 生成 compute_80 的 PTX（留给
未来新卡 JIT）。即卷二/06 的 fat binary 方案 C 的 CMake 写法。

**Q3：out-of-source build 是什么，为什么推荐？**
把构建产物放进独立 `build/` 目录、不混进源码。好处：源码树干净、`build/` 可整个删除重来、
不污染版本控制。

**Q4：为什么性能测试必须用 Release？**
Debug 常带 `-G`（device 调试信息）会关闭优化，性能可能差数倍，不代表真实 release 速度
（卷二/06、卷五/01）。

**Q5：怎么让构建"可复现"？**
固定 CMake 最低版本、显式声明架构和 build type、用 `find_package` 找依赖而非硬编码路径、把
`CMakeLists.txt` 纳入版本控制、`build/` 加 `.gitignore`——消除一切隐式/机器相关依赖。

## 10. 资料映射

- CMake 官方文档：CUDA language support、`CUDAToolkit` find module。
- CUDA C++ Best Practices Guide：Building for Maximum Compatibility。
- 配套：[卷二第 06 章 NVCC、PTX 与编译流程](../volume02_programming_model/06_NVCC_PTX与编译流程.md)。
