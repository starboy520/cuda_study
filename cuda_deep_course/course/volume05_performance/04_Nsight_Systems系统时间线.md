# 04 Nsight Systems 系统时间线

## 1. 它回答什么问题

Nsight Systems 看系统级时间线：

- CPU 在做什么？
- CUDA API 调用何时发生？
- Memcpy 和 kernel 是否重叠？
- GPU 是否空闲？
- 是否频繁同步？
- 多 stream 是否真正并发？

先用 Systems 判断“时间花在哪里”，再用 Compute 深挖某个 kernel。

## 2. 基本命令

```bash
nsys profile --trace=cuda,osrt \
  -o reports/transpose_systems \
  ./labs/03_memory_system/transpose/transpose 4096 4096
```

会生成 `.nsys-rep`。文件已存在时可加：

```bash
--force-overwrite=true
```

具体选项以本机 `nsys profile --help` 为准。

某些服务器会警告 CPU sampling、backtrace 或 context-switch tracing 不可用，
但 CUDA trace 仍可正常生成。使用：

```bash
nsys status --environment
```

检查环境能力，并在报告中注明缺失的数据类型。

## 2.5 命令选项详解与常用工作流

上面是最小命令，实战里几个选项几乎每次都要用，逐个说清楚它们干什么。

### 关键选项速查

| 选项 | 作用 | 什么时候用 |
|---|---|---|
| `--trace=cuda,nvtx,osrt` | 选择采集哪些事件源 | `cuda`=CUDA API/kernel/memcpy；`nvtx`=你的代码标记；`osrt`=OS 系统调用；图形加 `opengl/vulkan` |
| `-o <名字>` | 输出报告名（`.nsys-rep`）| 每次都给有意义的名字，便于前后对比 |
| `--force-overwrite=true` | 覆盖同名报告 | 反复测同一个程序时 |
| `--stats=true` | 跑完直接在终端打印汇总表 | 想快速看一眼、不开 GUI 时 |
| `--cuda-memory-usage=true` | 记录显存分配/占用 | 怀疑显存或想看分配时间线 |
| `--gpu-metrics-device=all` | 采集 GPU 硬件计数器（SM/显存利用率曲线）| 想在时间线上叠加 GPU 利用率（需权限/较新版本）|
| `--capture-range=cudaProfilerApi` | 只采集 `cudaProfilerStart/Stop` 之间的区间 | 程序很长、只想 profile 稳态那几次迭代 |
| `--delay=<秒>` / `--duration=<秒>` | 延迟开始 / 限制采集时长 | 跳过 warmup、控制报告大小 |
| `-w true`（`--show-output`）| 同时显示被测程序的 stdout | 调试时方便 |

### 工作流 A：一条命令看全貌（最常用）

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  -o reports/transpose_systems \
  ./labs/03_memory_system/transpose/transpose 4096 4096
```

`--stats=true` 会在终端直接打出 CUDA API / kernel / memcpy 的汇总，先看这个定方向，
再用 `nsys-ui reports/transpose_systems.nsys-rep` 打开时间线确认关系。

### 工作流 B：只 profile 稳态迭代（长程序必备）

长程序前面有初始化、warmup，全程采集既慢、报告又大。用 `cudaProfilerStart/Stop` 在
代码里圈出你真正关心的几次迭代：

```cpp
#include <cuda_profiler_api.h>
// ... warmup 若干次 ...
cudaProfilerStart();
for (int i = 0; i < 10; ++i) {     // 只采集这 10 次稳态迭代
    myKernel<<<grid, block>>>(...);
}
cudaProfilerStop();
```

```bash
nsys profile --capture-range=cudaProfilerApi \
  --trace=cuda,nvtx -o reports/steady ./my_program
```

### 工作流 C：命令行直接出汇总，不开 GUI

服务器上没有图形界面时，全程命令行也能拿到结论：

```bash
# 1) 采集
nsys profile --trace=cuda -o reports/run ./my_program

# 2) 列出可用报表
nsys stats --help-reports reports/run.nsys-rep

# 3) 出具体汇总（例：按 kernel 耗时排序）
nsys stats --report cuda_gpu_kern_sum reports/run.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum reports/run.nsys-rep   # memcpy 耗时
nsys stats --report cuda_api_sum         reports/run.nsys-rep   # API 调用耗时
```

报表名随版本变化，先用 `--help-reports` 列出当前可用的。需要二次分析时还可
`nsys export --type sqlite reports/run.nsys-rep` 导出成 SQLite 自己查询。

## 3. 先看 Summary

```bash
nsys stats reports/transpose_systems.nsys-rep
```

关注：

- CUDA API summary。
- Kernel summary。
- Memcpy summary。
- OS runtime waits。

Summary 找方向，时间线确认关系。

## 4. 时间线阅读顺序

1. 总运行时间。
2. CPU thread 和 CUDA API。
3. GPU context/stream。
4. Memcpy。
5. Kernel。
6. 同步和空洞。

## 5. 常见模式

### 频繁小 Kernel

大量 launch 间隙，单 kernel 很短。可能考虑：

- Kernel fusion。
- CUDA Graph。
- 增大 batch。

### 传输串行

H2D、kernel、D2H 完全排队。检查：

- 是否使用 pinned memory。
- 是否分块。
- 是否使用不同 stream。
- 设备是否支持相应并发。

### CPU 阻塞

频繁 `cudaDeviceSynchronize` 或同步 memcpy 让 CPU 等待。

### GPU 空洞

可能是 CPU 准备慢、依赖、同步或问题规模不足。

## 5.1 实战：一条时间线该怎么读

光知道"看什么"不够，下面用一个**典型的串行传输时间线**演示完整判读过程。假设你 profile
一个"拷入→计算→拷出"循环，时间线（简化）长这样：

```text
CPU:  [cudaMemcpy H2D]  [launch]  [cudaMemcpy D2H]  [cudaMemcpy H2D] ...
GPU:  ───[ H2D ]────────[kernel]──────[ D2H ]───────────[ H2D ]──────
                  ↑空洞            ↑空洞        ↑空洞
时间 ─────────────────────────────────────────────────────────────►
```

**第 1 步 看总时间和 GPU 占用率**：GPU 行有大片空白（空洞），说明 GPU 大部分时间在等，
利用率低。结论：瓶颈不在 kernel 本身，而在**数据流水线**。

**第 2 步 量化空洞**：从 Summary（`nsys stats`）看，假设 H2D=3ms、kernel=1ms、D2H=3ms，
一轮 7ms 里 kernel 只占 1ms——**计算只占 14%**，其余全是传输和等待。

**第 3 步 找空洞成因**：H2D / kernel / D2H 完全首尾相接、没有任何重叠，说明三者被**串行
排队**。再看 memcpy 是否用了 pinned memory（可分页内存无法异步重叠）。

**第 4 步 形成假设 + 改法**：

```text
症状：GPU 70%+ 时间空闲，传输与计算完全串行
假设：用了同步 memcpy + 可分页内存 + 单 stream，无法重叠
改法：pinned memory + cudaMemcpyAsync + 多 stream 分块流水线
预测：H2D(块i+1) 与 kernel(块i)、D2H(块i-1) 在时间线上重叠，空洞收窄
```

**第 5 步 改完复测**：理想的重叠时间线应该变成这样，总时间从 `3+1+3` 向 `max(3,1,3)` 靠拢：

```text
stream0:  [H2D0][k0][D2H0]
stream1:       [H2D1][k1][D2H1]
stream2:            [H2D2][k2][D2H2]   <- 三条流水线交错，GPU 不再大段空闲
```

这一整套"看空洞 → 量化 → 找成因 → 改 → 复测"就是 Systems 层面优化的标准动作。注意它
**只解决系统级编排问题**（重叠、同步、launch 太碎）；如果时间线显示 GPU 一直很忙、kernel
本身就慢，那才轮到 Nsight Compute 去深挖那个 kernel（第 05 章）。

## 6. NVTX

使用 NVTX 标记业务阶段：

```cpp
nvtxRangePushA("preprocess");
// work
nvtxRangePop();
```

命令加入：

```bash
--trace=cuda,nvtx,osrt
```

这样时间线能对应业务语义，而不是只有 kernel 名。

## 7. 实践

对 vector add 或 transpose 分别 profile：

1. 只运行一次。
2. 循环运行 100 次。
3. 每次循环都 `cudaDeviceSynchronize`。
4. 只在循环末尾同步。

比较 CPU/GPU 时间线差异。

## 8. 报告问题

每份 Systems 报告回答：

```text
端到端瓶颈是什么？
GPU 利用是否连续？
最大空洞在哪里？
哪些同步是必要的？
下一步应深入哪个 kernel？
```

## 9. 资料映射

- Nsight Systems User Guide：`nsys profile`、stats、timeline。
- Best Practices Guide：Application Profiling。
