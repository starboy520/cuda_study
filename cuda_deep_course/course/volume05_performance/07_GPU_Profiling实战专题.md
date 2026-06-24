# 07 GPU Profiling 实战专题

> 本章是把前面 01~06 的工具和指标**串成一套可照着做的端到端流程**。第 04/05 章分别讲透了
> Nsight Systems 和 Nsight Compute 的细节，这里不重复，只回答一个问题：**拿到一个 GPU 程序，
> 我从零到给出有据可依的优化结论，每一步具体敲什么命令、看什么、怎么决策。**

## 1. 工具地图：什么活用什么

```text
nsys (Nsight Systems)   系统级时间线   找"哪段慢/谁在等谁/有没有重叠"   <- 永远先用它
ncu  (Nsight Compute)   单 kernel 微观  找"这个 kernel 为什么慢"          <- 锁定热点后才用
compute-sanitizer       正确性检查      越界/race/未初始化/同步误用       <- profile 前先过
nvidia-smi              粗粒度监控      显存/功耗/温度/利用率              <- 随手看一眼
CUDA Event / NVTX       代码内埋点      精确测时 / 时间线打业务标记        <- 长期回归
-Xptxas=-v / cuobjdump  编译期/反汇编   寄存器/shared 用量 / SASS 指令     <- 解释 occupancy
```

一句话分工：**`nsys` 找热点，`ncu` 挖热点，sanitizer 保正确，smi/event 做监控。**

## 2. 黄金顺序：别跳步

这个顺序是本专题的核心纪律，跳步会让你在错误的地方浪费时间：

```text
1. 建立正确基线   CPU reference + 边界测试
2. 查正确性       compute-sanitizer（越界/race 的 kernel 不值得 profile）
3. 可靠测时       CUDA Event / CPU timer + warmup + 多次中位数
4. 系统级定位     nsys：时间花在哪？GPU 有没有空洞？传输和计算重叠没？
5. 决定方向       是系统编排问题（重叠/同步/launch 太碎）还是某个 kernel 慢？
   ├─ 系统问题 -> 回到卷七：stream 重叠 / CUDA Graph / pinned memory
   └─ kernel 慢 -> 进第 6 步
6. kernel 级深挖   ncu：Speed of Light 分流 -> 对应 section -> 定位源码行
7. 单一假设 + 改   一次只改一处，预测"哪个指标会变"
8. 复测验证        指标按预期变 + 时间下降 = 假设成立；否则推翻重来
9. 回归            存报告、记 before/after，确保没退化
```

> 反模式（卷五 06 章列过，这里再强调）：没基线就优化、只测一次、把 profiler 时间当
> benchmark、看到一个指标高就乱改、一次改一堆无法归因。

## 3. 环境与权限：先把路铺平

开工前确认工具在、权限够，免得跑到一半报错。

```bash
nsys --version            # Nsight Systems 在不在
ncu  --version            # Nsight Compute 在不在
compute-sanitizer --version
nvidia-smi                # 看 GPU 型号、驱动、显存、是否有别人在用

nsys status --environment # 看哪些采集能力可用（CPU sampling/backtrace 可能受限）
```

**常见权限坑**：`ncu` 采硬件计数器需要 GPU performance counter 权限，否则报

```text
ERR_NVGPUCTRPERM  The user does not have permission to access NVIDIA GPU Performance Counters
```

这不是程序错了，是当前用户无权采计数器。解决要由管理员按 NVIDIA 的 performance-counter
权限说明开放——**不要为绕过限制擅自改生产服务器配置**。`nsys` 的基础 CUDA trace 通常不
需要这个权限，所以即便 `ncu` 受限，`nsys` 那一步一般还能做。

## 4. 决策流程图：profile 时脑子里跑的那张图

```text
                    ┌─────────────────────────┐
                    │ 跑 nsys，看 GPU 时间线   │
                    └───────────┬─────────────┘
                                │
              GPU 大段空闲？ ────┴──── GPU 一直很忙？
                    │                      │
        ┌───────────▼──────────┐   ┌───────▼────────────┐
        │ 系统编排问题          │   │ 某个 kernel 是热点  │
        │ 传输/计算没重叠?      │   │ 跑 ncu 深挖它       │
        │ 同步太多? launch 太碎?│   └───────┬────────────┘
        └───────────┬──────────┘           │
                    │                ┌──────▼─────────────────┐
        卷七方案:    │                │ Speed of Light 分流      │
        pinned+async │                ├─────────┬──────┬───────┤
        多 stream    │            Memory高    Compute高   都低
        CUDA Graph   │                │         │        │
        kernel fusion│         访存不合并?  指令效率?  warp不够/
                     │         看 Mem        看指令     延迟没隐藏
                     │         Workload      组合       看 Warp State
                     ▼                                  + occupancy
              改完回到第 2 步复测
```

## 5. 完整案例：从零 profile 一次 transpose

把前面所有命令串成一次真实操作，照着敲即可。

### 5.1 先保正确

```bash
# 非方阵 + 非整除规模最容易抓边界 bug
compute-sanitizer --tool memcheck \
  ./labs/03_memory_system/transpose/transpose 1003 769
# 期望：========= ERROR SUMMARY: 0 errors
```

### 5.2 系统级看全貌

```bash
nsys profile --trace=cuda,nvtx --stats=true --force-overwrite=true \
  -o reports/transpose_sys \
  ./labs/03_memory_system/transpose/transpose 4096 4096
```

看终端汇总：如果 kernel 本身占了绝大部分 GPU 时间、没有明显空洞，说明**瓶颈在 kernel
内部**，转 5.3 用 ncu 深挖。如果时间线全是 H2D/D2H 串行、GPU 大片空闲，那是**系统编排
问题**，去卷七做重叠（本案例聚焦 kernel，假设是前者）。

### 5.3 kernel 级深挖

```bash
# 先 basic 快速分流
ncu --kernel-name regex:transpose --launch-count 1 --set basic \
  ./labs/03_memory_system/transpose/transpose 4096 4096
```

读 Speed of Light：

```text
Memory  [%]  80   <- 接近上限
Compute [%]  5    <- 极低
=> memory-bound，去看 Memory Workload
```

```bash
# 只抓关键访存指标，快
ncu --kernel-name regex:transpose --launch-count 1 --csv \
  --metrics gld_efficiency,gst_efficiency \
  ./labs/03_memory_system/transpose/transpose 4096 4096
```

```text
gld_efficiency ~100%   读是合并的 ✅
gst_efficiency ~12%    写不合并 ❌  <- 找到了：global store 不合并
```

### 5.4 单一假设 + 改 + 复测

```text
假设：naive 转置的 global store 跨步不合并，效率约 1/32
改法：shared memory tile（+1 padding 消 bank conflict），把跨步写换成合并写
预测：gst_efficiency 从 ~12% 升到 ~100%，有效 GB/s 显著上升，时间下降
```

改完重复 5.1~5.3：

```text
gst_efficiency ~100%   假设被验证 ✅
有效带宽逼近 T4 ~320 GB/s 量级
memcheck/racecheck 全过
```

这条"症状→假设→指标→改→复测"闭环，就是面试时最有说服力的讲法。

## 6. 命令速查表（贴墙版）

```bash
# ── 监控 ──────────────────────────────────────────────
nvidia-smi                              # GPU 概览
nvidia-smi dmon                         # 利用率/功耗实时刷新

# ── 正确性（profile 前必过）────────────────────────────
compute-sanitizer --tool memcheck  ./prog args   # 越界/非法访问
compute-sanitizer --tool racecheck ./prog args   # shared race
compute-sanitizer --tool initcheck ./prog args   # 未初始化读
compute-sanitizer --tool synccheck ./prog args   # 同步误用

# ── 系统级（先用）─────────────────────────────────────
nsys profile --trace=cuda,nvtx --stats=true -o rep ./prog args
nsys stats --report cuda_gpu_kern_sum rep.nsys-rep   # 按 kernel 排序
nsys-ui rep.nsys-rep                                 # 时间线 GUI

# ── kernel 级（后用）──────────────────────────────────
ncu --kernel-name regex:NAME --launch-count 1 --set basic ./prog args   # 分流
ncu --kernel-name regex:NAME --set full -f -o rep ./prog args           # 存档
ncu --kernel-name regex:NAME --csv --metrics gld_efficiency,gst_efficiency ./prog args
ncu-ui rep.ncu-rep                                                       # GUI

# ── 编译期信息 ────────────────────────────────────────
nvcc -Xptxas=-v -lineinfo ...           # 每 kernel 寄存器/shared 用量 + 源码关联
cuobjdump -sass ./prog                  # 看真实 SASS 指令
```

## 7. 本章自检

- [ ] 能说清 `nsys` 与 `ncu` 的分工，以及为什么先 system 后 kernel。
- [ ] 能独立完成一次 transpose 的完整 profile（正确性→系统→kernel→验证）。
- [ ] 看到 Speed of Light 能分流出 memory / compute / latency-bound。
- [ ] 能针对一个假设只抓 1~2 个指标做 before/after 对比。
- [ ] 遇到 `ERR_NVGPUCTRPERM` 知道是权限问题、知道找谁、知道 `nsys` 仍可用。

## 8. 资料映射

- Nsight Systems User Guide：`nsys profile` / `nsys stats` / timeline。
- Nsight Compute CLI 与 Profiling Guide：sections、metrics、replay。
- Compute Sanitizer Documentation。
- Best Practices Guide：Application Profiling、Performance Metrics。
