# CUDA 面试完整准备指南

> 一份独立、自包含的 CUDA 面试准备文档。覆盖从基础到进阶的**全部高频考点**，包括你可能还没
> 系统学的主题。不含 HPC / 多 GPU（按需另备）。
>
> 用法：先看 §1 了解"考什么、怎么考"，再用 §2 的知识地图查漏补缺，§3-§6 是题库（概念/手写/
> 性能/系统设计），§7 是进阶主题，§8 是复习计划，§9 是速查表。

---

## 1. CUDA 面试全景:考什么、怎么考

### 1.1 岗位类型与侧重

| 岗位 | 侧重 | 必考 |
|---|---|---|
| **通用 CUDA / GPU 工程师** | 编程模型 + 内存 + 基础优化 | 合并访问、shared memory、reduction |
| **算子开发工程师**（最热）| GEMM、卷积、softmax、融合、性能 | GEMM tiling、数值稳定、profiler |
| **性能优化工程师** | profiling、Roofline、瓶颈定位 | Nsight、occupancy、bound 分析 |
| **推理/框架工程师** | 算子 + 部署 + 框架集成 | 融合、量化、TensorRT、PyTorch extension |

### 1.2 面试形式(通常 3-5 轮)

```text
1. 电话/视频初筛：概念题为主(SM/warp/内存/同步)
2. 技术面 1：手写 kernel(reduction/transpose/GEMM)+ 讲优化
3. 技术面 2：性能分析(给场景定位瓶颈)+ profiler 经验
4. 系统设计：多流/显存预算/吞吐目标(高级岗)
5. 项目深挖：你做过的优化,问到底"为什么快"
```

### 1.3 考点权重(经验估计)

```text
内存系统(合并/shared/bank)     ★★★★★  必考,权重最高
执行模型(warp/SM/occupancy)    ★★★★★  必考
经典算法(reduction/scan)       ★★★★☆  手写高频
GEMM 及优化                    ★★★★☆  算子岗天花板
性能分析(Roofline/profiler)    ★★★★☆  区分度高
同步与正确性(race/atomic)      ★★★☆☆  常考
编译/部署(PTX/CC/JIT)          ★★★☆☆  加分项
异步/Stream/Graph              ★★★☆☆  中高级岗
算子(softmax/layernorm/融合)   ★★★★☆  算子岗必考
Tensor Core/WMMA               ★★☆☆☆  进阶/AI 岗
```

### 1.4 面试官在评估什么

```text
不只是"会不会写",而是:
  ✓ 能不能从硬件解释"为什么"(不是背答案)
  ✓ 能不能量化(快了几倍、哪个指标证明)
  ✓ 能不能讲清思维过程(问题→假设→改法→验证)
  ✓ 工程素养(边界、正确性、可测试)
  ✓ 知道边界(什么时候调库、什么时候手写)
```

---

## 2. 完整知识地图(查漏补缺用)

下面是 CUDA 面试需要覆盖的**全部知识点**。标 🔥 = 高频必会,⭐ = 进阶加分。

### 2.1 GPU 架构与执行模型 🔥

```text
□ CPU vs GPU 设计哲学(低延迟 vs 高吞吐)
□ GPU 硬件层次:die → GPC → TPC → SM → CUDA Core
□ SM 内部:CUDA Core / Tensor Core / SFU / LSU / warp scheduler / 寄存器文件 / shared
□ Grid → Block → Thread 软件层次,与硬件的映射
□ Warp = 32 线程,SIMT 执行
□ 为什么 block 不能跨 SM(资源驻留在 SM)
□ 一个 SM 可驻留多个 block
□ Warp scheduler / scoreboard / 指令发射
□ 延迟隐藏:为什么要大量 warp
□ latency vs throughput
□ 独立线程调度(Volta+):不能依赖 warp 锁步 ⭐
```

### 2.2 编程模型 🔥

```text
□ __global__ / __device__ / __host__ / __host__ __device__ 修饰符
□ 为什么 kernel 返回 void(异步 + 多线程)
□ <<<grid, block, sharedBytes, stream>>> 四个参数
□ 1D/2D/3D 索引计算:blockIdx*blockDim+threadIdx
□ grid-stride loop(处理超大输入)
□ dim3,边界判断,非方阵/非整除
□ cudaMalloc/cudaMemcpy/cudaFree
□ 同步 vs 异步,cudaDeviceSynchronize
□ 错误模型:launch error(cudaGetLastError) vs execution error(同步后)
□ 粘性错误
```

### 2.3 内存系统 🔥🔥(权重最高)

```text
□ 内存空间:register/local/shared/global/constant/texture
□ local memory 不在片上(其实在 global)⭐ 易错点
□ 合并访问(coalescing):一个 warp 访问连续对齐地址
□ 跨步访问的代价(利用率暴跌到 1/N)
□ 内存事务、cache line、对齐
□ 行主序 vs 列主序、pitch、cudaMallocPitch
□ AoS vs SoA(只用部分字段时 SoA 合并更好)
□ shared memory 的三种价值:复用/重排/协作
□ bank conflict:32 个 bank,同 bank 不同地址串行
□ padding 消 bank conflict([32][33])
□ broadcast(同地址)vs conflict(同 bank 不同地址)
□ 静态 vs 动态 shared memory
□ L1/L2 cache,数据复用
□ constant memory 的"同址广播"
□ texture/read-only 路径 ⭐
□ pinned memory(为什么快、为什么是异步重叠前提)
□ mapped/zero-copy ⭐
□ Unified Memory(按需页迁移、prefetch/advise)⭐
□ 向量化访问(float4)⭐
```

### 2.4 同步与正确性 🔥

```text
□ data race 的三个构成条件
□ __syncthreads():到齐 + 内存可见
□ divergent barrier 死锁(不能放进 thread-dependent 分支)
□ __syncwarp(mask)
□ memory fence(__threadfence*),fence ≠ barrier ⭐
□ fence + flag 的 producer-consumer 协议 ⭐
□ atomic 操作(atomicAdd/CAS/Exch...)
□ atomic 竞争(contention)与串行化
□ 分层聚合(privatization)降竞争
□ atomic 不等于 barrier
□ 不同 block 如何全局同步(多 kernel / cooperative launch)
□ cooperative groups ⭐
```

### 2.5 Warp 级原语 🔥(算法岗常考)

```text
□ lane = warp 内编号(0-31)
□ __shfl_sync / __shfl_down_sync / __shfl_up_sync / __shfl_xor_sync
□ 为什么要 _sync 后缀和 mask(独立线程调度)
□ warp shuffle 做归约(免 shared、免 barrier)
□ vote:__ballot_sync / __any_sync / __all_sync
□ __match_sync ⭐
□ warp aggregation(histogram 优化)⭐
```

### 2.6 经典并行算法 🔥(手写高频)

```text
□ Reduction:naive→shared树→每线程多元素→warp shuffle→多阶段
□ sequential addressing(避免 divergence)
□ Scan:inclusive vs exclusive
□ Hillis-Steele(O(n log n))vs Blelloch(O(n))⭐
□ 多 block scan(三趟:块内/块和/加回)⭐
□ Histogram:global atomic → privatization → warp aggregation
□ Stream compaction(标记→scan→搬运)⭐
□ Convolution/stencil:halo、ping-pong ⭐
□ 浮点归约的非确定性
```

### 2.7 性能优化 🔥🔥

```text
□ APOD(Assess/Parallelize/Optimize/Deploy)
□ 正确 benchmark:warmup、重复、中位数、Event 计时
□ CUDA Event 正确计时
□ latency/throughput/GFLOPS/有效带宽/speedup
□ Roofline 模型、arithmetic intensity、拐点
□ memory-bound vs compute-bound 判断
□ occupancy:定义、限制因素(寄存器/shared/block)
□ occupancy 高不一定快、低不一定慢
□ warp divergence(warp 内、串行、最坏 32x)
□ register pressure 与 spill
□ Amdahl / Gustafson、strong/weak scaling
□ Nsight Systems(系统时间线)vs Nsight Compute(单 kernel)
□ Speed of Light、Memory Workload、Warp State
□ Compute Sanitizer(memcheck/racecheck/synccheck/initcheck)
□ stall reasons(Long Scoreboard/Barrier/MIO...)⭐
```

### 2.8 GEMM 与算子 🔥(算子岗)

```text
□ GEMM 数学:C[M×N]=A[M×K]×B[K×N],约 2MNK FLOP
□ naive GEMM 为什么 memory-bound(无复用、AI 低)
□ shared-memory tiling(提升 AI)
□ register tiling(每线程多输出、ILP)⭐
□ 向量化加载、双缓冲、软件流水 ⭐
□ Tensor Core / WMMA ⭐
□ Softmax 数值稳定(减最大值防溢出)⭐
□ 多级归约(max + sum)
□ LayerNorm / RMSNorm ⭐
□ kernel 融合(减少 global 往返)⭐
□ im2col 卷积 ⭐
□ cuBLAS / cuDNN / CUTLASS 的边界
```

### 2.9 异步与 Stream ⭐(中高级岗)

```text
□ stream 语义:同 stream 顺序、跨 stream 并发
□ 默认 stream 的坑
□ cudaMemcpyAsync + pinned 重叠传输与计算
□ 多流分块流水线
□ event:计时与依赖
□ 并发 kernel、stream 优先级
□ CUDA Graph(capture/instantiate/replay,消除 launch 开销)⭐
□ cudaMallocAsync(stream-ordered allocator)⭐
```

### 2.10 编译与部署 ⭐(加分项)

```text
□ NVCC 编译流程(host/device 分离)
□ PTX(虚拟中间码)vs SASS(真实机器码)
□ cubin、fat binary
□ compute_XX vs sm_XX
□ -arch / -gencode / arch= / code=
□ PTX JIT、前向兼容规则
□ Compute Capability
□ __CUDA_ARCH__(编译期、多次编译、host 未定义)⭐
□ -Xptxas=-v(看寄存器/spill)
□ -lineinfo vs -G
□ separate compilation(-rdc=true)⭐
□ no kernel image 报错
```

### 2.11 工程化 ⭐

```text
□ CUDA_CHECK 错误包装宏
□ RAII 管理 CUDA 资源(防泄漏、double free)
□ CPU reference + 多规模 + 边界测试
□ 浮点用容差比较(不用 ==)
□ 性能基线与回归测试
□ CMake CUDA 工程、CUDA_ARCHITECTURES
```

---

## 3. 高频概念题库(50 题,带答案要点)

### 执行模型

**1. warp 是什么?为什么是 32?**
GPU 基本执行单位,32 个线程以 SIMT 执行同一指令。32 是硬件设计:warp scheduler 一拍发射的
指令正好驱动 32 条 lane。

**2. SM 和 block 是硬件还是逻辑?**
SM 硬件(物理执行单元),block 逻辑(程序员定义的协作单元,由调度器映射到 SM)。

**3. 为什么 block 不能跨 SM?**
block 的寄存器、shared memory、barrier 状态物理驻留在一个 SM,跨 SM 无法共享。

**4. 为什么 GPU 需要大量线程?**
延迟隐藏:某 warp 等内存(几百拍)时,scheduler 切到其他就绪 warp,让执行单元不空转。

**5. thread 等于 CUDA Core 吗?**
不。thread 是带状态的逻辑实例,CUDA Core 是物理 ALU,驻留 thread 远多于 core,时分复用。

**6. SIMT 和 SIMD 区别?**
SIMD 一条指令处理一个向量;SIMT 是多个独立线程执行同一指令,每线程有自己寄存器和分支能力。

**7. latency 和 throughput 区别?GPU 优化哪个?**
latency 单操作完成时间,throughput 单位时间完成总数。GPU 优化吞吐(靠海量并行 + 延迟隐藏)。

**8. occupancy 是什么?越高越好吗?**
驻留并发 warp 数 / 上限。不是越高越好:它为隐藏延迟,但带宽墙/充足 ILP 时低 occupancy 也快。

### 内存

**9. 什么是合并访问?**
一个 warp 的 32 条 lane 访问连续对齐地址,硬件用最少事务搬完。

**10. 跨步访问为什么慢?**
每条 lane 落到不同事务,搬运远多于需要的数据。stride=32 时利用率可低到 12.5%。

**11. CUDA 有哪些内存空间?**
register(线程私有)、local(私有但在 global)、shared(block 共享、片上)、global(全局、大慢)、
constant(只读广播)、texture(只读 cache)。

**12. local memory 在片上吗?**
不在。物理在 global,因 spill、动态索引数组、过大私有对象产生,是性能警告。

**13. shared memory 为什么快?**
SM 内片上 SRAM(~几十拍),global 在片外显存(~几百拍),快一个数量级。shared 和 L1 是同一块 SRAM。

**14. 什么是 bank conflict?怎么解决?**
shared 分 32 bank,多 lane 访问同 bank 不同地址被迫串行。padding(如 [32][33])错开 bank 解决。

**15. 同地址访问会冲突吗?**
不会,触发 broadcast(一次广播)。只有同 bank 不同地址才冲突。

**16. AoS 和 SoA 怎么选?**
只用部分字段时 SoA(访问连续合并);整体用结构时 AoS 也可。GPU 默认优先 SoA。

**17. pinned memory 为什么快?**
页锁定不被换出,DMA 直接搬,省驱动暂存拷贝;且是 cudaMemcpyAsync 真异步重叠的前提。

**18. constant memory 什么时候快?**
一个 warp 所有 lane 读同一地址时一次广播最快;读散地址退化。适合小型只读众线程共用参数。

**19. Unified Memory 自动变快吗?**
不。提升可编程性,数据靠按需页迁移,CPU/GPU 争用会频繁迁移变慢,要 prefetch/advise 管理。

**20. 什么是 zero-copy?何时用?**
GPU 直接访问映射的 host 内存,省 memcpy。但每次访问跨 PCIe,只适合一次性/零散访问或集成 GPU。

### 同步

**21. __syncthreads 只是"等一下"吗?**
不止。到齐 + 内存可见性(barrier 前的写对 barrier 后全 block 可见,建立 happens-before)。

**22. 为什么不能把 __syncthreads 放进 if 分支?**
它要求全 block 都到达。部分线程进不去会死等 → 死锁。条件要全 block 一致(如 blockIdx)。

**23. fence 和 barrier 区别?**
barrier 让一群线程到齐 + 可见(等待);fence 只保证本线程写入按序可见(不让任何线程等待)。

**24. atomic 等于 barrier 吗?**
不。atomic 只保护单地址读改写不可分割,不让线程等待、不保证到齐。

**25. atomic 为什么会慢?怎么缓解?**
百万线程争一个地址被串行化。分层聚合:寄存器局部攒 → block 内 shared atomic → 少量 global atomic。

**26. 不同 block 怎么全局同步?**
普通 kernel 没有全 grid barrier。最常用拆成多个 kernel(kernel 边界=同步墙);或 cooperative launch。

**27. 为什么不能依赖 warp 天然锁步?**
Volta+ 独立线程调度让 lane 可能不锁步,依赖它会在新架构间歇出错。要显式 __syncwarp/_sync。

### 算法

**28. reduction 怎么从 naive 优化到最优?**
v1 global atomic(竞争)→ v2 shared 树(竞争入 block)→ v3 每线程加2个(砍第一轮)→ v4 warp shuffle
收尾(免 barrier)→ 多阶段跨 block。每步单一假设 + 量化收益。

**29. warp shuffle 为什么比 shared 交换轻量?**
lane 间直接读寄存器,不经 shared、不需 block barrier(warp 天然同步)。

**30. shuffle mask 代表什么?**
声明哪些 lane 参与本次操作(32 位)。独立线程调度下必须如实反映存活 lane,整组在用 0xffffffff。

**31. scan 的 inclusive 和 exclusive 区别?**
inclusive 含自己([a,a+b,a+b+c]),exclusive 不含([0,a,a+b])。exclusive 输出=每个元素的写入位置。

**32. Hillis-Steele 和 Blelloch 哪个快?**
深度都 O(log n),但工作量 O(n log n) vs O(n)。大数组 Blelloch 快(访存少);H-S 简单适合小 tile。

**33. histogram privatization 为什么有效?**
每 block 先在 shared 私有直方图 atomic(范围小、快),最后每 bin 只回写 global 一次,降 global atomic。

**34. stream compaction 怎么做?**
标记(0/1)→ exclusive scan 求位置 → 按位置搬运。scan 结果就是保留元素的目标下标。

### 性能

**35. 怎么判断 memory-bound 还是 compute-bound?**
ncu 看 Speed of Light,memory/compute 哪个接近 100%;或 Roofline 看 AI 在拐点哪侧。

**36. 有效带宽怎么算?**
算法必须读写的字节 / 时间。如 transpose 读 N 写 N = 2N 字节,GB/s = 2N/time/1e9。

**37. arithmetic intensity 是什么?**
每搬 1 字节做多少 FLOP(FLOP/byte),Roofline 横轴。低则 memory-bound。

**38. Roofline 的局限?**
只给上限,不保证达到;不建模 latency、occupancy 不足、divergence、cache 波动。

**39. warp divergence 发生在哪、代价?**
warp 内(32 lane 走不同路径)。路径串行执行、被关 lane 空转,最坏 32 路慢约 32 倍。

**40. register pressure 怎么影响性能?**
用多了每 SM 驻留 warp 少(occupancy 降);强压又 spill 到 local 反而慢。要权衡。

**41. 为什么测性能要 warmup?**
首次含 context 初始化、JIT、首次内存触碰,会污染稳态。还有频率/温度进入稳态。

**42. profiler 时间为什么和普通运行不同?**
profiler 插桩、可能重放 kernel、串行化,有 overhead。不能当 benchmark 数字。

### 编译/部署

**43. PTX 和 SASS 区别?**
PTX 跨架构虚拟中间码(无限虚拟寄存器),SASS 某代真实 GPU 机器码(物理寄存器、绑定架构)。

**44. compute_75 和 sm_75 区别?**
compute_ 虚拟架构(决定 PTX),sm_ 真实架构(决定 SASS)。-arch=sm_75 隐含先 PTX 再 SASS。

**45. 为什么发布程序同时含 cubin 和 PTX?**
cubin 目标卡启动快但只认那代;PTX 在更新卡 JIT 前向兼容。fat binary 打包两者兼顾。

**46. no kernel image 报错怎么回事?**
编译架构和运行 GPU 不匹配,没有可用 cubin 也没有可 JIT 的 PTX。补 -gencode 或留 PTX。

### 工程

**47. CUDA API 出错为什么易被忽略?怎么防?**
不抛异常不崩溃,只返回错误码。用 CUDA_CHECK 宏包每个调用,出错打印文件:行并退出。

**48. kernel 错误为什么查两次?**
异步:cudaGetLastError 抓 launch error(配置非法),cudaDeviceSynchronize 抓 execution error(越界)。

**49. RAII 怎么解决显存泄漏?**
构造 cudaMalloc、析构 cudaFree,对象离开作用域自动释放(正常/异常都释放),根除泄漏。

**50. 浮点为什么不能用 == 比较?**
不满足结合律,并行求和顺序不同导致末位差异。用容差 abs(a-b)<=absTol+relTol*abs(b)。

---

## 4. 手写 Kernel 题(必须能默写)

> 评分点永远是:**边界判断、__syncthreads 位置、合并访问**。写完口头过这三项。

### 4.1 Vector Add(热身)

```cpp
__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];          // 边界判断
}
// vecAdd<<<(n+255)/256, 256>>>(a, b, c, n);
```

### 4.2 Reduction(最高频,默写到 warp-shuffle)

```cpp
__global__ void reduce(const float* in, float* out, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;       // 每线程先加 2 个
    float sum = 0;
    if (i < n)              sum += in[i];
    if (i + blockDim.x < n) sum += in[i + blockDim.x];
    s[tid] = sum;
    __syncthreads();
    for (int off = blockDim.x/2; off >= 32; off >>= 1) {   // shared 树降到 32
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }
    if (tid < 32) {                                  // 最后一个 warp 用 shuffle
        float v = s[tid];
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) out[blockIdx.x] = v;
    }
}
```

### 4.3 Tiled Transpose(shared + bank conflict)

```cpp
__global__ void transpose(float* out, const float* in, int w, int h) {
    __shared__ float tile[32][33];                   // +1 消 bank conflict
    int x = blockIdx.x*32 + threadIdx.x;
    int y = blockIdx.y*32 + threadIdx.y;
    if (x < w && y < h) tile[threadIdx.y][threadIdx.x] = in[y*w + x];  // 合并读
    __syncthreads();
    int tx = blockIdx.y*32 + threadIdx.x;
    int ty = blockIdx.x*32 + threadIdx.y;
    if (tx < h && ty < w) out[ty*h + tx] = tile[threadIdx.x][threadIdx.y];  // 合并写
}
```

### 4.4 Tiled GEMM(算子岗天花板)

```cpp
#define T 16
__global__ void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[T][T], Bs[T][T];
    int row = blockIdx.y*T + threadIdx.y;
    int col = blockIdx.x*T + threadIdx.x;
    float acc = 0;
    for (int k0 = 0; k0 < K; k0 += T) {
        As[threadIdx.y][threadIdx.x] =
            (row < M && k0+threadIdx.x < K) ? A[row*K + k0+threadIdx.x] : 0;
        Bs[threadIdx.y][threadIdx.x] =
            (k0+threadIdx.y < K && col < N) ? B[(k0+threadIdx.y)*N + col] : 0;
        __syncthreads();
        for (int k = 0; k < T; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                             // 算完再加载下一个 tile
    }
    if (row < M && col < N) C[row*N + col] = acc;
}
```

### 4.5 Softmax(数值稳定,算子岗高频)

```cpp
// 一个 block 处理一行,数值稳定版
__global__ void softmax(const float* in, float* out, int cols) {
    extern __shared__ float s[];
    int row = blockIdx.x, tid = threadIdx.x;
    const float* x = in + row * cols;
    float* y = out + row * cols;

    // 1. 求最大值(grid-stride + 归约)
    float m = -INFINITY;
    for (int c = tid; c < cols; c += blockDim.x) m = fmaxf(m, x[c]);
    s[tid] = m; __syncthreads();
    for (int o = blockDim.x/2; o > 0; o >>= 1) {
        if (tid < o) s[tid] = fmaxf(s[tid], s[tid+o]);
        __syncthreads();
    }
    m = s[0]; __syncthreads();

    // 2. 求 exp 和(减最大值防溢出)
    float sum = 0;
    for (int c = tid; c < cols; c += blockDim.x) sum += expf(x[c] - m);
    s[tid] = sum; __syncthreads();
    for (int o = blockDim.x/2; o > 0; o >>= 1) {
        if (tid < o) s[tid] += s[tid+o];
        __syncthreads();
    }
    sum = s[0]; __syncthreads();

    // 3. 归一化
    for (int c = tid; c < cols; c += blockDim.x) y[c] = expf(x[c] - m) / sum;
}
```

### 4.6 单 block Scan(Hillis-Steele)

```cpp
// inclusive scan,n <= blockDim.x
__global__ void scan(float* data, int n) {
    extern __shared__ float tmp[];
    int t = threadIdx.x;
    tmp[t] = (t < n) ? data[t] : 0.0f;
    __syncthreads();
    for (int off = 1; off < n; off <<= 1) {
        float add = (t >= off) ? tmp[t - off] : 0.0f;  // 先读旧值
        __syncthreads();                               // 等大家读完
        tmp[t] += add;                                 // 再写
        __syncthreads();                               // 等大家写完
    }
    if (t < n) data[t] = tmp[t];
}
// 关键:两道 __syncthreads 分隔"读旧值"和"写新值",否则 race
```

### 4.7 Histogram(两版)

```cpp
// 版本1:global atomic(简单但竞争)
__global__ void histGlobal(const int* in, int* hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&hist[in[i]], 1);
}

// 版本2:privatization(每 block 私有直方图,降竞争)
__global__ void histShared(const int* in, int* hist, int n, int bins) {
    extern __shared__ int local[];
    for (int b = threadIdx.x; b < bins; b += blockDim.x) local[b] = 0;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&local[in[i]], 1);            // shared atomic,快
    __syncthreads();
    for (int b = threadIdx.x; b < bins; b += blockDim.x)
        atomicAdd(&hist[b], local[b]);                 // 每 bin 只回写一次
}
```

### 4.8 Warp 归约(纯 shuffle)

```cpp
// 一个 warp(32 线程)内归约,lane 0 拿到结果
__inline__ __device__ float warpReduceSum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);     // 偏移 16/8/4/2/1
    return v;                                          // lane 0 持有总和
}
```

### 4.9 SAXPY / Elementwise(送分题)

```cpp
// SAXPY: y = a*x + y
__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];                 // 编译器会用 FMA
}

// ReLU
__global__ void relu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(in[i], 0.0f);
}
```

### 4.10 矩阵-向量乘(每行一个 warp)

```cpp
// y = A * x,A 是 M×N 行主序,每个 warp 算一行
__global__ void gemv(const float* A, const float* x, float* y, int M, int N) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= M) return;
    float sum = 0;
    for (int c = lane; c < N; c += 32)                 // warp 内分担一行
        sum += A[row * N + c] * x[c];
    for (int off = 16; off > 0; off >>= 1)             // warp 归约
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) y[row] = sum;
}
```

---

## 5. 性能分析题(考思维过程,不要直接给答案)

### 通用答题框架

```text
1. 先定性:这个 kernel 大概是 memory 还是 compute bound?(看计算量/访存比)
2. 看访存模式:合并吗?有复用吗?
3. 形成假设:瓶颈是 X(如写不合并)
4. 上工具验证:ncu 看对应指标(如 gst_efficiency)
5. 优化:针对假设(如 shared tile)
6. 复测验证:那个指标按预期变化 + 时间下降
```

### 例题与思路

**例 1:"transpose 很慢,怎么排查?"**
```text
→ 计算量≈0,必然 memory-bound
→ naive:读连续、写跨步 → global store 不合并(假设)
→ ncu 看 gst_efficiency,很低则证实
→ shared tile 把跨步写换成合并写 + padding 消 bank conflict
→ 复测 gst_efficiency 接近 100%、时间下降
```

**例 2:"kernel 计时快,但整个程序慢?"**
```text
→ Amdahl 信号,瓶颈不在 kernel
→ nsys 看系统时间线:H2D/D2H 占比、GPU 空洞、过度同步
→ pinned + 多 stream 重叠传输与计算;减少 cudaDeviceSynchronize
```

**例 3:"occupancy 只有 25%,一定要提高吗?"**
```text
→ 不一定。先看是不是已经 memory-bound/带宽打满
→ 如果是,提 occupancy 无用;如果 latency-bound,才考虑提
→ 看寄存器/shared 是哪个限制了 occupancy(-Xptxas=-v / ncu)
```

**例 4:"加了 shared memory 反而没变快?"**
```text
→ 可能:输入太小(噪声淹没)、测的是 profiler 时间、没 warmup
→ 或:本来就不是访存瓶颈、引入了 bank conflict、同步开销抵消收益
→ 加大规模 + warmup + ncu 看 shared 指标
```

---

## 6. 系统设计题(高级岗)

### 答题五步框架

```text
1. 澄清约束:数据规模?延迟还是吞吐?什么 GPU?精度?(不澄清就答是大忌)
2. 粗估算:显存够吗?算力够吗?瓶颈在哪?(展示定量能力)
3. 给基线:最简单能跑的方案
4. 优化迭代:针对瓶颈逐步改进
5. 谈取舍:每个选择的代价
```

### 估算用的关键数字(T4)

```text
FP32 算力 8.1 TFLOP/s | 显存带宽 320 GB/s | 显存 16GB | PCIe ~16 GB/s
Roofline 拐点 ≈ 25 FLOP/byte(算力/带宽)
算时间 = FLOP/算力,访存时间 = 字节/带宽,传输时间 = 字节/PCIe,取最大=瓶颈
```

### 常见题型

```text
□ 显存预算:"N×N FP32 矩阵在 16GB 上 N 最大多少?"
   → GEMM 三矩阵 3×N²×4 ≤ 16GB → N ≈ 3.6 万
□ 延迟 vs 吞吐:推理服务怎么权衡?
   → 吞吐优先大 batch;延迟优先小 batch + CUDA Graph;折中动态 batching
□ 传输瓶颈:H2D→kernel→D2H 流水线怎么优化?
   → pinned + 多 stream 重叠,总时间从 3+1+3 向 max(3,1,3)
□ 精度 vs 性能:能用 FP16 吗?
   → 算力密集容忍精度的用 FP16,敏感累加用 FP32(混合精度),验证误差在容差内
```

---

## 7. 你可能还没学但会考的进阶主题

下面这些**超出基础**,但中高级/算子岗常考。逐个补。

### 7.1 Tensor Core 与 WMMA ⭐⭐

```text
是什么:专门做小矩阵乘加的硬件单元,一条指令做一堆乘加,矩阵吞吐高一个数量级
为什么重要:AI 算力主要来自它,大模型 GEMM 全靠它
怎么用:WMMA API(wmma::fragment / load_matrix / mma_sync)或直接 cuBLAS/CUTLASS
限制:主要低精度(FP16/BF16/INT8/FP8),数据要特定形状对齐
```

**参考答案:**
- *"AI 算力为什么暴涨?"* — 主要靠 Tensor Core。它一条指令做一个小矩阵乘加（几十个
  乘加），而且每代支持更低精度（FP16→FP8→FP4），低精度单位时间能做更多乘加，矩阵
  吞吐指数增长。AI 算力主要来自 Tensor Core 而非 CUDA Core。
- *"怎么用 Tensor Core?"* — 两条路：直接调 cuBLAS/cuDNN/CUTLASS（底层用 Tensor Core）；
  或手写 WMMA API（声明 `wmma::fragment` → `load_matrix_sync` 加载 → `mma_sync` 乘加 →
  `store_matrix_sync` 写回）。数据要是 FP16/BF16 等低精度、形状对齐。

### 7.2 CUDA Graph ⭐⭐

```text
是什么:把一串 kernel/memcpy"录制"成图,实例化后可重放,消除重复 launch 开销
为什么重要:很多小 kernel 时,launch 开销占比大;推理低延迟场景关键
怎么用:cudaStreamBeginCapture → 操作 → EndCapture → cudaGraphInstantiate → Launch
```

**参考答案:**
- *"很多小 kernel 怎么优化?"* — 两条路：能融合的先 **kernel 融合**（减少 kernel 数和访存）；
  融不了、但重复启动的用 **CUDA Graph** —— 把一串固定的 kernel/memcpy 录制成图，实例化后
  重放，每次重放只一次提交开销，消除逐个 launch 的 CPU 开销。推理低延迟场景收益明显。
- *"CUDA Graph 适合什么场景?"* — 同一串操作反复执行、且结构固定（如推理每个请求跑同样的
  kernel 序列）；launch 开销占比高（很多小 kernel）时。

### 7.3 Cooperative Groups ⭐

```text
是什么:显式线程组抽象,让同步范围更清晰(thread_block / tiled_partition<32> / grid_group)
价值:比 __syncthreads/__syncwarp 更明确;支持任意大小 tile;grid.sync()(需 cooperative launch)
```

**参考答案:**
- *"除了 __syncthreads 还有什么同步方式?"* — `__syncwarp(mask)`（warp 内）、Cooperative
  Groups（`this_thread_block().sync()` 等价 syncthreads，`tiled_partition<32>` 的 `.sync()` 等价
  syncwarp，`grid_group.sync()` 做全 grid 同步但需 cooperative launch）、fence（顺序不等待）、
  kernel 边界（跨 block 全局同步）。核心区别在“同步范围”和“是否等待”。

### 7.4 kernel 融合 ⭐⭐(算子岗核心)

```text
是什么:把多个算子(如 bias+relu、layernorm 的 mean/var/normalize)融成一个 kernel
为什么:减少 global memory 往返(每个独立 kernel 都要读写一遍 global)
典型:elementwise 融合、reduction 融合、激活融合
```

**参考答案:**
- *"怎么减少访存?"* — 两招：数据复用（搬进 shared 多次用）+ kernel 融合（把多个连续算子
  合成一个 kernel，避免每个独立 kernel 都读写一遍 global）。访存受限算子靠融合，计算受限
  算子靠复用提 AI。
- *"softmax/layernorm 怎么优化?"* — 一个 block 处理一行，用多级归约（max + sum）算出
  统计量，warp shuffle 收尾；把 max/sub/exp/sum/div（或 mean/var/normalize）**融成一个 kernel**，
  数据只读写 global 一遍；数值稳定要减最大值防溢出。

### 7.5 向量化访问 ⭐

```text
float4/float2:一条指令搬 16/8 字节,减少指令数、更易满总线
前提:地址对齐(16 字节)、尾部处理、本来就是访存瓶颈
关键:向量化不修复不合并的布局,要先合并再向量化
```

### 7.6 双缓冲 / 软件流水 ⭐⭐(GEMM 进阶)

```text
是什么:加载下一个 tile 的同时计算当前 tile,隐藏加载延迟
怎么:两套 shared buffer 交替(ping-pong),Ampere+ 用 cp.async 异步加载
为什么 cuBLAS 难超:它做了双缓冲 + 向量化 + register tiling + Tensor Core
```

### 7.7 PyTorch CUDA Extension ⭐(框架/推理岗)

```text
是什么:把自定义 CUDA kernel 包成 PyTorch 算子
怎么:torch.utils.cpp_extension,写 .cu + pybind,torch tensor 转指针
```

**参考答案:**
- *"怎么把你的 kernel 用进 PyTorch?"* — 用 `torch.utils.cpp_extension`（`load` 即时编译或
  `setup.py` + `CUDAExtension`）。写一个 `.cu`，用 `pybind11` 导出函数；接口收 `torch::Tensor`，
  用 `.data_ptr<float>()` 拿到 device 指针传给 kernel，用 `.size()` 拿形状；返回 Tensor。要检查
  输入是否 contiguous、在 CUDA 上。需反向传播则再写 backward。

### 7.8 量化与低精度推理 ⭐(推理岗)

```text
FP16/BF16/INT8/FP8:用低精度换速度和显存
INT8 量化:scale + zero point,推理常用
混合精度:敏感部分高精度、其余低精度
```

**参考答案:**
- *"怎么加速推理?"* — 多管齐下：① 低精度（FP16/INT8 用 Tensor Core，算力高、显存省）；
  ② 算子融合（减访存）；③ CUDA Graph（减 launch 开销、降延迟）；④ KV cache / batching 等系统
  优化。量化要验证精度损失在业务容忍内。
- *"INT8 量化怎么做?"* — 用 scale + zero point 把 FP32 映射到 INT8（对称/非对称），计算用
  INT8、累加用 INT32，再反量化。需校准（用代表性数据统计分布定 scale）。

### 7.9 其他可能问到的(附答案)

**`__ldg` / `__restrict__`**
`__ldg(ptr)` 显式走只读数据 cache(对只读 global 数据有利)。`__restrict__` 告诉编译器指针不
重叠(no aliasing),让它更激进地优化(缓存复用、重排)。两者都是给编译器的"这数据只读/不别名"
提示,常一起用在只读输入指针上。

**launch bounds / maxrregcount**
`__launch_bounds__(maxThreads, minBlocks)` 或编译选项 `--maxrregcount=N` 限制每线程寄存器数。
目的:压低寄存器用量以提高 occupancy。但压太狠会 spill 到 local 反而变慢,要实测权衡(卷五/03)。

**动态并行(Dynamic Parallelism)**
kernel 内部再启动 kernel(`<<<>>>` 套娃)。用于递归/自适应算法(如自适应网格细分)。需要
`-rdc=true`(可重定位 device 代码),有额外开销,初学少用。

**persistent kernel(常驻 kernel)**
让 kernel 长期驻留、循环处理任务队列,而非每批数据都 launch 一次。用于减少反复 launch 的开销、
低延迟流式处理。代价是编程复杂、要自己管理任务分发和同步。

**texture memory 的采样/插值**
texture 路径除了只读 cache,还提供硬件**寻址/采样**:归一化坐标、自动边界处理(clamp/wrap)、
硬件双线性插值。图像处理(缩放、重采样)用它能省手写插值代码且快。普通数值 kernel 用不上。

**bank conflict 在不同 word size 下的行为**
shared bank 按 4 字节(32-bit word)交错。访问 `double`(8 字节)时一个元素跨 2 个 bank,冲突
规则更复杂;访问 `char` 时多个元素挤一个 word。分析 bank conflict 要按实际访问的字节宽度算
`bank = (字节地址/4) % 32`。

---

## 8. 复习计划(按时间)

### 如果有 2 周

```text
Week 1(打基础):
  D1-2: 执行模型 + 内存系统(合并/shared/bank)—— 权重最高,反复练
  D3-4: 手写 reduction + transpose 到能默写,讲清每步
  D5:   同步/race/atomic + warp 原语
  D6-7: 手写 GEMM(naive→tiled),Roofline 定位

Week 2(冲刺):
  D1-2: 性能分析(Roofline/occupancy/divergence)+ ncu 概念
  D3:   softmax/layernorm/融合(算子岗)
  D4:   编译/部署(PTX/CC/JIT)+ 工程化(RAII/测试)
  D5:   进阶主题扫读(Tensor Core/CUDA Graph/量化)
  D6:   §3 概念题全过一遍,答错的补
  D7:   §4 手写题默写 + 模拟面试(讲思维过程)
```

### 如果有 3-4 天(突击)

```text
D1: §2 知识地图查漏 + §3 概念题(只看 🔥)
D2: §4 手写 reduction/transpose/GEMM 默写
D3: §5 性能分析框架 + §3 性能部分
D4: §6 系统设计 + 进阶主题扫读 + 模拟讲项目
```

### 每天必做

```text
✓ 手写至少一个 kernel(不看答案)
✓ 口头答 10 道概念题(逼自己说出来,不是看)
✓ 讲一遍某个优化的"问题→假设→改法→验证"
```

---

## 9. 速查表(面试前一晚看)

### 关键数字(T4)

```text
warp = 32 线程 | block ≤ 1024 线程 | T4: 40 SM, 8.1 TFLOPS, 320 GB/s, 16GB, CC 7.5
shared bank = 32 | Roofline 拐点 ≈ 25 FLOP/byte
延迟量级:register ~1拍, shared ~几十拍, global ~几百拍
```

### 优化决策树

```text
慢?
├─ memory-bound(SoL memory 高)
│  ├─ 不合并 → 改访问模式/SoA(卷三)
│  ├─ 无复用 → shared tile
│  ├─ bank conflict → padding
│  └─ 带宽已满 → 减少数据搬运/融合
├─ compute-bound(SoL compute 高)
│  └─ 减少指令/用 Tensor Core/FMA
└─ latency-bound(都低)
   └─ 看 stall reason → 提并发/减依赖/减同步
```

### 手写 kernel 三查

```text
1. 边界判断了吗?(if idx < n)
2. __syncthreads 位置对吗?(读写之间、全员到达)
3. 访存合并吗?(threadIdx.x 对应连续维度)
```

### 高频"为什么"一句话

```text
为什么 block 不能跨 SM:资源驻留在 SM
为什么要大量线程:延迟隐藏
为什么 shared 快:片上 SRAM
为什么 bank conflict 慢:同 bank 串行
为什么 local 慢:其实在 global
为什么 kernel 返回 void:异步+多线程
为什么浮点用容差:不满足结合律
为什么 occupancy 高不一定快:带宽可能才是墙
```

### 答题口诀

```text
概念题:先一句话结论,再按追问展开(别背一大段)
手写题:边界/同步/合并三查,讲清比写快重要
性能题:问题→假设→改法→数据验证(讲思维过程)
设计题:先澄清约束,边估算边说,谈取舍
讲项目:问题→假设→改法→快了几倍→哪个指标证明(不要罗列技术名词)
```

### 常见陷阱(别踩)

```text
✗ 浮点用 == 比较
✗ 没 warmup 就测性能
✗ 用 Debug(-G)测性能
✗ __syncthreads 放进 if(threadIdx...)分支
✗ 依赖 warp 天然锁步省同步
✗ 只查 launch error 不查 execution error
✗ 用 width 算 pitch 内存的行偏移
✗ -arch=sm_75 以为会留 PTX(不会)
✗ 系统设计不问约束直接答
✗ 讲优化只说"我用了 tiling"没有数据
```

---

## 10. 最后的话

```text
面试官要的不是"会背",而是"懂原理 + 能动手 + 会量化 + 讲得清"。

三个层次:
  初级:能写对、知道概念
  中级:能优化、能讲为什么、会用 profiler
  高级:能定位瓶颈、做取舍、设计系统、有项目背书

最有说服力的永远是:一个有代码、有数据、有报告的项目,
配一段流畅的"问题→假设→改法→快了几倍"的讲解。

刷题是为了应对,做项目才是底气。两手都要抓。
```

> 配套:本指南的详细原理见 `cuda_deep_course/course/` 各卷;实验见 `labs/`。本文档是"面试视角"
> 的索引和题库,深入某个点时回查对应卷。
