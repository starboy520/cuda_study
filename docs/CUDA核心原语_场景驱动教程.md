# CUDA 核心原语 · 场景驱动教程

> 不是罗列 API，而是：每个原语给一个**真实问题**，用它**写出完整解法**，再讲**为什么这么用**和**坑**。
> 建议边读边把代码抠出来跑、改参数看变化。平台标注 ⚠️sm_80+ = Ampere 及以上。

---

## 一、Warp 表决：__ballot_sync（真实场景：流压缩 / 直方图分桶）

### 问题
一个数组，把所有**正数**紧凑地搬到输出数组前面（stream compaction）。难点：每个线程不知道自己该写到输出的第几个位置——因为前面有几个正数是动态的。

### 朴素做法的痛点
用 `atomicAdd(&counter,1)` 抢位置 → 每个正数都原子操作 → 高争用、慢。

### 用 ballot 的解法
让**一个 warp 一次性算出"我前面有几个正数"**，只做一次 atomic：
```cpp
__global__ void compact_positive(const float* in, int n, float* out, int* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    bool keep = (i < n) && (in[i] > 0);

    // ① warp 内 32 个线程，每位代表"该 lane 是否 keep"
    unsigned mask = __ballot_sync(0xffffffff, keep);

    // ② 我在本 warp 内排第几个 keep？= 比我低的位里有几个 1
    int my_rank = __popc(mask & ((1u << lane) - 1));

    // ③ 整个 warp 有几个 keep
    int total = __popc(mask);

    // ④ 只让 lane0 做一次 atomic 申请连续空间
    int warp_base;
    if (lane == 0) warp_base = atomicAdd(count, total);
    warp_base = __shfl_sync(0xffffffff, warp_base, 0);  // 广播给全 warp

    // ⑤ 各自写到 base + 自己的 rank
    if (keep) out[warp_base + my_rank] = in[i];
}
```

### 为什么这么用
```text
__ballot_sync : 把"32 个线程的布尔结果"压成一个 32bit 掩码 → 一次拿到全 warp 信息
__popc        : 数掩码里有几个 1 → 算排名 / 计数
效果：32 次 atomic 降为 1 次（warp 级聚合），高争用场景大幅提速
```

### 坑
```text
- mask 必须用 0xffffffff（或 __activemask()），且所有线程都要执行 ballot（即使 keep=false）
- 分支内调用要小心：未参与的 lane 行为未定义
- __popc(mask & ((1<<lane)-1)) 是"前缀和"技巧，记住这个套路
```

---

## 二、Shuffle：__shfl_*_sync（真实场景：warp 内归约 / 广播）

### 问题
求一个 warp 内 32 个线程各自持有的值之和，**不想用 shared memory**。

### 用 shuffle 的解法
```cpp
__device__ float warp_sum(float v) {
    // 蝶形归约：每步把距离 offset 的值加过来
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;   // lane 0 持有总和
}
```
广播（把 lane0 的结果发给全 warp）：
```cpp
float total = __shfl_sync(0xffffffff, v, 0);  // 所有 lane 读 lane0 的 v
```

### 四种 shuffle 的场景
```text
__shfl_down_sync(v, d)  : 读 lane+d  → 归约（求和/max）
__shfl_up_sync(v, d)    : 读 lane-d  → 前缀和（scan）
__shfl_xor_sync(v, m)   : 读 lane^m  → 蝶形/bitonic 排序、全归约
__shfl_sync(v, src)     : 读 src lane → 广播、转置
```

### 为什么这么用
```text
shuffle = 寄存器之间直接交换，比 shared memory 快、不用 __syncthreads
warp 归约比 shared 树形归约少一次写读 shared + 少同步
```

### 坑
```text
- 一次只传 32bit：double/long long 要拆两半，或 reinterpret 成 int2 分别 shuffle
- 全 warp 都要参与（mask 全 1），否则未参与 lane 返回未定义
- __shfl_xor 做"全归约"时所有 lane 都拿到结果（不止 lane0）
```

---

## 三、atomicCAS：自己造原子操作（真实场景：float 的 atomicMax）

### 问题
CUDA 有 `atomicMax(int*)` 但**没有 `atomicMax(float*)`**（老架构）。你要对 float 数组求全局最大值。

### 用 atomicCAS 的解法
`atomicCAS` 是"比较并交换"，是所有原子操作的基石。用它+自旋循环造任意原子：
```cpp
__device__ float atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        float cur = __int_as_float(assumed);
        if (cur >= val) break;                  // 已经更大，不用换
        old = atomicCAS(addr_as_int, assumed,   // 期望还是 assumed
                        __float_as_int(val));    // 就换成 val
    } while (assumed != old);                    // 被别人改了→重试
    return __int_as_float(old);
}
```

### 为什么这么用
```text
atomicCAS(ptr, expected, desired):
  if (*ptr == expected) { *ptr = desired; }  原子完成
  返回旧值
循环模式："读旧值→算新值→CAS→失败就重试"可实现任意原子操作
float 当成 int 位模式操作（__float_as_int / __int_as_float）
```

### 何时该用 / 不该用 atomic
```text
该用  ：低频更新（如全局 max、计数器收尾）
不该用：高频（如每元素 atomicAdd 做归约）→ 先 warp/block 聚合再 atomic
```

### 坑
```text
- 自旋 CAS 在高争用下退化严重（大量重试）
- float 的位模式比较对负数/NaN 要小心（这里 >= 提前退出规避了大部分）
- 现代架构(sm_80)其实已有 atomicMax 的 float 变体，但 CAS 套路通用，必须会
```

---

## 四、cp.async + pipeline：double buffering（真实场景：边搬边算）⚠️sm_80+

### 问题
一个 kernel 反复"从 global 搬一块到 shared → 算 → 再搬下一块"。搬的时候计算单元空转，延迟没藏住。

### 朴素流程（串行，慢）
```cpp
for (tile) {
    load(smem, gmem[tile]);   // 线程卡住等 global 返回
    __syncthreads();
    compute(smem);            // 算的时候内存单元空转
    __syncthreads();
}
```

### 用 cp.async + 双 buffer 的解法
```cpp
#include <cuda_pipeline.h>

__shared__ float buf[2][TILE];   // 两块，ping-pong

// 预取第 0 块
__pipeline_memcpy_async(&buf[0][threadIdx.x], &g[0*TILE + threadIdx.x], sizeof(float));
__pipeline_commit();

for (int t = 0; t < numTiles; t++) {
    int cur = t & 1, next = cur ^ 1;

    // 发起下一块（异步，不阻塞）
    if (t + 1 < numTiles) {
        __pipeline_memcpy_async(&buf[next][threadIdx.x],
                                &g[(t+1)*TILE + threadIdx.x], sizeof(float));
        __pipeline_commit();
    }

    __pipeline_wait_prior(1);   // 等"当前块"就绪（允许 1 个下块还在飞）
    __syncthreads();

    compute(buf[cur]);          // 算当前块，下块同时在后台搬 ← 重叠！

    __syncthreads();
}
```

### 为什么这么用
```text
cp.async         : global→shared 直达，不过寄存器、不阻塞线程
两块 buffer      : 算 buf[cur] 时往 buf[next] 搬，互不覆盖（ping-pong）
wait_prior(1)    : 保留 1 个在途拷贝，实现"算这块/搬下块"重叠
效果：global 延迟被计算掩盖，省寄存器、提 occupancy
```

### 坑
```text
- 只能 global→shared，粒度 4/8/16B，地址要对齐
- commit/wait_prior 的计数要和 buffer 数匹配，错了会读到没搬完的数据
- sm_75(T4) 不支持，会编译失败或退化
```

---

## 五、Cooperative Groups：灵活分组归约（真实场景：让归约代码不再手写 mask）

### 问题
warp 归约要手写 `__shfl_down_sync(0xffffffff, ...)` 循环，mask 易错、可读性差。想要更清晰的写法。

### 用 cooperative groups 的解法
```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void block_sum(const float* in, int n, float* out) {
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);   // 把 block 切成 warp 组

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? in[i] : 0.0f;

    // 一行完成 warp 归约，mask 自动管理
    float wsum = cg::reduce(warp, v, cg::plus<float>());

    __shared__ float s[32];
    if (warp.thread_rank() == 0) s[warp.meta_group_rank()] = wsum;
    block.sync();

    // 第一个 warp 再归约各 warp 的和
    if (warp.meta_group_rank() == 0) {
        float bv = (warp.thread_rank() < blockDim.x/32) ? s[warp.thread_rank()] : 0.0f;
        float bsum = cg::reduce(warp, bv, cg::plus<float>());
        if (warp.thread_rank() == 0) atomicAdd(out, bsum);
    }
}
```

### 为什么这么用
```text
tiled_partition<32>  : 把 block 显式切成 warp 组，语义清晰
cg::reduce           : 一行做组内归约，自动选硬件指令（sm_80 用 __reduce）
warp.thread_rank()   : 组内编号（= lane）
warp.meta_group_rank(): 这是第几个 warp（= warp_id）
好处：不手写 mask、不手写 shuffle 循环，可读、可移植
```

### 坑
```text
- cg::reduce 需要 CUDA 11+ 和 <cooperative_groups/reduce.h>
- grid.sync() 要 cudaLaunchCooperativeKernel + 所有 block 同时驻留，慎用
- 比手写 shuffle 略有抽象开销，极致性能场景仍可能手写
```

---

## 六、Warp Match：按值分组（真实场景：MoE 路由 / warp 内直方图）⚠️sm_70+

### 问题
warp 内每个线程持有一个"类别 id"（如 MoE 里 token 选的 expert）。想知道**哪些 lane 和我是同一类**，好把同类的聚到一起。

### 用 __match_any_sync 的解法
```cpp
__global__ void group_by_value(const int* labels, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int my_label = (i < n) ? labels[i] : -1;

    // 返回掩码：哪些 lane 的 label 和我相同
    unsigned same = __match_any_sync(0xffffffff, my_label);

    // 我这一组有几个？
    int group_size = __popc(same);

    // 组内的"组长"（最低位的那个 lane）负责聚合
    int leader = __ffs(same) - 1;          // 最低 set bit 的位置
    bool is_leader = (lane == leader);

    // is_leader 的线程可以代表整组做一次 atomic / 写出
}
```

### 为什么这么用
```text
__match_any_sync(mask, val):
  返回一个掩码，标出 warp 内所有"val 和我相同"的 lane
用于：按值分组——MoE 路由（同 expert 聚合）、warp 内直方图、去重
配合 __popc(组大小) + __ffs(选组长) 做组内聚合
```

### 坑
```text
- sm_70+ 才有
- 和 ballot 区别：ballot 是"布尔→掩码"，match 是"按值相等→掩码"
- 组长选举用 __ffs(same)-1（最低位 lane），保证每组唯一代表
```

---

## 七、横向对比：什么场景用什么原语

```text
要统计"warp 内多少线程满足条件"         → __ballot_sync + __popc
要算"我在满足条件的线程里排第几"         → __ballot + __popc(低位掩码)  （前缀和）
要 warp 内求和/max（无 shared）          → __shfl_down_sync 循环 / cg::reduce
要前缀和(scan)                          → __shfl_up_sync
要全归约(所有 lane 都拿结果)             → __shfl_xor_sync
要造 CUDA 没有的原子操作                 → atomicCAS + 自旋循环
高频归约别用 atomic                     → warp/block 先聚合，再 1 次 atomic
要边搬数据边算、藏 global 延迟           → cp.async + 双 buffer + pipeline
要清晰的分组同步/归约                    → cooperative groups
要按"值"把 warp 内线程分组               → __match_any_sync + __ffs + __popc
```

---

## 八、动手练习（建议每个写成一个可跑的小 demo）

```text
1. ballot 流压缩：把正数压到前面，和 CPU 结果对比
2. shuffle 归约：warp_sum，对 32 个数求和验证
3. atomicMaxFloat：对随机 float 数组求全局 max，对比 CPU
4. cp.async 双缓冲：向量加/拷贝，nsys 看是否重叠（A100）
5. cg::reduce：block 求和，对比手写 shuffle 版
6. match 分组：随机 label，统计每 warp 各组大小
```

每个 demo 目标：**能编译、有 CPU 参考校验、能讲清这个原语解决了什么**。

> 想练哪个，我给你搭可运行脚手架（CPU 参考 + main 框架，核心 kernel 你填），就像之前 reduce/relu 那样。
