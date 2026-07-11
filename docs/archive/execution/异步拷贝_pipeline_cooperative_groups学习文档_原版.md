# cp.async / pipeline / cooperative groups 学习文档

> 目标：搞懂 Ampere 异步拷贝（cp.async）、流水线（pipeline）、协作组（cooperative groups）的用法，为 GEMM double buffering 打基础。
> 适用：sm_80+（A100）。T4(sm_75) 不支持 cp.async。

---

## 一、cp.async：异步 global→shared 拷贝

### 1. 解决什么问题
```text
普通 load： global --LD--> 寄存器 --STS--> shared
            占寄存器 + 线程阻塞等数据

cp.async： global ======直达====== shared
            不占寄存器 + 异步（发了就走）
```

### 2. 两套 API

#### A. 底层 intrinsic（<cuda_pipeline.h>）—— GEMM 常用
```cpp
#include <cuda_pipeline.h>

// 发起一次异步拷贝：global → shared（4/8/16 字节）
__pipeline_memcpy_async(&smem[i], &gmem[i], sizeof(float4));

// 提交：把前面发起的拷贝打包成一个 "阶段(stage)"
__pipeline_commit();

// 等待：保证还有最多 N 个未完成阶段（N=0 即全部完成）
__pipeline_wait_prior(0);

__syncthreads();   // 等待后通常还要 barrier，确保全 block 可见
```

典型流程：
```cpp
// 发起本轮所有 tile 的异步拷贝
for (...) __pipeline_memcpy_async(&sa[...], &a[...], 16);
for (...) __pipeline_memcpy_async(&sb[...], &b[...], 16);
__pipeline_commit();          // 提交这一批
__pipeline_wait_prior(0);     // 等它们到位
__syncthreads();
// 现在 shared 数据可用，开始算
```

#### B. 高层 C++（<cuda/pipeline>）—— 更易读
```cpp
#include <cuda/pipeline>

__shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss;
auto pipe = cuda::make_pipeline(block, &pss);

pipe.producer_acquire();
cuda::memcpy_async(&smem[i], &gmem[i], sizeof(float4), pipe);
pipe.producer_commit();

pipe.consumer_wait();
// 用 smem 计算
pipe.consumer_release();
```

### 3. 约束
```text
- 只支持 global → shared（不能反向）
- 拷贝粒度：4 / 8 / 16 字节（16B 最高效，配 float4）
- 地址要对齐
- sm_80+ 才有；老架构会退化或编译失败
```

---

## 二、pipeline：软件流水线（double buffering 的灵魂）

### 1. 核心思想
```text
把"加载下一块"和"计算当前块"重叠：
  stage N   : 计算 buf[cur]
  同时       : cp.async 把下一块装进 buf[next]
靠多个 buffer + 多个 pipeline stage 实现深度流水。
```

### 2. double buffering 骨架（2 stage）
```cpp
__shared__ float sa[2][BM][BK];   // 两块 buffer
__shared__ float sb[2][BK][BN];

// 预取第 0 块
load_async(sa[0], sb[0], step=0);
__pipeline_commit();

for (int step = 0; step < numTiles; step++) {
    int cur  = step & 1;
    int next = cur ^ 1;

    // 发起下一块（如果还有）
    if (step + 1 < numTiles) {
        load_async(sa[next], sb[next], step+1);
        __pipeline_commit();
    }

    __pipeline_wait_prior(1);  // 等到"当前块"就绪（保留1个在飞的下块）
    __syncthreads();

    compute(sa[cur], sb[cur]); // 算当前块（下块同时在后台搬）

    __syncthreads();
}
```

### 3. 关键点
```text
- buffer 数 = pipeline 深度（2 块=双缓冲；3 块=三级流水更能藏延迟）
- __pipeline_wait_prior(N)：允许 N 个阶段还在飞，平衡"等待"与"重叠"
- 计算和下一块加载真正并行，靠 cp.async 的异步性
```

---

## 三、cooperative groups：协作组

### 1. 解决什么问题
```text
传统：__syncthreads() 只能同步整个 block，粒度粗。
cooperative groups：把线程灵活分组（warp / tile / 整个 grid），
                    在组内同步、归约、shuffle，代码更清晰。
```

### 2. 常用用法
```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// 当前 block
auto block = cg::this_thread_block();
block.sync();                    // 等价 __syncthreads()

// 把 block 切成 32 线程的 warp 组
auto warp = cg::tiled_partition<32>(block);
warp.sync();                     // warp 内同步
int v = warp.shfl_down(x, 1);    // 组内 shuffle（不用手写 mask）

// 组内归约（CUDA 11+）
#include <cooperative_groups/reduce.h>
int sum = cg::reduce(warp, x, cg::plus<int>());
```

### 3. grid 级同步（高级）
```cpp
auto grid = cg::this_grid();
grid.sync();   // 整个 grid 同步（需用 cudaLaunchCooperativeKernel 启动）
```
> 注意：grid.sync() 要求所有 block 同时驻留，受 occupancy 限制，少用。

### 4. 和 GEMM 的关系
```text
GEMM double buffering 里：
- 用 block.sync() 替代 __syncthreads()（可读性）
- 配合 pipeline 做生产者/消费者同步
cooperative groups 不是必须，但让异步流水代码更清晰。
```

---

## 四、三者关系总结

```text
cp.async           : 提供"异步、不过寄存器"的 global→shared 拷贝（硬件能力）
pipeline           : 用多 buffer + 多 stage 把加载和计算重叠（软件结构）
cooperative groups : 灵活的线程分组与同步（组织方式）

double buffering = cp.async（异步搬） + pipeline（多buffer轮流） + sync（同步）
```

---

## 五、学习 / 实践路线

```text
1. 先无 cp.async：双 buffer + 普通 load + step%2 ping-pong（练结构）
2. 加 cp.async：__pipeline_memcpy_async + commit + wait_prior（练异步）
3. ncu 对比：看 global stall 是否下降、SM busy 是否上升
4. 可选：换 cuda::pipeline 高层 API，或用 cooperative groups 重构同步
```

## 六、验收（能讲）
```text
[ ] 普通 load vs cp.async 的数据通路区别
[ ] 为什么 cp.async 省寄存器 + 能异步
[ ] ping-pong 为什么需要两块 buffer
[ ] __pipeline_commit / wait_prior 各做什么
[ ] cooperative groups 相比 __syncthreads 的好处
```

## 七、资料
```text
NVIDIA Blog: Controlling Data Movement to Boost Performance on Ampere
CUDA C++ Programming Guide → Asynchronous Data Copies
PTX ISA → cp.async
libcu++ → cuda::memcpy_async / cuda::pipeline
CUTLASS（工业级 cp.async + pipeline 实现）
```
