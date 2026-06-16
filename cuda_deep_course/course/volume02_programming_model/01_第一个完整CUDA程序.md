# 01 第一个完整 CUDA 程序

## 1. 本章目标

本章使用向量加法理解一个完整 CUDA 程序的生命周期：

```text
准备 Host 数据
  -> 分配 Device 内存
  -> H2D 复制
  -> 启动 Kernel
  -> 等待并检查
  -> D2H 复制
  -> 验证结果
  -> 释放资源
```

配套代码：

[`vector_add.cu`](../../labs/02_programming_model/vector_add/vector_add.cu)

## 2. 问题定义

给定两个长度为 `count` 的向量：

```text
c[i] = a[i] + b[i]
```

每个输出元素彼此独立，因此可以让一个 CUDA thread 负责一个元素。

## 2.1 程序运行时的数据状态

开始：

```text
Host A/B 有数据
Host C 为空
Device A/B/C 不存在
```

分配后：

```text
Device A/B/C 有地址，但内容未定义
```

H2D 后：

```text
Device A/B 与 Host 输入一致
Device C 仍未定义
```

Kernel 后：

```text
Device C 有结果
Host C 仍未更新
```

D2H 后：

```text
Host C 才能由 CPU 验证
```

把这个状态表画清楚，许多“为什么 CPU 看不到结果”的问题会消失。

## 3. CPU Reference

```cpp
void vectorAddCpu(const float* a, const float* b, float* c, int count) {
  for (int index = 0; index < count; ++index) {
    c[index] = a[index] + b[index];
  }
}
```

CPU reference 的作用不是追求速度，而是提供可信答案。

## 4. Kernel

```cpp
__global__ void vectorAdd(const float* a,
                          const float* b,
                          float* c,
                          int count) {
  const int index =
      blockIdx.x * blockDim.x + threadIdx.x;

  if (index < count) {
    c[index] = a[index] + b[index];
  }
}
```

`__global__` 表示：

```text
调用位置：Host
执行位置：Device
```

Kernel 代码会被许多 thread 各执行一次。

## 5. Host 数据和 Device 数据

```cpp
std::vector<float> hostA(count);
std::vector<float> hostB(count);
std::vector<float> actual(count);

float* deviceA = nullptr;
float* deviceB = nullptr;
float* deviceC = nullptr;
```

命名中的 `host` 和 `device` 能减少把错误指针传入 kernel 的风险。

## 6. 分配 GPU 内存

```cpp
const std::size_t bytes =
    static_cast<std::size_t>(count) * sizeof(float);

CUDA_CHECK(cudaMalloc(&deviceA, bytes));
CUDA_CHECK(cudaMalloc(&deviceB, bytes));
CUDA_CHECK(cudaMalloc(&deviceC, bytes));
```

`cudaMalloc` 由 Host 调用，但分配的是 Device 可访问的内存。

### 6.1 为什么参数是二级指针风格

API 需要修改调用方的指针值，所以接收地址：

```cpp
cudaMalloc(&deviceA, bytes);
```

C++ CUDA headers 提供类型友好重载，但本质是将分配结果写回变量。

## 7. H2D 复制

```cpp
CUDA_CHECK(cudaMemcpy(
    deviceA,
    hostA.data(),
    bytes,
    cudaMemcpyHostToDevice));
```

四个参数分别表示：

```text
目标地址
源地址
字节数
复制方向
```

`hostB` 同样需要复制。`deviceC` 是输出，不需要提前复制输入值。

## 8. 配置和启动 Kernel

```cpp
constexpr int threads = 256;
const int blocks = (count + threads - 1) / threads;

vectorAdd<<<blocks, threads>>>(
    deviceA, deviceB, deviceC, count);
```

这里：

```text
threads = 每个 block 的 thread 数
blocks  = Grid 中的 block 数
```

`<<<blocks, threads>>>` 是 CUDA kernel launch 语法。

完整形式：

```cpp
kernel<<<grid, block, dynamicSharedBytes, stream>>>(arguments);
```

前两个必需，后两个默认是 0。

## 9. 为什么要边界判断

假设：

```text
count = 1000
threads = 256
blocks = ceil(1000 / 256) = 4
总 thread = 4 x 256 = 1024
```

最后 24 个 thread 没有对应元素，所以 kernel 中必须有：

```cpp
if (index < count) {
  // safe access
}
```

## 10. Kernel Launch 是异步的

Host 发出 launch 后，通常不会等待 kernel 完成：

```cpp
vectorAdd<<<blocks, threads>>>(...);
// CPU 可能立刻执行这里
```

因此需要理解两类错误。

### Launch 配置错误

例如 block thread 数超过硬件限制，可用：

```cpp
CUDA_CHECK(cudaGetLastError());
```

在 launch 后立即检查。

### 执行期间错误

例如非法显存访问可能在 GPU 真正执行 kernel 时发生。需要同步点暴露：

```cpp
CUDA_CHECK(cudaDeviceSynchronize());
```

教学阶段建议两者都写：

```cpp
kernel<<<grid, block>>>(...);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

后续异步流水线不会在每个 launch 后全局同步，而是在正确的位置检查。

## 11. D2H 复制

```cpp
CUDA_CHECK(cudaMemcpy(
    actual.data(),
    deviceC,
    bytes,
    cudaMemcpyDeviceToHost));
```

同步 `cudaMemcpy` 会在需要时等待前面的 Device 工作完成。

## 12. 验证

```cpp
for (int index = 0; index < count; ++index) {
  if (std::fabs(expected[index] - actual[index]) > tolerance) {
    // report first mismatch
  }
}
```

对于当前简单加法，小容差足够。复杂归约和矩阵运算需要更认真处理浮点误差。

### 12.1 为什么报告第一个 mismatch

只输出 `FAIL` 不利于定位。第一个 mismatch 提供：

```text
index
expected
actual
```

二维问题还应反推出 row/col。

## 13. 释放资源

```cpp
CUDA_CHECK(cudaFree(deviceA));
CUDA_CHECK(cudaFree(deviceB));
CUDA_CHECK(cudaFree(deviceC));
```

Event 等资源也要销毁。后续工程卷会使用 RAII 降低手工管理风险。

## 14. CUDA Event 计时

配套实验使用：

```cpp
cudaEventRecord(start);
kernel<<<...>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsedMs, start, stop);
```

Event 位于 GPU 工作流中，适合测量 GPU 操作时间。

实验会：

1. 先执行一次 warmup。
2. 连续运行多个 measured iteration。
3. 用总时间除以迭代数。

## 15. 实践

### L1

构建并运行：

```bash
make -C labs/02_programming_model/vector_add clean all
./labs/02_programming_model/vector_add/vector_add
```

确认以下规模全部 PASS：

```text
1, 31, 32, 33, 255, 256, 257, 1000003
```

### L2

不看原文件，重新写一个版本，只要求：

- 支持任意正整数 `count`。
- CPU reference 验证。
- 每个 CUDA API 都检查错误。
- 运行 `count=1000003`。

### 故障注入

在小测试上删除：

```cpp
if (index < count)
```

使用：

```bash
compute-sanitizer ./labs/02_programming_model/vector_add/vector_add
```

观察越界报告，然后恢复代码。

### 修改实验

依次使用：

```text
threads = 32
threads = 128
threads = 256
threads = 512
```

先预测 blocks 数，再记录时间。不要仅凭一次短时间结果下结论。

### 完整重写任务

不看 sample，从空文件写出：

```text
include
error helper
kernel
CPU reference
input generation
allocation
copies
launch
checks
verification
cleanup
```

能够独立重写，才算真正掌握第一个 CUDA 程序。

## 16. 面试问题

1. `cudaMalloc` 在哪里分配内存？
2. Kernel launch 为什么说是异步的？
3. `cudaGetLastError` 和同步检查分别发现什么错误？
4. 为什么 block 数要向上取整？
5. 为什么必须有 CPU reference？

## 17. 小结

```text
CUDA 程序由 Host 组织，由 Device 执行 kernel。
Host 与 Device 数据需要明确管理。
Kernel launch 后要理解异步和错误暴露时机。
正确性验证必须先于性能结论。
```

## 18. 资料映射

- CUDA Programming Guide：Intro to CUDA C++。
- CUDA Programming Guide：Asynchronous Execution。
- CUDA C++ Best Practices Guide：Getting the Right Answer、Performance Metrics。
- CUDA Samples：vectorAdd。
