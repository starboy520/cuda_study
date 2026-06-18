# 02 错误包装、RAII 与资源生命周期

## 0. 先建立大局观：CUDA 的两个"坑"

CUDA 编程有两个会反复咬你的工程问题，本章一次解决：

```text
坑1：错误静默丢失
  CUDA API 出错不会抛异常、不会崩溃，只是返回一个错误码。
  你不查，它就被吞掉 —— bug 在后面某个莫名其妙的地方爆发。

坑2：资源泄漏
  cudaMalloc 了忘 cudaFree、中途 return / 抛异常没释放
  —— 显存慢慢漏光，长跑程序最终 OOM。
```

两个坑的解法都来自同一个工程思想：**别靠人记得，靠机制自动保证**。错误用**统一包装宏**
强制检查，资源用 **RAII** 自动释放。

用类比：

```text
手动 cudaFree    ≈ 出门前默念"关煤气、锁门、带钥匙"，迟早漏一项
RAII             ≈ 自动门锁：人走门自动锁，根本不用记
```

## 0.1 术语速查表

| 术语 | 一句话定义 |
|---|---|
| **错误码** | CUDA API 返回的 `cudaError_t`，`cudaSuccess` 才是成功 |
| **`CUDA_CHECK`** | 包装宏：调用后立即查错，出错就报位置并退出 |
| **粘性错误** | 一旦出错，后续 CUDA 调用持续返回同一错误，直到被清掉 |
| **RAII** | 资源获取即初始化：构造时拿资源、析构时自动还 |
| **生命周期** | 资源从分配到释放的全过程 |
| **`cudaGetLastError`** | 取出并清掉当前错误状态（用于查 launch 错误）|

## 1. 错误包装：让错误无处遁形

CUDA 的 runtime API 几乎都返回 `cudaError_t`。**最致命的习惯是忽略它**：

```cpp
cudaMalloc(&ptr, bytes);   // ❌ 出错了你也不知道，ptr 可能是野指针
```

正确做法是**每个调用都查**。但每次手写 `if (err != cudaSuccess) {...}` 太啰嗦，所以用一个
**统一的检查宏**：

```cpp
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\"\n",       \
                    __FILE__, __LINE__, (int)err_,                        \
                    cudaGetErrorString(err_));                            \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)
```

用法：

```cpp
CUDA_CHECK(cudaMalloc(&ptr, bytes));
CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
```

**为什么用 `do { ... } while(0)`？** 这是 C 宏的经典技巧——让宏在任何上下文（包括没加
大括号的 `if`）里都表现得像一条普通语句，不会因为分号或大括号位置出意外。

**为什么要打印 `__FILE__:__LINE__`？** 因为 CUDA 错误常常滞后爆发，知道"在哪一行第一次
出错"能省下大量排查时间。

## 2. Kernel 的错误怎么查（两个时刻）

回忆卷二/05：kernel launch 是**异步**的，所以它的错误分两类、在两个时刻暴露：

```cpp
myKernel<<<grid, block>>>(args);

// ① launch error：配置非法（block>1024、shared 超限），提交时就能查
CUDA_CHECK(cudaGetLastError());

// ② execution error：运行时越界等，必须同步后才暴露
CUDA_CHECK(cudaDeviceSynchronize());
```

为什么要两步（这是高频面试点）：

```text
cudaGetLastError()      抓"提交 kernel 时"就能发现的错（配置非法）
cudaDeviceSynchronize() 等 kernel 真跑完，抓"执行中"才发生的错（越界、非法地址）
```

**粘性错误**要注意：一旦发生 execution error，后续每个 CUDA 调用都会继续返回这个错误，直到
你用 `cudaGetLastError()` 清掉。所以排查时要找**第一个**报错处，不是最后一个。

> 注意：`CUDA_CHECK(cudaDeviceSynchronize())` 会强制 Host 等 GPU——它**只该用于调试**或必要
> 的同步点。release 热路径里不要无脑加，否则破坏异步并发（卷七）。

## 3. RAII：资源生命周期自动化

C++ 的 **RAII（Resource Acquisition Is Initialization）** 思想：**在构造函数里获取资源，在
析构函数里释放**。对象一旦离开作用域，析构自动触发，资源必被释放——无论是正常 return 还是
抛异常。

手动管理的痛点：

```cpp
void f() {
    float* d = nullptr;
    cudaMalloc(&d, bytes);
    if (some_condition) return;     // ❌ 漏了 cudaFree(d) —— 泄漏！
    use(d);
    cudaFree(d);                    // 只有走到这才释放
}
```

RAII 封装后，泄漏在语言层面被根除：

```cpp
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) {            // 构造：拿显存
        CUDA_CHECK(cudaMalloc(&ptr_, bytes));
        bytes_ = bytes;
    }
    ~DeviceBuffer() {                                // 析构：自动还显存
        if (ptr_) cudaFree(ptr_);                    // 析构里不要 CUDA_CHECK exit
    }

    // 禁止拷贝（两个对象指向同一块显存会 double free）
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // 允许移动（转移所有权）
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_) {
        o.ptr_ = nullptr;                            // 源对象交出所有权
    }

    float* get() const { return ptr_; }
    size_t bytes() const { return bytes_; }
private:
    float* ptr_ = nullptr;
    size_t bytes_ = 0;
};
```

现在用起来再也不会漏：

```cpp
void f() {
    DeviceBuffer d(bytes);
    if (some_condition) return;     // ✅ d 离开作用域，析构自动 cudaFree
    use(d.get());
}                                   // ✅ 正常路径也自动释放
```

## 4. RAII 的三条关键纪律

封装 CUDA 资源时，这三点不能省（否则会引入更隐蔽的 bug）：

```text
① 禁止拷贝：两个对象持同一指针 -> 析构两次 -> double free 崩溃
   用 = delete 删掉拷贝构造和拷贝赋值。

② 支持移动：要能转移所有权（如从函数返回 buffer），移动后源对象指针置空。

③ 析构里别抛异常/别 exit：析构函数应"安静地"释放。出错最多记日志，
   不要在析构里 CUDA_CHECK(...exit())，那会在异常展开时引发更糟的问题。
```

> 这正是"为什么 `cudaFree` 在析构里不包 `CUDA_CHECK`"：析构阶段（尤其程序退出时）device
> 可能已被部分销毁，`cudaFree` 报错很常见也无害，强行 `exit` 反而掩盖真正的问题。

## 5. 同样的 RAII 模式适用于所有 CUDA 资源

显存只是一例。stream、event、cublas handle 等都该用 RAII 包：

```cpp
class CudaStream {
public:
    CudaStream()  { CUDA_CHECK(cudaStreamCreate(&s_)); }
    ~CudaStream() { if (s_) cudaStreamDestroy(s_); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    cudaStream_t get() const { return s_; }
private:
    cudaStream_t s_ = nullptr;
};
```

```text
该 RAII 包起来的 CUDA 资源：
  cudaMalloc        -> DeviceBuffer
  cudaMallocHost    -> PinnedBuffer（pinned 内存，卷三/04）
  cudaStreamCreate  -> CudaStream
  cudaEventCreate   -> CudaEvent
  cublasCreate      -> CublasHandle
```

> 工程上也可以直接用现成轮子：`thrust::device_vector`（自动管显存）、智能指针配自定义 deleter
> （`std::unique_ptr<float, CudaDeleter>`）。理解 RAII 原理后，用库还是自己写都行。

## 6. 用 `unique_ptr` 极简实现（进阶）

如果不想写整个类，可以用 `std::unique_ptr` + 自定义删除器，几行搞定：

```cpp
struct CudaDeleter {
    void operator()(void* p) const { if (p) cudaFree(p); }
};
using DeviceMem = std::unique_ptr<float, CudaDeleter>;

DeviceMem make_device(size_t n) {
    float* raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, n * sizeof(float)));
    return DeviceMem(raw);          // unique_ptr 接管，离开作用域自动 cudaFree
}
```

`unique_ptr` 天生禁拷贝、支持移动，正好满足第 4 节的纪律，省去手写五大函数。

## 7. 实践

1. 把你 Week1/Week2 任意一个程序里所有裸 `cudaMalloc/cudaFree` 换成 `DeviceBuffer`，确认
   功能不变、代码更短。
2. 故意在 kernel 里制造越界，验证"只查 launch 不查 sync 会漏掉它"，再加 `cudaDeviceSynchronize`
   的检查抓出来。
3. 写一个会中途 `return` 的函数，对比裸指针版（泄漏）和 RAII 版（不漏）——用 `nvidia-smi` 或
   `cuda-memcheck --leak-check` 观察显存。
4. 给 `DeviceBuffer` 加移动构造，写一个"从函数返回 buffer"的例子验证所有权转移正确。

## 8. 面试题（附参考答案）

**Q1：CUDA API 出错为什么容易被忽略？怎么防？**
它不抛异常、不崩溃，只返回 `cudaError_t`，不查就被吞。防法：用统一 `CUDA_CHECK` 宏包每个调用，
出错立即打印 `文件:行` 并退出。

**Q2：kernel 错误为什么要查两次？**
launch 是异步的：`cudaGetLastError()` 抓提交时就能发现的 launch error（配置非法）；
`cudaDeviceSynchronize()` 抓 kernel 真正执行时才暴露的 execution error（越界等）。少查一个会
漏掉一类错误。

**Q3：什么是 RAII？它怎么解决 CUDA 显存泄漏？**
资源获取即初始化：构造时 `cudaMalloc`、析构时 `cudaFree`。对象离开作用域析构自动触发，无论
正常 return 还是抛异常都会释放，从语言层面根除泄漏。

**Q4：RAII 封装 CUDA 资源要注意什么？**
① 禁拷贝（防 double free）；② 支持移动（转移所有权）；③ 析构里别抛异常/别 exit（安静释放）。

**Q5：什么是粘性错误？**
一旦发生 execution error，后续每个 CUDA 调用都持续返回同一错误，直到 `cudaGetLastError()`
清掉。所以排查要找第一个报错处，不是最后一个。

## 9. 资料映射

- CUDA Runtime API：Error Handling。
- CUDA C++ Best Practices Guide：Error Handling。
- 配套：[卷二第 05 章 异步执行、同步与错误模型](../volume02_programming_model/05_异步执行_同步与错误模型.md)。
