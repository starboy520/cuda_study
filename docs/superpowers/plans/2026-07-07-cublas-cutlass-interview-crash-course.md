# cuBLAS 与 CUTLASS 面试速成教材 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一份 2–3 小时可学完、面试够用的 cuBLAS/CUTLASS 教材。

**Architecture:** 单一 Markdown 文档以现有 `gemm_cublas.cu` 为主线，从非方阵参数和 layout 推导进入 API 选择，再用数据流和配置片段解释 CUTLASS。只写少量代码片段，不创建或修改 CUDA 工程。

**Tech Stack:** Markdown、cuBLAS 13.3、cuBLASLt、CUTLASS 3.x、CUDA C++、A100 项目实测

---

### Task 1: 建立项目与官方资料基线

**Files:**
- Create: `docs/cuBLAS与CUTLASS面试速成.md`
- Reference: `week05_gemm_advanced/gemm_cublas.cu`
- Reference: `week05_gemm_advanced/benchmark.md`
- Reference: `week06_tensorcore/tensor_core_profile.md`

- [ ] **Step 1: 运行目标不存在的 RED 检查**

```bash
test ! -e 'docs/cuBLAS与CUTLASS面试速成.md'
```

- [ ] **Step 2: 核对项目证据**

读取现有 cuBLAS 调用、FP32/TF32 数据、row-major 参数技巧和 GEMM 优化阶梯，确认所有引用路径和数字。

- [ ] **Step 3: 核对官方资料**

用 NVIDIA 官方 cuBLAS 13.3、CUTLASS 3.x/GEMM API 文档核对 API、column-major、cuBLASLt heuristics、CUTLASS 3.x 层级和当前术语。

- [ ] **Step 4: 创建文档骨架**

加入定位、2–3 小时路线、cuBLAS、cuBLASLt、CUTLASS、面试题和验收章节。

### Task 2: 编写 cuBLAS 导读

**Files:**
- Modify: `docs/cuBLAS与CUTLASS面试速成.md`

- [ ] **Step 1: 解释库边界**

区分 cuBLAS、cuBLASLt、CUTLASS 的预编译库、灵活 matmul API 和源码模板库定位。

- [ ] **Step 2: 逐行导读现有代码**

解释 include、handle、math mode、`cublasSgemm`、event、destroy、错误检查缺口和 stream 绑定。

- [ ] **Step 3: 用非方阵推导参数**

使用 `A[2,3]×B[3,4]=C[2,4]`，分别画 row-major/column-major 内存，推导 `m/n/k/lda/ldb/ldc`，解释交换 A/B 与 M/N 的技巧。

- [ ] **Step 4: 解释 leading dimension**

明确它是相邻逻辑列（column-major）或相应视图的物理跨度，不是总元素数；加入 padding/submatrix 例子。

- [ ] **Step 5: 解释 API 选择**

比较 `Sgemm`、`GemmEx`、strided-batched、cuBLASLt；区分 storage、compute、accumulator、output。

- [ ] **Step 6: 解释 handle/stream/异步与错误**

加入生命周期、stream、pointer mode、host/device alpha beta 和同步边界。

### Task 3: 编写 benchmark 与精度章节

**Files:**
- Modify: `docs/cuBLAS与CUTLASS面试速成.md`

- [ ] **Step 1: 写公平比较清单**

对齐 shape/layout/dtype/compute/math mode/warmup/repeat/计时/容差/GPU/toolkit。

- [ ] **Step 2: 解读现有 A100 数据**

说明手写 FP32、cuBLAS FP32、TF32 数据的意义和不可直接等精度比较的边界。

- [ ] **Step 3: 写常见错误表**

覆盖结果转置、lda 错、非方阵失败、TF32 误解、stream 计时错、handle 重建和同步破坏并发。

### Task 4: 编写 CUTLASS 面试边界

**Files:**
- Modify: `docs/cuBLAS与CUTLASS面试速成.md`

- [ ] **Step 1: 解释 CUTLASS 为什么存在**

标准 GEMM 优先 cuBLAS；特殊 dtype/layout/epilogue/fusion/研究场景使用 cuBLASLt 或 CUTLASS 的原因。

- [ ] **Step 2: 解释 mainloop 与 epilogue**

用数据流图解释 global→shared→register/MMA→accumulator→epilogue/store。

- [ ] **Step 3: 解释两套层级视角**

同时解释 CTA/warp/instruction tile 与 Device→Kernel→Collective→Tiled MMA/Copy→Atom，说明二者分别是硬件并行和 3.x 软件组件视角。

- [ ] **Step 4: 添加配置阅读模板**

给一个简化配置片段，要求找到 architecture、dtype、layout、alignment、tile、stage、MMA、epilogue；不要求模板可独立编译。

- [ ] **Step 5: 解释 CuTe 和架构迁移**

只解释 layout 映射用途，以及 Ampere `cp.async/mma.sync` 与 Hopper TMA/WGMMA 的 mainloop 变化。

### Task 5: 面试题、两小时清单与验证

**Files:**
- Test: `docs/cuBLAS与CUTLASS面试速成.md`

- [ ] **Step 1: 添加高频题和标准回答**

覆盖 row-major、leading dimension、GemmEx、TF32、stream、cuBLASLt、CUTLASS 层级、mainloop/epilogue、库选择和手写差距。

- [ ] **Step 2: 添加 2–3 小时路线**

按 30/45/45/30 分钟组织代码导读、参数手推、CUTLASS 概念和模拟口述。

- [ ] **Step 3: 结构与事实检查**

确认所有必备主题存在、围栏成对、官方链接有效、本地链接存在、无深入模板 TODO、`git diff --check` 通过。

- [ ] **Step 4: 修改范围和提交**

```bash
git add -- 'docs/cuBLAS与CUTLASS面试速成.md'
git commit -m 'docs: add cuBLAS CUTLASS interview crash course'
```
