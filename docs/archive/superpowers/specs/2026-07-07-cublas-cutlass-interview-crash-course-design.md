# cuBLAS 与 CUTLASS 面试速成教材设计

## 目标

新增一份约 2–3 小时可学完的 cuBLAS/CUTLASS 面试教材，帮助已经完成手写 GEMM、A100 profiling 和 WMMA 的学习者：

- 正确理解和修改现有 cuBLAS 调用；
- 能解释 row/column-major、`m/n/k` 和 leading dimension；
- 知道 `Sgemm`、`GemmEx`、cuBLASLt 的选择边界；
- 会用 cuBLAS 做正确性与性能参考；
- 能解释 CUTLASS 的 mainloop、epilogue、多层 tile 和 3.x 组件层级；
- 能看懂一个 CUTLASS GEMM 配置的大意；
- 应付 CUDA 性能/Kernel 岗常见面试追问。

本教材不以掌握复杂 CUTLASS 模板编程为目标。

## 交付形式

- 新增一份 Markdown 文档；
- 建议路径：`docs/cuBLAS与CUTLASS面试速成.md`；
- 只新增教材，不创建新的 CUDA 工程或修改已有源码；
- 以现有 `week05_gemm_advanced/gemm_cublas.cu` 为真实导读对象；
- 技术信息以当前 NVIDIA cuBLAS 13.3 与 CUTLASS 3.x/当前官方文档为准，同时说明旧版面试术语。

## 学习边界

### cuBLAS

要求：

- 看懂 handle、stream、pointer mode 和生命周期；
- 看懂 `cublasSgemm` 参数；
- 能用非方阵推导 `m/n/k/lda/ldb/ldc`；
- 理解传统接口 column-major 与 row-major 调用技巧；
- 理解 `cublasGemmEx` 的输入、输出、compute type 和算法选择；
- 知道 strided-batched 的用途；
- 知道 cuBLASLt 的 descriptor、layout、preference、heuristic、workspace 和 epilogue；
- 会设计公平 benchmark。

不要求：

- 背完整 BLAS 函数表；
- 默写 cuBLASLt 全部样板代码；
- 覆盖稀疏、分布式或所有数据类型组合。

### CUTLASS

要求：

- 理解 CUTLASS 与 cuBLAS 的产品/工程边界；
- 理解 GEMM mainloop 与 epilogue；
- 理解 CTA/warp/instruction tile 这一硬件视角；
- 理解 CUTLASS 3.x 的 Device→Kernel→Collective→Tiled MMA/Copy→Atom 层级；
- 能从配置片段找到 dtype、layout、tile、stage、MMA、epilogue 和架构；
- 知道 Ampere 的 `cp.async+mma.sync` 与 Hopper 的 TMA/WGMMA 差异；
- 知道 CuTe layout 表达逻辑坐标到线程/值/地址映射。

不要求：

- 默写 CUTLASS 模板；
- 深入 CuTe layout algebra；
- 自己组合 production GEMM；
- 阅读完整模板调用栈；
- 为当前项目安装/编译 CUTLASS。

## 章节设计

1. cuBLAS、cuBLASLt、CUTLASS 分别是什么；
2. 逐行导读现有 `gemm_cublas.cu`；
3. 用 `2×3 × 3×4` 非方阵手推传统 BLAS 参数；
4. column-major、row-major 和转置技巧；
5. `lda/ldb/ldc` 的内存跨度含义；
6. `Sgemm`、`GemmEx`、strided-batched、cuBLASLt 的选择；
7. FP32、TF32、FP16/BF16 输入与 FP32 accumulate；
8. handle、stream、异步性和错误处理；
9. 公平 benchmark 与现有 A100 数据；
10. CUTLASS 为什么存在；
11. mainloop、epilogue 与多层 tile；
12. CUTLASS 3.x 五层结构；
13. 一个 CUTLASS 配置片段的阅读方法；
14. Ampere/Hopper 的 CUTLASS 主循环变化；
15. 高频面试题、标准回答和 2 小时复习清单。

## 教学方式

- 先用小型非方阵手算，避免方阵掩盖参数错误；
- 所有 API 参数先解释 shape 和内存，再给函数签名；
- 明确区分 storage type、compute type、accumulator 和 output type；
- 明确区分库选择、kernel 选择与精度选择；
- CUTLASS 以数据流图和配置阅读为主，不贴大段模板；
- 将已有项目结果作为面试证据，但注明 FP32/TF32 不是公平的同精度对比；
- 对当前版本易变的 API/层级采用官方链接，并标注版本语境。

## 项目连接

教材链接：

- `week05_gemm_advanced/gemm_cublas.cu`；
- `week05_gemm_advanced/benchmark.md`；
- `week05_gemm_advanced/gemm_optimization_ladder.md`；
- `week06_tensorcore/tensor_core_profile.md`；
- `docs/Week3_TensorCore学习文档.md`；
- `docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md`；
- `docs/cuBLAS函数速查.md`。

## 验收标准

学习者完成后应能：

1. 用非方阵解释 `m/n/k/lda/ldb/ldc`；
2. 解释现有 row-major `cublasSgemm` 为什么交换 A/B 与 M/N；
3. 说明何时选 `Sgemm`、`GemmEx`、strided-batched 或 cuBLASLt；
4. 说明 TF32 性能和 FP32 精度路径不能直接混为一谈；
5. 设计手写 GEMM 与 cuBLAS 的公平对比；
6. 用一张图解释 CUTLASS mainloop/epilogue；
7. 同时说清旧硬件 tile 视角和 CUTLASS 3.x 软件层级；
8. 打开一个 CUTLASS 配置并指出七个关键字段；
9. 解释为什么标准 GEMM 优先 cuBLAS、定制融合可能考虑 cuBLASLt/CUTLASS；
10. 不夸大自己对 CUTLASS/CuTe 的实践深度。

Markdown 标题、围栏、表格、本地链接和官方链接必须有效；提交只包含新教材，不纳入其他工作区变更。
