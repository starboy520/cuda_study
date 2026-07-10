# Docs 与 Study Plan 彻底整理设计

日期：2026-07-10

## 1. 背景

仓库当前有 65 份 `docs/**/*.md` 和 11 份 `study_plan/*.md`。问题不是知识量本身，而是教材、专题、速查、面试题、执行计划和历史过程文件混在同一级，同时存在多份“主计划”和大量移动后未修复的相对链接。

当前静态扫描已发现约 47 个与 `docs`、`study_plan` 直接相关的失效相对链接。工作区还包含未提交的 kernel 修改、未跟踪作品集、私有简历目录和生成物，本次整理不得触碰这些内容。

## 2. 目标

1. 让根目录、`docs` 和 `study_plan` 各只有一个明确入口。
2. 当前四周计划成为唯一执行计划；旧八周路线和旧冲刺计划只作历史参考。
3. 按职责建立课程、专题、性能、面试、参考和归档目录。
4. 合并真正重复且维护成本高的文档，不粗暴合并职责不同的材料。
5. 保留历史内容和 Git 可追踪性，不因整理丢失技术细节。
6. 修复仓库内可确认的 Markdown 相对文件链接，并加入可重复运行的链接检查。
7. 不修改 kernel、实验数据、作品集源码或私有简历。

## 3. 核心原则

### 3.1 一个问题只有一个事实所有者

- 当前学习顺序：`study_plan/四周聚焦计划_AIInfra与CUDA深水区.md`
- CUDA 机制与追问：CUDA 面试核心题库
- AI Infra 系统知识：AI Infra 面试核心题库及推理专题
- PTX/SASS：CUDA 深水区教材
- ncu：Nsight Compute 专题
- KV Cache 系统概念：KV Cache 系统指南
- KV 字节公式：KV Cache 显存账
- 单步 decode 算子链：decode 数据流速查
- 分页 KV：PagedAttention 专题
- Online Softmax 正确性：独立证明文档

其他文档可以保留最小自包含说明，但深入内容必须指向事实所有者，不再维护完整副本。

### 3.2 课程、专题、速查不强行合并

同一主题不代表同一职责。例如 KV Cache 系统教材、显存账练习、decode 数据流和 PagedAttention 分别回答“是什么”“占多少”“一步怎么走”“如何分页”，应并列在同一专题目录，而不是拼成巨型文件。

### 3.3 历史资料归档，不直接删除

旧计划、旧指南、重复讲义和已完成的过程性 spec/plan 移入归档。只有已经完整吸收到 canonical 文档、且归档中已有原文的拆分文件，才从活跃目录移除。

## 4. 目标目录结构

```text
docs/
├── README.md
├── courses/
│   ├── cuda/
│   ├── attention/
│   ├── inference/
│   └── ml/
├── topics/
│   ├── performance/
│   ├── gemm_tensorcore/
│   ├── execution/
│   ├── kv_cache/
│   └── distributed/
├── interview/
│   ├── README.md
│   └── specialized/
├── reference/
├── proofs/
├── superpowers/
│   ├── plans/              # 只保留当前 GEMM 实施计划
│   └── specs/              # 只保留当前 GEMM 与本次整理设计
└── archive/
    ├── README.md
    ├── curricula/
    ├── interview/
    ├── ml_companions/
    ├── hardware/
    └── superpowers/
        ├── plans/
        └── specs/

study_plan/
├── README.md
├── 四周聚焦计划_AIInfra与CUDA深水区.md
└── archive/
    ├── README.md
    └── legacy-8-week/
        ├── week01/
        ├── week02/
        ├── bridge/
        ├── week03/
        ├── week04/
        ├── week05/
        ├── week06/
        ├── week07/
        └── week08/
```

## 5. 活跃文档分类

### 5.1 课程教材

- `Programming_Model详解.md` → `docs/courses/cuda/`
- `Week3_TensorCore学习文档.md` → `docs/courses/cuda/`
- `Week4_Attention与FlashAttention完整学习资料.md` → `docs/courses/attention/`
- `Week5增强版_LLM推理优化与decode.md` → `docs/courses/inference/`
- `ML基础_训练侧入门.md` → `docs/courses/ml/`

### 5.2 性能与底层专题

- `CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md` → `docs/topics/performance/`
- `Nsight_Compute_ncu详解.md` → `docs/topics/performance/`
- `Occupancy详解_从入门到调优.md` → `docs/topics/performance/`
- `CUDA核心原语_场景驱动教程.md` → `docs/topics/execution/`
- `CooperativeGroups与CUDAGraph深度教程.md` → `docs/topics/execution/`

### 5.3 GEMM 与 Tensor Core

- `cuBLAS与CUTLASS面试速成.md` → `docs/topics/gemm_tensorcore/`

Tensor Core 周教材仍留在课程层，负责第一次完整学习；深水区教材负责 PTX/SASS/MMA 下钻；cuBLAS/CUTLASS 专题负责工业库和面试。三者互链但不合并。

### 5.4 Attention、KV Cache 与推理系统

- `大模型KVCache系统学习指南.md`
- `kv_cache_accounting.md`
- `decode_step_dataflow.md`
- `PagedAttention详解.md`

以上统一进入 `docs/topics/kv_cache/`，并新增该目录 README，明确四份材料的职责和两天 PagedAttention 学习顺序。

- `Online_Softmax正确性证明.md` → `docs/proofs/`
- `MoE与多卡并行_系统学习.md` → `docs/topics/distributed/`

### 5.5 面试资料

活跃区最终保留：

- `docs/interview/CUDA面试核心题库.md`
- `docs/interview/AI_Infra面试核心题库.md`
- `docs/interview/CUDA工程师面试_14天突击计划.md`：保留为历史验收题来源，但顶部明确“非当前执行计划”；四周计划结束后再移入 archive。
- `docs/interview/specialized/cuBLAS与CUTLASS面试速成.md` 不采用；该文件属于技术专题，只从面试 README 链接，避免同一文件两处所有权。

### 5.6 参考资料

- `Programming_Guide学习路径.md`
- `学习资料索引.md`
- `GPU卡型专项学习指南.md`
- `GPU架构图资源.md`
- `项目清单.md`

统一进入 `docs/reference/`。官方 Programming Guide 章节映射只由 `Programming_Guide学习路径.md` 维护，其他路线文档只链接过去。

## 6. 真正执行的合并

### 6.1 CUDA 面试主文与两份追问

合并源：

- `CUDA面试八股全集.md`
- `CUDA面试八股_追问答案.md`
- `CUDA面试八股_追问答案_续.md`

目标：`docs/interview/CUDA面试核心题库.md`

合并策略：

1. 主文 Q1–Q40 保持原顺序。
2. 两份追问原文作为按 Q 编号组织的“追问与边界”大章节并入同一文件，保证无内容丢失。
3. 后续再逐题嵌回主文；本次整理不以大规模重写技术答案为代价。
4. 修正已知不可靠表述：旧 reduction 91% 带宽不进入 canonical 结论；FP32 累加仍有舍入，不能写成误差完全不来自累加。
5. 三份原文完整保存在 `docs/archive/interview/`，活跃目录不再保留拆分版。

### 6.2 cuBLAS 函数速查

将 `cuBLAS函数速查.md` 完整并入 `cuBLAS与CUTLASS面试速成.md` 的附录，然后把原文件归档。函数速查不再作为第二个活跃事实源。

### 6.3 异步拷贝短教程

将 `异步拷贝_pipeline_cooperative_groups学习文档.md` 中独有的 cp.async/pipeline 示例和易错点并入 `CUDA核心原语_场景驱动教程.md` 的异步拷贝章节；Cooperative Groups 深入内容只链接到高级教程。原短教程完整归档。

### 6.4 ML 第一、第二卷精讲

`ML基础_第一卷_精讲.md` 和 `ML基础_第二卷_精讲.md` 不再作为活跃教材。其“一句话记忆”和独有类比形成 `docs/courses/ml/ML零基础记忆卡.md`，原全文移入 `docs/archive/ml_companions/`。完整训练教材是唯一主线。

## 7. 只归档、不合并

- `CUDA学习路线图.md`：旧 T4 八周长期路线，归档为历史课程地图。
- `DeepSeek_CUDA_2月冲刺计划.md`：被当前四周计划替代。
- `CUDA面试完整准备指南.md`：旧一站式指南；保留原文，活跃面试入口改为新核心题库。
- `CUDA复习资料_知识体系.md`：与新核心题库高度重复，作为历史复习讲义归档。
- `docs/archive/T4实战指南.md`：迁至硬件归档。
- 旧 Week1–Week8 计划：全部迁至 `study_plan/archive/legacy-8-week/`。

不从这些大文档机械抽取所有独有段落。本次目标是降低入口与维护重复；原文在归档中仍可检索，避免为“完美合并”投入数天并引入技术错误。

## 8. Superpowers 过程文档

活跃区仅保留：

- `2026-07-10-gemm-portfolio-rebuild-design.md`
- `2026-07-10-gpu-kernel-engineering-gemm.md`
- 本设计文档及后续实施计划

其余已完成或已被替代的 specs/plans 移入 `docs/archive/superpowers/`，归档 README 标记 `completed` 或 `superseded`。简历套件只写“私有交付完成”，不得链接或复制 `resume_private/`。

## 9. 导航设计

### 9.1 根 README

只保留五个稳定入口：

1. 当前计划
2. 知识目录
3. 深度教材
4. 实验代码
5. 当前 worklog

硬件更新为 A100 80GB、`sm_80`。旧 T4/八周路线只留归档说明。

### 9.2 docs README

按“当前四周该看什么”优先：

- Week 1：GEMM 设计、cuBLAS/CUTLASS 按需参考
- Week 2：PTX/SASS 深水区、ncu、Tensor Core
- Week 3：Attention 教材、Online Softmax 证明
- Week 4：KV Cache、PagedAttention、面试题库

其后再列长期课程和专题目录，不展示 superpowers 历史文件。

### 9.3 study_plan README

只展示当前计划、当前进度、冻结主题和历史计划入口。不再展开旧八周全文表格。

## 10. 链接迁移与验证

### 10.1 迁移规则

1. 先记录所有旧路径到新路径的映射。
2. 移动文件时同时修复移动文件内部出链和全仓库入链。
3. 不使用不理解语义的全局字符串替换。
4. 对历史过程文件允许保留指向当时路径的纯文本，但标准 Markdown 链接必须有效。
5. 对高价值旧入口不创建大量兼容 stub；Git 历史和 archive README 提供迁移映射。仓库内全部调用方更新到新路径。

### 10.2 链接检查器

新增只读脚本 `scripts/check_markdown_links.py`，检查：

- 相对文件目标存在；
- 相对目录包含 README 或明确存在；
- 路径大小写与实际文件一致；
- 忽略 `http(s)`、`mailto` 和代码块；
- 输出源文件、原始链接和解析目标；
- 非零退出码表示存在失效文件目标。

中文标题锚点的跨渲染器规则复杂，首版只验证文件目标；关键导航锚点人工抽查。

### 10.3 验收

- 新入口文档无失效相对文件链接；
- `docs`、`study_plan` 和根 README 的失效链接为 0；
- 全仓旧路径标准 Markdown 入链为 0；
- 不增加 kernel、二进制、SASS、profiler 报告或私有文件；
- Git diff 中移动尽量识别为 rename；
- 未提交的 kernel 修改、`portfolio/`、`resume_private/` 和 `manual_write/` 状态保持不变。

## 11. 实施顺序

1. 创建目标目录、归档 README 和链接检查器。
2. 整理 `study_plan`，固定唯一计划入口。
3. 分类移动职责独立的 docs。
4. 完成四组明确合并。
5. 归档旧路线、旧指南和过程文档。
6. 重写根 README、docs README、study_plan README。
7. 根据映射修复全仓链接。
8. 运行链接检查、旧路径扫描、Git diff 和错误检查。
9. 只提交 docs/study_plan/scripts 范围，绝不使用全量 `git add .`。

## 12. 非目标

- 不重新验证或改写所有技术结论。
- 不把全部知识压缩成一份巨型文档。
- 不修改 CUDA 核心代码。
- 不整理 `notes`、`portfolio`、`resume_private` 或代码目录。
- 不删除历史学习证据。
- 不在本次整理中继续扩展 MoE、PagedAttention 或其他新知识。

## 13. 成功标准

整理完成后，新读者只需作出两次选择：

1. “现在做什么？”进入 `study_plan/README.md`。
2. “某个知识去哪里查？”进入 `docs/README.md`。

不再需要在多份主计划、多份 CUDA 总复习和多份 KV Cache 解释之间自行判断哪一份有效。历史资料仍可在 archive 中检索，但不会和当前主线竞争注意力。
