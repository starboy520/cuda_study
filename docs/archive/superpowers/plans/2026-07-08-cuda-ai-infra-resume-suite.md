# CUDA 与 AI Infra 简历套件 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在远程工作区生成一套包含事实母版、CUDA+AI Infra 通用版、四份岗位专项版和投递建议的中文 ATS 简历，并以不进入公开 Git 历史的方式打包交付。

**Architecture:** 所有包含真实联系方式或完整简历内容的输出只写入本地 `resume_private/`，通过 `.git/info/exclude` 在当前克隆内排除，不修改共享 `.gitignore`。先审计工作经历与 CUDA 仓库证据并建立事实母版，再从同一事实底座派生通用版和四份专项版，最后做跨版本一致性、隐私、ATS 和压缩包验证。

**Tech Stack:** Markdown、Git 本地 exclude、shell 只读验证、ZIP、CUDA 项目仓库证据。

---

## File Structure

所有下列文件均为本地私有文件，不提交 Git：

- Create: `resume_private/戚成杰_事实母版.md`
  - 保存完整职业事实、CUDA 项目事实、指标口径、可用素材和不可写边界。
- Create: `resume_private/戚成杰_CUDA与AI_Infra高级工程师_通用版.md`
  - 默认投递版本，平衡 C++、AI Infra、在线推理和 CUDA 性能工程。
- Create: `resume_private/戚成杰_CUDA性能优化高级工程师.md`
  - 强调 CUDA kernel、GEMM、内存、profiling、PTX/SASS。
- Create: `resume_private/戚成杰_LLM推理算子高级工程师.md`
  - 强调在线推理、Attention、GEMV、Tensor Core 和算子工程。
- Create: `resume_private/戚成杰_AI_Infra性能工程师.md`
  - 强调 C++ serving、TensorFlow、brpc、延迟/资源和 GPU 性能。
- Create: `resume_private/戚成杰_GPU_CUDA系统高级工程师.md`
  - 强调 C++ 系统、CUDA 执行/内存/异步、编译链和性能诊断。
- Create: `resume_private/岗位版本差异与投递建议.md`
  - 说明各版本用途、岗位匹配、差距、关键词边界和面试准备。
- Create: `resume_private/戚成杰_CUDA_AI_Infra_简历套件.zip`
  - 包含上述七份 Markdown，不包含自身。
- Modify locally only: `.git/info/exclude`
  - 增加 `/resume_private/`，不提交。

### Task 1: 建立隐私隔离和输出目录

**Files:**
- Modify locally only: `.git/info/exclude`
- Create directory: `resume_private/`

- [ ] **Step 1: 确认当前仓库和排除文件位置**

  Run:

  ```bash
  git rev-parse --show-toplevel
  git rev-parse --git-path info/exclude
  ```

  Expected: 仓库根目录为当前 CUDA 学习项目，并返回当前克隆的 `.git/info/exclude` 路径。

- [ ] **Step 2: 在本地 exclude 增加私有目录规则**

  使用补丁方式在 `.git/info/exclude` 中仅追加一次：

  ```text
  /resume_private/
  ```

  不修改 `.gitignore`，不把手机号、邮箱或私有目录名写入 commit message。

- [ ] **Step 3: 创建私有目录并验证排除**

  Run:

  ```bash
  mkdir -p resume_private
  touch resume_private/.privacy_check
  git check-ignore -v resume_private/.privacy_check
  rm resume_private/.privacy_check
  ```

  Expected: 输出规则来源为 `.git/info/exclude`，路径命中 `/resume_private/`。

- [ ] **Step 4: 记录初始工作区状态**

  Run: `git status --short`

  Expected: 不显示 `resume_private/`；用户原有修改和未跟踪文件保持原样。

### Task 2: 审计职业事实和 CUDA 项目证据

**Files:**
- Create: `resume_private/戚成杰_事实母版.md`
- Reference: `docs/CUDA工程师面试_14天突击计划.md`
- Reference: `notes/week03.md`
- Reference: `notes/week05.md`
- Reference: `week05_gemm_advanced/benchmark.md`
- Reference: `week05_gemm_advanced/ncu_notes.md`
- Reference: `week06_tensorcore/tensor_core_profile.md`
- Reference: `week04_attention/ncu_pipeline_notes.md`

- [ ] **Step 1: 写候选人基本信息和时间线**

  使用用户提供的真实姓名、手机号、邮箱、GitHub、教育和完整公司时间线。将 2026.04～至今单列为“CUDA 性能工程个人项目”，不将 2025.12～2026.03 休整包装成工作经历。

- [ ] **Step 2: 写完整职业事实库**

  保留依图、连尚、拼多多、蚂蚁、百度的原始职责、技术、团队背景和量化指标。对“节省服务器/核数、延迟下降、代码量下降、收入提升”等数字增加“面试需准备口径”的内部提示，但投递用正文不添加未经用户提供的新数字。

- [ ] **Step 3: 审计 CUDA 可用证据**

  按“可直接写入投递版、只可写技术覆盖、完成突击后再升级、禁止使用”四类记录证据。至少核对：A100 GEMM 12681 GFLOPS 的 shape/条件、WMMA 16484 GFLOPS 与 HMMA、Attention `cp.async` stall 证据、GEMV 19 倍映射优化、PTX/SASS 学习状态和 reduction 91% 口径问题。

- [ ] **Step 4: 标明未完成和不可夸大项**

  将尚未完成或仅有脚手架的 CUDA Graph、Fused RMSNorm、Dequant GEMV，以及尚未形成生产经验的 CUTLASS、生产级 FlashAttention、Hopper 实测标为不可写成已完成成果。

- [ ] **Step 5: 验证事实母版没有内部矛盾**

  Run:

  ```bash
  rg -n '2025\.01|2025\.11|2026\.04|个人项目|GitHub|A100|12681|16484|HMMA|91%' resume_private/戚成杰_事实母版.md
  ```

  Expected: 时间线、个人项目、证据和被禁止使用的旧 91% 口径均有明确记录。

### Task 3: 编写 CUDA + AI Infra 通用投递版

**Files:**
- Create: `resume_private/戚成杰_CUDA与AI_Infra高级工程师_通用版.md`
- Reference: `resume_private/戚成杰_事实母版.md`

- [ ] **Step 1: 写 ATS 头部、目标职位和专业摘要**

  标题定位为 CUDA 与 AI Infra 高级工程师/资深 IC。摘要必须说明 12 年 C++ 高性能系统与在线推理经验，并将 CUDA 表述为 2026 年开始、可由 GitHub 验证的个人性能工程实践，不能写“12 年 CUDA”。

- [ ] **Step 2: 写核心技术能力**

  按 CUDA/GPU 性能、C++/AI Infra、在线推理、分布式系统、工具与平台分组。只列能由事实母版支撑的技能；尚未完成内容使用“了解”或不写。

- [ ] **Step 3: 写 CUDA 个人项目**

  使用 3～5 条最有辨识度且条件明确的要点，覆盖 GEMM、Tensor Core/WMMA、Attention/异步流水和 profiling/指令证据。每条采用“动作 + 方法 + 结果 + 条件”，附 GitHub。

- [ ] **Step 4: 压缩并重排工作经历**

  依图、连尚和拼多多获得主要篇幅；蚂蚁和百度保留最强结果。优先 C++、在线推理、检索、DAG、延迟、吞吐、资源和稳定性，管理人数只作为推动复杂项目的辅助证据。

- [ ] **Step 5: 做 ATS 和真实性检查**

  Run:

  ```bash
  rg -n '^#|^##|^- ' resume_private/戚成杰_CUDA与AI_Infra高级工程师_通用版.md
  rg -n '12年.*CUDA|精通 CUDA|生产级 FlashAttention|CUTLASS 专家|91%' resume_private/戚成杰_CUDA与AI_Infra高级工程师_通用版.md
  ```

  Expected: 标准标题和项目符号存在；第二条命令无输出。

### Task 4: 编写四份岗位专项版

**Files:**
- Create: `resume_private/戚成杰_CUDA性能优化高级工程师.md`
- Create: `resume_private/戚成杰_LLM推理算子高级工程师.md`
- Create: `resume_private/戚成杰_AI_Infra性能工程师.md`
- Create: `resume_private/戚成杰_GPU_CUDA系统高级工程师.md`
- Reference: `resume_private/戚成杰_事实母版.md`
- Reference: `resume_private/戚成杰_CUDA与AI_Infra高级工程师_通用版.md`

- [ ] **Step 1: 编写 CUDA 性能优化版**

  把 CUDA 执行/内存模型、warp 原语、GEMM、shared/register tiling、向量化、`cp.async`、Roofline、Nsight Compute、Compute Sanitizer、PTX/SASS 放在摘要和技能前部；工作经历强调性能方法和 C++ 能力。

- [ ] **Step 2: 编写 LLM 推理算子版**

  把 TensorFlow C++ 在线推理、Attention、Online Softmax、GEMV、WMMA/Tensor Core、prefill/decode 放在前部。不得把脚手架状态的 RMSNorm、Dequant GEMV、CUDA Graph 写成已完成成果。

- [ ] **Step 3: 编写 AI Infra 性能版**

  把 brpc、TensorFlow serving、在线检索/排序、延迟/资源、DAG 和稳定性置于核心；CUDA 作为 GPU 性能深化能力。这份版本应与既有商业经历衔接最自然。

- [ ] **Step 4: 编写 GPU/CUDA 系统版**

  把 C++、CUDA 内存/同步/stream、异步执行、GPU 架构、编译链、PTX/SASS、benchmark 和性能诊断放在前部，弱化广告业务和 ML 术语。

- [ ] **Step 5: 验证四版不是机械复制**

  对比四份文件的目标职位、专业摘要、技能前五项、CUDA 项目前两条和工作经历首条。每份至少在这五类位置中的四类有方向性差异，同时公司、日期和量化事实保持一致。

### Task 5: 编写岗位差异与投递建议

**Files:**
- Create: `resume_private/岗位版本差异与投递建议.md`
- Reference: six resume Markdown files in `resume_private/`

- [ ] **Step 1: 写版本选择指南**

  说明事实母版不可投递、通用版用于无 JD 场景，以及四份专项版各自匹配的职位名称和典型关键词。

- [ ] **Step 2: 写竞争力和差距分析**

  将 AI Infra 性能、C++ 在线推理列为当前最强匹配；将纯 CUDA 性能和 LLM 算子列为可投但需依靠项目证明；将 CUTLASS 专家、多 GPU/HPC、资深 CUDA 架构师列为当前风险较高。

- [ ] **Step 3: 写技能措辞边界**

  给出“可写掌握、建议写熟悉、只能写了解、暂时不写”的能力清单，尤其覆盖 CUDA、PTX/SASS、Hopper、CUTLASS、PyTorch、FlashAttention、CUDA Graph 和多 GPU。

- [ ] **Step 4: 写拿到 JD 后的定制步骤**

  提供关键词提取、职位标题匹配、摘要调整、技能排序、项目要点替换和缺口检查的清单，不建议为了 ATS 添加无法证明的关键词。

- [ ] **Step 5: 写面试证据清单**

  列出商业指标需要准备的口径、CUDA 项目需要准备的命令/数据/失败优化，以及不能混淆仓库旧记录和当天重跑结果的规则。

### Task 6: 跨版本一致性、隐私和 ATS 终验

**Files:**
- Verify: all seven Markdown files in `resume_private/`

- [ ] **Step 1: 验证七份文件存在**

  Run:

  ```bash
  find resume_private -maxdepth 1 -type f -name '*.md' -printf '%f\n' | sort
  ```

  Expected: 恰好列出设计说明中的七份 Markdown。

- [ ] **Step 2: 验证公司和时间一致**

  对六份简历逐一检查依图、连尚、拼多多、蚂蚁、百度的公司名与起止时间。专项版可以压缩内容，但不能改动事实。

- [ ] **Step 3: 验证隐私只存在于私有目录**

  Run:

  ```bash
  test -n "$RESUME_PHONE" && test -n "$RESUME_EMAIL"
  rg -l -F "$RESUME_PHONE" resume_private
  rg -l -F "$RESUME_EMAIL" resume_private
  ! git grep -n -F "$RESUME_PHONE" -- . ':!resume_private'
  ! git grep -n -F "$RESUME_EMAIL" -- . ':!resume_private'
  ```

  `RESUME_PHONE` 和 `RESUME_EMAIL` 只在执行会话中设置为用户提供的真实值，不把值写进计划、命令历史文件或 tracked 文件。Expected: 两条私有目录搜索只列简历文件；两条 tracked 文件搜索无输出。

- [ ] **Step 4: 验证禁止表述和 ATS 结构**

  Run:

  ```bash
  rg -n '12年.*CUDA|精通 CUDA|生产级 FlashAttention|CUTLASS 专家|91%.*峰值' resume_private/*.md
  rg -n '^\|' resume_private/戚成杰_*.md
  ```

  Expected: 两条命令均无输出；简历没有夸大表述或 Markdown 表格。

- [ ] **Step 5: 验证 Git 隔离**

  Run:

  ```bash
  git check-ignore -v resume_private/戚成杰_事实母版.md
  git status --short
  ```

  Expected: 私有文件命中 `.git/info/exclude`，`git status` 不显示 `resume_private/`；用户既有工作区改动保持不变。

### Task 7: 生成可下载压缩包并核验内容

**Files:**
- Create: `resume_private/戚成杰_CUDA_AI_Infra_简历套件.zip`

- [ ] **Step 1: 从私有目录生成 ZIP**

  Run:

  ```bash
  cd resume_private
  zip -9 戚成杰_CUDA_AI_Infra_简历套件.zip ./*.md
  cd ..
  ```

  Expected: ZIP 命令只加入七份 Markdown，不递归加入 Git 文件或压缩包自身。

- [ ] **Step 2: 检查 ZIP 文件列表**

  Run: `unzip -l resume_private/戚成杰_CUDA_AI_Infra_简历套件.zip`

  Expected: 压缩包中恰好有七份 Markdown，无绝对路径、无 `.git`、无临时文件。

- [ ] **Step 3: 最终确认没有 Git 提交**

  Run:

  ```bash
  git status --short
  git log -1 --oneline
  ```

  Expected: `resume_private/` 和 ZIP 均不出现在状态中；简历生成阶段没有新增 Git commit。

- [ ] **Step 4: 交付文件链接**

  交付时提供通用版、事实母版、四个专项版、投递建议和 ZIP 的绝对路径链接，并提醒用户在发送前复核个人信息、数字口径和目标岗位标题。
