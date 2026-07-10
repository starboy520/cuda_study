# Docs 与 Study Plan 彻底整理实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `docs` 与 `study_plan` 从平铺、多主线、重复维护状态整理为“唯一当前计划 + 分类知识目录 + 合并后的 canonical 文档 + 可检索归档”。

**Architecture:** 先建立移动映射与只读链接检查器，再按 study plan、独立专题、明确合并组、历史文档和过程文档的顺序迁移。所有路径移动完成后统一重写三个入口并修复全仓相对链接；最后通过链接检查、旧路径扫描和 Git 范围审计证明没有破坏当前代码工作区。

**Tech Stack:** Markdown、Python 3 标准库、Git、VS Code Markdown diagnostics。

---

## 0. 安全边界

本计划只允许修改：

```text
README.md
docs/**
study_plan/**
scripts/check_markdown_links.py
```

明确禁止修改或暂存：

```text
resume_private/**
portfolio/**
notes/**
week*/**
operator_practice/**
manual_write/**
*.sass
*.ptx
*.ncu-rep
*.nsys-rep
```

每次提交只能使用显式路径，不允许 `git add .`、`git add -A`、`git clean`、`git reset --hard` 或全工作区 stash。

---

## Task 1：记录基线并建立链接检查器

**Files:**
- Create: `scripts/check_markdown_links.py`
- Create: `/tmp/cuda_docs_reorg_before_status.txt`（临时证据，不入 Git）
- Create: `/tmp/cuda_docs_reorg_before_links.txt`（临时证据，不入 Git）

- [ ] **Step 1: 记录整理前 Git 状态**

Run:

```bash
cd /home/qichengjie/workspace/cuda_study
git status --short > /tmp/cuda_docs_reorg_before_status.txt
git --no-pager diff --name-status > /tmp/cuda_docs_reorg_before_diff.txt
```

Expected: 文件中保留现有 kernel 修改、`manual_write` 删除、未跟踪 `portfolio` 和当前计划状态，后续用来证明未误动。

- [ ] **Step 2: 创建只读 Markdown 文件链接检查器**

将以下完整实现写入 `scripts/check_markdown_links.py`：

```python
#!/usr/bin/env python3
"""Check relative file targets in Markdown links without modifying files."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from urllib.parse import unquote


LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
IGNORED_SCHEMES = ("http://", "https://", "mailto:", "tel:")


def markdown_files(inputs: list[Path], excluded: tuple[str, ...]) -> list[Path]:
  files: set[Path] = set()
  for item in inputs:
    if item.is_file() and item.suffix.lower() == ".md":
      candidates = [item]
    elif item.is_dir():
      candidates = item.rglob("*.md")
    else:
      continue
    for candidate in candidates:
      normalized = candidate.as_posix()
      if any(normalized == prefix or normalized.startswith(f"{prefix}/")
           for prefix in excluded):
        continue
      files.add(candidate)
  return sorted(files)


def links_outside_fences(text: str):
  in_fence = False
  fence_marker = ""
  for line_number, line in enumerate(text.splitlines(), start=1):
    fence = FENCE_RE.match(line)
    if fence:
      marker = fence.group(1)
      if not in_fence:
        in_fence = True
        fence_marker = marker
      elif marker == fence_marker:
        in_fence = False
        fence_marker = ""
      continue
    if in_fence:
      continue
    for match in LINK_RE.finditer(line):
      yield line_number, match.group(1).strip()


def relative_target(raw_target: str) -> str | None:
  target = raw_target
  if target.startswith("<") and target.endswith(">"):
    target = target[1:-1]
  target = target.split(maxsplit=1)[0]
  if not target or target.startswith("#") or target.startswith(IGNORED_SCHEMES):
    return None
  return unquote(target.split("#", 1)[0])


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("paths", nargs="*", default=["."], type=Path)
  parser.add_argument("--exclude-prefix", action="append", default=[])
  args = parser.parse_args()

  excluded = tuple(Path(prefix).as_posix().rstrip("/")
           for prefix in args.exclude_prefix)
  broken: list[tuple[Path, int, str, Path]] = []

  for source in markdown_files(args.paths, excluded):
    text = source.read_text(encoding="utf-8", errors="replace")
    for line_number, raw_target in links_outside_fences(text):
      target = relative_target(raw_target)
      if target is None:
        continue
      resolved = (source.parent / target).resolve()
      if not resolved.exists():
        broken.append((source, line_number, raw_target, resolved))

  for source, line_number, raw_target, resolved in broken:
    print(f"{source}:{line_number}: {raw_target} -> {resolved}")
  if broken:
    print(f"broken relative Markdown targets: {len(broken)}", file=sys.stderr)
    return 1
  print("broken relative Markdown targets: 0")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
```

脚本行为固定为：

```text
输入：一个或多个文件/目录，默认仓库根目录
扫描：Markdown 正文中的 [text](target)
忽略：http://、https://、mailto:、纯 #anchor、代码 fenced block
检查：相对文件或目录是否存在，目录链接是否存在
输出：source -> raw target -> resolved target
退出码：0 表示无失效文件目标；1 表示存在失效目标
参数：--exclude-prefix 可重复，排除 .git、resume_private 等目录
```

实现只使用 `argparse`、`pathlib`、`re`、`sys`，不得联网，不修改文件。

- [ ] **Step 3: 验证检查器能发现已知坏链**

Run:

```bash
python3 scripts/check_markdown_links.py README.md docs study_plan \
  --exclude-prefix docs/superpowers \
  > /tmp/cuda_docs_reorg_before_links.txt
```

Expected: 退出码为 1，输出包含旧 `Week1详细步骤.md` 或 `T4实战指南.md` 路径。

- [ ] **Step 4: 检查脚本自身**

Run:

```bash
python3 -m py_compile scripts/check_markdown_links.py
git diff --check -- scripts/check_markdown_links.py
```

Expected: 两条命令均退出 0。

- [ ] **Step 5: 提交检查器**

```bash
git add -- scripts/check_markdown_links.py
git commit -m "docs: add markdown link checker"
```

---

## Task 2：收口 Study Plan

**Files:**
- Modify: `study_plan/README.md`
- Keep active: `study_plan/四周聚焦计划_AIInfra与CUDA深水区.md`
- Create: `study_plan/archive/README.md`
- Move: `docs/archive/Week1详细步骤.md` → `study_plan/archive/legacy-8-week/week01/Week1详细步骤.md`
- Move: `docs/archive/Week1_Day2-Day5学习清单.md` → `study_plan/archive/legacy-8-week/week01/Week1_Day2-Day5学习清单.md`
- Move: `study_plan/Week2详细步骤.md` → `study_plan/archive/legacy-8-week/week02/Week2详细步骤.md`
- Move: `study_plan/Week2_Day1-Day7学习清单.md` → `study_plan/archive/legacy-8-week/week02/Week2_Day1-Day7学习清单.md`
- Move: `study_plan/Week2.5_补缺学习计划.md` → `study_plan/archive/legacy-8-week/bridge/Week2.5_补缺学习计划.md`
- Move: `study_plan/Week3_Day1-Day7学习清单.md` → `study_plan/archive/legacy-8-week/week03/Week3_Day1-Day7学习清单.md`
- Move: `study_plan/Week4_Day1-Day7学习清单.md` → `study_plan/archive/legacy-8-week/week04/Week4_Day1-Day7学习清单.md`
- Move: `study_plan/Week5_性能工程与Nsight实战.md` → `study_plan/archive/legacy-8-week/week05/Week5_性能工程与Nsight实战.md`
- Move: `study_plan/Week6_核心算子开发.md` → `study_plan/archive/legacy-8-week/week06/Week6_核心算子开发.md`
- Move: `study_plan/Week7_作品集项目.md` → `study_plan/archive/legacy-8-week/week07/Week7_作品集项目.md`
- Move: `study_plan/Week8_面试冲刺.md` → `study_plan/archive/legacy-8-week/week08/Week8_面试冲刺.md`

- [ ] **Step 1: 创建归档目录并移动旧计划**

使用 Git 感知移动；未跟踪的当前四周计划不移动。

- [ ] **Step 2: 重写 `study_plan/README.md`**

只保留：

```markdown
# CUDA 学习计划

## 当前唯一执行计划
- 四周聚焦计划

## 当前目标
- AI Infra / Model Serving 是求职落点
- CUDA kernel / performance 是差异化

## 冻结主题
- MoE、多卡、完整 vLLM/CUTLASS、Hopper 实现、零散算子

## 历史计划
- 链接到 archive/README.md
```

不得再展开旧八周每日计划，也不得把 DeepSeek 两月计划称为主计划。

- [ ] **Step 3: 创建 `study_plan/archive/README.md`**

列出旧八周路线、历史硬件 T4、替代计划和使用规则：旧计划只用于查实验/阅读映射，不用于判断当前进度。

- [ ] **Step 4: 修复移动后旧计划内部链接**

按新目录层级修复其对 `docs`、`notes`、`week*` 的相对链接。归档文件也必须通过文件目标检查。

- [ ] **Step 5: 验证 Study Plan 入口**

Run:

```bash
python3 scripts/check_markdown_links.py study_plan
find study_plan -maxdepth 1 -type f -name '*.md' -printf '%f\n' | sort
```

Expected:

```text
README.md
四周聚焦计划_AIInfra与CUDA深水区.md
```

链接检查退出 0。

- [ ] **Step 6: 提交计划收口**

```bash
git add -- study_plan docs/archive/Week1详细步骤.md docs/archive/Week1_Day2-Day5学习清单.md
git commit -m "docs: archive legacy study plans"
```

---

## Task 3：建立分类目录并移动职责独立的文档

**Files:**
- Create: `docs/topics/kv_cache/README.md`
- Create: `docs/interview/README.md`
- Move all files according to the mapping below.

### 3.1 移动映射

```text
docs/Programming_Model详解.md
  -> docs/courses/cuda/Programming_Model详解.md

docs/Week3_TensorCore学习文档.md
  -> docs/courses/cuda/Week3_TensorCore学习文档.md

docs/Week4_Attention与FlashAttention完整学习资料.md
  -> docs/courses/attention/Week4_Attention与FlashAttention完整学习资料.md

docs/Week5增强版_LLM推理优化与decode.md
  -> docs/courses/inference/Week5增强版_LLM推理优化与decode.md

docs/ML基础_训练侧入门.md
  -> docs/courses/ml/ML基础_训练侧入门.md

docs/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md
  -> docs/topics/performance/CUDA深水区_PTX_SASS_MMA_异步流水与Hopper.md

docs/Nsight_Compute_ncu详解.md
  -> docs/topics/performance/Nsight_Compute_ncu详解.md

docs/Occupancy详解_从入门到调优.md
  -> docs/topics/performance/Occupancy详解_从入门到调优.md

docs/CUDA核心原语_场景驱动教程.md
  -> docs/topics/execution/CUDA核心原语_场景驱动教程.md

docs/CooperativeGroups与CUDAGraph深度教程.md
  -> docs/topics/execution/CooperativeGroups与CUDAGraph深度教程.md

docs/cuBLAS与CUTLASS面试速成.md
  -> docs/topics/gemm_tensorcore/cuBLAS与CUTLASS面试速成.md

docs/大模型KVCache系统学习指南.md
  -> docs/topics/kv_cache/大模型KVCache系统学习指南.md

docs/kv_cache_accounting.md
  -> docs/topics/kv_cache/kv_cache_accounting.md

docs/decode_step_dataflow.md
  -> docs/topics/kv_cache/decode_step_dataflow.md

docs/PagedAttention详解.md
  -> docs/topics/kv_cache/PagedAttention详解.md

docs/MoE与多卡并行_系统学习.md
  -> docs/topics/distributed/MoE与多卡并行_系统学习.md

docs/Online_Softmax正确性证明.md
  -> docs/proofs/Online_Softmax正确性证明.md

docs/AI_Infra面试八股全集.md
  -> docs/interview/AI_Infra面试核心题库.md

docs/CUDA工程师面试_14天突击计划.md
  -> docs/interview/CUDA工程师面试_14天突击计划.md

docs/Programming_Guide学习路径.md
  -> docs/reference/Programming_Guide学习路径.md

docs/学习资料索引.md
  -> docs/reference/学习资料索引.md

docs/GPU卡型专项学习指南.md
  -> docs/reference/GPU卡型专项学习指南.md

docs/GPU架构图资源.md
  -> docs/reference/GPU架构图资源.md

docs/项目清单.md
  -> docs/reference/项目清单.md
```

- [ ] **Step 1: 创建目录并执行移动**

跟踪文件使用 `git mv`，未跟踪文件使用普通移动后在提交时显式加入。不得移动 `docs/README.md` 和 `docs/superpowers`。

- [ ] **Step 2: 创建 KV Cache 目录入口**

内容必须明确：

```text
第一次系统学习 → 大模型KVCache系统学习指南
只想算显存     → kv_cache_accounting
想看完整一步   → decode_step_dataflow
两天分页实验   → PagedAttention详解 + 当前四周计划
```

同时声明完整 prefill 的序列交互复杂度、单步 decode 对历史长度的线性读取和 PagedAttention 不改变同一工作负载数学 FLOP 的边界。

- [ ] **Step 3: 创建面试目录入口**

列出 CUDA 核心题库、AI Infra 核心题库、14 天历史验收计划和技术专题。明确当前执行计划仍在 `study_plan`。

- [ ] **Step 4: 暂不批量修所有链接，只修新入口链接**

确保两个新 README 能打开目标文件；全仓链接在 Task 8 统一按映射修复。

- [ ] **Step 5: 提交独立文档分类**

```bash
git add -- docs/courses docs/topics docs/proofs docs/interview docs/reference
git commit -m "docs: organize active documentation by purpose"
```

提交前删除命令中多余前导空格并确认只暂存目标目录。

---

## Task 4：合并 CUDA 面试三份材料

**Files:**
- Create: `docs/interview/CUDA面试核心题库.md`
- Move originals to:
  - `docs/archive/interview/CUDA面试八股全集_拆分原版.md`
  - `docs/archive/interview/CUDA面试八股_追问答案_Q1-Q22.md`
  - `docs/archive/interview/CUDA面试八股_追问答案_Q23-Q40.md`

- [ ] **Step 1: 复制主文为 canonical 基础**

新文件必须完整保留主文 Q1–Q40、手写题、模拟面试和项目证据章节。

- [ ] **Step 2: 将两份追问答案作为同文件附录并入**

追加固定分隔：

```markdown
---

# 附录 A：Q1–Q22 追问与边界

> 本附录由原拆分追问答案合并而来；题号与主文一致。

[原 Q1–Q22 全文]

---

# 附录 B：Q23–Q40 追问、手写评分与项目证据

> 本附录由原续篇合并而来；题号与主文一致。

[原 Q23–Q40 全文]
```

不得摘要替代原文；目标是先消除跨文件跳转且不丢内容。

- [ ] **Step 3: 修正两处已知危险结论**

- 删除或改写 reduction “91% 峰值带宽”结论，说明旧字节口径未验证，不作为 canonical 项目证据。
- 将“误差不来自 FP32 累加”改为“输入量化通常是主要误差源之一；FP32 累加显著降低但不能消除舍入误差”。

- [ ] **Step 4: 原文移入归档**

原文保留完整，不创建活跃旧路径 stub。全仓调用方在 Task 8 更新为 canonical 或归档路径。

- [ ] **Step 5: 验证内容数量**

Run:

```bash
grep -c '^## Q[0-9]' docs/interview/CUDA面试核心题库.md
grep -n '^# 附录 [AB]' docs/interview/CUDA面试核心题库.md
```

Expected: 主问题数量与原主文一致，两个附录标题均存在。

- [ ] **Step 6: 提交面试题库合并**

```bash
git add -- docs/interview/CUDA面试核心题库.md docs/archive/interview \
  docs/CUDA面试八股全集.md docs/CUDA面试八股_追问答案.md docs/CUDA面试八股_追问答案_续.md
git commit -m "docs: consolidate CUDA interview materials"
```

---

## Task 5：合并三个小型重复组

**Files:**
- Modify: `docs/topics/gemm_tensorcore/cuBLAS与CUTLASS面试速成.md`
- Move: `docs/cuBLAS函数速查.md` → `docs/archive/reference/cuBLAS函数速查_原版.md`
- Modify: `docs/topics/execution/CUDA核心原语_场景驱动教程.md`
- Move: `docs/异步拷贝_pipeline_cooperative_groups学习文档.md` → `docs/archive/execution/异步拷贝_pipeline_cooperative_groups学习文档_原版.md`
- Create: `docs/courses/ml/ML零基础记忆卡.md`
- Move:
  - `docs/ML基础_第一卷_精讲.md` → `docs/archive/ml_companions/ML基础_第一卷_精讲.md`
  - `docs/ML基础_第二卷_精讲.md` → `docs/archive/ml_companions/ML基础_第二卷_精讲.md`

- [ ] **Step 1: 合并 cuBLAS 速查**

在主文末尾追加：

```markdown
---

# 附录：cuBLAS 函数速查

> 本附录是 API 名称导航；语义和参数边界以 NVIDIA 官方文档为准。

[原速查全文，去掉重复的一级标题和返回链接]
```

- [ ] **Step 2: 合并异步拷贝短教程**

在原语教程 cp.async 章节后增加“pipeline 与双缓冲补充”，吸收短教程中的：

- 同步拷贝与异步拷贝数据流；
- `producer_acquire/commit`、`consumer_wait/release`；
- 双缓冲序幕、稳态、尾声；
- `thread_scope_block`；
- 边界、对齐和等待点易错项。

Cooperative Groups 深入只链接高级教程，不复制完整定义。

- [ ] **Step 3: 创建 ML 记忆卡**

只保留两份精讲中最适合快速回忆的内容：

```text
训练四步：forward → loss → backward → optimizer.step
参数 / 梯度 / 激活 / optimizer state
FP16 与 BF16 的范围/精度取舍
batch / epoch / learning rate
过拟合与验证集
训练侧与推理侧的 CUDA 关注点
```

每节使用“一句话记忆 + 一个最小例子 + 深入链接”，不再复制两卷全文。

- [ ] **Step 4: 原文归档并验证 canonical 链接**

Run:

```bash
python3 scripts/check_markdown_links.py \
  docs/topics/gemm_tensorcore docs/topics/execution docs/courses/ml docs/archive
```

Expected: 退出 0。

- [ ] **Step 5: 提交小型合并**

```bash
git add -- docs/topics/gemm_tensorcore docs/topics/execution docs/courses/ml \
  docs/archive/reference docs/archive/execution docs/archive/ml_companions \
  docs/cuBLAS函数速查.md docs/异步拷贝_pipeline_cooperative_groups学习文档.md \
  docs/ML基础_第一卷_精讲.md docs/ML基础_第二卷_精讲.md
git commit -m "docs: consolidate overlapping reference guides"
```

---

## Task 6：归档旧路线、旧指南和硬件材料

**Files:**
- Move: `docs/CUDA学习路线图.md` → `docs/archive/curricula/CUDA学习路线图_legacy.md`
- Move: `docs/DeepSeek_CUDA_2月冲刺计划.md` → `docs/archive/curricula/DeepSeek_CUDA_2月冲刺计划_legacy.md`
- Move: `docs/CUDA面试完整准备指南.md` → `docs/archive/interview/CUDA面试完整准备指南_legacy.md`
- Move: `docs/CUDA复习资料_知识体系.md` → `docs/archive/interview/CUDA复习资料_知识体系_legacy.md`
- Move: `docs/archive/T4实战指南.md` → `docs/archive/hardware/T4实战指南.md`
- Create: `docs/archive/README.md`

- [ ] **Step 1: 移动历史材料**

每份文档顶部增加归档声明：

```markdown
> **归档状态**：历史资料，不是当前执行入口。保留用于检索旧课程设计和技术解释；当前导航见相应 README。
```

- [ ] **Step 2: 创建归档总入口**

按 `curricula`、`interview`、`ml_companions`、`hardware`、`superpowers` 分类，说明归档原因和当前替代文档。

- [ ] **Step 3: 修复归档文件内部相对链接**

归档不是坏链豁免区；所有可确认的相对文件目标必须存在。

- [ ] **Step 4: 验证归档**

```bash
python3 scripts/check_markdown_links.py docs/archive
```

Expected: 退出 0。

- [ ] **Step 5: 提交历史材料归档**

```bash
git add -- docs/archive docs/CUDA学习路线图.md docs/DeepSeek_CUDA_2月冲刺计划.md \
  docs/CUDA面试完整准备指南.md docs/CUDA复习资料_知识体系.md
git commit -m "docs: archive superseded curricula and guides"
```

---

## Task 7：归档已完成的 Superpowers 过程文件

**Files:**
- Keep active:
  - `docs/superpowers/specs/2026-07-10-gemm-portfolio-rebuild-design.md`
  - `docs/superpowers/specs/2026-07-10-docs-study-plan-reorganization-design.md`
  - `docs/superpowers/plans/2026-07-10-gpu-kernel-engineering-gemm.md`
  - `docs/superpowers/plans/2026-07-10-docs-study-plan-reorganization.md`
- Move all other files under `docs/superpowers/specs` and `docs/superpowers/plans` to matching `docs/archive/superpowers/...` directories.
- Create: `docs/archive/superpowers/README.md`

- [ ] **Step 1: 移动历史过程文档**

保持文件名不变，避免丢失日期和主题。

- [ ] **Step 2: 创建状态索引**

至少标记：

- Tensor Core、Attention、ML、Week5、cuBLAS/CUTLASS、Transformer 扩写、简历套件：`completed`
- CUDA 深水区两周、14 天面试冲刺：`superseded by 四周聚焦计划`
- 简历套件：`private delivery completed`，不链接私有文件。

- [ ] **Step 3: 验证活跃区只剩四份过程文档**

Run:

```bash
find docs/superpowers -type f -name '*.md' -printf '%P\n' | sort
```

Expected: 只列出本任务 Files 中的四份文件。

- [ ] **Step 4: 提交过程归档**

```bash
git add -- docs/superpowers docs/archive/superpowers
git commit -m "docs: archive completed design records"
```

---

## Task 8：统一修复全仓 Markdown 相对链接

**Files:**
- Modify: Markdown files across the repository that reference moved paths.
- Do not modify code files merely because comments mention old paths unless the comment is an active user-facing build instruction.

- [ ] **Step 1: 建立旧→新路径映射表**

映射至少包含 Task 2–7 的全部移动。对三份合并源：

```text
CUDA面试八股全集.md                  -> interview/CUDA面试核心题库.md
CUDA面试八股_追问答案.md             -> interview/CUDA面试核心题库.md
CUDA面试八股_追问答案_续.md          -> interview/CUDA面试核心题库.md
```

- [ ] **Step 2: 修复活跃入口和当前四周计划**

优先修复：

```text
README.md
docs/README.md
study_plan/README.md
study_plan/四周聚焦计划_AIInfra与CUDA深水区.md
docs/courses/**
docs/topics/**
docs/interview/**
docs/reference/**
notes/**
operator_practice/**
```

- [ ] **Step 3: 修复归档文件与历史过程文件**

所有标准 Markdown 相对文件链接都应指向真实新路径。纯代码块中的历史命令不自动改写。

- [ ] **Step 4: 运行链接检查器**

Run:

```bash
python3 scripts/check_markdown_links.py README.md docs study_plan notes \
  cuda_deep_course operator_practice portfolio
```

Expected: 退出 0；如 `portfolio` 中存在与本次无关的预存坏链，记录基线差异并只要求本次新增/迁移坏链为 0。优先目标是 `README.md docs study_plan` 必须为 0。

- [ ] **Step 5: 搜索旧路径残留**

Run:

```bash
grep -RInE 'docs/(CUDA学习路线图|DeepSeek_CUDA_2月冲刺计划|CUDA面试八股全集|CUDA面试完整准备指南|Week4_Attention与FlashAttention完整学习资料|PagedAttention详解)\.md|\.\./docs/(CUDA学习路线图|CUDA深水区_PTX_SASS_MMA_异步流水与Hopper|Nsight_Compute_ncu详解)\.md' \
  --include='*.md' .
```

Expected: 无活跃旧路径；归档迁移说明中允许出现纯文本旧文件名，但不允许失效 Markdown 链接。

- [ ] **Step 6: 提交链接迁移**

```bash
git add -- README.md docs study_plan notes cuda_deep_course operator_practice portfolio
```

在提交前必须检查 staged 文件，排除 `portfolio` 未跟踪源码和任何非 Markdown 文件：

```bash
git diff --cached --name-only
```

只保留实际被修复的 Markdown 文件，然后提交：

```bash
git commit -m "docs: repair links after documentation reorganization"
```

---

## Task 9：重写三个入口导航

**Files:**
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `study_plan/README.md`

- [ ] **Step 1: 重写根 README**

只包含：

- 仓库定位：私人 CUDA/AI Infra 学习与实验仓库；
- 当前环境：A100 80GB、`sm_80`；
- 当前唯一计划；
- 知识目录；
- 长期深度教材；
- 实验代码与当前 worklog；
- 学习仓库与未来公开作品仓的边界；
- 历史 T4/八周路线的归档入口。

删除过时的“下一步从 Week 1 开始”和旧目录树。

- [ ] **Step 2: 重写 docs README**

顶部先按当前四周导航，再列：

```text
课程教材
性能与底层
GEMM/Tensor Core
Attention/KV/推理
AI Infra/多卡
面试
参考
归档
```

不得把 `superpowers` 历史过程列为日常学习入口。

- [ ] **Step 3: 最终确认 study_plan README**

确保它不重复 docs 导航，只回答“现在执行哪份计划”和“旧计划在哪里”。

- [ ] **Step 4: 验证入口**

```bash
python3 scripts/check_markdown_links.py README.md docs/README.md study_plan/README.md
```

Expected: 退出 0。

- [ ] **Step 5: 提交导航**

```bash
git add -- README.md docs/README.md study_plan/README.md
git commit -m "docs: establish focused repository navigation"
```

---

## Task 10：最终验证与工作区保护审计

**Files:**
- No new source files.
- Evidence in `/tmp`, not committed.

- [ ] **Step 1: 完整链接检查**

```bash
python3 scripts/check_markdown_links.py README.md docs study_plan
```

Expected: 退出 0，0 个失效相对文件目标。

- [ ] **Step 2: Markdown 与空白检查**

```bash
git diff --check HEAD~8..HEAD
```

若实际提交数不同，改用本次整理第一个提交的父提交到 `HEAD`。Expected: 退出 0。

- [ ] **Step 3: 检查顶层收口结果**

```bash
find docs -maxdepth 1 -type f -printf '%f\n' | sort
find study_plan -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected:

```text
docs 顶层：README.md
study_plan 顶层：README.md、四周聚焦计划_AIInfra与CUDA深水区.md
```

- [ ] **Step 4: 对比整理前后非目标工作区状态**

生成当前状态并与 `/tmp/cuda_docs_reorg_before_status.txt` 对照。允许 docs/study_plan/scripts/README 的预期变化；以下状态必须保持原样：

- `manual_write` 的已跟踪删除；
- 四个已修改 kernel；
- 未跟踪 `portfolio`；
- `resume_private` 仍不出现在 Git status；
- SASS/PTX/报告生成物未被暂存。

Run:

```bash
git status --short > /tmp/cuda_docs_reorg_after_status.txt
git diff --cached --name-only
```

Expected: 无 staged 文件；非目标状态与基线一致。

- [ ] **Step 5: 检查提交范围**

```bash
git --no-pager log --oneline --stat -10
```

Expected: 本次提交只包含导航、Markdown、路径移动和链接检查脚本。

- [ ] **Step 6: 最终文档诊断**

在 VS Code 中检查 `README.md`、`docs/README.md`、`study_plan/README.md`、当前四周计划和新 canonical 文档，无 Markdown diagnostics。

- [ ] **Step 7: 输出整理报告**

最终报告包含：

- 活跃入口；
- 合并了哪些文档；
- 归档了哪些文档；
- 链接检查结果；
- 未触碰的工作区改动；
- 当前四周计划下一步。
