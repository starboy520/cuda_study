# CUDA Interview Answers Inline Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Q1–Q40 的追问答案和第 15 章参考答案移动到对应问题正下方，删除重复附录，并补充少量高价值资料链接。

**Architecture:** 保持单文件题库和现有 15 个章节不变，以 Q 编号作为附录答案到正文的唯一映射键。先分别迁移附录 A、B、C，再统一删除附录和失效提示，最后用结构计数、链接检查及人工语义检查验证没有答案丢失或重复。

**Tech Stack:** GitHub Flavored Markdown、仓库内 `scripts/check_markdown_links.py`、Git diff/grep/Python 结构检查。

---

## 文件结构

- 修改：`docs/interview/CUDA面试核心题库.md`——唯一的题库正文，承载主答案、追问答案、第 15 章参考答案和资料链接。
- 参考：`docs/superpowers/specs/2026-07-15-cuda-interview-question-bank-inline-design.md`——已批准的结构和验收标准，不修改。
- 不创建运行时代码、测试代码或临时迁移脚本；所有正文修改使用补丁完成，避免把一次性工具留在仓库。

### Task 1: 内联 Q1–Q22 追问答案

**Files:**
- Modify: `docs/interview/CUDA面试核心题库.md:37-610`
- Source content: `docs/interview/CUDA面试核心题库.md:1171-1376`（附录 A）

- [ ] **Step 1: 建立 Q1–Q22 映射检查表**

确认正文 Q1–Q22 和附录 A 中的 Q1–Q22 均连续且各出现一次。运行：

```bash
python3 - <<'PY'
from pathlib import Path
import re
text = Path('docs/interview/CUDA面试核心题库.md').read_text(encoding='utf-8')
main = text.split('# 附录 A：', 1)[0]
appendix = text.split('# 附录 A：', 1)[1].split('# 附录 B：', 1)[0]
main_ids = [int(x) for x in re.findall(r'^### Q(\d+)：', main, re.M) if int(x) <= 22]
appendix_ids = [int(x) for x in re.findall(r'^## Q(\d+)\b', appendix, re.M)]
assert main_ids == list(range(1, 23)), main_ids
assert appendix_ids == list(range(1, 23)), appendix_ids
print('Q1-Q22 mapping OK')
PY
```

Expected: `Q1-Q22 mapping OK`。

- [ ] **Step 2: 将每个追问改成问题后紧跟答案**

对 Q1–Q22 逐题把附录 A 的两个答案原文移动到正文“面试官继续追问”位置，并统一为：

```markdown
**追问 1：latency hiding 和降低 latency 有什么区别？**

**参考答案**

- **降低 latency**：让单次操作本身更快。
- **latency hiding**：操作延迟仍存在，但等待时调度其他 ready warp。

**追问 2：为什么小 kernel 可能被 launch 开销主导？**

**参考答案**

每次 kernel launch 有固定提交与调度开销；当 kernel 本身只有几微秒时，固定开销会占据主要比例。
```

迁移时保留原答案中的数字、限制条件、项目引用和反例；不要在此步骤删除附录 A，便于 diff 对照。

- [ ] **Step 3: 检查 Q1–Q22 的答案数量和邻接关系**

运行：

```bash
python3 - <<'PY'
from pathlib import Path
import re
text = Path('docs/interview/CUDA面试核心题库.md').read_text(encoding='utf-8')
main = text.split('# 附录 A：', 1)[0]
for q in range(1, 23):
    start = re.search(rf'^### Q{q}：', main, re.M).start()
    next_match = re.search(rf'^### Q{q + 1}：', main[start:], re.M) if q < 22 else None
    section = main[start:start + next_match.start()] if next_match else main[start:]
    assert section.count('**追问 1：') == 1, q
    assert section.count('**追问 2：') == 1, q
    assert section.count('**参考答案**') == 2, q
print('Q1-Q22 inline answers OK')
PY
```

Expected: `Q1-Q22 inline answers OK`。

- [ ] **Step 4: 人工抽查迁移内容**

抽查 Q2、Q8、Q11、Q13、Q20、Q22，确认：答案与追问语义对应；代码符号仍使用代码跨度；原有“常见误区”和“项目证据”没有被移动到错误题目。

- [ ] **Step 5: 提交 Q1–Q22 内联结果**

```bash
git add docs/interview/CUDA面试核心题库.md
git commit -m "docs: inline CUDA interview answers Q1-Q22"
```

Expected: 只提交题库文件；附录 A 暂时仍保留。

### Task 2: 内联 Q23–Q40 追问答案

**Files:**
- Modify: `docs/interview/CUDA面试核心题库.md:613-1088`
- Source content: `docs/interview/CUDA面试核心题库.md:1387-1598`（附录 B）

- [ ] **Step 1: 建立 Q23–Q40 映射检查表**

运行与 Task 1 同类检查，要求正文和附录 B 都严格覆盖 23–40：

```bash
python3 - <<'PY'
from pathlib import Path
import re
text = Path('docs/interview/CUDA面试核心题库.md').read_text(encoding='utf-8')
main = text.split('# 附录 A：', 1)[0]
appendix = text.split('# 附录 B：', 1)[1].split('# 附录 C：', 1)[0]
main_ids = [int(x) for x in re.findall(r'^### Q(\d+)：', main, re.M) if int(x) >= 23]
appendix_ids = [int(x) for x in re.findall(r'^## Q(\d+)\b', appendix, re.M)]
assert main_ids == list(range(23, 41)), main_ids
assert appendix_ids == list(range(23, 41)), appendix_ids
print('Q23-Q40 mapping OK')
PY
```

Expected: `Q23-Q40 mapping OK`。

- [ ] **Step 2: 迁移 Q23–Q40 的两个追问答案**

使用与 Task 1 相同的“追问 N → 参考答案”结构。保留 Tensor Core、`cp.async`、PTX/SASS、profiling、Sanitizer、多 GPU、Hopper 答案中的架构边界和“当前未实测”说明。

- [ ] **Step 3: 检查 Q23–Q40 的答案数量和邻接关系**

```bash
python3 - <<'PY'
from pathlib import Path
import re
text = Path('docs/interview/CUDA面试核心题库.md').read_text(encoding='utf-8')
main = text.split('# 附录 A：', 1)[0]
for q in range(23, 41):
    start = re.search(rf'^### Q{q}：', main, re.M).start()
    next_match = re.search(rf'^### Q{q + 1}：', main[start:], re.M) if q < 40 else re.search(r'^# 15\.', main[start:], re.M)
    section = main[start:start + next_match.start()]
    assert section.count('**追问 1：') == 1, q
    assert section.count('**追问 2：') == 1, q
    assert section.count('**参考答案**') == 2, q
print('Q23-Q40 inline answers OK')
PY
```

Expected: `Q23-Q40 inline answers OK`。

- [ ] **Step 4: 人工抽查迁移内容**

抽查 Q23、Q26、Q29、Q32、Q38、Q40，重点确认 `ldmatrix`、`wait_group`、Long/Short Scoreboard、TMA/WGMMA 的答案没有串题，且无 H100 实测的边界仍保留。

- [ ] **Step 5: 提交 Q23–Q40 内联结果**

```bash
git add docs/interview/CUDA面试核心题库.md
git commit -m "docs: inline CUDA interview answers Q23-Q40"
```

### Task 3: 重排第 15 章并内联附录 C

**Files:**
- Modify: `docs/interview/CUDA面试核心题库.md:1091-1168`
- Source content: `docs/interview/CUDA面试核心题库.md:1608-1768`（附录 C）

- [ ] **Step 1: 内联 A100 GEMM 六个深挖答案**

在 `### A100 GEMM` 下，将六个问题分别改成四级标题，并紧跟附录 C.1 的完整参考回答：

```markdown
#### 问题 1：如何证明 12.88 TFLOPS / 73.0% cuBLAS 公平且可复现？

**参考回答**

先限定比较对象：A100 80GB PCIe、`sm_80`、row-major FP32、M=N=K=2048……
```

保留 12.88 TFLOPS、17.65 TFLOPS、73.0%、1.333719 ms、1.398333 ms、4.7% 等原始口径，不新增未经验证的数据。

- [ ] **Step 2: 内联其他项目深挖答案**

把 Reduction、Scan、Stream、WMMA 的五个问题分别与附录 C.2 的五个回答绑定。每个问题都使用 `#### 问题 N` 和紧随其后的 `**参考回答**`，并保留“旧 91% 口径作废”“当前公开 GEMM 不含 Tensor Core”等边界。

- [ ] **Step 3: 展开七类手写题答题骨架**

保留原“题目/必须写对/加分项”总览表，在表格后按 Vector Add、Reduction、Scan、Transpose、Histogram、Tiled GEMM、WMMA 骨架建立七个四级小节，并迁移附录 C.3 的三条答题骨架与自检点。

- [ ] **Step 4: 内联三档模拟面试参考思路**

在 B/A/C 三档题单中，每个考察项下面紧跟一段 `**参考思路：**`。使用附录 C.4 原文，不增加完整 kernel 实现。

- [ ] **Step 5: 保留通用项目回答模板**

在第 15 章末新增 `## 15.5 项目回答模板`，迁移附录 C.5 的四段式模板和公开 GEMM 一句话版本。

- [ ] **Step 6: 验证第 15 章覆盖完整**

人工对照附录 C.1–C.5，确认六个 A100 GEMM 回答、五个其他项目回答、七个手写题骨架、三档模拟面试和一个通用模板均已在正文出现一次。

- [ ] **Step 7: 提交第 15 章重排结果**

```bash
git add docs/interview/CUDA面试核心题库.md
git commit -m "docs: inline CUDA project interview references"
```

### Task 4: 补充相关资料并删除重复附录

**Files:**
- Modify: `docs/interview/CUDA面试核心题库.md`

- [ ] **Step 1: 添加高价值官方资料入口**

只在相关章节或问题下增加以下直接资料，不机械地为 40 题各加一个链接：

- Q2–Q4：[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 与 [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)。
- Q6–Q8：[CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)。
- Q14–Q16、Q35：[CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)。
- Q23–Q29：[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) 与 [CUTLASS Documentation](https://docs.nvidia.com/cutlass/)。
- Q30–Q32：[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。
- Q33：[Compute Sanitizer Documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)。
- Q36–Q37：[NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)。
- Q38–Q40：[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)。

使用统一格式：

```markdown
**相关资料**

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
```

已有相同链接时复用，不在同一题重复列出。

- [ ] **Step 2: 删除附录 A、B、C**

确认迁移完成后，从 `# 附录 A：Q1-Q22 追问答案（原文并入）` 起删除到文件末尾的全部重复附录内容。第 15 章迁移后的正文和文末官方资料索引必须保留。

- [ ] **Step 3: 清理过时提示和自引用**

删除或改写“见续篇”“确认这个形式合用”“配套文件指向自身”“原文并入”等迁移遗留措辞。运行：

```bash
if grep -nE '见续篇|确认这个形式合用|原文并入|# 附录 [ABC]' docs/interview/CUDA面试核心题库.md; then
  echo 'stale appendix references remain'
  exit 1
else
  echo 'No stale appendix references'
fi
```

Expected: `No stale appendix references`。

- [ ] **Step 4: 验证全文件结构**

```bash
python3 - <<'PY'
from pathlib import Path
import re
p = Path('docs/interview/CUDA面试核心题库.md')
text = p.read_text(encoding='utf-8')
q_ids = [int(x) for x in re.findall(r'^### Q(\d+)：', text, re.M)]
assert q_ids == list(range(1, 41)), q_ids
assert len(re.findall(r'^\*\*追问 1：', text, re.M)) == 40
assert len(re.findall(r'^\*\*追问 2：', text, re.M)) == 40
assert len(re.findall(r'^\*\*参考答案\*\*$', text, re.M)) == 80
assert '# 附录 A' not in text and '# 附录 B' not in text and '# 附录 C' not in text
assert '## 15.5 项目回答模板' in text
print('40 questions, 80 inline follow-up answers, no duplicate appendices')
PY
```

Expected: `40 questions, 80 inline follow-up answers, no duplicate appendices`。

- [ ] **Step 5: 检查相对 Markdown 链接**

```bash
python3 scripts/check_markdown_links.py docs/interview/CUDA面试核心题库.md
```

Expected: `broken relative Markdown targets: 0`，退出码为 0。

- [ ] **Step 6: 检查 Markdown 空白和基础结构**

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('docs/interview/CUDA面试核心题库.md')
lines = p.read_text(encoding='utf-8').splitlines()
issues = [f'{i}: trailing whitespace' for i, line in enumerate(lines, 1) if line != line.rstrip()]
issues += [f'{i}: tab' for i, line in enumerate(lines, 1) if '\t' in line]
assert not issues, '\n'.join(issues)
print('Markdown whitespace check OK')
PY
```

Expected: `Markdown whitespace check OK`。

- [ ] **Step 7: 审查最终差异**

运行：

```bash
git --no-pager diff --check
git --no-pager diff --stat
git --no-pager diff -- docs/interview/CUDA面试核心题库.md
```

检查结果应主要是答案移动、标题规范化、相关资料增加和重复附录删除；不得静默改变技术结论、性能数字或项目边界。

- [ ] **Step 8: 提交最终清理**

```bash
git add docs/interview/CUDA面试核心题库.md
git commit -m "docs: finalize inline CUDA interview guide"
```

### Task 5: 最终验收

**Files:**
- Verify: `docs/interview/CUDA面试核心题库.md`

- [ ] **Step 1: 重新运行全部自动检查**

依次重新运行 Task 4 的失效提示检查、40 题/80 答案结构检查、相对链接检查、空白检查和 `git diff --check`。所有命令必须退出码为 0。

- [ ] **Step 2: 完成语义抽样检查**

至少检查以下十题：Q2、Q8、Q11、Q13、Q20、Q23、Q26、Q32、Q38、Q40。每题确认“15 秒回答 → 1 分钟展开 → 追问及答案 → 常见误区 → 项目证据/相关资料”顺序正确。

- [ ] **Step 3: 完成第 15 章抽样检查**

确认 A100 GEMM 的成功与失败优化都保留；Reduction 的旧口径仍明确作废；手写题没有出现可直接照抄的完整 kernel；Hopper/Tensor Core 的未实测边界没有被删除。

- [ ] **Step 4: 检查提交范围**

```bash
git status --short
git --no-pager log -4 --oneline
```

Expected: 本计划产生的提交只包含题库重排；工作区中用户原有的其他改动保持原状态。
