# Week 3 Tensor Core Self-Contained Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `docs/Week3_TensorCore学习文档.md` into a self-contained Week 3 textbook that teaches the concepts, supplies runnable demos, requires an independent rewrite, and covers correctness, benchmarking, profiling, FP8, DeepGEMM, and interview review.

**Architecture:** Keep the entire required learning path in one Markdown document. Organize it from mental model to API, then from minimal demonstration to complete GEMM and independent practice, followed by measurement and advanced context. External references remain optional and cannot be prerequisites.

**Tech Stack:** Markdown, CUDA C++17, CUDA WMMA (`<mma.h>`), CUDA FP16 (`<cuda_fp16.h>`), cuBLAS, Nsight Compute, T4 SM75, A100 SM80.

---

### Task 1: Establish the self-contained learning path

**Files:**
- Modify: `docs/Week3_TensorCore学习文档.md`

- [x] **Step 1: Replace the short introduction with an explicit learning contract**

Add the prerequisites, T4/A100/FP8 hardware boundary, required outputs, and the four-step Demo workflow: read, run and modify, close the answer and rewrite, then profile.

- [x] **Step 2: Build the conceptual progression**

Explain scalar FMA versus warp-level MMA, why Tensor Core throughput is high, why data feeding still matters, and how global/shared/register/fragment tiling relates to the existing GEMM optimization ladder.

- [x] **Step 3: Add a technically precise precision table**

Cover FP32, TF32, FP16, BF16, FP8 E4M3, FP8 E5M2, and INT8. Include storage size, exponent/fraction bits, approximate range, Tensor Core availability on T4/A100, accumulation type, use cases, and numerical risks.

- [x] **Step 4: Validate terminology and scope**

Run:

```bash
rg -n "不溢出|精度低无所谓|自动用|一条指令.*16.x16.x16" docs/Week3_TensorCore学习文档.md
```

Expected: no unqualified versions of these claims.

### Task 2: Teach WMMA through two complete demonstrations

**Files:**
- Modify: `docs/Week3_TensorCore学习文档.md`

- [x] **Step 1: Explain the WMMA abstraction before code**

Define `fragment`, warp ownership, opaque lane/register mapping, `matrix_a`/`matrix_b`/`accumulator`, row/column-major layout, leading dimension, pointer alignment, shape multiples, and uniform warp participation.

- [x] **Step 2: Add a minimal one-warp 16x16x16 demonstration**

Provide a complete CUDA program that allocates 16x16 FP16 A/B and FP32 C, launches exactly one 32-thread warp, calls `load_matrix_sync`, `fill_fragment`, `mma_sync`, and `store_matrix_sync`, and checks the result on the host.

- [x] **Step 3: Add a complete multi-tile FP16 GEMM demonstration**

Provide a complete CUDA program for `M`, `N`, and `K` divisible by 16. Map one warp to one 16x16 C tile, loop over K tiles, use FP16 inputs and FP32 accumulation, time with CUDA events, compute GFLOPS, and compare against a CPU reference for a small correctness case.

- [x] **Step 4: Add exact compile and run commands**

Include:

```bash
nvcc -O3 -std=c++17 -arch=sm_75 wmma_fp16_gemm.cu -o wmma_fp16_gemm
nvcc -O3 -std=c++17 -arch=sm_80 wmma_fp16_gemm.cu -o wmma_fp16_gemm_a100
```

State that the learner copies the chosen code block into `wmma_fp16_gemm.cu` for the experiment.

- [x] **Step 5: Add the independent rewrite skeleton**

Show only the function signature, warp/tile index calculations, K-tile loop, and named blanks for fragment declaration, loads, MMA, and store. Include checkpoints for launch shape, layout, leading dimensions, and error behavior.

### Task 3: Make correctness and performance experiments executable

**Files:**
- Modify: `docs/Week3_TensorCore学习文档.md`

- [x] **Step 1: Define correctness criteria**

Explain absolute and relative error, FP16 input conversion effects, why exact bitwise equality is inappropriate, and a practical tolerance with explicit caveats for value distribution and K.

- [x] **Step 2: Add a fair comparison matrix**

Compare FP32 CUDA Core GEMM, FP16-input/FP32-accumulate WMMA, and cuBLAS using the same M/N/K and enough warmups/repetitions. Require GPU, CUDA version, compiler flags, input/accumulation/output types, time, GFLOPS, and error.

- [x] **Step 3: Add cuBLAS reference guidance**

Document the row-major versus column-major trap and give a concrete `cublasGemmEx` strategy, including data types, compute type, and the need to state whether Tensor Core math is allowed.

- [x] **Step 4: Add Nsight Compute commands and interpretation**

Include a version-tolerant first pass:

```bash
ncu --set full --kernel-name regex:wmma.* ./wmma_fp16_gemm
ncu --query-metrics | rg -i "tensor|mma|hmma"
cuobjdump --dump-sass ./wmma_fp16_gemm | rg "HMMA|MMA"
```

Explain SM busy, DRAM throughput, occupancy, registers, Tensor pipe activity, and why instruction evidence plus performance is stronger than a single utilization metric.

- [x] **Step 5: Add expected-result and troubleshooting tables**

Cover misaligned pointers/strides, wrong A/B layout, partial warp participation, dimensions not divisible by 16, missing synchronization assumptions, excessive validation time, unexpectedly low GFLOPS, and misleading first-run timing.

### Task 4: Complete the mixed-precision and AI-infrastructure context

**Files:**
- Modify: `docs/Week3_TensorCore学习文档.md`

- [x] **Step 1: Add T4 and A100 experiment branches**

Specify what works on SM75 and SM80. Explain that TF32 is an Ampere Tensor Core path used through supporting libraries/instructions and is not an automatic property of arbitrary FP32 source code.

- [x] **Step 2: Add FP8 scaling from first principles**

Explain E4M3/E5M2 range/precision trade-offs, quantization and dequantization equations, `amax`, scale and inverse scale, clipping, per-tensor versus per-block scaling, delayed/current scaling, and why FP32 accumulation or higher-precision state remains important.

- [x] **Step 3: Add DeepGEMM reading guidance without making it a prerequisite**

Explain FP8/BF16/FP4 GEMM, grouped GEMM for MoE, JIT specialization, scale layout/swizzling, TMA/WGMMA hardware context, and why current DeepGEMM requires newer architectures than T4/A100 for its primary kernels.

- [x] **Step 4: Connect Tensor Core behavior to LLM workloads**

Contrast prefill's larger GEMMs with decode's thinner or memory-bound operations, and explain why peak Tensor TFLOPS alone does not determine token throughput.

### Task 5: Turn the material into a seven-day training program

**Files:**
- Modify: `docs/Week3_TensorCore学习文档.md`

- [x] **Step 1: Define Day 1 through Day 7**

For every day include learning objective, reading section, coding task, measurement task, artifact, and one closed-book interview question.

- [x] **Step 2: Add reusable record templates**

Add Markdown tables for precision notes, benchmark results, NCU observations, and a final optimization explanation.

- [x] **Step 3: Add final acceptance checks**

Require the learner to explain Tensor Core versus CUDA Core, WMMA versus MMA/WGMMA, fragment semantics, FP16/BF16/TF32 differences, FP32 accumulation, FP8 scaling, Tensor Core verification, and DeepGEMM's role.

- [x] **Step 4: Keep external sources optional**

Place official CUDA WMMA, Transformer Engine FP8, CUTLASS, and DeepGEMM links in the final appendix with a clear “optional deep dive” label.

### Task 6: Verify the final document

**Files:**
- Modify: `docs/Week3_TensorCore学习文档.md`

- [x] **Step 1: Check structure and required coverage**

Run:

```bash
rg -n "^#|Demo|独立重写|正确性|cuBLAS|Nsight Compute|TF32|BF16|FP8|DeepGEMM|Day 1|Day 7|验收" docs/Week3_TensorCore学习文档.md
```

Expected: each required topic appears in a dedicated section.

- [x] **Step 2: Extract and compile the complete CUDA demo when NVCC is available**

Extract the marked complete program into `/tmp/wmma_fp16_gemm.cu`, then run:

```bash
nvcc -O3 -std=c++17 -arch=sm_75 /tmp/wmma_fp16_gemm.cu -o /tmp/wmma_fp16_gemm
```

Expected: exit code 0. If no compatible GPU is available, compilation is still required and runtime verification is explicitly reported as unavailable.

- [x] **Step 3: Check Markdown hygiene and stale wording**

Run:

```bash
git diff --check -- docs/Week3_TensorCore学习文档.md
rg -n "TBD|TODO|以后补|精度低无所谓|BF16.*不溢出" docs/Week3_TensorCore学习文档.md
```

Expected: no whitespace errors, placeholders, or misleading claims.

- [x] **Step 4: Compare against the approved design**

Confirm that every section of `docs/superpowers/specs/2026-06-29-week3-tensorcore-self-contained-design.md` is represented and that no other local document is required reading.
