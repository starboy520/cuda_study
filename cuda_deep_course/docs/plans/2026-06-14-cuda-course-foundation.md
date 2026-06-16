# CUDA Course Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first usable batch of the CUDA deep-learning course: navigation, conventions, practice contract, Volume 1, and the first programming-model labs.

**Architecture:** Keep explanatory material under `course/` and executable exercises under `labs/`. Every chapter links to a lab where appropriate and follows the approved sequence: intuition, formal model, code, verification, failure injection, measurement, and interview review.

**Tech Stack:** Markdown, Mermaid, CUDA C++17, NVCC, Make, CUDA Runtime API

---

## File Map

- `course/README.md`: canonical course entry point and progress tracker.
- `course/术语与符号约定.md`: global vocabulary, coordinate, size, and indexing conventions.
- `course/实验方法与完成标准.md`: practice levels, benchmark rules, evidence, and chapter completion contract.
- `course/volume01_gpu_basics/README.md`: Volume 1 navigation and completion checklist.
- `course/volume01_gpu_basics/01_CPU与GPU为什么不同.md`: CPU/GPU design goals and workload fit.
- `course/volume01_gpu_basics/02_GPU_SM_Warp_Thread.md`: minimum hardware hierarchy required for CUDA.
- `course/volume01_gpu_basics/03_内存层次第一印象.md`: first-pass register/shared/global memory model.
- `course/volume01_gpu_basics/04_延迟隐藏与大量线程.md`: latency, throughput, warps, and latency hiding.
- `course/volume01_gpu_basics/05_T4设备观察实验.md`: guided device-property lab and interpretation.
- `course/volume01_gpu_basics/06_卷一复习与面试题.md`: review, explanation drills, and interview questions.
- `course/volume02_programming_model/README.md`: Volume 2 navigation.
- `course/volume02_programming_model/01_第一个完整CUDA程序.md`: host/device lifecycle using vector addition.
- `course/volume02_programming_model/02_Grid_Block_Thread索引.md`: 1D/2D indexing, boundary checks, and block shapes.
- `labs/common/cuda_check.cuh`: reusable CUDA Runtime error-checking helper.
- `labs/01_gpu_basics/device_query/Makefile`: build commands for device query.
- `labs/01_gpu_basics/device_query/device_query.cu`: query and explain essential device properties.
- `labs/02_programming_model/vector_add/Makefile`: build commands for vector addition.
- `labs/02_programming_model/vector_add/vector_add.cu`: CPU reference, GPU kernel, boundary cases, and event timing.
- `labs/02_programming_model/index_mapping/Makefile`: build commands for index mapping.
- `labs/02_programming_model/index_mapping/index_mapping.cu`: print small 1D and 2D thread mappings.

### Task 1: Course Entry Point And Conventions

**Files:**
- Create: `course/README.md`
- Create: `course/术语与符号约定.md`
- Create: `course/实验方法与完成标准.md`

- [ ] **Step 1: Create the canonical course navigation**

Include:

```text
十卷顺序
每卷目标
当前可用章节
章节完成复选框
学习方式：阅读 -> 手写 -> 验证 -> 破坏 -> 测量 -> 表达
```

- [ ] **Step 2: Define global coordinate and matrix conventions**

The conventions must state:

```text
x = column = col = 横向
y = row = row = 纵向

input: height rows x width columns
row-major index: row * width + col
transpose output: width rows x height columns
```

- [ ] **Step 3: Define the practice contract**

Specify L1-L4 practice levels, CPU-reference requirements, boundary inputs, CUDA Event measurement, sanitizer/profiler evidence, and interview-style explanation checks.

- [ ] **Step 4: Validate the documents**

Run:

```bash
rg -n 'T[B]D|T[O]DO|待定' course
rg -n '^(<<<<<<<|=======|>>>>>>>)|[[:blank:]]+$' course
```

Expected: no output.

### Task 2: Shared CUDA Error Helper

**Files:**
- Create: `labs/common/cuda_check.cuh`

- [ ] **Step 1: Add the header**

Implement:

```cpp
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

inline void checkCuda(cudaError_t result, const char* expression,
                      const char* file, int line) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at %s:%d\n  expression: %s\n  reason: %s\n",
                 file, line, expression, cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(expression) \
  checkCuda((expression), #expression, __FILE__, __LINE__)
```

- [ ] **Step 2: Compile it through the first lab**

The header is verified by Tasks 3 and 5 rather than as a standalone translation unit.

### Task 3: Device Query Lab

**Files:**
- Create: `labs/01_gpu_basics/device_query/device_query.cu`
- Create: `labs/01_gpu_basics/device_query/Makefile`

- [ ] **Step 1: Implement essential device-property output**

Print:

```text
GPU name
compute capability
SM count
warp size
max threads per block
max block dimensions
max grid dimensions
registers per block and per SM
shared memory per block and per SM
global memory
memory clock and bus width
L2 cache size
concurrent kernel support
async engine count
```

- [ ] **Step 2: Add derived observations**

Compute and print:

```text
theoretical peak resident threads = SM count * maxThreadsPerMultiProcessor
approximate peak DRAM bandwidth from memory clock and bus width
```

Label the bandwidth value as a specification-derived approximation, not a measured result.

- [ ] **Step 3: Add the Makefile**

Use:

```make
NVCC ?= nvcc
TARGET := device_query
NVCCFLAGS := -O2 -std=c++17

all: $(TARGET)

$(TARGET): device_query.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) device_query.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
```

- [ ] **Step 4: Build and run**

Run:

```bash
make -C labs/01_gpu_basics/device_query clean all
./labs/01_gpu_basics/device_query/device_query
```

Expected: exit code 0 and a coherent device-property report.

### Task 4: Volume 1 Text

**Files:**
- Create: `course/volume01_gpu_basics/README.md`
- Create: `course/volume01_gpu_basics/01_CPU与GPU为什么不同.md`
- Create: `course/volume01_gpu_basics/02_GPU_SM_Warp_Thread.md`
- Create: `course/volume01_gpu_basics/03_内存层次第一印象.md`
- Create: `course/volume01_gpu_basics/04_延迟隐藏与大量线程.md`
- Create: `course/volume01_gpu_basics/05_T4设备观察实验.md`
- Create: `course/volume01_gpu_basics/06_卷一复习与面试题.md`

- [ ] **Step 1: Write the volume navigation**

List chapter order, required lab, completion conditions, and links.

- [ ] **Step 2: Explain CPU and GPU design goals**

Cover latency-oriented CPU design, throughput-oriented GPU design, workload suitability, Amdahl's law intuition, and why data transfer matters.

- [ ] **Step 3: Explain the minimum hierarchy**

Cover GPU, SM, block scheduling, warp, thread, logical versus physical concepts, and why a block cannot migrate after it starts.

- [ ] **Step 4: Give the first memory hierarchy model**

Cover register, shared, and global memory without prematurely teaching detailed cache policy. Include lifetime, sharing scope, capacity, and first-order speed intuition.

- [ ] **Step 5: Explain latency hiding**

Use a timeline example to contrast CPU cache/branch strategies with GPU warp switching. Clarify that occupancy and thread count are means, not final performance goals.

- [ ] **Step 6: Write the T4 observation lab guide**

Map each `device_query` field to its meaning. Require the learner to record actual output and answer interpretation questions.

- [ ] **Step 7: Add review and interview drills**

Include:

```text
concept checks
draw-from-memory tasks
short interview answers
misconception corrections
one written mini-report
```

- [ ] **Step 8: Validate links and formatting**

Run:

```bash
rg -n 'T[B]D|T[O]DO|待定' course/volume01_gpu_basics
rg -n '^(<<<<<<<|=======|>>>>>>>)|[[:blank:]]+$' course/volume01_gpu_basics
```

Expected: no output.

### Task 5: Vector Add Lab

**Files:**
- Create: `labs/02_programming_model/vector_add/vector_add.cu`
- Create: `labs/02_programming_model/vector_add/Makefile`

- [ ] **Step 1: Implement the CPU reference and verification**

Use deterministic input and report the first mismatch:

```cpp
void vectorAddCpu(const float* a, const float* b, float* c, int count);
bool verify(const float* expected, const float* actual, int count);
```

- [ ] **Step 2: Implement the CUDA kernel**

Use:

```cpp
__global__ void vectorAdd(const float* a, const float* b, float* c, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    c[index] = a[index] + b[index];
  }
}
```

- [ ] **Step 3: Implement the complete host lifecycle**

Include allocation, H2D copies, warmup, CUDA Event timing, launch error checking, synchronization, D2H copy, verification, and cleanup.

- [ ] **Step 4: Include boundary cases**

Run these sizes in one invocation:

```text
1
31
32
33
255
256
257
1,000,003
```

- [ ] **Step 5: Add the Makefile**

Use C++17 and `-O2`, with `all`, `run`, and `clean`.

- [ ] **Step 6: Build and run**

Run:

```bash
make -C labs/02_programming_model/vector_add clean all
./labs/02_programming_model/vector_add/vector_add
```

Expected: every size reports `PASS`.

### Task 6: Index Mapping Lab

**Files:**
- Create: `labs/02_programming_model/index_mapping/index_mapping.cu`
- Create: `labs/02_programming_model/index_mapping/Makefile`

- [ ] **Step 1: Implement a 1D mapping kernel**

For a small launch, print:

```text
blockIdx.x, threadIdx.x, blockDim.x, global index
```

- [ ] **Step 2: Implement a 2D mapping kernel**

Use the course convention:

```cpp
const int col = blockIdx.x * blockDim.x + threadIdx.x;
const int row = blockIdx.y * blockDim.y + threadIdx.y;
```

Print block coordinates, local coordinates, global row/column, and row-major linear index.

- [ ] **Step 3: Keep output deterministic enough to study**

Use one block for the detailed 2D print and explain that device `printf` ordering across threads is not a synchronization guarantee.

- [ ] **Step 4: Build and run**

Run:

```bash
make -C labs/02_programming_model/index_mapping clean all
./labs/02_programming_model/index_mapping/index_mapping
```

Expected: exit code 0 and visible 1D/2D coordinate mappings.

### Task 7: First Two Volume 2 Chapters

**Files:**
- Create: `course/volume02_programming_model/README.md`
- Create: `course/volume02_programming_model/01_第一个完整CUDA程序.md`
- Create: `course/volume02_programming_model/02_Grid_Block_Thread索引.md`

- [ ] **Step 1: Write the Volume 2 navigation**

Mark only the first two chapters as available and list the planned remaining topics without placeholder prose.

- [ ] **Step 2: Explain the complete CUDA program lifecycle**

Walk through vector add from CPU input to GPU allocation, copies, launch, verification, timing, and cleanup. Distinguish launch errors from asynchronous execution errors.

- [ ] **Step 3: Explain 1D mapping**

Derive:

```cpp
index = blockIdx.x * blockDim.x + threadIdx.x;
blocks = (count + threads - 1) / threads;
```

Use concrete values and boundary cases.

- [ ] **Step 4: Explain 2D mapping and block shapes**

Cover:

```text
x = column
y = row
row-major = row * width + col
block may be m x n
block shape and data tile shape are different concepts
```

- [ ] **Step 5: Attach practice tasks**

Require the learner to modify thread counts, remove the boundary check in a controlled small test, predict the failure, and use Compute Sanitizer when available.

### Task 8: Batch Verification

**Files:**
- Verify all files created by Tasks 1-7.

- [ ] **Step 1: Build all first-batch labs**

Run:

```bash
make -C labs/01_gpu_basics/device_query clean all
make -C labs/02_programming_model/vector_add clean all
make -C labs/02_programming_model/index_mapping clean all
```

Expected: all commands exit 0.

- [ ] **Step 2: Run all labs**

Run:

```bash
./labs/01_gpu_basics/device_query/device_query
./labs/02_programming_model/vector_add/vector_add
./labs/02_programming_model/index_mapping/index_mapping
```

Expected: device report, all vector-add cases `PASS`, and valid mapping output.

- [ ] **Step 3: Run document checks**

Run:

```bash
find course -type f -name '*.md' -print0 | xargs -0 grep -nE 'T[B]D|T[O]DO|待定'
find course -type f -name '*.md' -print0 | xargs -0 grep -nE '^(<<<<<<<|=======|>>>>>>>)|[[:blank:]]+$'
```

Expected: no output. Because `grep` returns 1 for no matches, execute these checks with explicit result handling.

- [ ] **Step 4: Verify Markdown fences and relative links**

Use a shell check to ensure every Markdown file has an even number of code fences and every relative Markdown link resolves to an existing file.

- [ ] **Step 5: Record repository limitation**

State in the completion report that `/home/qichengjie/workspace/cuda_study/cuda_deep_course` has no `.git`, so no commits were created.
