# GPU Kernel Engineering GEMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the standalone public `starboy520/gpu-kernel-engineering` repository and deliver a reproducible FP32 GEMM optimization ladder whose core CUDA kernels are handwritten by the repository author.

**Architecture:** The public repository is created as a sibling of `cuda_study`, with GEMM under `projects/gemm/`. A single runner owns input generation, CPU/cuBLAS references, validation, timing, and CSV output; each kernel file owns only one optimization stage and a launcher implementing a shared interface. Fast `float4` and `cp.async` paths explicitly fall back to the safe register-tiled implementation for unsupported shapes.

**Tech Stack:** CUDA 13.3, CUDA C++17, CMake 3.28 with Unix Makefiles, cuBLAS, compute-sanitizer, Nsight Compute, cuobjdump, Python 3 standard library, Git, GitHub CLI.

**Ownership rule:** The assistant may implement build files, runner/reference/validation code, scripts, documentation structure, and tests. The author must personally write every CUDA core kernel body: naive, shared-memory tiled, 2D register tiled, `float4`, and `cp.async`. Kernel tasks below are explicit author gates and intentionally provide contracts and staged hints rather than implementation bodies.

---

## Repository Map

The implementation creates this independent repository at `/home/qichengjie/workspace/gpu-kernel-engineering`:

```text
gpu-kernel-engineering/
├── CMakeLists.txt                    # Root project and CUDA architecture policy
├── LICENSE                           # MIT license
├── README.md                         # Stable résumé entry point and project index
├── .gitignore                        # Build/profiler/generated-file exclusions
└── projects/
    └── gemm/
        ├── CMakeLists.txt            # GEMM targets and tests
        ├── README.md                 # Reproduction guide and final optimization table
        ├── include/gemm/
        │   ├── cuda_check.hpp        # CUDA/cuBLAS error checking
        │   ├── kernel.hpp            # Shared launcher contract and registry API
        │   ├── reference.hpp         # CPU and cuBLAS reference declarations
        │   ├── runner.hpp            # CLI and experiment configuration
        │   └── validation.hpp        # Error metrics and pass policy
        ├── common/
        │   ├── reference.cpp         # Double-accumulation CPU GEMM
        │   ├── reference_cublas.cu   # Pedantic FP32 row-major cuBLAS reference
        │   └── validation.cpp        # Finite/error/worst-index analysis
        ├── kernels/
        │   ├── naive.cu              # Author-owned core kernel
        │   ├── shared_tiled.cu       # Author-owned core kernel
        │   ├── register_tiled.cu     # Author-owned core kernel and safe fallback
        │   ├── vectorized.cu         # Author-owned float4 fast path
        │   └── async_pipeline.cu     # Author-owned cp.async fast path
        ├── runner/
        │   ├── main.cu               # Unified executable entry point
        │   ├── registry.cu           # Kernel descriptors and launch lookup
        │   └── runner.cu             # Allocation, validation, timing, CSV output
        ├── tests/
        │   ├── common_tests.cpp       # CPU reference/validation unit tests
        │   └── correctness_cases.csv  # Required GPU shapes and expected path
        ├── scripts/
        │   ├── validate.sh            # Full correctness matrix
        │   ├── sanitize.sh            # memcheck/racecheck/synccheck gates
        │   ├── benchmark.sh           # Stable benchmark protocol
        │   ├── profile.sh             # Fixed ncu metric collection
        │   ├── extract_sass.sh        # Focused SASS evidence extraction
        │   └── render_results.py      # CSV-to-Markdown table generation
        ├── results/
        │   └── README.md              # Generated result schema; no invented numbers
        └── docs/
            ├── methodology.md         # Experiment protocol and hardware metadata
            ├── naive.md
            ├── shared-tiled.md
            ├── register-tiled.md
            ├── vectorized.md
            ├── async-pipeline.md
            └── cublas-baseline.md
```

No file in this repository may include a path into `cuda_study`.

---

### Task 1: Create the Standalone Public Repository Skeleton

**Files:**
- Create: `/home/qichengjie/workspace/gpu-kernel-engineering/CMakeLists.txt`
- Create: `/home/qichengjie/workspace/gpu-kernel-engineering/.gitignore`
- Create: `/home/qichengjie/workspace/gpu-kernel-engineering/LICENSE`
- Create: `/home/qichengjie/workspace/gpu-kernel-engineering/README.md`
- Create: `/home/qichengjie/workspace/gpu-kernel-engineering/projects/gemm/CMakeLists.txt`
- Create: `/home/qichengjie/workspace/gpu-kernel-engineering/projects/gemm/README.md`
- Create: `/home/qichengjie/workspace/gpu-kernel-engineering/projects/gemm/results/README.md`

- [ ] **Step 1: Create a sibling directory and initialize Git**

Run:

```bash
cd /home/qichengjie/workspace
mkdir gpu-kernel-engineering
cd gpu-kernel-engineering
git init -b main
```

Expected: an empty repository on branch `main`; `git rev-parse --show-toplevel` prints `/home/qichengjie/workspace/gpu-kernel-engineering`.

- [ ] **Step 2: Write the root CMake project**

Write `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.25)
project(gpu_kernel_engineering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

enable_testing()
add_subdirectory(projects/gemm)
```

- [ ] **Step 3: Write initial GEMM CMake and documentation stubs**

Write `projects/gemm/CMakeLists.txt` with a configuration-only target so the scaffold builds before source files exist:

```cmake
add_custom_target(gemm_scaffold ALL
  COMMAND ${CMAKE_COMMAND} -E echo "GEMM scaffold configured for sm_${CMAKE_CUDA_ARCHITECTURES}"
)
```

Write the root README with the title `GPU Kernel Engineering`, a one-paragraph repository purpose, an active link to `projects/gemm/`, and a roadmap listing FlashAttention and CUDA operators as future work without claiming implementation.

Write the GEMM README with these explicit labels:

```markdown
# FP32 GEMM Optimization Ladder

Status: active rebuild; published performance numbers will appear only after validation.

## Scope

Row-major `C = A × B`, FP32 accumulation, NVIDIA A100 (`sm_80`).

## Optimization ladder

1. Naive
2. Shared-memory tiling
3. 2D register tiling
4. `float4` vectorized load
5. `cp.async` double buffering
6. cuBLAS pedantic FP32 baseline
```

Write `results/README.md` stating that raw benchmark CSV files must contain hardware, toolchain, git commit, shape, kernel, selected path, latency, GFLOPS, correctness status, and timestamp.

- [ ] **Step 4: Write repository exclusions and MIT license**

`.gitignore` must contain:

```gitignore
/build/
**/build/
*.o
*.a
*.so
*.out
*.ncu-rep
*.nsys-rep
*.sass
*.ptx
__pycache__/
*.pyc
projects/gemm/results/raw/*.csv
projects/gemm/results/generated/*.md
```

Write `LICENSE` exactly as:

```text
MIT License

Copyright (c) 2026 qichengjie

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 5: Configure and build the empty scaffold**

Run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
```

Expected: configure succeeds and build prints `GEMM scaffold configured for sm_80`.

- [ ] **Step 6: Commit only the scaffold**

Run:

```bash
git add CMakeLists.txt .gitignore LICENSE README.md projects/gemm
git commit -m "chore: initialize GPU kernel portfolio"
```

Expected: one root commit with no copied CUDA source.

---

### Task 2: Add Test-Driven CPU Reference and Validation Infrastructure

**Files:**
- Create: `projects/gemm/include/gemm/reference.hpp`
- Create: `projects/gemm/include/gemm/validation.hpp`
- Create: `projects/gemm/common/reference.cpp`
- Create: `projects/gemm/common/validation.cpp`
- Create: `projects/gemm/tests/common_tests.cpp`
- Modify: `projects/gemm/CMakeLists.txt`

- [ ] **Step 1: Write failing unit tests first**

Define the public contracts:

```cpp
namespace gemm {
void reference_cpu(const float* a, const float* b, float* c,
                   int m, int n, int k);

struct ErrorMetrics {
  double max_abs;
  double max_rel;
  std::size_t worst_index;
  float expected;
  float actual;
  bool finite;
};

ErrorMetrics compare(const float* expected, const float* actual,
                     std::size_t count);
bool passes(const ErrorMetrics& metrics, double atol, double rtol);
}
```

Write tests that verify:

1. `reference_cpu` computes a hand-calculated `2×3` by `3×2` product.
2. `compare` finds the exact worst index.
3. `compare` marks NaN and Inf as non-finite.
4. `passes` accepts when either absolute or relative combined tolerance holds and rejects a clear mismatch.

The hand-calculated case must use:

```text
A = [1, 2, 3; 4, 5, 6]
B = [7, 8; 9, 10; 11, 12]
C = [58, 64; 139, 154]
```

Create `common/reference.cpp` and `common/validation.cpp` with only their corresponding header include and no function definitions. This allows configuration and compilation to reach the intentional linker failure.

- [ ] **Step 2: Add the test target and verify the expected link failure**

Replace the scaffold-only GEMM CMake with targets for `gemm_common` and `gemm_common_tests`, listing the two intentionally definition-free implementation files created in Step 1.

Run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
```

Expected: FAIL because `reference_cpu`, `compare`, and `passes` are not yet defined.

- [ ] **Step 3: Implement the CPU reference**

Implement exactly row-major indexing:

```cpp
for (int row = 0; row < m; ++row) {
  for (int col = 0; col < n; ++col) {
    double sum = 0.0;
    for (int inner = 0; inner < k; ++inner) {
      sum += static_cast<double>(a[row * k + inner]) *
             static_cast<double>(b[inner * n + col]);
    }
    c[row * n + col] = static_cast<float>(sum);
  }
}
```

Reject null pointers and non-positive dimensions with `std::invalid_argument`.

- [ ] **Step 4: Implement deterministic validation**

For each element, compute:

```cpp
const double abs_error = std::abs(double(actual[i]) - double(expected[i]));
const double rel_error = abs_error / std::max(std::abs(double(expected[i])), 1.0e-12);
```

Track the element with largest absolute error. `passes` must evaluate the worst element using:

```cpp
metrics.finite &&
std::abs(double(metrics.actual) - double(metrics.expected)) <=
    atol + rtol * std::abs(double(metrics.expected))
```

- [ ] **Step 5: Run the unit tests**

Run:

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: `gemm_common_tests` passes with 100% tests passed.

- [ ] **Step 6: Commit the common correctness layer**

```bash
git add projects/gemm/CMakeLists.txt projects/gemm/include projects/gemm/common projects/gemm/tests/common_tests.cpp
git commit -m "test: add GEMM reference and validation"
```

---

### Task 3: Add CUDA Error Handling, Kernel Contract, and Unified Runner

**Files:**
- Create: `projects/gemm/include/gemm/cuda_check.hpp`
- Create: `projects/gemm/include/gemm/kernel.hpp`
- Create: `projects/gemm/include/gemm/runner.hpp`
- Create: `projects/gemm/runner/main.cu`
- Create: `projects/gemm/runner/runner.cu`
- Create: `projects/gemm/runner/registry.cu`
- Create: `projects/gemm/tests/correctness_cases.csv`
- Modify: `projects/gemm/CMakeLists.txt`

- [ ] **Step 1: Define the shared kernel contract**

Use these exact types:

```cpp
namespace gemm {
struct Problem { int m; int n; int k; };

struct LaunchResult {
  const char* selected_path;
  bool used_fallback;
};

using LaunchFn = LaunchResult (*)(const float*, const float*, float*,
                                  Problem, cudaStream_t);

struct KernelDescriptor {
  const char* name;
  LaunchFn launch;
  bool author_kernel;
};

const KernelDescriptor* find_kernel(std::string_view name);
std::vector<KernelDescriptor> registered_kernels();
}
```

The registry initially returns an empty vector. It must not contain fake implementations.

- [ ] **Step 2: Add CUDA/cuBLAS error macros**

Provide `CUDA_CHECK(expression)` and `CUBLAS_CHECK(expression)` macros that throw `std::runtime_error` containing the expression, source file, source line, and runtime error string/status. Do not print-and-continue.

- [ ] **Step 3: Write runner CLI failure tests before GPU kernels exist**

The runner must support:

```text
--list
--kernel <name>
--m <positive int> --n <positive int> --k <positive int>
--mode validate|benchmark
--warmup <non-negative int>
--iterations <positive int>
--seed <unsigned int>
--csv <path>
```

Run after adding the runner target:

```bash
./build/projects/gemm/gemm_runner --kernel naive --m 17 --n 19 --k 23 --mode validate
```

Expected: non-zero exit and `unknown kernel: naive`. This is the failing acceptance test for the first author kernel.

Also verify malformed dimensions:

```bash
./build/projects/gemm/gemm_runner --kernel naive --m 0 --n 19 --k 23 --mode validate
```

Expected: non-zero exit and `m, n, and k must be positive`.

- [ ] **Step 4: Implement runner infrastructure without any GEMM kernel body**

The runner must:

1. Parse all options without external libraries.
2. Use checked `std::size_t` multiplication before allocation.
3. Fill A and B with deterministic `std::mt19937(seed)` values in `[-0.5, 0.5]`.
4. Allocate device buffers with RAII wrappers.
5. Compute CPU reference in validate mode while problem FLOPs are below a documented threshold.
6. Launch warmups, check `cudaPeekAtLastError`, and synchronize before validation.
7. Time only repeated kernel launches with CUDA events.
8. Print one stable line containing kernel, selected path, shape, pass/fail, max errors, milliseconds, and GFLOPS.
9. Return non-zero for invalid arguments, unknown kernels, CUDA errors, or validation failures.

Use small-reference tolerances `atol=1e-3` and `rtol=1e-3`. Passing is based on the combined tolerance for the worst-absolute-error element; still report global `max_rel` as diagnostic data.

- [ ] **Step 5: Populate the required correctness matrix**

Write CSV rows with columns `kernel,m,n,k,expected_path`. The special value `all` expands to every registered author kernel; include at least:

```csv
kernel,m,n,k,expected_path
all,1,1,1,any
all,2,3,4,any
all,17,19,23,any
all,31,32,33,any
all,32,32,32,any
all,33,65,17,any
all,63,64,65,any
all,64,64,64,any
all,65,127,130,any
all,127,130,33,any
all,130,127,65,any
```

- [ ] **Step 6: Build and verify the intentional empty-registry failure**

Run:

```bash
cmake --build build -j
./build/projects/gemm/gemm_runner --list
./build/projects/gemm/gemm_runner --kernel naive --m 17 --n 19 --k 23 --mode validate
```

Expected: build succeeds, `--list` reports no kernels, and the naive command fails as unknown.

- [ ] **Step 7: Commit the reusable runner**

```bash
git add projects/gemm
git commit -m "feat: add unified GEMM runner"
```

---

### Task 4: Author Gate — Handwrite and Validate Naive GEMM

**Files:**
- Create: `projects/gemm/kernels/naive.cu`
- Modify: `projects/gemm/include/gemm/kernel.hpp`
- Modify: `projects/gemm/runner/registry.cu`
- Modify: `projects/gemm/CMakeLists.txt`
- Create: `projects/gemm/docs/naive.md`

- [ ] **Step 1: Preserve the failing acceptance test**

Run:

```bash
./build/projects/gemm/gemm_runner --kernel naive --m 17 --n 19 --k 23 --mode validate
```

Expected: FAIL with `unknown kernel: naive`.

- [ ] **Step 2: Author writes the kernel from a blank file**

Required contract:

```cpp
LaunchResult launch_naive(const float* a, const float* b, float* c,
                          Problem problem, cudaStream_t stream);
```

Behavioral requirements:

- One CUDA thread owns one output element.
- Bounds checks support arbitrary positive rectangular shapes.
- Each valid thread accumulates K products in FP32.
- Launch uses the supplied stream.
- Result is `{ "naive", false }`.

Hints are revealed only if requested:

1. Level 1: map block x to columns and block y to rows.
2. Level 2: use a 2D block such as `dim3(16, 16)` and ceil-div grid dimensions.
3. Level 3: inspect only the index equations and launch configuration; do not provide a complete kernel body.

- [ ] **Step 3: Register and compile the author implementation**

Add `naive.cu` to the kernel target and add descriptor `{ "naive", launch_naive, true }`.

Apply the same CUDA compile options to the kernel target for every later stage:

```cmake
target_compile_options(gemm_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-warn-spills>
)
```

Build with compiler resource output:

```bash
cmake --build build -j --verbose
```

Expected: no CUDA compilation errors and no unresolved launcher symbol.

- [ ] **Step 4: Run the full small-shape validation matrix for naive**

For every CSV row, invoke validate mode with `--kernel naive`. Expected: every case reports `PASS`, `selected_path=naive`, and no fallback.

- [ ] **Step 5: Run memory safety validation**

```bash
compute-sanitizer --tool memcheck --error-exitcode=99 \
  ./build/projects/gemm/gemm_runner --kernel naive --m 130 --n 127 --k 65 --mode validate
```

Expected: `ERROR SUMMARY: 0 errors` and runner PASS.

- [ ] **Step 6: Review before committing**

Review only; do not rewrite the author kernel. Check row-major indexing, integer bounds, stream use, launch error propagation, and absence of copied teaching comments. Author fixes findings personally.

- [ ] **Step 7: Record the baseline reasoning and commit**

In `docs/naive.md`, record mapping, global-memory reuse limitation, arithmetic-intensity hypothesis, exact validation command, and no performance claim yet.

```bash
git add projects/gemm
git commit -m "feat: implement naive FP32 GEMM"
```

---

### Task 5: Author Gate — Handwrite Shared-Memory Tiled GEMM

**Files:**
- Create: `projects/gemm/kernels/shared_tiled.cu`
- Modify: `projects/gemm/include/gemm/kernel.hpp`
- Modify: `projects/gemm/runner/registry.cu`
- Modify: `projects/gemm/CMakeLists.txt`
- Create: `projects/gemm/docs/shared-tiled.md`

- [ ] **Step 1: Add and run the failing shared-kernel acceptance test**

```bash
./build/projects/gemm/gemm_runner --kernel shared --m 33 --n 65 --k 17 --mode validate
```

Expected: FAIL with `unknown kernel: shared`.

- [ ] **Step 2: Author writes the shared-memory kernel**

Required launcher result: `{ "shared", false }`.

Required signature:

```cpp
LaunchResult launch_shared_tiled(const float* a, const float* b, float* c,
                                 Problem problem, cudaStream_t stream);
```

Behavioral requirements:

- One block owns one output tile.
- Threads cooperatively load one A tile and one B tile.
- Out-of-range loads write zero to shared memory.
- Every shared-memory producer/consumer phase has correct block synchronization.
- K tail tiles and arbitrary M/N tails pass without a fallback.
- Accumulation stays FP32.

Hint ladder:

1. Level 1: separate global coordinates from tile-local coordinates.
2. Level 2: load A with `(row, tile_k)` and B with `(tile_k, col)`, storing zero for invalid coordinates.
3. Level 3: inspect one failing phase or index equation only; do not provide the finished kernel.

- [ ] **Step 3: Register, build, and run correctness tests**

Run all CSV shapes for `shared`, including `1×1×1`, `33×65×17`, and `130×127×65`.

Expected: all PASS with path `shared`.

- [ ] **Step 4: Run synchronization and memory tools**

```bash
compute-sanitizer --tool memcheck --error-exitcode=99 ./build/projects/gemm/gemm_runner --kernel shared --m 130 --n 127 --k 65 --mode validate
compute-sanitizer --tool racecheck --error-exitcode=99 ./build/projects/gemm/gemm_runner --kernel shared --m 65 --n 127 --k 33 --mode validate
compute-sanitizer --tool synccheck --error-exitcode=99 ./build/projects/gemm/gemm_runner --kernel shared --m 65 --n 127 --k 33 --mode validate
```

Expected: zero errors from all tools.

- [ ] **Step 5: Benchmark only after validation**

Run naive and shared for aligned `1024³` and `2048³` with identical seed, warmup, and iterations. Do not add numbers to the root README yet.

- [ ] **Step 6: Document evidence and commit**

Record expected DRAM-traffic reduction, measured speedup, exact commands, and any discrepancy from the hypothesis in `docs/shared-tiled.md`.

```bash
git add projects/gemm
git commit -m "feat: add shared-memory tiled GEMM"
```

---

### Task 6: Author Gate — Handwrite 2D Register-Tiled GEMM

**Files:**
- Create: `projects/gemm/kernels/register_tiled.cu`
- Modify: `projects/gemm/include/gemm/kernel.hpp`
- Modify: `projects/gemm/runner/registry.cu`
- Modify: `projects/gemm/CMakeLists.txt`
- Create: `projects/gemm/docs/register-tiled.md`

- [ ] **Step 1: Establish the failing registration test**

```bash
./build/projects/gemm/gemm_runner --kernel register --m 64 --n 64 --k 64 --mode validate
```

Expected: unknown kernel failure.

- [ ] **Step 2: Author writes the register-tiled kernel**

Required launcher result: `{ "register", false }`.

Required signature:

```cpp
LaunchResult launch_register_tiled(const float* a, const float* b, float* c,
                                   Problem problem, cudaStream_t stream);
```

Behavioral requirements:

- A block owns a `BM×BN` output tile.
- A thread owns a `TM×TN` accumulator tile.
- Threads cooperatively load shared tiles independently of compute mapping.
- Inner K loop loads fragments from shared memory into registers and performs an outer product.
- Boundary loads zero-fill; boundary stores are guarded.
- Tile constants are named and documented with compile-time resource assumptions.

Start with one conservative configuration. Tune only after the untuned version passes all tests.

Hint ladder:

1. Level 1: derive `blockDim=(BN/TN, BM/TM)` and verify it is legal.
2. Level 2: flatten threads for cooperative loads so loading does not assume `TM==TN`.
3. Level 3: review register-fragment indexing and one outer-product iteration without supplying the full implementation.

- [ ] **Step 3: Validate the untuned implementation**

Run every CSV shape plus at least one rectangular aligned shape such as `M=256,N=512,K=128`.

Expected: all PASS and no fallback.

- [ ] **Step 4: Run sanitizer gates**

Run memcheck, racecheck, and synccheck on both aligned and non-aligned shapes. Expected: zero errors.

- [ ] **Step 5: Perform a bounded parameter experiment**

Compare a small declared matrix of configurations, for example `TM×TN` in `{4×4, 8×4, 8×8}` while keeping legal block resources. For every compiled configuration, record registers/thread, correctness, occupancy, and GFLOPS. Reject configurations that fail launch or correctness instead of omitting them.

- [ ] **Step 6: Select and explain one default configuration**

The selection criterion is highest stable aligned-shape performance after correctness, not highest occupancy. Record why the losing configurations lose.

- [ ] **Step 7: Review and commit**

Review index ownership, shared bank behavior, register count, occupancy, edge stores, and configuration invariants. Author applies fixes.

```bash
git add projects/gemm
git commit -m "feat: add 2D register-tiled GEMM"
```

---

### Task 7: Author Gate — Add `float4` Fast Path and Explicit Fallback

**Files:**
- Create: `projects/gemm/kernels/vectorized.cu`
- Modify: `projects/gemm/include/gemm/kernel.hpp`
- Modify: `projects/gemm/runner/registry.cu`
- Modify: `projects/gemm/CMakeLists.txt`
- Modify: `projects/gemm/tests/correctness_cases.csv`
- Create: `projects/gemm/docs/vectorized.md`

- [ ] **Step 1: Add path-sensitive failing cases**

Add cases expecting `fast-float4` for an aligned shape and `fallback-register` for a non-aligned shape:

```csv
vectorized,128,128,128,fast-float4
vectorized,130,127,65,fallback-register
```

Run vectorized validation. Expected: unknown kernel failure before implementation.

- [ ] **Step 2: Define the fast-path predicate before writing the kernel**

The launcher must use one named predicate that checks all assumptions required by the actual implementation. At minimum it must account for 16-byte row-start alignment and vector-width divisibility. The predicate and kernel must agree; do not claim broader support than tested.

Required launcher behavior:

```text
supported shape -> run vectorized kernel, return {"fast-float4", false}
unsupported shape -> call launch_register_tiled, return {"fallback-register", true}
```

Required signature:

```cpp
LaunchResult launch_vectorized(const float* a, const float* b, float* c,
                               Problem problem, cudaStream_t stream);
```

- [ ] **Step 3: Author writes the vectorized load kernel**

Behavioral requirements:

- Global A/B loads use valid, aligned `float4` operations only.
- Shared-memory layout and scalar unpacking are explicit.
- The compute mapping remains author-understood register tiling.
- No vector load can cross an allocation or row boundary.
- Tail handling is delegated to the declared fallback, not undefined access.

Hint ladder:

1. Level 1: prove alignment for every row start, not only the allocation base.
2. Level 2: derive vector indices in units of four floats and separately derive shared scalar destinations.
3. Level 3: inspect one load loop and predicate mismatch; never supply the complete kernel.

- [ ] **Step 4: Validate both execution paths**

Run aligned and non-aligned path-sensitive cases and assert the printed path exactly matches the CSV expectation. Compare fallback output to the same CPU reference.

- [ ] **Step 5: Reproduce and close the historical N=130 defect**

```bash
compute-sanitizer --tool memcheck --error-exitcode=99 \
  ./build/projects/gemm/gemm_runner --kernel vectorized --m 130 --n 130 --k 130 --mode validate
```

Expected: runner PASS, selected path `fallback-register`, zero sanitizer errors. This explicitly prevents recurrence of the old misaligned 16-byte read.

- [ ] **Step 6: Verify SASS evidence on the aligned fast path**

Compile with line information, dump SASS, and confirm the fast-path kernel contains a 128-bit global-load form appropriate to the generated architecture. Save only a concise extract and command, not the full binary dump.

- [ ] **Step 7: Benchmark and commit**

Compare register and vectorized fast paths on the same aligned shapes. Record whether instruction reduction produces a wall-clock gain.

```bash
git add projects/gemm
git commit -m "feat: add vectorized GEMM fast path"
```

---

### Task 8: Author Gate — Add `cp.async` Double Buffering

**Files:**
- Create: `projects/gemm/kernels/async_pipeline.cu`
- Modify: `projects/gemm/include/gemm/kernel.hpp`
- Modify: `projects/gemm/runner/registry.cu`
- Modify: `projects/gemm/CMakeLists.txt`
- Modify: `projects/gemm/tests/correctness_cases.csv`
- Create: `projects/gemm/docs/async-pipeline.md`

- [ ] **Step 1: Add path-sensitive async acceptance cases**

Expected paths:

```csv
async,128,128,128,fast-async
async,130,127,65,fallback-register
```

Run before registration. Expected: unknown kernel failure.

- [ ] **Step 2: Author writes a two-stage asynchronous pipeline**

Required behavior:

- Use two shared-memory stages.
- Establish a correct prologue, steady-state loop, and epilogue.
- Never overwrite a stage still consumed by the block.
- Wait for the required producer stage before reading it.
- Synchronize block-visible shared-memory consumption correctly.
- Aligned supported shapes return `fast-async`.
- Unsupported shapes call register fallback and return `fallback-register`.

Required signature:

```cpp
LaunchResult launch_async_pipeline(const float* a, const float* b, float* c,
                                   Problem problem, cudaStream_t stream);
```

Hint ladder:

1. Level 1: draw stage ownership for iterations 0, 1, and 2 before coding.
2. Level 2: separate `prefetch(tile, stage)` from `compute(stage)` and state which stage is safe at each point.
3. Level 3: review only the failing prologue/wait/stage transition; do not write the final pipeline.

- [ ] **Step 3: Validate edge tile counts**

Test shapes producing one tile, two tiles, an odd number of tiles, and a non-aligned fallback. Expected: all PASS with exact selected paths.

- [ ] **Step 4: Run all sanitizer tools**

Use memcheck, racecheck, synccheck, and initcheck on at least one multi-stage aligned shape. Expected: zero reported errors.

- [ ] **Step 5: Verify generated asynchronous-copy evidence**

Use cuobjdump to confirm the target kernel generates the expected Ampere global-to-shared asynchronous instruction form. If it does not, do not describe the implementation as proven `cp.async`; investigate compilation and API usage first.

- [ ] **Step 6: Benchmark without assuming a win**

Compare vectorized and async versions on the same shapes. Record registers, occupancy, long-scoreboard stalls, instruction count, and latency. If async is slower, preserve the result and explain whether the kernel was already compute-bound, copy granularity was too small, or occupancy/resource pressure increased.

- [ ] **Step 7: Close the historical non-aligned correctness failure and commit**

Run `130³` through memcheck. Expected: safe fallback and PASS, unlike the old double-buffering implementation.

```bash
git add projects/gemm
git commit -m "feat: add asynchronous GEMM pipeline"
```

---

### Task 9: Add a Pedantic FP32 cuBLAS Reference and Baseline

**Files:**
- Create: `projects/gemm/common/reference_cublas.cu`
- Modify: `projects/gemm/include/gemm/reference.hpp`
- Modify: `projects/gemm/runner/runner.cu`
- Modify: `projects/gemm/runner/registry.cu`
- Modify: `projects/gemm/CMakeLists.txt`
- Create: `projects/gemm/docs/cublas-baseline.md`

- [ ] **Step 1: Write a failing row-major cuBLAS comparison test**

For a small rectangular problem, compare the cuBLAS result to `reference_cpu`. Expected before implementation: unresolved or unavailable cuBLAS reference path.

- [ ] **Step 2: Implement row-major cuBLAS correctly**

Use `cublasCreate`, bind the runner stream, call `cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)`, and map row-major `C=A×B` through the column-major identity `Cᵀ=Bᵀ×Aᵀ`. Use `cublasSgemm` with B passed before A and dimensions `(n, m, k)`.

The wrapper must expose both reference generation and a timed `cublas-fp32` descriptor. It must report path `cublas-pedantic-fp32`.

- [ ] **Step 3: Verify TF32 is not used**

Record the explicit math-mode call and profile a baseline launch. Do not compare against default cuBLAS without documenting its math mode.

- [ ] **Step 4: Validate rectangular and large shapes**

Compare cuBLAS to CPU on small rectangular cases. For `512`, `1024`, `2048`, and optionally `4096`, use cuBLAS as the reference for hand-written kernels to avoid slow CPU reference execution.

Use large-reference tolerances `atol=5e-2` and `rtol=2e-3`, documented as fixed policy for differing FP32 reduction order.

- [ ] **Step 5: Commit the fair baseline**

```bash
git add projects/gemm
git commit -m "feat: add pedantic FP32 cuBLAS baseline"
```

---

### Task 10: Automate Validation and Sanitizer Gates

**Files:**
- Create: `projects/gemm/scripts/validate.sh`
- Create: `projects/gemm/scripts/sanitize.sh`
- Modify: `projects/gemm/CMakeLists.txt`
- Modify: `projects/gemm/README.md`

- [ ] **Step 1: Write the validation script and make a deliberate failure observable**

The script must use `set -euo pipefail`, locate the repository root relative to itself, and parse `correctness_cases.csv`. A row whose kernel is `all` expands to every registered author kernel; a named row runs only that kernel. Verify both PASS status and exact expected path when the CSV value is not `any`.

Temporarily pass an invalid kernel name. Expected: script exits non-zero. Remove the deliberate failure after observing it.

- [ ] **Step 2: Add CTest integration**

Add a `gemm_correctness` test invoking `validate.sh` against the built runner. Mark it with a `gpu` label.

- [ ] **Step 3: Write sanitizer tiers**

`sanitize.sh quick` runs memcheck for every kernel on one aligned and one non-aligned shape.

`sanitize.sh full` additionally runs racecheck, synccheck, and initcheck for shared/register/vectorized/async kernels. Every command uses `--error-exitcode=99`.

- [ ] **Step 4: Run the complete gates**

```bash
ctest --test-dir build --output-on-failure
projects/gemm/scripts/validate.sh ./build/projects/gemm/gemm_runner
projects/gemm/scripts/sanitize.sh quick ./build/projects/gemm/gemm_runner
projects/gemm/scripts/sanitize.sh full ./build/projects/gemm/gemm_runner
```

Expected: all commands exit 0. If full sanitizer runtime is long, it remains a release gate rather than being weakened.

- [ ] **Step 5: Document and commit reproducible validation**

```bash
git add projects/gemm
git commit -m "test: automate GEMM correctness and sanitizer gates"
```

---

### Task 11: Automate Stable Benchmark Collection and Table Rendering

**Files:**
- Create: `projects/gemm/scripts/benchmark.sh`
- Create: `projects/gemm/scripts/render_results.py`
- Modify: `projects/gemm/runner/runner.cu`
- Modify: `projects/gemm/results/README.md`
- Create at run time: `projects/gemm/results/raw/a100-fp32.csv`
- Create at run time: `projects/gemm/results/generated/a100-fp32.md`

- [ ] **Step 1: Define a stable CSV schema and test malformed data**

Use columns:

```text
timestamp,git_commit,gpu,cuda,nvcc,kernel,path,m,n,k,warmup,iterations,latency_ms,gflops,passed,max_abs,max_rel
```

Write a small malformed fixture missing `gflops`; run the renderer and expect a non-zero exit with `missing required column: gflops`. Remove the fixture after the test.

- [ ] **Step 2: Implement runner CSV append mode**

The runner creates the header for a new file and appends one row per measured kernel/shape. It must JSON/CSV-escape string fields correctly or restrict generated metadata to safe CSV values and document that restriction.

- [ ] **Step 3: Implement the benchmark protocol**

`benchmark.sh` must:

- verify the working tree is clean before an official run;
- capture git commit, GPU name, CUDA runtime, nvcc version, and UTC timestamp;
- run kernels in a fixed order;
- use shapes `1024³`, `2048³`, and `4096³` when memory permits;
- use the same seed, 10 warmups, and at least 50 timed iterations;
- repeat each point three times and retain the median latency;
- stop on validation or CUDA failure;
- never benchmark a fallback row as if it were the fast-path result.

- [ ] **Step 4: Implement Markdown rendering**

The Python standard-library renderer groups rows by shape and prints latency, GFLOPS, speedup over previous stage, and percentage of cuBLAS FP32. It must not invent or interpolate missing stages.

- [ ] **Step 5: Run a smoke benchmark before the official run**

Use `512³`, 2 warmups, and 3 iterations to validate the pipeline. Expected: CSV and Markdown are generated and all rows report PASS.

- [ ] **Step 6: Commit automation, not preliminary performance claims**

```bash
git add projects/gemm/scripts projects/gemm/runner projects/gemm/results/README.md
git commit -m "feat: automate GEMM benchmark reporting"
```

Do not commit smoke benchmark numbers to the final README.

---

### Task 12: Collect Nsight Compute and SASS Evidence

**Files:**
- Create: `projects/gemm/scripts/profile.sh`
- Create: `projects/gemm/scripts/extract_sass.sh`
- Modify: `projects/gemm/docs/naive.md`
- Modify: `projects/gemm/docs/shared-tiled.md`
- Modify: `projects/gemm/docs/register-tiled.md`
- Modify: `projects/gemm/docs/vectorized.md`
- Modify: `projects/gemm/docs/async-pipeline.md`
- Create: `projects/gemm/docs/methodology.md`

- [ ] **Step 1: Define profiler commands before collecting conclusions**

`profile.sh` accepts kernel and shape, performs one warmup-aware target launch, and collects only named metrics required for the hypothesis. Include:

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
sm__warps_active.avg.pct_of_peak_sustained_active
launch__registers_per_thread
lts__throughput.avg.pct_of_peak_sustained_elapsed
dram__throughput.avg.pct_of_peak_sustained_elapsed
l1tex__throughput.avg.pct_of_peak_sustained_elapsed
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
```

Add shared bank-conflict metrics only for kernels that use shared memory and verify metric names with `ncu --query-metrics` on the installed version.

- [ ] **Step 2: Make profiler output non-versioned by default**

Write full `.ncu-rep` files under a gitignored temporary/results directory. Commit only concise Markdown tables, exact commands, hardware metadata, and selected metric values.

- [ ] **Step 3: Extract focused SASS evidence**

`extract_sass.sh` must build the selected target and use `cuobjdump --dump-sass` to extract only the target kernel region and counts/snippets for FFMA, wide loads, spills, and Ampere async-copy instructions. Full `.sass` output stays ignored.

- [ ] **Step 4: Profile each optimization as a hypothesis test**

For every stage, fill the same four headings:

1. Previous bottleneck hypothesis
2. Change made
3. Metric evidence
4. Wall-clock result and interpretation

Do not call a subsystem the bottleneck from one percentage alone. Correlate throughput, stalls, occupancy, registers, and grid size.

- [ ] **Step 5: Preserve negative results**

If async is slower than vectorized, keep the row and explain it. If a parameter configuration fails resources or correctness, record it as a rejected experiment rather than deleting it from the narrative.

- [ ] **Step 6: Commit evidence separately from kernels**

```bash
git add projects/gemm/scripts/profile.sh projects/gemm/scripts/extract_sass.sh projects/gemm/docs
git commit -m "docs: add GEMM profiler and SASS evidence"
```

---

### Task 13: Run the Official A100 Release Experiment

**Files:**
- Create: `projects/gemm/results/raw/a100-fp32.csv`
- Create: `projects/gemm/results/generated/a100-fp32.md`
- Modify: `projects/gemm/docs/methodology.md`

- [ ] **Step 1: Verify release preconditions**

Run:

```bash
git status --short
ctest --test-dir build --output-on-failure
projects/gemm/scripts/validate.sh ./build/projects/gemm/gemm_runner
projects/gemm/scripts/sanitize.sh full ./build/projects/gemm/gemm_runner
```

Expected: clean working tree before generated results; all tests and sanitizers pass.

- [ ] **Step 2: Capture environment metadata**

Record exact GPU model, driver, CUDA runtime, nvcc version, clock/persistence state when available, CMake configure command, compile flags, git commit, and date. Do not state generic A100 peak numbers without identifying PCIe/SXM model and source.

- [ ] **Step 3: Execute the official benchmark**

Run the fixed benchmark protocol without other GPU workloads. Keep all raw repetitions until the median table is generated.

Expected: every official row is correct, uses its intended fast path, and has finite positive latency/GFLOPS.

- [ ] **Step 4: Re-run suspicious points**

Any point with large variance, thermal/clock anomaly, fallback selection, or unexpected regression is invalid until repeated or explained. Do not manually edit the CSV to improve results.

- [ ] **Step 5: Render and review results**

Verify all README-bound numbers exist in raw CSV, percentage calculations use pedantic FP32 cuBLAS for the same shape, and no old `cuda_study` number appears.

- [ ] **Step 6: Commit reproducible results**

Temporarily force-add the reviewed raw and generated release files if the ignore policy excludes raw files, or adjust `.gitignore` to allow the single canonical dataset while continuing to ignore ad-hoc runs.

```bash
git add -f projects/gemm/results/raw/a100-fp32.csv projects/gemm/results/generated/a100-fp32.md
git add projects/gemm/docs/methodology.md
git commit -m "perf: publish A100 FP32 GEMM results"
```

---

### Task 14: Finish Public Documentation and Publish GitHub Repository

**Files:**
- Modify: `README.md`
- Modify: `projects/gemm/README.md`
- Modify: `projects/gemm/results/README.md`
- Verify: all tracked repository files

- [ ] **Step 1: Write the final root résumé entry point**

Include:

- one-sentence technical positioning;
- tested hardware/toolchain;
- GEMM project link;
- one compact table of verified headline results;
- methodology summary;
- roadmap that clearly labels unimplemented projects;
- no badges claiming CI/GPU tests that do not exist.

- [ ] **Step 2: Complete the GEMM README from generated evidence**

Include build, validation, sanitizer, benchmark, ncu, and SASS reproduction commands. Link every performance claim to the generated result table or relevant evidence document. State fast-path constraints and fallback behavior.

- [ ] **Step 3: Run an AI/template-trace editorial audit**

Search tracked public files for instructional residue:

```bash
git grep -n -E '你来写|[T]ODO|[F]IXME|Day[0-9]|Week[0-9]|教学版|照搬|由你实现'
```

Expected: no matches in public source or documentation. Technical use of the term “AI” is allowed only when it means arithmetic intensity and is defined on first use.

Also verify that comments explain invariants and trade-offs rather than narrating every line.

- [ ] **Step 4: Verify standalone reproducibility from a fresh clone directory**

Create a temporary local clone from the repository path, then run configure, build, common tests, one GPU correctness case, and one smoke benchmark without referencing `cuda_study`.

Expected: every command succeeds from the clone.

- [ ] **Step 5: Perform final repository verification**

```bash
git status --short
git ls-files | grep -E '(build/|\.o$|\.ncu-rep$|\.sass$)' && exit 1 || true
ctest --test-dir build --output-on-failure
projects/gemm/scripts/validate.sh ./build/projects/gemm/gemm_runner
```

Expected: clean tracked state, no build/profiler artifacts, all tests pass.

- [ ] **Step 6: Commit final documentation**

```bash
git add README.md projects/gemm/README.md projects/gemm/results/README.md
git commit -m "docs: publish GEMM optimization portfolio"
```

- [ ] **Step 7: Create the public GitHub repository and push**

The name is currently available and GitHub CLI is authenticated as `starboy520` over SSH.

Run:

```bash
gh repo create starboy520/gpu-kernel-engineering \
  --public \
  --source=. \
  --remote=origin \
  --push \
  --description "Profiler-driven CUDA kernel optimization projects"
```

Expected:

```text
https://github.com/starboy520/gpu-kernel-engineering
```

- [ ] **Step 8: Verify the public repository**

Run:

```bash
gh repo view starboy520/gpu-kernel-engineering --web
git remote -v
git status --short
```

Expected: `origin` points to the SSH GitHub repository, main is pushed, and the working tree is clean.

---

## Final Release Checklist

- [ ] Public repository contains no copied learning-history source.
- [ ] Every core CUDA kernel was written and can be explained by the author.
- [ ] Every required correctness shape passes.
- [ ] `float4` and async non-aligned cases select safe fallback.
- [ ] memcheck, racecheck, synccheck, and initcheck release gates pass.
- [ ] cuBLAS comparison explicitly uses `CUBLAS_PEDANTIC_MATH`.
- [ ] Official numbers come only from the new repository and canonical CSV.
- [ ] Profiler and SASS claims have commands and compact evidence.
- [ ] Negative optimization results are retained and explained.
- [ ] Fresh-clone build and smoke test pass without `cuda_study`.
- [ ] Root README is suitable as the stable résumé link.
