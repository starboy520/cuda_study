# Deepen CUDA Volumes 1-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Volumes 1 and 2 from introductory notes into a self-contained, deeply explained CUDA foundation with extensive runnable samples.

**Architecture:** Volume 1 explains only the hardware concepts needed to reason about early CUDA programs, using T4 observations and numerical examples. Volume 2 follows the complete CUDA program lifecycle, adds language, memory, execution, compilation, timing, debugging, and 2D/GEMM samples.

**Tech Stack:** Markdown, Mermaid, CUDA C++17, NVCC, PTX inspection, CUDA Runtime API, Make, Compute Sanitizer

---

### Task 1: Deepen Volume 1

- [ ] Expand CPU/GPU design trade-offs, parallelism types, latency/throughput, transfer economics, Amdahl examples, and workload diagnosis.
- [ ] Expand logical versus physical hierarchy, block scheduling, warp formation, lanes, SIMT, divergence introduction, and resource residence examples.
- [ ] Expand memory hierarchy intuition with latency/bandwidth, register allocation, local spilling, shared/global scope, and arithmetic-intensity examples.
- [ ] Expand latency hiding with ready/active/stalled warp concepts, issue examples, occupancy limits, and T4 resource arithmetic.
- [ ] Expand the T4 lab into guided predictions, output interpretation, calculations, and follow-up micro-experiments.
- [ ] Add exercises with answers and interview follow-ups.

### Task 2: Complete And Deepen Volume 2

- [ ] Deepen the complete vector-add lifecycle and indexing chapters.
- [ ] Add CUDA function qualifiers and execution-space chapter.
- [ ] Add allocation, copy, initialization, cleanup, RAII-intuition, and memory-error chapter.
- [ ] Add launch asynchrony, synchronization, streams introduction, and error timing chapter.
- [ ] Add NVCC compilation pipeline, host/device compilation, PTX, cubin, fatbin, and architecture target chapter.
- [ ] Add correct timing and benchmark chapter.
- [ ] Add 2D matrix-add sample chapter.
- [ ] Add naive GEMM derivation and sample chapter.
- [ ] Add Volume 2 review, exercises, answers, and interview drills.

### Task 3: Add Runnable Samples

- [ ] Add function-qualifier sample.
- [ ] Add memory-lifecycle sample.
- [ ] Add asynchronous-error sample with safe modes.
- [ ] Add compilation-inspection sample and Make targets for PTX/SASS metadata.
- [ ] Add event-timing sample.
- [ ] Add 2D matrix-add sample with non-square boundaries.
- [ ] Add naive GEMM sample with CPU reference and non-multiple dimensions.

### Task 4: Verify

- [ ] Build all Volume 1-2 samples.
- [ ] Run normal and boundary cases.
- [ ] Run Compute Sanitizer on memory, matrix-add, and GEMM samples.
- [ ] Validate required chapters, links, code fences, headings, and generated-file cleanup.

