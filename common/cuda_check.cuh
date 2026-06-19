#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

inline void checkCuda(cudaError_t result, const char* expression,
                      const char* file, int line) {
  if (result != cudaSuccess) {
    std::fprintf(stderr,
                 "CUDA error at %s:%d\n"
                 "  expression: %s\n"
                 "  reason: %s\n",
                 file, line, expression, cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(expression) \
  checkCuda((expression), #expression, __FILE__, __LINE__)
