// 卷四/01 §5.2 fence + flag producer-consumer 协议
//
//   nvcc -O3 -arch=sm_75 -o fence_flag fence_flag.cu
//   ./fence_flag              # 有 fence
//   ./fence_flag --no-fence   # 删 fence（语义有错，可能难复现）
//   ./fence_flag --broken     # 源码里先 flag 后 data（稳定 stale read）

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

constexpr int kProducerLane = 0;
constexpr int kConsumerLane = 32;  // 不同 warp → Volta+ 独立调度，fence 才有意义

// mode: 0=with fence, 1=no fence, 2=broken (flag before data)
__global__ void fence_flag_kernel(int* data, int* flag, int* wrong_count,
                                  int expected, int mode) {
  if (threadIdx.x == kProducerLane) {
    if (mode == 2) {
      // 先发布 flag，再故意延迟写 data，稳定让 consumer 读到旧值
      atomicExch(flag, 1);
      for (int d = 0; d < 512; ++d) {
        __nanosleep(1000);
      }
      *data = expected;
      return;
    }

    *data = expected;

    if (mode == 0) {
      __threadfence();  // device 级：data 先于 flag 对其他 thread 可见
    }

    atomicExch(flag, 1);
  }

  if (threadIdx.x == kConsumerLane) {
    while (atomicAdd(flag, 0) == 0) {
    }

    if (mode == 0) {
      __threadfence();
    }

    int v = *data;
    if (v != expected) {
      atomicAdd(wrong_count, 1);
    }
  }
}

__global__ void reset_kernel(int* data, int* flag, int* wrong_count) {
  if (threadIdx.x == 0) {
    *data = 0;
    *flag = 0;
    *wrong_count = 0;
  }
}

static void run_trials(const char* label, int mode, int trials, int expected) {
  int *d_data = nullptr, *d_flag = nullptr, *d_wrong = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_wrong, sizeof(int)));

  int wrong_total = 0;
  for (int i = 0; i < trials; ++i) {
    reset_kernel<<<1, 1>>>(d_data, d_flag, d_wrong);
    fence_flag_kernel<<<1, 64>>>(d_data, d_flag, d_wrong, expected, mode);
    CUDA_CHECK(cudaDeviceSynchronize());

    int wrong = 0;
    CUDA_CHECK(cudaMemcpy(&wrong, d_wrong, sizeof(int), cudaMemcpyDeviceToHost));
    wrong_total += wrong;
  }

  printf("[%s] trials=%d, stale reads=%d (expected=%d)\n", label, trials,
         wrong_total, expected);

  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_flag));
  CUDA_CHECK(cudaFree(d_wrong));
}

static void print_why_stale() {
  printf("\n--- 为什么可能读到旧值？ ---\n");
  printf("• fence ≠ barrier：只约束【本 thread 的写顺序/可见范围】，不让别人等。\n");
  printf("• flag 是通知：consumer 见 flag==1 就去读 data。\n");
  printf("• 删 producer 的 fence 后，「写 data」与 atomicExch(flag) 可能被重排，\n");
  printf("  consumer 先看到 flag==1 时 data 仍是 0 → 写后读 race。\n");
  printf("• 独立 warp 调度下跨 warp 更明显；同 warp 锁步时有时「碰巧」对。\n");
  printf("• --broken 在源码里先发布 flag，等价于最坏重排，便于观察。\n");
}

int main(int argc, char** argv) {
  const int kTrials = 100000;
  const int kExpected = 42;
  int mode = 0;

  if (argc > 1) {
    if (strcmp(argv[1], "--no-fence") == 0) {
      mode = 1;
    } else if (strcmp(argv[1], "--broken") == 0) {
      mode = 2;
    }
  }

  printf("§5.2 fence + flag (1 block, warp0=producer, warp1=consumer)\n\n");

  if (mode == 0) {
    printf("=== WITH fence ===\n");
    printf("*data=42; __threadfence(); atomicExch(flag,1)\n\n");
    run_trials("with-fence", 0, kTrials, kExpected);
    printf("\n  ./fence_flag --no-fence\n  ./fence_flag --broken\n");
  } else if (mode == 1) {
    printf("=== WITHOUT fence ===\n");
    printf("*data=42; atomicExch(flag,1)   // 无 __threadfence\n\n");
    run_trials("no-fence", 1, kTrials, kExpected);
    printf("\n若 stale=0：内存模型仍禁止这种写法，只是本次调度未暴露。\n");
    print_why_stale();
  } else {
    printf("=== BROKEN: flag before data ===\n\n");
    run_trials("broken-order", 2, kTrials, kExpected);
    print_why_stale();
  }

  return 0;
}
