#include <cstddef>
#include <cstdio>


/**

1.写一个单 warp（32 线程）归约：
2.用 5 次 __shfl_down_sync（偏移 16/8/4/2/1）。
  代码中的 for 循环从 offset=16 开始，每次除以 2：
  16 -> 8 -> 4 -> 2 -> 1，一共 5 次 shuffle-down。

3.手工推演 8 个 lane 的 shuffle-down，确认每步谁加谁。
  如果只看 lane 0~7，并假设初始值是：
  lane:   0  1  2  3  4  5  6  7
  value:  0  1  2  3  4  5  6  7

  offset = 4:
  lane0 += lane4 -> 0 + 4 = 4
  lane1 += lane5 -> 1 + 5 = 6
  lane2 += lane6 -> 2 + 6 = 8
  lane3 += lane7 -> 3 + 7 = 10

  offset = 2:
  lane0 += lane2 -> 4 + 8 = 12
  lane1 += lane3 -> 6 + 10 = 16

  offset = 1:
  lane0 += lane1 -> 12 + 16 = 28

  所以 8 个 lane 时，lane0 最终得到 0+1+2+3+4+5+6+7 = 28。
  32 个 lane 同理，只是多了 offset=16 和 offset=8 两步。

4.验证 lane 0 拿到 32 个数之和。

*/
__global__ void warp_reduce(int* input, int size) {
    int idx  = blockDim.x * blockIdx.x + threadIdx.x;
    int value = 0;
    if (idx < size) {
        value = input[idx];
    }

    int lane = threadIdx.x & 31;
    for (int offset = 16; offset > 0; offset /=2) {
        value += __shfl_down_sync(0xffffffffU, value, offset);

    }
    if (lane == 0) {
        printf("lane 0 value: %d\n", value);
    }
}

int main() {
    int* a = NULL;
    cudaMallocManaged(&a, 64*sizeof(int));
    for (int i = 0; i < 64; i++)a[i]=i;
    warp_reduce<<<1,64>>>(a, 64);
    cudaDeviceSynchronize();
    return 0;
}
