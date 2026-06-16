#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include <cuda_runtime.h>

__global__ void vec_add(const float* a, const float* b, float* c, int vectorLength) {
  int workerIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (workerIndex < vectorLength) {
    c[workerIndex] = a[workerIndex] + b[workerIndex];
  }
}

void initArray(float* array, int length) {
  std::srand(static_cast<unsigned>(std::time(nullptr)));
  for (int i = 0; i < length; i++) {
    array[i] = static_cast<float>(std::rand()) / RAND_MAX;
  }
}

void serialAdd(const float* a, const float* b, float* c, int length) {
  for (int i = 0; i < length; i++) {
    c[i] = a[i] + b[i];
  }
}

bool vectorApproximateEqual(const float* a, const float* b, int length, float epsilon) {
  for (int i = 0; i < length; i++) {
    if (std::fabs(a[i] - b[i]) > epsilon) {
      printf("Error at index %d: %f != %f\n", i, a[i], b[i]);
      return false;
    }
  }
  return true;
}

void explicitMemoryManagement(int vectorLength) {
  float* a = nullptr;
  float* b = nullptr;
  float* c = nullptr;
  float* comparisonResult =
      static_cast<float*>(std::malloc(vectorLength * sizeof(float)));

  float* deviceA = nullptr;
  float* deviceB = nullptr;
  float* deviceC = nullptr;

  cudaMallocHost(&a, vectorLength * sizeof(float));
  cudaMallocHost(&b, vectorLength * sizeof(float));
  cudaMallocHost(&c, vectorLength * sizeof(float));

  initArray(a, vectorLength);
  initArray(b, vectorLength);

  cudaMalloc(&deviceA, vectorLength * sizeof(float));
  cudaMalloc(&deviceB, vectorLength * sizeof(float));
  cudaMalloc(&deviceC, vectorLength * sizeof(float));

  cudaMemcpy(deviceA, a, vectorLength * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, b, vectorLength * sizeof(float), cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (vectorLength + threads - 1) / threads;

  vec_add<<<blocks, threads>>>(deviceA, deviceB, deviceC, vectorLength);
  cudaDeviceSynchronize();

  cudaMemcpy(c, deviceC, vectorLength * sizeof(float), cudaMemcpyDeviceToHost);

  serialAdd(a, b, comparisonResult, vectorLength);

  if (vectorApproximateEqual(c, comparisonResult, vectorLength, 1e-6f)) {
    printf("PASS: GPU and CPU results match (n=%d)\n", vectorLength);
  } else {
    printf("FAIL: GPU and CPU results differ (n=%d)\n", vectorLength);
  }

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);
  std::free(comparisonResult);
}

int main(int argc, char** argv) {
  int vectorLength = 1 << 20;
  if (argc >= 2) {
    vectorLength = std::atoi(argv[1]);
  }
  explicitMemoryManagement(vectorLength);
  return 0;
}
