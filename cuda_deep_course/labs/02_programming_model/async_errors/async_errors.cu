#include <cstring>
#include <iostream>

#include "../../common/cuda_check.cuh"

__global__ void validKernel(int* output) {
  if (threadIdx.x == 0) {
    output[0] = 42;
  }
}

__global__ void illegalAccessKernel(int* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  output[index + 1'000'000] = index;
}

int main(int argc, char** argv) {
  const bool triggerIllegal =
      argc >= 2 && std::strcmp(argv[1], "--illegal-access") == 0;

  int* deviceOutput = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceOutput, sizeof(int)));

  if (!triggerIllegal) {
    validKernel<<<1, 32>>>(deviceOutput);
    CUDA_CHECK(cudaGetLastError());
    std::cout << "Launch accepted. Host reached this line before an explicit "
                 "device synchronization.\n";
    CUDA_CHECK(cudaDeviceSynchronize());

    int value = 0;
    CUDA_CHECK(cudaMemcpy(&value, deviceOutput, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceOutput));
    std::cout << "Synchronized result=" << value << '\n';
    std::cout << (value == 42 ? "PASS" : "FAIL") << '\n';
    return value == 42 ? 0 : 1;
  }

  std::cout << "Triggering an intentional illegal access.\n";
  illegalAccessKernel<<<1, 32>>>(deviceOutput);

  const cudaError_t launchStatus = cudaGetLastError();
  std::cout << "Immediate launch check: "
            << cudaGetErrorString(launchStatus) << '\n';

  const cudaError_t executionStatus = cudaDeviceSynchronize();
  std::cout << "Synchronization check: "
            << cudaGetErrorString(executionStatus) << '\n';

  cudaFree(deviceOutput);
  return executionStatus == cudaSuccess ? 1 : 0;
}

