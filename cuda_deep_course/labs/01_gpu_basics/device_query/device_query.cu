#include <iomanip>
#include <iostream>

#include "../../common/cuda_check.cuh"

namespace {

double bytesToGiB(std::size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

double bytesToKiB(std::size_t bytes) {
  return static_cast<double>(bytes) / 1024.0;
}

}  // namespace

int main() {
  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
    std::cerr << "No CUDA-capable device found.\n";
    return 1;
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp property{};
  CUDA_CHECK(cudaGetDeviceProperties(&property, device));

  int memoryClockRateKHz = 0;
  int memoryBusWidthBits = 0;
  int asyncEngineCount = 0;
  int concurrentKernels = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &memoryClockRateKHz, cudaDevAttrMemoryClockRate, device));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &memoryBusWidthBits, cudaDevAttrGlobalMemoryBusWidth, device));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &asyncEngineCount, cudaDevAttrAsyncEngineCount, device));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &concurrentKernels, cudaDevAttrConcurrentKernels, device));

  const long long peakResidentThreads =
      static_cast<long long>(property.multiProcessorCount) *
      property.maxThreadsPerMultiProcessor;

  // The memory clock attribute is in kHz and represents the physical clock.
  // DDR transfers data twice per clock.
  const double approximateBandwidthGBs =
      2.0 * static_cast<double>(memoryClockRateKHz) * 1000.0 *
      (static_cast<double>(memoryBusWidthBits) / 8.0) / 1.0e9;

  std::cout << "CUDA device observation\n"
            << "=======================\n";
  std::cout << "Device index: " << device << " / " << deviceCount << '\n';
  std::cout << "Name: " << property.name << '\n';
  std::cout << "Compute capability: " << property.major << '.'
            << property.minor << "\n\n";

  std::cout << "Execution resources\n"
            << "-------------------\n";
  std::cout << "SM count: " << property.multiProcessorCount << '\n';
  std::cout << "Warp size: " << property.warpSize << '\n';
  std::cout << "Max threads per block: " << property.maxThreadsPerBlock
            << '\n';
  std::cout << "Max threads per SM: "
            << property.maxThreadsPerMultiProcessor << '\n';
  std::cout << "Max block dimensions: (" << property.maxThreadsDim[0] << ", "
            << property.maxThreadsDim[1] << ", "
            << property.maxThreadsDim[2] << ")\n";
  std::cout << "Max grid dimensions: (" << property.maxGridSize[0] << ", "
            << property.maxGridSize[1] << ", " << property.maxGridSize[2]
            << ")\n";
  std::cout << "Registers per block: " << property.regsPerBlock << '\n';
  std::cout << "Registers per SM: " << property.regsPerMultiprocessor << '\n';
  std::cout << "Specification-limit resident threads across all SMs: "
            << peakResidentThreads << "\n\n";

  std::cout << "Memory resources\n"
            << "----------------\n";
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Global memory: " << bytesToGiB(property.totalGlobalMem)
            << " GiB\n";
  std::cout << "Shared memory per block: "
            << bytesToKiB(property.sharedMemPerBlock) << " KiB\n";
  std::cout << "Shared memory per SM: "
            << bytesToKiB(property.sharedMemPerMultiprocessor) << " KiB\n";
  std::cout << "L2 cache: " << bytesToKiB(property.l2CacheSize) << " KiB\n";
  std::cout << "Memory clock: "
            << static_cast<double>(memoryClockRateKHz) / 1000.0 << " MHz\n";
  std::cout << "Memory bus width: " << memoryBusWidthBits << " bits\n";
  std::cout << "Approximate specification-derived DRAM bandwidth: "
            << approximateBandwidthGBs << " GB/s\n";
  std::cout << "  This is an estimate from clock and bus width, not a measured "
               "kernel result.\n\n";

  std::cout << "Concurrency capabilities\n"
            << "------------------------\n";
  std::cout << "Concurrent kernels: "
            << (concurrentKernels ? "yes" : "no") << '\n';
  std::cout << "Async engine count: " << asyncEngineCount << '\n';
  std::cout << "Unified addressing: "
            << (property.unifiedAddressing ? "yes" : "no") << '\n';
  std::cout << "Managed memory: "
            << (property.managedMemory ? "yes" : "no") << '\n';

  return 0;
}
