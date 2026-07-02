#include <cstdio>
#include <vector>

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}


int main() {
  int n = 1 << 20;
  std::vector<float> h_a(n), h_b(n), h_c(n);


  for (int i = 0; i < n; i++) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(i * 2);
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  size_t bytes = n * sizeof(float);
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  int thread= 256;
  int blocks = (n + thread-1)/thread;
  vec_add<<<blocks, thread>>>(d_a, d_b, d_c, n);

  cudaDeviceSynchronize();

  
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  bool ok = true;
  for (int i = 0; i < n; i++) {
    if (h_c[i] != h_a[i] + h_b[i]) {
      ok = false;
      break;
    }
  }

  printf("n = %d, result=%s\n", n, ok ? "PASS": "FAIL");
  cudaFree(d_c);
  cudaFree(d_b);
  cudaFree(d_a);

  return ok? 0: 1;

}
