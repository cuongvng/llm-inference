// M0 done-signal #1: GPU result byte-for-byte equal to the CPU reference.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "llm/cuda_check.h"
#include "llm/kernels.h"

int main() {
  // Deliberately not a multiple of the block size, so the grid-stride tail and
  // the bounds check both get exercised.
  const std::size_t n = 1'000'003;
  const std::size_t bytes = n * sizeof(float);

  std::vector<float> h_a(n), h_b(n), h_c(n), h_ref(n);
  for (std::size_t i = 0; i < n; ++i) {
    // Exactly representable in fp32, so GPU and CPU must agree bitwise.
    h_a[i] = static_cast<float>(i % 1024);
    h_b[i] = static_cast<float>((i % 77) * 2);
    h_ref[i] = h_a[i] + h_b[i];
  }

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  LLM_CUDA_CHECK(cudaMalloc(&d_a, bytes));
  LLM_CUDA_CHECK(cudaMalloc(&d_b, bytes));
  LLM_CUDA_CHECK(cudaMalloc(&d_c, bytes));

  LLM_CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  LLM_CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
  LLM_CUDA_CHECK(cudaMemset(d_c, 0xff, bytes));

  llm::launch_vector_add(d_a, d_b, d_c, n);
  LLM_CUDA_CHECK_LAUNCH();

  LLM_CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (h_c[i] != h_ref[i]) {
      if (mismatches < 5) {
        std::fprintf(stderr, "mismatch at %zu: gpu=%f cpu=%f\n", i, h_c[i], h_ref[i]);
      }
      ++mismatches;
    }
  }

  LLM_CUDA_CHECK(cudaFree(d_a));
  LLM_CUDA_CHECK(cudaFree(d_b));
  LLM_CUDA_CHECK(cudaFree(d_c));

  if (mismatches != 0) {
    std::fprintf(stderr, "FAIL: %zu / %zu elements mismatched\n", mismatches, n);
    return EXIT_FAILURE;
  }
  std::printf("PASS: vector_add matches CPU reference for %zu elements\n", n);
  return EXIT_SUCCESS;
}
