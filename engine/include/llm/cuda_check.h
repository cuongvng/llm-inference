#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace llm {

inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[CUDA] %s:%d: %s failed: %s (%s)\n", file, line, expr,
                 cudaGetErrorName(err), cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace llm

// Wrap every runtime call. Fails loudly at the call site instead of surfacing
// as a confusing error from an unrelated later call.
#define LLM_CUDA_CHECK(expr) ::llm::cuda_check((expr), #expr, __FILE__, __LINE__)

// Launches report errors asynchronously; this catches both the launch-config
// error (cudaGetLastError) and any fault during execution (the sync).
#define LLM_CUDA_CHECK_LAUNCH()                  \
  do {                                           \
    LLM_CUDA_CHECK(cudaGetLastError());          \
    LLM_CUDA_CHECK(cudaDeviceSynchronize());     \
  } while (0)
