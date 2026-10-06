#include <cuda_runtime.h>

#include "kernels.cuh"
#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

__global__ void kv_cache_write_kernel(const float* __restrict__ new_k,
                                      const float* __restrict__ new_v, float* __restrict__ k_cache,
                                      float* __restrict__ v_cache, std::size_t n,
                                      std::size_t cache_offset) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
       i += stride) {
    k_cache[cache_offset + i] = new_k[i];
    v_cache[cache_offset + i] = new_v[i];
  }
}

}  // namespace

void launch_kv_cache_write(const float* new_k, const float* new_v, float* k_cache, float* v_cache,
                           int n_new, int start_pos, int kv_dim, void* stream) {
  if (n_new == 0) return;
  const std::size_t n = static_cast<std::size_t>(n_new) * kv_dim;
  const std::size_t cache_offset = static_cast<std::size_t>(start_pos) * kv_dim;
  const int grid = clamped_grid(static_cast<int>((n + kDefaultBlockSize - 1) / kDefaultBlockSize));
  kv_cache_write_kernel<<<grid, kDefaultBlockSize, 0, static_cast<cudaStream_t>(stream)>>>(
      new_k, new_v, k_cache, v_cache, n, cache_offset);
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
