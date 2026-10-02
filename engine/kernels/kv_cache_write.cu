#include <cuda_runtime.h>

#include "kernels.cuh"
#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

// Appends this step's K and V rows into one layer's cache slab.
//
// Pure copy, no arithmetic: what it costs is exactly what it moves, which makes
// it the one kernel whose measured bandwidth is a clean read of the memory
// system rather than of a blocking decision.
//
// K and V ride in one launch. They are the same shape and land at the same row
// offset, so splitting them would pay launch overhead twice to move the same
// bytes, and the two streams interleave in flight rather than competing.
//
// The copy is contiguous today, which makes it equivalent to a device-to-device
// memcpy. It is a kernel because the row offset becomes a per-page lookup once
// the cache is drawn from a paged allocator, and that scatter has nowhere else
// to live.
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
