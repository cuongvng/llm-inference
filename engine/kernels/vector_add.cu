#include <cuda_runtime.h>

#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

constexpr int kBlockSize = 256;

// Purely bandwidth-bound: 2 loads + 1 store per element, zero reuse, so nothing
// here belongs in shared memory. Grid-stride keeps the block count bounded by
// occupancy rather than by n.
__global__ void vector_add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                  float* __restrict__ c, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x; 
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
       i += stride) {
    c[i] = a[i] + b[i];
  }
}

}  // namespace

void launch_vector_add(const float* d_a, const float* d_b, float* d_c, std::size_t n,
                       void* stream) {
  if (n == 0) return;
  const std::size_t want = (n + kBlockSize - 1) / kBlockSize;
  const int grid = static_cast<int>(want > 65535 ? 65535 : want);
  vector_add_kernel<<<grid, kBlockSize, 0, static_cast<cudaStream_t>(stream)>>>(d_a, d_b, d_c, n);
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
