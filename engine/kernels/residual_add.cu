#include <cuda_runtime.h>

#include "kernels.cuh"
#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

// The cheapest kernel in the engine and, per byte moved, one of the least
// efficient uses of a launch: two loads and a store for one flop. It exists as
// its own kernel only because the residual stream has to be kept intact across
// the norm; folding the add into the following rmsnorm's read is the fusion
// that removes it, and is worth doing once the layer graph settles.
__global__ void residual_add_kernel(float* __restrict__ x, const float* __restrict__ y,
                                    std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
       i += stride) {
    x[i] += y[i];
  }
}

}  // namespace

void launch_residual_add(float* x, const float* y, std::size_t n, void* stream) {
  if (n == 0) return;
  const int grid = clamped_grid(static_cast<int>((n + kDefaultBlockSize - 1) / kDefaultBlockSize));
  residual_add_kernel<<<grid, kDefaultBlockSize, 0, static_cast<cudaStream_t>(stream)>>>(x, y, n);
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
