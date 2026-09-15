#include <cuda_runtime.h>

#include "kernels.cuh"
#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

// Two loads, one store, a handful of flops: purely bandwidth-bound, and the
// reason gate and up are computed into separate buffers and combined here
// rather than each being written back through a norm. Fusing this into the
// down-projection's input read is the obvious next saving -- it would remove an
// entire intermediate_size-wide round trip to VRAM per layer -- but that
// changes the GEMM's interface, so it waits until the profile says it is worth
// the coupling.
__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                              float* __restrict__ out, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
       i += stride) {
    const float g = gate[i];
    out[i] = (g / (1.0f + __expf(-g))) * up[i];
  }
}

}  // namespace

void launch_swiglu(const float* gate, const float* up, float* out, std::size_t n, void* stream) {
  if (n == 0) return;
  const int grid = clamped_grid(static_cast<int>((n + kDefaultBlockSize - 1) / kDefaultBlockSize));
  swiglu_kernel<<<grid, kDefaultBlockSize, 0, static_cast<cudaStream_t>(stream)>>>(gate, up, out, n);
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
