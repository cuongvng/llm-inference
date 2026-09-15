#include <cuda_runtime.h>

#include "kernels.cuh"
#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

// Row-wise softmax, one block per row, in place.
//
// Three passes over the row -- max, sum, normalize -- but only the first of them
// reaches DRAM: the row is a few KB, so the block's own L1 serves the other two.
// The reductions themselves never leave the SM at all.
//
// Materializing the full [heads, query, keys] score matrix at all is the thing
// this design eventually gives up: an online (flash-attention-style) softmax
// fused into the QK and AV kernels keeps the row in shared memory and never
// writes it out. That is the natural follow-on once the VRAM ceiling starts
// binding; this kernel is the correct-and-obvious baseline it gets measured
// against.
__global__ void softmax_rows_kernel(float* __restrict__ x, int row_len) {
  extern __shared__ float smem[];

  float* row = x + static_cast<std::size_t>(blockIdx.x) * row_len;

  // Subtracting the row max before exp is what keeps this from overflowing.
  float thread_max = -FLT_MAX;
  for (int i = threadIdx.x; i < row_len; i += blockDim.x) thread_max = fmaxf(thread_max, row[i]);
  const float row_max = block_reduce_max(thread_max, smem);

  // Masked entries arrive as -INFINITY and exp(-inf - row_max) is 0 for any
  // finite max, so they need no special case. A row that were *entirely*
  // -INFINITY would give 0/0, which causal masking rules out: the diagonal is
  // always live.
  float thread_sum = 0.0f;
  for (int i = threadIdx.x; i < row_len; i += blockDim.x) thread_sum += __expf(row[i] - row_max);
  const float inv_sum = 1.0f / block_reduce_sum(thread_sum, smem);

  for (int i = threadIdx.x; i < row_len; i += blockDim.x) {
    row[i] = __expf(row[i] - row_max) * inv_sum;
  }
}

}  // namespace

void launch_softmax_rows(float* x, int n_rows, int row_len, void* stream) {
  if (n_rows == 0 || row_len == 0) return;
  const int block = kReduceBlockSize;
  const std::size_t shared_bytes = (block / kWarpSize) * sizeof(float);
  softmax_rows_kernel<<<n_rows, block, shared_bytes, static_cast<cudaStream_t>(stream)>>>(x,
                                                                                          row_len);
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
