#include <cuda_runtime.h>

#include "kernels.cuh"
#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

// Pure gather: every output element is read exactly once from a row of the
// table and written exactly once, so there is no reuse to exploit and nothing
// belongs in shared memory. The only thing that matters is that the reads of a
// row are coalesced across the threads that cooperate on it -- one block per
// token, threads striding along `hidden`, gives fully coalesced 2-byte reads
// that the memory system can widen.
__global__ void embedding_lookup_kernel(const __half* __restrict__ table,
                                        const std::int32_t* __restrict__ ids,
                                        float* __restrict__ out, int n_tokens, int hidden) {
  const int token = blockIdx.x;
  if (token >= n_tokens) return;

  const __half* src = table + static_cast<std::size_t>(ids[token]) * hidden;
  float* dst = out + static_cast<std::size_t>(token) * hidden;

  // The block strides `hidden` rather than covering it: hidden (2048 for
  // TinyLlama) is larger than any block, and striding keeps consecutive threads
  // on consecutive columns at every step, which is what coalesces the reads.
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    dst[col] = load_half(src + col);
  }
}

}  // namespace

void launch_embedding_lookup(const __half* table, const std::int32_t* ids, float* out,
                             int n_tokens, int hidden, void* stream) {
  if (n_tokens == 0 || hidden == 0) return;
  embedding_lookup_kernel<<<n_tokens, kDefaultBlockSize, 0, static_cast<cudaStream_t>(stream)>>>(
      table, ids, out, n_tokens, hidden);
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
