#include <cuda_runtime.h>

#include "kernels.cuh"
#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

// Elementwise and in-place: two loads, two stores, and a sincos per rotated
// pair. Nothing to cache -- the angles are cheaper to recompute per thread than
// to fetch from a precomputed table in global memory, which is why this engine
// has no cos/sin cache at all.
//
// One thread per (token, head, pair), where a pair is (i, i + head_dim/2).
// q and k are rotated by the same kernel launch: the grid covers q's heads
// followed by k's, so the GQA head-count mismatch costs no second launch.
__global__ void rope_kernel(float* __restrict__ q, float* __restrict__ k,
                            const std::int32_t* __restrict__ positions, int n_tokens, int n_heads,
                            int n_kv_heads, int head_dim, float theta) {
  const int pairs = head_dim / 2;
  const int heads_total = n_heads + n_kv_heads;
  const std::size_t total = static_cast<std::size_t>(n_tokens) * heads_total * pairs;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

  for (std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
       idx < total; idx += stride) {
    const int pair = static_cast<int>(idx % pairs);
    const std::size_t rest = idx / pairs;
    const int head = static_cast<int>(rest % heads_total);
    const int token = static_cast<int>(rest / heads_total);

    // The grid covers q's heads followed by k's, so the GQA head-count mismatch
    // costs no second launch -- only this branch, which is uniform across a warp
    // for all but one head boundary.
    float* base = head < n_heads
                      ? q + (static_cast<std::size_t>(token) * n_heads + head) * head_dim
                      : k + (static_cast<std::size_t>(token) * n_kv_heads + (head - n_heads)) *
                                head_dim;

    // The precise sincosf, not __sincosf: absolute positions reach into the
    // thousands, and the fast intrinsic loses accuracy on large arguments --
    // exactly where a long-context decode lives.
    const float inv_freq = 1.0f / powf(theta, (2.0f * pair) / head_dim);
    const float angle = static_cast<float>(positions[token]) * inv_freq;
    float sin_a, cos_a;
    sincosf(angle, &sin_a, &cos_a);

    // Llama's rotate_half pairing: element i rotates with element i + head_dim/2.
    const float lo = base[pair];
    const float hi = base[pair + pairs];
    base[pair] = lo * cos_a - hi * sin_a;
    base[pair + pairs] = hi * cos_a + lo * sin_a;
  }
}

}  // namespace

void launch_rope(float* q, float* k, const std::int32_t* positions, int n_tokens, int n_heads,
                 int n_kv_heads, int head_dim, float theta, void* stream) {
  if (n_tokens == 0 || head_dim == 0) return;
  const int pairs = head_dim / 2;
  const int total = n_tokens * (n_heads + n_kv_heads) * pairs;
  const int grid = clamped_grid(ceil_div(total, kDefaultBlockSize));
  rope_kernel<<<grid, kDefaultBlockSize, 0, static_cast<cudaStream_t>(stream)>>>(
      q, k, positions, n_tokens, n_heads, n_kv_heads, head_dim, theta);
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
