#include <cuda_runtime.h>

#include <cfloat>

#include "kernels.cuh"
#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

// Greedy sampling's last step: one pass over the vocabulary-sized logits vector
// to find the winning index. The LM head projection itself is an ordinary GEMV
// (gemv_fp16) -- this file is only the reduction that follows it.
//
// A single block: the vector is one row of ~32K floats, a fraction of a
// millisecond of streaming, and one block avoids a second kernel launch to
// combine per-block winners. Copying the whole logits vector back to the host
// to argmax it there would cost more in PCIe transfer than the reduction saves.
constexpr int kArgmaxBlockSize = 256;

__global__ void argmax_kernel(const float* __restrict__ logits, std::int32_t* __restrict__ out,
                              int n) {
  // The reduction carries the index alongside the value, which is why it is
  // written out by hand instead of calling block_reduce_max().
  __shared__ float best_value[kArgmaxBlockSize];
  __shared__ std::int32_t best_index[kArgmaxBlockSize];

  float value = -FLT_MAX;
  // `n` as the empty marker, not 0: it loses every tie against a real index, so
  // a thread that gets no elements can never win one.
  std::int32_t index = n;

  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const float v = logits[i];
    if (v > value) {
      value = v;
      index = i;
    }
  }

  best_value[threadIdx.x] = value;
  best_index[threadIdx.x] = index;
  __syncthreads();

  // Tie-breaking is not incidental: torch.argmax returns the lowest index and
  // the greedy-decode test compares token ids exactly, so equal values keep the
  // smaller index at every merge step.
  for (int half = blockDim.x / 2; half > 0; half >>= 1) {
    if (threadIdx.x < half) {
      const float other = best_value[threadIdx.x + half];
      const std::int32_t other_index = best_index[threadIdx.x + half];
      if (other > best_value[threadIdx.x] ||
          (other == best_value[threadIdx.x] && other_index < best_index[threadIdx.x])) {
        best_value[threadIdx.x] = other;
        best_index[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) *out = best_index[0] < n ? best_index[0] : 0;
}

}  // namespace

void launch_argmax(const float* logits, std::int32_t* out_index, int n, void* stream) {
  if (n == 0) return;
  argmax_kernel<<<1, kArgmaxBlockSize, 0, static_cast<cudaStream_t>(stream)>>>(logits, out_index, n);
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
