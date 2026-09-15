#include <cuda_runtime.h>

#include "kernels.cuh"
#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

// The decode-time shape: M = 1, so every weight byte is loaded, used for a
// single multiply-add, and thrown away. There is no reuse of the weight matrix
// to exploit and the kernel is hard-bound by how fast the weights can be pulled
// out of VRAM -- the arithmetic is free by comparison.
//
// That fixes the design: one warp per output row, lanes striding along k so the
// row's FP16 reads coalesce into full cache lines, then a warp-shuffle
// reduction. The input vector x is small (hidden-sized) and read by every warp,
// so it lives in L2/L1 without any explicit staging.
constexpr int kWarpsPerBlock = kReduceBlockSize / kWarpSize;

__global__ void gemv_fp16_kernel(const __half* __restrict__ weight, const float* __restrict__ x,
                                 float* __restrict__ y, int n_out, int k_in) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int row = blockIdx.x * kWarpsPerBlock + warp;

  // Warp-uniform, so the whole warp leaves together and the full-mask shuffles
  // below never see a partially retired warp.
  if (row >= n_out) return;

  const __half* w_row = weight + static_cast<std::size_t>(row) * k_in;

  // Lanes walk k together: 32 consecutive FP16 values per step is a 64-byte
  // request, which the memory system fills from one or two sectors.
  float acc = 0.0f;
  for (int i = lane; i < k_in; i += kWarpSize) {
    acc += load_half(w_row + i) * x[i];
  }

  acc = warp_reduce_sum(acc);
  if (lane == 0) y[row] = acc;
}

}  // namespace

void launch_gemv_fp16(const __half* weight, const float* x, float* y, int n_out, int k_in,
                      void* stream) {
  if (n_out == 0 || k_in == 0) return;
  const int grid = ceil_div(n_out, kWarpsPerBlock);
  gemv_fp16_kernel<<<grid, kReduceBlockSize, 0, static_cast<cudaStream_t>(stream)>>>(weight, x, y,
                                                                                    n_out, k_in);
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
