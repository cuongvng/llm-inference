#include <cuda_runtime.h>

#include "kernels.cuh"
#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

// One block per row, so the sum-of-squares reduction never leaves the SM: the
// row is read once into registers, reduced through shuffles + a warp-count-sized
// shared array, then scaled and written. Reading the row twice (once to reduce,
// once to scale) would double the traffic on a kernel that is entirely
// bandwidth-bound, so the values are kept live instead where the row fits, or
// re-read from L1/L2 where it doesn't.
__global__ void rmsnorm_kernel(const float* __restrict__ x, const __half* __restrict__ weight,
                               float* __restrict__ out, int hidden, float eps) {
  extern __shared__ float smem[];

  const int row = blockIdx.x;
  const float* x_row = x + static_cast<std::size_t>(row) * hidden;
  float* out_row = out + static_cast<std::size_t>(row) * hidden;

  // The row is longer than the block, so each thread accumulates a strided
  // partial first and only warp-count floats reach shared memory.
  float sum_squares = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = x_row[i];
    sum_squares += v * v;
  }
  // eps applies to the mean square, not the sum -- getting that wrong is a
  // scale error that only shows up on short rows.
  const float mean_square = block_reduce_sum(sum_squares, smem) / hidden;
  const float scale = rsqrtf(mean_square + eps);

  // The second pass re-reads the row, but it comes back out of L1 rather than
  // DRAM: this block just walked it, and the whole row is a few KB.
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    out_row[i] = x_row[i] * scale * load_half(weight + i);
  }
}

}  // namespace

void launch_rmsnorm(const float* x, const __half* weight, float* out, int n_rows, int hidden,
                    float eps, void* stream) {
  if (n_rows == 0 || hidden == 0) return;
  const int block = kReduceBlockSize;
  const std::size_t shared_bytes = (block / kWarpSize) * sizeof(float);
  rmsnorm_kernel<<<n_rows, block, shared_bytes, static_cast<cudaStream_t>(stream)>>>(x, weight, out,
                                                                                    hidden, eps);
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
