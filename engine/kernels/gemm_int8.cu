#include <cuda_runtime.h>

#include <cstdint>

#include "kernels.cuh"
#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

// Identical tiling to gemm_fp16; only the weight-tile load changes: it turns
// q into q * scale on the way into shared memory, so the inner loop is untouched.
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 16;

__global__ void gemm_int8_kernel(const std::int8_t* __restrict__ weight,
                                 const __half* __restrict__ scales, const float* __restrict__ x,
                                 float* __restrict__ y, int m_rows, int n_out, int k_in,
                                 int group_size) {
  __shared__ float x_tile[kTileM][kTileK];
  __shared__ float w_tile[kTileN][kTileK + 1];

  const int groups = k_in / group_size;
  const int m = blockIdx.y * kTileM + threadIdx.y;
  const int n = blockIdx.x * kTileN + threadIdx.x;

  float acc = 0.0f;
  for (int k0 = 0; k0 < k_in; k0 += kTileK) {
    const int k = k0 + threadIdx.x;

    x_tile[threadIdx.y][threadIdx.x] =
        (m < m_rows && k < k_in) ? x[static_cast<std::size_t>(m) * k_in + k] : 0.0f;

    const int w_row = blockIdx.x * kTileN + threadIdx.y;
    float w = 0.0f;
    if (w_row < n_out && k < k_in) {
      const float q = static_cast<float>(weight[static_cast<std::size_t>(w_row) * k_in + k]);
      w = q * load_half(scales + static_cast<std::size_t>(w_row) * groups + k / group_size);
    }
    w_tile[threadIdx.y][threadIdx.x] = w;
    __syncthreads();

    for (int t = 0; t < kTileK; ++t) {
      acc += x_tile[threadIdx.y][t] * w_tile[threadIdx.x][t];
    }
    __syncthreads();
  }

  if (m < m_rows && n < n_out) y[static_cast<std::size_t>(m) * n_out + n] = acc;
}

}  // namespace

void launch_gemm_int8(const std::int8_t* weight, const __half* scales, const float* x, float* y,
                      int m_rows, int n_out, int k_in, int group_size, void* stream) {
  if (m_rows == 0 || n_out == 0 || k_in == 0) return;
  const dim3 block(kTileN, kTileM);
  const dim3 grid(ceil_div(n_out, kTileN), ceil_div(m_rows, kTileM));
  gemm_int8_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      weight, scales, x, y, m_rows, n_out, k_in, group_size);
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
