#include <cuda_runtime.h>

#include "kernels.cuh"
#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

// The prefill shape: M = seq_len rows share the same weight matrix, so unlike
// the GEMV there *is* reuse to buy back. Every weight element loaded from VRAM
// can serve all M rows if it is staged in shared memory first -- that is the
// whole reason this kernel exists separately from gemv_fp16.
//
// Tiling: a block computes a kTileM x kTileN patch of the output, walking k in
// kTileK chunks. Each chunk stages x's tile and weight's tile into shared
// memory, syncs, and accumulates into per-thread registers. Bytes moved from
// global memory drop by roughly the tile dimension compared with the naive
// one-thread-per-output version.
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 16;

__global__ void gemm_fp16_kernel(const __half* __restrict__ weight, const float* __restrict__ x,
                                 float* __restrict__ y, int m_rows, int n_out, int k_in) {
  // y[m, n] = sum_k x[m * k_in + k] * weight[n * k_in + k], accumulated in FP32.
  //
  // Note both operands are indexed by k in their *last* dimension, so a tile of
  // weight is read along rows -- coalesced -- and the transpose that the math
  // implies happens for free in how the shared tile is indexed on read-back.
  //
  // Shape of the loop:
  //   acc = 0
  //   for k0 in steps of kTileK:
  //       stage x tile and weight tile into __shared__
  //       __syncthreads()
  //       acc += dot over the staged tile
  //       __syncthreads()          // before the next chunk overwrites it
  //   write y if this thread's (m, n) is in range
  //
  // The partial tile at the end of k (and the ragged edges in m/n) must be
  // zero-filled in shared memory rather than skipped, so the dot product below
  // needs no per-element bounds test.
  // The +1 pad breaks the bank conflict in the accumulate loop below, where
  // threads of a warp read w_tile[threadIdx.x][t] -- without it, a 16-float row
  // stride puts every other thread on the same bank.
  __shared__ float x_tile[kTileM][kTileK];
  __shared__ float w_tile[kTileN][kTileK + 1];

  const int m = blockIdx.y * kTileM + threadIdx.y;
  const int n = blockIdx.x * kTileN + threadIdx.x;

  float acc = 0.0f;
  for (int k0 = 0; k0 < k_in; k0 += kTileK) {
    const int k = k0 + threadIdx.x;

    // Both operands are indexed by k in their last dimension, so both tiles are
    // read along rows -- coalesced -- and the transpose the math implies happens
    // for free in how w_tile is indexed on read-back.
    //
    // Ragged edges are zero-filled rather than skipped, which is what lets the
    // accumulate loop run without a per-element bounds test.
    x_tile[threadIdx.y][threadIdx.x] =
        (m < m_rows && k < k_in) ? x[static_cast<std::size_t>(m) * k_in + k] : 0.0f;

    const int w_row = blockIdx.x * kTileN + threadIdx.y;
    w_tile[threadIdx.y][threadIdx.x] =
        (w_row < n_out && k < k_in) ? load_half(weight + static_cast<std::size_t>(w_row) * k_in + k)
                                    : 0.0f;
    __syncthreads();

    for (int t = 0; t < kTileK; ++t) {
      acc += x_tile[threadIdx.y][t] * w_tile[threadIdx.x][t];
    }
    // Before the next chunk overwrites the tiles.
    __syncthreads();
  }

  if (m < m_rows && n < n_out) y[static_cast<std::size_t>(m) * n_out + n] = acc;
}

}  // namespace

void launch_gemm_fp16(const __half* weight, const float* x, float* y, int m_rows, int n_out,
                      int k_in, void* stream) {
  if (m_rows == 0 || n_out == 0 || k_in == 0) return;
  const dim3 block(kTileN, kTileM);
  const dim3 grid(ceil_div(n_out, kTileN), ceil_div(m_rows, kTileM));
  gemm_fp16_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(weight, x, y, m_rows,
                                                                         n_out, k_in);
  LLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace llm
