#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "llm/config.h"

namespace llm {

constexpr int kMaxTensorDims = 4;

// Non-owning view over a tensor's bytes: either host memory owned by a
// ModelFile or a device pointer owned by the allocator.
// TensorView never allocates or frees -- ownership always lives elsewhere.
struct TensorView {
  const std::uint8_t* data = nullptr;
  DType dtype = DType::kFP32;
  int ndim = 0;
  std::array<std::uint32_t, kMaxTensorDims> shape{};
  std::size_t nbytes = 0;

  std::uint32_t dim(int i) const { return shape[i]; }

  // Row-major offset (in elements, not bytes) for a leading-dimension index.
  // Only meaningful for unpacked dtypes (FP32/FP16); INT4 callers must handle
  // the 2-values-per-byte packing themselves.
  std::size_t row_stride_elems() const {
    std::size_t stride = 1;
    for (int i = 1; i < ndim; ++i) stride *= shape[i];
    return stride;
  }

  const std::uint8_t* row(std::size_t row_index) const {
    return data + row_index * row_stride_elems() * dtype_size_bytes(dtype);
  }
};

}  // namespace llm
