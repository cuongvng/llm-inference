#pragma once

#include <cstddef>

namespace llm {

// Toolchain smoke test: c[i] = a[i] + b[i] over device pointers.
// Grid-stride loop so the launch config is independent of `n`.
void launch_vector_add(const float* d_a, const float* d_b, float* d_c, std::size_t n,
                       void* stream = nullptr);

}  // namespace llm
