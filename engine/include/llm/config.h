#pragma once

#include <cstdint>

namespace llm {

enum class DType : std::uint32_t {
  kFP32 = 0,
  kFP16 = 1,
  kINT8 = 2,
  kINT4 = 3,
};

inline std::size_t dtype_size_bytes(DType dtype) {
  switch (dtype) {
    case DType::kFP32: return 4;
    case DType::kFP16: return 2;
    case DType::kINT8: return 1;
    case DType::kINT4: return 1;  // packed 2/byte; callers must account for this
  }
  return 0;
}

struct ModelConfig {
  std::uint32_t vocab_size = 0;
  std::uint32_t hidden_size = 0;
  std::uint32_t num_layers = 0;
  std::uint32_t num_heads = 0;
  std::uint32_t num_kv_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t intermediate_size = 0;
  std::uint32_t max_seq_len = 0;
  float rope_theta = 0.0f;
  float rms_norm_eps = 0.0f;
};

}  // namespace llm
