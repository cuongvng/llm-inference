#pragma once

#include <cstddef>

#include "llm/config.h"
#include "llm/device_buffer.h"

namespace llm {

class KVCache {
 public:
  void allocate(const ModelConfig& config, int max_tokens);

  float* layer_k(int layer);
  float* layer_v(int layer);
  const float* layer_k(int layer) const;
  const float* layer_v(int layer) const;

  int length() const { return length_; }

  void advance(int n_new);

  void reset() { length_ = 0; }

  int max_tokens() const { return max_tokens_; }
  int kv_dim() const { return kv_dim_; }
  std::size_t bytes() const { return k_.bytes() + v_.bytes(); }

 private:
  std::size_t layer_stride() const;

  int max_tokens_ = 0;
  int num_layers_ = 0;
  int kv_dim_ = 0;
  int length_ = 0;

  DeviceBuffer k_;  // float [num_layers, max_tokens, kv_dim]
  DeviceBuffer v_;  // float [num_layers, max_tokens, kv_dim]
};

}  // namespace llm
