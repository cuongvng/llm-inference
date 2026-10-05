#include "llm/kv_cache.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace llm {
namespace {

[[noreturn]] void fail(const std::string& message) {
  std::fprintf(stderr, "[llm] kv_cache: %s\n", message.c_str());
  std::exit(EXIT_FAILURE);
}

}  // namespace

std::size_t KVCache::layer_stride() const {
  return static_cast<std::size_t>(max_tokens_) * kv_dim_;
}

void KVCache::allocate(const ModelConfig& config, int max_tokens) {
  if (max_tokens <= 0) fail("max_tokens must be positive");

  max_tokens_ = max_tokens;
  num_layers_ = static_cast<int>(config.num_layers);
  kv_dim_ = static_cast<int>(config.num_kv_heads * config.head_dim);
  length_ = 0;

  const std::size_t elems = static_cast<std::size_t>(num_layers_) * layer_stride();
  k_.allocate(elems * sizeof(float));
  v_.allocate(elems * sizeof(float));
}

float* KVCache::layer_k(int layer) {
  return const_cast<float*>(static_cast<const KVCache*>(this)->layer_k(layer));
}

float* KVCache::layer_v(int layer) {
  return const_cast<float*>(static_cast<const KVCache*>(this)->layer_v(layer));
}

const float* KVCache::layer_k(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    fail("layer " + std::to_string(layer) + " out of range, cache holds " +
         std::to_string(num_layers_));
  }
  return k_.as<float>() + static_cast<std::size_t>(layer) * layer_stride();
}

const float* KVCache::layer_v(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    fail("layer " + std::to_string(layer) + " out of range, cache holds " +
         std::to_string(num_layers_));
  }
  return v_.as<float>() + static_cast<std::size_t>(layer) * layer_stride();
}

void KVCache::advance(int n_new) {
  if (n_new < 0) fail("advance called with a negative count");
  if (length_ + n_new > max_tokens_) {
    // The context limit, reported where it is actually hit.
    fail("sequence of " + std::to_string(length_ + n_new) + " exceeds cache capacity " +
         std::to_string(max_tokens_));
  }
  length_ += n_new;
}

}  // namespace llm
