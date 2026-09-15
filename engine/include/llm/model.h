#pragma once

#include <cuda_fp16.h>

#include <cstdint>
#include <string>
#include <vector>

#include "llm/config.h"
#include "llm/device_buffer.h"
#include "llm/loader.h"

namespace llm {

// Device pointers for one transformer block. All weights are FP16 and point
// into the single weight buffer Model owns -- none of these are separate
// allocations.
struct LayerWeights {
  const __half* attn_norm = nullptr;   // [hidden]
  const __half* wq = nullptr;          // [num_heads * head_dim, hidden]
  const __half* wk = nullptr;          // [num_kv_heads * head_dim, hidden]
  const __half* wv = nullptr;          // [num_kv_heads * head_dim, hidden]
  const __half* wo = nullptr;          // [hidden, num_heads * head_dim]
  const __half* ffn_norm = nullptr;    // [hidden]
  const __half* gate_proj = nullptr;   // [intermediate, hidden]
  const __half* up_proj = nullptr;     // [intermediate, hidden]
  const __half* down_proj = nullptr;   // [hidden, intermediate]
};

// Every intermediate buffer a forward pass needs, sized for a fixed maximum
// token count and reused across passes.
//
// Allocated once and never freed mid-run on purpose: the point of pre-sizing is
// that a forward pass performs zero allocations, which is also what makes the
// arena allocator a drop-in replacement later -- reset a bump pointer instead of
// keeping these handles.
struct Activations {
  int max_tokens = 0;

  DeviceBuffer ids;        // int32  [max_tokens]
  DeviceBuffer positions;  // int32  [max_tokens]

  DeviceBuffer x;          // float  [max_tokens, hidden]        residual stream
  DeviceBuffer normed;     // float  [max_tokens, hidden]        norm output
  DeviceBuffer q;          // float  [max_tokens, num_heads * head_dim]
  DeviceBuffer k;          // float  [max_tokens, num_kv_heads * head_dim]
  DeviceBuffer v;          // float  [max_tokens, num_kv_heads * head_dim]
  DeviceBuffer scores;     // float  [num_heads, max_tokens, max_tokens]
  DeviceBuffer attn_out;   // float  [max_tokens, num_heads * head_dim]
  DeviceBuffer proj;       // float  [max_tokens, hidden]        wo / down_proj output
  DeviceBuffer gate;       // float  [max_tokens, intermediate]
  DeviceBuffer up;         // float  [max_tokens, intermediate]
  DeviceBuffer act;        // float  [max_tokens, intermediate]  swiglu output
  DeviceBuffer logits;     // float  [max_tokens, vocab]
  DeviceBuffer next_token; // int32  [1]

  void allocate(const ModelConfig& config, int max_tokens);
  std::size_t total_bytes() const;
};

// A loaded model: weights resident in VRAM, plus the workspace and the forward
// pass that runs them.
//
// Single-sequence and cacheless: each forward pass recomputes the whole
// sequence. That is the deliberate baseline the KV cache gets measured against
// -- and the reason `forward` takes the full id array rather than one token.
class Model {
 public:
  // Loads a .llmbin file and uploads every weight to VRAM. Exits on any
  // missing or wrongly shaped tensor, matching the loader's fail-fast policy.
  static Model load(const std::string& path, int max_tokens);

  const ModelConfig& config() const { return config_; }
  std::size_t weight_bytes() const { return weights_.bytes(); }
  std::size_t workspace_bytes() const { return act_.total_bytes(); }
  int max_tokens() const { return act_.max_tokens; }

  // Runs `ids` (length n_tokens) through the network. Leaves logits for every
  // position in the workspace: [n_tokens, vocab_size], row-major, device-side.
  // Does not synchronize.
  void forward(const std::int32_t* host_ids, int n_tokens);

  // Copies the logits of the last forward pass back to the host.
  std::vector<float> logits_host(int n_tokens);

  // Argmax over the final position's logits, on the device. Synchronizes.
  std::int32_t argmax_last(int n_tokens);

  // Greedy decode: repeatedly forward the whole sequence and append the argmax.
  // Returns only the newly generated ids.
  std::vector<std::int32_t> generate(const std::vector<std::int32_t>& prompt, int max_new_tokens);

 private:
  ModelConfig config_;
  DeviceBuffer weights_;  // one allocation holding every tensor's bytes
  Activations act_;

  const __half* tok_embeddings_ = nullptr;  // [vocab, hidden]
  const __half* output_norm_ = nullptr;     // [hidden]
  const __half* lm_head_ = nullptr;         // [vocab, hidden]
  std::vector<LayerWeights> layers_;

  int q_dim() const { return static_cast<int>(config_.num_heads * config_.head_dim); }
  int kv_dim() const { return static_cast<int>(config_.num_kv_heads * config_.head_dim); }
};

}  // namespace llm
