#include "llm/model.h"

#include <cstdio>
#include <cstdlib>

#include "llm/cuda_check.h"
#include "llm/kernels.h"

namespace llm {
namespace {

// Device allocations are 256-byte aligned by cudaMalloc; keeping every tensor
// inside the shared weight buffer on the same boundary means a tensor's rows
// start where the memory system expects them to, and costs a few KB total.
constexpr std::size_t kWeightAlign = 256;

std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

[[noreturn]] void fail(const std::string& message) {
  std::fprintf(stderr, "[llm] model: %s\n", message.c_str());
  std::exit(EXIT_FAILURE);
}

// Every weight must be FP16 at this milestone; the quantized dtypes get their
// own load path rather than being silently widened here.
void check_tensor(const std::string& name, const TensorView& t, std::size_t expect_elems) {
  if (t.dtype != DType::kFP16) {
    fail("tensor '" + name + "' is not FP16 (quantized weights need the quantized load path)");
  }
  const std::size_t elems = t.nbytes / dtype_size_bytes(DType::kFP16);
  if (elems != expect_elems) {
    fail("tensor '" + name + "' has " + std::to_string(elems) + " elements, expected " +
         std::to_string(expect_elems));
  }
}

}  // namespace

void Activations::allocate(const ModelConfig& config, int tokens) {
  max_tokens = tokens;
  const std::size_t t = static_cast<std::size_t>(tokens);
  const std::size_t hidden = config.hidden_size;
  const std::size_t inter = config.intermediate_size;
  const std::size_t q_dim = static_cast<std::size_t>(config.num_heads) * config.head_dim;
  const std::size_t kv_dim = static_cast<std::size_t>(config.num_kv_heads) * config.head_dim;

  ids.allocate(t * sizeof(std::int32_t));
  positions.allocate(t * sizeof(std::int32_t));

  x.allocate(t * hidden * sizeof(float));
  normed.allocate(t * hidden * sizeof(float));
  q.allocate(t * q_dim * sizeof(float));
  k.allocate(t * kv_dim * sizeof(float));
  v.allocate(t * kv_dim * sizeof(float));
  // The score matrix is the one workspace buffer that grows quadratically with
  // sequence length -- the term that decides how long a prompt fits in 4GB, and
  // the first thing an online-softmax attention would delete.
  scores.allocate(static_cast<std::size_t>(config.num_heads) * t * t * sizeof(float));
  attn_out.allocate(t * q_dim * sizeof(float));
  proj.allocate(t * hidden * sizeof(float));
  gate.allocate(t * inter * sizeof(float));
  up.allocate(t * inter * sizeof(float));
  act.allocate(t * inter * sizeof(float));
  logits.allocate(t * config.vocab_size * sizeof(float));
  next_token.allocate(sizeof(std::int32_t));
}

std::size_t Activations::total_bytes() const {
  return ids.bytes() + positions.bytes() + x.bytes() + normed.bytes() + q.bytes() + k.bytes() +
         v.bytes() + scores.bytes() + attn_out.bytes() + proj.bytes() + gate.bytes() + up.bytes() +
         act.bytes() + logits.bytes() + next_token.bytes();
}

Model Model::load(const std::string& path, int max_tokens) {
  const ModelFile file = ModelFile::load(path);
  const ModelConfig& c = file.config();

  Model model;
  model.config_ = c;

  const std::size_t hidden = c.hidden_size;
  const std::size_t inter = c.intermediate_size;
  const std::size_t q_dim = static_cast<std::size_t>(c.num_heads) * c.head_dim;
  const std::size_t kv_dim = static_cast<std::size_t>(c.num_kv_heads) * c.head_dim;

  // Name, expected element count. Order here is also the upload order, so
  // tensors used together in a layer land next to each other in VRAM.
  std::vector<std::pair<std::string, std::size_t>> wanted = {
      {"tok_embeddings.weight", c.vocab_size * hidden},
      {"output_norm.weight", hidden},
      {"lm_head.weight", c.vocab_size * hidden},
  };
  for (std::uint32_t i = 0; i < c.num_layers; ++i) {
    const std::string p = "layers." + std::to_string(i) + ".";
    wanted.push_back({p + "attn_norm.weight", hidden});
    wanted.push_back({p + "attn.wq.weight", q_dim * hidden});
    wanted.push_back({p + "attn.wk.weight", kv_dim * hidden});
    wanted.push_back({p + "attn.wv.weight", kv_dim * hidden});
    wanted.push_back({p + "attn.wo.weight", hidden * q_dim});
    wanted.push_back({p + "ffn_norm.weight", hidden});
    wanted.push_back({p + "mlp.gate_proj.weight", inter * hidden});
    wanted.push_back({p + "mlp.up_proj.weight", inter * hidden});
    wanted.push_back({p + "mlp.down_proj.weight", hidden * inter});
  }

  // Size the whole thing first, then make exactly one cudaMalloc for it. This
  // is the bump allocator's job in miniature: weights are loaded once and live
  // for the process, so they need no free tracking and no per-tensor handle.
  std::size_t total = 0;
  std::vector<std::size_t> offsets;
  offsets.reserve(wanted.size());
  for (const auto& [name, expect] : wanted) {
    const TensorView& t = file.require(name);
    check_tensor(name, t, expect);
    offsets.push_back(total);
    total = align_up(total + t.nbytes, kWeightAlign);
  }
  model.weights_.allocate(total);

  std::vector<const __half*> device_ptrs;
  device_ptrs.reserve(wanted.size());
  auto* base = model.weights_.as<std::uint8_t>();
  for (std::size_t i = 0; i < wanted.size(); ++i) {
    const TensorView& t = file.require(wanted[i].first);
    LLM_CUDA_CHECK(
        cudaMemcpy(base + offsets[i], t.data, t.nbytes, cudaMemcpyHostToDevice));
    device_ptrs.push_back(reinterpret_cast<const __half*>(base + offsets[i]));
  }

  model.tok_embeddings_ = device_ptrs[0];
  model.output_norm_ = device_ptrs[1];
  model.lm_head_ = device_ptrs[2];

  model.layers_.resize(c.num_layers);
  for (std::uint32_t i = 0; i < c.num_layers; ++i) {
    const std::size_t b = 3 + static_cast<std::size_t>(i) * 9;
    LayerWeights& l = model.layers_[i];
    l.attn_norm = device_ptrs[b + 0];
    l.wq = device_ptrs[b + 1];
    l.wk = device_ptrs[b + 2];
    l.wv = device_ptrs[b + 3];
    l.wo = device_ptrs[b + 4];
    l.ffn_norm = device_ptrs[b + 5];
    l.gate_proj = device_ptrs[b + 6];
    l.up_proj = device_ptrs[b + 7];
    l.down_proj = device_ptrs[b + 8];
  }

  if (max_tokens <= 0) fail("max_tokens must be positive");
  if (static_cast<std::uint32_t>(max_tokens) > c.max_seq_len) {
    fail("max_tokens " + std::to_string(max_tokens) + " exceeds the model's max_seq_len " +
         std::to_string(c.max_seq_len));
  }
  model.act_.allocate(c, max_tokens);

  return model;
}

void Model::forward(const std::int32_t* host_ids, int n_tokens) {
  if (n_tokens <= 0) fail("forward called with no tokens");
  if (n_tokens > act_.max_tokens) {
    fail("forward called with " + std::to_string(n_tokens) + " tokens, workspace holds " +
         std::to_string(act_.max_tokens));
  }

  // Positions are 0..n_tokens-1 for a cacheless full-sequence pass. They are
  // uploaded rather than assumed by the kernel because a cached decode step
  // passes a single, much larger position.
  std::vector<std::int32_t> positions(n_tokens);
  for (int i = 0; i < n_tokens; ++i) positions[i] = i;
  act_.ids.upload(host_ids, static_cast<std::size_t>(n_tokens) * sizeof(std::int32_t));
  act_.positions.upload(positions.data(),
                        static_cast<std::size_t>(n_tokens) * sizeof(std::int32_t));

  const int hidden = static_cast<int>(config_.hidden_size);
  const int inter = static_cast<int>(config_.intermediate_size);
  const int heads = static_cast<int>(config_.num_heads);
  const int kv_heads = static_cast<int>(config_.num_kv_heads);
  const int head_dim = static_cast<int>(config_.head_dim);
  const float eps = config_.rms_norm_eps;
  const std::size_t hidden_elems = static_cast<std::size_t>(n_tokens) * hidden;

  const std::int32_t* d_ids = act_.ids.as<std::int32_t>();
  const std::int32_t* d_positions = act_.positions.as<std::int32_t>();
  float* x = act_.x.as<float>();
  float* normed = act_.normed.as<float>();
  float* q = act_.q.as<float>();
  float* k = act_.k.as<float>();
  float* v = act_.v.as<float>();
  float* scores = act_.scores.as<float>();
  float* attn_out = act_.attn_out.as<float>();
  float* proj = act_.proj.as<float>();
  float* gate = act_.gate.as<float>();
  float* up = act_.up.as<float>();
  float* activated = act_.act.as<float>();
  float* logits = act_.logits.as<float>();

  // The layer graph. Every step is a launch from llm/kernels.h -- no allocation,
  // no host round trip, nothing between the launches.
  launch_embedding_lookup(tok_embeddings_, d_ids, x, n_tokens, hidden);

  for (const LayerWeights& layer : layers_) {
    // Writes to a separate buffer, not back over x: the residual add below
    // needs the un-normed stream.
    launch_rmsnorm(x, layer.attn_norm, normed, n_tokens, hidden, eps);

    launch_gemm_fp16(layer.wq, normed, q, n_tokens, q_dim(), hidden);
    launch_gemm_fp16(layer.wk, normed, k, n_tokens, kv_dim(), hidden);
    launch_gemm_fp16(layer.wv, normed, v, n_tokens, kv_dim(), hidden);

    launch_rope(q, k, d_positions, n_tokens, heads, kv_heads, head_dim, config_.rope_theta);

    launch_attention_qk(q, k, scores, n_tokens, n_tokens, heads, kv_heads, head_dim,
                        /*query_pos_offset=*/0);
    // The score matrix is heads * n_tokens independent rows of length n_tokens;
    // the causal mask is already baked into it as -inf.
    launch_softmax_rows(scores, heads * n_tokens, n_tokens);
    launch_attention_av(scores, v, attn_out, n_tokens, n_tokens, heads, kv_heads, head_dim);

    launch_gemm_fp16(layer.wo, attn_out, proj, n_tokens, hidden, q_dim());
    launch_residual_add(x, proj, hidden_elems);

    launch_rmsnorm(x, layer.ffn_norm, normed, n_tokens, hidden, eps);
    launch_gemm_fp16(layer.gate_proj, normed, gate, n_tokens, inter, hidden);
    launch_gemm_fp16(layer.up_proj, normed, up, n_tokens, inter, hidden);
    launch_swiglu(gate, up, activated, static_cast<std::size_t>(n_tokens) * inter);
    launch_gemm_fp16(layer.down_proj, activated, proj, n_tokens, hidden, inter);
    launch_residual_add(x, proj, hidden_elems);
  }

  launch_rmsnorm(x, output_norm_, normed, n_tokens, hidden, eps);
  // Logits for every position, not just the last: the reference checks all of
  // them, which localizes a failure the token ids alone would report as one
  // wrong number. With a KV cache only the final row needs computing.
  launch_gemm_fp16(lm_head_, normed, logits, n_tokens, static_cast<int>(config_.vocab_size),
                   hidden);
}

std::vector<float> Model::logits_host(int n_tokens) {
  const std::size_t count = static_cast<std::size_t>(n_tokens) * config_.vocab_size;
  std::vector<float> out(count);
  LLM_CUDA_CHECK(cudaDeviceSynchronize());
  act_.logits.download(out.data(), count * sizeof(float));
  return out;
}

std::int32_t Model::argmax_last(int n_tokens) {
  const float* last_row =
      act_.logits.as<float>() + static_cast<std::size_t>(n_tokens - 1) * config_.vocab_size;
  launch_argmax(last_row, act_.next_token.as<std::int32_t>(),
                static_cast<int>(config_.vocab_size));
  std::int32_t token = -1;
  LLM_CUDA_CHECK(cudaDeviceSynchronize());
  act_.next_token.download(&token, sizeof(token));
  return token;
}

std::vector<std::int32_t> Model::generate(const std::vector<std::int32_t>& prompt,
                                          int max_new_tokens, const TokenCallback& on_token) {
  if (prompt.empty()) fail("generate called with an empty prompt");

  std::vector<std::int32_t> sequence = prompt;
  std::vector<std::int32_t> generated;
  generated.reserve(max_new_tokens);

  for (int step = 0; step < max_new_tokens; ++step) {
    if (static_cast<int>(sequence.size()) > act_.max_tokens) break;
    // Recomputing the whole prefix every step is quadratic in the sequence
    // length. That cost is the baseline the KV cache is measured against, so it
    // stays exactly this naive here.
    forward(sequence.data(), static_cast<int>(sequence.size()));
    const std::int32_t next = argmax_last(static_cast<int>(sequence.size()));
    generated.push_back(next);
    sequence.push_back(next);
    if (on_token) on_token(step, next);
  }
  return generated;
}

}  // namespace llm
