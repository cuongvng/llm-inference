#include "llm/model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "llm/cuda_check.h"
#include "llm/kernels.h"
#include "llm/profile.h"

namespace llm {
namespace {

constexpr std::size_t kWeightAlign = 256;

std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

[[noreturn]] void fail(const std::string& message) {
  std::fprintf(stderr, "[llm] model: %s\n", message.c_str());
  std::exit(EXIT_FAILURE);
}

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

// Analytic cost models for the kernel profiler.
constexpr double kF32 = 4.0;
constexpr double kF16 = 2.0;

struct Cost {
  double flops = 0.0;
  double ideal = 0.0;
  double moved = 0.0;
  double dram = 0.0;
};

double dram_term(double bytes, double factor) {
  return bytes <= l2_cache_bytes() ? bytes : bytes * factor;
}

Cost embedding_cost(int tokens, int hidden) {
  const double n = static_cast<double>(tokens) * hidden;
  return {0.0, n * (kF16 + kF32), n * (kF16 + kF32), n * (kF16 + kF32)};
}

Cost rmsnorm_cost(int rows, int hidden) {
  const double n = static_cast<double>(rows) * hidden;
  // Square-and-accumulate, then scale by the norm and by the weight.
  const double ideal = 2.0 * n * kF32 + hidden * kF16;
  return {4.0 * n, ideal, 3.0 * n * kF32 + hidden * kF16, ideal};
}

Cost rope_cost(int tokens, int q_dim, int kv_dim) {
  const double n = static_cast<double>(tokens) * (q_dim + kv_dim);
  // Each rotated pair is 4 multiplies and 2 adds; the sin/cos are not counted.
  return {3.0 * n, 2.0 * n * kF32, 2.0 * n * kF32, 2.0 * n * kF32};
}

Cost gemm_cost(int m, int n, int k) {
  constexpr double kTile = 16.0;
  const double M = m, N = n, K = k;
  const double tiles_m = std::ceil(M / kTile);
  const double tiles_n = std::ceil(N / kTile);
  Cost c;
  c.flops = 2.0 * M * N * K;
  c.ideal = N * K * kF16 + M * K * kF32 + M * N * kF32;
  c.moved = tiles_m * tiles_n * kTile * K * (kF32 + kF16) + M * N * kF32;
  c.dram = dram_term(N * K * kF16, tiles_m) + dram_term(M * K * kF32, tiles_n) + M * N * kF32;
  return c;
}

// Query i sits at absolute position n_keys - n_query + i, so it may see that
// many keys plus itself; summing over the queries gives the live score count.
double live_scores(double n_query, double n_keys) {
  return n_query * n_keys - n_query * (n_query - 1.0) / 2.0;
}

Cost attention_qk_cost(int n_query, int n_keys, int heads, int kv_heads, int head_dim) {
  const double h = heads, nq = n_query, nk = n_keys, d = head_dim;
  // Causal masking leaves only part of the score matrix live; the masked
  // entries cost a store and no arithmetic.
  const double live = h * live_scores(nq, nk);
  Cost c;
  c.flops = 2.0 * d * live;
  c.ideal = nq * h * d * kF32 + nk * kv_heads * d * kF32 + h * nq * nk * kF32;
  // One block per (query, head) stages its query once and streams the keys it
  // is allowed to see, so K is re-read once per query row.
  c.moved = h * nq * d * kF32 + live * d * kF32 + h * nq * nk * kF32;
  c.dram = h * nq * d * kF32 + dram_term(nk * kv_heads * d * kF32, live / nq) + h * nq * nk * kF32;
  return c;
}

Cost softmax_cost(int rows, int row_len) {
  const double n = static_cast<double>(rows) * row_len;
  // 3 passes -- max, exp-sum, normalize -- so 4 row-sized accesses.
  return {4.0 * n, 2.0 * n * kF32, 4.0 * n * kF32, 2.0 * n * kF32};
}

// Unlike qk this reads the masked entries too -- they are zeros after softmax,
// so the full n_query * n_keys matrix is touched.
Cost attention_av_cost(int n_query, int n_keys, int heads, int kv_heads, int head_dim) {
  const double h = heads, nq = n_query, nk = n_keys, d = head_dim;
  Cost c;
  c.flops = 2.0 * h * nq * nk * d;
  c.ideal = h * nq * nk * kF32 + nk * kv_heads * d * kF32 + nq * h * d * kF32;
  c.moved = h * nq * nk * kF32 + h * nq * nk * d * kF32 + nq * h * d * kF32;
  c.dram = h * nq * nk * kF32 + dram_term(nk * kv_heads * d * kF32, h * nq) + nq * h * d * kF32;
  return c;
}

// A straight copy of K and V into the cache: each is read once and written once.
Cost kv_write_cost(int n_new, int kv_dim) {
  const double b = 4.0 * static_cast<double>(n_new) * kv_dim * kF32;
  return {0.0, b, b, b};
}

Cost swiglu_cost(std::size_t n) {
  const double e = static_cast<double>(n);
  return {5.0 * e, 3.0 * e * kF32, 3.0 * e * kF32, 3.0 * e * kF32};
}

Cost residual_cost(std::size_t n) {
  const double e = static_cast<double>(n);
  return {e, 3.0 * e * kF32, 3.0 * e * kF32, 3.0 * e * kF32};
}

Cost argmax_cost(int n) {
  const double e = static_cast<double>(n);
  return {0.0, e * kF32, e * kF32, e * kF32};
}

void profile_begin_cost(const char* name, const Cost& c) {
  profile_begin(name, c.flops, c.ideal, c.moved, c.dram);
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
  model.cache_.allocate(c, max_tokens);

  return model;
}

void Model::forward(const std::int32_t* host_ids, int n_tokens) {
  if (n_tokens <= 0) fail("forward called with no tokens");
  if (n_tokens > act_.max_tokens) {
    fail("forward called with " + std::to_string(n_tokens) + " tokens, workspace holds " +
         std::to_string(act_.max_tokens));
  }

  const int cache_len = cache_.length();
  if (cache_len + n_tokens > cache_.max_tokens()) {
    fail("sequence of " + std::to_string(cache_len + n_tokens) + " tokens exceeds cache capacity " +
         std::to_string(cache_.max_tokens()));
  }

  std::vector<std::int32_t> positions(n_tokens);
  for (int i = 0; i < n_tokens; ++i) positions[i] = cache_len + i;
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
  const int n_keys = cache_len + n_tokens;

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

  profile_begin_cost("embedding", embedding_cost(n_tokens, hidden));
  launch_embedding_lookup(tok_embeddings_, d_ids, x, n_tokens, hidden);
  profile_end();

  for (std::size_t li = 0; li < layers_.size(); ++li) {
    const LayerWeights& layer = layers_[li];
    float* k_cache = cache_.layer_k(static_cast<int>(li));
    float* v_cache = cache_.layer_v(static_cast<int>(li));

    profile_begin_cost("rmsnorm", rmsnorm_cost(n_tokens, hidden));
    launch_rmsnorm(x, layer.attn_norm, normed, n_tokens, hidden, eps);
    profile_end();

    Cost qkv = gemm_cost(n_tokens, q_dim(), hidden);
    const Cost kv_one = gemm_cost(n_tokens, kv_dim(), hidden);
    qkv.flops += 2.0 * kv_one.flops;
    qkv.ideal += 2.0 * kv_one.ideal;
    qkv.moved += 2.0 * kv_one.moved;
    qkv.dram += 2.0 * kv_one.dram;
    profile_begin_cost("gemm.qkv", qkv);
    launch_gemm_fp16(layer.wq, normed, q, n_tokens, q_dim(), hidden);
    launch_gemm_fp16(layer.wk, normed, k, n_tokens, kv_dim(), hidden);
    launch_gemm_fp16(layer.wv, normed, v, n_tokens, kv_dim(), hidden);
    profile_end();

    profile_begin_cost("rope", rope_cost(n_tokens, q_dim(), kv_dim()));
    launch_rope(q, k, d_positions, n_tokens, heads, kv_heads, head_dim, config_.rope_theta);
    profile_end();

    profile_begin_cost("kv_write", kv_write_cost(n_tokens, kv_dim()));
    launch_kv_cache_write(k, v, k_cache, v_cache, n_tokens, cache_len, kv_dim());
    profile_end();

    // Attention reads KV cache
    profile_begin_cost("attn.qk", attention_qk_cost(n_tokens, n_keys, heads, kv_heads, head_dim));
    launch_attention_qk(q, k_cache, scores, n_tokens, n_keys, heads, kv_heads, head_dim,
                        cache_len);
    profile_end();

    profile_begin_cost("attn.softmax", softmax_cost(heads * n_tokens, n_keys));
    launch_softmax_rows(scores, heads * n_tokens, n_keys);
    profile_end();

    profile_begin_cost("attn.av", attention_av_cost(n_tokens, n_keys, heads, kv_heads, head_dim));
    launch_attention_av(scores, v_cache, attn_out, n_tokens, n_keys, heads, kv_heads, head_dim);
    profile_end();

    profile_begin_cost("gemm.wo", gemm_cost(n_tokens, hidden, q_dim()));
    launch_gemm_fp16(layer.wo, attn_out, proj, n_tokens, hidden, q_dim());
    profile_end();

    profile_begin_cost("residual", residual_cost(hidden_elems));
    launch_residual_add(x, proj, hidden_elems);
    profile_end();

    profile_begin_cost("rmsnorm", rmsnorm_cost(n_tokens, hidden));
    launch_rmsnorm(x, layer.ffn_norm, normed, n_tokens, hidden, eps);
    profile_end();

    Cost gate_up = gemm_cost(n_tokens, inter, hidden);
    gate_up.flops *= 2.0;
    gate_up.ideal *= 2.0;
    gate_up.moved *= 2.0;
    gate_up.dram *= 2.0;
    profile_begin_cost("gemm.gate_up", gate_up);
    launch_gemm_fp16(layer.gate_proj, normed, gate, n_tokens, inter, hidden);
    launch_gemm_fp16(layer.up_proj, normed, up, n_tokens, inter, hidden);
    profile_end();

    profile_begin_cost("swiglu", swiglu_cost(static_cast<std::size_t>(n_tokens) * inter));
    launch_swiglu(gate, up, activated, static_cast<std::size_t>(n_tokens) * inter);
    profile_end();

    profile_begin_cost("gemm.down", gemm_cost(n_tokens, hidden, inter));
    launch_gemm_fp16(layer.down_proj, activated, proj, n_tokens, hidden, inter);
    profile_end();

    profile_begin_cost("residual", residual_cost(hidden_elems));
    launch_residual_add(x, proj, hidden_elems);
    profile_end();
  }

  cache_.advance(n_tokens);

  profile_begin_cost("rmsnorm", rmsnorm_cost(n_tokens, hidden));
  launch_rmsnorm(x, output_norm_, normed, n_tokens, hidden, eps);
  profile_end();

  profile_begin_cost("gemm.lm_head",
                     gemm_cost(n_tokens, static_cast<int>(config_.vocab_size), hidden));
  launch_gemm_fp16(lm_head_, normed, logits, n_tokens, static_cast<int>(config_.vocab_size),
                   hidden);
  profile_end();
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
  profile_begin_cost("argmax", argmax_cost(static_cast<int>(config_.vocab_size)));
  launch_argmax(last_row, act_.next_token.as<std::int32_t>(),
                static_cast<int>(config_.vocab_size));
  profile_end();
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

  reset_cache();
  forward(prompt.data(), static_cast<int>(prompt.size()));
  int last_forward_tokens = static_cast<int>(prompt.size());

  for (int step = 0; step < max_new_tokens; ++step) {
    if (static_cast<int>(sequence.size()) > act_.max_tokens) break;
    const std::int32_t next = argmax_last(last_forward_tokens);
    generated.push_back(next);
    sequence.push_back(next);
    if (on_token) on_token(step, next);
    if (step + 1 < max_new_tokens && static_cast<int>(sequence.size()) <= act_.max_tokens) {
      forward(&next, 1);
      last_forward_tokens = 1;
    }
  }
  return generated;
}

}  // namespace llm
