#include "llm/model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "llm/cuda_check.h"
#include "llm/kernels.h"
#include "llm/profile.h"

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

// Analytic cost models for the kernel profiler.
//
// `ideal` is the compulsory traffic: every distinct input read once, every
// output written once -- what a perfect implementation of the operator would
// move. `moved` is what this implementation's blocking actually asks the memory
// system for. Where the two differ, the difference is re-read, and the ratio is
// the first thing to look at when a kernel is slower than its bytes suggest.
//
// Activations are FP32 and weights FP16 throughout.
constexpr double kF32 = 4.0;
constexpr double kF16 = 2.0;

struct Cost {
  double flops = 0.0;
  double ideal = 0.0;
  double moved = 0.0;
  double dram = 0.0;
};

// An operand read `factor` times costs DRAM bandwidth only if it is too big for
// L2 to hold between those reads. This is the one place a cache is modelled at
// all, and it is what separates "the tiling asks for 110 GB/s" from "the memory
// bus actually carries 35 GB/s".
double dram_term(double bytes, double factor) {
  return bytes <= l2_cache_bytes() ? bytes : bytes * factor;
}

Cost embedding_cost(int tokens, int hidden) {
  const double n = static_cast<double>(tokens) * hidden;
  // Distinct rows of a table far larger than L2: every gathered byte is a miss.
  return {0.0, n * (kF16 + kF32), n * (kF16 + kF32), n * (kF16 + kF32)};
}

Cost rmsnorm_cost(int rows, int hidden) {
  const double n = static_cast<double>(rows) * hidden;
  // Square-and-accumulate, then scale by the norm and by the weight.
  // The second pass over the row is a re-read at request level even though a
  // row this small is served from L1 in practice.
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
  // Every output tile stages its own 16 rows of x and 16 rows of the weight
  // across the whole of k, so the weight matrix is re-read once per row-tile of
  // m. That factor -- ceil(m / 16) -- is why a longer sequence costs more per
  // token here than the FLOP count alone predicts.
  c.moved = tiles_m * tiles_n * kTile * K * (kF32 + kF16) + M * N * kF32;
  // The weight matrix is megabytes and is re-read once per row-tile: all of
  // that crosses the bus. The x tile is a few hundred KB at these sequence
  // lengths, so its tiles_n re-reads are served by L2.
  c.dram = dram_term(N * K * kF16, tiles_m) + dram_term(M * K * kF32, tiles_n) + M * N * kF32;
  return c;
}

Cost attention_qk_cost(int tokens, int heads, int kv_heads, int head_dim) {
  const double h = heads, t = tokens, d = head_dim;
  // Causal masking leaves half the score matrix live; the masked half costs a
  // store and no arithmetic.
  const double live = h * t * (t + 1.0) / 2.0;
  Cost c;
  c.flops = 2.0 * d * live;
  c.ideal = t * h * d * kF32 + t * kv_heads * d * kF32 + h * t * t * kF32;
  // One block per (query, head) stages its query once and streams the keys it
  // is allowed to see, so K is re-read once per query row.
  c.moved = h * t * d * kF32 + live * d * kF32 + h * t * t * kF32;
  c.dram = h * t * d * kF32 + dram_term(t * kv_heads * d * kF32, live / t) + h * t * t * kF32;
  return c;
}

Cost softmax_cost(int rows, int row_len) {
  const double n = static_cast<double>(rows) * row_len;
  // Three passes -- max, exp-sum, normalize -- so four row-sized accesses.
  return {4.0 * n, 2.0 * n * kF32, 4.0 * n * kF32, 2.0 * n * kF32};
}

Cost attention_av_cost(int tokens, int heads, int kv_heads, int head_dim) {
  const double h = heads, t = tokens, d = head_dim;
  Cost c;
  // Masked keys carry probability 0 but are still multiplied through, so the
  // full rectangle is executed.
  c.flops = 2.0 * h * t * t * d;
  c.ideal = h * t * t * kF32 + t * kv_heads * d * kF32 + t * h * d * kF32;
  // The probability row broadcasts across the block; V is re-read in full by
  // every (query, head) block.
  c.moved = h * t * t * kF32 + h * t * t * d * kF32 + t * h * d * kF32;
  c.dram = h * t * t * kF32 + dram_term(t * kv_heads * d * kF32, h * t) + t * h * d * kF32;
  return c;
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
  model.cache_.allocate(c, max_tokens);

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
  //
  // The profile_begin/profile_end pairs are inert unless the profiler is
  // enabled. Each names a region of the graph rather than a single launch, so
  // the q/k/v projections (identical shapes, launched back to back) report as
  // one line.
  profile_begin_cost("embedding", embedding_cost(n_tokens, hidden));
  launch_embedding_lookup(tok_embeddings_, d_ids, x, n_tokens, hidden);
  profile_end();

  for (const LayerWeights& layer : layers_) {
    // Writes to a separate buffer, not back over x: the residual add below
    // needs the un-normed stream.
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

    profile_begin_cost("attn.qk", attention_qk_cost(n_tokens, heads, kv_heads, head_dim));
    launch_attention_qk(q, k, scores, n_tokens, n_tokens, heads, kv_heads, head_dim,
                        /*query_pos_offset=*/0);
    profile_end();

    // The score matrix is heads * n_tokens independent rows of length n_tokens;
    // the causal mask is already baked into it as -inf.
    profile_begin_cost("attn.softmax", softmax_cost(heads * n_tokens, n_tokens));
    launch_softmax_rows(scores, heads * n_tokens, n_tokens);
    profile_end();

    profile_begin_cost("attn.av", attention_av_cost(n_tokens, heads, kv_heads, head_dim));
    launch_attention_av(scores, v, attn_out, n_tokens, n_tokens, heads, kv_heads, head_dim);
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

  profile_begin_cost("rmsnorm", rmsnorm_cost(n_tokens, hidden));
  launch_rmsnorm(x, output_norm_, normed, n_tokens, hidden, eps);
  profile_end();

  // Logits for every position, not just the last: the reference checks all of
  // them, which localizes a failure the token ids alone would report as one
  // wrong number. With a KV cache only the final row needs computing.
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
