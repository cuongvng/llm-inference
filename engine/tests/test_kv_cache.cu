// Checks the cached decode path against the cacheless one on the same model:
// incremental decode, then one full pass over the sequence it produced. A
// cache, position or masking bug shows up here without routing through torch.
#include <cstdio>
#include <cstdlib>

#include "llm/model.h"
#include "ref_fixture.h"

namespace {

constexpr int kSteps = 200;

}  // namespace

int main(int argc, char** argv) {
  using namespace llm::testing;
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <ref_model.llmbin>\n", argv[0]);
    return EXIT_FAILURE;
  }

  std::vector<std::int32_t> prompt = {3, 17, 42, 8, 55};
  const int max_tokens = static_cast<int>(prompt.size()) + kSteps;
  llm::Model model = llm::Model::load(argv[1], max_tokens);
  for (std::int32_t& id : prompt) id %= static_cast<std::int32_t>(model.config().vocab_size);

  // Incremental: prefill once, then one token per step out of the cache.
  const std::vector<std::int32_t> incremental = model.generate(prompt, kSteps);
  LLM_CUDA_CHECK_LAUNCH();

  std::vector<std::int32_t> sequence = prompt;
  sequence.insert(sequence.end(), incremental.begin(), incremental.end());

  // generate() stops before forwarding its last token; this extends the cache
  // over the whole sequence so the two logits rows describe the same thing.
  model.forward(&sequence.back(), 1);
  const std::vector<float> cached_row = model.logits_host(1);

  // Cacheless: re-run the whole prefix every step, which is what the cache
  // claims to be an optimization of.
  std::vector<std::int32_t> cacheless;
  std::vector<std::int32_t> ids = prompt;
  for (int step = 0; step < kSteps; ++step) {
    model.reset_cache();
    model.forward(ids.data(), static_cast<int>(ids.size()));
    const std::int32_t next = model.argmax_last(static_cast<int>(ids.size()));
    cacheless.push_back(next);
    ids.push_back(next);
  }

  model.reset_cache();
  model.forward(sequence.data(), static_cast<int>(sequence.size()));
  const std::vector<float> full = model.logits_host(static_cast<int>(sequence.size()));
  const std::size_t vocab = model.config().vocab_size;
  const std::vector<float> cacheless_row(full.end() - vocab, full.end());

  std::printf("model: %u layers, hidden %u, vocab %u -- cache %.1f KiB over %d tokens\n",
              model.config().num_layers, model.config().hidden_size, model.config().vocab_size,
              model.cache_bytes() / 1024.0, max_tokens);

  // Zero tolerance: both paths reduce over the same keys in the same order, so
  // the cache may not perturb the result at all, not merely stay close.
  bool ok = check_int_equal("decoded ids", incremental, cacheless);
  ok &= check_close("final logits", cached_row, cacheless_row, 0.0f, 0.0f);

  if (!ok) return EXIT_FAILURE;
  std::printf("PASS: %d incremental decode steps match a cacheless pass exactly\n", kSteps);
  return EXIT_SUCCESS;
}
