// The milestone's end-to-end check: a whole random model run through the
// engine, compared against the same model run through torch.
//
// Two levels, and both matter. The prefill logits catch a kernel that is
// slightly wrong -- a drifting accumulation, a norm applied to the wrong
// buffer. The greedy token ids catch nothing extra when the logits are right,
// but they are the signal the milestone is actually stated in, and they fail
// loudly and unambiguously when the layer graph is wired wrong.
#include <cstdio>
#include <cstdlib>

#include "llm/model.h"
#include "ref_fixture.h"

int main(int argc, char** argv) {
  using namespace llm::testing;
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <ref_model.llmbin> <ref_forward.llmbin>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const llm::ModelFile ref = llm::ModelFile::load(argv[2]);
  const std::vector<std::int32_t> prompt = load_int32(ref, "prompt_ids");
  const std::vector<float> prefill_expected = load_fp32(ref, "prefill_logits");
  const std::vector<std::int32_t> generated_expected = load_int32(ref, "generated_ids");

  const int max_tokens = static_cast<int>(prompt.size() + generated_expected.size());
  llm::Model model = llm::Model::load(argv[1], max_tokens);

  std::printf("model: %u layers, hidden %u, vocab %u -- weights %.1f KiB, workspace %.1f KiB\n",
              model.config().num_layers, model.config().hidden_size, model.config().vocab_size,
              model.weight_bytes() / 1024.0, model.workspace_bytes() / 1024.0);

  model.forward(prompt.data(), static_cast<int>(prompt.size()));
  LLM_CUDA_CHECK_LAUNCH();
  const std::vector<float> prefill_got = model.logits_host(static_cast<int>(prompt.size()));

  // Looser than the per-kernel tolerances by design: this is the accumulated
  // error of every kernel in the stack, run twice per layer, and it is checked
  // here only to localize a failure that the token ids would report as a single
  // wrong number.
  bool ok = check_close("prefill logits", prefill_got, prefill_expected, 5e-3f, 5e-3f);

  const std::vector<std::int32_t> generated =
      model.generate(prompt, static_cast<int>(generated_expected.size()));
  ok &= check_int_equal("greedy ids", generated, generated_expected);

  if (!ok) return EXIT_FAILURE;
  std::printf("PASS: forward matches torch reference over %zu prompt + %zu generated tokens\n",
              prompt.size(), generated_expected.size());
  return EXIT_SUCCESS;
}
