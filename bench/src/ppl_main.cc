// `llm-ppl`: perplexity of a model over an offline-tokenized text.
//
// The ids are cut into non-overlapping windows of --ctx tokens: a BOS, then
// ctx - 1 text ids. Each window is one fresh prefill; every position's logits
// score the id that follows it, so a window yields ctx - 1 predictions.
// Perplexity is exp(mean negative log-likelihood) over all of them.
//
// Validation: tools/references/hf_perplexity.py computes the same windows with torch

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "llm/model.h"

namespace {

constexpr std::int32_t kBosId = 1;

void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s --model <file.llmbin> --ids <file.ids> [--ctx N] [--windows N]\n"
               "\n"
               "  --model    path to a converted .llmbin model\n"
               "  --ids      flat little-endian int32 token ids (tools/convert/tokenize_text.py)\n"
               "  --ctx      window length in tokens, BOS included (default 512)\n"
               "  --windows  evaluate only the first N windows (default: all full windows)\n",
               argv0);
}

std::vector<std::int32_t> read_ids(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    std::fprintf(stderr, "[llm] cannot open %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  const std::streamsize bytes = in.tellg();
  std::vector<std::int32_t> ids(static_cast<std::size_t>(bytes) / sizeof(std::int32_t));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(ids.size() * sizeof(std::int32_t)));
  return ids;
}

// -log softmax(row)[target], in double so the sum over the vocab doesn't lose the small terms.
double nll(const float* row, std::size_t vocab, std::int32_t target) {
  const float max = *std::max_element(row, row + vocab);
  double sum = 0.0;
  for (std::size_t v = 0; v < vocab; ++v) sum += std::exp(static_cast<double>(row[v]) - max);
  return max + std::log(sum) - row[target];
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string ids_path;
  int ctx = 512;
  int max_windows = 0;

  for (int i = 1; i < argc; ++i) {
    const bool has_value = i + 1 < argc;
    if (std::strcmp(argv[i], "--model") == 0 && has_value) {
      model_path = argv[++i];
    } else if (std::strcmp(argv[i], "--ids") == 0 && has_value) {
      ids_path = argv[++i];
    } else if (std::strcmp(argv[i], "--ctx") == 0 && has_value) {
      ctx = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--windows") == 0 && has_value) {
      max_windows = std::atoi(argv[++i]);
    } else {
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  if (model_path.empty() || ids_path.empty() || ctx < 2) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  const std::vector<std::int32_t> text = read_ids(ids_path);
  const int stride = ctx - 1;
  int windows = static_cast<int>(text.size() / stride);
  if (max_windows > 0) windows = std::min(windows, max_windows);
  if (windows == 0) {
    std::fprintf(stderr, "[llm] %zu ids is less than one window of %d\n", text.size(), stride);
    return EXIT_FAILURE;
  }

  llm::Model model = llm::Model::load(model_path, ctx);
  const std::size_t vocab = model.config().vocab_size;
  std::printf("[llm] %zu ids, %d windows of %d (BOS + %d), %d predictions\n", text.size(), windows,
              ctx, stride, windows * stride);

  const auto start = std::chrono::steady_clock::now();
  std::vector<std::int32_t> window(ctx);
  double total_nll = 0.0;
  long long scored = 0;

  for (int w = 0; w < windows; ++w) {
    window[0] = kBosId;
    std::copy_n(text.begin() + static_cast<std::ptrdiff_t>(w) * stride, stride, window.begin() + 1);

    model.reset_cache();
    model.forward(window.data(), ctx);
    const std::vector<float> logits = model.logits_host(ctx);

    // Position t predicts window[t + 1]; the last position has nothing to score.
    for (int t = 0; t + 1 < ctx; ++t) total_nll += nll(logits.data() + t * vocab, vocab, window[t + 1]);
    scored += stride;

    if ((w + 1) % 50 == 0 || w + 1 == windows) {
      std::printf("  window %4d/%d  running ppl %.4f\n", w + 1, windows, std::exp(total_nll / scored));
      std::fflush(stdout);
    }
  }

  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  const double mean_nll = total_nll / scored;
  std::printf("tokens scored %lld\nmean nll      %.6f\nperplexity    %.4f\ntime          %.1f s\n", scored,
              mean_nll, std::exp(mean_nll), seconds);
  return EXIT_SUCCESS;
}
