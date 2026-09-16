// `llm-bench`: request-level latency and throughput for one model.
//
// Runs the same generation request repeatedly and reports distributions, not a
// single number: a single run hides the tail, and the tail is what a user of an
// interactive system actually feels.
//
// Metrics, all measured on the host with a steady clock -- i.e. as a caller of
// Model::generate() experiences them, including launch overhead and the
// device-to-host copy of each token:
//
//   TTFT  time to first token: request start -> first generated id on the host.
//         Dominated by prefill over the prompt.
//   TPOT  time per output token after the first: the gaps between consecutive
//         token callbacks. Also called inter-token latency.
//   E2E   request start -> last generated id on the host.
//   TPS   decode throughput, (tokens - 1) / (E2E - TTFT), so prefill doesn't
//         dilute it; plus the end-to-end rate tokens / E2E for reference.
//
// Model load (disk read + VRAM upload) is timed separately and excluded from
// every request metric -- it happens once per process, not once per request.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llm/cuda_check.h"
#include "llm/model.h"

namespace {

using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s --model <file.llmbin> (--prompt <id,id,...> | --prompt-len N)\n"
               "          [--tokens N] [--runs N] [--warmup N]\n"
               "\n"
               "  --model       path to a converted .llmbin model\n"
               "  --prompt      comma-separated int32 token ids (offline-tokenized)\n"
               "  --prompt-len  use a synthetic prompt of N ids instead; latency depends on\n"
               "                length, not content, so this is how to sweep prompt size\n"
               "  --tokens      tokens to generate per request (default 32)\n"
               "  --runs        measured requests (default 20)\n"
               "  --warmup      unmeasured requests first, to settle clocks and caches (default 2)\n",
               argv0);
}

std::vector<std::int32_t> parse_ids(const std::string& text) {
  std::vector<std::int32_t> ids;
  std::size_t pos = 0;
  while (pos <= text.size()) {
    const std::size_t comma = text.find(',', pos);
    const std::string piece =
        text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    if (!piece.empty()) ids.push_back(static_cast<std::int32_t>(std::strtol(piece.c_str(), nullptr, 10)));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return ids;
}

// BOS followed by a deterministic spread of ids. Content is irrelevant to
// latency; what matters is that the same prompt is used for every run.
std::vector<std::int32_t> synthetic_prompt(int length, std::uint32_t vocab) {
  std::vector<std::int32_t> ids(length);
  ids[0] = 1;
  for (int i = 1; i < length; ++i) {
    ids[i] = static_cast<std::int32_t>(100 + (static_cast<std::uint32_t>(i) * 7919u) % (vocab - 100));
  }
  return ids;
}

// Nearest-rank percentile: the smallest sample with at least p% of samples at
// or below it. No interpolation, so every reported value is one that was
// actually observed.
double percentile(std::vector<double> samples, double p) {
  std::sort(samples.begin(), samples.end());
  const std::size_t rank = static_cast<std::size_t>(std::ceil(p / 100.0 * samples.size()));
  return samples[rank == 0 ? 0 : rank - 1];
}

double mean(const std::vector<double>& samples) {
  double sum = 0.0;
  for (double s : samples) sum += s;
  return sum / samples.size();
}

void print_row(const char* name, const char* unit, const std::vector<double>& samples) {
  std::printf("  %-14s %-5s %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f\n", name, unit,
              mean(samples), percentile(samples, 50), percentile(samples, 90),
              percentile(samples, 99), *std::min_element(samples.begin(), samples.end()),
              *std::max_element(samples.begin(), samples.end()));
}

struct RunResult {
  double ttft_ms = 0.0;
  double e2e_ms = 0.0;
  std::vector<double> tpot_ms;  // one per token after the first
};

RunResult run_request(llm::Model& model, const std::vector<std::int32_t>& prompt, int tokens) {
  std::vector<Clock::time_point> stamps;
  stamps.reserve(tokens);

  const Clock::time_point start = Clock::now();
  model.generate(prompt, tokens, [&](int, std::int32_t) { stamps.push_back(Clock::now()); });

  RunResult r;
  r.ttft_ms = ms_between(start, stamps.front());
  r.e2e_ms = ms_between(start, stamps.back());
  for (std::size_t i = 1; i < stamps.size(); ++i) r.tpot_ms.push_back(ms_between(stamps[i - 1], stamps[i]));
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string prompt_text;
  int prompt_len = 0;
  int tokens = 32;
  int runs = 20;
  int warmup = 2;

  for (int i = 1; i < argc; ++i) {
    const bool has_value = i + 1 < argc;
    if (std::strcmp(argv[i], "--model") == 0 && has_value) {
      model_path = argv[++i];
    } else if (std::strcmp(argv[i], "--prompt") == 0 && has_value) {
      prompt_text = argv[++i];
    } else if (std::strcmp(argv[i], "--prompt-len") == 0 && has_value) {
      prompt_len = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--tokens") == 0 && has_value) {
      tokens = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--runs") == 0 && has_value) {
      runs = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0 && has_value) {
      warmup = std::atoi(argv[++i]);
    } else {
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  if (model_path.empty() || (prompt_text.empty() == (prompt_len <= 0)) || tokens < 2 || runs < 1 ||
      warmup < 0) {
    // Exactly one prompt source; at least two tokens so TPOT has a sample.
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  // Force context creation before the baseline reading, so the load delta
  // below is the model and nothing else.
  LLM_CUDA_CHECK(cudaFree(nullptr));
  std::size_t free_before = 0, total = 0;
  LLM_CUDA_CHECK(cudaMemGetInfo(&free_before, &total));

  // The workspace is sized before the model exists, so only the prompt's
  // *length* is needed here. A synthetic prompt's ids need the vocab size and
  // are filled in after load.
  std::vector<std::int32_t> prompt = prompt_text.empty() ? std::vector<std::int32_t>{} : parse_ids(prompt_text);
  const int prompt_tokens = prompt_text.empty() ? prompt_len : static_cast<int>(prompt.size());
  if (prompt_tokens <= 0) {
    std::fprintf(stderr, "[llm] empty prompt\n");
    return EXIT_FAILURE;
  }

  const Clock::time_point load_start = Clock::now();
  llm::Model model = llm::Model::load(model_path, prompt_tokens + tokens);
  LLM_CUDA_CHECK(cudaDeviceSynchronize());
  const double load_ms = ms_between(load_start, Clock::now());

  std::size_t free_after = 0;
  LLM_CUDA_CHECK(cudaMemGetInfo(&free_after, &total));

  if (prompt.empty()) prompt = synthetic_prompt(prompt_len, model.config().vocab_size);

  const double mib = 1024.0 * 1024.0;
  std::printf("model      %s\n", model_path.c_str());
  std::printf("request    prompt %d tokens, generate %d tokens | %d warmup + %d measured runs\n",
              prompt_tokens, tokens, warmup, runs);
  std::printf("load       %.0f ms (disk read + VRAM upload, excluded below)\n", load_ms);
  std::printf("vram       %.1f MiB measured by cudaMemGetInfo | %.1f weights + %.1f workspace "
              "accounted\n\n",
              (free_before - free_after) / mib, model.weight_bytes() / mib,
              model.workspace_bytes() / mib);

  for (int i = 0; i < warmup; ++i) run_request(model, prompt, tokens);

  std::vector<double> ttft, e2e, tps_decode, tps_e2e, tpot_all;
  std::vector<double> tpot_first, tpot_last;  // per run: first and last decode gap
  for (int i = 0; i < runs; ++i) {
    const RunResult r = run_request(model, prompt, tokens);
    ttft.push_back(r.ttft_ms);
    e2e.push_back(r.e2e_ms);
    tps_decode.push_back((tokens - 1) / ((r.e2e_ms - r.ttft_ms) / 1000.0));
    tps_e2e.push_back(tokens / (r.e2e_ms / 1000.0));
    tpot_all.insert(tpot_all.end(), r.tpot_ms.begin(), r.tpot_ms.end());
    tpot_first.push_back(r.tpot_ms.front());
    tpot_last.push_back(r.tpot_ms.back());
  }

  std::printf("  %-14s %-5s %10s %10s %10s %10s %10s %10s\n", "metric", "unit", "mean", "p50", "p90",
              "p99", "min", "max");
  print_row("TTFT", "ms", ttft);
  print_row("TPOT (all)", "ms", tpot_all);
  print_row("TPOT (1st)", "ms", tpot_first);
  print_row("TPOT (last)", "ms", tpot_last);
  print_row("E2E latency", "ms", e2e);
  print_row("TPS (decode)", "tok/s", tps_decode);
  print_row("TPS (e2e)", "tok/s", tps_e2e);

  if (runs < 100) {
    // Nearest-rank p99 of fewer than 100 samples is simply the maximum.
    std::printf("\n  note: p99 over %d runs is the max sample; use --runs 100+ for a real tail\n", runs);
  }
  return EXIT_SUCCESS;
}
