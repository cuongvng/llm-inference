#include "llm/profile.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>

#include "llm/cuda_check.h"

namespace llm {
namespace {

// Bound on event pairs held before an automatic flush. A flush synchronizes, so
// it must not land in the middle of a pass; this is sized well above the ~350
// regions one forward pass records, and the generation loop synchronizes per
// token anyway.
constexpr std::size_t kMaxPending = 8192;

}  // namespace

KernelProfiler& KernelProfiler::get() {
  static KernelProfiler profiler;
  return profiler;
}

std::size_t KernelProfiler::acquire_event() {
  if (cursor_ == events_.size()) {
    cudaEvent_t event = nullptr;
    LLM_CUDA_CHECK(cudaEventCreate(&event));
    events_.push_back(event);
  }
  return cursor_++;
}

void KernelProfiler::begin(const char* name, double flops, double ideal_bytes, double moved_bytes,
                           double dram_bytes) {
  if (!enabled_) return;
  if (in_region_) {
    std::fprintf(stderr, "[llm] profile: region '%s' opened inside another\n", name);
    return;
  }
  if (pending_.size() >= kMaxPending) flush();

  const std::size_t start = acquire_event();
  LLM_CUDA_CHECK(cudaEventRecord(static_cast<cudaEvent_t>(events_[start])));
  pending_.push_back({name, flops, ideal_bytes, moved_bytes, dram_bytes, start, 0});
  in_region_ = true;
}

void KernelProfiler::end() {
  if (!enabled_ || !in_region_) return;
  const std::size_t stop = acquire_event();
  LLM_CUDA_CHECK(cudaEventRecord(static_cast<cudaEvent_t>(events_[stop])));
  pending_.back().stop = stop;
  in_region_ = false;
}

void KernelProfiler::flush() {
  if (pending_.empty()) return;
  LLM_CUDA_CHECK(cudaDeviceSynchronize());

  for (const Pending& p : pending_) {
    float ms = 0.0f;
    LLM_CUDA_CHECK(cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(events_[p.start]),
                                        static_cast<cudaEvent_t>(events_[p.stop])));
    auto it = std::find_if(stats_.begin(), stats_.end(),
                           [&](const KernelStat& s) { return s.name == p.name; });
    if (it == stats_.end()) {
      stats_.push_back(KernelStat{p.name, 0, 0.0, 0.0, 0.0, 0.0, 0.0});
      it = stats_.end() - 1;
    }
    it->calls += 1;
    it->ms += ms;
    it->flops += p.flops;
    it->ideal_bytes += p.ideal_bytes;
    it->moved_bytes += p.moved_bytes;
    it->dram_bytes += p.dram_bytes;
  }
  pending_.clear();
  cursor_ = 0;
}

std::vector<KernelStat> KernelProfiler::stats() {
  flush();
  std::vector<KernelStat> out = stats_;
  std::sort(out.begin(), out.end(),
            [](const KernelStat& a, const KernelStat& b) { return a.ms > b.ms; });
  return out;
}

double KernelProfiler::total_ms() {
  flush();
  double total = 0.0;
  for (const KernelStat& s : stats_) total += s.ms;
  return total;
}

void KernelProfiler::reset() {
  pending_.clear();
  cursor_ = 0;
  stats_.clear();
  in_region_ = false;
}

void profile_begin(const char* name, double flops, double ideal_bytes, double moved_bytes,
                   double dram_bytes) {
  KernelProfiler::get().begin(name, flops, ideal_bytes, moved_bytes, dram_bytes);
}

double l2_cache_bytes() {
  static const double bytes = [] {
    int device = 0;
    LLM_CUDA_CHECK(cudaGetDevice(&device));
    int l2 = 0;
    LLM_CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device));
    return static_cast<double>(l2);
  }();
  return bytes;
}

void profile_end() { KernelProfiler::get().end(); }

}  // namespace llm
