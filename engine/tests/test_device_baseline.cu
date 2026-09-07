// M0 done-signal #2: the cudaMemGetInfo baseline is measurable, and the device
// we build for is the device we run on.
#include <cstdio>
#include <cstdlib>

#include "llm/device_info.h"

int main() {
  const llm::DeviceBaseline b = llm::query_device_baseline(0);
  std::printf("%s", llm::format_baseline(b).c_str());
  std::printf("baseline: %s\n", llm::format_baseline_kv(b).c_str());

  int rc = EXIT_SUCCESS;

  // The build targets LLM_CUDA_ARCH; a mismatch means the binary is running via
  // JIT (or not at all), which would silently skew every later benchmark.
  const int built_for = LLM_BUILD_SM;
  const int actual = b.cc_major * 10 + b.cc_minor;
  if (built_for != actual) {
    std::fprintf(stderr, "FAIL: built for sm_%d but running on sm_%d -- set -DLLM_CUDA_ARCH=%d\n",
                 built_for, actual, actual);
    rc = EXIT_FAILURE;
  }

  if (b.total_bytes == 0 || b.free_after_context == 0) {
    std::fprintf(stderr, "FAIL: cudaMemGetInfo returned nothing usable\n");
    rc = EXIT_FAILURE;
  }

  // Sanity bound, not a tuning knob: if the context alone ate more than half the
  // card, the VRAM budget in PLAN.md is built on sand.
  if (b.reserved_bytes() > b.total_bytes / 2) {
    std::fprintf(stderr, "FAIL: %.1f MiB of %.1f MiB already reserved before we allocate\n",
                 b.reserved_bytes() / 1048576.0, b.total_bytes / 1048576.0);
    rc = EXIT_FAILURE;
  }

  if (rc == EXIT_SUCCESS) std::printf("PASS: device baseline recorded\n");
  return rc;
}
