#pragma once

#include <cstddef>
#include <string>

namespace llm {

// Snapshot of the target device plus the empirically measured cost of simply
// having a CUDA context alive. PLAN.md budgets 200-400MB for this; M0 exists to
// replace that guess with a number from this machine.
struct DeviceBaseline {
  int device = 0;
  std::string name;
  int cc_major = 0;
  int cc_minor = 0;
  int sm_count = 0;
  std::size_t shared_mem_per_block = 0;
  std::size_t shared_mem_per_sm = 0;
  int regs_per_block = 0;
  int warp_size = 0;
  int l2_cache_bytes = 0;
  int memory_bus_width_bits = 0;
  int memory_clock_khz = 0;

  std::size_t total_bytes = 0;         // cudaMemGetInfo total
  std::size_t free_before_context = 0; // free VRAM before our context exists
  std::size_t free_after_context = 0;  // free VRAM once the context is created

  // total - free_after_context: context overhead *plus* whatever else on the
  // system holds VRAM (desktop compositor, other processes). Treat as the
  // ceiling on what we can actually allocate, not as a pure driver cost.
  std::size_t reserved_bytes() const { return total_bytes - free_after_context; }

  // Delta attributable to creating our own context, assuming nothing else
  // allocated concurrently.
  std::size_t context_overhead_bytes() const {
    return free_before_context > free_after_context ? free_before_context - free_after_context : 0;
  }

  double theoretical_bandwidth_gbps() const {
    // DDR: 2 transfers per clock. kHz -> GB/s.
    return 2.0 * memory_clock_khz * 1e3 * (memory_bus_width_bits / 8.0) / 1e9;
  }
};

// Queries device 0 (or `device`), forcing context creation in between the two
// cudaMemGetInfo calls. Exits the process on any CUDA error.
DeviceBaseline query_device_baseline(int device = 0);

// Human-readable multi-line report.
std::string format_baseline(const DeviceBaseline& b);

// Single machine-readable line: key=value pairs, for logging into docs/.
std::string format_baseline_kv(const DeviceBaseline& b);

}  // namespace llm
