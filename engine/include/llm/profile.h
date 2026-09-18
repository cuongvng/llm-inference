#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llm {

// Per-kernel device timing for the forward pass.
//
// A region is a named span of one or more launches on the default stream,
// bracketed by CUDA events. Events are recorded *in* the stream, so what they
// measure is device execution time -- not the host-side launch call, which
// returns long before the kernel runs. Reading them back requires a
// synchronize, so that happens once at flush() rather than per region.
//
// Alongside the time, each region carries an analytic cost model supplied by
// the caller, which is what turns a millisecond number into a diagnosis:
//
//   flops        floating-point operations the kernel actually executes
//   ideal_bytes  compulsory DRAM traffic: every distinct input read once,
//                every output written once. What a perfect implementation of
//                this operator would move.
//   moved_bytes  traffic this implementation requests, given its blocking. The
//                gap between the two is re-read -- the tiling's reuse failure.
//   dram_bytes   the part of that request the caches cannot absorb: an operand
//                that fits in L2 is counted once however often it is re-read,
//                one larger than L2 is counted with its re-reads. Without
//                hardware counters this is the closest estimate available of
//                what actually crosses the memory bus.
//
// Disabled by default and compiled unconditionally: a disabled region is a
// branch on a bool, so the un-profiled path costs nothing worth measuring.
struct KernelStat {
  std::string name;
  std::uint64_t calls = 0;
  double ms = 0.0;
  double flops = 0.0;
  double ideal_bytes = 0.0;
  double moved_bytes = 0.0;
  double dram_bytes = 0.0;
};

class KernelProfiler {
 public:
  static KernelProfiler& get();

  void enable(bool on) { enabled_ = on; }
  bool enabled() const { return enabled_; }

  // No-ops unless enabled. Regions must not nest.
  void begin(const char* name, double flops, double ideal_bytes, double moved_bytes,
             double dram_bytes);
  void end();

  // Drains pending event pairs into the accumulated stats. Synchronizes.
  void flush();

  // flush()es, then returns the accumulated regions, slowest first.
  std::vector<KernelStat> stats();
  double total_ms();
  void reset();

 private:
  struct Pending {
    const char* name;
    double flops;
    double ideal_bytes;
    double moved_bytes;
    double dram_bytes;
    std::size_t start;
    std::size_t stop;
  };

  // Grabs the next event from the pool, creating it on first use. The pool is
  // never shrunk: a steady-state forward pass reuses exactly the same events.
  std::size_t acquire_event();

  bool enabled_ = false;
  bool in_region_ = false;
  std::vector<void*> events_;  // cudaEvent_t, type-erased to keep this header CUDA-free
  std::size_t cursor_ = 0;
  std::vector<Pending> pending_;
  std::vector<KernelStat> stats_;
};

// Free functions so call sites read as two plain statements around the launches
// they time.
void profile_begin(const char* name, double flops, double ideal_bytes, double moved_bytes,
                   double dram_bytes);

// Bytes of L2 on the current device: the threshold that decides whether a
// re-read operand is absorbed by cache or paid for at DRAM speed.
double l2_cache_bytes();
void profile_end();

}  // namespace llm
