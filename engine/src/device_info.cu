#include "llm/device_info.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <sstream>

#include "llm/cuda_check.h"

namespace llm {
namespace {

// Touching device memory is what actually forces the driver to build the
// context; cudaSetDevice alone can be lazy.
void force_context_init() {
  void* p = nullptr;
  LLM_CUDA_CHECK(cudaMalloc(&p, 1));
  LLM_CUDA_CHECK(cudaFree(p));
  LLM_CUDA_CHECK(cudaDeviceSynchronize());
}

double to_mib(std::size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

}  // namespace

DeviceBaseline query_device_baseline(int device) {
  int count = 0;
  LLM_CUDA_CHECK(cudaGetDeviceCount(&count));
  if (count <= 0) {
    std::fprintf(stderr, "[llm] no CUDA device found\n");
    std::exit(EXIT_FAILURE);
  }

  DeviceBaseline b;
  b.device = device;
  LLM_CUDA_CHECK(cudaSetDevice(device));

  cudaDeviceProp prop{};
  LLM_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  b.name = prop.name;
  b.cc_major = prop.major;
  b.cc_minor = prop.minor;
  b.sm_count = prop.multiProcessorCount;
  b.shared_mem_per_block = prop.sharedMemPerBlock;
  b.shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
  b.regs_per_block = prop.regsPerBlock;
  b.warp_size = prop.warpSize;
  b.l2_cache_bytes = prop.l2CacheSize;
  b.memory_bus_width_bits = prop.memoryBusWidth;
  b.memory_clock_khz = prop.memoryClockRate;

  // cudaMemGetInfo itself requires a context, so "before" here means "before we
  // allocate anything of our own" -- see the note on context_overhead_bytes().
  std::size_t free_bytes = 0, total_bytes = 0;
  LLM_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  b.free_before_context = free_bytes;
  b.total_bytes = total_bytes;

  force_context_init();

  LLM_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  b.free_after_context = free_bytes;
  b.total_bytes = total_bytes;
  return b;
}

std::string format_baseline(const DeviceBaseline& b) {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(1);
  os << "device " << b.device << ": " << b.name << " (sm_" << b.cc_major << b.cc_minor << ")\n"
     << "  SMs                      : " << b.sm_count << "\n"
     << "  warp size                : " << b.warp_size << "\n"
     << "  shared mem / block       : " << b.shared_mem_per_block / 1024 << " KiB\n"
     << "  shared mem / SM          : " << b.shared_mem_per_sm / 1024 << " KiB\n"
     << "  32-bit regs / block      : " << b.regs_per_block << "\n"
     << "  L2 cache                 : " << b.l2_cache_bytes / 1024 << " KiB\n"
     << "  memory bus               : " << b.memory_bus_width_bits << " bit @ "
     << b.memory_clock_khz / 1000 << " MHz\n"
     << "  theoretical bandwidth    : " << b.theoretical_bandwidth_gbps() << " GB/s\n"
     << "  VRAM total               : " << to_mib(b.total_bytes) << " MiB\n"
     << "  VRAM free (pre-alloc)    : " << to_mib(b.free_before_context) << " MiB\n"
     << "  VRAM free (post-context) : " << to_mib(b.free_after_context) << " MiB\n"
     << "  context overhead delta   : " << to_mib(b.context_overhead_bytes()) << " MiB\n"
     << "  total reserved (unusable): " << to_mib(b.reserved_bytes()) << " MiB\n"
     << "  => allocatable budget    : " << to_mib(b.free_after_context) << " MiB\n";
  return os.str();
}

std::string format_baseline_kv(const DeviceBaseline& b) {
  std::ostringstream os;
  os << "name=\"" << b.name << "\""
     << " sm=" << b.cc_major << b.cc_minor << " sms=" << b.sm_count
     << " total_mib=" << b.total_bytes / (1024 * 1024)
     << " free_pre_mib=" << b.free_before_context / (1024 * 1024)
     << " free_post_mib=" << b.free_after_context / (1024 * 1024)
     << " ctx_overhead_mib=" << b.context_overhead_bytes() / (1024 * 1024)
     << " reserved_mib=" << b.reserved_bytes() / (1024 * 1024);
  return os.str();
}

}  // namespace llm
