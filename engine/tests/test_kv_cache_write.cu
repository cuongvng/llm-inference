// The cache write is a copy, so the reference is the input itself and no torch
// fixture is involved.
//
// What the test is actually looking for is the offset arithmetic. A copy kernel
// that writes the right bytes to the wrong rows, or that spills past the rows it
// was given, still produces plausible-looking attention output a few layers
// later; here it is a bitwise mismatch. So every row carries its own absolute
// row index in its values, the cache starts full of a sentinel, and the rows
// that were never written must still hold that sentinel afterwards.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "llm/cuda_check.h"
#include "llm/device_buffer.h"
#include "llm/kernels.h"

namespace {

// Deliberately not a multiple of the block size, the warp size, or four: the
// grid-stride tail is where a hand-written copy goes wrong. A real run has
// kv_dim = n_kv_heads * head_dim, which is none of those awkward things.
constexpr int kKvDim = 258;
constexpr int kMaxSeq = 64;
constexpr float kSentinel = -7.5f;

// Small integers, exactly representable in fp32, so the comparison is bitwise.
// K and V differ in sign so a kernel that writes one into the other's slab fails.
float k_value(int row, int col) { return static_cast<float>(row * 1000 + col); }
float v_value(int row, int col) { return -k_value(row, col); }

}  // namespace

int main() {
  // Three writes: a prefill chunk, the single-row append a decode step makes,
  // and a jump to a later row that a kernel quietly assuming "append at the end"
  // would get wrong.
  const struct {
    int start_pos;
    int n_new;
  } writes[] = {{0, 5}, {5, 1}, {40, 3}};

  const std::size_t cache_elems = static_cast<std::size_t>(kMaxSeq) * kKvDim;
  std::vector<float> h_k_cache(cache_elems, kSentinel);
  std::vector<float> h_v_cache(cache_elems, kSentinel);

  llm::DeviceBuffer d_k_cache = llm::make_device_buffer(h_k_cache);
  llm::DeviceBuffer d_v_cache = llm::make_device_buffer(h_v_cache);

  for (const auto& w : writes) {
    const std::size_t n = static_cast<std::size_t>(w.n_new) * kKvDim;
    std::vector<float> h_new_k(n), h_new_v(n);
    for (int r = 0; r < w.n_new; ++r) {
      for (int c = 0; c < kKvDim; ++c) {
        const std::size_t i = static_cast<std::size_t>(r) * kKvDim + c;
        h_new_k[i] = k_value(w.start_pos + r, c);
        h_new_v[i] = v_value(w.start_pos + r, c);
        // The same values the kernel should leave in the cache.
        const std::size_t dst = static_cast<std::size_t>(w.start_pos + r) * kKvDim + c;
        h_k_cache[dst] = h_new_k[i];
        h_v_cache[dst] = h_new_v[i];
      }
    }

    llm::DeviceBuffer d_new_k = llm::make_device_buffer(h_new_k);
    llm::DeviceBuffer d_new_v = llm::make_device_buffer(h_new_v);

    llm::launch_kv_cache_write(d_new_k.as<float>(), d_new_v.as<float>(), d_k_cache.as<float>(),
                               d_v_cache.as<float>(), w.n_new, w.start_pos, kKvDim);
    LLM_CUDA_CHECK_LAUNCH();
  }

  std::vector<float> got_k(cache_elems), got_v(cache_elems);
  d_k_cache.download(got_k.data(), cache_elems * sizeof(float));
  d_v_cache.download(got_v.data(), cache_elems * sizeof(float));

  std::size_t bad = 0;
  for (std::size_t i = 0; i < cache_elems; ++i) {
    const bool k_ok = got_k[i] == h_k_cache[i];
    const bool v_ok = got_v[i] == h_v_cache[i];
    if (!k_ok || !v_ok) {
      if (bad < 5) {
        const std::size_t row = i / kKvDim, col = i % kKvDim;
        std::fprintf(stderr, "  row %zu col %zu: k got %g want %g, v got %g want %g\n", row, col,
                     got_k[i], h_k_cache[i], got_v[i], h_v_cache[i]);
      }
      ++bad;
    }
  }

  if (bad != 0) {
    std::fprintf(stderr, "FAIL kv_cache_write: %zu / %zu cache elements wrong\n", bad, cache_elems);
    return EXIT_FAILURE;
  }
  std::printf("PASS: kv_cache_write placed 9 rows correctly and left the other %d untouched\n",
              kMaxSeq - 9);
  return EXIT_SUCCESS;
}
