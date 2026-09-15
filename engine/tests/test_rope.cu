// Rotary embedding vs the torch reference, for both q and the narrower k.
#include <cstdio>
#include <cstdlib>

#include "llm/kernels.h"
#include "ref_fixture.h"

int main(int argc, char** argv) {
  using namespace llm::testing;
  const llm::ModelFile fixture = llm::ModelFile::load(fixture_path(argc, argv, "ref_rope.llmbin"));

  const llm::TensorView& q_view = fixture.require("q");
  const llm::TensorView& k_view = fixture.require("k");
  const int n_tokens = static_cast<int>(q_view.dim(0));
  const int n_heads = static_cast<int>(q_view.dim(1));
  const int head_dim = static_cast<int>(q_view.dim(2));
  const int n_kv_heads = static_cast<int>(k_view.dim(1));
  const float theta = fixture.config().rope_theta;

  const std::vector<std::int32_t> positions = load_int32(fixture, "positions");
  const std::vector<float> q_expected = load_fp32(fixture, "q_expected");
  const std::vector<float> k_expected = load_fp32(fixture, "k_expected");

  llm::DeviceBuffer d_q = upload_fp32(fixture, "q");
  llm::DeviceBuffer d_k = upload_fp32(fixture, "k");
  llm::DeviceBuffer d_pos = llm::make_device_buffer(positions);

  llm::launch_rope(d_q.as<float>(), d_k.as<float>(), d_pos.as<std::int32_t>(), n_tokens, n_heads,
                   n_kv_heads, head_dim, theta);
  LLM_CUDA_CHECK_LAUNCH();

  bool ok = check_close("rope(q)", download_fp32(d_q, q_expected.size()), q_expected, 1e-5f, 1e-5f);
  ok &= check_close("rope(k)", download_fp32(d_k, k_expected.size()), k_expected, 1e-5f, 1e-5f);
  if (!ok) return EXIT_FAILURE;

  std::printf("PASS: rope matches reference (%d tokens, %d q heads / %d kv heads, dim %d)\n",
              n_tokens, n_heads, n_kv_heads, head_dim);
  return EXIT_SUCCESS;
}
