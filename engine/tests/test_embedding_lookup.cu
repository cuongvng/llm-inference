// Embedding gather vs the torch reference.
#include <cstdio>
#include <cstdlib>

#include "llm/kernels.h"
#include "ref_fixture.h"

int main(int argc, char** argv) {
  using namespace llm::testing;
  const llm::ModelFile fixture = llm::ModelFile::load(fixture_path(argc, argv, "ref_embedding.llmbin"));

  const llm::TensorView& table_view = fixture.require("table");
  const int hidden = static_cast<int>(table_view.dim(1));

  const std::vector<std::int32_t> ids = load_int32(fixture, "ids");
  const int n_tokens = static_cast<int>(ids.size());
  const std::vector<float> expected = load_fp32(fixture, "expected");

  llm::DeviceBuffer d_table = upload_half(fixture, "table");
  llm::DeviceBuffer d_ids = llm::make_device_buffer(ids);
  llm::DeviceBuffer d_out(static_cast<std::size_t>(n_tokens) * hidden * sizeof(float));
  d_out.zero();

  llm::launch_embedding_lookup(d_table.as<__half>(), d_ids.as<std::int32_t>(), d_out.as<float>(),
                               n_tokens, hidden);
  LLM_CUDA_CHECK_LAUNCH();

  const std::vector<float> got = download_fp32(d_out, expected.size());
  // A gather copies values; there is no arithmetic to lose precision in, so the
  // only tolerance needed is for the FP16 -> FP32 widening being exact.
  if (!check_close("embedding", got, expected, 0.0f, 0.0f)) return EXIT_FAILURE;

  std::printf("PASS: embedding_lookup matches reference (%d tokens x %d)\n", n_tokens, hidden);
  return EXIT_SUCCESS;
}
