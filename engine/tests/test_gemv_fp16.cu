// FP16-weight GEMV (the decode-shape matmul) vs the torch reference.
#include <cstdio>
#include <cstdlib>

#include "llm/kernels.h"
#include "ref_fixture.h"

int main(int argc, char** argv) {
  using namespace llm::testing;
  const llm::ModelFile fixture = llm::ModelFile::load(fixture_path(argc, argv, "ref_gemv.llmbin"));

  const llm::TensorView& w_view = fixture.require("weight");
  const int n_out = static_cast<int>(w_view.dim(0));
  const int k_in = static_cast<int>(w_view.dim(1));
  const std::vector<float> expected = load_fp32(fixture, "expected");

  llm::DeviceBuffer d_w = upload_half(fixture, "weight");
  llm::DeviceBuffer d_x = upload_fp32(fixture, "input");
  llm::DeviceBuffer d_y(expected.size() * sizeof(float));
  d_y.zero();

  llm::launch_gemv_fp16(d_w.as<__half>(), d_x.as<float>(), d_y.as<float>(), n_out, k_in);
  LLM_CUDA_CHECK_LAUNCH();

  const std::vector<float> got = download_fp32(d_y, expected.size());
  // A K-long FP32 accumulation in a different order than torch's: error grows
  // with sqrt(K), so the tolerance is loose in absolute terms but still far
  // tighter than any real bug in the reduction would land inside.
  if (!check_close("gemv", got, expected, 2e-3f, 2e-3f)) return EXIT_FAILURE;

  std::printf("PASS: gemv_fp16 matches reference (%d x %d)\n", n_out, k_in);
  return EXIT_SUCCESS;
}
