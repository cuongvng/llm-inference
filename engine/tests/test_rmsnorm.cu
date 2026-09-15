// RMSNorm vs the torch reference.
#include <cstdio>
#include <cstdlib>

#include "llm/kernels.h"
#include "ref_fixture.h"

int main(int argc, char** argv) {
  using namespace llm::testing;
  const llm::ModelFile fixture = llm::ModelFile::load(fixture_path(argc, argv, "ref_rmsnorm.llmbin"));

  const llm::TensorView& input_view = fixture.require("input");
  const int n_rows = static_cast<int>(input_view.dim(0));
  const int hidden = static_cast<int>(input_view.dim(1));
  const float eps = fixture.config().rms_norm_eps;

  const std::vector<float> expected = load_fp32(fixture, "expected");

  llm::DeviceBuffer d_x = upload_fp32(fixture, "input");
  llm::DeviceBuffer d_w = upload_half(fixture, "weight");
  llm::DeviceBuffer d_out(expected.size() * sizeof(float));
  d_out.zero();

  llm::launch_rmsnorm(d_x.as<float>(), d_w.as<__half>(), d_out.as<float>(), n_rows, hidden, eps);
  LLM_CUDA_CHECK_LAUNCH();

  const std::vector<float> got = download_fp32(d_out, expected.size());
  if (!check_close("rmsnorm", got, expected, 1e-5f, 1e-5f)) return EXIT_FAILURE;

  // In-place must work too: the forward pass has no spare buffer to spend on a
  // norm that can only write somewhere else.
  llm::DeviceBuffer d_inplace = upload_fp32(fixture, "input");
  llm::launch_rmsnorm(d_inplace.as<float>(), d_w.as<__half>(), d_inplace.as<float>(), n_rows,
                      hidden, eps);
  LLM_CUDA_CHECK_LAUNCH();
  const std::vector<float> got_inplace = download_fp32(d_inplace, expected.size());
  if (!check_close("rmsnorm(in-place)", got_inplace, expected, 1e-5f, 1e-5f)) return EXIT_FAILURE;

  std::printf("PASS: rmsnorm matches reference (%d rows x %d)\n", n_rows, hidden);
  return EXIT_SUCCESS;
}
