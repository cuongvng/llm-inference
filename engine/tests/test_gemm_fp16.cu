// FP16-weight GEMM (the prefill-shape matmul) vs the torch reference.
#include <cstdio>
#include <cstdlib>

#include "llm/kernels.h"
#include "ref_fixture.h"

int main(int argc, char** argv) {
  using namespace llm::testing;
  const llm::ModelFile fixture = llm::ModelFile::load(fixture_path(argc, argv, "ref_gemm.llmbin"));

  const llm::TensorView& w_view = fixture.require("weight");
  const llm::TensorView& x_view = fixture.require("input");
  const int n_out = static_cast<int>(w_view.dim(0));
  const int k_in = static_cast<int>(w_view.dim(1));
  const int m_rows = static_cast<int>(x_view.dim(0));
  const std::vector<float> expected = load_fp32(fixture, "expected");

  llm::DeviceBuffer d_w = upload_half(fixture, "weight");
  llm::DeviceBuffer d_x = upload_fp32(fixture, "input");
  llm::DeviceBuffer d_y(expected.size() * sizeof(float));
  d_y.zero();

  llm::launch_gemm_fp16(d_w.as<__half>(), d_x.as<float>(), d_y.as<float>(), m_rows, n_out, k_in);
  LLM_CUDA_CHECK_LAUNCH();

  const std::vector<float> got = download_fp32(d_y, expected.size());
  if (!check_close("gemm", got, expected, 2e-3f, 2e-3f)) return EXIT_FAILURE;

  // M and K are both ragged against the tile size in the fixture, so the
  // partial-tile path is already covered above. This second call pins the
  // single-row case: prefill and decode must go through the same kernel and
  // agree, since decode is just M = 1.
  llm::DeviceBuffer d_y1(static_cast<std::size_t>(n_out) * sizeof(float));
  d_y1.zero();
  llm::launch_gemm_fp16(d_w.as<__half>(), d_x.as<float>(), d_y1.as<float>(), 1, n_out, k_in);
  LLM_CUDA_CHECK_LAUNCH();
  const std::vector<float> got_row0 = download_fp32(d_y1, n_out);
  const std::vector<float> expected_row0(expected.begin(), expected.begin() + n_out);
  if (!check_close("gemm(M=1)", got_row0, expected_row0, 2e-3f, 2e-3f)) return EXIT_FAILURE;

  std::printf("PASS: gemm_fp16 matches reference (%d x %d x %d)\n", m_rows, n_out, k_in);
  return EXIT_SUCCESS;
}
