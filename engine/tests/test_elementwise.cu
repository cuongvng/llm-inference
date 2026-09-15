// The three small kernels that don't warrant a file each: SwiGLU, the residual
// add, and the greedy-sampling argmax.
#include <cstdio>
#include <cstdlib>

#include "llm/kernels.h"
#include "ref_fixture.h"

int main(int argc, char** argv) {
  using namespace llm::testing;
  const llm::ModelFile fixture = llm::ModelFile::load(fixture_path(argc, argv, "ref_elementwise.llmbin"));

  bool ok = true;

  {
    const std::vector<float> expected = load_fp32(fixture, "swiglu_expected");
    llm::DeviceBuffer d_gate = upload_fp32(fixture, "gate");
    llm::DeviceBuffer d_up = upload_fp32(fixture, "up");
    llm::DeviceBuffer d_out(expected.size() * sizeof(float));
    d_out.zero();

    llm::launch_swiglu(d_gate.as<float>(), d_up.as<float>(), d_out.as<float>(), expected.size());
    LLM_CUDA_CHECK_LAUNCH();
    ok &= check_close("swiglu", download_fp32(d_out, expected.size()), expected, 1e-6f, 1e-5f);
  }

  {
    const std::vector<float> expected = load_fp32(fixture, "res_expected");
    llm::DeviceBuffer d_x = upload_fp32(fixture, "res_x");
    llm::DeviceBuffer d_y = upload_fp32(fixture, "res_y");

    llm::launch_residual_add(d_x.as<float>(), d_y.as<float>(), expected.size());
    LLM_CUDA_CHECK_LAUNCH();
    ok &= check_close("residual_add", download_fp32(d_x, expected.size()), expected, 0.0f, 0.0f);
  }

  {
    const std::vector<float> logits = load_fp32(fixture, "logits");
    const std::vector<std::int32_t> expected = load_int32(fixture, "argmax_expected");
    llm::DeviceBuffer d_logits = upload_fp32(fixture, "logits");
    llm::DeviceBuffer d_index(sizeof(std::int32_t));
    d_index.zero();

    llm::launch_argmax(d_logits.as<float>(), d_index.as<std::int32_t>(),
                       static_cast<int>(logits.size()));
    LLM_CUDA_CHECK_LAUNCH();
    std::int32_t got = -1;
    d_index.download(&got, sizeof(got));
    ok &= check_int_equal("argmax", {got}, expected);
  }

  if (!ok) return EXIT_FAILURE;
  std::printf("PASS: swiglu, residual_add and argmax match reference\n");
  return EXIT_SUCCESS;
}
