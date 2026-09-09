// Done signal: the C++ loader reads a file written by
// tools/convert/make_test_fixture.py, and the embedding row for TOKEN_ID
// byte-matches what that Python script wrote independently to
// expected_row.bin. TOKEN_ID here must match TOKEN_ID in make_test_fixture.py.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "llm/loader.h"

namespace {

constexpr int kTokenId = 5;

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "FAIL: cannot open %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <tiny_model.llmbin> <expected_row.bin>\n", argv[0]);
    return EXIT_FAILURE;
  }
  const std::string model_path = argv[1];
  const std::string expected_row_path = argv[2];

  const llm::ModelFile model = llm::ModelFile::load(model_path);
  int rc = EXIT_SUCCESS;

  const llm::ModelConfig& cfg = model.config();
  if (cfg.vocab_size != 17 || cfg.hidden_size != 8) {
    std::fprintf(stderr, "FAIL: config mismatch (vocab_size=%u hidden_size=%u)\n",
                 cfg.vocab_size, cfg.hidden_size);
    rc = EXIT_FAILURE;
  }

  const llm::TensorView& emb = model.require("tok_embeddings.weight");
  if (emb.dtype != llm::DType::kFP16 || emb.ndim != 2 || emb.shape[0] != 17 || emb.shape[1] != 8) {
    std::fprintf(stderr, "FAIL: tok_embeddings.weight has unexpected shape/dtype\n");
    rc = EXIT_FAILURE;
  }

  const std::vector<std::uint8_t> expected = read_file(expected_row_path);
  const std::size_t row_bytes = emb.row_stride_elems() * llm::dtype_size_bytes(emb.dtype);
  if (row_bytes != expected.size()) {
    std::fprintf(stderr, "FAIL: row size %zu != expected file size %zu\n", row_bytes, expected.size());
    rc = EXIT_FAILURE;
  } else {
    const std::uint8_t* actual_row = emb.row(kTokenId);
    if (std::memcmp(actual_row, expected.data(), row_bytes) != 0) {
      std::fprintf(stderr, "FAIL: embedding row for token %d does not byte-match Python source\n", kTokenId);
      rc = EXIT_FAILURE;
    }
  }

  if (model.find("no_such_tensor") != nullptr) {
    std::fprintf(stderr, "FAIL: find() returned non-null for a missing tensor\n");
    rc = EXIT_FAILURE;
  }

  if (rc == EXIT_SUCCESS) std::printf("PASS: loader reads .llmbin, embedding row byte-matches Python source\n");
  return rc;
}
