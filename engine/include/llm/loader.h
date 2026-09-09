#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "llm/config.h"
#include "llm/tensor.h"

namespace llm {

// Reads a .llmbin file (docs/binary_format.md) fully into host memory and
// exposes its tensors by name. Written offline by tools/convert/binformat.py.
//
// Loading is eager and whole-file: v1 targets single-digit-GB models on a
// dev machine, and mmap-vs-read tradeoffs aren't worth the complexity until
// profiling says otherwise.
class ModelFile {
 public:
  // Reads and parses `path`. Exits the process (via LLM_CUDA_CHECK-style
  // fail-fast, matching the rest of the codebase) on any format error:
  // missing file, bad magic, unsupported version, truncated data.
  static ModelFile load(const std::string& path);

  const ModelConfig& config() const { return config_; }

  // Returns nullptr if no tensor with that name exists.
  const TensorView* find(const std::string& name) const;

  // Like find(), but exits the process if the tensor is missing -- for
  // required tensors where a missing entry means the file doesn't match the
  // architecture the engine expects.
  const TensorView& require(const std::string& name) const;

  std::size_t tensor_count() const { return tensors_.size(); }

 private:
  ModelConfig config_;
  std::vector<std::uint8_t> storage_;  // owns all tensor bytes
  std::unordered_map<std::string, TensorView> tensors_;
};

}  // namespace llm
