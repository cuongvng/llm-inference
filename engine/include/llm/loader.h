#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "llm/config.h"
#include "llm/tensor.h"

namespace llm {

class ModelFile {
 public:
  static ModelFile load(const std::string& path);

  const ModelConfig& config() const { return config_; }

  const TensorView* find(const std::string& name) const;

  const TensorView& require(const std::string& name) const;

  std::size_t tensor_count() const { return tensors_.size(); }

 private:
  ModelConfig config_;
  std::vector<std::uint8_t> storage_;  // owns all tensor bytes
  std::unordered_map<std::string, TensorView> tensors_;
};

}  // namespace llm
