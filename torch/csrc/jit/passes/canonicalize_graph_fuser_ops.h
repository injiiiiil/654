#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

TORCH_API void CanonicalizeOps(const std::shared_ptr<Graph>& graph);

}
