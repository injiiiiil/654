#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

void CheckInplace(std::shared_ptr<Graph>& graph);

}}
