#pragma once

#include <ATen/ATen.h>

namespace torch { namespace autograd {

struct TORCH_API SavedVariableHooks {
  virtual void call_pack_hook(at::Tensor &tensor) = 0;
  virtual at::Tensor call_unpack_hook() = 0;
  virtual void reset_data() = 0;
  virtual ~SavedVariableHooks() = default;
};

}}
