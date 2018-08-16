#if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
#include "torch/csrc/jit/fusers/cuda/tensor_desc.h"

#include "torch/csrc/jit/assertions.h"

#include <vector>

namespace torch { namespace jit { namespace cudafuser {

std::vector<bool> TensorDesc::findContiguous(
  const at::IntList& sizes
, const at::IntList& strides) {
  JIT_ASSERT(sizes.size() == strides.size());
  std::vector<bool> cont(sizes.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    int64_t expected_stride = (i + 1 < sizes.size()) ? sizes[i+1]*strides[i+1] : 1;
    cont[i] = strides[i] == expected_stride;
  }
  return cont;
}

} // namespace cudafuser
} // namespace jit
} // namespace torch

#endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
