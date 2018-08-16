#if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
#pragma once

#include "ATen/ATen.h"

#include "torch/csrc/jit/type.h"

#include <vector>
#include <iostream>
#include <algorithm>

namespace torch { namespace jit { namespace cudafuser {

// type information needed by the compiler for input/outputs
// contiguity[i] is true if the dim i is contiguous with dim i + 1.
// contiguity.back() == true means strides.back() == 1.
struct TensorDesc {
  at::ScalarType scalar_type;
  std::vector<bool> contiguity;

  TensorDesc(const at::ScalarType& type, const std::vector<bool>& contiguity)
  : scalar_type(type), contiguity(contiguity) {
    nDim_ = std::count(contiguity.begin(), contiguity.end(), false) + (lastIsContiguous() ? 1 : 0);
  }

  TensorDesc(
    const at::ScalarType& type
  , const at::IntList& sizes
  , const at::IntList& strides)
  : TensorDesc(type, TensorDesc::findContiguous(sizes, strides)) {}
  
  TensorDesc(const at::Tensor& t)
  : TensorDesc(t.type().scalarType(), t.sizes(), t.strides()) {}
  
  TensorDesc(TensorTypePtr type)
  : TensorDesc(type->scalarType(), type->sizes(), type->strides()) {}

  // number of dimensions after contiguity compression
  size_t nDim() const {
    return nDim_;
  }

  // do we have inner stride == 1?
  bool lastIsContiguous() const {
    return contiguity.size() == 0 || contiguity.back();
  }

  static std::vector<bool> findContiguous(
    const at::IntList& sizes
  , const at::IntList& strides);

private:
  size_t nDim_;
};

inline std::ostream& operator<<(std::ostream& out, const TensorDesc& d) {
  out << d.scalar_type << "[";
  for (auto b : d.contiguity) out << b << ";";
  out << "]";
  return out;
}

} // namespace cudafuser
} // namespace jit
} // namespace torch

#endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
