#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>

namespace at {

SparseTensorImpl::SparseTensorImpl(Type * type)
    : TensorImpl(type)
    , indices_(type->toDense().toScalarType(ScalarType::Long).tensor())
    , values_(type->toDense().tensor()) {
      AT_ASSERT(type->is_sparse() && !type->is_variable_or_undefined());
    }

const char * SparseTensorImpl::toString() const {
  // TODO: also give back type information
  return "SparseTensor";
}
IntList SparseTensorImpl::sizes() const {
  return size_;
}
IntList SparseTensorImpl::strides() const {
  AT_ERROR("sparse tensors do not have strides");
}
int64_t SparseTensorImpl::dim() const {
  return dimI_ + dimV_;
}
Scalar SparseTensorImpl::localScalar() {
  AT_ERROR("sparse tensors cannot be scalars");
}
void * SparseTensorImpl::unsafeGetTH(bool retain) {
  AT_ERROR("unsafeGetTH not supported for new style TensorImpl");
}
std::unique_ptr<Storage> SparseTensorImpl::storage() {
  AT_ERROR("sparse tensors do not have storage");
}

void SparseTensorImpl::set_indices_and_values(const Tensor& indices, const Tensor& values) {
  bool empty = values.dim() == 0;
  AT_CHECK(values.type().toSparse() == type(), "values type must match sparse tensor type");
  AT_CHECK(indices.type().scalarType() == kLong);
  AT_CHECK(indices.type().backend() == values.type().backend());
  if (!empty) {
    AT_CHECK(indices.dim() == 2, "indices must be nDim x nnz");
    AT_CHECK(indices.size(1) == values.size(0), "indices and values must have same nnz");
    AT_CHECK(indices.size(0) == dimI_, "indices has incorrect first dimension, expected ", dimI_, ", got ", indices.size(0));
    AT_CHECK(values.dim() == dimV_ + 1, "values has incorrect number of dimensions, expected ", dimV_ + 1, ", got ", values.dim());
  } else {
    AT_CHECK(indices.dim() == 0, "if values is empty, indices must be empty too");
  }
  indices_ = indices;
  values_ = values;
  nnz_ = empty ? 0 : values.size(0);
  coalesced_ = 0;
}


} // namespace at
