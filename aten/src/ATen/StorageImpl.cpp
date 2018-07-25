#include <ATen/StorageImpl.h>
#include <ATen/Context.h>

namespace at {

StorageImpl::StorageImpl(
    at::ScalarType scalar_type,
    ptrdiff_t size,
    at::DataPtr data_ptr,
    at::Allocator* allocator,
    char flag)
    : scalar_type(scalar_type),
      data_ptr(std::move(data_ptr)),
      size(size),
      refcount(1),
      weakcount(1), // from the strong reference
      flag(flag),
      allocator(allocator),
      finalizer(nullptr) {}

StorageImpl::StorageImpl(
    at::ScalarType scalar_type,
    ptrdiff_t size,
    at::Allocator* allocator,
    char flag)
    : StorageImpl(
          scalar_type,
          size,
          allocator->allocate(at::elementSize(scalar_type) * size),
          allocator,
          flag) {}

Type& StorageImpl::type() const {
  if (data_ptr.device().is_cuda())
    return globalContext().getType(at::Backend::CUDA, scalar_type);
  return globalContext().getType(at::Backend::CPU, scalar_type);
}

} // namespace at
