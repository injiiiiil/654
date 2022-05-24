#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/core/jit_type.h>
#include <c10/util/DimVector.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/zeros.h>
#endif

#include <cstdint>
#include <utility>

namespace torch { namespace autograd {

/**
 * Records TensorOptions, shape of the tensor, whether or not the Python dispatch key is set (tensor subclass),
 * and, where applicable, the stream the corresponding operation took place on.
 *
 * If is_valid() is false, then the corresponding input is not used and may be
 * an undefined tensor.
 */
struct InputMetadata {
  InputMetadata() = default;

  InputMetadata(
      const at::TensorOptions options,
      at::IntArrayRef shape,
      c10::optional<const at::Tensor> nested_shape,
      bool is_tensor_subclass)
      : options_{options},
        shape_{shape},
        nested_shape_(std::move(nested_shape)),
        is_tensor_subclass_{is_tensor_subclass} {
    auto device_ = options.device();
    stream_ = c10::impl::getDeviceGuardImpl(device_.type())->getStream(device_);
  }

  InputMetadata(const at::Tensor& t)
      : InputMetadata(
            t.options(),
            compute_shape(t),
            compute_nested_shape_from_tensor(t),
            t.unsafeGetTensorImpl()->is_python_dispatch()) {}

  const at::TensorOptions options() const {
    return options_;
  }

  // This Constructor is for prior behavior probably should not use this and fix external calling cites
  InputMetadata(
      const at::TensorOptions options,
      at::IntArrayRef shape,
      bool is_tensor_subclass)
      : options_{options},
        shape_{shape},
        nested_shape_(c10::optional<const at::Tensor>{}),
        is_tensor_subclass_{is_tensor_subclass} {
    auto device_ = options.device();
    stream_ = c10::impl::getDeviceGuardImpl(device_.type())->getStream(device_);
  }

  at::IntArrayRef shape() const {
    return shape_;
  }

  bool is_nested_tensor() const {
    return nested_shape_.has_value();
  }

  at::Tensor nested_shape() const {
    TORCH_CHECK(is_nested_tensor(), "The Tensor must be Nested to call nested_shape")
    return *nested_shape_;
  }

  caffe2::TypeMeta dtype() const {
    return options_.dtype();
  }

  at::Device device() const {
    return options_.device();
  }

  at::Layout layout() const {
    return options_.layout();
  }

  c10::Stream stream() const {
    return stream_;
  }

  bool is_tensor_subclass() const {
    return is_tensor_subclass_;
  }

  at::Tensor zeros_like() const {
    return at::zeros(shape_, options_);
  }

  at::IntArrayRef compute_shape(const at::Tensor& t){
    if(t.is_nested()){
      return at::IntArrayRef{};
    }
    return t.sizes();
  }

  c10::optional<const at::Tensor> compute_nested_shape_from_tensor(const at::Tensor& t){
    if(t.is_nested()){
      auto nested_impl = at::native::get_nested_tensor_impl(t);
      return nested_impl->get_nested_size_tensor();
    }
    return c10::optional<const at::Tensor> {};
  }

private:
  const at::TensorOptions options_;
  at::DimVector shape_;
  c10::optional<const at::Tensor> nested_shape_;
  c10::Stream stream_ = c10::Stream(c10::Stream::Default::DEFAULT, device());
  bool is_tensor_subclass_ = false;
};

}} // torch::autograd
