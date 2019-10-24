#include <torch/nn/modules/padding.h>

#include <torch/expanding_array.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

template <size_t D, typename Derived>
ReflectionPadImpl<D, Derived>::ReflectionPadImpl(const ReflectionPadOptions<D>& options_)
    : options(options_) {}

template <size_t D, typename Derived>
void ReflectionPadImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
Tensor ReflectionPadImpl<D, Derived>::forward(const Tensor& input) {
  return F::pad(input, PadOptions(at::ArrayRef<int64_t>(options.padding()).vec()).mode(torch::kReflect));
}

template <size_t D, typename Derived>
void ReflectionPadImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ReflectionPad" << D << "d"
         << "(padding=" << options.padding() << ")";
}

template class ReflectionPadImpl<1, ReflectionPad1dImpl>;
template class ReflectionPadImpl<2, ReflectionPad2dImpl>;

// ============================================================================

template <size_t D, typename Derived>
ReplicationPadImpl<D, Derived>::ReplicationPadImpl(const ReplicationPadOptions<D>& options_)
    : options(options_) {}

template <size_t D, typename Derived>
void ReplicationPadImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
Tensor ReplicationPadImpl<D, Derived>::forward(const Tensor& input) {
  return F::pad(input, PadOptions(at::ArrayRef<int64_t>(options.padding()).vec()).mode(torch::kReplicate));
}

template <size_t D, typename Derived>
void ReplicationPadImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ReplicationPad" << D << "d"
         << "(padding=" << options.padding() << ")";
}

template class ReplicationPadImpl<1, ReplicationPad1dImpl>;
template class ReplicationPadImpl<2, ReplicationPad2dImpl>;
template class ReplicationPadImpl<3, ReplicationPad3dImpl>;

// ============================================================================

ZeroPad2dImpl::ZeroPad2dImpl(const ZeroPad2dOptions& options_)
    : options(options_) {}

void ZeroPad2dImpl::reset() {}

void ZeroPad2dImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ZeroPad2d"
         << "(padding=" << options.padding() << ")";
}

Tensor ZeroPad2dImpl::forward(const Tensor& input) {
  return F::pad(input, PadOptions(at::ArrayRef<int64_t>(options.padding()).vec()).mode(torch::kConstant).value(0));
}

} // namespace nn
} // namespace torch
