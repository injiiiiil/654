#pragma once

/*
 * This op is only for testing the c10 dispatcher and might not support all
 * parameter combinations or backends the corresponding caffe2 op supports.
 * Please ignore this.
 * TODO Remove this comment once this is more final
 */

#include "caffe2/core/tensor.h"
#include "caffe2/utils/Array.h"

namespace caffe2 {
namespace ops {

struct BatchMatmul final {
  struct State final {
    std::shared_ptr<Tensor> scratch;
  };

  static constexpr const char* name = "batch_matmul";

  using Signature = void(
      const Tensor& A,
      const Tensor& B,
      Tensor* output,
      int trans_a,
      int trans_b,
      int broadcast,
      int use_scratch,
      State* state,
      BaseContext* context);

  static constexpr c10::guts::array<const char*, 9> parameter_names = {
      {"A",
       "B",
       "output",
       "trans_a",
       "trans_b",
       "broadcast",
       "use_scratch",
       "state",
       "context"}};
};

} // namespace ops
} // namespace caffe2
