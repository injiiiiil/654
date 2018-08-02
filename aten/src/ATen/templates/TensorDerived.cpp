// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

// ${generated_comment}

#include "ATen/${Tensor}.h"
#include "ATen/Scalar.h"
#include "ATen/Storage.h"
#include "ATen/core/Half.h"

$extra_cuda_headers

namespace at {

${Tensor}::${Tensor}(${THTensor} * tensor)
: TensorImpl(Backend::${Backend}, ScalarType::${ScalarName}, tensor, /* is variable */ false)
{}

}
