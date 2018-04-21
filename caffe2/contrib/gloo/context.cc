#include "context.h"

#include "caffe2/core/typeid.h"

#include <gloo/types.h>

namespace caffe2 {

CAFFE_KNOWN_TYPE(::gloo::float16);
CAFFE_KNOWN_TYPE(std::shared_ptr<::gloo::Context>);

} // namespace caffe2
