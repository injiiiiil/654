#include "types.h"

namespace torch {
namespace jit {
namespace fuser {

enum ScalarType {
  kScalarUninitialized,
  kScalarHandle,
  kScalarInt32,
  kScalarFloat32,
  kScalarNull
};

Dtype Dtype::scalar_type() const {
  switch (static_cast<ScalarType>(scalar_type_)) {
    case kScalarUninitialized:
      return kUninitialized;
    case kScalarHandle:
      return kHandle;
    case kScalarInt32:
      return kInt32;
    case kScalarFloat32:
      return kFloat32;
      //TODO switch to PyT LOG
      //default:
      //LOG(FATAL) << "invalid scalar type: " << scalar_type_;
  }
}

Dtype kInt32(kScalarInt32, 1);
Dtype kFloat32(kScalarFloat32, 1);
Dtype kHandle(kScalarHandle, 1);
Dtype kUninitialized(kScalarUninitialized, 1);
Dtype kNull(kScalarNull, 1);

std::ostream& operator<<(std::ostream& stream, const Dtype& dtype) {
  switch (static_cast<ScalarType>(dtype.scalar_type_)) {
    case kScalarUninitialized:
      stream << "uninitialized";
      break;
    case kScalarHandle:
      stream << "handle";
      break;
    case kScalarInt32:
      stream << "int32";
      break;
    case kScalarFloat32:
      stream << "float32";
      break;
      //TODO switch to PyT LOG
      //default:
      //LOG(FATAL) << "invalid scalar type: " << dtype.scalar_type_;
  }
  if (dtype.lanes() > 1) {
    stream << "x" << dtype.lanes();
    ;
  }
  return stream;
}

} // namespace fuser
} // namespace jit
} // namespace torch
