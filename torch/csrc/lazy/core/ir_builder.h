#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/ir.h>
#include <vector>

// This file is part of the backend interface. So, ops shouldn't be added or removed without due process
// The exception to this being the view ops which will be removed soon pending functionalization

namespace torch {
namespace lazy {


struct IrBuilder {
  virtual NodePtr MakeDeviceData(const std::shared_ptr<BackendData>& data) const = 0;
  virtual NodePtr MakeScalar(const at::Scalar& value, const at::ScalarType& type) const = 0;
  virtual NodePtr MakeExpand(const Value& input0, const std::vector<int64_t>& size, const bool& is_scalar_expand) const = 0;
  virtual NodePtr MakeView(const Value& input0, const std::vector<int64_t>& output_size) const = 0;
  virtual NodePtr MakeCast(const Value& input0, const at::ScalarType& dtype, const c10::optional<at::ScalarType>& stype = c10::nullopt) const = 0;
  virtual NodePtr MakeTensorList(const OpList& inputs) const = 0;
  virtual NodePtr MakeAsStridedViewUpdate(const Value& input0, const Value& input1, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, const int64_t& storage_offset) const = 0;
  virtual NodePtr MakeAsStrided(const Value& input0, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, const int64_t& storage_offset) const = 0;
  virtual NodePtr MakeDiagonalViewUpdate(const Value& input0, const Value& input1, const int64_t& offset, const int64_t& dim1, const int64_t& dim2) const = 0;
  virtual NodePtr MakeDiagonal(const Value& input0, const int64_t& offset, const int64_t& dim1, const int64_t& dim2) const = 0;
  virtual NodePtr MakeNarrowViewUpdate(const Value& input0, const Value& input1, const std::vector<int64_t>& base_indices) const = 0;
  virtual NodePtr MakeNarrow(const Value& input0, const std::vector<int64_t>& base_indices, const std::vector<int64_t>& sizes) const = 0;
  virtual NodePtr MakePermute(const Value& input0, const std::vector<int64_t>& dims) const = 0;
  virtual NodePtr MakeResize(const Value& input0, const std::vector<int64_t>& size) const = 0;
  virtual NodePtr MakeSelectViewUpdate(const Value& input0, const Value& input1, const int64_t& dim, const int64_t& start, const int64_t& end, const int64_t& stride) const = 0;
  virtual NodePtr MakeSelect(const Value& input0, const int64_t& dim, const int64_t& start, const int64_t& end, const int64_t& stride) const = 0;
  virtual NodePtr MakeSqueeze(const Value& input0, const int& dim) const = 0;
  virtual NodePtr MakeUnsqueeze(const Value& input0, const int& dim) const = 0;
};

static inline NodePtr MakeDeviceData(const std::shared_ptr<BackendData>& data) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeDeviceData(data);
}
static inline NodePtr MakeScalar(const at::Scalar& value, const at::ScalarType& type) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeScalar(value, type);
}
static inline NodePtr MakeExpand(const Value& input0, const std::vector<int64_t>& size, const bool& is_scalar_expand) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeExpand(input0, size, is_scalar_expand);
}
static inline NodePtr MakeView(const Value& input0, const std::vector<int64_t>& output_size) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeView(input0, output_size);
}
static inline NodePtr MakeCast(const Value& input0, const at::ScalarType& dtype, const c10::optional<at::ScalarType>& stype = c10::nullopt) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeCast(input0, dtype, stype);
}
static inline NodePtr MakeTensorList(const OpList& inputs) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeTensorList(inputs);
}
static inline NodePtr MakeAsStridedViewUpdate(const Value& input0, const Value& input1, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, const int64_t& storage_offset) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeAsStridedViewUpdate(input0, input1, size, stride, storage_offset);
}
static inline NodePtr MakeAsStrided(const Value& input0, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, const int64_t& storage_offset) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeAsStrided(input0, size, stride, storage_offset);
}
static inline NodePtr MakeDiagonalViewUpdate(const Value& input0, const Value& input1, const int64_t& offset, const int64_t& dim1, const int64_t& dim2) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeDiagonalViewUpdate(input0, input1, offset, dim1, dim2);
}
static inline NodePtr MakeDiagonal(const Value& input0, const int64_t& offset, const int64_t& dim1, const int64_t& dim2) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeDiagonal(input0, offset, dim1, dim2);
}
static inline NodePtr MakeNarrowViewUpdate(const Value& input0, const Value& input1, const std::vector<int64_t>& base_indices) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeNarrowViewUpdate(input0, input1, base_indices);
}
static inline NodePtr MakeNarrow(const Value& input0, const std::vector<int64_t>& base_indices, const std::vector<int64_t>& sizes) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeNarrow(input0, base_indices, sizes);
}
static inline NodePtr MakePermute(const Value& input0, const std::vector<int64_t>& dims) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakePermute(input0, dims);
}
static inline NodePtr MakeResize(const Value& input0, const std::vector<int64_t>& size) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeResize(input0, size);
}
static inline NodePtr MakeSelectViewUpdate(const Value& input0, const Value& input1, const int64_t& dim, const int64_t& start, const int64_t& end, const int64_t& stride) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeSelectViewUpdate(input0, input1, dim, start, end, stride);
}
static inline NodePtr MakeSelect(const Value& input0, const int64_t& dim, const int64_t& start, const int64_t& end, const int64_t& stride) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeSelect(input0, dim, start, end, stride);
}
static inline NodePtr MakeSqueeze(const Value& input0, const int& dim) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeSqueeze(input0, dim);
}
static inline NodePtr MakeUnsqueeze(const Value& input0, const int& dim) {
    static IrBuilder* builder = getBackend()->GetIrBuilder();
    return builder->MakeUnsqueeze(input0, dim);
}

} // namespace lazy
} // namespace torch
