#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {

// Will return a new value of type val with the DataType dtype, if it's a
// tensorview it will propagate the shape information from val.
TORCH_CUDA_API Val* newValLike(const Val* val, DataType dtype) {
  switch (val->getValType().value()) {
    case (ValType::TensorView):
      return val->as<TensorView>()->newForOutput(dtype);
    case (ValType::NamedScalar):
    case (ValType::Scalar):
      switch (dtype) {
        case (DataType::Bool):
          return new Bool();
        case (DataType::Float):
          return new Float();
        case (DataType::Half):
          return new Half();
        case (DataType::Int):
          return new Int();
        default:
          break;
      }
    default:
      break;
  }

  TORCH_CHECK(
      false,
      "Could not generate a new value of type ",
      val->getValType().value(),
      " with data type ",
      val->getDataType().value());
}

TORCH_CUDA_API Val* newValLike(const Val* val) {
  return newValLike(val, val->getDataType().value());
}

TORCH_CUDA_API Val* promoteNew(Val* v1, Val* v2) {
  // Can't promote two types if they aren't both
  // values with valid data types.
  TORCH_CHECK(v1->isVal() && v2->isVal());
  TORCH_CHECK(
      v1->getDataType() != DataType::Null &&
      v2->getDataType() != DataType::Null);

  ValType out_vtype =
      promote_type(v1->getValType().value(), v2->getValType().value());
  DataType out_dtype =
      promote_type(v1->getDataType().value(), v2->getDataType().value());

  if (out_vtype == v2->getValType().value())
    return newValLike(v2, out_dtype);

  return newValLike(v1, out_dtype);
}

Val* newConstScalar(DataType dtype, int val) {
  switch (dtype) {
    case (DataType::Int):
      return new Int(val);
    default:
      break;
  }
  TORCH_CHECK(
      false,
      "Could not generate a new Scalar with data type ",
      dtype,
      "and constant value: ",
      val);
}

Val* newConstScalar(DataType dtype, float val) {
  switch (dtype) {
    case (DataType::Float):
      return new Float(val);
    default:
      break;
  }
  TORCH_CHECK(
      false,
      "Could not generate a new Scalar with data type ",
      dtype,
      "and constant value: ",
      val);
}

TORCH_CUDA_API Val* castOp(DataType dtype, Val* v1) {
  if (v1->getDataType().value() == dtype)
    return v1;

  if (cast_func_str(std::make_pair(v1->getDataType().value(), dtype)) ==
      c10::nullopt) {
    TORCH_CHECK(
        false,
        "Illegal Cast value from  DataType: ",
        v1->getDataType().value(),
        " to DataType: ",
        dtype);
  }

  Val* out = newValLike(v1, dtype);
  new UnaryOp(UnaryOpType::Cast, out, v1);
  return out;
}

TORCH_CUDA_API TensorView* castOp(DataType dtype, TensorView* v1) {
  return castOp(dtype, v1->as<Val>())->as<TensorView>();
}

// UNARY OPERATIONS

TORCH_CUDA_API Val* unaryOp(UnaryOpType type, Val* v1) {
  Val* out = newValLike(v1);
  new UnaryOp(type, out, v1);
  return out;
}

TORCH_CUDA_API TensorView* unaryOp(UnaryOpType type, TensorView* v1) {
  return unaryOp(type, v1->as<Val>())->as<TensorView>();
}

TORCH_CUDA_API Val* neg(Val* v) {
  return unaryOp(UnaryOpType::Neg, v);
}
TORCH_CUDA_API TensorView* neg(TensorView* v) {
  return unaryOp(UnaryOpType::Neg, v);
}

// BINARY OPERATIONS

namespace {
// Helper function to reduce repetitive code
template <typename T1, typename T2>
TensorView* arithOpOverloads(Val* (*func)(Val*, Val*), T1* v1, T2* v2) {
  return func(v1->template as<Val>(), v2->template as<Val>())
      ->template as<TensorView>();
}
template <typename T1, typename T2>
TensorView* arithOpOverloads(BinaryOpType type, T1* v1, T2* v2) {
  return binaryOp(type, v1->template as<Val>(), v2->template as<Val>())
      ->template as<TensorView>();
}
template <typename T1, typename T2, typename T3>
TensorView* arithOpOverloads(
    Val* (*func)(Val*, Val*, Val*),
    T1* v1,
    T2* v2,
    T3* v3) {
  return func(
             v1->template as<Val>(),
             v2->template as<Val>(),
             v3->template as<Val>())
      ->template as<TensorView>();
}
template <typename T1, typename T2, typename T3, typename T4>
TensorView* arithOpOverloads(
    Val* (*func)(Val*, Val*, Val*, Val*),
    T1* v1,
    T2* v2,
    T3* v3,
    T4* v4) {
  return func(
             v1->template as<Val>(),
             v2->template as<Val>(),
             v3->template as<Val>(),
             v4->template as<Val>())
      ->template as<TensorView>();
}
} // namespace

TORCH_CUDA_API Val* binaryOp(BinaryOpType type, Val* v1, Val* v2) {
  Val* out = promoteNew(v1, v2);
  if (is_logical_op(type)) {
    if (out->getDataType().value() != DataType::Bool)
      out = newValLike(out, DataType::Bool);
  } else if (type >= BinaryOpType::Mod) {
    if (out->getDataType().value() != DataType::Int)
      out = newValLike(out, DataType::Int);
  }
  new BinaryOp(type, out, v1, v2);
  return out;
}
TORCH_CUDA_API TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    Val* v2) {
  return arithOpOverloads(type, v1, v2);
}
TORCH_CUDA_API TensorView* binaryOp(
    BinaryOpType type,
    Val* v1,
    TensorView* v2) {
  return arithOpOverloads(type, v1, v2);
}
TORCH_CUDA_API TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    TensorView* v2) {
  return arithOpOverloads(type, v1, v2);
}

// add
TORCH_CUDA_API Val* add(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::Add, v1, v2);
}
TORCH_CUDA_API TensorView* add(TensorView* v1, Val* v2) {
  return arithOpOverloads(add, v1, v2);
}
TORCH_CUDA_API TensorView* add(Val* v1, TensorView* v2) {
  return arithOpOverloads(add, v1, v2);
}
TORCH_CUDA_API TensorView* add(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(add, v1, v2);
}
// sub
TORCH_CUDA_API Val* sub(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::Sub, v1, v2);
}
TORCH_CUDA_API TensorView* sub(TensorView* v1, Val* v2) {
  return arithOpOverloads(sub, v1, v2);
}
TORCH_CUDA_API TensorView* sub(Val* v1, TensorView* v2) {
  return arithOpOverloads(sub, v1, v2);
}
TORCH_CUDA_API TensorView* sub(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(sub, v1, v2);
}
// mul
TORCH_CUDA_API Val* mul(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::Mul, v1, v2);
}
TORCH_CUDA_API TensorView* mul(TensorView* v1, Val* v2) {
  return arithOpOverloads(mul, v1, v2);
}
TORCH_CUDA_API TensorView* mul(Val* v1, TensorView* v2) {
  return arithOpOverloads(mul, v1, v2);
}
TORCH_CUDA_API TensorView* mul(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(mul, v1, v2);
}
// div
TORCH_CUDA_API Val* div(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::Div, v1, v2);
}
TORCH_CUDA_API TensorView* div(TensorView* v1, Val* v2) {
  return arithOpOverloads(div, v1, v2);
}
TORCH_CUDA_API TensorView* div(Val* v1, TensorView* v2) {
  return arithOpOverloads(div, v1, v2);
}
TORCH_CUDA_API TensorView* div(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(div, v1, v2);
}
// mod
TORCH_CUDA_API Val* mod(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::Mod, v1, v2);
}
TORCH_CUDA_API TensorView* mod(TensorView* v1, Val* v2) {
  return arithOpOverloads(mod, v1, v2);
}
TORCH_CUDA_API TensorView* mod(Val* v1, TensorView* v2) {
  return arithOpOverloads(mod, v1, v2);
}
TORCH_CUDA_API TensorView* mod(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(mod, v1, v2);
}
// lt
TORCH_CUDA_API Val* lt(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::LT, v1, v2);
}
TORCH_CUDA_API TensorView* lt(TensorView* v1, Val* v2) {
  return arithOpOverloads(lt, v1, v2);
}
TORCH_CUDA_API TensorView* lt(Val* v1, TensorView* v2) {
  return arithOpOverloads(lt, v1, v2);
}
TORCH_CUDA_API TensorView* lt(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(lt, v1, v2);
}
// ceilDiv
TORCH_CUDA_API Val* ceilDiv(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::CeilDiv, v1, v2);
}
TORCH_CUDA_API TensorView* ceilDiv(TensorView* v1, Val* v2) {
  return arithOpOverloads(ceilDiv, v1, v2);
}
TORCH_CUDA_API TensorView* ceilDiv(Val* v1, TensorView* v2) {
  return arithOpOverloads(ceilDiv, v1, v2);
}
TORCH_CUDA_API TensorView* ceilDiv(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(ceilDiv, v1, v2);
}
// andOp
TORCH_CUDA_API Val* andOp(Val* v1, Val* v2) {
  TORCH_CHECK(
      v1->getDataType().value() == DataType::Bool,
      "Input1 should be of type bool, not ",
      v1->getDataType().value());
  TORCH_CHECK(
      v2->getDataType().value() == DataType::Bool,
      "Input2 should be of type bool, not ",
      v2->getDataType().value());
  return binaryOp(BinaryOpType::And, v1, v2);
}
TORCH_CUDA_API TensorView* andOp(TensorView* v1, Val* v2) {
  return arithOpOverloads(andOp, v1, v2);
}
TORCH_CUDA_API TensorView* andOp(Val* v1, TensorView* v2) {
  return arithOpOverloads(andOp, v1, v2);
}
TORCH_CUDA_API TensorView* andOp(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(andOp, v1, v2);
}

// REDUCTION OPERATIONS

TensorView* reductionOp(
    BinaryOpType reduction_op_type,
    const std::vector<int>& axes,
    Val* init,
    TensorView* v1) {
  TORCH_CHECK(
      init->isConstScalar(),
      "Cannot create a reduction operation where the initial value is not a const scalar.");

  TORCH_CHECK(
      v1->getRootDomain() == v1->domain(),
      "Reducing a tensor once it's gone under transformations is not permitted at this time. Please set reductions before calling split/merge/reorder/computeAt.");

  std::vector<unsigned int> uint_axes;
  for (int axis : axes) {
    if (axis < 0)
      axis += int(v1->nDims());

    TORCH_CHECK(
        axis >= 0 && (unsigned int)axis < v1->nDims(),
        "Reduction on invalid axis, recieved: ",
        axis,
        " however tensor view only has ",
        v1->nDims(),
        " dims.");

    uint_axes.push_back((unsigned int)axis);
  }

  TensorView* out = v1->newForReduction(uint_axes);
  if (init->getDataType().value() != v1->getDataType().value())
    init = castOp(v1->getDataType().value(), init);
  new ReductionOp(reduction_op_type, init, out, v1);
  return out;
}

TORCH_CUDA_API TensorView* sum(TensorView* v1, const std::vector<int>& axes) {
  Val* init;
  switch (v1->getDataType().value()) {
    case (DataType::Float):
      init = new Float(0.0);
      break;
    case (DataType::Int):
      init = new Int(0);
      break;
    default:
      TORCH_CHECK(
          false,
          "Could not generate a sum op for tensor with type: ",
          v1->getDataType().value());
  }

  return reductionOp(BinaryOpType::Add, axes, init, v1);
}

// COMPOUND OPERATIONS

// add_alpha
TORCH_CUDA_API Val* add_alpha(Val* v1, Val* v2, Val* s) {
  TORCH_CHECK(
      s->getValType().value() == ValType::Scalar,
      "Alpha value should be a Scalar Valtype and not ",
      s->getValType().value());

  Val* intrm = binaryOp(BinaryOpType::Mul, v2, s);
  return binaryOp(BinaryOpType::Add, v1, intrm);
}
TORCH_CUDA_API TensorView* add_alpha(TensorView* v1, Val* v2, Val* v3) {
  return arithOpOverloads(add_alpha, v1, v2, v3);
}
TORCH_CUDA_API TensorView* add_alpha(Val* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(add_alpha, v1, v2, v3);
}
TORCH_CUDA_API TensorView* add_alpha(TensorView* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(add_alpha, v1, v2, v3);
}
// sub_alpha
TORCH_CUDA_API Val* sub_alpha(Val* v1, Val* v2, Val* s) {
  TORCH_CHECK(
      s->getValType().value() == ValType::Scalar,
      "Alpha value should be a Scalar Valtype and not ",
      s->getValType().value());

  Val* intrm = binaryOp(BinaryOpType::Mul, v2, s);
  return binaryOp(BinaryOpType::Sub, v1, intrm);
}
TORCH_CUDA_API TensorView* sub_alpha(TensorView* v1, Val* v2, Val* v3) {
  return arithOpOverloads(sub_alpha, v1, v2, v3);
}
TORCH_CUDA_API TensorView* sub_alpha(Val* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(sub_alpha, v1, v2, v3);
}
TORCH_CUDA_API TensorView* sub_alpha(TensorView* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(sub_alpha, v1, v2, v3);
}
// lerp
TORCH_CUDA_API Val* lerp(Val* start, Val* end, Val* weight) {
  Val* intrm1 = binaryOp(BinaryOpType::Sub, end, start);
  Val* intrm2 = binaryOp(BinaryOpType::Mul, weight, intrm1);
  return binaryOp(BinaryOpType::Add, start, intrm2);
}
TORCH_CUDA_API TensorView* lerp(TensorView* v1, Val* v2, Val* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TORCH_CUDA_API TensorView* lerp(Val* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TORCH_CUDA_API TensorView* lerp(Val* v1, Val* v2, TensorView* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TORCH_CUDA_API TensorView* lerp(TensorView* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TORCH_CUDA_API TensorView* lerp(TensorView* v1, Val* v2, TensorView* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TORCH_CUDA_API TensorView* lerp(Val* v1, TensorView* v2, TensorView* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TORCH_CUDA_API TensorView* lerp(
    TensorView* v1,
    TensorView* v2,
    TensorView* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
// addcmul
TORCH_CUDA_API Val* addcmul(Val* v1, Val* v2, Val* v3, Val* s) {
  TORCH_CHECK(
      s->getValType().value() == ValType::Scalar,
      "Alpha value should be a Scalar Valtype and not ",
      s->getValType().value());

  Val* intrm1 = binaryOp(BinaryOpType::Mul, v3, s);
  Val* intrm2 = binaryOp(BinaryOpType::Mul, v2, intrm1);
  return binaryOp(BinaryOpType::Add, v1, intrm2);
}
TORCH_CUDA_API TensorView* addcmul(TensorView* v1, Val* v2, Val* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TORCH_CUDA_API TensorView* addcmul(Val* v1, TensorView* v2, Val* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TORCH_CUDA_API TensorView* addcmul(Val* v1, Val* v2, TensorView* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TORCH_CUDA_API TensorView* addcmul(
    TensorView* v1,
    TensorView* v2,
    Val* v3,
    Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TORCH_CUDA_API TensorView* addcmul(
    TensorView* v1,
    Val* v2,
    TensorView* v3,
    Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TORCH_CUDA_API TensorView* addcmul(
    Val* v1,
    TensorView* v2,
    TensorView* v3,
    Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TORCH_CUDA_API TensorView* addcmul(
    TensorView* v1,
    TensorView* v2,
    TensorView* v3,
    Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}

// TERNARY OPERATIONS
// where
TORCH_CUDA_API Val* where(Val* c, Val* v1, Val* v2) {
  TORCH_CHECK(
      c->getDataType().value() == DataType::Bool,
      "Condition should be of DataType Bool, not ",
      c->getDataType().value());

  Val* out = promoteNew(v1, v2);
  new TernaryOp(TernaryOpType::Where, out, c, v1, v2);
  return out;
}
TORCH_CUDA_API TensorView* where(TensorView* v1, Val* v2, Val* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TORCH_CUDA_API TensorView* where(Val* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TORCH_CUDA_API TensorView* where(Val* v1, Val* v2, TensorView* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TORCH_CUDA_API TensorView* where(TensorView* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TORCH_CUDA_API TensorView* where(TensorView* v1, Val* v2, TensorView* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TORCH_CUDA_API TensorView* where(Val* v1, TensorView* v2, TensorView* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TORCH_CUDA_API TensorView* where(
    TensorView* v1,
    TensorView* v2,
    TensorView* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}

TORCH_CUDA_API Val* threshold(Val* in, Val* thresh, Val* value) {
  TORCH_CHECK(
      in->getDataType().value() == thresh->getDataType().value() &&
          in->getDataType().value() == value->getDataType().value(),
      "All input DataType values should match the input ",
      in->getDataType().value());
  TORCH_CHECK(
      thresh->getValType().value() == ValType::Scalar &&
          value->getValType().value() == ValType::Scalar,
      "Thresh and Value values should be Scalars");

  Val* out = newValLike(in);

  new TernaryOp(TernaryOpType::Threshold, out, in, thresh, value);
  return out;
}

TORCH_CUDA_API TensorView* threshold(TensorView* in, Val* thresh, Val* value) {
  return threshold(in->as<Val>(), thresh, value)->as<TensorView>();
}

TORCH_CUDA_API Val* clamp(Val* in, Val* min_val, Val* max_val) {
  TORCH_CHECK(
      in->getDataType().value() == min_val->getDataType().value() &&
          in->getDataType().value() == max_val->getDataType().value(),
      "All input DataType values should match the input ",
      in->getDataType().value());
  TORCH_CHECK(
      min_val->getValType().value() == ValType::Scalar &&
          max_val->getValType().value() == ValType::Scalar,
      "Min and Max values should be Scalars");

  Val* out = newValLike(in);

  new TernaryOp(TernaryOpType::Clamp, out, in, min_val, max_val);
  return out;
}

TORCH_CUDA_API TensorView* clamp(TensorView* in, Val* min_val, Val* max_val) {
  return clamp(in->as<Val>(), min_val, max_val)->as<TensorView>();
}

} // namespace fuser
} // namespace jit
} // namespace torch
