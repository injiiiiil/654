
#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>

#include <torch/csrc/jit/ir/ir.h>

#include <c10/util/Exception.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {

Statement::Statement(const Statement* src, IrCloner* ir_cloner) {
  ir_cloner->registerClone(src, this);
  name_ = src->name_;
  fusion_ = ir_cloner->fusion();
}

Val* Statement::asVal() {
  TORCH_INTERNAL_ASSERT(isVal(), "Cannot cast to Val as this is not a Val.");
  return this->as<Val>();
}

Expr* Statement::asExpr() {
  TORCH_INTERNAL_ASSERT(isExpr(), "Cannot cast to Expr as this is not a Expr.");
  return this->as<Expr>();
}

void Statement::print() const {
  IRPrinter ir_printer(std::cout);
  ir_printer.handle(this);
  std::cout << std::endl;
}

// When we create a Val we immediately register them with the active fusion.
Val::Val(ValType _vtype, DataType _dtype, bool register_val)
    : vtype_{_vtype}, dtype_{_dtype} {
  Fusion* fusion = FusionGuard::getCurFusion();
  TORCH_CHECK(
      fusion != nullptr, "No active fusion group found when creating a Val.");
  this->fusion_ = fusion;
  if (register_val)
    this->name_ = this->fusion_->registerVal(this);
}

Val::Val(const Val* src, IrCloner* ir_cloner)
    : Statement(src, ir_cloner), vtype_(src->vtype_), dtype_(src->dtype_) {}

// Traverse origin of all values involved in constructing the provided val.
// Check if all values involved are constant values, meaning the provided
// val is also a constant value.
namespace {

class ConstCheck : OptOutConstDispatch {
 private:
  bool is_const_ = true;

  void handle(const Bool* b) override {
    is_const_ = is_const_ && b->isConst();
  }

  void handle(const Float* f) override {
    is_const_ = is_const_ && f->isConst();
  }

  void handle(const Half* h) override {
    is_const_ = is_const_ && h->isConst();
  }

  void handle(const Int* i) override {
    is_const_ = is_const_ && i->isConst();
  }

  void handle(const NamedScalar* ns) override {
    is_const_ = is_const_ && false;
  }

  void handle(const Expr* expr) override {
    for (auto inp : expr->inputs()) {
      handle(inp);
    }
  }

  void handle(const Val* val) override {
    const Expr* orig = FusionGuard::getCurFusion()->origin(val);
    if (orig != nullptr)
      handle(orig);
    else
      OptOutConstDispatch::handle(val);
  }

 public:
  static bool isConst(const Val* val) {
    ConstCheck cc;
    cc.handle(val);
    return cc.is_const_;
  }
};

} // namespace

bool Val::isConstScalar() const {
  if (!isScalar())
    return false;
  return ConstCheck::isConst(this);
}

bool Val::isZeroInt() const {
  if (isConstScalar() && getValType().value() == ValType::Scalar &&
      getDataType().value() == DataType::Int &&
      this->as<Int>()->value().has_value() &&
      this->as<Int>()->value() == Int::ScalarType(0))
    return true;
  return false;
}

bool Val::isOneInt() const {
  if (isConstScalar() && getValType().value() == ValType::Scalar &&
      getDataType().value() == DataType::Int &&
      this->as<Int>()->value().has_value() &&
      this->as<Int>()->value() == Int::ScalarType(1))
    return true;
  return false;
}

c10::optional<DataType> Val::getDataType() const {
  TORCH_INTERNAL_ASSERT(
      dtype_ != DataType::Null, "Value does not have a data type.");
  return dtype_;
}

Expr* Val::getOrigin() {
  return fusion_->origin(this);
}

const Expr* Val::getOrigin() const {
  return fusion_->origin(this);
}

// We don't register with the active fusion in Expr as this needs to be done
// after inputs and outputs are registered with the Expr
Expr::Expr(ExprType _type) : type_{_type} {
  Fusion* fusion = FusionGuard::getCurFusion();
  if (fusion == nullptr)
    TORCH_CHECK(false, "No active fusion group found when creating an Expr.");
  this->fusion_ = fusion;
}

Expr::Expr(const Expr* src, IrCloner* ir_cloner)
    : Statement(src, ir_cloner),
      type_(src->type_),
      inputs_(ir_cloner->clone(src->inputs_)),
      outputs_(ir_cloner->clone(src->outputs_)) {}

bool Expr::sameAs(const Expr* const other) const {
  if (getExprType() != other->getExprType())
    return false;
  if (inputs().size() != other->inputs().size() ||
      outputs().size() != other->outputs().size())
    return false;
  for (size_t i = 0; i < inputs().size(); i++) {
    if (!input(i)->sameAs(other->input(i)))
      return false;
  }
  return true;
}

} // namespace fuser
} // namespace jit
} // namespace torch
