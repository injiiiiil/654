#include <torch/csrc/jit/constants.h>
#include <ATen/core/functional.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

namespace {
c10::OperatorOptions aliasAnalysisInternalSpecialCase() {
  c10::OperatorOptions options;
  options.setAliasAnalysis(AliasAnalysisKind::INTERNAL_SPECIAL_CASE);
  return options;
}
} // namespace

Value* insertConstant(
    Graph& g,
    const IValue& val,
    const c10::TypePtr& result_type,
    c10::optional<SourceRange> loc,
    c10::optional<ScopePtr> scope) {
  auto value = tryInsertConstant(g, val, result_type, loc, scope);
  if (value) {
    return *value;
  }
  throw constant_not_supported_error(
      "Unsupported value kind: " + val.tagKind());
}

// IValue -> Constant node
c10::optional<Value*> tryInsertConstant(
    Graph& g,
    const IValue& val,
    const c10::TypePtr& result_type,
    c10::optional<SourceRange> loc,
    c10::optional<ScopePtr> scope) {
  Node* n = g.create(prim::Constant);
  if (val.isTensor()) {
    at::Tensor ref = val.toTensor();
    if (!ref.defined()) {
      n->destroy();
      return g.insertNode(g.createNone(TensorType::get()))->output();
    }
    // TODO: fix all cases where we are not passing in a variable,
    // and then change this to an AT_ASSERT
    if (!ref.is_variable()) {
      ref = autograd::make_variable(ref, /*requires_grad=*/false);
    } else {
      AT_ASSERT(!ref.requires_grad());
    }
    n->output()->inferTypeFrom(
        ref); // note: before t_ because of std::move(ref)
    n->t_(attr::value, std::move(ref));
  } else if (val.isInt()) {
    n->i_(attr::value, val.toInt());
    n->output()->setType(IntType::get());
  } else if (val.isDouble()) {
    n->f_(attr::value, val.toDouble());
    n->output()->setType(FloatType::get());
  } else if (val.isBool()) {
    n->i_(attr::value, val.toBool());
    n->output()->setType(BoolType::get());
  } else if (val.isBoolList()) {
    auto bool_list = val.toBoolList();
    n->is_(
        attr::value, std::vector<int64_t>(bool_list.begin(), bool_list.end()));
    n->output()->setType(ListType::ofBools());
  } else if (val.isIntList()) {
    n->is_(attr::value, val.toIntListRef().vec());
    n->output()->setType(ListType::ofInts());
  } else if (val.isTensorList()) {
    n->ts_(
        attr::value,
        fmap(val.toTensorListRef(), [](const at::Tensor& t) {
          AT_ASSERT(t.is_variable() && !t.requires_grad());
          return t;
        }));
    n->output()->setType(ListType::ofTensors());
  } else if (val.isString()) {
    n->s_(attr::value, val.toString()->string());
    n->output()->setType(StringType::get());
  } else if (val.isDevice()) {
    std::stringstream ss;
    ss << val.toDevice();
    n->s_(attr::value, ss.str());
    n->output()->setType(DeviceObjType::get());
  } else if (val.isNone()) {
    n->output()->setType(NoneType::get());
  } else {
    n->destroy();
    return c10::nullopt;
  }
  if (loc)
    n->setSourceRange(*loc);
  if (scope)
    n->setScope(*scope);
  if (result_type) {
    auto inferred_type = n->output()->type();

    if (inferred_type->isSubtypeOf(NoneType::get()) &&
        !inferred_type->isSubtypeOf(result_type)) {
      // None doesn't subtype Optional, but an Optional can be None, so handle
      // that here
      TORCH_CHECK(
          result_type->kind() == TypeKind::OptionalType,
          "Expected OptionalType or NoneType, got ",
          result_type->python_str());
      n->output()->setType(result_type);
    } else if (!(inferred_type->isSubtypeOf(TensorType::get()) &&
                 result_type->isSubtypeOf(inferred_type))) {
      // Retain more type information in case of tensor constant
      n->output()->setType(result_type);
    }
  }
  return g.insertNode(n)->output();
}

RegisterOperators reg({
    Operator(
        FunctionSchema(
            prim::Constant,
            "",
            {},
            {},
            /*is_vararg=*/false,
            /*is_varret=*/true),
        [](const Node* node) -> Operation {
          TypePtr type = node->output()->type();
          if (type->isSubtypeOf(TensorType::get())) {
            auto t = node->t(attr::value);
            return [t](Stack& stack) {
              push(stack, t);
              return 0;
            };
          } else if (type->isSubtypeOf(BoolType::get())) {
            bool b = node->i(attr::value);
            return [b](Stack& stack) {
              push(stack, b);
              return 0;
            };
          } else if (
              type->isSubtypeOf(NumberType::get()) &&
              node->kindOf(attr::value) == AttributeKind::i) {
            auto i = node->i(attr::value);
            return [i](Stack& stack) {
              push(stack, i);
              return 0;
            };
          } else if (
              type->isSubtypeOf(NumberType::get()) &&
              node->kindOf(attr::value) == AttributeKind::f) {
            auto f = node->f(attr::value);
            return [f](Stack& stack) {
              push(stack, f);
              return 0;
            };
          } else if (type->isSubtypeOf(ListType::ofInts())) {
            const auto& is = node->is(attr::value);
            return [is](Stack& stack) {
              push(stack, is);
              return 0;
            };
          } else if (type->isSubtypeOf(ListType::ofBools())) {
            const auto bs = fmap<bool>(node->is(attr::value));
            return [bs](Stack& stack) {
              push(stack, bs);
              return 0;
            };
          } else if (type->isSubtypeOf(ListType::ofTensors())) {
            const auto& ts = node->ts(attr::value);
            return [ts](Stack& stack) {
              push(stack, ts);
              return 0;
            };
          } else if (type == StringType::get()) {
            const auto& s = node->s(attr::value);
            return [s](Stack& stack) {
              push(stack, s);
              return 0;
            };
          } else if (type == DeviceObjType::get()) {
            auto d = c10::Device(node->s(attr::value));
            return [d](Stack& stack) {
              push(stack, d);
              return 0;
            };
          } else if (node->mustBeNone()) {
            return [](Stack& stack) {
              push(stack, IValue());
              return 0;
            };
          } else if ( type->isSubtypeOf(OptionalType::create(ListType::ofInts()) ) ) {
              std::cout << "optional list of ints constant = " << *node << std::endl;
              if (node->hasAttribute(attr::value))
              {
                const auto& is = node->is(attr::value);
                std::cout << "getting the value list\n";
                return [is](Stack& stack) {
                  push(stack, is);
                  return 0;
                };
              }
              else
              {
                return [](Stack& stack) {
                  std::cout << "getting none\n";
                  push(stack, IValue());
                  return 0;
                };
              }
           } else {
            std::stringstream ss;
            ss << "constant literal not supported for: " << type->str();
            throw std::runtime_error(ss.str());
          }
        },
        aliasAnalysisInternalSpecialCase()),
});

c10::optional<IValue> toIValue(const Value* v) {
  if (v->node()->kind() != prim::Constant) {
    return c10::nullopt;
  }
  // use implemenation of prim::Constant to compute the output IValue
  auto op = getOperation(v->node());
  Stack stack;
  op(stack);
  return stack.back();
}
} // namespace jit
} // namespace torch
