#include "torch/csrc/jit/ir.h"

#include <algorithm>
#include <unordered_map>

#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/node_hashing.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/utils/hash.h"

namespace torch { namespace jit {

namespace {

bool tensorEqual(const at::Tensor& lhs, const at::Tensor& rhs) {
  return &lhs.type() == &rhs.type() && lhs.equal(rhs);
}

bool tensorListEqual(const std::vector<at::Tensor>& lhs, const std::vector<at::Tensor>& rhs) {
  if (lhs.size() != rhs.size()) return false;
  return std::equal(lhs.begin(), lhs.end(), rhs.begin(), tensorEqual);
}


// Check whether two nodes have the same attributes in CSE.
// This function may be too conservative for general use.
// Do NOT support t/ts/g/gs attributes.
// If t/ts are supported, CONSTANT node comparison may need to consider device.
bool attributesEqualCSE(const Node* lhs, const Node* rhs) {
  JIT_ASSERT(lhs != nullptr);
  JIT_ASSERT(rhs != nullptr);
  // One has attributes, the other does not.
  if (lhs->hasAttributes() != rhs->hasAttributes()) return false;
  // Neither has attributes.
  if (!lhs->hasAttributes() && !rhs->hasAttributes()) return true;

  auto lnames = lhs->attributeNames();
  auto rnames = rhs->attributeNames();
  std::sort(lnames.begin(), lnames.end());
  std::sort(rnames.begin(), rnames.end());
  if (lnames != rnames) return false;

  for (auto name : lnames) {
    if (lhs->kindOf(name) != rhs->kindOf(name)) return false;

    #define COMPARE_ATTRIBUTEVALUE(type) \
      case AttributeKind::type: \
        { if (lhs->type(name) != rhs->type(name)) return false; } break;

    switch(lhs->kindOf(name)) {
      COMPARE_ATTRIBUTEVALUE(f)
      COMPARE_ATTRIBUTEVALUE(fs)
      COMPARE_ATTRIBUTEVALUE(i)
      COMPARE_ATTRIBUTEVALUE(is)
      COMPARE_ATTRIBUTEVALUE(s)
      COMPARE_ATTRIBUTEVALUE(ss)
      case AttributeKind::t: {
        if (!tensorEqual(lhs->t(name), rhs->t(name))) return false;
        break;
      }
      case AttributeKind::ts: {
        if (!tensorListEqual(lhs->ts(name), rhs->ts(name))) return false;
        break;
      }
      case AttributeKind::g:
      case AttributeKind::gs:
        return false;
    }

    #undef COMPARE_ATTRIBUTEVALUE
  }

  return true;
}

} // anonymous namespace


size_t HashNodeCSE::operator()(const Node* k) const {
  JIT_ASSERT(k != nullptr);
  return get_hash(k->kind(),
                  fmap(k->outputs(), [](const Value *v) { return v->type()->kind(); }),
                  fmap(k->inputs(), [](const Value *v) { return v->unique(); }));
};

bool EqualNodeCSE::operator()(const Node* lhs, const Node* rhs) const {
  if (lhs == nullptr && rhs == nullptr) return true;
  if (lhs == nullptr || rhs == nullptr) return false;

  if (lhs->kind() != rhs->kind()) return false;

  // Check whether the output types are the same.
  auto lhs_outputs = lhs->outputs();
  auto rhs_outputs = rhs->outputs();
  if (lhs_outputs.size() != rhs_outputs.size()) return false;
  for (size_t i = 0; i < lhs_outputs.size(); ++i) {
    if (*lhs_outputs[i]->type() != *rhs_outputs[i]->type())
      return false;
  }

  // Check whether the inputs are the same.
  auto lhs_inputs = lhs->inputs();
  auto rhs_inputs = rhs->inputs();
  if (lhs_inputs.size() != rhs_inputs.size()) return false;
  if (!std::equal(lhs_inputs.begin(), lhs_inputs.end(), rhs_inputs.begin())) return false;

  if (!attributesEqualCSE(lhs, rhs)) return false;

  return true;
};

}}
