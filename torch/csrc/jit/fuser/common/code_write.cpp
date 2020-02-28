#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/iriostream.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/transform_replay.h>
#include <torch/csrc/jit/fuser/common/type.h>

#include <torch/csrc/jit/fuser/common/code_write.h>

namespace torch {
namespace jit {
namespace fuser {

std::vector<Int*> CodeWrite::getLoopIndices() {
  std::vector<Int*> inds;
  for (auto pair : fors)
    inds.push_back(pair.first);
  return inds;
}

std::ostream& CodeWrite::print_indices(
    std::ostream& os,
    const std::vector<Int*>& indices) {
  os << "[";
  for (const auto& ind : indices) {
    print_inline(os, ind);
    if (ind != *(indices.end() - 1))
      os << ", ";
  }
  return os << "]";
}

std::ostream& CodeWrite::print(std::ostream& os, const TensorView* const tv) {
  TensorDomain* td = tv->domain();
  if (producer && consumer != nullptr) {
    // get new reference so replay inline doesn't change the original.
    TensorView* tv_ = tv->clone();

    tv_->resetView();

    TransformReplay::fullReplay(consumer, tv_);

    if (tv_->tensor() != nullptr) {
      os << "%T" << tv_->tensor()->name();
    } else {
      os << "%TV" << tv_->name();
    }

    std::vector<Int*> indices =
        IndexCompute::computeIndices(tv_, getLoopIndices());

    print_indices(os, indices);

  } else if (producer) {
    throw std::runtime_error(
        "Could not find consumer for producer in CodeWrite.");
  } else {
    if (tv->tensor() != nullptr)
      os << "%T" << tv->tensor()->name();
    else
      os << "%TV" << tv->name();

    std::vector<Int*> indices =
        IndexCompute::computeIndices(tv, getLoopIndices());

    print_indices(os, indices);
  }

  return os;
}

std::ostream& CodeWrite::print(std::ostream& os, const Val* const val) {
  if (*(val->getValType()) == ValType::TensorView) {
    return print(os, static_cast<const TensorView* const>(val));
  }

  return os << val;
}

std::ostream& CodeWrite::print(std::ostream& os, const UnaryOp* const uop) {
  if (!parse_inline) {
    print(os, uop->out());
    os << " = ";
  }

  if (uop->out()->getValType().value() == ValType::TensorView) {
    consumer = static_cast<TensorView*>(uop->out());
    producer = true;
  }

  if (auto inline_uop = inline_op_str(uop->type())) {
    os << inline_uop.value();
    print(os, uop->in());
  } else {
    os << uop->type() << "(";
    print(os, uop->in());
    os << ")";
  }
  consumer = nullptr;
  producer = false;

  if (!parse_inline)
    os << std::endl;

  return os;
}

std::ostream& CodeWrite::print(std::ostream& os, const BinaryOp* const bop) {
  if (!parse_inline) {
    print(os, bop->out());
    os << " = ";
  }

  if (bop->out()->getValType().value() == ValType::TensorView) {
    consumer = static_cast<TensorView*>(bop->out());
    producer = true;
  }

  if (auto inline_bop = inline_op_str(bop->type())) {
    print(os, bop->lhs());
    os << " " << inline_bop.value() << " ";
    print(os, bop->rhs());
  } else {
    os << bop->type() << "(";
    print(os, bop->lhs());
    os << ", ";
    print(os, bop->rhs());
    os << ")";
  }

  consumer = nullptr;
  producer = false;

  if (!parse_inline)
    os << std::endl;

  return os;
}

void CodeWrite::indent() {
  for (int i = 0; i < fors.size(); i++)
    std::cout << "  ";
}

void CodeWrite::closeScope() {
  fors.pop_back();
  indent();
  std::cout << "}" << std::endl;
}

void CodeWrite::openFor(IterDomain* id) {
  indent();
  fors.push_back(std::pair<Int*, Int*>{new Int(), id->size()});

  std::cout << "for( " << fors.back().first << " : ";
  print_inline(std::cout, id->size());
  std::cout << " ) {" << std::endl;
}

void CodeWrite::clearActiveView() {
  active_view_axis = 0;
  active_view = nullptr;
}

void CodeWrite::resetFors() {
  while (!fors.empty())
    closeScope();

  reset_fors = false;
  clearActiveView();
}

void CodeWrite::updateView(TensorView* tv) {
  if (reset_fors)
    resetFors();

  // Hit the final statment in a string of compute_at's. Need to close the for
  // loops down to the previous compute_at axis, then need to put my remaining
  // for loops on there. Also need to set reset_fors flag.
  if (!tv->hasComputeAt()) {
    if (active_view != nullptr && tv->same_as(active_view)) {
      int depth = fors.size();
      for (int i = active_view_axis; i < depth; i++)
        closeScope();
      clearActiveView();
    } else {
      resetFors();
    }
    for (int i = fors.size(); i < tv->domain()->size(); i++)
      openFor(tv->domain()->axis(i));
    reset_fors = true;
  } else {
    active_view_axis = tv->getComputeAtAxis();
    active_view = tv->getComputeAtView();

    int depth = fors.size();
    for (int i = active_view_axis; i < depth; i++)
      closeScope();
    for (int i = fors.size(); i < tv->domain()->size(); i++)
      openFor(tv->domain()->axis(i));
  }
}

void CodeWrite::handle(UnaryOp* uop) {
  if (uop->out()->getValType().value() == ValType::TensorView) {
    updateView(static_cast<TensorView*>(uop->out()));
    indent();
    print(std::cout, uop);
  }
}
void CodeWrite::handle(BinaryOp* bop) {
  if (bop->out()->getValType().value() == ValType::TensorView) {
    updateView(static_cast<TensorView*>(bop->out()));
    indent();
    print(std::cout, bop);
  }
}

void CodeWrite::traverse(
    const Fusion* const fusion,
    bool from_outputs_only,
    bool breadth_first,
    std::unordered_set<ValType> val_types) {
  IterVisitor::traverse(fusion, from_outputs_only, breadth_first, val_types);
  resetFors();
}

} // namespace fuser
} // namespace jit
} // namespace torch