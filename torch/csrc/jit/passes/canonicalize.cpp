#include <torch/csrc/jit/passes/canonicalize.h>

namespace torch { namespace jit {

// Canonicalize a graph, renumbering it so that all structurally equivalent
// graphs have same numbers.
// keep_unique_names: If false, removes unique names and renumbers those.
//   Otherwise, ignores values with unique names.
std::shared_ptr<Graph> Canonicalize(
    const std::shared_ptr<Graph>& graph, bool keep_unique_names) {
  auto r = std::make_shared<Graph>(graph->current_scope());
  std::unordered_map<Value*, Value*> rn_env;
  auto rn_fn = [&](Value* v) { return rn_env.at(v); };
  for (auto* input : graph->inputs()) {
    auto* r_input = r->addInput();
    r_input->copyMetadata(input, keep_unique_names);
    rn_env[input] = r_input;
  }
  for (auto* node : graph->nodes()) {
    auto* r_node = r->createClone(
        node, rn_fn, /*copy_blocks=*/true, keep_unique_names);
    r->appendNode(r_node);
    auto outputs = node->outputs();
    auto r_outputs = r_node->outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
      rn_env[outputs.at(i)] = r_outputs.at(i);
    }
    if (node->hasAttribute(attr::Subgraph)) {
      r_node->g_(attr::Subgraph, Canonicalize(node->g(attr::Subgraph)));
    }
  }
  for (auto* output : graph->outputs()) {
    r->registerOutput(rn_fn(output));
  }

  return r;

}

}}
