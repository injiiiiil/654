#pragma once
#include <torch/csrc/jit/graph_executor_impl.h>

namespace torch {
namespace jit {

struct ProfilingGraphExecutorImpl : public GraphExecutorImplBase {
  ProfilingGraphExecutorImpl(const std::shared_ptr<Graph>& graph);

  ExecutionPlan getPlanFor(Stack& stack) override;
  GraphExecutorState getDebugState() override;
  ~ProfilingGraphExecutorImpl() override = default;
  std::shared_ptr<Graph> _getProfiledGraph() const override;

 private:
  std::shared_ptr<Graph> prepareGraph(
      const std::shared_ptr<Graph>& graph,
      Stack& stack);
  std::unique_ptr<ProfilingRecord> pr_;
  c10::optional<ExecutionPlan>
      profiling_plan_; // plan to run in order to profiling the code
  c10::optional<ExecutionPlan> optimized_plan_;
};

} // namespace jit
} // namespace torch
