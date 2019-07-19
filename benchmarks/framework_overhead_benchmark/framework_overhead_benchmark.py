from __future__ import absolute_import, division, print_function, unicode_literals
from utils import ms_to_us, benchmark_module, BenchmarkConfig, ModuleConfig
import argparse

from SimpleAddModule import SimpleAddModule, add_tensors_loop
from pt_wrapper_module import WrapperModule

""" Framework overhead benchmark script.
Benchmark framework overhead.
Currently supported ops: add.
As of now runs only forward pass.
Supports both graph mode and eager mode. In graph mode the module is traced via JIT tracing.
Debug option prints the traced graph is graph_mode is enabled.
Graph can be saved via save option. Saved in the directory where benchmark is run.
Example build/run:
buck run @mode/opt <path-to-framework_overhead_benchmark>:framework_overhead_benchmark --
 --add_op --graph_mode --eager_mode (Runs both graph mode and eager mode)
buck run @mode/opt <path-to-framework_overhead_benchmark>:framework_overhead_benchmark --
 --add_op --graph_mode (Runs only graph mode)
"""

SUPPORTED_OPS = {"add_op"}

def parse_op_args(op):
    op_list = ops.split(",")

def print_results(result):
    print("===================================")
    for key, value in result.items():
        print("{}, latency per iter (us):{}".format(key, ms_to_us(value)))
    print("===================================")

def benchmark_simple_fn(args, config, module_config, module_type, result):
    """ Benchmarks a PyTorch traceable function specified in the config.
    Instantiates a wrapper object that wraps the object of module_type and runs the forward
    method using benchmark_module.
    Args:
        config:         contains number of warmup and benchmark iterations.
        module_config:  module_config which contains op, number of parameters that op takes
                    and wether graph mode is enabled or not.
        module_type:    Type of the module to be wrapped. e.g. SimpleAddModule for add op.
        result:         dictionary instance to be populated with the benchmark result (latency per iter).
    """
    print("Benchmarking {}".format(module_type.__name__))
    f_name = module_config.pt_fn.__name__ + ":Num Operands=" + str(module_config.num_params)
    graph_mode_str = "Graph mode" + ":" + str(module_config.graph_mode)
    result_key = ','.join((f_name, graph_mode_str))
    module = WrapperModule(module_type, module_config, args.debug, args.save)
    latency_per_iter_ms = benchmark_module(config, module)
    result[result_key] = latency_per_iter_ms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", default="add_op", dest="op", type=str)
    parser.add_argument("--debug", default=False, dest="debug", action="store_true")
    parser.add_argument("--save", default=False, dest="save", action="store_true")
    parser.add_argument("--eager_mode", default=False, dest="eager_mode", action="store_true")
    parser.add_argument("--num_warmup_iters", type=int, default=100)
    parser.add_argument("--num_iters", type=int, default=1000)
    args = parser.parse_args()

    if args.op not in SUPPORTED_OPS:
        print("Op {} is not supported: Supported ops are:{}".format(args.op, SUPPORTED_OPS))
        return

    num_warmup_iters = args.num_warmup_iters
    num_iters = args.num_iters
    config = BenchmarkConfig(num_warmup_iters, num_iters)
    graph_mode = True
    if args.eager_mode:
        graph_mode = False
    result = {}
    if args.op == "add_op":
        num_params = 2
        module_config = ModuleConfig(add_tensors_loop, num_params, graph_mode)
        benchmark_simple_fn(args, config, module_config, SimpleAddModule, result)
    print_results(result)

if __name__ == "__main__":
    main()
