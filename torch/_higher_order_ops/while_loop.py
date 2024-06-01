from typing import Callable, Tuple, Union

import torch
import torch.utils._pytree as pytree

from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization

from torch._higher_order_ops.utils import (
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    _set_compilation_env,
    reenter_make_fx,
    UnsupportedAliasMutationException,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import (
    disable_functional_mode,
    FunctionalTensor,
)
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.multiprocessing.reductions import StorageWeakRef
from .utils import (
    _from_fun, 
    clone_outputs_aliasing_inputs, 
    prepare_fw_with_masks, 
    create_fw_bw_graph,
    _unstack_pytree
)

def create_fw_bw_graph_manual(cond_fn, body_fn, bw_cond_fn, num_mapped_args, *args):
    from torch._functorch.aot_autograd import AOTConfig, create_joint
    dummy_aot_config = AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )
    
    mapped_xs = args[:num_mapped_args]
    pos_args = args[num_mapped_args:]

    # Note: We create "clean" environments for make_fx by suspending all dispatch keys
    # between Autograd and Python key. Currently, we only suspend functionalization but more can be
    # added when required. Will encounter two problems if we don't suspend functionalization:
    #
    # 1. make_fx fails to capture operations on input: the inputs are wrapped as _to_functional_tensor_wrapper,
    # but they will be unwrapped before entering ProxyTorchDispatchMode as part of the dispatching.
    # However, it's the outside wrapper that tracer creates proxies for. This casuses tracer fail to
    # fetch the proxy for the inputs and fail to capture any operations on them.
    #
    # 2. make_fx fails to capture output: the outputs after ProxyTorchDispatchMode are further
    # wrapped as FunctionalTensorWrapper in Functionalize key after return. However, the tracer
    # only associates the inner tensor with proxy in ProxyTorchDispatchMode. Therefore,
    # when creating the output node, it fails to associate the wrapped tensor with its proxy.
    # Instead, it will create _tensor_constant as output.

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():

            unwrapped_mapped_xs = pytree.tree_map(_from_fun, mapped_xs)
            example_xs = _unstack_pytree(unwrapped_mapped_xs)[0]

            example_pos_args = [
                _from_fun(arg) if isinstance(arg, torch.Tensor) else arg
                for arg in pos_args
            ]

            example_flat_out = pytree.tree_map(
                _from_fun, body_fn(*example_xs)
            )
            if any(
                not isinstance(out, torch.Tensor) for out in example_flat_out if out is not None
            ):
                raise RuntimeError(
                    "Expect outputs of map only contains tensors or None. "
                    f"Got types {[type(out) for out in example_flat_out]}."
                )
                
            example_xs_out = list(example_xs) + list(example_flat_out)
            
            example_grad = tuple([_from_fun(out) for out in example_xs])
            # fw_graph = make_fx(body_fn)(*example_xs)
            cond_graph = make_fx(cond_fn)(*example_xs)
            bw_cond_graph = make_fx(bw_cond_fn)(example_grad, example_xs_out)
            operands = list(example_xs) + list(pos_args)
            
            bw_path = False
        
            if len(pos_args) > 0 and pos_args[0]:
                bw_path = True

            if bw_path:
                fw_graph, joint_graph = make_fx(body_fn)(*example_xs), make_fx(body_fn)(*example_xs)
            else:
                fw_graph, joint_graph = create_fw_bw_graph(body_fn, True, *operands)
    
        def joint_f(grad, example_fw_outs):
        # def joint_f(*args):
            # grad = args[:grad_num]
            # example_fw_outs = args[:grad_num]
            mapped_grads = grad
            mapped_input = list(example_fw_outs[:xs_out_num_mapped])
            pos_args = list(example_fw_outs[xs_out_num_mapped:])
            bw_path = False
            
            if len(pos_args) > 0 and pos_args[0]:
                mapped_grads = grad[0]
                mapped_input = list(mapped_input[1])
                bw_path = True

            def fw_with_masks(*args):
                fw_out = body_fn(*args)
                return fw_out, [
                    True
                    if isinstance(ret, torch.Tensor) and ret.requires_grad
                    else False
                    for ret in fw_out
                ]

            #TODO: may need dynamic list support here
            #TODO: When the create_fw_bw_graph is invoked within the backward path,
            #then there a special treatment is necessary
            if bw_path:
                g_list = list(mapped_grads)
            else:
                # outp = mapped_input.pop()
                inp = mapped_input[-1:]
                joint = create_joint(fw_with_masks, aot_config=dummy_aot_config)
                _, grads = joint(
                    # list(inp) if not bw_path else (list(mapped_grads), list(inp)),
                    list(inp),
                    [
                        grad
                        for grad in mapped_grads
                        if grad is not None and grad.requires_grad
                    ],
                )
                g_list = list(grads)

            # In order to keep while_loop functional for backward graph,
            # we clone outputs that are aliasing inputs
            input_storage = {
                StorageWeakRef(arg._typed_storage())
                for arg in example_fw_outs
                if isinstance(arg, torch.Tensor)
            }

            def maybe_clone(t):
                if (
                    isinstance(t, torch.Tensor)
                    and StorageWeakRef(t._typed_storage()) in input_storage
                ):
                    return t.clone()
                return t

            return pytree.tree_map(maybe_clone, tuple(g_list))#, pytree.tree_map(maybe_clone, mapped_input_initial)
        
        # xs_out_num_mapped = len(example_xs_out)
        # grad_num = list(example_grad)
        # example_xs_out = example_xs_out + list(pos_args)
        # joint_graph = make_fx(joint_f)(example_grad, example_xs_out)
        # # joint_grads_operands = (list(example_grad), list(example_xs_out))
        # # joint_graph = make_fx(joint_f)(*joint_grads_operands)
        return cond_graph, fw_graph, bw_cond_graph, joint_graph

class WhileLoopOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("while_loop")

    def __call__(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        carried_inputs: Tuple[Union[torch.Tensor, int, float, bool]],
        additional_inputs: Tuple[Union[torch.Tensor, int, float, bool]],
        /,
    ):
        if not isinstance(carried_inputs, tuple):
            raise RuntimeError(
                f"carried_inputs must be a tuple, got {type(carried_inputs)}"
            )
        if not isinstance(additional_inputs, tuple):
            raise RuntimeError(
                f"additional_inputs must be a tuple, got {type(additional_inputs)}"
            )
        if not all(
            isinstance(t, (torch.Tensor, int, float, bool, tuple)) for t in carried_inputs
        ):
            raise RuntimeError(
                "carried_inputs must be a tuple of tensors, ints, floats, or bools, got "
                f"{carried_inputs}"
            )

        if not all(
            isinstance(t, (torch.Tensor, int, float, bool)) for t in additional_inputs
        ):
            raise RuntimeError(
                "additional_inputs must be a tuple of tensors, ints, floats, or bools, got "
                f"{additional_inputs}"
            )
        return super().__call__(cond_fn, body_fn, carried_inputs, additional_inputs)


while_loop_op = WhileLoopOp()
# Override while_loop_op.__module__ to "torch.ops.higher_order" so that in the generated
# graph module, while_loop node's target is correctedly printed as torch.ops.higher_order.while_loop
while_loop_op.__module__ = "torch.ops.higher_order"


def while_loop(cond_fn, body_fn, carried_inputs):
    r"""
    Run body_fn(*carried_inputs) while cond_fn(*carried_inputs) returns a True scalar tensor. Returns the output of body_fn or
    initial carried_inputs.

    .. warning::
        `torch.while_loop` is a prototype feature in PyTorch. It has limited support for input and output types and
        doesn't support training currently. Please look forward to a more stable implementation in a future version of PyTorch.
        Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    `while_loop` is a structured control flow operator. It preserves the loop semantic across the torch.compile and torch.export.

    `while_loop` is equivalent to the following:

        def while_loop(cond_fn, body_fn, carried_inputs):
            val = carried_inputs
            while cond_fn(*val):
                val = body_fn(*val)
            return val

    Args:
        cond_fn (Callable): A callable function that returns a boolean Scalar tensor.

        body_fn (Callable): A callable function that takes the same inputs as `cond_fn` and returns a tuple of tensors

        carried_inputs (Tuple of possibly nested dict/list/tuple of tensors): A tuple of inputs to cond_fn and body_fn. It's also
            the initial value of states that are carried across iterations.

    Example:

        def cond_fn(iter, x):
            return iter.sum() < 10

        def body_fn(iter, x):
            return iter + 1, x.sin()

        while_loop(cond_fn, body_fn, (torch.zeros(1), torch.randn(3, 4)))

    Restrictions:

        - body_fn must return tensors with the same metadata (e.g.shape, dtype) as inputs.

        - body_fn and cond_fn must not in-place mutate the carried_inputs. A clone before the mutation is required.

        - body_fn and cond_fn must not mutate python varialbles (e.g. list/dict) created outside of the body_fn.

        - body_fn and cond_fn's output cannot aliase any of the inputs. A clone is required.

    .. warning::
        Temporal Limitations:

        - 'while_loop' only supports **inference** right now. Autograd will be supported in the future.

    """

    # Currently, additional_inputs is not a user-facing input. It will be automatically set in dynamo.
    # parameters and buffers accessed in cond_fn or body_fn or tensor closures will become additional_inputs.
    additional_inputs: Tuple = tuple()
    
    if torch.compiler.is_dynamo_compiling():
        return while_loop_op(cond_fn, body_fn, carried_inputs, additional_inputs)

    def _validate_input(cond_fn, body_fn, carried_inputs):
        if not callable(cond_fn) or not callable(body_fn):
            raise RuntimeError("Expect cond_fn and body_fn to be callbale.")

        if not isinstance(carried_inputs, (tuple, list)) or pytree.tree_any(
            lambda t: not isinstance(t, torch.Tensor), carried_inputs
        ):
            raise RuntimeError(
                "Expect carried_inputs to be a tuple of possibly nested dict/list/tuple that only"
                f"consists of tensor leaves, but got {carried_inputs}."
            )

    _validate_input(cond_fn, body_fn, carried_inputs)

    with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
        return torch.compile(while_loop_op, backend="eager", fullgraph=True)(
            cond_fn, body_fn, carried_inputs, additional_inputs
        )


@while_loop_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def while_loop_dense(cond_fn, body_fn, carried_inputs, additional_inputs):
    carried_vals = carried_inputs
    outs = []
    
    bw_path = False
    
    if len(additional_inputs) > 0 and additional_inputs[0]:
        bw_path = True
        
    def _is_boolean_scalar_tensor(pred):
        return (
            isinstance(pred, torch.Tensor)
            and pred.size() == torch.Size([])
            and pred.dtype == torch.bool
        )

    #TODO: The while_loop_dense behaves a little different in the forward
    #as well as in the backward path, is this okay?
    if not bw_path:
        outs.append(carried_vals)
        
        if not isinstance(carried_inputs, tuple):
            raise RuntimeError(
                f"carried_inputs must be a tuple but got {type(carried_inputs)}"
            )

        while pred := cond_fn(*carried_vals, *additional_inputs):
            if not _is_boolean_scalar_tensor(pred):
                raise RuntimeError(
                    f"cond_fn must return a boolean scalar tensor but got {pred}"
                )
            out = body_fn(*carried_vals, *additional_inputs)
            assert isinstance(
                out, tuple
            ), f"body_fn should return a tuple but got {type(out)}"
            assert len(out) == len(
                carried_inputs
            ), "body_fn should return the same number of elements as carried_inputs"
            
            carried_vals = out
            outs.append(out)
        
        return tuple(outs)
    else:
        grad = carried_inputs[0]
        outs.append(grad)
        carried_vals = carried_vals_ch = carried_vals_inp = carried_inputs[1]
        
        if not isinstance(grad, tuple):
            raise RuntimeError(
                f"grad of while backward must be a tuple but got {type(grad)}"
            )
            
        if not isinstance(carried_inputs, tuple):
            raise RuntimeError(
                f"carried_inputs of while backward must be a tuple but got {type(carried_inputs)}"
            )
            
        grad_acc = grad
            
        #TODO: Compiled for a fixed length currently
        while len(carried_vals_ch) > 1:
            
            pred = cond_fn(grad, carried_vals_inp)
            if not _is_boolean_scalar_tensor(pred) and not type(pred) == bool:
                raise RuntimeError(
                    f"cond_fn must return a boolean scalar tensor but got {pred}"
                )
            grads = body_fn(grad, carried_vals_inp)
            outs.append(grads)
            grad_acc = [g_acc * g for g_acc, g in zip(grad_acc, grads)]
            
            carried_vals_ch = carried_vals_ch[:-1]
            carried_vals_inp = tuple([carried_vals_inp[-1:] + carried_vals_inp[:-1]][0])
            assert isinstance(
                grad, tuple
            ), f"body_fn in the backward path should return a tuple but got {type(grad)}"
        
        return tuple(grad_acc)

class WhileLoopAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cond_graph, fw_graph, bw_cond_graph, bw_graph, num_mapped_args, *flat_args):
        ctx._cond_graph = cond_graph
        ctx._bw_cond_graph = bw_cond_graph
        ctx._fw_graph = fw_graph
        ctx._bw_graph = bw_graph
        ctx._num_mapped_args = num_mapped_args
        ctx._num_pos_args = len(flat_args[num_mapped_args:])
        
        with torch._C._AutoDispatchBelowAutograd():
            # with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            #     res = torch.compile(while_loop_op, backend="eager", fullgraph=True)(
            #         cond_graph, fw_graph, flat_args[:num_mapped_args], flat_args[num_mapped_args:]
            #     )
            #     flattened_res = []
            #     for r1 in res:
            #         for r in r1:
            #             flattened_res.append(r)
            #     ctx.save_for_backward(*(flattened_res + list(flat_args[num_mapped_args:])))
            #     return res[-1]
                
            #TODO: The compilation does not work, because the produced tuple inside the while_loop_op is not generated properly
            res = while_loop_op(cond_graph, fw_graph, flat_args[:num_mapped_args], flat_args[num_mapped_args:])
            flattened_res = []
            for r1 in res:
                for r in r1:
                    flattened_res.append(r)
            ctx.save_for_backward(*(flattened_res + list(flat_args[num_mapped_args:])))
            return res[-1]

    @staticmethod
    def backward(ctx, *flat_grads):       
        fw_flattened = ctx.saved_tensors
        fw_xs_out = fw_flattened[:len(fw_flattened)-ctx._num_pos_args]
        fw_mapped_posargs = fw_flattened[len(fw_flattened)-ctx._num_pos_args:]

        # with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
        #     grads = torch.compile(while_loop_op, backend="eager", fullgraph=True)(
        #         ctx._bw_cond_graph, ctx._bw_graph, (flat_grads, fw_xs_out), (True,)
        #     )
        #     return None, None, None, None, None, grads
        
        grads = while_loop_op(ctx._bw_cond_graph, ctx._bw_graph, (flat_grads, fw_xs_out), (True,))
        # operands = tuple(list(flat_grads) + list(fw_xs_out))
        # grads = while_loop_op(ctx._bw_cond_graph, ctx._bw_graph, operands, (True,))
        return None, None, None, None, None, grads

@while_loop_op.py_impl(DispatchKey.Autograd)
def while_loop_autograd(cond_fn, body_fn, xs, pos_args):
    num_mapped_args = len(xs)
    
    #def bw_cond_fn(grad, fw_outputs):
    def bw_cond_fn(*args):
        return len(args[-1]) > 1
    
    cond_graph, fw_graph, bw_cond_graph, bw_graph = create_fw_bw_graph_manual(cond_fn, body_fn, bw_cond_fn, num_mapped_args, *xs, *pos_args)
    flat_out = WhileLoopAutogradOp.apply(cond_graph, fw_graph, bw_cond_graph, bw_graph, num_mapped_args, *xs, *pos_args)
    return flat_out

@while_loop_op.py_impl(ProxyTorchDispatchMode)
def while_loop_tracing(mode, cond_fn, body_fn, carried_inputs, additional_inputs):
    def _trace_while_loop(
        proxy_mode, while_loop_op, cond_fn, body_fn, carried_inputs, additional_inputs
    ):
        cond_graph = reenter_make_fx(cond_fn)(*carried_inputs, *additional_inputs)
        body_graph = reenter_make_fx(body_fn)(*carried_inputs, *additional_inputs)

        next_name = None
        i = 0
        while not next_name:
            candidate = f"while_loop_cond_graph_{i}"
            if hasattr(proxy_mode.tracer.root, candidate):
                i += 1
            else:
                next_name = candidate
        cond_graph_name = next_name
        body_graph_name = f"while_loop_body_graph_{i}"
        assert not hasattr(proxy_mode.tracer.root, body_graph_name)

        proxy_mode.tracer.root.register_module(cond_graph_name, cond_graph)
        proxy_mode.tracer.root.register_module(body_graph_name, body_graph)

        args = (cond_graph, body_graph, carried_inputs, additional_inputs)

        proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function", while_loop_op, proxy_args, {}, name="while_loop"
        )

        # body_fn return output with the same pytree and tensor meta data as carried_inputs
        # so we could just return the output after one iteration.
        out = body_fn(*carried_inputs, *additional_inputs)
        return track_tensor_tree(
            out, out_proxy, constant=None, tracer=proxy_mode.tracer
        )

    if mode.enable_tracing:
        return _trace_while_loop(
            mode, while_loop_op, cond_fn, body_fn, carried_inputs, additional_inputs
        )
    else:
        return while_loop_op(cond_fn, body_fn, carried_inputs, additional_inputs)


@while_loop_op.py_impl(FakeTensorMode)
def while_loop_fake_tensor_mode(
    mode, cond_fn, body_fn, carried_inputs, additional_inputs
):
    with mode:
        return body_fn(*carried_inputs, *additional_inputs)


@while_loop_op.py_functionalize_impl
def while_loop_func(ctx, cond_fn, body_fn, carried_inputs, additional_inputs):
    unwrapped_carried_inputs = ctx.unwrap_tensors(carried_inputs)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)
    unwrapped_inputs = unwrapped_carried_inputs + unwrapped_additional_inputs
    with ctx.redispatch_to_next() as m:
        functional_cond_fn = ctx.functionalize(cond_fn)
        functional_body_fn = ctx.functionalize(body_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        for fn, fn_name in [
            (functional_cond_fn, "cond_fn"),
            (functional_body_fn, "body_fn"),
        ]:
            if _has_potential_branch_input_mutation(
                fn, unwrapped_inputs, pre_dispatch=pre_dispatch
            ):
                raise UnsupportedAliasMutationException(
                    f"torch.while_loop's {fn_name} might be modifying the input!"
                )

            if _has_potential_branch_input_alias(
                fn, unwrapped_inputs, pre_dispatch=pre_dispatch
            ):
                raise UnsupportedAliasMutationException(
                    f"torch.while_loop's {fn_name} might be aliasing the input!"
                )
        ret = while_loop_op(
            functional_cond_fn,
            functional_body_fn,
            unwrapped_carried_inputs,
            unwrapped_additional_inputs,
        )
        return ctx.wrap_tensors(ret)
