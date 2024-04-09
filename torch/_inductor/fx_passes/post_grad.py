import functools
import itertools
import logging
import operator
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Union

from sympy import Expr

import torch
import torch._inductor as inductor
import torch.utils._pytree as pytree
from torch import fx
from torch._decomp import register_decomposition
from torch._dynamo.utils import counters, optimus_scuba_log

from torch._prims_common import is_boolean_dtype, is_expandable_to, is_integer_dtype

from torch._utils_internal import upload_graph
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq

from .. import config, ir, pattern_matcher
from ..fx_utils import FakeTensorUpdater, get_fake_args_kwargs, get_node_storage

from ..lowering import lowerings as L
from ..pattern_matcher import (
    _return_true,
    Arg,
    CallFunction,
    CallFunctionVarArgs,
    filter_nodes,
    get_arg_value,
    get_mutation_region_id,
    Ignored,
    init_once_fakemode,
    KeywordArg,
    ListOf,
    Match,
    MULTIPLE,
    PatternMatcherPass,
    register_graph_pattern,
    stable_topological_sort,
)
from ..utils import decode_device, is_pointwise_use
from ..virtualized import V
from .ddp_fusion import fuse_ddp_communication
from .group_batch_fusion import group_batch_fusion_passes
from .pre_grad import is_same_dict, save_inductor_dict
from .reinplace import reinplace_inplaceable_ops
from .split_cat import POST_GRAD_PATTERNS


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

pattern_matcher_passes = POST_GRAD_PATTERNS.values()
# First pass_patterns[0] are applied, then [1], then [2]
pass_patterns = [
    PatternMatcherPass(),
    PatternMatcherPass(),
    PatternMatcherPass(),
]
# patterns applied only in inference
inference_patterns = PatternMatcherPass()


def post_grad_passes(gm: torch.fx.GraphModule, is_inference: bool):
    """
    Passes that run on after grad.  This is called once on the forwards
    graph and once on the backwards graph.

    The IR here has been normalized and functionalized.
    """
    if config.dce:
        # has some issues with mutation in inference mode
        gm.graph.eliminate_dead_code()

    if is_inference and config.reorder_for_locality:
        reorder_for_locality(gm.graph)

    fake_tensor_updater = FakeTensorUpdater(gm.graph)

    if config.post_grad_custom_pre_pass is not None:
        config.post_grad_custom_pre_pass(gm.graph)

    if config.pattern_matcher:
        lazy_init()
        optimus_scuba_log["before_recompile_post_grad"] = upload_graph(gm.graph)
        group_batch_fusion_passes(gm.graph, pre_grad=False)
        remove_noop_ops(gm.graph)
        for patterns in pass_patterns:
            patterns.apply(gm.graph)  # type: ignore[arg-type]
        for pattern_matcher_pass in pattern_matcher_passes:
            inductor_before_change = save_inductor_dict(
                [pattern_matcher_pass.pass_name]
            )
            pattern_matcher_pass.apply(gm.graph)  # type: ignore[arg-type]
            if not is_same_dict(counters["inductor"], inductor_before_change):
                optimus_scuba_log[
                    f"{pattern_matcher_pass.pass_name}_post_grad"
                ] = upload_graph(gm.graph)
        if is_inference:
            inference_patterns.apply(gm.graph)  # type: ignore[arg-type]

    if config._fuse_ddp_communication:
        fuse_ddp_communication(
            gm.graph,
            config._fuse_ddp_communication_passes,
            config._fuse_ddp_bucket_size,
        )

    if config.post_grad_custom_post_pass is not None:
        config.post_grad_custom_post_pass(gm.graph)

    stable_topological_sort(gm.graph)

    move_constructors_to_cuda(gm.graph)

    fake_tensor_updater.incremental_update()

    # Keep these last, since they introduces mutation. Look at
    # ./fx_passes/README.md for a discussion of mutation invariants.
    reinplace_inplaceable_ops(gm.graph)
    decompose_auto_functionalized(gm.graph)

    gm.recompile()
    optimus_scuba_log["after_recompile_post_grad"] = upload_graph(gm.graph)
    gm.graph.lint()


@init_once_fakemode
def lazy_init():
    if torch._C._has_mkldnn:
        from . import decompose_mem_bound_mm  # noqa: F401
        from .mkldnn_fusion import _mkldnn_fusion_init

        _mkldnn_fusion_init()


def reorder_for_locality(graph: torch.fx.Graph):
    def visit(other_node):
        if (
            other_node.op == "call_function"
            and other_node.target != operator.getitem
            and all((n in seen_nodes) for n in other_node.users)
            and get_mutation_region_id(graph, node)
            == get_mutation_region_id(graph, other_node)
        ):
            # move node's producers right before it
            node.prepend(other_node)

    seen_nodes = set()

    # only reorder nodes before the first copy_ in the graph.
    # copy_ will appear at the end of functionalized graphs when there is mutation on inputs,
    # and this reordering doesnt work well with mutation
    first_copy = next(
        iter(graph.find_nodes(op="call_function", target=torch.ops.aten.copy_.default)),
        None,
    )
    past_mutating_epilogue = True if first_copy is None else False

    for node in reversed(graph.nodes):
        seen_nodes.add(node)
        if not past_mutating_epilogue:
            past_mutating_epilogue = node is first_copy
            continue

        torch.fx.map_arg((node.args, node.kwargs), visit)


def register_lowering_pattern(pattern, extra_check=_return_true, pass_number=1):
    """
    Register an aten to inductor IR replacement pattern
    """
    return pattern_matcher.register_lowering_pattern(
        pattern, extra_check, pass_dict=pass_patterns[pass_number]
    )


################################################################################
# Actual patterns below this point.
# Priority of patterns is:
#   - later output nodes first
#   - order patterns are defined in
################################################################################


def is_valid_mm_plus_mm(match: Match):
    *b1, m1, k1 = match.kwargs["mat1"].meta.get("tensor_meta").shape
    *b2, k2, n1 = match.kwargs["mat2"].meta.get("tensor_meta").shape
    if k1 != k2:
        return False

    *b1, m2, k3 = match.kwargs["mat3"].meta.get("tensor_meta").shape
    *b2, k4, n2 = match.kwargs["mat4"].meta.get("tensor_meta").shape
    if k3 != k4:
        return False

    if m1 != m2 or n1 != n2:
        return False

    return True


@register_lowering_pattern(
    CallFunction(
        aten.add,
        CallFunction(aten.mm, KeywordArg("mat1"), KeywordArg("mat2")),
        CallFunction(aten.mm, KeywordArg("mat3"), KeywordArg("mat4")),
    ),
    extra_check=is_valid_mm_plus_mm,
)
def mm_plus_mm(match: Match, mat1, mat2, mat3, mat4):
    return inductor.kernel.mm_plus_mm.tuned_mm_plus_mm(mat1, mat2, mat3, mat4)


def cuda_and_enabled_mixed_mm(match):
    return (config.use_mixed_mm or config.force_mixed_mm) and getattr(
        match.kwargs["mat1"].meta.get("val"), "is_cuda", False
    )


def cuda_and_enabled_mixed_mm_and_not_int8(match):
    return (
        cuda_and_enabled_mixed_mm(match)
        and getattr(match.kwargs["mat1"].meta.get("val"), "is_cuda", False)
        and getattr(match.kwargs["mat2"].meta.get("val"), "dtype", torch.int8)
        != torch.int8
    )  # bitshift numerics in triton and pytorch don't match for torch.int8


"""
    this is intended to be used to unpack a [K,N] int4 tensor from a [K/2, N] uint4x2 tensor
    (where the int4 and uint4x2 are represented with int8 and uint8 respectively)
    where every other row of the int4 is packed with the row above it as:
    uint4x2[k,n] = (8+int4[2*k,n])+(8+int4[2*k+1,n])<<4

    unpack formulas:
    int4[2*k,n]=(uint4x2[k,n] & 0xF) - 8
    int4[2*k+1,n]=(uint4x2[k,n] >> 4) - 8

    thus matching on unpack formula:
    torch.mm(mat1, torch.cat((mat2 & 0xF, mat2>>4),1).reshape(mat2_mm_shape).to(mat2_dtype).sub(8))

    note: although the unpack formula in pytorch and the triton kernel is designed for a uint8 mat2, the behavior
    of the kernel matches the pytorch formula for all dtypes except torch.int8
    where the bitwise numerics in triton do not match those in pytorch.
"""


@register_lowering_pattern(
    CallFunction(
        aten.mm.default,
        KeywordArg("mat1"),
        CallFunction(
            aten.sub.Tensor,
            CallFunction(
                prims.convert_element_type.default,
                CallFunction(
                    aten.reshape.default,
                    CallFunction(
                        aten.cat.default,
                        ListOf(
                            CallFunction(
                                aten.bitwise_and.Scalar,
                                KeywordArg("mat2"),
                                0xF,
                            ),
                            CallFunction(
                                aten.__rshift__.Scalar,
                                KeywordArg("mat2"),
                                4,
                            ),
                        ),
                        1,
                    ),
                    KeywordArg("mat2_mm_shape"),
                ),
                KeywordArg("mat2_dtype"),
            ),
            8,
        ),
    ),
    extra_check=cuda_and_enabled_mixed_mm_and_not_int8,
)
def uint4x2_mixed_mm(match: Match, mat1, mat2, mat2_mm_shape, mat2_dtype):
    return inductor.kernel.unpack_mixed_mm.tuned_uint4x2_mixed_mm(
        mat1, mat2, mat2_mm_shape, mat2_dtype
    )


"""
    torch.mm(mat1, mat2.to(mat2_dtype))
"""


@register_lowering_pattern(
    CallFunction(
        aten.mm,
        KeywordArg("mat1"),
        CallFunction(
            prims.convert_element_type.default,
            KeywordArg("mat2"),
            KeywordArg("mat2_dtype"),
        ),
    ),
    extra_check=cuda_and_enabled_mixed_mm,
)
def mixed_mm(match: Match, mat1, mat2, mat2_dtype):
    return inductor.kernel.mm.tuned_mixed_mm(mat1, mat2, mat2_dtype)


@register_graph_pattern(
    CallFunction(
        aten.cumsum.default,
        CallFunction(
            torch.ops.aten.full.default,
            KeywordArg("shape"),
            KeywordArg("fill_value"),
            dtype=KeywordArg("dtype"),
            layout=Ignored(),
            device=KeywordArg("device"),
            pin_memory=False,
            _users=MULTIPLE,
        ),
        KeywordArg("dim"),
        _users=MULTIPLE,
    ),
    pass_dict=pass_patterns[1],
)
def pointless_cumsum_replacement(match: Match, shape, fill_value, device, dtype, dim):
    """Based on a pattern in OPTForCausalLM"""

    if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
        # cumsum promotes all integral types to int64
        dtype = torch.int64

    def repl(*shape):
        dim_size = shape[dim]
        idx = torch.arange(1, dim_size + 1, device=device, dtype=dtype)

        inter_shape = [1] * len(shape)
        inter_shape[dim] = dim_size
        return (idx * fill_value).view(inter_shape).expand(shape)

    # only replace the output node, not all nodes
    match.nodes = [match.output_node()]
    with V.fake_mode:
        match.replace_by_example(repl, list(shape))


def shape_of_mm(a, b):
    m, _ = a.get_size()
    _, n = b.get_size()
    return [m, n]


@register_lowering_pattern(
    CallFunction(aten.cat, ListOf(CallFunction(aten.mm, Arg(), Arg())), Arg()),
)
def cat_mm(match, inputs, dim):
    return cat_tuned_op(match, inputs, dim, op=L[aten.mm], shape_of=shape_of_mm)


@register_lowering_pattern(
    CallFunction(
        aten.cat, ListOf(CallFunction(aten.addmm, Arg(), Arg(), Arg())), Arg()
    ),
)
def cat_addmm(match, inputs, dim):
    def shape_of(bias, a, b):
        m, _ = a.get_size()
        _, n = b.get_size()
        return [m, n]

    return cat_tuned_op(match, inputs, dim, op=L[aten.addmm], shape_of=shape_of)


def cat_tuned_op(match, inputs, dim, *, op, shape_of):
    """
    Memory planning to remove cat. We can't use the stock memory
    planner since autotuning matmuls needs to know the output layout.
    """
    if len(inputs) == 1:
        return op(*inputs[0])

    # TODO(jansel): rewrite this as a bmm?
    if dim < 0:
        dim += len(shape_of(*inputs[0]))
    assert dim in (0, 1)
    notdim = 1 - dim

    new_size: Optional[Union[List[Expr], List[int]]] = None
    offsets_start = []
    offsets_end = []

    # compute output sizes
    for i in range(len(inputs)):
        shape = shape_of(*inputs[i])
        if new_size is None:
            new_size = shape
        else:
            new_size[notdim] = V.graph.sizevars.guard_equals(  # type: ignore[call-overload]
                shape[notdim], new_size[notdim]
            )
            new_size[dim] += shape[dim]
        offsets_start.append(new_size[dim] - shape[dim])
        offsets_end.append(new_size[dim])

    assert new_size is not None
    dtype = functools.reduce(
        torch.promote_types,
        [x.get_dtype() for x in itertools.chain.from_iterable(inputs)],
    )
    device = inputs[0][0].get_device()
    kernel = ir.ConcatKernel(
        name=None,
        layout=ir.FixedLayout(device, dtype, new_size),
        inputs=[],
    )
    kernel_tensor = ir.TensorBox.create(kernel)

    for i in range(len(inputs)):
        dst = ir.SliceView.create(kernel_tensor, dim, offsets_start[i], offsets_end[i])
        src = op(*inputs[i], layout=dst.get_layout()).data.data
        assert isinstance(src, (ir.ExternKernelOut, ir.TemplateBuffer))
        src.layout = ir.NonOwningLayout(dst)
        kernel.inputs.append(src)

    kernel.name = V.graph.register_buffer(kernel)
    kernel.inputs = ir.ConcatKernel.unwrap_storage(kernel.inputs)
    return kernel_tensor


_cat_1 = CallFunction(aten.cat, Arg(), 1, _users=2)


@register_lowering_pattern(
    CallFunction(
        aten.cat,
        [
            _cat_1,
            CallFunction(
                aten.slice,
                _cat_1,
                1,
                0,
                KeywordArg("size"),
            ),
        ],
        1,
    )
)
def cat_slice_cat(match, cat_input, size, dim=1):
    """
    This is an example of a more complex pattern where cat_1 is used
    multiple times inside the pattern.  We fold 2 calls to cat into one.

    Matches:
        cat_1: f32[1024, 4077] = torch.ops.aten.cat.default([add_26, primals_217], 1)
        slice_1: f32[1024, 4077] = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
        slice_2: f32[1024, 19] = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 19)
        cat_2: f32[1024, 4096] = torch.ops.aten.cat.default([cat_1, slice_2], 1)


    Rewrite to:
        slice_2 = torch.ops.aten.slice.Tensor(add_26, 1, 0, 19)
        cat_2 = torch.ops.aten.cat.default([add_26, primals_217, slice2], 1)
    """
    first, *rest = cat_input
    # Optimization is optional, because we can just not fold the cat
    # size should be within first.get_size()[dim] such that the optimization is valid.
    # For negative `end`, we currently fallback to not optimizing.
    if size >= 0 and V.graph.sizevars.statically_known_leq(size, first.get_size()[dim]):
        # fold 2 cats into 1 cat
        return L[aten.cat](
            [
                first,
                *rest,
                L[aten.slice](first, dim, 0, size),
            ],
            dim,
        )
    else:
        # don't expect to hit this case, just fall back
        tmp = L[aten.cat](cat_input, dim)
        return L[aten.cat](
            [
                tmp,
                L[aten.slice](tmp, dim, 0, size),
            ],
            dim,
        )


def is_valid_splitwithsizes_cat(match):
    split_nodes = filter_nodes(match.nodes, aten.split_with_sizes)
    cat_nodes = filter_nodes(match.nodes, aten.cat)
    get_item_nodes = filter_nodes(match.nodes, operator.getitem)
    if len(split_nodes) != 1 or len(cat_nodes) != 1:
        return False
    split_node, cat_node = split_nodes[0], cat_nodes[0]
    # The dim of split and cat should match for passthrough
    if get_arg_value(split_node, 2, "dim") != get_arg_value(cat_node, 1, "dim"):
        return False
    get_item_args = {
        get_arg_value(get_item_node, 1) for get_item_node in get_item_nodes
    }
    assert None not in get_item_args
    split_sizes = get_arg_value(split_node, 1, "split_sizes")
    # All parts of split should be included in the cat
    if get_item_args != set(range(len(split_sizes))):
        return False
    # The order of get_item_args should same with cat_node used.
    # For example, if the split_node like split_with_sizes(input, [2, 2, 3], 1),
    # the cat node should be like cat([get_item(0), get_item(1), get_item(2)], 1).
    cat_items_args_order = [
        get_arg_value(item_node, 1) for item_node in get_arg_value(cat_node, 0)
    ]
    if cat_items_args_order != list(range(len(split_sizes))):
        return False

    return True


def same_meta(node1: torch.fx.Node, node2: torch.fx.Node):
    """True if two nodes have the same metadata"""
    val1 = node1.meta.get("val")
    val2 = node2.meta.get("val")
    return (
        val1 is not None
        and val2 is not None
        and statically_known_true(sym_eq(val1.size(), val2.size()))
        and val1.layout == val2.layout
        and val1.dtype == val2.dtype
        and val1.device == val2.device
        and (
            val1.layout != torch.strided
            or statically_known_true(sym_eq(val1.stride(), val2.stride()))
        )
    )


noop_registry: Dict[Any, Any] = {}


def register_noop_decomp(targets, nop_arg=0):
    def register_fun(cond):
        register_decomposition(targets, registry=noop_registry, unsafe=True)(
            (cond, nop_arg)
        )
        return cond

    return register_fun


@register_noop_decomp(aten.slice)
def slice_noop(self, dim=0, start=None, end=None, step=1):
    if start is None or end is None:
        return False
    if start == 0 and end >= 2**63 - 1 and step == 1:
        return True
    return False


@register_noop_decomp(aten.slice_scatter, 1)
def slice_scatter_noop(self, src, dim=0, start=None, end=None, step=1):
    if start is None:
        start = 0
    if end is None:
        end = 2**63 - 1
    if start == 0 and end >= 2**63 - 1 and step == 1:
        return True
    return False


@register_noop_decomp(aten.repeat)
def repeat_noop(self, repeats):
    return all(r == 1 for r in repeats)


@register_noop_decomp(aten.constant_pad_nd)
def constant_pad_nd(x, padding, fill_value=0):
    return all(p == 0 for p in padding)


@register_noop_decomp(torch.ops.prims.convert_element_type)
def convert_element_type_noop(x, dtype: torch.dtype):
    return x.dtype == dtype


@register_noop_decomp(torch.ops.prims.device_put)
def device_put_noop(x, device):
    return x.device == decode_device(device)


@register_noop_decomp([aten.ceil, aten.floor, aten.round, aten.trunc])
def int_noop(x):
    return is_integer_dtype(x.dtype)


@register_noop_decomp([aten.pow])
def pow_noop(a, b):
    return isinstance(b, int) and b == 1


@register_noop_decomp([aten.cat], lambda args: args[0][0])
def cat_noop(inputs, dim=0):
    return len(inputs) == 1


@register_noop_decomp(aten.view)
def view_noop(arg, size):
    return arg.shape == size


# Note, we also always have a check for identical metadata, which is why these
# are safe
@register_noop_decomp([aten.copy], nop_arg=1)
@register_noop_decomp([aten.alias, aten.clone])
def true_noop(*args, **kwargs):
    return True


def remove_noop_ops(graph: torch.fx.Graph):
    """
    Removes both operations that are essentially aten.clone and operations that are essentially aten.alias from the graph.
    """
    inputs = set()
    input_storages = set()
    output_storages = set()

    for node in graph.find_nodes(op="placeholder"):
        inputs.add(node)
        input_storages.add(get_node_storage(node))

    output_node = next(iter(reversed(graph.nodes)))
    assert output_node.op == "output"
    outputs = output_node.args[0]
    if not isinstance(outputs, (list, tuple)):
        # nested subgraphs can have singleton outputs
        outputs = (outputs,)
    for out in outputs:
        if isinstance(out, torch.fx.Node):
            output_storages.add(get_node_storage(out))

    for node in graph.nodes:
        if node.target in noop_registry:
            cond, src_index = noop_registry[node.target]
            if isinstance(src_index, int):
                src = node.args[src_index]
            else:
                src = src_index(node.args)
            if not isinstance(src, torch.fx.Node):
                continue
            # Don't introduce new aliasing between inputs and outputs.
            # See fx_passes/README.md for a discussion of why this is
            # necessary.
            node_storage = get_node_storage(node)
            src_storage = get_node_storage(src)
            node_is_view = node_storage == src_storage
            if (
                not node_is_view
                and node_storage in output_storages
                and (src_storage in input_storages or src_storage in output_storages)
            ):
                continue

            # Even if input and outputs are expected to alias,
            # don't make "node is src" True
            if (
                node_is_view
                and node in output_node.args
                and (src in inputs or src in output_node.args)
            ):
                continue

            is_valid, args, kwargs = get_fake_args_kwargs(node)
            if not is_valid:
                continue
            if same_meta(node, src) and cond(*args, **kwargs):
                node.replace_all_uses_with(src)
                graph.erase_node(node)


def decompose_auto_functionalized(graph):
    graph_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.higher_order.auto_functionalized),
        pass_dict=graph_pass,
    )
    def replacement(match: Match, *args, **kwargs):
        from torch._higher_order_ops.auto_functionalize import auto_functionalized_dense

        only_clone_these_tensors = tuple(
            match.nodes[0].meta.get("only_clone_these_tensors", [])
        )

        flat_args, spec = pytree.tree_flatten((args, kwargs))

        # NB: we combine (args, kwargs) into flat args for replacing.
        # This is replace_by_example uses make_fx which does not support
        # tracing a function with kwargs.
        def decomp(*flat_args):
            args, kwargs = pytree.tree_unflatten(flat_args, spec)
            return auto_functionalized_dense(*args, only_clone_these_tensors, **kwargs)

        with V.fake_mode:
            match.replace_by_example(decomp, flat_args, run_dce=False)

    graph_pass.apply(graph)
    for node in graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.auto_functionalized
    ):
        raise AssertionError("auto_functionalized was not removed")


@register_lowering_pattern(
    CallFunction(
        aten.cat,
        ListOf(
            CallFunction(
                operator.getitem,
                CallFunction(
                    aten.split_with_sizes,
                    KeywordArg("input_"),
                    Ignored(),
                    Ignored(),
                    _users=MULTIPLE,
                ),
                Ignored(),
            ),
        ),
        Ignored(),
    ),
    pass_number=2,
    extra_check=is_valid_splitwithsizes_cat,
)
def splitwithsizes_cat_replace(match, input_):
    return input_


def is_valid_cat_splitwithsizes(match):
    cat_nodes = filter_nodes(match.nodes, aten.cat)
    split_nodes = filter_nodes(match.nodes, aten.split_with_sizes)
    if len(split_nodes) != 1 or len(cat_nodes) != 1:
        return False
    split_node, cat_node = split_nodes[0], cat_nodes[0]

    # the cat node has other users: can't eliminate
    if len(cat_node.users) > 1:
        return False

    # the dim of the cat and split should match
    dim = get_arg_value(split_node, 2, "dim")
    if dim != get_arg_value(cat_node, 1, "dim"):
        return False

    cat_inputs = list(get_arg_value(cat_node, 0))
    split_sizes = get_arg_value(split_node, 1, "split_sizes")
    # the number of input tensors in cat and the
    # length of the split sizes should match
    if len(cat_inputs) != len(split_sizes):
        return False

    for cat_input, split_size in zip(cat_inputs, split_sizes):
        # each cat input tensor's size along dim
        # should match the corresponding split size
        if "val" not in cat_input.meta:
            return False
        cat_input_size = cat_input.meta["val"].size(dim)
        if cat_input_size != split_size:
            return False

    return True


@register_lowering_pattern(
    CallFunction(
        aten.split_with_sizes,
        CallFunction(
            aten.cat,
            KeywordArg("input_"),
            Ignored(),
            _users=MULTIPLE,
        ),
        Ignored(),
        Ignored(),
    ),
    pass_number=2,
    extra_check=is_valid_cat_splitwithsizes,
)
def cat_splitwithsizes_replace(match, input_):
    return input_


def view_to_reshape(gm):
    """
    Replace view ops in the GraphModule to reshape ops.
    """
    for nd in gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.view.default
    ):
        nd.target = torch.ops.aten.reshape.default


def should_prefer_unfused_addmm(match):
    inp = match.kwargs["inp"]
    if not inp.meta["val"].is_cuda:
        return False

    output = match.output_node()
    return all(is_pointwise_use(use) for use in output.users)


@register_graph_pattern(
    CallFunction(aten.addmm, KeywordArg("inp"), Arg(), Arg()),
    pass_dict=pass_patterns[2],
    extra_check=should_prefer_unfused_addmm,
)
def unfuse_bias_add_to_pointwise(match: Match, mat1, mat2, *, inp):
    def repl(inp, x1, x2):
        return x1 @ x2 + inp

    with V.fake_mode:
        match.replace_by_example(repl, [inp, mat1, mat2])


def is_valid_addmm_fusion(match):
    mat1, mat2 = match.args
    inp = match.kwargs["inp"]

    if not (
        isinstance(inp, torch.fx.Node) and isinstance(inp.meta["val"], torch.Tensor)
    ):
        return False  # Input is a number

    in_shape = inp.meta["val"].shape
    mm_shape = mat1.meta["val"].shape[0], mat2.meta["val"].shape[1]
    matched = is_expandable_to(in_shape, mm_shape)
    if not matched:
        return False  # Shape mismatch

    return not should_prefer_unfused_addmm(match)


@register_graph_pattern(
    CallFunction(
        aten.add,
        CallFunction(aten.mm, Arg(), Arg()),
        KeywordArg("inp"),
    ),
    pass_dict=pass_patterns[2],
    extra_check=is_valid_addmm_fusion,
)
@register_graph_pattern(
    CallFunction(
        aten.add,
        KeywordArg("inp"),
        CallFunction(aten.mm, Arg(), Arg()),
    ),
    pass_dict=pass_patterns[2],
    extra_check=is_valid_addmm_fusion,
)
def addmm(match, mat1, mat2, *, inp):
    def repl(inp, mat1, mat2):
        return aten.addmm(inp, mat1, mat2)

    with V.fake_mode:
        match.replace_by_example(repl, [inp, mat1, mat2])


def check_shape_cuda_and_fused_int_mm_mul_enabled(match):
    return (
        config.force_fuse_int_mm_with_mul
        and len(getattr(match.args[2].meta.get("val"), "shape", [])) == 2
        and getattr(match.args[2].meta.get("val"), "is_cuda", False)
    )


@register_lowering_pattern(
    CallFunction(
        prims.convert_element_type.default,
        CallFunction(
            aten.mul,
            CallFunction(
                aten._int_mm,
                Arg(),
                Arg(),
            ),
            Arg(),
        ),
        Arg(),
    ),
    check_shape_cuda_and_fused_int_mm_mul_enabled,
)
@register_lowering_pattern(
    CallFunction(
        aten.mul,
        CallFunction(
            aten._int_mm,
            Arg(),
            Arg(),
        ),
        Arg(),
    ),
    check_shape_cuda_and_fused_int_mm_mul_enabled,
)
def fused_int_mm_mul(match: Match, mat1, mat2, mat3, out_dtype=None):
    return inductor.kernel.mm.tuned_fused_int_mm_mul(mat1, mat2, mat3, out_dtype)


class ConstructorMoverPass:
    def __init__(self, target: str, allow_outputs: bool = False) -> None:
        """
        Move constructors from cpu to the target_device.

        Sweeps through the module, looking for constructor nodes that can be moved
        to the target_device.

        A constructor node can be moved to the target_device iff all of its users
        can also be moved (tested by cannot_be_moved). Otherwise, all dependent
        constructor nodes won't be moved.

        - target: target device type
        - allow_outputs: allow outputs to be moved
        """

        self.target = target
        self.allow_outputs = allow_outputs

        assert isinstance(target, str), (
            "target should be a string representing the device type. "
            f"Got: {type(target).__name__}"
        )

    def allow_cpu_device(self, node: fx.Node) -> bool:
        """
        Returns whether a node that returns a tensor on the target device may have
        cpu tensors as input.
        """
        return node.target in (
            torch.ops.aten.index.Tensor,
            torch.ops.aten.index_put.default,
            torch.ops.aten.index_put_.default,
            torch.ops.aten.copy.default,
            torch.ops.aten.copy_.default,
            torch.ops.aten.slice_scatter.default,
        )

    def cannot_be_moved(self, node: fx.Node) -> bool:
        """
        Returns whether a node can be moved to the target device.

        If this function returns False, it means that this node and all of its users
        won't be moved into the target device.
        """
        if node.target == "output":
            return not self.allow_outputs

        if not (
            isinstance(node.target, torch._ops.OpOverload)
            and node.target.namespace in ("prims", "aten")
        ):
            return True

        return False

    def get_node_device(self, node: fx.Node) -> Optional[torch.device]:
        """
        Get the device of a node.
        """
        ten = node.meta.get("val")
        return None if not isinstance(ten, torch.Tensor) else ten.device

    def get_cpu_indeg_count(self, graph: fx.Graph) -> Dict[fx.Node, int]:
        """
        Get the number of cpu inputs to a node
        """
        cpu_indeg: Dict[fx.Node, int] = Counter()

        for node in graph.nodes:
            cpu_count = 0

            def add_cpu_inp(node):
                nonlocal cpu_count
                device = self.get_node_device(node)
                cpu_count += device is not None and device.type == "cpu"

            pytree.tree_map_only(fx.Node, add_cpu_inp, (node.args, node.kwargs))

            if cpu_count:
                cpu_indeg[node] = cpu_count

        return cpu_indeg

    def __call__(self, graph: fx.Graph) -> None:
        target_devices = set()
        constructors = []

        for node in graph.nodes:
            device = self.get_node_device(node)
            if device and device.type == self.target:
                target_devices.add(device)

            if not (
                isinstance(node.target, torch._ops.OpOverload)
                and node.target.namespace in ("prims", "aten")
            ):
                continue

            if not torch._subclasses.fake_tensor._is_tensor_constructor(node.target):
                continue

            if not node.kwargs.get("device") == torch.device("cpu"):
                continue

            constructors.append(node)

        # not handling multiple target devices initially
        if not constructors or len(target_devices) != 1:
            return

        movable_constructors = self.find_movable_constructors(graph, constructors)

        for node in movable_constructors:
            kwargs = node.kwargs.copy()
            kwargs["device"] = next(iter(target_devices))
            node.kwargs = kwargs

    def find_movable_constructors(
        self, graph: fx.Graph, constructors: List[fx.Node]
    ) -> Set[fx.Node]:
        """
        Starting from the cpu constructors, iterate through the graph and test that all of their
        downstream uses can safely be moved to cpu.
        """
        cpu_indeg: Dict[fx.Node, int] = self.get_cpu_indeg_count(graph)

        # which constructors cannot be moved to cuda
        cannot_move_to_cuda: Set[fx.Node] = set()

        # For any node in the graph, which constructors does it have a dependency on
        constructor_dependencies: Dict[fx.Node, Set[fx.Node]] = defaultdict(set)

        # if a cpu node has a dependency on two different cpu constructors,
        # then if either constructor cannot be moved to cuda, the other cannot as well.
        # In this case any node with a dependency on one will have a dependency on the other
        equal_constructor_sets: Dict[fx.Node, Set[fx.Node]] = {
            c: {c} for c in constructors
        }

        def make_dependencies_equivalent(
            set1: Set[fx.Node], set2: Set[fx.Node]
        ) -> Set[fx.Node]:
            # could use union find but not worth complexity here
            set1.update(set2)
            for obj in set1:
                equal_constructor_sets[obj] = set1
            return set1

        queue: List[fx.Node] = list(constructors)

        for c in queue:
            constructor_dependencies[c].add(c)

        while queue:
            node = queue.pop()
            dependencies = constructor_dependencies[node]

            for user in node.users:
                if self.cannot_be_moved(user):
                    cannot_move_to_cuda.update(dependencies)
                    break

                # this node was used on a op which takes in multiple devices and output a cuda
                # tensor. we can convert its cpu input to cuda without making further changes
                node_device = self.get_node_device(user)
                if (
                    self.allow_cpu_device(user)
                    and node_device
                    and node_device.type == self.target
                ):
                    del cpu_indeg[user]
                else:
                    # otherwise, we should continue look at its downstream uses
                    cpu_indeg[user] -= 1
                    if cpu_indeg[user] == 0:
                        del cpu_indeg[user]
                        queue.append(user)

                unioned_set = make_dependencies_equivalent(
                    dependencies, constructor_dependencies[user]
                )
                constructor_dependencies[user] = unioned_set

        for node in cpu_indeg:
            if constructor_dependencies[node]:
                cannot_move_to_cuda.update(constructor_dependencies[node])

        all_cannot_move_to_cuda = cannot_move_to_cuda.copy()
        for constructor in cannot_move_to_cuda:
            all_cannot_move_to_cuda.update(equal_constructor_sets[constructor])

        return set(constructors) - all_cannot_move_to_cuda


def move_constructors_to_cuda(graph: fx.Graph) -> None:
    """
    Moves intermediary tensors which are constructed on the cpu to cuda when safe
    """
    ConstructorMoverPass("cuda")(graph)
