import torch
from . import nested
from .codegen import tensorextension

torch.nestedtensor = nested.make_nested_tensor
NestedTensor = nested.NestedTensor

def _create_out(input1, out):
    if out is None:
        out = input1.clone()
    assert len(out) == len(input1)
    return out


def _unary(func, input1, out=None):
    out = _create_out(input1, out)
    for i in range(len(out)):
        # NOTE: We are disabling broadcasting for now
        assert out.tensors[i].size() == input1.tensors[i].size()
        func(input1.tensors[i], out=out.tensors[i])
    return out


# The contract is that func only works with torch.Tensor
def _binary(func, input1, input2, out=None):
    out = _create_out(input1, out)
    assert len(input1) == len(input2)
    for i in range(len(out)):
        # NOTE: We are disabling broadcasting for now
        assert out.tensors[i].size() == input1.tensors[i].size()
        assert input2.tensors[i].size() == input1.tensors[i].size()
        func(input1.tensors[i], input2.tensors[i], out=out.tensors[i])
    return out


def _comparison(func, input1, input2, out=None):
    out = _create_out(input1, out)
    assert len(input1) == len(input2)
    for i in range(len(out)):
        out.tensors[i] = out.tensors[i].to(torch.uint8)
    for i in range(len(out)):
        # NOTE: We are disabling broadcasting for now
        assert out.tensors[i].size() == input1.tensors[i].size()
        assert input2.tensors[i].size() == input1.tensors[i].size()
        func(input1.tensors[i], input2.tensors[i], out=out.tensors[i])
    return out


torch, NestedTensor = tensorextension.add_pointwise_unary_functions(torch, NestedTensor, _unary)
torch, NestedTensor = tensorextension.add_pointwise_binary_functions(torch, NestedTensor, _binary)
torch, NestedTensor = tensorextension.add_pointwise_comparison_functions(torch, NestedTensor, _comparison)

__all__ = ["NestedTensor"]
