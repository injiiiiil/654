# torch.ao is a package with a lot of interdependencies.
# We will use lazy import to avoid cyclic dependencies here.

import typing

if typing.TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features.
    from . import nn, ns, quantization, pruning

__all__ = [
    "nn",
    "ns",
    "quantization",
    "pruning",
]

def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
