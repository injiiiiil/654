import ctypes

lib = None

__all__ = ['range_push', 'range_pop', 'mark']


def _libnvToolsExt():
    global lib
    if lib is None:
        lib = ctypes.cdll.LoadLibrary(None)
        lib.nvtxMarkA.restype = None
    return lib


def range_push(msg):
    """
    Pushes a range onto a stack of nested range span.  Returns zero-based
    depth of the range that is started.

    Arguments:
        msg (string): ASCII message to associate with range
    """
    if _libnvToolsExt() is None:
        raise RuntimeError('Unable to load nvToolsExt library')
    return lib.nvtxRangePushA(ctypes.c_char_p(msg.encode("ascii")))


def range_pop():
    """
    Pops a range off of a stack of nested range spans.  Returns the
    zero-based depth of the range that is ended.
    """
    if _libnvToolsExt() is None:
        raise RuntimeError('Unable to load nvToolsExt library')
    return lib.nvtxRangePop()


def mark(msg):
    """
    Describe an instantaneous event that occurred at some point.

    Arguments:
        msg (string): ASCII message to associate with the event.
    """
    if _libnvToolsExt() is None:
        raise RuntimeError('Unable to load nvToolsExt library')
    return lib.nvtxMarkA(ctypes.c_char_p(msg.encode("ascii")))

class nvtxEventAttributes_t(ctypes.Structure):
    """
    A C struct containing essential attributes and optional
    attributes about a CUDA event. The optional attributes currently
    implemented are 'msg' and 'color'.

    The data types of the fields are based on cupy's implementation.
    """
    _fields_ = = [('version', ctypes.c_ushort),
                  ('size', ctypes.c_ushort),
                  ('color', ctypes.c_char_p),
                  ('msg', ctypes.c_char_p)
                 ]
