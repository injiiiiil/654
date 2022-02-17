from typing import Tuple
from torch.utils.data import IterDataPipe


class StreamReaderIterDataPipe(IterDataPipe[Tuple[str, bytes]]):
    r"""
    Given IO streams and their label names, yields bytes with label name in a tuple.

    Args:
        datapipe: Iterable DataPipe provides label/URL and byte stream
        chunk: Number of bytes to be read from stream per iteration.
            If ``None``, all bytes will be read util the EOF.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, StreamReader
        >>> from io import StringIO
        >>> dp = IterableWrapper([("alphabet", StringIO("abcde"))])
        >>> list(StreamReader(dp, chunk=1))
        [('alphabet', 'a'), ('alphabet', 'b'), ('alphabet', 'c'), ('alphabet', 'd'), ('alphabet', 'e')]
    """
    def __init__(self, datapipe, chunk=None):
        self.datapipe = datapipe
        self.chunk = chunk

    def __iter__(self):
        for furl, stream in self.datapipe:
            while True:
                d = stream.read(self.chunk)
                if not d:
                    break
                yield (furl, d)
