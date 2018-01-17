class Bijector:
    """
    Abstract class `Bijector`. `Bijector` are bijective transformations with computable
    inverse log det jacobians. They are meant for use in `TransformedDistribution`.
    """

    def __init__(self):
        self.add_inverse_to_cache = False
        self._intermediates_cache = {}

    def __call__(self, value):
        """
        Invokes the bijection x=>y
        """
        return self.forward(value)

    def forward(self, value):
        """
        Invokes the bijection x=>y
        """
        raise NotImplementedError()

    def inverse(self, value):
        """
        Inverts the bijection y => x.
        """
        if self.add_inverse_to_cache:
            if (value, 'x') in self._intermediates_cache:
                x = self._intermediates_cache.pop((value, 'x'))
                return x
            else:
                print(self._intermediates_cache)
                raise KeyError("Key {} in wasn't found in intermediates cache of {}".format(
                    (value, 'x'),
                    self.__class__.__name__))

        raise NotImplementedError()

    def inverse_log_det_jacobian(self, value):
        """
        Computes the inverse log det jacobian `log |dx/dy|`
        """
        raise NotImplementedError()

    def _add_intermediate_to_cache(self, intermediate, y, name):
        """
        Internal function used to cache intermediate results computed during the forward call
        """
        if (y, name) in self._intermediates_cache:
            raise ValueError("Key collision in _add_intermediate_to_cache. Key:{}".format((y, name)))
        self._intermediates_cache[(y, name)] = intermediate


class ExpBijector(Bijector):
    """
    Bijector for the mapping y = exp(x)
    """

    def __init__(self):
        super(ExpBijector, self).__init__()

    def forward(self, value):
        return value.exp()

    def inverse(self, value):
        return value.log()

    def inverse_log_det_jacobian(self, value):
        return -value.log()


__all__ = [
    'Bijector',
    'ExpBijector',
]
