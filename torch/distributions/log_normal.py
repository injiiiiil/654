import torch
from torch.distributions import constraints
from torch.distributions.transforms import ExpTransform
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution


class LogNormal(TransformedDistribution):
    r"""
    Creates a log-normal distribution parameterized by
    :attr:`loc` and :attr:`scale` where::

        X ~ Normal(loc, scale)
        Y = exp(X) ~ LogNormal(loc, scale)

    Example::

        >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of log of distribution
        scale (float or Tensor): standard deviation of log of the distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self._base_dist = Normal(loc, scale)
        super(LogNormal, self).__init__(self._base_dist, ExpTransform(), validate_args=validate_args)

    def expand(self, batch_shape=torch.Size()):
        new = self.__new__(LogNormal)
        new._base_dist = self._base_dist.expand(batch_shape)
        super(LogNormal, new).__init__(new._base_dist, ExpTransform(), validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        return (self.loc + self.scale.pow(2) / 2).exp()

    @property
    def variance(self):
        return (self.scale.pow(2).exp() - 1) * (2 * self.loc + self.scale.pow(2)).exp()

    def entropy(self):
        return self.base_dist.entropy() + self.loc
