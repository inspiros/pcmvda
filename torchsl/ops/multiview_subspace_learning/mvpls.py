import torch

import torchsl
from torchsl._extensions import _has_ops
from ._helpers import *

__all__ = ['mvpls']


# ===========================================================
# Multiview Partial Least Squared Regression
# ===========================================================
def mvpls(Xs):
    if _has_ops():
        return torchsl.ops.mvpls(Xs)

    options = dict(dtype=Xs[0].dtype, device=Xs[0].device)
    num_views = len(Xs)
    num_components = Xs[0].size(0)
    num_samples = num_views * num_components
    dims = dimensions(Xs)

    I = torch.eye(num_components, **options)
    B = torch.ones(num_components, num_components, **options) / num_samples

    SI = torch.eye(dims.sum(), **options)
    Sb = multiview_covariance_matrix(dims,
                                     lambda j, r: Xs[j].t().mm(I - B).mm(Xs[r]) if j != r
                                     else 0,
                                     options)

    return SI, Sb
