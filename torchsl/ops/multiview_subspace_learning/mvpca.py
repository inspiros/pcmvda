import torch

import torchsl
from torchsl._extensions import _has_ops
from ._helpers import *

__all__ = ['mvpca']


# ===========================================================
# Multiview Principal Componant Analysis
# ===========================================================
def mvpca(Xs, cross_covariance=True):
    if cross_covariance and len(set(X.size(1) for X in Xs)) > 1:
        raise ValueError(f"MvPCA with cross covariance only works with multiview data with "
                         f"same dimensions, got dimensions of {tuple(X.size(1) for X in Xs)}")
    if _has_ops():
        return torchsl.ops.mvpca(Xs, cross_covariance)

    options = dict(dtype=Xs[0].dtype, device=Xs[0].device)
    num_components = Xs[0].size(0)
    dims = dimensions(Xs)

    Xs_centered = [X - X.mean(0) for X in Xs]
    if cross_covariance:
        cov = multiview_covariance_matrix(dims,
                                          lambda j, r: Xs_centered[j].t().mm(Xs_centered[r]),
                                          options)
    else:
        cov = multiview_covariance_matrix(dims,
                                          lambda j, r: Xs_centered[j].t().mm(Xs_centered[r])
                                          if j == r else 0,
                                          options)
    cov /= num_components - 1
    SI = torch.eye(sum(dims), **options)

    return SI, cov
