import torch

import torchsl
from torchsl._extensions import _has_ops
from ._helpers import *

__all__ = ['lpp']


# ===========================================================
# Locality Preserving Projection
# ===========================================================
# noinspection DuplicatedCode
def lpp(X):
    # if _has_ops():
    #     return torchsl.ops.lpp(X, y, y_unique)

    options = dict(dtype=X.dtype, device=X.device)
    num_samples = X.size(0)

    A = torchsl.rbf_kernel(X)
    W = A
    D = W.sum(-1).diag()

    Sw = X.t().mm(D - W).mm(X)
    Sb = X.t().mm(D).mm(X)

    return Sw, Sb
