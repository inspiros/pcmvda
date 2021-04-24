import torch

import torchsl
from torchsl._extensions import _has_ops
from ._helpers import *

__all__ = ['lfda', 'lfda_lle']


# ===========================================================
# Local Fisher Discriminant Analysis
# ===========================================================
# noinspection DuplicatedCode
def lfda(X, y, y_unique=None):
    if y_unique is None:
        y_unique = torch.unique(y)
    # if _has_ops():
    #     return torchsl.ops.lfda(X, y, y_unique)

    options = dict(dtype=X.dtype, device=X.device)
    num_samples = X.size(0)
    num_classes = y_unique.size(0)
    ecs = class_vectors(y, y_unique).to(dtype=options['dtype'])
    y_unique_counts = ecs.sum(1)

    W = torch.zeros(num_samples, num_samples, **options)
    for ci in range(num_classes):
        W += ecs[ci].unsqueeze(1).mm(ecs[ci].unsqueeze(0)).div_(y_unique_counts[ci])
    W_mask = torch.where(W.bool())
    A = torchsl.rbf_kernel(X)
    W_lw = A * W
    W_lb = 1 / num_samples - W
    W_lb[W_mask] *= A[W_mask]

    # Sw = X.t().mm(I - W).mm(X)
    # Sb = X.t().mm((1 - W.sum(-1)) * I - (1 / num_samples - W) * B).mm(X)
    Sw = X.t().mm(W_lw.sum(-1).diag() - W_lw).mm(X)
    Sb = X.t().mm(W_lb.sum(-1).diag() - W_lb).mm(X)

    return Sw, Sb


# ===========================================================
# Local Fisher Discriminant Analysis with Locally Linear Embedding Affinity Matrix
# ===========================================================
# noinspection DuplicatedCode
def lfda_lle(X):
    # if _has_ops():
    #     return torchsl.ops.lfda_lle(X, y, y_unique)

    options = dict(dtype=X.dtype, device=X.device)
    num_samples = X.size(0)

    W_l = torchsl.rbf_kernel(X).softmax(-1)
    I = torch.eye(num_samples, **options)

    Sw = X.t().mm(I - W_l).mm(X)
    Sb = X.t().mm(W_l - 1 / num_samples).mm(X)

    return Sw, Sb
