import torch

import torchsl
from torchsl._extensions import _has_ops
from ._helpers import *

__all__ = ['lda', 'lda_loss']


# ===========================================================
# Linear Discriminant Analysis
# ===========================================================
# noinspection DuplicatedCode
def lda(X, y, y_unique=None):
    if y_unique is None:
        y_unique = torch.unique(y)
    if _has_ops():
        return torchsl.ops.lda(X, y, y_unique)

    options = dict(dtype=X.dtype, device=X.device)
    num_samples = X.size(0)
    num_classes = y_unique.size(0)
    ecs = class_vectors(y, y_unique).to(dtype=options['dtype'])
    y_unique_counts = ecs.sum(1)

    W = torch.zeros(num_samples, num_samples, **options)
    for ci in range(num_classes):
        W += ecs[ci].unsqueeze(1).mm(ecs[ci].unsqueeze(0)).div_(y_unique_counts[ci])
    I = torch.eye(num_samples, **options)
    B = torch.ones(num_samples, num_samples, **options) / num_samples

    Sw = X.t().mm(I - W).mm(X)
    Sb = X.t().mm(W - B).mm(X)

    return Sw, Sb


# noinspection DuplicatedCode
def lda_loss(Y, y, y_unique=None, eps=1e-6):
    if y_unique is None:
        y_unique = torch.unique(y)

    options = dict(dtype=Y.dtype, device=Y.device)
    num_samples = Y.size(0)
    num_classes = y_unique.size(0)
    ecs = class_vectors(y, y_unique).to(dtype=options['dtype'])
    y_unique_counts = ecs.sum(1)

    W = torch.zeros(num_samples, num_samples, **options)
    for ci in range(num_classes):
        W += ecs[ci].unsqueeze(1).mm(ecs[ci].unsqueeze(0)).div_(y_unique_counts[ci])
    I = torch.eye(num_samples, **options)
    B = torch.ones(num_samples, num_samples, **options) / num_samples

    Sw = Y.t().mm(I - W).mm(Y)
    Sb = Y.t().mm(W - B).mm(Y)

    return Sw.trace() / (Sb.trace() + eps)
