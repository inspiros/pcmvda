import torch
import torchsl

from .singleview_helpers import *

__all__ = ['lda']


# ===========================================================
# Linear Discriminant Analysis
# ===========================================================
def lda(X, y, y_unique=None, C=True):
    if y_unique is None:
        y_unique = torch.unique(y)
    if C:
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
