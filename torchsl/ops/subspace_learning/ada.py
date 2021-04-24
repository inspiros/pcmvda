import torch

import torchsl
from torchsl._extensions import _has_ops
from ._helpers import *

__all__ = ['ada']


# ===========================================================
# Angular Discriminant Analysis
# ===========================================================
# noinspection DuplicatedCode
def ada(X, y, y_unique=None):
    if y_unique is None:
        y_unique = torch.unique(y)
    # if _has_ops():
    #     return torchsl.ops.ada(X, y, y_unique)

    options = dict(dtype=X.dtype, device=X.device)
    num_samples = X.size(0)
    num_classes = y_unique.size(0)
    ecs = class_vectors(y, y_unique).to(dtype=options['dtype'])
    ucs = class_means(X, ecs)
    u = ucs.mean(0)
    y_unique_counts = ecs.sum(1)

    # normalize
    X = X / X.norm(dim=-1, keepdim=True)
    ucs = ucs / ucs.norm(dim=-1, keepdim=True)
    u = u / u.norm()

    Sw = sum(ucs[ci].unsqueeze(1) @ X[ecs[ci].long()].sum(0).unsqueeze(0) for ci in range(num_classes))
    Sb = sum(u.unsqueeze(1) @ ucs[ci].unsqueeze(0) * y_unique_counts[ci] for ci in range(num_classes))
    # print(Sw)
    # print(Sb)
    # exit()

    # W = torch.zeros(num_samples, num_samples, **options)
    # for ci in range(num_classes):
    #     W += ecs[ci].unsqueeze(1).mm(ecs[ci].unsqueeze(0)) * num_samples
    # B = torch.zeros(num_samples, num_samples, **options)
    # for ca in range(num_classes):
    #     for cb in range(num_classes):
    #         B += ecs[ca].unsqueeze(1).mm(ecs[cb].unsqueeze(0))  # .div_(y_unique_counts[ca] * y_unique_counts[cb])

    # Sw = X.t().mm(W).mm(X)
    # Sb = X.t().mm(B).mm(X)

    return Sw, Sb
