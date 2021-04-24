import torch

import torchsl
from torchsl._extensions import _has_ops
from ._helpers import *

__all__ = ['mvmda']


# ===========================================================
# Multiview Modular Discriminant Analysis
# ===========================================================
# noinspection DuplicatedCode
def mvmda(Xs, y, y_unique=None):
    if y_unique is None:
        y_unique = torch.unique(y)
    if _has_ops():
        return torchsl.ops.mvmda(Xs, y, y_unique)

    options = dict(dtype=Xs[0].dtype, device=Xs[0].device)
    num_views = len(Xs)
    num_components = y.size(0)
    num_classes = y_unique.size(0)
    ecs = class_vectors(y, y_unique).to(dtype=options['dtype'])
    y_unique_counts = ecs.sum(1)
    dims = dimensions(Xs)

    W = torch.zeros(num_components, num_components, **options)
    J = torch.zeros_like(W)
    B = torch.zeros_like(W)
    for ci in range(num_classes):
        tmp = ecs[ci].unsqueeze(1).mm(ecs[ci].unsqueeze(0)).div(y_unique_counts[ci])
        W += tmp.div(y_unique_counts[ci])
        J += tmp
        del tmp
    W *= num_classes / (num_views * num_views)
    J /= num_views
    for ca in range(num_classes):
        for cb in range(num_classes):
            B += ecs[ca].unsqueeze(1).mm(ecs[cb].unsqueeze(0)).div(y_unique_counts[ca] * y_unique_counts[cb])
    B /= num_views * num_views
    I = torch.eye(num_components, **options)

    Sw = multiview_covariance_matrix(dims,
                                     lambda j, r: Xs[j].t().mm(I - J).mm(Xs[r]) if j == r
                                     else Xs[j].t().mm(-J).mm(Xs[r]),
                                     options)
    Sb = multiview_covariance_matrix(dims,
                                     lambda j, r: Xs[j].t().mm(W - B).mm(Xs[r]),
                                     options)

    return Sw, Sb
