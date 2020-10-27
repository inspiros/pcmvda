import torch
import torchsl

from .multiview_helpers import *

__all__ = ['mvda']


# ===========================================================
# Multiview Discriminant Analysis
# ===========================================================
def mvda(Xs, y, y_unique=None, alpha_vc=0, reg_vc=1e-4, C=True):
    if y_unique is None:
        y_unique = torch.unique(y)
    if C:
        return torchsl.ops.mvda(Xs, y, y_unique, alpha_vc, reg_vc)

    options = dict(dtype=Xs[0].dtype, device=Xs[0].device)
    num_views = len(Xs)
    num_components = y.size(0)
    num_samples = num_views * num_components
    num_classes = y_unique.size(0)
    ecs = class_vectors(y, y_unique).to(dtype=options['dtype'])
    y_unique_counts = ecs.sum(1)
    dims = dimensions(Xs)

    W = torch.zeros(num_components, num_components, **options)
    for ci in range(num_classes):
        W += ecs[ci].unsqueeze(1).mm(ecs[ci].unsqueeze(0)).div(num_views * y_unique_counts[ci])
    I = torch.eye(num_components, **options)
    B = torch.ones(num_components, num_components, **options) / num_samples

    Sw = multiview_covariance_matrix(dims,
                                     lambda j, r: Xs[j].t().mm(I - W).mm(Xs[r]) if j == r
                                     else Xs[j].t().mm(- W).mm(Xs[r]),
                                     options)
    Sb = multiview_covariance_matrix(dims,
                                     lambda j, r: Xs[j].t().mm(W - B).mm(Xs[r]),
                                     options)

    if alpha_vc != 0:
        Ps = [Xs[vi].mm(Xs[vi].t()) for vi in range(num_views)]
        Ps = [(Ps[vi].add_(torch.zeros(num_components, num_components, **options)
                           .fill_diagonal_(reg_vc * Ps[vi].trace()))).inverse().mm(Xs[vi]) for vi in range(num_views)]
        Sb += multiview_covariance_matrix(dims,
                                          lambda j, r: 2 * (num_views - 1) * Ps[j].t().mm(Ps[r]) if j == r
                                          else -2 * Ps[j].t().mm(Ps[r]),
                                          options)
    return Sw, Sb
