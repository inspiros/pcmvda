import torch
import torchsl

from .multiview_helpers import *

__all__ = ['pcmvda']


# ===========================================================
# Pairwise-Covariance Multiview Discriminant Analysis
# ===========================================================
def pcmvda(Xs, y, y_unique=None, beta=1, q=1, C=True):
    if len(set(X.size(1) for X in Xs)) > 1:
        raise ValueError(f"pc-MvDA only works on projected data with same dimensions, "
                         f"got dimensions of {tuple(X.size(1) for X in Xs)}.")
    if y_unique is None:
        y_unique = torch.unique(y)
    if C:
        return torchsl.ops.pcmvda(Xs, y, y_unique, beta, q)

    Xs_cat = torch.cat(Xs, 0)
    options = dict(dtype=Xs_cat.dtype, device=Xs_cat.device)
    num_views = len(Xs)
    num_components = y.size(0)
    num_classes = y_unique.size(0)
    ecs = class_vectors(y, y_unique).to(dtype=options['dtype'])
    ucs = class_means(Xs, ecs)
    y_unique_counts = ecs.sum(1)
    out_dimension = Xs_cat.size(1)
    covariance_dims = torch.tensor(num_components, dtype=torch.long).repeat(num_views)

    pairs = torch.combinations(torch.arange(num_classes, dtype=torch.long), r=2)

    class_W = torch.empty(num_classes, num_components, num_components, **options)
    class_I = torch.empty(num_classes, num_components, num_components, **options)
    for ci in range(num_classes):
        class_W[ci] = ecs[ci].unsqueeze(0).t().mm(ecs[ci].unsqueeze(0)).div_(num_views * y_unique_counts[ci])
        class_I[ci] = torch.eye(num_components, **options) * ecs[ci]
    W = class_W.sum(dim=0)
    I = torch.eye(num_components, **options)

    class_Sw = torch.empty(num_classes, out_dimension, out_dimension, **options)
    for ci in range(num_classes):
        class_Sw[ci] = Xs_cat.t().mm(multiview_covariance_matrix(covariance_dims,
                                                                 lambda j, r: class_I[ci] - class_W[ci] if j == r
                                                                 else - class_W[ci],
                                                                 options)).mm(Xs_cat)
    Sw = Xs_cat.t().mm(multiview_covariance_matrix(covariance_dims,
                                                   lambda j, r: I - W if j == r else -W,
                                                   options)).mm(Xs_cat)

    out = 0
    for ca, cb in pairs:
        Sw_ab = beta * (y_unique_counts[ca] * class_Sw[ca] + y_unique_counts[cb] * class_Sw[cb])
        Sw_ab.div_(y_unique_counts[ca] + y_unique_counts[cb]).add_((1 - beta) * Sw)

        du_ab = sum(uc[ca] for uc in ucs).sub(sum(uc[cb] for uc in ucs)).div_(num_views).unsqueeze_(0)
        Sb_ab = du_ab.t().mm(du_ab)
        out += y_unique_counts[ca] * y_unique_counts[cb] * (torch.trace(Sb_ab) / torch.trace(Sw_ab)).pow_(-q)
        # out += y_unique_counts[ca] * y_unique_counts[cb] * (du_ab.mm(Sw_ab.inverse()).mm(du_ab.t())).pow_(-q)
    out /= num_components * num_components
    return out
