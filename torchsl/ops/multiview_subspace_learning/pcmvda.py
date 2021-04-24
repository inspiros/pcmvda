import torch

import torchsl
from torchsl._extensions import _has_ops
from ._helpers import *

__all__ = ['pcmvda_loss', 'fast_pcmvda_loss']


# ===========================================================
# Pairwise-Covariance Multiview Discriminant Analysis
# ===========================================================
# noinspection DuplicatedCode
def pcmvda_loss(Ys, y, y_unique=None, beta=1, q=1):
    if len(set(X.size(1) for X in Ys)) > 1:
        raise ValueError(f"pc-MvDA only works on projected data with same dimensions, "
                         f"got dimensions of {tuple(X.size(1) for X in Ys)}.")
    if y_unique is None:
        y_unique = torch.unique(y)
    if _has_ops():
        return torch.ops.torchsl.pcmvda(Ys, y, y_unique, beta, q)

    Xs_cat = torch.cat(Ys, 0)
    options = dict(dtype=Xs_cat.dtype, device=Xs_cat.device)
    num_views = len(Ys)
    num_components = y.size(0)
    num_classes = y_unique.size(0)
    ecs = class_vectors(y, y_unique).to(dtype=options['dtype'])
    ucs = class_means(Ys, ecs)
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


# noinspection DuplicatedCode
def fast_pcmvda_loss(Ys, y, y_unique=None, beta=1, q=1):
    if len(set(X.size(1) for X in Ys)) > 1:
        raise ValueError(f"pc-MvDA only works on projected data with same dimensions, "
                         f"got dimensions of {tuple(X.size(1) for X in Ys)}.")
    y_unique_present = torch.unique(y)
    if y_unique is None:
        y_unique = y_unique_present
        y_present_mask = torch.zeros_like(y_unique, dtype=torch.bool)
        y_present_mask[y_unique_present.tolist()] = 1
    else:
        y_present_mask = torch.ones_like(y_unique, dtype=torch.bool)
    y_present_mask = torch.where(y_present_mask)[0]

    options = dict(dtype=Ys[0].dtype, device=Ys[0].device)
    num_views = len(Ys)
    num_components = y.size(0)
    num_classes = y_unique.size(0)
    ecs = class_vectors(y, y_unique).to(dtype=options['dtype'])
    ucs = torch.stack(class_means(Ys, ecs))
    us = ucs.mean(0)
    y_unique_counts = ecs.sum(1)
    out_dimension = Ys[0].size(1)

    pairs = torch.combinations(y_unique_present, r=2)
    class_Sw = torch.zeros(num_classes, out_dimension, out_dimension, **options)

    for ci in y_unique_present:
        for vj in range(num_views):
            for k in torch.where(ecs[ci])[0]:
                d_ijk = Ys[vj][k] - us[ci]
                class_Sw[ci] += d_ijk.unsqueeze(1) @ d_ijk.unsqueeze(0)
    Sw = class_Sw[y_present_mask].sum(0)

    out = torch.tensor(0, **options)
    for ca, cb in pairs:
        Sw_ab = beta * (y_unique_counts[ca] * class_Sw[ca] + y_unique_counts[cb] * class_Sw[cb])
        Sw_ab.div_(y_unique_counts[ca] + y_unique_counts[cb]).add_((1 - beta) * Sw)

        du_ab = sum(uc[ca] for uc in ucs).sub(sum(uc[cb] for uc in ucs)).div_(num_views).unsqueeze_(0)
        Sb_ab = du_ab.t().mm(du_ab)
        out += y_unique_counts[ca] * y_unique_counts[cb] * (torch.trace(Sb_ab) / torch.trace(Sw_ab)).pow_(-q)
    out /= num_components * num_components
    return out
