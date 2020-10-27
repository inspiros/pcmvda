import torch
import torchsl

from .singleview_helpers import *

__all__ = ['pclda']


# ===========================================================
# Pairwise-Covariance Linear Discriminant Analysis
# ===========================================================
def pclda(X, y, y_unique=None, beta=1, q=1, C=True):
    if y_unique is None:
        y_unique = torch.unique(y)
    if C:
        return torchsl.ops.pclda(X, y, y_unique, beta, q)

    options = dict(dtype=X.dtype, device=X.device)
    num_samples = y.size(0)
    num_classes = y_unique.size(0)
    ecs = class_vectors(y, y_unique).to(dtype=options['dtype'])
    ucs = class_means(X, ecs)
    y_unique_counts = ecs.sum(1)
    out_dimension = X.size(1)

    pairs = torch.combinations(torch.arange(num_classes, dtype=torch.long), r=2)

    class_W = torch.empty(num_classes, num_samples, num_samples, **options)
    class_I = torch.empty(num_classes, num_samples, num_samples, **options)
    for ci in range(num_classes):
        class_W[ci] = ecs[ci].unsqueeze(0).t().mm(ecs[ci].unsqueeze(0)).div_(y_unique_counts[ci])
        class_I[ci] = torch.eye(num_samples, **options) * ecs[ci]
    W = class_W.sum(dim=0)
    I = torch.eye(num_samples, **options)

    class_Sw = torch.empty(num_classes, out_dimension, out_dimension, **options)
    for ci in range(num_classes):
        class_Sw[ci] = X.t().mm(class_I[ci] - class_W[ci]).mm(X)
    Sw = X.t().mm(I - W).mm(X)

    out = 0
    for ca, cb in pairs:
        Sw_ab = beta * (y_unique_counts[ca] * class_Sw[ca] + y_unique_counts[cb] * class_Sw[cb])
        Sw_ab.div_(y_unique_counts[ca] + y_unique_counts[cb]).add_((1 - beta) * Sw)

        du_ab = ucs[ca].sub(ucs[cb]).unsqueeze_(0)
        # Sb_ab = du_ab.t().mm(du_ab)
        # out += y_unique_counts[ca] * y_unique_counts[cb] * (torch.trace(Sb_ab) / torch.trace(Sw_ab)).pow_(-q)
        out += y_unique_counts[ca] * y_unique_counts[cb] * (du_ab.mm(Sw_ab.inverse()).mm(du_ab.t())).pow_(-q)
    out /= num_samples * num_samples
    return out
