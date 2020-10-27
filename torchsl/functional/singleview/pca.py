import torch
import torchsl

__all__ = ['pca']


# ===========================================================
# Principle Component Analysis
# ===========================================================
def pca(X, C=True):
    if C:
        return torchsl.ops.pca(X)

    options = dict(dtype=X.dtype, device=X.device)
    num_samples = X.size(0)
    dim = X.size(1)
    X_centered = X - X.mean(0)
    cov = X_centered.t().mm(X_centered).div(num_samples - 1)
    SI = torch.eye(dim, **options)

    return SI, cov
