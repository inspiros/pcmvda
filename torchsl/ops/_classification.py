import torch

__all__ = ['accuracy_score',
           'confusion_matrix']


def accuracy_score(y_true, y_pred):
    return y_true.eq(y_pred).float().sum() / y_true.nelement()


def confusion_matrix(y_true, y_pred, y_unique=None, normalize=None):
    if y_unique is None:
        y_unique = y_true.unique()
    out = torch.empty(y_unique.nelement(), y_unique.nelement(), dtype=torch.float, device=y_pred.device)
    for i, ci in enumerate(y_unique):
        d = torch.where(y_true.eq(ci))
        for j, cj in enumerate(y_unique):
            out[i, j] = y_pred[d].eq(cj).sum()
    if normalize is None or normalize == 'none':
        out = out.long()
    elif normalize == 'true':
        out /= out.sum(dim=1, keepdim=True)
    elif normalize == 'pred':
        out /= out.sum(dim=0, keepdim=True)
    elif normalize == 'all':
        out /= out.sum()
    else:
        raise AttributeError("normalize must be either 'none', 'true', 'pred' or 'all'")
    return out
