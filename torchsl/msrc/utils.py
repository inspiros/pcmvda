import torch

__all__ = ['from_matlab']


def from_matlab(mat, dtype=None, device=None, requires_grad=False):
    return torch.tensor(mat._data,
                        dtype=dtype,
                        device=device,
                        requires_grad=requires_grad).view(mat.size).t().contiguous()
