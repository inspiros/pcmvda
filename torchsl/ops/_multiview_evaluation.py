import numpy as np
import torch

__all__ = ['multiview_average',
           'singleview_average',
           'crossview_average']


def multiview_average(multiview_scores):
    if torch.is_tensor(multiview_scores):
        multiview_scores = multiview_scores.cpu().detach().numpy()
    return np.mean(multiview_scores)


def singleview_average(multiview_scores):
    if torch.is_tensor(multiview_scores):
        multiview_scores = multiview_scores.cpu().detach().numpy()
    return np.mean(np.diag(multiview_scores))


def crossview_average(multiview_scores):
    if torch.is_tensor(multiview_scores):
        multiview_scores = multiview_scores.cpu().detach().numpy()
    return (np.sum(multiview_scores) -
            (np.diag(multiview_scores).sum())) / (np.size(multiview_scores) - np.shape(multiview_scores)[0])
