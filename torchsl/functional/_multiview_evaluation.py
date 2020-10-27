__all__ = ['multiview_average',
           'singleview_average',
           'crossview_average']


def multiview_average(multiview_scores):
    return multiview_scores.mean()


def singleview_average(mutiview_scores):
    return mutiview_scores.diag().mean()


def crossview_average(mutiview_scores):
    return (mutiview_scores.sum() - (mutiview_scores.diag().sum())) / (mutiview_scores.numel() - mutiview_scores.size(0))
