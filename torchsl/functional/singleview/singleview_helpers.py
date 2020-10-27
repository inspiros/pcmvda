import torch

__all__ = ['centerize',
           'class_vectors',
           'class_means',
           'unique_counts']


def centerize(X):
    return X - X.mean(0)


def class_vectors(y, y_unique):
    return torch.stack([y.eq(cls) for cls in y_unique])


def class_means(X, ecs):
    return ecs @ X / ecs.sum(1).unsqueeze(1)


def unique_counts(y, y_unique):
    return torch.tensor([y.eq(cls).sum() for cls in y_unique], dtype=torch.long)
