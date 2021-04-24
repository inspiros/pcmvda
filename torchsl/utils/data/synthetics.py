import numpy as np
import torch
from scipy.stats import ortho_group
from sklearn.datasets import make_blobs, make_gaussian_quantiles


# noinspection DuplicatedCode
def make_multiview_blobs_old(n_classes=3,
                             n_views=3,
                             n_features=3,
                             n_samples='auto',
                             rotate=True,
                             shuffle=True,
                             seed=None):
    np.random.seed(seed)
    n_samples = n_classes * 20 if n_samples == 'auto' else n_samples
    X_0, y = make_blobs(n_features=n_features, centers=n_classes, n_samples=n_samples, random_state=seed)
    Xs = [X_0]
    for i in range(n_views - 1):
        X_i = X_0 + np.random.randn(n_features) * np.random.randint(1, 3)
        X_i = np.array([x + np.random.normal(0, 1, len(x.shape)) for x in X_i])
        if rotate:
            X_i = X_i @ ortho_group.rvs(n_features, random_state=seed)
        Xs.append(X_i)
    if shuffle:
        indexes = np.random.permutation(np.arange(len(y)))
        y = y[indexes]
        for _ in range(n_views):
            Xs[_] = Xs[_][indexes, :]
    if n_views > 1:
        return [torch.tensor(X).float() for X in Xs], torch.from_numpy(y)
    return torch.tensor(Xs).squeeze(0).float(), torch.from_numpy(y)


# noinspection DuplicatedCode
def make_multiview_blobs(n_classes=3,
                         n_views=3,
                         n_features=3,
                         n_samples='auto',
                         cluster_std=1.0,
                         center_box=(-10.0, 10.0),
                         rotate=True,
                         shuffle=True,
                         seed=None):
    np.random.seed(seed)
    n_samples = n_classes * 20 if n_samples == 'auto' else n_samples
    X_0, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes,
                        cluster_std=cluster_std,
                        center_box=center_box,
                        random_state=seed)
    std = np.std(X_0)
    Xs = [X_0]
    for i in range(n_views - 1):
        X_i = X_0 + np.tile(np.random.normal(loc=0.0, scale=10 * std, size=n_features), (len(X_0), 1))
        X_i += np.random.normal(loc=0.0, scale=std / 100, size=X_i.shape)
        if rotate:
            X_i = X_i @ ortho_group.rvs(n_features, random_state=seed)
        Xs.append(X_i)
    if shuffle:
        indexes = np.random.permutation(np.arange(len(y)))
        y = y[indexes]
        for _ in range(n_views):
            Xs[_] = Xs[_][indexes, :]
    if n_views > 1:
        return [torch.tensor(X).float() for X in Xs], torch.from_numpy(y)
    return torch.tensor(Xs).squeeze(0).float(), torch.from_numpy(y)


# noinspection DuplicatedCode
def make_multiview_dual_blobs(n_classes=2,
                              n_views=3,
                              n_features=3,
                              n_samples='auto',
                              cluster_std=1.0,
                              rotate=True,
                              shuffle=True,
                              seed=None):
    np.random.seed(seed)
    n_samples = n_classes * 20 if n_samples == 'auto' else n_samples
    X_0, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes,
                        cluster_std=cluster_std, random_state=seed)
    X_c0 = X_0[y == 0]
    X_c1 = X_0[y == 1]
    X_c2 = X_c0 + (2 + np.random.normal(loc=0.0, scale=0.1)) * (X_c1.mean(0) - X_c0.mean(0))
    X_c2 += np.random.normal(loc=0.0, scale=np.std(X_c0) / 10, size=X_c2.shape)
    X_0 = np.concatenate((X_0, X_c2))
    y = np.concatenate((y, [0] * len(X_c2)))
    std = np.std(X_0)
    Xs = [X_0]
    for i in range(n_views - 1):
        X_i = X_0 + np.tile(np.random.normal(loc=0.0, scale=10 * std, size=n_features), (len(X_0), 1))
        X_i += np.random.normal(loc=0.0, scale=std / 100, size=X_i.shape)
        if rotate:
            X_i = X_i @ ortho_group.rvs(n_features, random_state=seed)
        Xs.append(X_i)
    if shuffle:
        indexes = np.random.permutation(np.arange(len(y)))
        y = y[indexes]
        for _ in range(n_views):
            Xs[_] = Xs[_][indexes, :]
    if n_views > 1:
        return [torch.tensor(X).float() for X in Xs], torch.from_numpy(y)
    return torch.tensor(Xs).squeeze(0).float(), torch.from_numpy(y)


# noinspection DuplicatedCode
def make_multiview_gaussian_quantiles(n_classes=3,
                                      n_views=3,
                                      n_features=3,
                                      n_samples='auto',
                                      rotate=True,
                                      shuffle=True,
                                      seed=None):
    np.random.seed(seed)
    n_samples = n_classes * 20 if n_samples == 'auto' else n_samples
    X_0, y = make_gaussian_quantiles(n_features=n_features, n_samples=n_samples,
                                     n_classes=n_classes, random_state=seed)
    std = np.std(X_0)
    Xs = [X_0]
    for i in range(n_views - 1):
        X_i = X_0 + np.tile(np.random.normal(loc=0.0, scale=10 * std, size=n_features), (len(X_0), 1))
        X_i += np.random.normal(loc=0.0, scale=std / 100, size=X_i.shape)
        if rotate:
            X_i = X_i @ ortho_group.rvs(n_features)
        Xs.append(X_i)
    if shuffle:
        indexes = np.random.permutation(np.arange(len(y)))
        y = y[indexes]
        for _ in range(n_views):
            Xs[_] = Xs[_][indexes, :]
    if n_views > 1:
        return [torch.tensor(X).float() for X in Xs], torch.from_numpy(y)
    return torch.tensor(Xs).squeeze(0).float(), torch.from_numpy(y)
