import torchsl

__all__ = ['euclidean_distances',
           'manhattan_distances',
           'cosine_distances',
           'linear_kernel',
           'polynomial_kernel',
           'sigmoid_kernel',
           'rbf_kernel',
           'laplacian_kernel',
           'cosine_similarity',
           'additive_chi2_kernel',
           'chi2_kernel',
           'neighbors_mask']


def __check_X_Y(X, Y):
    if Y is None:
        Y = X  # TODO: X.clone()?
    return X, Y


def euclidean_distances(X, Y=None):
    X, Y = __check_X_Y(X, Y)
    return torchsl.ops.euclidean_distances(X, Y)


def manhattan_distances(X, Y=None):
    X, Y = __check_X_Y(X, Y)
    return torchsl.ops.manhattan_distances(X, Y)


def cosine_distances(X, Y=None, eps=1e-6):
    X, Y = __check_X_Y(X, Y)
    return torchsl.ops.cosine_distances(X, Y, eps)


def linear_kernel(X, Y=None):
    X, Y = __check_X_Y(X, Y)
    return torchsl.ops.linear_kernel(X, Y)


def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
    X, Y = __check_X_Y(X, Y)
    if gamma is None:
        gamma = 1 / X.size(1)
    return torchsl.ops.polynomial_kernel(X, Y, degree, gamma, coef0)


def sigmoid_kernel(X, Y=None, gamma=None, coef0=1):
    X, Y = __check_X_Y(X, Y)
    if gamma is None:
        gamma = 1 / X.size(1)
    return torchsl.ops.sigmoid_kernel(X, Y, gamma, coef0)


def rbf_kernel(X, Y=None, gamma=None):
    X, Y = __check_X_Y(X, Y)
    if gamma is None:
        gamma = 1 / X.size(1)
    return torchsl.ops.rbf_kernel(X, Y, gamma)


def laplacian_kernel(X, Y=None, gamma=None):
    X, Y = __check_X_Y(X, Y)
    if gamma is None:
        gamma = 1 / X.size(1)
    return torchsl.ops.laplacian_kernel(X, Y, gamma)


def cosine_similarity(X, Y=None, eps=1e-6):
    X, Y = __check_X_Y(X, Y)
    return torchsl.ops.cosine_similarity(X, Y, eps)


def additive_chi2_kernel(X, Y=None):
    X, Y = __check_X_Y(X, Y)
    return torchsl.ops.additive_chi2_kernel(X, Y)


def chi2_kernel(X, Y=None, gamma=1):
    X, Y = __check_X_Y(X, Y)
    return torchsl.ops.chi2_kernel(X, Y, gamma)


def neighbors_mask(X, Y=None, n_neighbors=1):
    X, Y = __check_X_Y(X, Y)
    return torchsl.ops.neighbors_mask(X, Y, n_neighbors)
