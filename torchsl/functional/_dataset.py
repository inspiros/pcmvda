import torchsl

__all__ = ['zero_mean',
           'unit_var',
           'standardize']


def zero_mean(X, X_mean=None):
    return torchsl.ops.zero_mean(X, X.mean(0) if X_mean is None else X_mean)


def unit_var(X, X_std=None):
    return torchsl.ops.unit_var(X, X.std() if X_std is None else X_std)


def standardize(X, X_mean=None, X_std=None):
    return torchsl.ops.standardize(X,
                                   X.mean(0) if X_mean is None else X_mean,
                                   X.std() if X_std is None else X_std)
