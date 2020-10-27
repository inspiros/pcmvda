import torch.nn as nn

__all__ = ['Kernel',
           'Kernels']


def identity(X):
    return X


class Kernel(nn.Module):

    def __init__(self, metric, params_dict=None, landmark=None):
        super(Kernel, self).__init__()
        self.metric = metric if metric is not None else identity
        self.params_dict = dict() if params_dict is None else params_dict
        self.landmark = landmark
        assert callable(self.metric), f"Metric is not callable."
        assert isinstance(self.params_dict, dict), f"Metric parameters must be of type dictionary."

    def forward(self, X):
        return self.metric(X, self.landmark, **self.params_dict)


class Kernels(nn.Module):

    def __init__(self, metrics, params_kwargs=None, landmarks=None):
        super(Kernels, self).__init__()
        if not hasattr(metrics, '__len__'):
            metrics = list(metrics)
        self.metrics = [metric if metric is not None else identity for metric in metrics]
        if params_kwargs is not None:
            self.params_kwargs = [params_dict if params_dict is not None else dict() for params_dict in params_kwargs]
        else:
            self.params_kwargs = [dict() for _ in range(len(self.metrics))]
        self.landmarks = landmarks if landmarks is not None else [None for _ in range(len(self.metrics))]

        if not all(callable(metric) for metric in self.metrics):
            raise AttributeError(f"All metrics must be callable.")
        if not all(isinstance(kwargs, dict) for kwargs in self.params_kwargs):
            raise AttributeError(f"Metric parameters must be of type dictionary, "
                                 f"got {(type(kwargs) for kwargs in self.params_kwargs)}.")

    def forward(self, Xs):
        return [metric(X, Y, **kwargs) for X, Y, metric, kwargs
                in zip(Xs, self.landmarks, self.metrics, self.params_kwargs)]
