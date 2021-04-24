import torch
import torch.nn as nn

__all__ = ['Linear',
           'Linears']


class Linear(nn.Module):

    def __init__(self, in_dimension, out_dimension, bias=False, requires_grad=True):
        super(Linear, self).__init__()
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self._bias = bias
        self._requires_grad = requires_grad
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = nn.Parameter(torch.randn(self.in_dimension, self.out_dimension), requires_grad=self._requires_grad)
        if self._bias:
            self.bias = nn.Parameter(torch.zeros(self.out_dimension), requires_grad=self._requires_grad)
        else:
            self.register_parameter('bias', None)

    def forward(self, X):
        out = torch.mm(X, self.weight)
        if self.bias is not None:
            out += self.bias
        return out

    @property
    def W(self):
        return self.weight

    @W.setter
    def W(self, W, strict=True):
        if strict:
            self.weight.data = W.data
        else:
            self.weight.data = W.data[:self.weight.size(1)]

    def requires_grad_(self, mode=False):
        for param in self.parameters():
            param.requires_grad_(mode)

    def __repr__(self):
        rep = f"{self.__class__.__name__}("
        rep += f"in_dimension={self.in_dimension}"
        rep += f", out_dimension={self.out_dimension}"
        if self.bias is None:
            rep += f", bias=False"
        return rep + ")"


class Linears(nn.Module):

    def __init__(self, in_dimensions, out_dimensions, bias=False, requires_grad=True):
        super(Linears, self).__init__()
        if not hasattr(in_dimensions, '__len__'):
            in_dimensions = list(in_dimensions)
        if not hasattr(out_dimensions, '__len__') or len(out_dimensions) == 1:
            out_dimensions = [out_dimensions for _ in range(len(in_dimensions))]
        self.in_dimensions = in_dimensions
        self.out_dimensions = out_dimensions
        self.num_views = len(in_dimensions)
        self._bias = bias
        self._requires_grad = requires_grad
        self.reset_parameters()

    def reset_parameters(self):
        self.weights = torch.nn.ParameterList(nn.Parameter(torch.randn(in_dim, out_dim), requires_grad=self._requires_grad)
                                              for in_dim, out_dim in zip(self.in_dimensions, self.out_dimensions))
        if self._bias:
            self.biases = torch.nn.ParameterList(nn.Parameter(torch.zeros(out_dim), requires_grad=self._requires_grad)
                                                 for out_dim in self.out_dimensions)
        else:
            self.register_parameter('biases', None)

    def forward(self, Xs):
        outs = [torch.mm(X, w) for X, w in zip(Xs, self.weights)]
        if self.biases is not None:
            outs = [torch.add(out, bias) for out, bias in zip(outs, self.biases)]
        return outs

    @property
    def W(self):
        return torch.cat(list(self.weights))

    @W.setter
    def W(self, W):
        assert sum(w.size(0) for w in self.weights) == W.size(0), f"In-dimensions not match {W.size(0)}"
        for di in range(self.num_views):
            self.weights[di].data = W[sum(self.in_dimensions[:di]):sum(self.in_dimensions[:di + 1]),
                                      :self.weights[di].size(1)]

    @property
    def ws(self):
        return list(self.weights)

    def requires_grad_(self, mode=False):
        self._requires_grad = mode
        for param in self.parameters():
            param.requires_grad_(mode)

    def __repr__(self):
        rep = f"{self.__class__.__name__}("
        rep += f"in_dimensions={self.in_dimensions}"
        rep += f", out_dimensions={self.out_dimensions}"
        if self.biases is None:
            rep += f", bias=False"
        return rep + ")"
