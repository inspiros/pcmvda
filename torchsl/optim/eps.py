import torch
from torch.optim.optimizer import Optimizer

from ..msrc import get_engine, init_engine, from_matlab

__all__ = ['EPS']


class EPS(Optimizer):

    def __init__(self, params, solver='eig', reg=1e-4, minimize=False):
        if solver not in ['eig', 'eigen', 'svd', 'matlab', 'ldax']:
            raise ValueError(f"Unknown solver {solver}, expect either eig, svd or matlab.")
        params = list(params)
        if not all(param.ndimension() == 2 for param in params):
            raise ValueError("Invalid dimensions of parameters, must be equal to 2.")
        if len(set(param.size(1) for param in params)) > 1:
            raise ValueError("Invalid output dimensions, must be equal for all parameters.")
        if solver == 'matlab':
            init_engine()

        defaults = dict()
        super(EPS, self).__init__(params, defaults)
        self.num_views = len(params)
        self.in_dimensions = [param.size(0) for param in params]
        self.solver = solver
        self.reg = reg
        self.minimize = minimize

    def step(self, Sw, Sb):
        W = self.solve(Sw, Sb)

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                p.data = W[sum(self.in_dimensions[:i]):sum(self.in_dimensions[:i + 1]), :p.size(1)]

    @torch.no_grad()
    def solve(self, Sw, Sb):
        if Sw.size() != Sb.size():
            raise AttributeError(f"Covariance matrices have incompatible sizes {Sw.size()} and {Sb.size()}.")
        if Sw.size(0) != sum(self.in_dimensions):
            raise AttributeError(f"Covariance matrix dimension ({Sw.size(0)}) must be equals to "
                                 f"sum of parameters input dimensions ({sum(self.in_dimensions)}).")
        dtype, device = Sw.dtype, Sw.device

        if Sw.eq(0).all():
            Sw = torch.eye(Sw.size(0), dtype=dtype, device=device)
        if Sb.eq(0).all():
            Sb = torch.eye(Sb.size(0), dtype=dtype, device=device)

        if self.solver in ['eig', 'eigen']:
            evals, evecs = self._regularize(Sw).inverse().mm(Sb).eig(eigenvectors=True)
            return evecs[:, torch.argsort(evals[:, 0], descending=not self.minimize)]
        elif self.solver in ['svd']:
            U, S, V = self._regularize(Sw).inverse().mm(Sb).svd()
            evecs = U * S
            return evecs.flip([1]) if self.minimize else evecs
        elif self.solver in ['matlab', 'ldax']:
            import matlab
            ret = get_engine().LDAX_SwSb(matlab.single(Sw.cpu().tolist()), matlab.single(Sb.cpu().tolist()))
            evecs = from_matlab(ret, dtype=Sw.dtype, device=Sw.device)
            return evecs.flip([1]) if self.minimize else evecs

    def _regularize(self, tensor):
        if self.reg is None:
            return tensor
        return tensor + torch.zeros_like(tensor).fill_diagonal_(self.reg)
