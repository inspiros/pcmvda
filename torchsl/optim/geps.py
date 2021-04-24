import torch
from torch.optim.optimizer import Optimizer

__all__ = ['GEPSolver']


# noinspection PyMethodOverriding
class GEPSolver(Optimizer):

    def __init__(self, params, solver='svd', eps=1e-5, minimize=False):
        if solver not in ['eig', 'eigen', 'svd']:
            raise ValueError(f"Unknown solver {solver}, expect either eigen or svd.")
        params = list(params)
        if not all(param.ndimension() == 2 for param in params):
            raise ValueError("Invalid dimensions of parameters, must be equal to 2.")
        if len(set(param.size(1) for param in params)) > 1:
            raise ValueError("Invalid output dimensions, must be equal for all parameters.")

        defaults = dict()
        super(GEPSolver, self).__init__(params, defaults)
        self.num_views = len(params)
        self.in_dimensions = [param.size(0) for param in params]
        self.solver = solver
        self.eps = eps
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
        options = dict(dtype=Sw.dtype, device=Sw.device)

        if Sw.eq(0).all():
            Sw = torch.eye(Sw.size(0), **options)
        if Sb.eq(0).all():
            Sb = torch.eye(Sb.size(0), **options)

        if self.solver in ['svd']:
            h, A, H = torch.svd(Sw)
            M = H / (A.sqrt() + self.eps)
            u, EV, U = torch.svd(M.t() @ Sb @ M)
            W = M @ U
            return W[:, torch.argsort(EV, descending=not self.minimize)]
        elif self.solver in ['eig', 'eigen']:
            evals, evecs = torch.eig(self._regularize(Sw).inverse().mm(Sb), eigenvectors=True)
            return evecs[:, torch.argsort(evals[:, 0], descending=not self.minimize)]

    def _regularize(self, tensor):
        return tensor + torch.zeros_like(tensor).fill_diagonal_(self.eps)
