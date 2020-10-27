import torch

from scipy.linalg import fractional_matrix_power
from torch.optim.optimizer import Optimizer


class SPGD(Optimizer):

    def __init__(self, params, lr, restore_rate=5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if restore_rate < 1:
            raise ValueError(f"Invalid restore rate: {restore_rate}")

        defaults = dict(lr=lr, restore_rate=restore_rate)
        super(SPGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p.add_(-p.data @ d_p.t() @ p.data)
                n = group['lr'] * p.data.norm(1) / d_p.norm(1)
                p.data.add_(-n, d_p)

                param_state = self.state[p]
                if 'restore_counter' not in param_state:
                    param_state['restore_counter'] = 1
                else:
                    param_state['restore_counter'] += 1
                if param_state['restore_counter'] >= group['restore_rate']:
                    param_state['restore_counter'] -= group['restore_rate']
                    p.data = torch.from_numpy(fractional_matrix_power(p.data.t().mm(p.data),
                                                                      -.5)).type_as(p).to(p.device)
        return loss
