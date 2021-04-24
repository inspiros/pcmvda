import torch

__all__ = ['dimensions',
           'class_vectors',
           'class_means',
           'unique_counts',
           'multiview_covariance_matrix']


def dimensions(Xs):
    return torch.tensor([X.size(1) for X in Xs], dtype=torch.long)


def class_vectors(y, y_unique):
    return torch.stack([y.eq(cls) for cls in y_unique])


def class_means(Xs, ecs):
    return [ecs @ Xs[vi] / ecs.sum(1).clamp_min_(1).unsqueeze(1) for vi in range(len(Xs))]


def unique_counts(y, y_unique):
    return torch.tensor([y.eq(cls).sum() for cls in y_unique], dtype=torch.long)


def multiview_covariance_matrix(dims,
                                constructor,
                                options=dict(),
                                symmetric=True,
                                **kwargs):
    num_views = dims.size(0)
    indices = torch.arange(num_views, dtype=torch.long)
    diag_indices = indices.repeat(2, 1).t_()
    upper_indices = torch.combinations(indices, r=2)
    diag_and_upper_indices = torch.cat([diag_indices, upper_indices])
    sum_dimensions = dims.sum()

    options.update(kwargs)
    out = torch.empty(sum_dimensions, sum_dimensions, **options)
    for j, r in diag_and_upper_indices:
        j_indices = slice(dims[:j].sum(), dims[:j + 1].sum())
        r_indices = slice(dims[:r].sum(), dims[:r + 1].sum())
        jr_cross_covariance = constructor(j, r)
        if not torch.is_tensor(jr_cross_covariance) or jr_cross_covariance.nelement() == 1:
            jr_cross_covariance = torch.empty(dims[j], dims[r], **options).fill_(jr_cross_covariance)
        elif jr_cross_covariance.nelement() == 0:
            jr_cross_covariance = torch.zeros(dims[j], dims[r], **options)
        out[j_indices, r_indices] = jr_cross_covariance
        if j != r:
            out[r_indices, j_indices] = jr_cross_covariance.t() if symmetric else constructor(r, j)
    return out
