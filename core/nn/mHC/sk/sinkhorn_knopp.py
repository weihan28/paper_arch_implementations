import torch


def naive_sinkhorn_knopp(x, n_iters=4):
    x = torch.exp(x)
    # A is of shape [..., row=n, col=n]
    for _ in range(n_iters):
        # row normalization
        x = x / x.sum(axis=-2, keepdims=True, dtype=x.dtype)
        # column normalization
        x = x / x.sum(axis=-1, keepdims=True, dtype=x.dtype)
    return x


def sinkhorn_knopp(x, n_iters=4):
    """ Iterative sinkhorn_knopp algorithm.

    Algorithm:
    - scales initialized as one vector (standard summation with weights as 1)
    - col scale = 1 / sum of column weighted by row scales (each element in column is a element in each row)
    - row scale = 1 / sum of row weighted by col scales
    whichever comes first doesn't rly matter.

    Note:
    - the algorithm when scale is 1 is simply the normal sum of the corresponding row/col.
    - x.transpose(-1, -2) makes each row the column of x.
    """
    x = torch.exp(x)  # ensure non-negative
    row_scale = torch.ones(*x.shape[:-2], x.shape[-1], 1, device=x.device, dtype=x.dtype)
    col_scale = torch.ones_like(row_scale)

    for _ in range(n_iters):
        col_scale = 1. / (x.transpose(-1, -2) @ row_scale)  # [n, n] @ [n, 1] = [n, 1]
        row_scale = 1. / (x @ col_scale)  # [n, n] @ [n, 1]

    return row_scale * x * col_scale.transpose(-1, -2)  # [n, 1] * [n,n] * [1, n]


def sinkhorn_knopp_log(x, n_iters=4):
    row_scale = torch.zeros(*x.shape[:-2], x.shape[-1], device=x.device, dtype=x.dtype)
    col_scale = torch.zeros_like(row_scale)

    for _ in range(n_iters):
        col_scale = -torch.logsumexp(x + row_scale.unsqueeze(-1), dim=-2)
        row_scale = -torch.logsumexp(x.transpose(-1, -2) + col_scale.unsqueeze(-1), dim=-2)

    return torch.exp(row_scale.unsqueeze(-1) + x + col_scale.unsqueeze(-2))


if __name__ == '__main__':
    n = 4
    torch.manual_seed(72)
    x = torch.randn((2, 4, n, n), dtype=torch.float)

    y = naive_sinkhorn_knopp(x.clone().detach(), n_iters=4)
    print(y.sum(-2))
    print(y.sum(-1))

    y = sinkhorn_knopp(x.clone().detach(), n_iters=4)
    print(y.sum(-2))
    print(y.sum(-1))

    y = sinkhorn_knopp_log(x.clone().detach(), n_iters=4)
    print(y.sum(-1))
    print(y.sum(-2))
