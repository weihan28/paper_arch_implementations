import torch
from torch import nn
from torch.nn import functional as F

from dataclasses import dataclass, asdict


@dataclass
class MockParams:
    dim = 4096
    ffn_dim_multiplier = 1.3
    multiple_of = 1024
    n_heads = 32
    n_kv_heads = 8
    n_layers = 32
    norm_eps = 1e-05
    rope_theta = 500000.0
    use_scaled_rope = True
    vocab_size = 128256
    max_batch_size = 4
    max_seq_length = 128
    n_kv_head_rep = 4
    ffn_dim = 4 * 4096
    device = torch.device('mps')

    # to add
    n_groups = 4
    n_routed_experts = 16
    n_shared_experts = 2
    moe_inter_dim = ffn_dim // 2
    moe_top_k = 2
    moe_route_scale = 1
    moe_score_func = "softmax"
    n_limited_groups = 2
    moe_use_bias = True


def to_dict(self):
    return asdict(self)


class Expert(nn.Module):
    def __init__(self, dim, inter_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(dim, inter_dim)
        self.w3 = nn.Linear(inter_dim, dim)

    def forward(self, x):
        # swiGLU
        return self.w3(F.silu(self.w1(x) * self.w2(x)))


class Router(nn.Module):
    def __init__(self, params: MockParams):
        super().__init__()
        self.params = params
        self.n_groups = params.n_groups
        self.n_limited_groups = params.n_limited_groups  # only choose from n_limited_groups group by score
        self.k = params.moe_top_k

        self.score_func = params.moe_score_func

        self.w1 = nn.Linear(params.dim, params.n_routed_experts)
        self.bias = nn.Parameter(torch.empty(params.n_routed_experts)) if params.moe_use_bias else None
        self.route_scale = params.moe_route_scale

    def forward(self, x):
        _, _ = x.shape  # [B*T, h]
        # routes each token to their top k experts.
        scores = self.w1(x)  # [B*T, N]

        # generate scores
        if self.score_func == "softmax":
            scores = F.softmax(scores, dim=-1, dtype=torch.float32)  # [B*T, N]
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()  # [B*T, N]
            scores /= scores.sum(dim=-1, keepdim=True)  # normalise to probability
        else:
            raise ValueError(f"Unsupported score function {self.score_func}")

        # add learnt expert bias
        if self.bias is not None:
            scores += self.bias  # [B*T, N]

        # mask non-top-n_limited_groups and set others to -inf
        original_scores = scores
        scores = self._mask_groups(scores, x)  # [B*T, N]

        # extract top k scores
        indices = scores.topk(dim=-1, k=self.k).indices  # [B*T, k]
        values = original_scores.gather(1, indices)  # [B*T, k]
        values *= self.route_scale  # [B*T, k]
        return values.type_as(x), indices  # [B*T, k], [B*T, k]

    def _mask_groups(self, scores, x):
        if self.n_groups > 1:
            # generate group scores
            scores = scores.view(x.shape[0], self.n_groups, -1)  # [B*T, G, N//G]

            if self.bias is not None:
                # max expert score in the group
                group_scores = scores.amax(dim=-1)  # [B*T, G]
            else:
                # sum of top 2 expert scores in the group
                group_scores = scores.topk(2, dim=-1).values.sum(dim=-1)  # [B*T, G]

            # set the scores of the not chosen groups to be -inf
            indices = group_scores.topk(self.n_limited_groups, dim=-1).indices  # [B*T, k_group]
            mask = torch.ones((x.shape[0], self.n_groups), dtype=torch.bool, device=x.device).scatter_(-1, indices,
                                                                                                       False)  # [B*T, G]
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)  # [B*T, G, N//G] -> [B*T, N]
        return scores


class Moe(nn.Module):
    def __init__(self, params: MockParams):
        super().__init__()
        self.dim = params.dim
        self.inter_dim = params.moe_inter_dim
        self.n_routed_experts = params.n_routed_experts
        self.n_shared_experts = params.n_shared_experts

        # experts
        self.routed_experts = nn.ModuleList(Expert(self.dim, self.inter_dim) for _ in range(self.n_routed_experts))
        self.shared_experts = Expert(self.dim, self.n_shared_experts * self.inter_dim)

        self.router = Router(params)

    def forward(self, x):
        shape = x.shape
        x = x.view(-1, self.dim)  # [B*T, h]
        scores, indices = self.router(x)  # [B*T, k]
        y = torch.zeros_like(x)  # [B*T, h]
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()  # [n_routed_experts]

        # fill outputs: should be parallelized in practise
        for i in range(self.n_routed_experts):
            count = counts[i]
            # skip if no tokens are routed to expert
            if count == 0:
                pass
            expert = self.routed_experts[i]
            token_idx, col = torch.where(indices == i)  # row , col
            y[token_idx] += expert(x[token_idx]) * scores[token_idx, col].unsqueeze(
                -1)  # [len(idx), h] * [len(idx), 1], weight output with scores

        # run shared experts
        z = self.shared_experts(x)  # [B*T, h]
        return (y + z).view(shape)  # [B, T, h]


if __name__ == '__main__':
    params = MockParams()
    B, T, h = 2, 8, params.dim
    x = torch.randn(B, T, h).to(params.device)

    moe = Moe(params).to(params.device)
    res = moe(x)
    print(x.shape)
    print(res.shape)
