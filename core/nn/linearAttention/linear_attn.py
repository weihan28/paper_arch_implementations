from typing import override

import torch
from torch import nn
from torch.nn import functional as F


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1


class LinearAttention(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.attn.n_heads
        self.qk_dim = args.attn.qk_dim
        self.v_dim = args.attn.v_dim
        self.feature_map = args.attn.feature_map

        self.WQ = nn.Linear(self.dim, self.n_heads * self.qk_dim, bias=False)
        self.WK = nn.Linear(self.dim, self.n_heads * self.qk_dim, bias=False)
        self.WV = nn.Linear(self.dim, self.n_heads * self.v_dim, bias=False)

        self.proj = nn.Linear(self.v_dim, self.dim, bias=False)

    def forward(self, x, start_pos) -> torch.Tensor:
        B, T, _ = x.shape
        # Q @ K^T @ V = [.., T, D] @ [.., D, T] @ [.., T, M]
        q = self.WQ(x).view(B, T, self.n_heads, self.qk_dim)  # [B, T, H, D]
        k = self.WK(x).view(B, T, self.n_heads, self.qk_dim)  # [B, T, H, D]
        v = self.WV(x).view(B, T, self.n_heads, self.v_dim)  # [B, T, H, M]
        q, k = self.feature_map(q), self.feature_map(k)

        kv = torch.einsum("bthd, bthm -> bhdm", k, v)  # K^T @ V
        z = 1 / torch.einsum("bthd, bhd -> bth", q, k.sum(1))  # Σ_d q_t @ (K1+K2..KT)^T = Σ_d (q_t @ Σ_T (Ki^T))
        o = torch.einsum("bthd, bhdm, bth-> bthm", q, kv, z)  # Σ_d (Q_bthd KV_bhdm Z_bth)
        return self.proj(o)  # [B, T, H, e]


class MaskedLinearAttention(LinearAttention):
    def __init__(self, args):
        super().__init__(args)
        self.max_batch_size = args.max_batch_size
        self.max_seq_length = args.max_seq_length

        # denominator cache
        self.register_buffer(
            "cache_k_sum",
            torch.zeros(
                self.max_batch_size,
                1,
                self.n_heads,
                self.qk_dim)
        )

        # numerator cache
        self.register_buffer(
            "cache_kv_sum",
            torch.zeros(
                self.max_batch_size,
                1,
                self.n_heads,
                self.qk_dim,
                self.v_dim)
        )


    @override
    def forward(self, x, start_pos) -> torch.Tensor:
        assert x.ndim == 3, "x must have 3 dimensions, [B,T,D]"
        B, T, _ = x.shape
        q = self.WQ(x).view(B, T, self.n_heads, self.qk_dim)
        k = self.WK(x).view(B, T, self.n_heads, self.qk_dim)
        v = self.WV(x).view(B, T, self.n_heads, self.v_dim)
        q, k = self.feature_map(q), self.feature_map(k)

        # denominator: q_t @ (K1 + K2..Kt)
        k_sum = k.cumsum(dim=1)  # Σ_t (Ki^T)
        k_sum = self.retrieve_cached_k_sum(B, T, k_sum, start_pos) # Σ_1->t (Ki^T) + Σ_t->T (Ki^T)
        z = 1 / torch.einsum("bthd, bthd -> bth", q, k_sum)  # Σ_d (q_t @ Σ_T (Ki^T))

        # numerator: q @ Σ_t(KtVt)
        kv_sum = torch.einsum("bthd, bthm -> bthdm", k, v).cumsum(dim=1)  # kv = Σ_t (KtVt)
        kv_sum = self.retrieve_cached_kv_sum(B, T, kv_sum, start_pos)
        y = torch.einsum("bthd, bthdm, bth -> bthm", q, kv_sum, z)  # (q @ kv) * z = [bthm]
        return self.proj(y)  # [btho]

    def retrieve_cached_kv_sum(self, B, T, kv_sum, start_pos):
        if not self.training:
            # write cache
            kv_sum = self.cache_kv_sum[:B, -1:] + kv_sum
            self.cache_kv_sum[:B, start_pos:start_pos + T] = kv_sum[:, -1:] # [B, 1, H, D]
        return kv_sum

    def retrieve_cached_k_sum(self, B, T, k_sum, start_pos):
        if not self.training:
            # write cache
            k_sum = self.cache_k_sum[:B, -1:] + k_sum  # [B, 1, H, D] + [B, T, H, D] = [B, T, H, D]
            self.cache_k_sum[:B, start_pos:start_pos + T] = k_sum[:, -1:] # [B, 1, H, D]
        return k_sum


if __name__ == "__main__":
    from types import SimpleNamespace

    elu_feature_map = lambda x: F.elu(x) + 1

    args = SimpleNamespace(
        max_batch_size=10,
        max_seq_length=200,
        dim=512,
        attn=SimpleNamespace(
            n_heads=8,
            qk_dim=64,
            v_dim=64,
            feature_map= elu_feature_map,
        )
    )

    start_pos = 0
    attn = MaskedLinearAttention(args).eval()
    b, t = 2, 3
    x = torch.randn(b, t, args.dim)
    out = attn(x, start_pos)
    print(out.shape)
    start_pos += t

    b, t = 2, 2
    x = torch.randn(b, t, args.dim)
    out = attn(x, start_pos)
    print(out.shape)



