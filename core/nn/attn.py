from torch import nn


class AttentionLayer(nn.Module):

    def __init__(self, qkv_proj, attn):
        super().__init__()
        self.qkv_proj = qkv_proj
        self.attn = attn

    def forward(self, x_q, x_kv=None, mask=None, **kwargs):
        if x_kv is None:
            x_kv = x_q
        q, k, v = self.qkv_proj(x_q, x_kv, **kwargs)
        res = self.attn(q, k, v, mask=mask, **kwargs)
        return self.qkv_proj.out_proj(res)


class NaiveQKVProj(nn.Module):
    def __init__(self, n_heads, source_dim, qk_dim, v_dim, target_dim=None, bias=False):
        super().__init__()
        if target_dim is None:
            target_dim = source_dim

        self.W_q = nn.Linear(source_dim, n_heads * qk_dim, bias=bias)
        self.W_k = nn.Linear(target_dim, n_heads * qk_dim, bias=bias)
        self.W_v = nn.Linear(target_dim, n_heads * v_dim, bias=bias)
        self.W_o = nn.Linear(n_heads * v_dim, source_dim, bias=bias)
        self.n_heads = n_heads

    def out_proj(self, x):
        return self.W_o(x.flatten(start_dim=-2))

    def forward(self, x_q, x_kv, **kwargs):
        q = self.W_q(x_q).view(*x_q.shape[:-1], self.n_heads, -1)
        k = self.W_k(x_kv).view(*x_kv.shape[:-1], self.n_heads, -1)
        v = self.W_v(x_kv).view(*x_kv.shape[:-1], self.n_heads, -1)
        return q, k, v


if __name__ == '__main__':
    import torch
    from core.nn.linearAttention.linear_attn import LinearAttention

    x = torch.randn(2, 3, 64)

    feature_map = lambda x: x

    qkv_proj = NaiveQKVProj(4, 64, 21, 42, None)
    attn = AttentionLayer(qkv_proj, LinearAttention(feature_map))
    print(attn(x).shape)
