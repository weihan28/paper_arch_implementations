import torch
from torch import nn
from torch.nn import functional as F
from core.utils.params import ParamsLlama3
from core.pos_embed.rotary_embed import apply_rotary_emb


# GQA with Standard KV caching
class GroupedQueryAttention(nn.Module):

    def _init_hyper_params(self, params):
        # hyper parameters
        self.dim = params.dim
        self.n_heads = params.attn.n_heads
        self.head_dim = params.attn.head_dim
        self.n_kv_heads = params.attn.n_kv_heads
        self.max_batch_size = params.max_batch_size
        self.max_seq_length = params.max_seq_length
        self.n_kv_head_rep = self.n_heads // self.n_kv_heads

    def __init__(self, params: ParamsLlama3):
        super().__init__()
        self._init_hyper_params(params)

        # weights
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        # kv cache
        self.register_buffer(
            "cache_k",
            torch.zeros(
                self.max_batch_size,
                self.max_seq_length,
                self.n_kv_heads,
                self.head_dim,
            ),
            persistent=False
        )
        self.register_buffer(
            "cache_v",
            torch.zeros(
                self.max_batch_size,
                self.max_seq_length,
                self.n_kv_heads,
                self.head_dim
            ),
            persistent=False
        )

    def forward(self, x, start_pos, freqs_cis, mask):
        B, T, _ = x.shape

        # extract query, keys and values
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        # apply rope
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # kv cache
        k, v = self.extract_full_kv_from_cache(k, v, start_pos)

        # Duplicate the KV heads for MQA in all GPUs (llama2)
        k = torch.repeat_interleave(v, self.n_kv_head_rep, dim=2)  # [B, T, n_heads, dim_head]
        v = torch.repeat_interleave(v, self.n_kv_head_rep, dim=2)  # [B, T, n_heads, dim_head]

        # Reshaping for scaled_dot_product_attention. expected = [B, ..., T, head_dim]
        q = q.transpose(1, 2)  # [B, n_heads, T, dim_head]
        k = k.transpose(1, 2)  # [B, n_heads, T, dim_head]
        v = v.transpose(1, 2)  # [B, n_heads, T, dim_head]

        # Normal Attn
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # [B, n_heads, T, dim_head]
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, n_heads*dim_head]

        return self.proj(out)  # [B, T, dim]

    def extract_full_kv_from_cache(self, k, v, start_pos):
        B, T, _, _ = k.shape
        # write cache
        self.cache_k[:B, start_pos: start_pos + T] = k
        self.cache_v[:B, start_pos: start_pos + T] = v
        # retrieve complete kv from cache
        k = self.cache_k[:B, : start_pos + T]
        v = self.cache_v[:B, : start_pos + T]
        return k, v


if __name__ == '__main__':
    from core.utils.device import get_device

    device = get_device()
    params = ParamsLlama3(
        device=device,
    )

    B, T = 2, 8
    dummy_start_pos = 0
    dummy_input = torch.randn(B, T, params.dim).to(device)
    dummy_freqs_cis = torch.randn(T, params.attn.head_dim // 2).to(device)
    dummy_mask = torch.rand(T, T).to(device)  # mask is size (T, T)

    attn = GroupedQueryAttention(params).to(device)
    output = attn(dummy_input, dummy_start_pos, dummy_freqs_cis, dummy_mask)
    assert dummy_input.shape == output.shape
    print(dummy_input.shape)  # (B, T, h)
    print(output.shape)  # (B, T, h)
