import torch
from torch import nn
from torch.nn import functional as F
from core.utils.params import ParamsLLama3
from core.pos_embed.rotary_embed import apply_rotary_emb


# GQA with KV caching
# Note this is the naive implementation of the KV cache,
# the optimized version is Multi-head Latent Attention introduced in Deepseek V3, after Llama3 was introduced.
class GroupedQueryAttention(nn.Module):
    def __init__(self, params: ParamsLLama3):
        super(GroupedQueryAttention, self).__init__()
        self.params = params
        self.wq = nn.Linear(params.dim, params.n_heads * params.dim_head, bias=False)
        self.wk = nn.Linear(params.dim, params.n_kv_heads * params.dim_head, bias=False)
        self.wv = nn.Linear(params.dim, params.n_kv_heads * params.dim_head, bias=False)

        self.proj = nn.Linear(params.n_heads * params.dim_head, params.dim, bias=False)

        # kv cache
        self.cache_k = torch.zeros(
            (
                params.max_batch_size,
                params.max_seq_length,
                params.n_kv_heads,
                params.dim_head
            )
        )

        self.cache_v = torch.zeros(
            (
                params.max_batch_size,
                params.max_seq_length,
                params.n_kv_heads,
                params.dim_head
            )
        )

    def forward(self, x, start_pos, freqs_cis, mask):
        B, T, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(
            x)  # q -> (B, T, n_head*dim_head) | k, v -> (B, T, n_kv_head*dim_head)

        q = q.view(B, T, self.params.n_heads, self.params.dim_head)
        k = k.view(B, T, self.params.n_kv_heads, self.params.dim_head)
        v = v.view(B, T, self.params.n_kv_heads, self.params.dim_head)

        # apply rotary embedding to only query & key (not values)
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # ensure cache tensors are in the same device the input
        self.cache_k = self.cache_k.to(q.device)
        self.cache_v = self.cache_v.to(q.device)
        # append k & v into cache
        self.cache_k[:B, start_pos: start_pos + T] = k
        self.cache_v[:B, start_pos: start_pos + T] = v
        # retrieve complete k & v from cache for attention
        k = self.cache_k[:B, start_pos: start_pos + T]
        v = self.cache_v[:B, start_pos: start_pos + T]

        # In these runs we duplicate the KV heads for MQA in all GPUs (llama2)
        k = torch.repeat_interleave(
            k, dim=2, repeats=self.params.n_kv_head_rep
        )  # (B, T, n_kv_heads, dim_head) -> (B, T, n_heads, dim_head)
        v = torch.repeat_interleave(
            v, dim=2, repeats=self.params.n_kv_head_rep
        )  # (B, T, n_kv_heads, dim_head) -> (B, T, n_heads, dim_head)

        # Reshaping for scaled_dot_product_attention. expected = (B, ..., T, head_dim)
        q = q.transpose(1, 2)  # (B, T, n_heads, dim_head) -> (B, n_heads, T, dim_head)
        k = k.transpose(1, 2)  # (B, T, n_heads, dim_head) -> (B, n_heads, T, dim_head)
        v = v.transpose(1, 2)  # (B, T, n_heads, dim_head) -> (B, n_heads, T, dim_head)

        # takes care of the entire attention process
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (B, n_heads, T, dim_head)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, n_heads, dim_head) -> (B, T, n_heads*dim_head)
        return self.proj(out)  # (B, T, n_heads*dim_head) -> (B, T, dim)


if __name__ == '__main__':
    from core.utils.params import params_llama3
    from core.utils.device import get_device

    device = get_device()
    params = params_llama3(device)
    params.dim = 4096

    dummy_input = torch.randn(2, 8, params.dim).to(device)  # (B, T, h)
    dummy_start_pos = 0
    dummy_freqs_cis = torch.randn(8, 64).to(device) # (T, dim_head / 2)
    dummy_mask = torch.rand(8, 8).to(device)  # mask is size (T, T)

    attn = GroupedQueryAttention(params).to(device=device)
    output = attn(dummy_input, dummy_start_pos, dummy_freqs_cis, dummy_mask)
    print(dummy_input.shape)  # (B, T, h)
    print(output.shape)  # (B, T, h)
