import torch
from torch import nn
from torch.nn import functional as F
from core.pos_embed.rotary_embed import apply_rotary_emb

class MultiHeadLatentAttention(nn.Module):

    def __init__(self, params):
        super(MultiHeadLatentAttention, self).__init__()
        self.params = params

        # Q low rank compression matrices
        self.W_dq = nn.Linear(params.dim, params.q_compress_dim, bias=False)
        self.W_uq = nn.Linear(params.q_compress_dim, params.n_heads * params.dim_head, bias=False)

        # KV low rank compression matrices
        self.W_dkv = nn.Linear(params.dim, params.kv_compress_dim, bias=False)
        self.W_uk = nn.Linear(params.q_compress_dim, params.n_heads * params.dim_head, bias=False)
        self.W_uv = nn.Linear(params.q_compress_dim, params.n_heads * params.dim_head, bias=False)

        # Decoupled rope matrices
        self.W_qr = nn.Linear(params.q_compress_dim, params.n_heads * params.decoupled_dim, bias=False)
        self.W_kr = nn.Linear(params.dim, params.decoupled_dim, bias=False)

        # Output projection matrix
        self.W_o = nn.Linear(params.n_heads * params.dim_head, params.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis, mask):
        # get compressed latent vectors for query & kv
        q_compressed = self.W_dq(x) # [B, T, q_compress_dim]
        kv_compressed = self.W_dkv(x) # [B, T, kv_compress_dim]

        # reshape decoupled query
        q_rope = self.W_qr(q_compressed)  # [B, T, n_head * decoupled_dim]
        q_rope = q_rope.view(B, T, self.params.n_heads, -1)  # [B, T, n_head, decoupled_dim]

        # repeat shared decoupled key for each head
        kv_rope = self.W_kr(x)  # [B, T, decoupled_dim]

        # https://discuss.pytorch.org/t/torch-repeat-and-torch-expand-which-to-use/27969
        kv_rope = kv_rope.unsqueeze(-2).expand(-1, -1, self.params.n_heads, -1)  # [B, T, n_head, decoupled_dim]

        # perform rotational embed to decoupled query & key
        q_rope, kv_rope = apply_rotary_emb(q_rope, kv_rope, freqs_cis) # # [B, T, n_head, decoupled_dim]

        # generate q, k, v from latent vectors
        q = self.W_uq(q_compressed).view(*q_compressed.shape[:2], self.params.n_heads, self.params.dim_head)  # [B, T, n_heads, dim_head]
        k = self.W_uk(kv_compressed).view(*kv_compressed.shape[:2], self.params.n_heads, self.params.dim_head)  # [B, T, n_heads, dim_head]
        v = self.W_uv(kv_compressed).view(*kv_compressed.shape[:2], self.params.n_heads, self.params.dim_head)  # [B, T, n_heads, dim_head]

        # concatenate decoupled rotational embeds
        q = torch.cat((q, q_rope), dim=-1)  # [B, T, n_heads, dim_head + decoupled_dim]
        k = torch.cat((k, kv_rope), dim=-1) # [B, T, n_heads, dim_head + decoupled_dim]

        # attention
        q = q.transpose(1, 2) # [B, n_heads, T, dim_head + decoupled_dim]
        k = k.transpose(1, 2) # [B, n_heads, T, dim_head + decoupled_dim]
        v = v.transpose(1, 2) # [B, n_heads, T, dim_head]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (B, n_heads, T, dim_head)

        # project out
        out = out.transpose(1, 2).contiguous().view(*x.shape[:2], self.params.n_heads * self.params.dim_head) # (B, T, n_heads * dim_head)
        return self.W_o(out) # (B, T, dim)


if __name__ == '__main__':
    import torch
    from core.utils.params import params_llama3
    from core.utils.device import get_device

    device = get_device()
    params = params_llama3(device)
    params.dim = 4096
    # todo: remove after params redesign
    params.q_compress_dim = 4
    params.decoupled_dim = 4
    params.kv_compress_dim = 4
    print(params)

    mla = MultiHeadLatentAttention(params).to(device)

    B, T = 2, 8
    dummy_input = torch.randn(B, T, params.dim).to(device)  # (B, T, h)
    dummy_freqs_cis = torch.randn(T, params.decoupled_dim // 2).to(device) # (T, qr_dim_head / 2)
    dummy_mask = torch.rand(T, T).to(device)  # mask is size (T, T)
    mla.forward(dummy_input, dummy_freqs_cis, dummy_mask)









