import torch
from torch import nn
from torch.nn import functional as F
from core.pos_embed.rotary_embed import apply_rotary_emb_mla


class MultiHeadLatentAttention(nn.Module):

    def __init__(self, params):
        super(MultiHeadLatentAttention, self).__init__()
        self.params = params

        # Q low rank compression matrices
        self.W_dq = nn.Linear(params.dim, params.q_compress_dim, bias=False)
        self.W_uq = nn.Linear(params.q_compress_dim, params.n_heads * params.dim_head, bias=False)
        self.norm_q = nn.RMSNorm(params.q_compress_dim, eps=params.norm_eps)

        # KV low rank compression matrices
        self.W_dkv = nn.Linear(params.dim, params.kv_compress_dim, bias=False)
        self.W_uk = nn.Linear(params.kv_compress_dim, params.n_heads * params.dim_head, bias=False)
        self.W_uv = nn.Linear(params.kv_compress_dim, params.n_heads * params.dim_head, bias=False)
        self.norm_kv = nn.RMSNorm(params.kv_compress_dim, eps=params.norm_eps)

        # Decoupled rope matrices
        self.W_qr = nn.Linear(params.q_compress_dim, params.n_heads * params.decoupled_dim, bias=False)
        self.W_kr = nn.Linear(params.dim, params.decoupled_dim, bias=False)

        # Output projection matrix
        self.W_o = nn.Linear(params.n_heads * params.dim_head, params.dim, bias=False)

        # kv latent cache
        self.register_buffer(
            "cache_kv_compressed",
            torch.zeros(
                params.max_batch_size,
                params.max_seq_length,
                params.kv_compress_dim,
            ),
            persistent=False  # This buffer will NOT be saved into state_dict() checkpoints.
        )

        # decoupled key cache
        self.register_buffer(
            "cache_k_rope",
            torch.zeros(
                params.max_batch_size,
                params.max_seq_length,
                1,  # This is due to my un-optimal implementation
                params.decoupled_dim,
            ),
            persistent=False
        )

    def forward(self, x: torch.Tensor, start_pos, freqs_cis, mask):
        B, t, _ = x.shape

        # get compressed latent vectors for query & kv
        q_compressed = self.W_dq(x)  # [B, t, q_compress_dim]
        kv_compressed = self.W_dkv(x)  # [B, t, kv_compress_dim]

        # normalization after compressed latent vectors
        kv_compressed = self.norm_kv(kv_compressed)
        q_compressed = self.norm_q(q_compressed)

        # reshape decoupled query
        q_rope = self.W_qr(q_compressed)  # [B, t, n_head * decoupled_dim]
        q_rope = q_rope.view(B, t, self.params.n_heads, -1)  # [B, t, n_head, decoupled_dim]
        q_rope = apply_rotary_emb_mla(q_rope, freqs_cis)  # [B, t, n_head, decoupled_dim]

        # repeat shared decoupled key for each head
        k_rope = self.W_kr(x)  # [B, t, decoupled_dim]
        k_rope = apply_rotary_emb_mla(k_rope.unsqueeze(2), freqs_cis)  # [B, t, 1, decoupled_dim]

        if not self.training:
            print("Getting kv from cache")
            # write cache
            self.cache_kv_compressed[:B, start_pos: start_pos + t] = kv_compressed
            self.cache_k_rope[:B, start_pos: start_pos + t] = k_rope

            # retrieve cache
            kv_compressed = self.cache_kv_compressed[:B, : start_pos + t]  # [B, T, 1, decoupled_dim], where T>=t
            k_rope = self.cache_k_rope[:B, : start_pos + t]  # [B, T, 1, decoupled_dim]
            print(f"Current kv_compressed shape: {kv_compressed.shape}")
            print(f"Current k_rope_shape: {k_rope.shape}")

        # https://discuss.pytorch.org/t/torch-repeat-and-torch-expand-which-to-use/27969
        k_rope = k_rope.expand(-1, -1, self.params.n_heads, -1)  # [B, T, n_head, decoupled_dim]

        # generate q, k, v from latent vectors
        q = self.W_uq(q_compressed).view(*q_compressed.shape[:2], self.params.n_heads,
                                         self.params.dim_head)  # [B, t, n_heads, dim_head]
        k = self.W_uk(kv_compressed).view(*kv_compressed.shape[:2], self.params.n_heads,
                                          self.params.dim_head)  # [B, T, n_heads, dim_head]
        v = self.W_uv(kv_compressed).view(*kv_compressed.shape[:2], self.params.n_heads,
                                          self.params.dim_head)  # [B, T, n_heads, dim_head]

        # concatenate decoupled rotational embeds
        q = torch.cat((q, q_rope), dim=-1)  # [B, T, n_heads, dim_head + decoupled_dim]
        k = torch.cat((k, k_rope), dim=-1)  # [B, T, n_heads, dim_head + decoupled_dim]

        # attention
        q = q.transpose(1, 2)  # [B, n_heads, T, dim_head + decoupled_dim]
        k = k.transpose(1, 2)  # [B, n_heads, T, dim_head + decoupled_dim]
        v = v.transpose(1, 2)  # [B, n_heads, T, dim_head]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (B, n_heads, T, dim_head)

        # project out
        out = out.transpose(1, 2).contiguous().view(*x.shape[:2],
                                                    self.params.n_heads * self.params.dim_head)  # (B, T, n_heads * dim_head)
        return self.W_o(out)  # (B, T, dim)


if __name__ == '__main__':
    import torch
    from core.utils.params import params_llama3
    from core.utils.device import get_device
    from core.pos_embed.rotary_embed import precompute_freqs_cis_mla

    device = get_device()
    params = params_llama3(device)
    params.dim = 4096
    # todo: remove after params redesign
    params.q_compress_dim = 4
    params.decoupled_dim = 4
    params.kv_compress_dim = 4
    # mla
    params.q_lora_rank = 0
    params.kv_lora_rank = params.kv_compress_dim
    params.qk_nope_head_dim = params.q_compress_dim
    params.qk_rope_head_dim = params.decoupled_dim
    params.v_head_dim = params.dim_head
    # rope
    params.original_seq_len: int = 4096
    params.rope_theta: float = 10000.0
    params.rope_factor: float = 40
    params.beta_fast: int = 32
    params.beta_slow: int = 1
    params.mscale: float = 1.
    params.max_seq_len = params.max_seq_length
    print(params)

    mla = MultiHeadLatentAttention(params).to(device)
    mla.eval()  # inference mode

    freqs_cis = precompute_freqs_cis_mla(params).to(device)
    print("Precomputed freqs_cis shape:", freqs_cis.shape)

    # Initial Input

    B, T = 2, 8
    dummy_start_pos = 0
    dummy_input = torch.randn(B, T, params.dim).to(device)  # (B, T, h)
    dummy_mask = torch.rand(T, T).to(device)  # mask is size (T, T)
    dummy_freqs_cis = freqs_cis[dummy_start_pos: dummy_start_pos + T]

    print("\nProcess Initial Input of shape: ", dummy_input.shape)
    print("Fetched freqs_cis shape:", dummy_freqs_cis.shape)
    out = mla.forward(dummy_input, dummy_start_pos, dummy_freqs_cis, dummy_mask)
    print("Output shape: ", out.shape)
    dummy_start_pos += T

    # next token (mla kv cache inference)
    T = 1
    dummy_next_input = torch.randn(B, T, params.dim).to(device)  # (B, 1, h)
    dummy_freqs_cis = freqs_cis[dummy_start_pos: dummy_start_pos + T]
    print("\nProcess next token of shape: ", dummy_next_input.shape)
    print("fetched freqs_cis shape:", dummy_freqs_cis.shape)
    out = mla.forward(dummy_next_input, dummy_start_pos, dummy_freqs_cis, None)  # no mask used when seq length is 1
    print("Output shape: ", out.shape)
