import torch
from torch import nn

from core.nn.llama3.grouped_query_attn import GroupedQueryAttention
from core.nn.llama3.ffn import FeedForwardSwiGLU, get_ffn_dim
from core.norm.rms_norm import RMSNorm
from core.utils.params import ParamsLlama3


class TransformerBlock(nn.Module):
    def __init__(self, params: ParamsLlama3):
        super().__init__()
        self.dim = params.dim
        self.norm_eps = params.rms_norm.norm_eps
        self.ffn_dim = get_ffn_dim(params.dim, params.ffn_dim_multiplier, params.multiple_of)

        self.attn = GroupedQueryAttention(params)
        self.ffn = FeedForwardSwiGLU(self.dim, self.ffn_dim)
        self.attn_norm = RMSNorm(self.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(self.dim, self.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask):
        h = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)  # (B, T, dim)
        return h + self.ffn(self.ffn_norm(h))  # (B, T, dim)


if __name__ == "__main__":
    from core.utils.params import ParamsLlama3
    from core.utils.device import get_device

    device = get_device()
    params = ParamsLlama3(device=device)
    B, T = 2, 8
    dummy_input = torch.randn(B, T, params.dim, device=params.device)
    dummy_start_pos = 0
    dummy_freqs_cis = torch.randn(T, params.attn.head_dim // 2, device=params.device)
    dummy_mask = torch.randn(T, T, device=params.device)

    transformer = TransformerBlock(params).to(device)
    out = transformer(dummy_input, dummy_start_pos, dummy_freqs_cis, dummy_mask)
    assert dummy_input.shape == out.shape
    print(dummy_input.shape)
    print(out.shape)
