import torch
from torch import nn

from core.nn.grouped_query_attn import GroupedQueryAttention
from core.nn.ffn import FeedForwardSwiGLU
from core.norm.rms_norm import RMSNorm
from core.utils.params import ParamsLLama3


class TransformerBlock(nn.Module):
    def __init__(self, params: ParamsLLama3):
        super(TransformerBlock, self).__init__()
        self.attention_norm = RMSNorm(params.dim, params.norm_eps)
        self.attention = GroupedQueryAttention(params)
        self.ffn_norm = RMSNorm(params.dim, params.norm_eps)
        self.feed_forward = FeedForwardSwiGLU(params)

    def forward(self, x, start_pos, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)  # (B, T, dim)
        return h + self.feed_forward(self.ffn_norm(h))  # (B, T, dim)


if __name__ == "__main__":
    from core.utils.params import params_llama3
    from core.utils.device import get_device

    device = get_device()
    params = params_llama3(device)

    dummy_input = torch.randn(2, 8, params.dim, device=params.device)
    dummy_start_pos = 0
    dummy_freqs_cis = torch.randn(8, 64, device=params.device)
    dummy_mask = torch.randn(8, 8, device=params.device)

    transformer = TransformerBlock(params).to(device)
    out = transformer(dummy_input, dummy_start_pos, dummy_freqs_cis, dummy_mask)
    assert dummy_input.shape == out.shape
    print(dummy_input.shape)
    print(out.shape)
