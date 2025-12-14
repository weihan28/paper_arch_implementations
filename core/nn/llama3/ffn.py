from torch import nn
from torch.nn import functional as F
from core.utils.params import ParamsLlama3


# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L202
def get_ffn_dim(hidden_dim, ffn_dim_multiplier, multiple_of):
    hidden_dim = int(2 * hidden_dim / 3)
    # custom dim factor multiplier
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, dim, inter_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(dim, inter_dim, bias=False)
        self.proj = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x):
        return self.proj(F.silu(self.w1(x)) * self.w2(x))


if __name__ == "__main__":
    import torch
    from core.utils.device import get_device

    device = get_device()

    params = ParamsLlama3(device=device)
    dummy_input = torch.randn(2, 8, params.dim)
    inter_dim = get_ffn_dim(params.dim, params.ffn_dim_multiplier, params.multiple_of)
    model = FeedForwardSwiGLU(params.dim, inter_dim)
    print(dummy_input.shape)
    print(model(dummy_input).shape)
