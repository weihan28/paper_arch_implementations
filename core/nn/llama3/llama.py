import torch
from torch import nn

from core.nn.llama3.transformer import TransformerBlock
from core.norm.rms_norm import RMSNorm
from core.pos_embed.rotary_embed import precompute_freqs_cis
from core.utils.params import ParamsLlama3


class LlamaTransformer(nn.Module):
    def __init__(self, params: ParamsLlama3):
        super().__init__()
        self.vocab_size = params.vocab_size
        self.max_seq_length = params.max_seq_length
        self.dim = params.dim
        self.norm_eps = params.rms_norm.norm_eps
        self.head_dim = params.attn.head_dim
        self.rope_theta = params.rope.rope_theta

        self.token_embeddings = nn.Embedding(self.vocab_size, self.dim)
        self.layers = nn.ModuleList()
        for _ in range(params.n_layers):
            self.layers.append(TransformerBlock(params))
        self.norm = RMSNorm(self.dim, self.norm_eps)
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.head_dim,
            2 * self.max_seq_length,
            self.rope_theta
        )

    @torch.inference_mode()
    def forward(self, tokens, start_pos):
        B, T = tokens.shape
        h = self.token_embeddings(tokens)  # [B, T, dim]
        self.freqs_cis = self.freqs_cis.to(tokens.device)
        freqs_cis = self.freqs_cis[start_pos:start_pos + T]

        mask = None
        # because of KV cache, we process only 1 token except for the first run, which has a seq_len>1.
        if T > 1:
            mask = torch.full((T, T), float('-inf'), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).to(tokens.device)  # only get upper triangle, lower triangle all 0.

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)  # [B, T, dim]

        return self.output(self.norm(h))  # [B, T, vocab_size]


if __name__ == "__main__":
    from core.utils.params import ParamsLlama3
    from core.utils.device import get_device

    device = get_device('cpu')
    params = ParamsLlama3(device=device)
    B, T = 2, 8
    # Use rand instead of randn, as nn.Embeddings dont accept neg nums.
    dummy_tokens = torch.rand(B, T, device=params.device).long()
    dummy_start_pos = 0
    llama3 = LlamaTransformer(params).to(params.device)
    output = llama3(dummy_tokens, dummy_start_pos)

    print(dummy_tokens.shape)  # [B, T]
    print(output.shape)  # [B, T, vocab_size]
