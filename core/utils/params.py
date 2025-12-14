from dataclasses import dataclass, field, asdict
from typing import Optional

import torch


@dataclass
class Common:
    dim: int = 4096
    vocab_size = 128256
    max_batch_size = 4
    max_seq_length = 128
    device: Optional[torch.device] = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Warning: using default device ({self.device}) as device was not provided")
        else:
            print(f"Current device: {self.device}")

    def to_dict(self):
        return asdict(self)


# RMS Norm
@dataclass
class RMSNorm:
    norm_eps = 1e-7


@dataclass
class Transformer(Common):
    n_layers: int = 4
    rms_norm: Optional[RMSNorm] = field(default_factory=RMSNorm)


# Standard Attention
@dataclass
class Attn:
    n_heads: int = 32
    head_dim: int = 128


# Grouped Query Attention
@dataclass
class GQA(Attn):
    n_kv_heads: int = 8


# Multi-head Latent Attention
@dataclass
class MLA(Attn):
    kv_compress_dim = 4
    q_compress_dim = 4
    decoupled_dim = 4


@dataclass
class FFN:
    ffn_dim: int = 4096


@dataclass
class Moe(FFN):
    n_groups = 4
    n_limited_groups = 2
    n_routed_experts = 16
    n_shared_experts = 2
    inter_dim: Optional[int] = None
    k = 2
    route_scale = 1
    score_func = "softmax"
    use_bias = True

    def __post_init__(self):
        if self.inter_dim is None:
            self.inter_dim = self.ffn_dim // 2


# ROPE positional embedding
@dataclass
class Rope:
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


@dataclass
class ParamsLlama3(Transformer):
    ffn_dim_multiplier: int = 1.3
    multiple_of: int = 1024
    attn: GQA = field(default_factory=GQA)
    rope: Rope = field(default_factory=Rope)


@dataclass
class DeepSeekV2(Transformer):
    attn: MLA = field(default_factory=MLA)
    rope: Rope = field(default_factory=Rope)
    moe: Moe = field(default_factory=Moe)
