from typing import Callable
from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
from torch.nn import functional as F

CACHE_K_SUM = "CACHE_K_SUM"
CACHE_KV_SUM = "CACHE_KV_SUM"


# feature maps
def elu_feature_map(x: Tensor, inplace: bool = False) -> Tensor:
    return F.elu(x, inplace=inplace) + 1

def relu_feature_map(x: Tensor, inplace: bool = False) -> Tensor:
    return F.relu(x, inplace=inplace)


class LinearAttentionBase(nn.Module, ABC):

    def __init__(self, feature_map: Callable):
        super().__init__()
        self.feature_map = feature_map

        self.register_buffer(CACHE_K_SUM, torch.tensor(0), persistent=False)
        self.register_buffer(CACHE_KV_SUM, torch.tensor(0), persistent=False)

    def reset_cache(self) -> None:
        setattr(self, CACHE_K_SUM, torch.tensor(0))
        setattr(self, CACHE_KV_SUM, torch.tensor(0))

    @abstractmethod
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        pass


class LinearAttention(LinearAttentionBase):

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Perform Linear Attention over a batch of queries and keys.

        Note:
        - Let Sk be the start position of the first token from the keys within the target sequence.
        - Let Tk be the end position of the last token from the keys within the target sequence.
        - For self-attention, the source sequence is the target sequence.

        Each q_i attends to all keys, namely k_0, k_1 ... k_Tk.

        # cached k_sum with shape [b, h, d]
        # cached kv_sum with shape [b, h, d, m]

        :param q: Query vector of shape [B, tq, H, D]
        :param k: Key vector of shape [B, tk, H, D]
        :param v: Value vector of shape [B, tk, H, M]
        :return: output vector of shape [B, tq, H, M]
        """
        q, k = self.feature_map(q), self.feature_map(k)

        # denominator: q_i @ Σ_{j=0..Tk}(k_j)
        k_sum = k.sum(dim=1)  # [b,h,d]: Σ_{j=Sk..Tk} (k_j)
        if not self.training:
            k_sum += getattr(self, CACHE_K_SUM)  # Σ_{j=0..Sk} k_j + Σ_{j=Sk..Tk} k_j
            setattr(self, CACHE_K_SUM, k_sum)
        z = 1 / torch.einsum("bthd, bhd -> bth", q, k_sum)

        # numerator: q_i @ Σ_{j=0..Tk}(k_j v_j)
        kv_sum = torch.einsum("bthd, bthm -> bhdm", k, v)  # K.T @ V = Σ_{j=Sk..Tk} (Kj ⨂ Vj)
        if not self.training:
            kv_sum += getattr(self, CACHE_KV_SUM)  # Σ_{j=0..Sk} (Kj ⨂ Vj) + Σ_{j=Sk..Tk} (Kj ⨂ Vj)
            setattr(self, CACHE_KV_SUM, kv_sum)
        return torch.einsum("bthd, bhdm, bth-> bthm", q, kv_sum, z)


class CausalLinearSelfAttention(LinearAttentionBase):

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Perform Causal Linear Attention over a batch of queries and keys.

        Each q_i attends to all keys, namely k_0, k_1 ... k_i.

        # cached k_sum with shape [b, 1, h, d]
        # cached kv_sum with shape [b, 1, h, d, m]
        # the extra dimension is or easy addition.

        :param q: Query vector of shape [B, Tq, H, D]
        :param k: Key vector of shape [B, Tq, H, D]
        :param v: Value vector of shape [B, Tq, H, M]
        :return: output vector of shape [B, Tq, H, M]
        """
        q, k = self.feature_map(q), self.feature_map(k)

        # denominator: q_i @ Σ_{t=0..i}(k_t)
        k_sum = k.cumsum(dim=1)  # Σ_{j=Sk..i} (k_j)
        if not self.training:
            k_sum += getattr(self, CACHE_K_SUM)  # Σ_{j=0..Sk} k_j + Σ_{j=Sk..i} k_j
            setattr(self, CACHE_K_SUM, k_sum[:, -1:])
        z = 1 / torch.einsum("bthd, bthd -> bth", q, k_sum)

        # numerator: q_i @ Σ_{j=0..i}(k_j v_j)
        kv_sum = torch.einsum("bthd, bthm -> bthdm", k, v).cumsum(dim=-2)  # K.T @ V = Σ_{j=Sk..i} (Kj ⨂ Vj)
        if not self.training:
            kv_sum += getattr(self, CACHE_KV_SUM)  # Σ_{j=0..Sk} (Kj ⨂ Vj) + Σ_{j=Sk..i} (Kj ⨂ Vj)
            setattr(self, CACHE_KV_SUM, kv_sum[:, -1:])
        return torch.einsum("bthd, bthdm, bth-> bthm", q, kv_sum, z)


if __name__ == "__main__":
    B, Tq, H, D = 4, 5, 6, 7
    Tk, M = 9, 10
    q = torch.randn(B, Tq, H, D)
    k = torch.randn(B, Tk, H, D)
    v = torch.randn(B, Tk, H, M)

    attn = LinearAttention(elu_feature_map).eval()
    print(attn(q, k, v).shape)
    print(attn(q, k, v).shape)

    q = torch.randn(B, Tq, H, D)
    k = torch.randn(B, Tq, H, D)
    v = torch.randn(B, Tq, H, M)
    attn = CausalLinearSelfAttention(elu_feature_map).eval()
    print(attn(q, k, v).shape)
    print(attn(q, k, v).shape)
