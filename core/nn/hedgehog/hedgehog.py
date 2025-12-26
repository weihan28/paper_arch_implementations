from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from core.nn.attn import AttentionLayer


# Attention Map and Loss Generation
def softmax_attention(q, k):
    scale = q.shape[-1] ** 0.5
    qk = torch.einsum("bthd, bThd -> bthT", q, k) / scale
    return torch.softmax(qk, dim=-1)  # c_il = exp(q_i @ k_j.T / scale) / Σ_{j=0..Tk} exp(q_i @ k_j.T / scale)


def quadratic_linear_attn(phi_q, phi_k):
    qk = torch.einsum("bthd, bThd-> bthT", phi_q, phi_k)
    return qk / qk.sum(dim=-1, keepdim=True)  # c_it = Φ(q_i) Φ(k_j.T) / Σ_{j=0..Tk} Φ(q_i) Φ(k_j.T))


def compute_hedgehog_loss(q, k, hhq_feature_map, hhk_feature_map):
    true_attn = softmax_attention(q, k)
    hh_attn = quadratic_linear_attn(hhq_feature_map(q), hhk_feature_map(k))  # bthT
    return soft_label_cross_entropy(hh_attn, true_attn)


def soft_label_cross_entropy(hh_attn, true_attn):
    loss = -(true_attn * torch.log(hh_attn + 1e-9)).sum(dim=-1)  # bth (loss calculated for each qi)
    return loss.mean()  # return a single value loss


# Learnable Feature Map
class HedgehogFeatureMap(nn.Module):
    def __init__(self, head_dim, act: Callable):
        super().__init__()
        self.act = act if act is not None else torch.exp
        self.ffn = nn.Linear(head_dim, head_dim)
        self._init_weights()

    def _init_weights(self):
        # set ffn as identity matrix.
        nn.init.eye_(self.ffn.weight)
        nn.init.zeros_(self.ffn.bias)

    def forward(self, x):
        x = self.ffn(x)
        return torch.cat([self.act(x), self.act(-x)], dim=-1)


# Hedgehog Attention Module (Wraps around an attention module)
class HedgehogAttention(nn.Module):
    def __init__(self, original_attn: AttentionLayer, qk_dim, training=True, act=None):
        super().__init__()
        self.fm_q = HedgehogFeatureMap(qk_dim, act=act)
        self.fm_k = HedgehogFeatureMap(qk_dim, act=act)
        self.training = training

        # freeze original params
        self.base_attn = original_attn
        for p in self.base_attn.parameters():
            p.requires_grad = False

        self.true_attn_map = None
        self.pred_attn_map = None

        self._register_hook()

    def forward(self, xq, **kwargs):
        out = self.base_attn(xq, **kwargs)
        return out

    def _register_hook(self):
        # using hook and recalculating the true_attn_map as base_attn currently doesn't support attn map output.
        def hook(module, inp, outp):
            q, k, v = outp[0], outp[1], outp[2]
            if self.training:
                self.true_attn_map = softmax_attention(q, k)
                self.pred_attn_map = quadratic_linear_attn(self.fm_q(q), self.fm_k(k))
        self.base_attn.qkv_proj.register_forward_hook(hook)


if __name__ == '__main__':
    from core.nn.attn import NaiveQKVProj

    def standard_attn(q, k, v, **kwargs):
        return F.scaled_dot_product_attention(q, k, v)

    attn_layer = AttentionLayer(
        attn=standard_attn,
        qkv_proj=NaiveQKVProj(2, 128, 64, 64),
    )

    x = torch.randn(3, 24, 128)
    print(attn_layer)

    # replace atn_layer with hedgehog attention
    attn_layer = HedgehogAttention(attn_layer, 64)
    print(attn_layer)
    print(attn_layer(x).shape)
    print(attn_layer.true_attn_map.shape)
    print(attn_layer.pred_attn_map.shape)
    loss = soft_label_cross_entropy(attn_layer.pred_attn_map, attn_layer.true_attn_map)
    print(loss)







