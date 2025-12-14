<img src="attachments/mla_2.png" width="500"><br>

papers (v2&v3 use the same architecture):

- deepseekv2: https://arxiv.org/pdf/2405.04434
- deepseekv3: https://arxiv.org/pdf/2412.19437
- deepseekMoe: https://arxiv.org/pdf/2401.06066

# Multi-head Latent Attention (MLA)

source(Deepseek v2): https://arxiv.org/pdf/2405.04434

<img src="attachments/mla_1.png" width="500"><br>
instead of caching the keys and values, we cache a compressed latent vector during inference.
We then generate the Keys and Values during runtime.

<img src="attachments/mla_excalidraw.png" width="1000"><br>

Hyper params from deepseek v2:<br>
<img src="attachments/mla_3.png" width="500"><br>

Cache Size:<br>
<img src="attachments/mla_4.png" width="500"><br>

# Deepseek Mixture Of Experts (MOE):<br>

<img src="attachments/moe_1.png" width="500"><br>

official github: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L664

#### Expert:<br>

- each expert is a standard swiGLU FFN.

#### Router:<br>

- maps each token to k routed experts.

softmax or sigmoid is used for the scores:<br>

```python
 _, _ = x.shape  # [B*T, h]
scores = self.w1(x)  # [B*T, N]
if self.score_func == "softmax":
    scores = F.softmax(scores, dim=-1, dtype=torch.float32)  # [B*T, N]
elif self.score_func == "sigmoid":
    scores = scores.sigmoid()  # [B*T, N]
    scores /= scores.sum(dim=-1, keepdim=True)  # normalise to probability
else:
    raise ValueError(f"Unsupported score function {self.score_func}")
```

a learnable expert level bias is added:<br>

- this is not added before the softmax as it is supposed to push (bias) the scores to other experts.

```python
self.bias = nn.Parameter(torch.empty(params.n_routed_experts)) if params.moe_use_bias else None
...

# add learnt expert bias
if self.bias is not None:
    scores += self.bias  # [B*T, N]
```

if experts are partitioned to groups, top_k_grp groups are chosen and the rest are muted.

- if bias is used, amax cant be used as the bias may push one specific expert to dominate.

```python
def _mask_groups(self, scores, x):
    if self.n_groups > 1:
        # generate group scores
        scores = scores.view(x.shape[0], self.n_groups, -1)  # [B*T, G, N//G]

        if self.bias is not None:
            # max expert score in the group
            group_scores = scores.amax(dim=-1)  # [B*T, G]
        else:
            # sum of top 2 expert scores in the group
            group_scores = scores.topk(2, dim=-1).values.sum(dim=-1)  # [B*T, G]

        # set the scores of the not chosen groups to be -inf
        indices = group_scores.topk(self.n_limited_groups, dim=-1).indices  # [B*T, k_group]
        mask = torch.ones((x.shape[0], self.n_groups), dtype=torch.bool, device=x.device).scatter_(-1, indices,
                                                                                                   False)  # [B*T, G]
        scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)  # [B*T, G, N//G] -> [B*T, N]
    return scores
```

the final top k experts and their scores are chosen.

```python
# extract top k scores
indices = scores.topk(dim=-1, k=self.k).indices  # [B*T, k]
values = original_scores.gather(1, indices)  # [B*T, k]
values *= self.route_scale  # [B*T, k]
return values.type_as(x), indices  # [B*T, k], [B*T, k]
```




