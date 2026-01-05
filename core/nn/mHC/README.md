paper:

- https://arxiv.org/pdf/2512.24880

<img src="attachments/mhc1.png" width="700">

# Architecture

- dynamic hyper-connection variant (H is "dynamically" generated and not a pre-learnt weight.)
<br>
<img src="attachments/mhc2.png" width="700">
<br>
# Problems of HC architecture.

<img src="attachments/mhc3.png" width="700">

# Solution

<img src="attachments/mhc4.png" width="700">

Note: it is not closed under hadamard product.
# Deepseek mHC hyperparameters

<b>
uses the same modules as deekseek-v3:
<br> — Moe (Mixture of Expert with top_k routing)
<br> — MLA (Multi-head Latent Attention)

The only difference is that the residual network is replaced
with the mHC block.

<img src="attachments/mhc5.png" width="700">

