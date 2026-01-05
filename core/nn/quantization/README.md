<br>Official torchao intro: https://docs.pytorch.org/ao/stable/quantization_overview.html#

Github:
<br>quantization: https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md
<br>qat: https://github.com/pytorch/ao/tree/main/torchao/quantization/qat

Blogs:
https://selek.tech/posts/static-vs-dynamic-quantization-in-machine-learning/

<img src="attachments/asym_v_sym.png" width="500"><br>

# Asymmetric Quantisation

<img src="attachments/asym_1.png" width="500"><br>

Derivation:<br>
Note: clamping values to min and max is omitted for brevity<br>

<img src="attachments/asym_2.png" width="1000"><br>

# Symmetric Quantisation

<img src="attachments/sym_1.png" width="500"><br>

Derivation:<br>
Note:

- clamping values to min and max is omitted for brevity<br>
- 1 bit is actually unused on the negative side to maintain symmetry
- if the original range is symmetric, the scaled unused range is small.
  <img src="attachments/asym_2.png" width="1000"><br>

# Different strategies to choose a, b

<img src="attachments/outlier_1.png" width="500"><br>
<img src="attachments/outlier_2.png" width="500"><br>

# Applying Quantization

<img src="attachments/quant_1.png" width="500"><br>
<img src="attachments/quant_2.png" width="500"><br>