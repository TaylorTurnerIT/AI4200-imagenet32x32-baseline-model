# title: Activator reference (arXiv:2405.15953v4)
# description: Patch embedding, four GEGLU residual blocks, global average pooling, and a classifier.
# category: MLPs
"""Standalone architecture sketch for opening and copying in Torch Canvas.

Source: Abdullah and Aydin, "Activator: GLU Activation Function as the Core
Component of a Vision Transformer", arXiv:2405.15953v4 (25 November 2025).
https://arxiv.org/html/2405.15953v4

Open this .py file through Torch Canvas's Open / import picker. Expand an
ActivatorBlock to inspect the two projection branches, multiplication, and
residual addition. Copy that block independently if you only want the gate.
This file does not change the project notebook or start training.

Default image path (B denotes batch size):
    RGB image                         [B, 3, 32, 32]
    4x4 patch embedding               [B, 256, 8, 8]
    flatten spatial axes, transpose   [B, 64, 256]
    four independent ActivatorBlocks  [B, 64, 256]
    average the 64 patch tokens       [B, 256]
    linear classifier                 [B, 10] logits

Within each block, D=256 is token width and H=512 is expanded width:
    n = LayerNorm_D(x)
    value = LayerNorm_H(Linear_D_to_H(n))
    gate = LayerNorm_H(GELU(Linear_D_to_H(n)))
    output = x + Linear_H_to_D(value * gate)

Interpretation choices and limits:
* Follow equations (6)-(8), including both branch LayerNorms. Procedure 3
  and Figures 1-2 omit those branch norms; the paper is inconsistent here.
  This is an equation-based teaching reference, not a verified reproduction
  of the authors' implementation or reported accuracy.
* Section 4 supplies the default patch size, widths, and four-block depth.
  Keep the full hidden width: the paper omits the usual 2/3 GLU adjustment.
* A stride-4, kernel-4 convolution implements the shared linear embedding of
  non-overlapping 4x4 patches. It learns patch features; it is not max pooling.
* Return raw logits for CrossEntropyLoss. For probabilities at inference,
  apply model(images).softmax(dim=-1), outside forward().
* Biases and LayerNorm epsilon use PyTorch defaults; the paper does not fully
  specify these details. No positional embedding or class token is added.
* Blocks transform each patch independently with shared weights. There is
  no attention or cross-patch mixing before global averaging. Consequently,
  rearranging whole patches preserves the output, up to floating-point error.
* The GELU gate is learned and continuous, not a random mask or a probability
  constrained to [0, 1]. Neither dropout nor pixel replacement is included.

Copying only ActivatorBlock into the current residual MLP applies this gate
to the whole-image feature vector. That is a different architecture from the
patch-based classifier below. D and H can differ because the final projection
restores D before adding the residual.

Inputs must be floating-point NCHW images, with height and width divisible
by patch_size. Non-divisible dimensions would lose border pixels in Conv2d.
"""

import torch
import torch.nn as nn


class ActivatorBlock(nn.Module):
    """One token-local, pre-normalized GEGLU residual block; last axis is D."""

    def __init__(self, width=256, hidden=512):
        super().__init__()
        self.input_norm = nn.LayerNorm(width)
        self.value_up = nn.Linear(width, hidden)
        self.value_norm = nn.LayerNorm(hidden)
        self.gate_up = nn.Linear(width, hidden)
        self.gate_activation = nn.GELU()
        self.gate_norm = nn.LayerNorm(hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        """Preserve x's shape; [B, tokens, D] and [B, D] both work."""
        normalized = self.input_norm(x)
        value = self.value_norm(self.value_up(normalized))
        gate = self.gate_norm(self.gate_activation(self.gate_up(normalized)))
        gated = value * gate
        update = self.down(gated)
        return x + update


class ActivatorReference(nn.Module):
    """Paper-sized CIFAR-10 sketch, with dimensions exposed for Canvas edits."""

    def __init__(self, in_channels=3, num_classes=10, patch_size=4,
                 width=256, hidden=512, depth=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, width, kernel_size=patch_size,
                                     stride=patch_size)
        self.blocks = nn.ModuleList([
            ActivatorBlock(width, hidden) for _ in range(depth)
        ])
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x):
        """Embed patches, transform each token, average tokens, return logits."""
        x = self.patch_embed(x)       # [B, 3, 32, 32] -> [B, 256, 8, 8]
        x = x.flatten(2).transpose(1, 2)  # [B, 64, 256], features last
        for block in self.blocks:
            x = block(x)             # Each block keeps [B, 64, 256].
        x = x.mean(1)                # Global average pooling: [B, 256].
        return self.classifier(x)    # [B, 10], no softmax during training.


if __name__ == "__main__":
    # Running this file checks shapes only; it loads no data and does not train.
    model = ActivatorReference()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        logits = model(images)
    print(f"Input: {tuple(images.shape)}; logits: {tuple(logits.shape)}")
