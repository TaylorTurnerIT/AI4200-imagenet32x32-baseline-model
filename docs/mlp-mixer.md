# MLP-Mixer exploration

The original Mixer implementation is preserved in `src/residual_mlp_baseline/mixer.py`
and `mixer_training.py`. It remains available through `train_mixer()` and
`predict_mixer()`; the frozen grading recipes and dataset notebook defaults
use the best recorded models. Mixer has synthetic checks but no full-dataset
score in the saved history. `train_mixer()` never submits to a leaderboard.

## Mixer implementation

[`mixer.py`](../src/residual_mlp_baseline/mixer.py) implements Section 2 and Appendix E:
shared non-overlapping linear patch embeddings; channel LayerNorm before each
residual branch; token mixing shared across channels; channel mixing shared
across tokens; two dense layers with GELU per MLP; final LayerNorm, token mean,
and a zero-initialized linear classifier. There are no position embeddings,
attention layers, or convolution layers. The Linear patch projection is
mathematically equivalent to the reference's patch-sized, strided Conv stem.
LayerNorm epsilon and GELU approximation match Flax defaults. Dense kernels use
Flax's LeCun truncated-normal convention; biases start at zero.

All seven Table 1 configurations are available through `mixer_config()`:
`S/32`, `S/16`, `B/32`, `B/16`, `L/32`, `L/16`, and `H/14`. Their defaults use
224×224 input. Example:

```python
from residual_mlp_baseline.mixer import MlpMixer, mixer_config

model = MlpMixer(mixer_config("B/16", num_classes=1000))
```

[`mixer_training.py`](../src/residual_mlp_baseline/mixer_training.py) implements
linear warmup and linear decay, AdamW with betas (0.9, 0.999), global gradient
norm clipping at 1, random resized crops, horizontal flips, two-operation
RandAugment, mixup, dropout, and stochastic depth. Drop probabilities increase
linearly across all residual MLP branches. `MixerRecipe()` records Table 4's
ImageNet B/L recipe: 300 epochs, batch 4096, learning rate 0.001, weight decay
0.1, warmup 10,000 updates, RandAugment magnitude 15, mixup alpha 0.5, dropout
0, and maximum stochastic depth 0.1. AdamW explicitly implements decoupled
weight decay. Pillow bilinear interpolation, black fill, the RandAugment
operation pool and 0–30 magnitude scale are implementation choices where the
paper does not fully specify behavior; run configuration records them.

The local defaults adapt this recipe to the repository's datasets: Mixer-S
widths/depth, 4×4 patches, original 28/32-pixel resolution, batch 256, and no
horizontal flips for MNIST. They are **not** the paper's full-resolution
ImageNet experiments. Warmup is capped at `total_steps - 1` for short runs;
both requested and effective warmup are recorded. The repository's seeded 10%
validation holdout and best-validation checkpoint selection remain local
evaluation conventions. Training images stay uint8 until each sample is
transformed; Mixer does not expand the full dataset to float32.

Configure architecture through `task_config()` and training through
`MixerRecipe`. The dataset notebooks retain their best measured defaults;
the separate Mixer API supports explicit experiments:

```python
from residual_mlp_baseline.mixer_training import MixerRecipe, task_config, train_mixer

model = train_mixer(
    "cifar10", X_train, y_train, device="cpu",
    config=task_config("cifar10"),
    recipe=MixerRecipe(batch_size=256),
    run_dir="runs/mixer-experiment",  # Must not already exist.
)
```

`model.resize_for_finetuning(448)` implements Appendix C's block-diagonal token
MLP expansion from 224 to 448, retaining patch size and ordering tokens by
spatial tile. It returns a new model. `MixerRecipe.finetuning(...)` supplies
SGD with momentum 0.9, batch 512, zero weight decay, cosine decay with warmup,
and no mixup, RandAugment, dropout, or stochastic depth. Choose task-specific
learning rate, epochs, and warmup explicitly; the paper sweeps these. Pass a
pre-trained `model=` to `train_mixer()` for transfer, using a matching architecture. The recipe updates
dropout and stochastic depth while retaining learned weights. Call
`model.reset_classifier(num_classes)` to initialize a new downstream task head. The grading interface always trains from scratch.
No pretrained checkpoints or datasets are bundled. Implementation tests use
synthetic data; they do not reproduce or establish the paper's accuracy.

The original worktree's tracked edits are preserved in
[`mlp-mixer-worktree.patch`](mlp-mixer-worktree.patch) as historical evidence.
Its former package/dashboard defaults are superseded by the consolidated
dataset notebooks and frozen grading recipes.

