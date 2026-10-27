# Final submission and reproduction

[Fresh A100 harness results and executed notebook](outputs/final-evaluation/README.md):
MNIST 99.15%, CIFAR-10 74.76%, ImageNet32 19.96%; total training 40.33 minutes.

The package-level `train()` and `predict()` use frozen recipes in
`src/residual_mlp_baseline/submission.py`. The marimo notebook remains an
experimental dashboard. Every public training call creates fresh weights;
validation selects the best complete epoch, breaking ties toward the earlier
epoch. No saved checkpoint is loaded by the grading path.

| Task | Model | Training | Maximum |
| --- | --- | --- | --- |
| MNIST | Seven residual blocks, width 256, hidden 512, pre-LayerNorm, GELU, Xavier normal | SGD, lr 0.05, weight decay 0.0001, no momentum; historical augmentation | 500 epochs / 15 minutes |
| CIFAR-10 | Six padded rolling-window MLP stages, hidden/output 128/64; width-512 classifier, two gated residual blocks, hidden 1024, dropout 0.1 | AdamW, lr 0.001, weight decay 0.00001, AMSGrad; historical augmentation | 60 epochs / 15 minutes |
| ImageNet32 | Same rolling-window architecture, 1000-class head | Same AdamW; augmentation off | 25 epochs / 60 minutes |

All tasks use batch size 256, constant learning rate, cross entropy, and a seeded
10% validation split from the supplied training data. The time allocations sum
to 90 minutes. An internal reserve permits restoring the best completed epoch;
an incomplete epoch is discarded. ImageNet normalization runs per minibatch.
Time guards are checked at batch boundaries, not a hard process kill.

Recipe evidence is preserved in `tests/fixtures/submission-recipes.json`:

- MNIST: `mnist-20261023-105208-872d486e`, validation 99.3333%, historical test 99.23%.
- CIFAR-10: `cifar10-20261026-224057-085c2c3b`, validation 76.10%, historical test 75.45%.
- ImageNet32: `imagenet32-20261027-000850-1c570f4f`, validation 20.4939%, historical test 18.496%.

These historical runs used different runtime conditions. The 20.004% continued
ImageNet result loaded prior weights and is analysis evidence only. The new
25-epoch maximum is a fresh-run experiment, not a claim of equivalent accuracy.

## Exact instructor harness

`project1_harness.ipynb` retains the instructor's cells except the four supported
submission settings. Its CUDA/CPU device choice, seed 0, datasets, train/predict
calls, output assertions, and accuracy calculation are unchanged. The nickname
is empty, disabling leaderboard posting. Section is 5000.

Publish the intended commit before testing. On a fresh CUDA runtime, execute the
notebook with `HARNESS_PACKAGE_URL=git+https://github.com/TaylorTurnerIT/AI4200-imagenet32x32-baseline-model.git@<commit>`.
This uses the instructor's existing environment override to pin the install.
The notebook installs with `--no-deps`; the public import therefore requires no
marimo dashboard dependency. Preserve the executed notebook and runtime receipt
with results, rather than treating synthetic checks as full reproduction.

The authorized final environment is an A100 with high RAM, using the official
Colab CLI. Keep CLI credentials and session state outside the repository.

## Optional training analysis

Set `AI4200_EVIDENCE_DIR` to save unique run folders with settings, per-epoch
metrics, selected weights, split indices, and a training receipt. No artifact is
loaded during training. Set `AI4200_ANALYSIS_DIR` to enable ImageNet diagnostics;
this option is off by default and consumes training time when enabled.

Diagnostics use the same 32 unaugmented training examples in eval mode at
initialization, epochs 1/5/10/15/20/25 when reached, and the selected checkpoint.
They measure activation variance, residual branch-to-stream RMS ratios, weight
gradients, and update-to-weight RMS ratios. A model and optimizer copy receives
one probe-batch step, leaving live training unchanged. GELU has no tanh/ReLU
saturation measurement. Probe update ratios describe this fixed-batch step,
not the actual random training minibatch update.

After training, run the separate analysis script against the selected ImageNet
run and official training cache:

```bash
python scripts/analyze_submission.py --run RUN --data data/imagenet32 \
  --probe PROBE --output NEW_OUTPUT --device cuda
```

This evaluates the full validation split, then removes each residual classifier
branch individually without retraining. It writes JSON and 300-DPI dynamics and
ablation figures. Window stages are feed-forward MLP stages without identity
skip connections; they are not residual branches and are not ablated by this
procedure. Keep this distinction explicit in the report.

## Consolidation scope

The final main branch includes the saved-run reconstruction, draft `LOG.md`,
report images, model walkthrough, and frozen submission path. Personal Why and
Expected sections remain TODOs unless journal wording was available. The
separate `t3/implement-google-paper` MLP-Mixer exploration is excluded.
