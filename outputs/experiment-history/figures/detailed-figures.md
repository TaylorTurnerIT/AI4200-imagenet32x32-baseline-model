# Development eras — figure design draft

Three figures use the same design: checkpoint accuracy across experiment attempts above, representative training and validation loss below. Each has a PNG and an editable SVG export.

- [MNIST](mnist-eras.png): capacity and weight decay; momentum and depth corrections; longer training, validation and augmentation; rolling-window transfer. An inset expands 95–100% accuracy while retaining the severe failures in the full view.
- [CIFAR-10](cifar10-eras.png): original baseline; flat MLP exploration; Happy Accident and audit; optimizer/width/depth experiments; bilinear gates, pooling and dropout; hierarchical rolling windows.
- [ImageNet32](imagenet32-eras.png): flat gated MLP; rolling-window MLP; continued training, explicitly labeled as analysis.

These are working labels. The separate MLP-Mixer implementation has no saved dataset experiment in this snapshot and contributes no invented score.

## Evidence and interpretation

The figures include 167 recorded attempts: 109 unique saved configurations and 58 earlier CSV-only results. Copied runs across worktrees are counted once. Unfinished runs get a cross below the accuracy axis, never a zero score. The audit cohort shares a directory timestamp; its result timestamps establish the order within that cohort.

Accuracy uses the recorded checkpoint result, usually the best validation checkpoint, rather than the final epoch's weights. Training and validation are separate from historical test observations. No best-so-far envelope hides regressions, and no test score selects a representative run automatically.

Loss curves show recorded epochs without smoothing. The horizontal axis is percentage of recorded training progress, not equal wall time or equal optimizer updates. Representative labels give the epoch counts. Training loss can include augmentation; validation loss uses held-out images. Comparisons change several settings and do not establish isolated causal effects.

The ImageNet continuation panel includes only new epochs 11–25; the copied parent prefix is excluded. Its optimizer and random state restarted at epoch 10. Its score is not fresh-from-scratch performance within the grading cap.

No original-baseline per-epoch loss trace was located, and no original ImageNet baseline accuracy was found. The CIFAR reference loss trace is the later augmentation-off audit, not a claim that it is the instructor's unchanged baseline.

## Revise or regenerate

Edit [eras.json](../eras.json) to rename eras, change representative runs or adjust annotations. Full run IDs remain attached to every choice. [plotted-data.json](plotted-data.json) records every plotted point and representative epoch range.

From the repository root:

```sh
uv run --locked python scripts/plot_development_eras.py
```

The script reads [all-runs.json](../all-runs.json) and does not modify `LOG.md`. Refresh that snapshot with later experiments before the final export. The current figures are a design draft, not a completed report or final model selection.
