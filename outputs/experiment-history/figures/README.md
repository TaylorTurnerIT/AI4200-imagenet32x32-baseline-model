# Experiment accuracy — design draft

One plot per task: every recorded result, a best-observed step line, era boundaries and a few numbered milestones.

- [MNIST](mnist-accuracy.png) · [SVG](mnist-accuracy.svg)
- [CIFAR-10](cifar10-accuracy.png) · [SVG](cifar10-accuracy.svg)
- [ImageNet32](imagenet32-accuracy.png) · [SVG](imagenet32-accuracy.svg)

The [interactive preview](../accuracy-preview.html) switches between historical test and validation accuracy, and between clipped linear, full linear and log-error scales. Clipping limits are editable. Hover or tap to inspect the actual score and run ID.

The default MNIST range is 95–99.5%. Three lower results appear as downward triangles; their original values remain in the data and tooltips. CIFAR-10 uses 48–77%; ImageNet32 uses 10–23%. Log-error spacing uses `100 − accuracy` logarithmically, expanding differences near 100% while retaining the low scores.

Best observed is a running maximum of the selected metric, computed from raw scores before clipping. The experiment dots retain regressions. Missing scores produce no dot. Of 167 recorded attempts, 147 have historical test scores. The ImageNet continuation is labeled separately because it starts from saved weights.

## Regenerate

```sh
uv run --locked python scripts/plot_experiment_accuracy.py
```

Use `--scale full` or `--scale log-error`, and `--metric validation`, to change the static exports. Add `--output /path/to/directory` to keep alternatives separately. Edit the `accuracy_view` section in [eras.json](../eras.json) for ranges and milestone choices. [accuracy-data.json](accuracy-data.json) preserves plotted values and run IDs.

The earlier [detailed loss comparisons](detailed-figures.md) remain available for the report's training-curve analysis.
