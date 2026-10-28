# Report figures

[Download all report images](final-report-images.zip), or [accuracy timelines only](accuracy-plots.zip).
All PNGs use 300 DPI. SVG versions are available for the accuracy timelines and
curve comparison.

- [MNIST accuracy by experiment](mnist-accuracy.png) · [SVG](mnist-accuracy.svg)
- [CIFAR-10 accuracy by experiment](cifar10-accuracy.png) · [SVG](cifar10-accuracy.svg)
- [ImageNet32 accuracy by experiment](imagenet32-accuracy.png) · [SVG](imagenet32-accuracy.svg)
- [CIFAR-10 training curves](cifar10-training-comparison.png) · [SVG](cifar10-training-comparison.svg)
- [ImageNet32 training dynamics](training-dynamics.png)
- [ImageNet32 extra-block experiment](imagenet32-pre-reduction-comparison.png) · [SVG](imagenet32-pre-reduction-comparison.svg) · [Validation-only result](../pre-reduction-experiment/README.md)
- [ImageNet32 residual branch ablation](ablation-depth.png)
- [ImageNet32 window-stage spatial ablation](window-stage-ablation.png) · [SVG](window-stage-ablation.svg) · [Method and results](../window-stage-ablation/README.md)

The timelines retain the clipped lower ranges from the design draft. Downward
triangles preserve failed results outside the visible range. A dark diamond
marks each fresh seed-0 CUDA harness result. Historical saved-weight continuation
is identified separately. [Plot data](accuracy-data.json) retain exact values and
run IDs; the [interactive preview](../experiment-history/accuracy-preview.html)
switches metric and scale.

The CIFAR-10 curves compare earlier saved experiment
`cifar10-20261023-112130-b25190a4` with final harness run `cifar10-32ae8cde`.
These runs differ in recipe, seed, and hardware; the figure does not isolate a
single change. Training accuracy uses training minibatch predictions, augmented
when enabled. [Curve data](cifar10-training-comparison.json).

ImageNet figures use the selected fresh epoch-23 model. Dynamics use a fixed
training batch in eval mode; ablations use the full training-data validation
split. Update ratios come from a copied model/optimizer probe step. The branch
ablation covers the two residual classifier blocks. [Measurement details](../final-evaluation/README.md).

Regenerate the timelines and CIFAR-10 comparison from the repository root:

```bash
python scripts/plot_experiment_accuracy.py \
  --qualification outputs/final-evaluation/qualification.json \
  --output outputs/report-figures --dpi 300
python scripts/plot_final_curves.py --run runs/cifar10-32ae8cde \
  --output outputs/report-figures/cifar10-training-comparison.png
```

Run `scripts/analyze_submission.py` separately to reproduce ImageNet dynamics and
ablation figures from the saved selected checkpoint and official training cache;
see [SUBMISSION.md](../../SUBMISSION.md).
