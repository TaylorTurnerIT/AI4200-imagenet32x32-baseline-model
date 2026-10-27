# Window-stage spatial ablation — October 27, 2026

Selected ImageNet checkpoint: `imagenet32-8856b095`, epoch 23. No retraining.
The same 128,116 training-data validation examples gave **22.022230%** for the
unmodified model, exactly matching the CUDA qualification. This separate
analysis used local Apple Metal with PyTorch 2.14.1 and took 154.74 seconds.
Weights and restored full-model accuracy were unchanged after all interventions.

## What was ablated

The six window stages perform learned spatial reduction and channel mapping;
they are not standalone pooling layers with an identity bypass. The first
stage changes 3 channels to 64. Skipping a stage directly breaks the geometry
or channel dimensions.

At one stage at a time, the controls change the gathered 2×2 neighborhood:

- **Single pixel:** repeat its top-left pixel in all four positions. This removes aggregation of neighboring pixels.
- **Mean only:** repeat its spatial mean in all four positions. This retains the mean but removes within-window differences and arrangement.

Both pass the controlled window through the original learned MLP, preserving
its padding, stride, output size, and channel mapping. All other stages remain
unchanged. These are spatial-information interventions, not deletion of an
entire learned stage. A drop can reflect loss of information and a changed
feature distribution; it is not a retrained smaller-model performance estimate.

| Stage | Input to output, C×H×W | Single-pixel accuracy | Drop, points | Mean-only accuracy | Drop, points |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 3×32×32 to 64×31×31 | 0.5917% | 21.4306 | 0.5823% | 21.4399 |
| 2 | 64×31×31 to 64×16×16 | 2.2846% | 19.7376 | 2.9465% | 19.0757 |
| 3 | 64×16×16 to 64×8×8 | 6.2443% | 15.7779 | 13.7914% | 8.2308 |
| 4 | 64×8×8 to 64×4×4 | 8.0685% | 13.9538 | 16.4515% | 5.5707 |
| 5 | 64×4×4 to 64×2×2 | 5.8166% | 16.2056 | 12.8571% | 9.1651 |
| 6 | 64×2×2 to 64×1×1 | 3.2400% | 18.7822 | 6.9367% | 15.0855 |

The first two stages are particularly sensitive to removing within-window
structure. The fourth stage tolerates the mean-only control best. The sixth
stage is also sensitive: its mean-only accuracy is 6.9367%. These measurements
are separate from the earlier residual-branch bypass experiment, which used a
different intervention.

[PNG plot](window-stage-ablation.png) · [SVG plot](window-stage-ablation.svg) ·
[Exact results and integrity checks](window-stage-ablation.json) ·
[Execution output](execution.log).

Reproduce without calling the graded training function:

```bash
PYTHONPATH=src python scripts/ablate_window_stages.py \
  --run runs/imagenet32-8856b095 --data data/imagenet32 \
  --output outputs/NEW-window-stage-ablation --device mps
```

The selected checkpoint is local under ignored `runs/`; its SHA-256 matches the
original qualification artifact manifest. CUDA can be selected instead. The
script measures its own baseline and checks restoration on the same backend.
