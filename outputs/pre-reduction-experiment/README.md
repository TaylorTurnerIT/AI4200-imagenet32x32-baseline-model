# One pre-reduction residual-block experiment

**The additional block did not improve validation accuracy in this trial.**
One fresh seed-0 candidate attempt completed all 25 epochs on the same A100 and
PyTorch version. No retries, continuation, test-set evaluation, or promotion.
The qualified submission and dataset notebook defaults remain unchanged.

| Measure | Baseline | Extra block |
| --- | ---: | ---: |
| Best validation accuracy | 22.0222% | 21.1067% |
| Selected epoch / completed epochs | 23 / 25 | 25 / 25 |
| Minimum validation loss | 4.069596 (epoch 15) | 4.101746 (epoch 25) |
| Epoch-25 training loss | 3.252028 | 3.419130 |
| Epoch-25 training-minibatch accuracy | 31.1640% | 28.3014% |
| Epoch-25 validation loss | 4.111427 | 4.101746 |
| Training minutes | 21.81 | 23.52 |
| Parameters | 3,923,048 | 4,186,472 |

Candidate best accuracy was **0.9156 percentage points lower**.
Its worse training fit suggests optimization under this recipe deserves
attention; the result does not establish that the extra block merely overfit.
Its final validation loss was slightly better than baseline's final loss, but
worse than baseline's minimum. Best candidate accuracy and minimum loss both
occurred at epoch 25, so its later trajectory remains unmeasured. This one
trial does not rule out benefits from depth under other settings.

[Learning curves PNG](validation-comparison.png) · [SVG](validation-comparison.svg) ·
[Interactive summary](preview.html) · [Exact summary](summary.json) ·
[Epoch metrics](run-record/metrics.csv).

## Fixed hypothesis and architecture

[Intent](intent.json), [launch protocol](protocol.json), and Change 146 in
[LOG.md](../../LOG.md) were recorded before training. The assistant-proposed
hypothesis, accepted by the user for one experiment, was that additional
processing before spatial compression might improve validation accuracy. The
primary threshold was the baseline's 22.022230% best validation accuracy.

Added one ordinary residual MLP before the final 2×2 to 1×1 window stage:
flatten 64×2×2 to 256; LayerNorm; Linear(256,512); GELU; Linear(512,256); add the
original 256 values; reshape to 64×2×2. It adds 263,424 parameters. Added linear
weights use Xavier normal with zero biases. A CPU RNG fork preserves every
shared baseline parameter's seed-matched initialization and the subsequent
shuffle RNG state. It does not load saved weights.

AdamW lr 0.001, weight decay 0.00001, AMSGrad, constant schedule, no augmentation,
batch 256, 25 epochs, a 60-minute ceiling, and the seed-0 10% holdout were fixed.
Both runs used high-RAM A100-SXM4-80GB, PyTorch 2.11.0+cu130, CUDA 13.0.

## Evidence and checks

- One attempt: [persistent launch marker](execution-started.json) and [completed attempt receipt](attempt.json). The runner refuses an existing attempt directory; no marker was cleared.
- Data: [array hashes](training-data-reference.json) matched all 1,281,167 training examples and labels exactly. Saved training and validation indices were identical to the baseline checkpoint.
- Source commit: `71c9c2d402bfe38ee374cf18261604b615b96355`; [exact source snapshots](source/) and [hash manifest](artifact-sha256.json). Frozen training core installed from baseline commit `b5ab6e1f75de4cfc257cf85b7e2535cfa217934c`.
- Five CPU tests cover shared seeded parameters/RNG, identity recovery when the new branch is zeroed, shape/gradient flow, diagnostics leaving training unchanged, and refusal of a second launch.
- [Runtime](runtime.json), [verification](verification.json), [training receipt](run-record/training-receipt.json), and [CLI completion](execution.log).
- [Eight fixed-batch snapshots](diagnostics/training-dynamics.json) include the extra residual branch. Measurements use copied model/optimizer states. Layer metrics are keyed by module name; execution order is available in activation-variance keys.
- Raw selected checkpoint retained locally at `runs/experiments/imagenet32-prereduction-20261028-s0/best.pt`, excluded from Git. It is kept outside the dashboard's ordinary run list until the experimental source is integrated.

The experiment is separate from the exact instructor-harness qualification.
It evaluates validation accuracy only. The earlier baseline's 19.96% test score
remains the qualified ImageNet result. The A100 was terminated after preserving
and verifying the complete archive.
