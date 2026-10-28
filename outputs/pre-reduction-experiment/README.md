# One pre-reduction residual-block experiment

Authorized October 28, 2026: one fresh ImageNet32 candidate run, seed 0, with no
retry or automatic promotion. The public grading model and dataset notebook
defaults remain unchanged.

[Pre-run hypothesis and fixed protocol](intent.json) were recorded before the
training attempt. The extra block processes the flattened 64×2×2 representation
with LayerNorm and Linear(256,512) → GELU → Linear(512,256), adds the original
representation, and reshapes it before the final window reduction. It adds
263,424 parameters (3,923,048 to 4,186,472).

All existing weights retain their baseline seed-matched initialization. A CPU
RNG fork prevents the new block's initialization from changing shuffle RNG
state. The training recipe remains AdamW lr 0.001, weight decay 0.00001, AMSGrad,
constant schedule, no augmentation, batch 256, 25 epochs, and a 60-minute cap.
The seed-0 holdout is identical. The training arrays are checked byte-for-byte
against [the existing cache](training-data-reference.json).

Baseline: `imagenet32-8856b095`, 22.022230% best validation accuracy, selected
epoch 23 of 25. Its original A100/PyTorch 2.11.0+cu130 environment is matched.
The primary outcome is best validation accuracy. Validation loss, train/holdout
gap, and time are secondary evidence; one seed does not establish a robust gain.
No test-set predictions are requested for this exploratory comparison.

The experiment is separate from the instructor harness and does not change the
qualified submission. It uses the same frozen training core. Candidate diagnostics
also record the added residual branch, using copied model/optimizer states.

Status: runtime/data preparation. Training has not started. Detailed results
will be saved here after the single attempt. The persistent attempt directory
refuses a second launch and is preserved even on failure.
