# Experiment architecture recovery

Final fresh CUDA qualification is recorded separately in [final-evaluation](../final-evaluation/README.md); this reconstruction snapshot is preserved.

This is a source-backed reconstruction for writing `LOG.md`, not a replacement development log. It preserves facts and identifies missing evidence. The student’s original journal supplies the reasons; no pre-run prediction has been invented.

## Scope and limits

- Checked all five currently registered worktrees, including uncommitted model changes. Found 53 unique saved attempts in scope: 42 with final summaries and 11 without. This includes the six audit runs; there are 47 attempts after that audit cohort.
- Saved artifacts are concentrated in the main checkout; the continuation worktree adds three MNIST attempts. Three copied rolling-window runs there have identical config bytes and are counted once.
- The MLP-Mixer implementation is a separate uncommitted worktree branch, described below. No dataset-run configuration was found there; its journal records synthetic checks only.
- Current branches and uncommitted files can change. `inventory.json` captures saved configurations, model/block/support sources, exact settings, scores, source-file SHA-256 values, all discovered locations and curve endpoints at inspection time.
- First saved occurrence establishes when an architecture was used, not necessarily when it was first written. A `restored_from` ID records loaded settings, not necessarily a direct experimental predecessor.
- Missing final summaries do not by themselves prove a crash; some attempts may have been stopped or interrupted. An empty metrics file only establishes no completed epoch was saved.
- Saved source and module shapes take precedence over class names, stale docstrings, constructor defaults and rounded journal summaries. Several models named `ResidualMLP` had no skip connections.
- This recovery inspected historical artifacts and current source only. It did not rerun training or independently reproduce historical accuracies.

## Architecture sequence

### Oct 24, audit reference — Plain residual MLP

Flatten image; input Linear(3072,256); seven pre-LayerNorm blocks, each Linear(256,512), GELU, Linear(512,256), plus skip; LayerNorm/512-hidden GELU head. 2,767,882 CIFAR parameters.

Observed result: 56.40% validation; 54.59% test. Audit also tested 128/128 widths and ReLU.

Evidence: [cifar10-20261024-222752-audit-baseline config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-222752-audit-baseline/config.json>) · [summary](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-222752-audit-baseline/summary.txt>).

### Oct 25, 00:13–12:48 — Activation, width and depth experiments

GELU to SiLU and back; residual width 256 to 512; depth seven to six to three and back to six. Hidden width stayed 512. Six-block 512/512 model: 5,000,202 parameters.

Observed result: Six-block run: 62.26% validation, 60.15% test. Three-block run: 59.12% validation, 57.55% test. Recipes also varied; these are observations, not isolated causal estimates.

Evidence: [cifar10-20261025-123955-7c8c8c1d config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-123955-7c8c8c1d/config.json>) · [summary](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-123955-7c8c8c1d/summary.txt>).

### Oct 25, 12:57–13:25 — Incomplete block edits

Four saved attempts have a block forward() that returns None. Some linear layers moved to the parent model but never enter its forward().

Observed result: All four have only the metrics header, no completed epochs and no final summary. No exception text was found in those artifacts; do not invent a specific crash message.

Evidence: [cifar10-20261025-125710-62ba38ce config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-125710-62ba38ce/config.json>).

### Oct 25, 13:54–14:23 — Bilinear blocks

Two Linear projections of LayerNorm(x) are multiplied elementwise. These saved blocks have no skip connection, and the parent simply calls blk(x). Early version also registers six unused Linear layers. Later version removes those layers and uses SiLU in the classifier head.

Observed result: Simplified version: 60.24% validation, 57.57% test, reproduced in three runs. SiLU here is the head activation; there is no SiLU inside the bilinear branch.

Evidence: [cifar10-20261025-140404-e8234a17 config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-140404-e8234a17/config.json>) · [summary](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-140404-e8234a17/summary.txt>).

### Oct 25, 19:51–20:12 — Max-pooling detour

A literal nn.MaxPool2d(kernel_size=4) reduces RGB inputs to 3×8×8; the projection now takes 192 values. Bilinear blocks restore x + branch.

Observed result: 57.38% validation, 55.73% test. Historical extra layer type: do not describe this as using only the assignment’s listed building blocks. A later saved run returns to the ordinary six-block GELU model.

Evidence: [cifar10-20261025-201234-869c059e config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-201234-869c059e/config.json>) · [summary](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-201234-869c059e/summary.txt>).

### Oct 25, 21:33 — Normalized value/gate architecture

Width 512, six blocks, branch hidden width 1024. Normalize input, separately project value and GELU gate, LayerNorm both branches, multiply, project down. Replace the old nonlinear head with one Linear(512,num_classes). These saved models have no residual addition. 11,061,770 CIFAR parameters.

Observed result: 61.52% validation, 59.67% test. Note: the second positional Block19 argument fills unused h1; actual hidden=1024 comes from the block default. Read saved module shapes, not misleading constructor names.

Evidence: [cifar10-20261025-213314-9c0e7949 config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-213314-9c0e7949/config.json>) · [summary](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-213314-9c0e7949/summary.txt>).

### Oct 26, 13:11–15:34 — Dropout on normalized gates

Dropout 0.1 on each branch output, then a 0.15 trial and a return to 0.1. Still six blocks without skips. Cosmetic class/variable renames are not separate conceptual architectures.

Observed result: 60-epoch dropout-0.1 run: 63.50% validation, 60.87% test. Later runs used 10–20 epochs, so their scores do not isolate dropout probability.

Evidence: [cifar10-20261026-131127-ba8e8b68 config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-131127-ba8e8b68/config.json>) · [summary](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-131127-ba8e8b68/summary.txt>).

### Oct 26, 19:58–21:03 — Same flat gated architecture on ImageNet32

Same 512-wide, six-block, 1024-hidden dropout-0.1 architecture, now 1000 classes: 11,569,640 parameters. This is a task/head-size change, not the rolling-window design.

Observed result: 10 epochs: 12.1632% validation, 11.106% test. 20 epochs: 12.7307% validation, 11.648% test.

Evidence: [imagenet32-20261026-210325-b9c04e03 config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-20261026-210325-b9c04e03/config.json>) · [summary](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-20261026-210325-b9c04e03/summary.txt>).

### Oct 26, 21:52 — Hierarchical 2×2 rolling windows

Keep spatial feature maps; gather 2×2 windows with Tensor.unfold. Apply one shared two-layer GELU MLP within each stage; different stages have different weights. Five unpadded stages: 32→31→15→7→3→1. Window hidden/output 32/16; classifier width/hidden 256/512 with TWO gated residual blocks and 0.1 dropout. Skip addition is now in the parent. 812,410 parameters.

Observed result: 49.98% validation, 48.27% test.

Evidence: [cifar10-20261026-215237-2ac3a7b3 config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-215237-2ac3a7b3/config.json>) · [summary](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-215237-2ac3a7b3/summary.txt>).

### Oct 26, 22:13 — Widened rolling windows

Window hidden/output 32/16 to 128/64; classifier width/hidden 256/512 to 512/1024. Still two residual blocks and five unpadded stages. 3,374,026 parameters.

Observed result: 60.80% validation, 59.15% test. This changes several capacities together; it cannot isolate the effect of any one dimension.

Evidence: [cifar10-20261026-221344-5d653ac6 config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-221344-5d653ac6/config.json>) · [summary](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-221344-5d653ac6/summary.txt>).

### Oct 26, 22:40 — Full-image coverage correction

Unpadded stride-2 stages had discarded edges, leaving the final vector connected only to the top-left 17×17 pixels. Pad unmatched bottom/right edges. Spatial sequence becomes 32→31→16→8→4→2→1, adding a sixth stage. Preserve the wider dimensions. 3,415,178 parameters.

Observed result: 76.10% validation, 75.45% test; selected epoch 51/60. Record this as padding/full-coverage plus an extra stage, not a pure one-variable comparison.

Evidence: [cifar10-20261026-224057-085c2c3b config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-224057-085c2c3b/config.json>) · [summary](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-224057-085c2c3b/summary.txt>).

### Oct 27, ImageNet and MNIST — Current rolling-window family

Refactor stages into RollingWindowBlock and move x + branch into Block19. Source structure changes but the intended mathematical graph is preserved; regression fixtures cover seeded weights, logits, gradients and checkpoint keys. ImageNet has 3,923,048 parameters; MNIST 3,414,154.

Observed result: ImageNet initial run: 20.4939% validation, 18.496% test. Continuation: 21.9809% validation, 20.004% test; continuation is extra training with restarted optimizer/random state, not a new architecture or a fresh graded run. Separate-worktree MNIST has two unfinished attempts and one completed 20-epoch Muon run: 95.7333% validation, 95.73% test.

Evidence: [imagenet32-20261027-000850-1c570f4f config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-20261027-000850-1c570f4f/config.json>) · [summary](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-20261027-000850-1c570f4f/summary.txt>).

### Separate worktree — MLP-Mixer

Uncommitted implementation in `t3/implement-google-paper`: non-overlapping 4×4 linear patch embeddings; eight S-sized Mixer blocks; token-mixing MLP hidden size 256; channel width 512; channel-mixing MLP hidden size 2048. Token and channel branches each have pre-LayerNorm and a residual addition. Final LayerNorm, average over tokens and linear classifier. CIFAR/ImageNet use 64 tokens; MNIST uses 49. This is a separate model family, not a stage in the rolling-window model. Code supports other presets and overrides; these are task defaults, not a demonstrated full-run configuration.

Evidence: [architecture](</Users/taylor/.t3/worktrees/AI4200-imagenet32x32-baseline-model/t3-d14dfe9e/src/residual_mlp_baseline/mixer.py>) · [task defaults/training](</Users/taylor/.t3/worktrees/AI4200-imagenet32x32-baseline-model/t3-d14dfe9e/src/residual_mlp_baseline/mixer_training.py>) · [worktree journal](</Users/taylor/.t3/worktrees/AI4200-imagenet32x32-baseline-model/t3-d14dfe9e/journal.md>). The journal reports synthetic checks including Metal, not full-dataset accuracy.

## Training and data changes to log separately

- Optimizer experiments include SGD, AdamW, Muon (with an auxiliary AdamW group) and RMSprop. AdamW `amsgrad` and other optimizer settings also vary. “AdamW with Adagrad” in the journal is not an exact saved optimizer identity; use the recorded settings.
- Epoch limits changed repeatedly (including 30/50/60 and 10/15/16/20). Batch size, learning rate, weight decay and augmentation changed independently of model shape. See per-run settings below and full JSON, rather than attributing all score changes to architecture.
- The rolling-window series switched to AdamW (AMSGrad enabled), learning rate 0.001, weight decay 0.00001, batch size 256. The nearby flat-model ImageNet runs used Muon, so the cross-family result is not an architecture-only ablation.
- The ImageNet continuation uses saved weights as a separate analysis experiment. It does not establish fresh-from-scratch performance within the grading time cap. Its parent and continuation have the same architecture.

## Converting this evidence into LOG.md

Use the assignment’s Change / Why / Expected / Result fields. One entry is needed for each changed model, recipe or pipeline; identical reruns can share an entry. A repeated model with changed training settings still needs the change recorded.

- **Change:** Use exact settings and source from the matching inventory entry. Include a run ID or commit; do not rely on a class number such as `Block19`.
- **Why:** Reuse the student’s existing words where they actually explain that change. If a reason is reconstructed from memory, label it retrospective. Do not infer intent from a saved configuration.
- **Expected:** Preserve genuine written predictions. Where none exists, say “Not recorded before this run.” A retrospective expectation must be labeled as such; do not present it as contemporaneous.
- **Result:** Record validation accuracy, observed test accuracy where historically measured, training time and what the saved loss curves show. Tune future choices on validation. Record failed and incomplete setups honestly, including missing error evidence.
- Keep the MaxPool2d and no-skip variants as historical unsuccessful or nonconforming explorations. They should not be represented as compliant residual MLPs.

## Every saved attempt in scope

Times below come from run IDs (local wall time); audit cohort IDs share one timestamp. “No final summary” is not an inferred failure reason. Accuracies use full-precision saved summary values before display rounding.

### cifar10-20261024-222752-audit-weight-decay-zero

**Architecture:** Plain residual MLP: 256/512/7 gelu. Width/hidden/depth: 256/512/7; parameters: 2,767,882.

**Training:** task `cifar10`, seed 42, 30 configured epochs, batch 256, optimizer `SGD`, learning rate 0.05, weight decay 0, augmentation enabled `False`.

**Result:** clean train 99.9978%; validation 55.7200%; test 54.6700%; 0.4944 minutes. Selected epoch 29; completed epochs 30.

**Curve endpoints:** training loss 1.679883 at epoch 1 to 0.002283 at epoch 30; validation loss 1.558608 at epoch 1 to 2.643826 at epoch 30. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-222752-audit-weight-decay-zero/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-222752-audit-weight-decay-zero/metrics.csv>).

### cifar10-20261024-222752-audit-relu

**Architecture:** Plain residual MLP: 256/512/7 relu. Width/hidden/depth: 256/512/7; parameters: 2,767,882.

**Training:** task `cifar10`, seed 42, 30 configured epochs, batch 256, optimizer `SGD`, learning rate 0.05, weight decay 0.0001, augmentation enabled `False`.

**Result:** clean train 99.9844%; validation 55.5200%; test 54.8100%; 0.5300 minutes. Selected epoch 30; completed epochs 30.

**Curve endpoints:** training loss 1.718357 at epoch 1 to 0.011148 at epoch 30; validation loss 1.577711 at epoch 1 to 2.306269 at epoch 30. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-222752-audit-relu/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-222752-audit-relu/metrics.csv>).

### cifar10-20261024-222752-audit-batch-128

**Architecture:** Plain residual MLP: 256/512/7 gelu. Width/hidden/depth: 256/512/7; parameters: 2,767,882.

**Training:** task `cifar10`, seed 42, 30 configured epochs, batch 128, optimizer `SGD`, learning rate 0.05, weight decay 0.0001, augmentation enabled `False`.

**Result:** clean train 99.9978%; validation 57.3800%; test 55.7700%; 0.8551 minutes. Selected epoch 27; completed epochs 30.

**Curve endpoints:** training loss 1.631168 at epoch 1 to 0.001167 at epoch 30; validation loss 1.497187 at epoch 1 to 2.619847 at epoch 30. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-222752-audit-batch-128/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-222752-audit-batch-128/metrics.csv>).

### cifar10-20261024-222752-audit-baseline

**Architecture:** Plain residual MLP: 256/512/7 gelu. Width/hidden/depth: 256/512/7; parameters: 2,767,882.

**Training:** task `cifar10`, seed 42, 30 configured epochs, batch 256, optimizer `SGD`, learning rate 0.05, weight decay 0.0001, augmentation enabled `False`.

**Result:** clean train 99.9867%; validation 56.4000%; test 54.5900%; 0.5187 minutes. Selected epoch 28; completed epochs 30.

**Curve endpoints:** training loss 1.679841 at epoch 1 to 0.004210 at epoch 30; validation loss 1.558701 at epoch 1 to 2.497566 at epoch 30. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-222752-audit-baseline/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-222752-audit-baseline/metrics.csv>).

### cifar10-20261024-222752-audit-weight-decay-1e-5

**Architecture:** Plain residual MLP: 256/512/7 gelu. Width/hidden/depth: 256/512/7; parameters: 2,767,882.

**Training:** task `cifar10`, seed 42, 30 configured epochs, batch 256, optimizer `SGD`, learning rate 0.05, weight decay 1e-05, augmentation enabled `False`.

**Result:** clean train 99.9978%; validation 55.3000%; test 54.7500%; 0.5125 minutes. Selected epoch 29; completed epochs 30.

**Curve endpoints:** training loss 1.679879 at epoch 1 to 0.002158 at epoch 30; validation loss 1.558617 at epoch 1 to 2.664308 at epoch 30. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-222752-audit-weight-decay-1e-5/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-222752-audit-weight-decay-1e-5/metrics.csv>).

### cifar10-20261024-222752-audit-width-hidden

**Architecture:** Plain residual MLP: 128/128/7 gelu. Width/hidden/depth: 128/128/7; parameters: 644,362.

**Training:** task `cifar10`, seed 42, 30 configured epochs, batch 256, optimizer `SGD`, learning rate 0.05, weight decay 0.0001, augmentation enabled `False`.

**Result:** clean train 72.8311%; validation 51.7400%; test 50.2900%; 0.4279 minutes. Selected epoch 15; completed epochs 30.

**Curve endpoints:** training loss 1.736499 at epoch 1 to 0.356411 at epoch 30; validation loss 1.599736 at epoch 1 to 2.102399 at epoch 30. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-222752-audit-width-hidden/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-222752-audit-width-hidden/metrics.csv>).

### cifar10-20261024-223819-2016e143

**Architecture:** Plain residual MLP: 256/512/7 gelu. Width/hidden/depth: 256/512/7; parameters: 2,767,882.

**Training:** task `cifar10`, seed 42, 30 configured epochs, batch 256, optimizer `SGD`, learning rate 0.05, weight decay 0.0001, augmentation enabled `False`.

**Result:** No final summary; 11 completed epoch rows saved.

**Curve endpoints:** training loss 1.679841 at epoch 1 to 0.641107 at epoch 11; validation loss 1.558701 at epoch 1 to 1.487677 at epoch 11. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-223819-2016e143/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-223819-2016e143/metrics.csv>).

### cifar10-20261024-234504-6badff50

**Architecture:** Plain residual MLP: 256/512/7 gelu. Width/hidden/depth: 256/512/7; parameters: 2,767,882.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `SGD`, learning rate 0.05, weight decay 0.0001, augmentation enabled `True`.

**Result:** clean train 87.4733%; validation 61.4800%; test 60.1100%; 2.8309 minutes. Selected epoch 53; completed epochs 60.

**Curve endpoints:** training loss 1.784356 at epoch 1 to 0.697558 at epoch 60; validation loss 1.552454 at epoch 1 to 1.318634 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-234504-6badff50/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-234504-6badff50/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-234504-6badff50/curves.png>).

### cifar10-20261024-234930-b4aa43ea

**Architecture:** Plain residual MLP: 256/512/7 gelu. Width/hidden/depth: 256/512/7; parameters: 2,767,882.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `AdamW`, learning rate 0.001, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 96.3133%; validation 61.2600%; test 59.0100%; 3.3840 minutes. Selected epoch 43; completed epochs 60.

**Curve endpoints:** training loss 1.799182 at epoch 1 to 0.332900 at epoch 60; validation loss 1.551959 at epoch 1 to 1.891902 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-234930-b4aa43ea/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-234930-b4aa43ea/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-234930-b4aa43ea/curves.png>).

### cifar10-20261024-235432-57963b71

**Architecture:** Plain residual MLP: 256/512/7 gelu. Width/hidden/depth: 256/512/7; parameters: 2,767,882.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `AdamW`, learning rate 0.005, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 93.7622%; validation 59.6400%; test 58.1100%; 3.3097 minutes. Selected epoch 52; completed epochs 60.

**Curve endpoints:** training loss 2.004996 at epoch 1 to 0.498903 at epoch 60; validation loss 1.748069 at epoch 1 to 1.863851 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-235432-57963b71/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-235432-57963b71/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261024-235432-57963b71/curves.png>).

### cifar10-20261025-000330-5cea5707

**Architecture:** Plain residual MLP: 256/512/7 gelu. Width/hidden/depth: 256/512/7; parameters: 2,767,882.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `AdamW`, learning rate 0.001, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 91.9044%; validation 61.4800%; test 59.4300%; 3.3901 minutes. Selected epoch 32; completed epochs 60.

**Curve endpoints:** training loss 1.799166 at epoch 1 to 0.317799 at epoch 60; validation loss 1.551933 at epoch 1 to 1.842555 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-000330-5cea5707/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-000330-5cea5707/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-000330-5cea5707/curves.png>).

### cifar10-20261025-000912-e7c42e1d

**Architecture:** Plain residual MLP: 256/512/7 gelu. Width/hidden/depth: 256/512/7; parameters: 2,767,882.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `AdamW`, learning rate 0.001, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 94.1978%; validation 61.6800%; test 59.0800%; 3.5287 minutes. Selected epoch 49; completed epochs 60.

**Curve endpoints:** training loss 1.832627 at epoch 1 to 0.503886 at epoch 60; validation loss 1.582198 at epoch 1 to 1.525676 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-000912-e7c42e1d/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-000912-e7c42e1d/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-000912-e7c42e1d/curves.png>).

### cifar10-20261025-001310-684974cb

**Architecture:** Plain residual MLP: 256/512/7 silu. Width/hidden/depth: 256/512/7; parameters: 2,767,882.

**Training:** task `cifar10`, seed 42, 30 configured epochs, batch 256, optimizer `SGD`, learning rate 0.05, weight decay 0.0001, augmentation enabled `True`.

**Result:** clean train 70.9911%; validation 60.2600%; test 58.8200%; 1.7885 minutes. Selected epoch 29; completed epochs 30.

**Curve endpoints:** training loss 1.783098 at epoch 1 to 1.040564 at epoch 30; validation loss 1.556317 at epoch 1 to 1.176018 at epoch 30. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-001310-684974cb/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-001310-684974cb/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-001310-684974cb/curves.png>).

### cifar10-20261025-001521-3603c62e

**Architecture:** Plain residual MLP: 256/512/7 silu. Width/hidden/depth: 256/512/7; parameters: 2,767,882.

**Training:** task `cifar10`, seed 42, 50 configured epochs, batch 256, optimizer `SGD`, learning rate 0.05, weight decay 0.0001, augmentation enabled `True`.

**Result:** clean train 78.8400%; validation 61.2800%; test 59.4700%; 2.9715 minutes. Selected epoch 46; completed epochs 50.

**Curve endpoints:** training loss 1.783098 at epoch 1 to 0.869601 at epoch 50; validation loss 1.556317 at epoch 1 to 1.177700 at epoch 50. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-001521-3603c62e/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-001521-3603c62e/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-001521-3603c62e/curves.png>).

### cifar10-20261025-002050-9941eabc

**Architecture:** Plain residual MLP: 256/512/7 gelu. Width/hidden/depth: 256/512/7; parameters: 2,767,882.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `AdamW`, learning rate 0.001, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 94.1978%; validation 61.6800%; test 59.0800%; 4.4547 minutes. Selected epoch 49; completed epochs 60.

**Curve endpoints:** training loss 1.832627 at epoch 1 to 0.503886 at epoch 60; validation loss 1.582198 at epoch 1 to 1.525676 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-002050-9941eabc/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-002050-9941eabc/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-002050-9941eabc/curves.png>).

### cifar10-20261025-002715-5a076b1b

**Architecture:** Plain residual MLP: 256/512/7 gelu. Width/hidden/depth: 256/512/7; parameters: 2,767,882.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.02, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 89.4933%; validation 61.5600%; test 59.2500%; 6.9284 minutes. Selected epoch 31; completed epochs 60.

**Curve endpoints:** training loss 1.747705 at epoch 1 to 0.427620 at epoch 60; validation loss 1.511235 at epoch 1 to 1.657752 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-002715-5a076b1b/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-002715-5a076b1b/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-002715-5a076b1b/curves.png>).

### cifar10-20261025-003949-24bf237f

**Architecture:** Plain residual MLP: 512/512/7 gelu. Width/hidden/depth: 512/512/7; parameters: 5,526,538.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.02, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 93.8378%; validation 61.9600%; test 59.6900%; 9.1604 minutes. Selected epoch 38; completed epochs 60.

**Curve endpoints:** training loss 1.763307 at epoch 1 to 0.443056 at epoch 60; validation loss 1.486495 at epoch 1 to 1.578831 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-003949-24bf237f/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-003949-24bf237f/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-003949-24bf237f/curves.png>).

### cifar10-20261025-005328-ee80b2f5

**Architecture:** Plain residual MLP: 512/512/7 gelu. Width/hidden/depth: 512/512/7; parameters: 5,526,538.

**Training:** task `cifar10`, seed 42, 30 configured epochs, batch 256, optimizer `RMSprop`, learning rate 0.001, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 75.6644%; validation 60.2400%; test 57.6800%; 2.2535 minutes. Selected epoch 27; completed epochs 30.

**Curve endpoints:** training loss 2.318147 at epoch 1 to 0.957198 at epoch 30; validation loss 1.786178 at epoch 1 to 1.227672 at epoch 30. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-005328-ee80b2f5/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-005328-ee80b2f5/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-005328-ee80b2f5/curves.png>).

### cifar10-20261025-005652-cd458538

**Architecture:** Plain residual MLP: 512/512/6 gelu. Width/hidden/depth: 512/512/6; parameters: 5,000,202.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.02, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 97.3133%; validation 62.0400%; test 60.1300%; 8.0145 minutes. Selected epoch 57; completed epochs 60.

**Curve endpoints:** training loss 1.751799 at epoch 1 to 0.452521 at epoch 60; validation loss 1.508540 at epoch 1 to 1.602175 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-005652-cd458538/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-005652-cd458538/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-005652-cd458538/curves.png>).

### cifar10-20261025-115606-bd7c0564

**Architecture:** Plain residual MLP: 512/512/3 gelu. Width/hidden/depth: 512/512/3; parameters: 3,421,194.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.02, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 95.6933%; validation 59.1200%; test 57.5500%; 4.2525 minutes. Selected epoch 27; completed epochs 60.

**Curve endpoints:** training loss 1.659445 at epoch 1 to 0.199116 at epoch 60; validation loss 1.442750 at epoch 1 to 2.177310 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-115606-bd7c0564/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-115606-bd7c0564/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-115606-bd7c0564/curves.png>).

### cifar10-20261025-123955-7c8c8c1d

**Architecture:** Plain residual MLP: 512/512/6 gelu. Width/hidden/depth: 512/512/6; parameters: 5,000,202.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 96.8067%; validation 62.2600%; test 60.1500%; 5.9099 minutes. Selected epoch 60; completed epochs 60.

**Curve endpoints:** training loss 1.762312 at epoch 1 to 0.518642 at epoch 60; validation loss 1.522307 at epoch 1 to 1.485432 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-123955-7c8c8c1d/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-123955-7c8c8c1d/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-123955-7c8c8c1d/curves.png>).

### cifar10-20261025-124849-bcdd1a30

**Architecture:** Plain residual MLP: 512/512/6 gelu. Width/hidden/depth: 512/512/6; parameters: 5,000,202.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.01, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 94.8800%; validation 61.9000%; test 60.2600%; 6.0614 minutes. Selected epoch 35; completed epochs 60.

**Curve endpoints:** training loss 1.742737 at epoch 1 to 0.394277 at epoch 60; validation loss 1.469321 at epoch 1 to 1.693536 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-124849-bcdd1a30/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-124849-bcdd1a30/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-124849-bcdd1a30/curves.png>).

### cifar10-20261025-125710-62ba38ce

**Architecture:** Incomplete block construction. Width/hidden/depth: 512/512/6; parameters: 4,994,058.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.01, weight decay 1e-05, augmentation enabled `True`.

**Result:** No final summary; 0 completed epoch rows saved.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-125710-62ba38ce/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-125710-62ba38ce/metrics.csv>).

### cifar10-20261025-125841-0e1749e2

**Architecture:** Incomplete block construction. Width/hidden/depth: 512/512/6; parameters: 4,994,058.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.01, weight decay 1e-05, augmentation enabled `True`.

**Result:** No final summary; 0 completed epoch rows saved.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-125841-0e1749e2/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-125841-0e1749e2/metrics.csv>).

### cifar10-20261025-130442-f759bea4

**Architecture:** Incomplete block construction. Width/hidden/depth: 512/512/6; parameters: 4,994,058.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** No final summary; 0 completed epoch rows saved.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-130442-f759bea4/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-130442-f759bea4/metrics.csv>).

### cifar10-20261025-132544-8610feee

**Architecture:** Incomplete block construction. Width/hidden/depth: 512/512/6; parameters: 4,994,058.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** No final summary; 0 completed epoch rows saved.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-132544-8610feee/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-132544-8610feee/metrics.csv>).

### cifar10-20261025-135437-5b0d7d45

**Architecture:** Bilinear blocks without skip and unused extra linear layers. Width/hidden/depth: 512/512/6; parameters: 6,576,138.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 98.7511%; validation 60.3400%; test 57.6600%; 5.9019 minutes. Selected epoch 44; completed epochs 60.

**Curve endpoints:** training loss 2.053201 at epoch 1 to 0.389171 at epoch 60; validation loss 1.771153 at epoch 1 to 1.715362 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-135437-5b0d7d45/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-135437-5b0d7d45/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-135437-5b0d7d45/curves.png>).

### cifar10-20261025-140146-70e4347d

**Architecture:** Bilinear blocks without skip and unused extra linear layers. Width/hidden/depth: 512/512/6; parameters: 6,576,138.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** No final summary; 22 completed epoch rows saved.

**Curve endpoints:** training loss 2.027322 at epoch 1 to 0.774405 at epoch 22; validation loss 1.753968 at epoch 1 to 1.371303 at epoch 22. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-140146-70e4347d/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-140146-70e4347d/metrics.csv>).

### cifar10-20261025-140404-e8234a17

**Architecture:** Bilinear blocks without skip. Width/hidden/depth: 512/512/6; parameters: 5,000,202.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 98.7733%; validation 60.2400%; test 57.5700%; 6.0311 minutes. Selected epoch 45; completed epochs 60.

**Curve endpoints:** training loss 2.032280 at epoch 1 to 0.387003 at epoch 60; validation loss 1.763730 at epoch 1 to 1.727932 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-140404-e8234a17/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-140404-e8234a17/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-140404-e8234a17/curves.png>).

### cifar10-20261025-141647-eeafe2e6

**Architecture:** Bilinear blocks without skip. Width/hidden/depth: 512/512/6; parameters: 5,000,202.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 98.7733%; validation 60.2400%; test 57.5700%; 5.9864 minutes. Selected epoch 45; completed epochs 60.

**Curve endpoints:** training loss 2.032280 at epoch 1 to 0.387003 at epoch 60; validation loss 1.763730 at epoch 1 to 1.727932 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-141647-eeafe2e6/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-141647-eeafe2e6/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-141647-eeafe2e6/curves.png>).

### cifar10-20261025-142317-26fef98f

**Architecture:** Bilinear blocks without skip. Width/hidden/depth: 512/512/6; parameters: 5,000,202.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 98.7733%; validation 60.2400%; test 57.5700%; 6.0893 minutes. Selected epoch 45; completed epochs 60.

**Curve endpoints:** training loss 2.032280 at epoch 1 to 0.387003 at epoch 60; validation loss 1.763730 at epoch 1 to 1.727932 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-142317-26fef98f/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-142317-26fef98f/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-142317-26fef98f/curves.png>).

### cifar10-20261025-195130-38a08235

**Architecture:** 4×4 max pooling + bilinear residual blocks. Width/hidden/depth: 512/512/6; parameters: 3,525,642.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** No final summary; 13 completed epoch rows saved.

**Curve endpoints:** training loss 1.827851 at epoch 1 to 0.983965 at epoch 13; validation loss 1.589860 at epoch 1 to 1.325175 at epoch 13. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-195130-38a08235/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-195130-38a08235/metrics.csv>).

### cifar10-20261025-201234-869c059e

**Architecture:** 4×4 max pooling + bilinear residual blocks. Width/hidden/depth: 512/512/6; parameters: 3,525,642.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 82.2622%; validation 57.3800%; test 55.7300%; 6.0256 minutes. Selected epoch 14; completed epochs 60.

**Curve endpoints:** training loss 1.827851 at epoch 1 to 0.380834 at epoch 60; validation loss 1.589860 at epoch 1 to 1.773807 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-201234-869c059e/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-201234-869c059e/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-201234-869c059e/curves.png>).

### cifar10-20261025-210636-d9406711

**Architecture:** Plain residual MLP: 512/512/6 gelu. Width/hidden/depth: 512/512/6; parameters: 5,000,202.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 96.8067%; validation 62.2600%; test 60.1500%; 6.0289 minutes. Selected epoch 60; completed epochs 60.

**Curve endpoints:** training loss 1.762312 at epoch 1 to 0.518642 at epoch 60; validation loss 1.522307 at epoch 1 to 1.485432 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-210636-d9406711/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-210636-d9406711/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-210636-d9406711/curves.png>).

### cifar10-20261025-213314-9c0e7949

**Architecture:** Normalized value/gate blocks, no skip, dropout 0. Width/hidden/depth: 512/1024/6; parameters: 11,061,770.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 97.0422%; validation 61.5200%; test 59.6700%; 11.4033 minutes. Selected epoch 54; completed epochs 60.

**Curve endpoints:** training loss 2.187845 at epoch 1 to 0.528838 at epoch 60; validation loss 1.798178 at epoch 1 to 1.475215 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-213314-9c0e7949/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-213314-9c0e7949/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261025-213314-9c0e7949/curves.png>).

### cifar10-20261026-131127-ba8e8b68

**Architecture:** Normalized value/gate blocks, no skip, dropout 0.1. Width/hidden/depth: 512/1024/6; parameters: 11,061,770.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 95.1200%; validation 63.5000%; test 60.8700%; 11.3476 minutes. Selected epoch 52; completed epochs 60.

**Curve endpoints:** training loss 2.303002 at epoch 1 to 0.648117 at epoch 60; validation loss 1.788643 at epoch 1 to 1.339543 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-131127-ba8e8b68/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-131127-ba8e8b68/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-131127-ba8e8b68/curves.png>).

### cifar10-20261026-151046-ee81c4cb

**Architecture:** Normalized value/gate blocks, no skip, dropout 0.15. Width/hidden/depth: 512/1024/6; parameters: 11,061,770.

**Training:** task `cifar10`, seed 42, 10 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 60.6178%; validation 56.3800%; test 55.3400%; 2.6668 minutes. Selected epoch 10; completed epochs 10.

**Curve endpoints:** training loss 2.389710 at epoch 1 to 1.368360 at epoch 10; validation loss 1.827088 at epoch 1 to 1.256800 at epoch 10. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-151046-ee81c4cb/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-151046-ee81c4cb/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-151046-ee81c4cb/curves.png>).

### cifar10-20261026-152251-5dfa7476

**Architecture:** Normalized value/gate blocks, no skip, dropout 0.1. Width/hidden/depth: 512/1024/6; parameters: 11,061,770.

**Training:** task `cifar10`, seed 42, 10 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 61.6800%; validation 56.6400%; test 55.6100%; 1.8841 minutes. Selected epoch 10; completed epochs 10.

**Curve endpoints:** training loss 2.303002 at epoch 1 to 1.344061 at epoch 10; validation loss 1.788643 at epoch 1 to 1.264035 at epoch 10. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-152251-5dfa7476/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-152251-5dfa7476/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-152251-5dfa7476/curves.png>).

### cifar10-20261026-152521-7a9bfc86

**Architecture:** Normalized value/gate blocks, no skip, dropout 0.1. Width/hidden/depth: 512/1024/6; parameters: 11,061,770.

**Training:** task `cifar10`, seed 42, 15 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 66.5511%; validation 58.5800%; test 57.4100%; 2.8223 minutes. Selected epoch 14; completed epochs 15.

**Curve endpoints:** training loss 2.303002 at epoch 1 to 1.229873 at epoch 15; validation loss 1.788643 at epoch 1 to 1.189914 at epoch 15. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-152521-7a9bfc86/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-152521-7a9bfc86/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-152521-7a9bfc86/curves.png>).

### cifar10-20261026-152851-db2487cc

**Architecture:** Normalized value/gate blocks, no skip, dropout 0.1. Width/hidden/depth: 512/1024/6; parameters: 11,061,770.

**Training:** task `cifar10`, seed 42, 16 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** No final summary; 0 completed epoch rows saved.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-152851-db2487cc/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-152851-db2487cc/metrics.csv>).

### cifar10-20261026-152951-e65d9ef5

**Architecture:** Normalized value/gate blocks, no skip, dropout 0.1. Width/hidden/depth: 512/1024/6; parameters: 11,061,770.

**Training:** task `cifar10`, seed 42, 20 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 72.8800%; validation 60.4200%; test 58.7000%; 3.7484 minutes. Selected epoch 20; completed epochs 20.

**Curve endpoints:** training loss 2.303002 at epoch 1 to 1.136440 at epoch 20; validation loss 1.788643 at epoch 1 to 1.165195 at epoch 20. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-152951-e65d9ef5/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-152951-e65d9ef5/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-152951-e65d9ef5/curves.png>).

### cifar10-20261026-153426-931b32ea

**Architecture:** Normalized value/gate blocks, no skip, dropout 0.1. Width/hidden/depth: 512/1024/6; parameters: 11,061,770.

**Training:** task `cifar10`, seed 42, 20 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 72.8800%; validation 60.4200%; test 58.7000%; 3.7896 minutes. Selected epoch 20; completed epochs 20.

**Curve endpoints:** training loss 2.303002 at epoch 1 to 1.136440 at epoch 20; validation loss 1.788643 at epoch 1 to 1.165195 at epoch 20. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-153426-931b32ea/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-153426-931b32ea/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-153426-931b32ea/curves.png>).

### imagenet32-20261026-195826-4b2dc142

**Architecture:** Normalized value/gate blocks, no skip, dropout 0.1. Width/hidden/depth: 512/1024/6; parameters: 11,569,640.

**Training:** task `imagenet32`, seed 42, 10 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `False`.

**Result:** clean train 18.7556%; validation 12.1632%; test 11.1060%; 42.2183 minutes. Selected epoch 10; completed epochs 10.

**Curve endpoints:** training loss 6.184651 at epoch 1 to 4.644329 at epoch 10; validation loss 5.829359 at epoch 1 to 4.963408 at epoch 10. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-20261026-195826-4b2dc142/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-20261026-195826-4b2dc142/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-20261026-195826-4b2dc142/curves.png>).

### imagenet32-20261026-204057-1f232b47

**Architecture:** Normalized value/gate blocks, no skip, dropout 0.1. Width/hidden/depth: 512/1024/6; parameters: 11,569,640.

**Training:** task `imagenet32`, seed 42, 10 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `False`.

**Result:** No final summary; 4 completed epoch rows saved.

**Curve endpoints:** training loss 6.184651 at epoch 1 to 5.263049 at epoch 4; validation loss 5.829359 at epoch 1 to 5.212148 at epoch 4. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-20261026-204057-1f232b47/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-20261026-204057-1f232b47/metrics.csv>).

### imagenet32-20261026-210325-b9c04e03

**Architecture:** Normalized value/gate blocks, no skip, dropout 0.1. Width/hidden/depth: 512/1024/6; parameters: 11,569,640.

**Training:** task `imagenet32`, seed 42, 20 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `False`.

**Result:** clean train 28.1384%; validation 12.7307%; test 11.6480%; 114.6338 minutes. Selected epoch 17; completed epochs 20.

**Curve endpoints:** training loss 6.184651 at epoch 1 to 3.953908 at epoch 20; validation loss 5.829359 at epoch 1 to 5.179975 at epoch 20. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-20261026-210325-b9c04e03/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-20261026-210325-b9c04e03/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-20261026-210325-b9c04e03/curves.png>).

### cifar10-20261026-215237-2ac3a7b3

**Architecture:** Rolling windows: small. Width/hidden/depth: 256/512/2; parameters: 812,410.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `AdamW`, learning rate 0.001, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 55.5444%; validation 49.9800%; test 48.2700%; 9.9126 minutes. Selected epoch 54; completed epochs 60.

**Curve endpoints:** training loss 2.389605 at epoch 1 to 1.386364 at epoch 60; validation loss 1.873977 at epoch 1 to 1.399145 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-215237-2ac3a7b3/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-215237-2ac3a7b3/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-215237-2ac3a7b3/curves.png>).

### cifar10-20261026-221344-5d653ac6

**Architecture:** Rolling windows: widened. Width/hidden/depth: 512/1024/2; parameters: 3,374,026.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `AdamW`, learning rate 0.001, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 71.3511%; validation 60.8000%; test 59.1500%; 14.8296 minutes. Selected epoch 57; completed epochs 60.

**Curve endpoints:** training loss 3.277407 at epoch 1 to 1.051925 at epoch 60; validation loss 1.973398 at epoch 1 to 1.153401 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-221344-5d653ac6/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-221344-5d653ac6/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-221344-5d653ac6/curves.png>).

### cifar10-20261026-224057-085c2c3b

**Architecture:** Rolling windows: full-image coverage correction. Width/hidden/depth: 512/1024/2; parameters: 3,415,178.

**Training:** task `cifar10`, seed 42, 60 configured epochs, batch 256, optimizer `AdamW`, learning rate 0.001, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 88.5156%; validation 76.1000%; test 75.4500%; 12.6071 minutes. Selected epoch 51; completed epochs 60.

**Curve endpoints:** training loss 3.459430 at epoch 1 to 0.443744 at epoch 60; validation loss 1.884440 at epoch 1 to 0.783493 at epoch 60. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-224057-085c2c3b/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-224057-085c2c3b/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/cifar10-20261026-224057-085c2c3b/curves.png>).

### imagenet32-20261027-000850-1c570f4f

**Architecture:** Rolling windows: padded, explicit stage modules (equivalent refactor). Width/hidden/depth: 512/1024/2; parameters: 3,923,048.

**Training:** task `imagenet32`, seed 42, 10 configured epochs, batch 256, optimizer `AdamW`, learning rate 0.001, weight decay 1e-05, augmentation enabled `False`.

**Result:** clean train 26.4322%; validation 20.4939%; test 18.4960%; 62.0101 minutes. Selected epoch 10; completed epochs 10.

**Curve endpoints:** training loss 5.691792 at epoch 1 to 3.826965 at epoch 10; validation loss 5.054592 at epoch 1 to 4.116780 at epoch 10. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-20261027-000850-1c570f4f/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-20261027-000850-1c570f4f/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-20261027-000850-1c570f4f/curves.png>).

### imagenet32-continuation-20261027-013419-8e7bc688

**Architecture:** Rolling windows: padded, explicit stage modules (equivalent refactor). Width/hidden/depth: 512/1024/2; parameters: 3,923,048.

**Training:** task `imagenet32`, seed 42, 100 configured epochs, batch 256, optimizer `AdamW`, learning rate 0.001, weight decay 1e-05, augmentation enabled `False`.

**Result:** clean train 37.1263%; validation 21.9809%; test 20.0040%; 134.6048 minutes. Selected epoch 25; completed epochs 25.

**Curve endpoints:** training loss 5.691792 at epoch 1 to 3.229355 at epoch 25; validation loss 5.054592 at epoch 1 to 4.105812 at epoch 25. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-continuation-20261027-013419-8e7bc688/config.json>) · [metrics](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-continuation-20261027-013419-8e7bc688/metrics.csv>) · [curves](</Users/taylor/Projects/AI4200-imagenet32x32-baseline-model/runs/imagenet32-continuation-20261027-013419-8e7bc688/curves.png>).

### mnist-20261027-105310-e7eed90e

**Architecture:** Rolling windows: padded, explicit stage modules (equivalent refactor). Width/hidden/depth: 512/1024/2; parameters: 3,414,154.

**Training:** task `mnist`, seed 42, 500 configured epochs, batch 256, optimizer `AdamW`, learning rate 0.001, weight decay 1e-05, augmentation enabled `False`.

**Result:** No final summary; 72 completed epoch rows saved.

**Curve endpoints:** training loss 1.810360 at epoch 1 to 0.000004 at epoch 72; validation loss 0.203701 at epoch 1 to 0.080584 at epoch 72. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/.t3/worktrees/AI4200-imagenet32x32-baseline-model/t3-0e1f69f0/runs/mnist-20261027-105310-e7eed90e/config.json>) · [metrics](</Users/taylor/.t3/worktrees/AI4200-imagenet32x32-baseline-model/t3-0e1f69f0/runs/mnist-20261027-105310-e7eed90e/metrics.csv>).

### mnist-20261027-111813-3f4dc443

**Architecture:** Rolling windows: padded, explicit stage modules (equivalent refactor). Width/hidden/depth: 512/1024/2; parameters: 3,414,154.

**Training:** task `mnist`, seed 42, 500 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** No final summary; 1 completed epoch rows saved.

**Curve endpoints:** training loss 0.793923 at epoch 1 to 0.793923 at epoch 1; validation loss 0.191342 at epoch 1 to 0.191342 at epoch 1. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/.t3/worktrees/AI4200-imagenet32x32-baseline-model/t3-0e1f69f0/runs/mnist-20261027-111813-3f4dc443/config.json>) · [metrics](</Users/taylor/.t3/worktrees/AI4200-imagenet32x32-baseline-model/t3-0e1f69f0/runs/mnist-20261027-111813-3f4dc443/metrics.csv>).

### mnist-20261027-111917-d3e67d1d

**Architecture:** Rolling windows: padded, explicit stage modules (equivalent refactor). Width/hidden/depth: 512/1024/2; parameters: 3,414,154.

**Training:** task `mnist`, seed 42, 20 configured epochs, batch 256, optimizer `Muon`, learning rate 0.03, weight decay 1e-05, augmentation enabled `True`.

**Result:** clean train 96.2463%; validation 95.7333%; test 95.7300%; 8.3334 minutes. Selected epoch 9; completed epochs 20.

**Curve endpoints:** training loss 0.793923 at epoch 1 to 2.304183 at epoch 20; validation loss 0.191342 at epoch 1 to 2.302797 at epoch 20. These endpoints do not describe every fluctuation; consult metrics/curves.

**Evidence:** [config](</Users/taylor/.t3/worktrees/AI4200-imagenet32x32-baseline-model/t3-0e1f69f0/runs/mnist-20261027-111917-d3e67d1d/config.json>) · [metrics](</Users/taylor/.t3/worktrees/AI4200-imagenet32x32-baseline-model/t3-0e1f69f0/runs/mnist-20261027-111917-d3e67d1d/metrics.csv>) · [curves](</Users/taylor/.t3/worktrees/AI4200-imagenet32x32-baseline-model/t3-0e1f69f0/runs/mnist-20261027-111917-d3e67d1d/curves.png>).
