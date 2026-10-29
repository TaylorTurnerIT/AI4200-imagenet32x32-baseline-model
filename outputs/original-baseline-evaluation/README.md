# Original ImageNet32 baseline — recovery run

The instructor’s original baseline completed one fresh run through the original harness on 2026-10-28. This fills the missing baseline measurement after development; it is not an October 15 result.

| Measure | Result |
| --- | ---: |
| Test accuracy | 6.766% |
| Correct / test examples | 3,383 / 50,000 |
| Training time | 15.42 minutes |
| Epochs completed | 10 |
| Training images | 1,281,167 |
| First-epoch training loss | 6.4107 |
| Final-epoch training loss | 5.3584 |
| Parameters | 2,668,904 |

Run ID: `imagenet32-originalbaseline-20261028-s0`. [Exact summary](summary.json), [original harness CSV](harness-results.csv), and [epoch losses](metrics.csv). Losses retain the original trainer’s four-decimal print precision. The supplied trainer uses no validation split and returns the last epoch; validation accuracy and training accuracy were not measured.

## Reproducible source and recipe

Source: [instructor repository](https://github.com/JesseTNRoberts/AI4200-imagenet32x32-baseline-model/tree/b56785a206069671420fbf5c6bbd6efb99f467b9), commit `b56785a206069671420fbf5c6bbd6efb99f467b9`. The model, trainer and package initializer match initial fork commit `c1f6802bf04a35fb12e1cd5429cb3faf74626df4` byte-for-byte.

| Setting | Value |
| --- | --- |
| Architecture | 30 residual blocks, width 128, hidden width 256, tanh, pre-LayerNorm |
| Initialization | Xavier normal; zero biases |
| Optimizer | SGD, learning rate 0.05, zero momentum and weight decay |
| Batch size / epochs / seed | 256 / 10 / 0 |
| Schedule / augmentation | Constant learning rate / none |
| Data | Full official training split; original normalization to [-1, 1] |
| Runtime | NVIDIA A100-SXM4-80GB, high RAM; PyTorch 2.11.0+cu130, CUDA 13.0 |

[Pre-run protocol](protocol.json) · [Installed source](source/) · [Configuration](config.json) · [Data hashes](data.json) · [Runtime](runtime.json).

## Harness and evidence

All ten code-cell sources in the [instructor notebook](project1_harness.ipynb) are unchanged in the [measurement notebook](baseline-harness.ipynb). Environment variables select ImageNet32 only, pin the package commit, and leave the leaderboard alias empty. Two added evidence cells verify source/data and archive the final weights after the training timer stops. The original `train()` function is never replaced. A print observer records its epoch losses; the original `predict()` performs the only test forward pass. Shape and probability-sum checks passed, along with finite/nonnegative checks.

[Executed notebook](baseline-harness_output.ipynb) · [Execution log](execution.log) · [Completed attempt](attempt.json) · [Verification receipt](verification.json).

Training-image and label hashes match the earlier final evaluation. The original baseline here differs from the “Baseline” in the extra-block experiment, which denotes the qualified rolling-window model.

The final checkpoint and test predictions are retained under `runs/experiments/imagenet32-originalbaseline-20261028-s0`, excluded from Git. [Local archive receipt](local-archive.json) records the checkpoint hash; [raw manifest](raw-run-artifact-sha256.json) covers that complete local directory. Graded training code and model defaults are unchanged.

The legacy CSV was extended by one row without rewriting its earlier bytes; [append receipt](csv-append-receipt.json). The original missing-baseline placeholder in Change 1 now points to this recovered measurement. The A100 was terminated after local archive verification; the server then reported no active sessions. [Runtime cleanup receipt](runtime-cleanup.json).
