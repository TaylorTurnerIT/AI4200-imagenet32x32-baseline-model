# Fresh instructor-harness evaluation — October 27, 2026

All three tasks passed the instructor's original evaluation cells, using fresh
weights, seed 0, CUDA, and a high-RAM NVIDIA A100-SXM4-80GB. Total measured
training time was **40.3339 minutes**, below the 90-minute limit. The Colab
runtime was terminated after downloading and verifying the evidence.

| Task | Test accuracy | Validation accuracy | Training minutes | Selected / completed epochs | Run ID |
| --- | ---: | ---: | ---: | --- | --- |
| MNIST | 99.150% | 99.3667% | 14.7003 | 270 / 382, time budget | `mnist-c628ed51` |
| CIFAR-10 | 74.760% | 76.6400% | 3.8194 | 56 / 60 | `cifar10-32ae8cde` |
| ImageNet32 | 19.960% | 22.0222% | 21.8142 | 23 / 25 | `imagenet32-8856b095` |

These are the final-report test scores from this evaluation. Historical runs
used other seeds, hardware, and stopping conditions. In particular, the
20.004% historical ImageNet continuation loaded prior weights; it remains
separate from this fresh 19.960% result.

## Exact source and execution

Qualified GitHub commit: `b5ab6e1f75de4cfc257cf85b7e2535cfa217934c`.
The package and harness files remain unchanged in the later evidence commit.
The installed package's source hashes matched this checkout exactly.

- [Executed instructor notebook](harness-output.ipynb): all ten code cells completed, no error outputs, and the three-task score table is present. The official CLI leaves `execution_count` null; completion is evidenced by the cell logs and saved outputs.
- [CLI output](cli-output.log), [precise results and installed origin](qualification.json), and [original harness CSV](harness-results.csv).
- [Runtime versions](runtime.json): Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0, 167.05 GiB system RAM.
- [Preparation checks](preparation.json): 73 tests, one optional visualization skip; seven clean-wheel tests; historical architecture equivalence checks.
- [Run records](run-records/): settings, recipe, epoch metrics, and training receipt for each run.
- [Artifact hashes](artifact-sha256.json): raw selected checkpoints are saved locally under `runs/<run ID>/`, excluded from Git. Graded `train()` never loads them.

The checked-in notebook differs from the upstream instructor notebook only in
the supported submission settings. It retains the original device selection,
data loading, seed, train/predict calls, output checks, and accuracy calculation.
The leaderboard nickname was empty, so this evaluation posted no scores.

The official CLI ran the notebook directly, with these existing environment
options:

```bash
colab --auth oauth2 --config /tmp/ai4200-session.json new \
  -s ai4200-final --gpu A100 --high-mem
colab --auth oauth2 --config /tmp/ai4200-session.json exec \
  -s ai4200-final -f project1_harness.ipynb --timeout 10800 \
  --env HARNESS_PACKAGE_URL=git+https://github.com/TaylorTurnerIT/AI4200-imagenet32x32-baseline-model.git@b5ab6e1f75de4cfc257cf85b7e2535cfa217934c \
  --env AI4200_EVIDENCE_DIR=/content/ai4200-evidence \
  --env AI4200_ANALYSIS_DIR=/content/ai4200-analysis
```

The diagnostic option is off by default. For this measurement run it was enabled
and its overhead was included in the harness's training timer. Download the
artifacts before running `colab ... stop -s ai4200-final`.

## ImageNet dynamics and ablations

[Fixed batch indices](analysis/fixed-batch.json) identify the same 32 training
examples for every eval-mode measurement. [Eight snapshots](analysis/training-dynamics.json)
cover initialization, epochs 1/5/10/15/20/25, and selected epoch 23.
[Layer indices](analysis/layer-index.json) map each plot position to its module.
Update ratios use one fixed-batch step on copied model and optimizer states;
they do not measure the live random minibatch update. GELU has no tanh/ReLU
saturation criterion.

The separate ablation script evaluated all 128,116 validation examples without
retraining. Restored full-model accuracy matched the selected checkpoint exactly.

| Residual classifier block | Ablated validation accuracy | Drop, percentage points | Selected branch/stream RMS |
| --- | ---: | ---: | ---: |
| 1 | 11.8588% | 10.1634 | 0.9455 |
| 2 | 18.4091% | 3.6131 | 0.6994 |

[Raw ablation results](analysis/ablation.json) and [collection output](analysis-output.log)
preserve the measurements. Window stages have no identity residual branch; the
two classifier blocks are the residual branches tested here. The report's
interpretation remains yours.

## Report images

[All images and export commands](../report-figures/README.md) include one accuracy
timeline per task, a CIFAR-10 curve comparison, ImageNet dynamics, and branch
ablation. Fresh harness results use separate markers in the timelines. Historical
continuation scores remain visibly identified; original evidence is preserved.
