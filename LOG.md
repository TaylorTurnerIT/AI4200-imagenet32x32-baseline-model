# Development Log — draft

Taylor Turner

Reconstructed from [journal.md](journal.md) and saved experiments. Original wording is reused in quotation marks. Factual descriptions and recorded results are draft scaffolding; `TODO` marks the parts to finish in my own words. Missing predictions have not been invented.

## Before finalizing

- Replace the `TODO` fields. If a prediction or reason is remembered now, label it retrospective; the position of text in the old journal does not establish when it was written.

## Era figures — design draft

Each task gets one accuracy timeline: a dot for each recorded result, a best-observed step line, and era boundaries. Low outliers are clipped with downward markers. The [interactive preview](outputs/experiment-history/accuracy-preview.html) also offers full-range and log-error scales.

![MNIST experiment accuracy](outputs/experiment-history/figures/mnist-accuracy.png)

![CIFAR-10 experiment accuracy](outputs/experiment-history/figures/cifar10-accuracy.png)

![ImageNet32 experiment accuracy](outputs/experiment-history/figures/imagenet32-accuracy.png)

## Baseline reference

### Change 1 — mnist / cifar10 / imagenet32 — 2026-10-15

_Original baseline_

**Change:** Run the supplied baseline: 30 residual blocks, width 128, hidden width 256, tanh, pre-LayerNorm, Xavier initialization and SGD at learning rate 0.05; batch 256. Baseline source: commit `c1f6802`. Configured epochs were MNIST 10, CIFAR-10 30 and ImageNet32 10.

**Why:** “I began by leaving the model architecture unchanged, only changing hyperparameters by hand to understand their impacts.” Aka, collect my baseline.

**Expected:** My initial thoughts had been that the beginning architecture would be immediately overfit and oversized.

**Result:**

| Dataset / record | Test accuracy | Training time |
|---|---:|---:|
| MNIST ([CSV row 2](results.csv)) | 95.65% | 1.1 minutes |
| CIFAR-10 ([CSV row 3](results.csv)) | 48.21% | 2.0 minutes |
| MNIST (journal baseline rerun; [CSV row 4](results.csv)) | 95.65% | 0.5 minutes |
| ImageNet32 (recovered 2026-10-28, seed 0) | 6.766% | 15.42 minutes |

The ImageNet32 baseline was recovered after development from the unchanged instructor source. Saved run: [`imagenet32-originalbaseline-20261028-s0`](outputs/original-baseline-evaluation/README.md). The historical MNIST and CIFAR-10 scores above come from the early CSV rows.

| ImageNet32 training measure | Result |
| --- | ---: |
| Epochs completed | 10 |
| First-epoch training loss | 6.4107 |
| Final-epoch training loss | 5.3584 |

TODO — add MNIST/CIFAR-10 baseline training-loss observations and confirm that the early CSV runs used the unchanged baseline.

## M1 — Baseline, capacity and regularization

### Change 2 — mnist — 2026-10-19

_Learning rate 0.01_

**Change:** Keep the baseline architecture and 10 epochs; set learning rate to 0.01, seed 42.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 95.49% |
| Training time | 0.6 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 5 (raw rows retained in the evidence snapshot).

### Change 3 — mnist — 2026-10-19

_Learning rate 0.10_

**Change:** Keep the baseline architecture and 10 epochs; set learning rate to 0.10, seed 42.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 96.74% |
| Training time | 0.5 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 6 (raw rows retained in the evidence snapshot).

### Change 4 — mnist — 2026-10-19

_Learning rate 0.15_

**Change:** Keep the baseline architecture and 10 epochs; set learning rate to 0.15, seed 42.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 96.11% |
| Training time | 0.5 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 7 (raw rows retained in the evidence snapshot).

### Change 5 — mnist — 2026-10-19

_50 epochs at learning rate 0.01_

**Change:** Set epochs to 50 and learning rate to 0.01; architecture unchanged.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.56% |
| Training time | 2.5 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 8 (raw rows retained in the evidence snapshot).

### Change 6 — mnist — 2026-10-19

_50 epochs at learning rate 0.10_

**Change:** Set epochs to 50 and learning rate to 0.10; architecture unchanged.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.12% |
| Training time | 2.4 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 9 (raw rows retained in the evidence snapshot).

### Change 7 — mnist — 2026-10-19

_50 epochs at learning rate 0.15_

**Change:** Set epochs to 50 and learning rate to 0.15; architecture unchanged.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.97% |
| Training time | 2.4 minutes |
| First loss spike | 0.5566 |
| Second loss spike | 0.2268 |

Journal reports that the loss spikes were followed by recovery. “We see here lots of big jumps followed by recovery. Not as good as a=0.10.”

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 10 (raw rows retained in the evidence snapshot).

### Change 8 — mnist — 2026-10-19

_Add a training-accuracy check_

**Change:** Measure training accuracy as well as test accuracy for the 50-epoch learning-rate comparisons.

**Why:** “I realized I didn't record train accuracy so I can't compare. Let's understand how train accuracy is looking.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

Journal results:

| Learning rate | Training accuracy | Test accuracy |
|---:|---:|---:|
| 0.05 | 100.00% | 98.12% |
| 0.10 | 99.97% | 98.10% |
| 0.15 | 99.94% | 97.97% |

| Measure | Result |
|---|---|
| Training time per setup | 2.4 minutes |

“We see that the baseline learning rate was perfect for this current configuration. All it truly needed was a few more epochs.” TODO — add a loss-curve comparison.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 11, 12, 13 (raw rows retained in the evidence snapshot).

**Draft notes:** The 0.15 CSV row appears to encode 99.93% train, while the journal says 99.94%. Preserve this discrepancy until checked.

### Change 9 — mnist — 2026-10-19

_40-layer comparison with 25 epochs_

**Change:** Journal records layers=40, epochs=25, seed=42, and `a=0.5`.

**Why:** “After my findings with accuracy revealed that the default learning rate was a solid starting point, I'm ready to move onto layers.”

**Expected:** “Right now I assume the issue is we're losing valuable signal as the layer layers add nothing but noise to an ultimately simple problem. I expect to see some small number of layers, perhaps 10-15 perform the same or better.”

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.79% |
| Training accuracy | 99.7% |
| Training time | 1.2 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 14 (raw rows retained in the evidence snapshot).

**Draft notes:** TODO — verify whether `a=0.5` was a typo for 0.05. The baseline had 30 blocks, while this entry says 40. Do not silently substitute either value.

### Change 10 — mnist — 2026-10-19

_Five layers_

**Change:** Reduce recorded depth from 40 to 5; keep 25 epochs and the preceding setup, subject to its learning-rate uncertainty.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** “I expect to see some small number of layers, perhaps 10-15 perform the same or better.”

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.89% |
| Training accuracy | 99.73% |
| Training time | 0.3 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. “An incredible gain! As predicted we actually see a performance increase by decreasing the layers and a gain of 1.2 minutes”

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 15 (raw rows retained in the evidence snapshot).

**Draft notes:** Journal’s claimed 1.2-minute gain conflicts with the recorded 1.2-to-0.3-minute change (0.9 minutes). TODO — revise the interpretation.

### Change 11 — mnist — 2026-10-19

_1 layers_

**Change:** Set depth to 1; retain the 25-epoch depth-comparison setup.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.73% |
| Training accuracy | 99.06% |
| Training time | 0.4 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 16 (raw rows retained in the evidence snapshot).

### Change 12 — mnist — 2026-10-19

_2 layers_

**Change:** Set depth to 2; retain the 25-epoch depth-comparison setup.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.77% |
| Training accuracy | 99.09% |
| Training time | 0.2 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 17 (raw rows retained in the evidence snapshot).

### Change 13 — mnist — 2026-10-19

_3 layers_

**Change:** Set depth to 3; retain the 25-epoch depth-comparison setup.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.8% |
| Training accuracy | 99.57% |
| Training time | 0.3 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 18 (raw rows retained in the evidence snapshot).

### Change 14 — mnist — 2026-10-19

_4 layers_

**Change:** Set depth to 4; retain the 25-epoch depth-comparison setup.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.92% |
| Training accuracy | 99.67% |
| Training time | 0.9 minutes |
| Loss milestone | 0.0x range within approximately seven epochs |

TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 19 (raw rows retained in the evidence snapshot).

### Change 15 — mnist — 2026-10-19

_6 layers_

**Change:** Set depth to 6; retain the 25-epoch depth-comparison setup.

**Why:** “For now, let's try a few more until we see a performance regression.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.03% |
| Training accuracy | 99.79% |
| Training time | 0.4 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 20 (raw rows retained in the evidence snapshot).

### Change 16 — mnist — 2026-10-19

_7 layers_

**Change:** Set depth to 7; retain the 25-epoch depth-comparison setup.

**Why:** “For now, let's try a few more until we see a performance regression.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.6% |
| Training accuracy | 99.67% |
| Training time | 0.5 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. “Regression. Let's roll with 6 layers for now. Going to try a run with 50 epochs and 6 layers.”

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 21 (raw rows retained in the evidence snapshot).

### Change 17 — mnist — 2026-10-19

_50 epochs after depth sweep_

**Change:** Set epochs to 50; recorded depth 6, probably actual depth 7.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.01% |
| Training accuracy | 100% |
| Training time | 0.9 minutes |
| Last four recorded losses (in order) | 0.0012, 0.0011, 0.0010, 0.0010 |

“Zero change in the last epoch. Clearly reached the floor of this configuration.”

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 22 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 18 — mnist — 2026-10-19

_Residual width 32_

**Change:** Set width=32, hidden=256, epochs=25, learning_rate=0.05; recorded depth 6, probably 7.

**Why:** “My ideas here are to change the size of the residual stream (width) and the size of the hidden layers (hidden).”

**Expected:** “Perhaps a smaller residual stream is equally capable, perhaps a larger residual layer will allow more signal through.”

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 95.92% |
| Training accuracy | 96.93% |
| Training time | 0.4 minutes |

“Immediately slower training. Didn't reach 0.0# until epoch 18.” TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 23 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 19 — mnist — 2026-10-19

_Residual width 256_

**Change:** Set width=256, hidden=256, epochs=25, learning_rate=0.05; recorded depth 6, probably 7.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** “perhaps a larger residual layer will allow more signal through.”

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.04% |
| Training accuracy | 99.99% |
| Training time | 0.5 minutes |
| Loss milestone | 0.0x range in six epochs |

“Incredible training accuracy but a trivial improvement in test. All we did was improve the models ability to overfit.”

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 24 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 20 — mnist — 2026-10-19

_50 epochs at width 256_

**Change:** Keep width=256 and hidden=256; extend training from 25 to 50 epochs.

**Why:** “Let's try a 50 epoch here since I see a bit more room for training loss.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.01% |
| Training accuracy | 100% |
| Training time | 0.9 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. “We gained an incredible -0.03%. I'd say that was just further overfitting pushing our performance worse.”

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 25 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 21 — mnist — 2026-10-19

_Hidden width 512_

**Change:** Set width=256, hidden=512, epochs=50, learning_rate=0.05; recorded depth 6, probably 7.

**Why:** “For now, let's see what happens when we tinker with the size of the hidden layers. Perhaps reducing the models capacity will serve a similar role.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.17% |
| Training accuracy | 100% |
| Training time | 1.0 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 26 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown. Journal settings increase hidden width, although the preceding idea mentions reducing capacity. TODO — explain the intended comparison. The printed result is 98.17%, while the following sentence says 98.18%.

### Change 22 — mnist — 2026-10-19

_Residual width 512_

**Change:** Set width=512, hidden=512, epochs=50, learning_rate=0.05; recorded depth 6, probably 7.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.16% |
| Training accuracy | 100% |
| Training time | 1.1 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 27 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 23 — mnist — 2026-10-19

_Hidden width 1024_

**Change:** Set width=512, hidden=1024, epochs=50, learning_rate=0.05; recorded depth 6, probably 7.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.15% |
| Training accuracy | 100% |
| Training time | 1.4 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. “Crazy experiments over with no real change. This is obviously a dead end. Onto Regularization.”

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 28 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 24 — mnist — 2026-10-19

_Weight decay 0.99_

**Change:** Restore width=256 and hidden=512; set weight_decay=0.99, learning_rate=0.05, epochs=50; recorded depth 6, probably 7.

**Why:** “Weight decay is a classic from ML. Let's try to force down those worthless weights and see if we can open up some capacity for the test accuracy to use.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 11.35% |
| Training accuracy | 11.24% |
| Training time | 1.0 minutes |
| Loss change across 50 epochs | about 0.0001 |

“Didn't learn a thing.”

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 29 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown. TODO — explain the SGD weight-decay update correctly. Weight decay 0.99 is not “99% of squared magnitude allowed to affect weight decay.”

### Change 25 — mnist — 2026-10-19

_Weight decay 0.01_

**Change:** Reduce weight_decay from 0.99 to 0.01; set epochs=25; width=256, hidden=512, learning_rate=0.05.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 87.79% |
| Training accuracy | 86.78% |
| Training time | 0.5 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 30 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 26 — mnist — 2026-10-19

_Weight decay 0.0001_

**Change:** Set weight_decay=0.0001; keep 25 epochs, width=256, hidden=512, learning_rate=0.05.

**Why:** “Let's try suggestion from the internet, 1e-4.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.09% |
| Training accuracy | 99.96% |
| Training time | 0.5 minutes |
| Recorded loss epochs | 22–25 |
| Recorded losses (in order) | 0.0093, 0.0077, 0.0060, 0.0047 |

TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 31 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown. TODO — identify and cite the source of the weight-decay suggestion if you remember it.

### Change 27 — mnist — 2026-10-19

_50 epochs with weight decay 0.0001_

**Change:** Extend the regularized configuration to 50 epochs; width=256, hidden=512, learning_rate=0.05.

**Why:** “This is interesting, we're almost back where we started but we still have a little wiggle room with train accuracy. I still see some more downward motion in the loss as well.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.24% |
| Training accuracy | 100% |
| Training time | 1.0 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. “The plan paid off! We closed the train accuracy to 100% again and gained 0.10 of test accuracy.”

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 32 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 28 — mnist — 2026-10-19

_Weight decay 0.001_

**Change:** Increase weight_decay to 0.001; keep 50 epochs, width=256, hidden=512, learning_rate=0.05.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.92% |
| Training accuracy | 99.27% |
| Training time | 1.0 minutes |

Journal reports oscillation. “Tried another jump in regularization and I'm starting to see a lot of oscillation. We've lost test accuracy in exchange for some train room.”

These values match the old row’s shifted layout. Journal reports oscillation.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 33 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 29 — mnist — 2026-10-19

_Learning rate 0.03 with decay 0.001_

**Change:** Set learning_rate=0.03; weight_decay=0.001, epochs=50, width=256, hidden=512.

**Why:** “We've lost test accuracy in exchange for some train room. Let me try a smaller learning rate.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.06% |
| Training accuracy | 99.66% |
| Training time | 1.0 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. “a=0.03 worked. I saw oscillation happening and it predictably solved it.”

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 34 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 30 — mnist — 2026-10-19

_Learning rate 0.01 with decay 0.001_

**Change:** Set learning_rate=0.01; weight_decay=0.001, epochs=50, width=256, hidden=512.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.97% |
| Training accuracy | 99.91% |
| Training time | 1.0 minutes |

“With 0.01 we're seeing incredibly slow training with more room for training by epoch 50.” TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 35 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

## M2 — Momentum and depth corrections

### Change 31 — mnist — 2026-10-20

_SGD momentum 0.9_

**Change:** Set momentum=0.9; learning_rate=0.01, weight_decay=0.001, epochs=50, width=256, hidden=512.

**Why:** “We're already using SGD, I have no interest in trying classical GD due to time and probable worse performance. So, let's try just adding some momentum and see what happens.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 96.66% |
| Training accuracy | 98.25% |
| Training time | 1.1 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. “Well that didn't work. Let's try less.”

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 36 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 32 — mnist — 2026-10-20

_SGD momentum 0.5_

**Change:** Reduce momentum to 0.5; keep learning_rate=0.01, weight_decay=0.001, epochs=50, width=256, hidden=512.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98% |
| Training accuracy | 100% |
| Training time | 1.1 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 37 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 33 — mnist — 2026-10-20

_Weight decay 0.005 with momentum 0.5_

**Change:** Set weight_decay=0.005; momentum=0.5, learning_rate=0.01, epochs=50, width=256, hidden=512.

**Why:** “Well that kinda worked. Maybe more regularization?”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 94.08% |
| Training accuracy | 94.67% |
| Training time | 1.1 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. “Huge overshoot. Let's try less momentum then bring in dampening. Could also try a smaller learning rate.”

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 38 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 34 — mnist — 2026-10-20

_Momentum 0.3 with decay 0.005_

**Change:** Set momentum=0.3; weight_decay=0.005, batch=256, learning_rate=0.01, epochs=50, width=256, hidden=512.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 96.65% |
| Training accuracy | 97.73% |
| Training time | 1.1 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 39 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 35 — mnist — 2026-10-20

_Batch 128 and weight decay 0.001_

**Change:** Set batch=128 and weight_decay=0.001; momentum=0.3, learning_rate=0.01, epochs=50, width=256, hidden=512.

**Why:** “Doing something else as a waiting move. Changing weight decay back down and turning down batch size so it must generalize more.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 96.36% |
| Training accuracy | 98.41% |
| Training time | 2.0 minutes |

“Still oscillating.” TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 42 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown. The nearby CSV-only result at 11:42 is not assigned this setup because the journal does not identify it.

### Change 36 — mnist — 2026-10-20

_Learning rate 0.001 and momentum 0.9_

**Change:** Set learning_rate=0.001 and momentum=0.9; batch=256, weight_decay=0.001, epochs=50, width=256, hidden=512.

**Why:** “https://distill.pub/2017/momentum/ This post opened my eyes. Let's experiment. Maybe our learning rate is too high? Let's try more momentum, less learning rate.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.76% |
| Training accuracy | 99.96% |
| Training time | 1.2 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. “Slow training, okay results.”

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 43 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 37 — mnist — 2026-10-20

_Batch 128 at learning rate 0.001_

**Change:** Reduce batch to 128; learning_rate=0.001, momentum=0.9, weight_decay=0.001, epochs=50, width=256, hidden=512.

**Why:** “Slow training, okay results. More batch to make more little jumps?”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.85% |
| Training accuracy | 99.93% |
| Training time | 2.0 minutes |

“Faster convergence, seeing some oscillation.” TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 44 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown. TODO — revise the explanation: a smaller batch means more optimizer updates per epoch, not a larger batch.

### Change 38 — mnist — 2026-10-20

_Learning rate 0.0001_

**Change:** Set learning_rate=0.0001; batch=128, momentum=0.9, weight_decay=0.001, epochs=50, width=256, hidden=512.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.94% |
| Training accuracy | 100% |
| Training time | 2.9 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 45 (raw rows retained in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 39 — mnist — 2026-10-20

_Accidental weight decay 0.00001_

**Change:** Set depth=7, width=256, hidden=512, batch=256, learning_rate=0.05, epochs=50; accidentally enter weight_decay=0.1e-4 (=0.00001).

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.22% |
| Training accuracy | 100% |
| Training time | 1.0 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. “I actually made a mistake, I forgot to remove the 0. before the 1 in weight_decay and I got this 98.22%.”

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 47 (raw rows retained in the evidence snapshot).

**Draft notes:** The preceding 12:08 CSV result also scored 98.22%, but its settings are not independently recorded. It remains in the unresolved list.

### Change 40 — mnist — 2026-10-20

_Correct decay to 0.0001_

**Change:** Keep depth=7, width=256, hidden=512, batch=256, learning_rate=0.05, epochs=50; set weight_decay=0.0001.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.16% |
| Training accuracy | 100% |
| Training time | 0.9 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 48 (raw rows retained in the evidence snapshot).

### Change 41 — mnist — 2026-10-20

_Explicitly test depth 6_

**Change:** Change depth from 7 to 6; width=256, hidden=512, batch=256, learning_rate=0.05, epochs=50, weight_decay=0.0001.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.24% |
| Training accuracy | 100% |
| Training time | 1.0 minutes |

TODO — add the loss-curve observation; no per-epoch artifact was located for this early run. TODO — explain whether this matched your expectation.

These values match the old row’s shifted layout.
**Evidence:** [journal.md](journal.md); legacy CSV line(s) 49 (raw rows retained in the evidence snapshot).

**Draft notes:** Later retrospective replay evidence in the journal associates 98.24% with seven blocks and 98.16% with six. These contemporary labels say the reverse. TODO — reconcile; do not silently relabel either historical result.

### Change 42 — mnist — 2026-10-20

_Unmatched legacy result at 11:42_

**Change:** TODO — identify the changed setup or mark this as a repeat of another entry. No run directory or exact configuration was saved with this row.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 97.65% |

The CSV column layout is shifted; printed time/train values require review. TODO — add confirmed training time, train accuracy, loss behavior and interpretation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 40 (raw rows retained in the evidence snapshot).

### Change 43 — mnist — 2026-10-20

_Unmatched legacy result at 11:44_

**Change:** TODO — identify the changed setup or mark this as a repeat of another entry. No run directory or exact configuration was saved with this row.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 96.36% |

The CSV column layout is shifted; printed time/train values require review. TODO — add confirmed training time, train accuracy, loss behavior and interpretation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 41 (raw rows retained in the evidence snapshot).

### Change 44 — mnist — 2026-10-20

_Unmatched legacy result at 12:08_

**Change:** TODO — identify the changed setup or mark this as a repeat of another entry. No run directory or exact configuration was saved with this row.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.22% |

The CSV column layout is shifted; printed time/train values require review. TODO — add confirmed training time, train accuracy, loss behavior and interpretation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 46 (raw rows retained in the evidence snapshot).

### Change 45 — mnist — 2026-10-20

_Unmatched legacy result at 12:40_

**Change:** TODO — identify the changed setup or mark this as a repeat of another entry. No run directory or exact configuration was saved with this row.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.18% |

The CSV column layout is shifted; printed time/train values require review. TODO — add confirmed training time, train accuracy, loss behavior and interpretation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 50 (raw rows retained in the evidence snapshot).

## M3 — Longer training, validation and augmentation

### Change 46 — mnist — 2026-10-22

_Unmatched legacy result at 14:49_

**Change:** TODO — identify the changed setup or mark this as a repeat of another entry. No run directory or exact configuration was saved with this row.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.18% |

The CSV column layout is shifted; printed time/train values require review. TODO — add confirmed training time, train accuracy, loss behavior and interpretation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 51 (raw rows retained in the evidence snapshot).

### Change 47 — mnist — 2026-10-22

_Unmatched legacy result at 14:54_

**Change:** TODO — identify the changed setup or mark this as a repeat of another entry. No run directory or exact configuration was saved with this row.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.18% |

The CSV column layout is shifted; printed time/train values require review. TODO — add confirmed training time, train accuracy, loss behavior and interpretation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 52 (raw rows retained in the evidence snapshot).

### Change 48 — mnist — 2026-10-22

_Unmatched legacy result at 15:33_

**Change:** TODO — identify the changed setup or mark this as a repeat of another entry. No run directory or exact configuration was saved with this row.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.37% |

The CSV column layout is shifted; printed time/train values require review. TODO — add confirmed training time, train accuracy, loss behavior and interpretation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 53 (raw rows retained in the evidence snapshot).

### Change 49 — mnist — 2026-10-22

_Unmatched legacy result at 17:30_

**Change:** TODO — identify the changed setup or mark this as a repeat of another entry. No run directory or exact configuration was saved with this row.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.07% |

The CSV column layout is shifted; printed time/train values require review. TODO — add confirmed training time, train accuracy, loss behavior and interpretation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 54 (raw rows retained in the evidence snapshot).

### Change 50 — mnist — 2026-10-22

_Unmatched legacy result at 17:38_

**Change:** TODO — identify the changed setup or mark this as a repeat of another entry. No run directory or exact configuration was saved with this row.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.24% |
| CSV warning | Column layout is shifted; printed time/train values require review |

TODO — add confirmed training time, train accuracy, loss behavior and interpretation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 55 (raw rows retained in the evidence snapshot).

### Change 51 — mnist — 2026-10-22

_Unmatched legacy result at 17:48_

**Change:** TODO — identify the changed setup or mark this as a repeat of another entry. No run directory or exact configuration was saved with this row.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 97.86% |
| CSV warning | Column layout is shifted; printed time/train values require review |

TODO — add confirmed training time, train accuracy, loss behavior and interpretation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 56 (raw rows retained in the evidence snapshot).

### Change 52 — mnist — 2026-10-22

_Unmatched legacy result at 17:49_

**Change:** TODO — identify the changed setup or mark this as a repeat of another entry. No run directory or exact configuration was saved with this row.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.24% |
| CSV warning | Column layout is shifted; printed time/train values require review |

TODO — add confirmed training time, train accuracy, loss behavior and interpretation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 57 (raw rows retained in the evidence snapshot).

### Change 53 — mnist — 2026-10-22

_Unmatched legacy result at 21:34_

**Change:** TODO — identify the changed setup or mark this as a repeat of another entry. No run directory or exact configuration was saved with this row.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.09% |
| CSV warning | Column layout is shifted; printed time/train values require review |

TODO — add confirmed training time, train accuracy, loss behavior and interpretation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 58 (raw rows retained in the evidence snapshot).

### Change 54 — mnist — 2026-10-22

_Unmatched legacy result at 21:47_

**Change:** TODO — identify the changed setup or mark this as a repeat of another entry. No run directory or exact configuration was saved with this row.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.37% |
| CSV warning | Column layout is shifted; printed time/train values require review |

TODO — add confirmed training time, train accuracy, loss behavior and interpretation.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 59 (raw rows retained in the evidence snapshot).

### Change 55 — mnist — 2026-10-22

_Validation-based checkpoint selection_

**Change:** legacy MLP (module shapes saved; forward source not captured). Exact module shapes are in the saved configuration; activation and forward source were not captured. Optimizer=not recorded, lr=0.05, weight_decay=0.0001, batch=256, epochs=500, seed=42; augmentation=not recorded. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “When I implemented augmentation, I used Codex and it made an executive decision I was initially confused by. It created a train/test/validation split.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261022-220617-e4881f95 | 100% | 98.48% | 98.16% | 9.10 minutes | 368 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.6946 → 0.0002703 | 0.0002227 (278) | 0.3219 → 0.07032 | 0.05885 (358) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [mnist-20261022-220617-e4881f95](outputs/experiment-history/run-index.md#mnist-20261022-220617-e4881f95).

**Draft notes:** Legacy snapshot: activation/forward code is missing. Do not infer it from a later restored version.

### Change 56 — mnist — 2026-10-22

_Legacy mlp (module shapes saved; forward source not captured)_

**Change:** legacy MLP (module shapes saved; forward source not captured). Exact module shapes are in the saved configuration; activation and forward source were not captured. Optimizer=not recorded, lr=0.01, weight_decay=0.0001, batch=256, epochs=1000, seed=42; augmentation=not recorded. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261022-230308-ed671f61 | — | — | — | — | — | 143 completed epoch rows; no final score |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.6247 → 0.000658 | 0.000658 (143) | 0.3862 → 0.09722 | 0.08053 (42) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [mnist-20261022-230308-ed671f61](outputs/experiment-history/run-index.md#mnist-20261022-230308-ed671f61).

**Draft notes:** Legacy snapshot: activation/forward code is missing. Do not infer it from a later restored version.

### Change 57 — mnist — 2026-10-22

_Legacy mlp (module shapes saved; forward source not captured)_

**Change:** legacy MLP (module shapes saved; forward source not captured). Exact module shapes are in the saved configuration; activation and forward source were not captured. Optimizer=not recorded, lr=0.03, weight_decay=0.0001, batch=256, epochs=1000, seed=42; augmentation=not recorded. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261022-230552-e687600f | 100% | 98.42% | 98.06% | 18.50 minutes | 653 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.5453 → 0.000259 | 0.0002077 (341) | 0.3552 → 0.08314 | 0.06415 (615) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [mnist-20261022-230552-e687600f](outputs/experiment-history/run-index.md#mnist-20261022-230552-e687600f).

**Draft notes:** Legacy snapshot: activation/forward code is missing. Do not infer it from a later restored version.

### Change 58 — mnist — 2026-10-23

_MNIST data augmentation_

**Change:** legacy MLP (module shapes saved; forward source not captured). Exact module shapes are in the saved configuration; activation and forward source were not captured. Optimizer=not recorded, lr=0.05, weight_decay=0.0001, batch=256, epochs=400, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “Then, also building from what we did in class I decided to add data augmentation. Move the image around, blur it, brighten it, etc.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261023-001324-8a4087e7 | 99.97% | 99.32% | 99.15% | 13.70 minutes | 315 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.095 → 0.01925 | 0.0156 (357) | 0.4271 → 0.02562 | 0.02282 (323) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [mnist-20261023-001324-8a4087e7](outputs/experiment-history/run-index.md#mnist-20261023-001324-8a4087e7).

**Draft notes:** Legacy snapshot: activation/forward code is missing. Do not infer it from a later restored version. Journal later mentions 99.32% at epoch 384. This run records 99.32% validation, 99.15% test and selected epoch 315. TODO — identify the intended run/metric; do not treat these as the same observation.

### Change 59 — mnist — 2026-10-23

_Legacy mlp (module shapes saved; forward source not captured)_

**Change:** legacy MLP (module shapes saved; forward source not captured). Exact module shapes are in the saved configuration; activation and forward source were not captured. Optimizer=not recorded, lr=0.05, weight_decay=0.0001, batch=256, epochs=500, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261023-100855-d6864421 | 99.97% | 99.32% | 99.15% | 17.40 minutes | 315 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.095 → 0.01711 | 0.0156 (357) | 0.4271 → 0.02901 | 0.02282 (323) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [mnist-20261023-100855-d6864421](outputs/experiment-history/run-index.md#mnist-20261023-100855-d6864421).

**Draft notes:** Legacy snapshot: activation/forward code is missing. Do not infer it from a later restored version.

### Change 60 — mnist — 2026-10-23

_Legacy mlp (module shapes saved; forward source not captured)_

**Change:** legacy MLP (module shapes saved; forward source not captured). Exact module shapes are in the saved configuration; activation and forward source were not captured. Optimizer=not recorded, lr=0.05, weight_decay=0.0001, batch=256, epochs=500, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261023-105208-872d486e | 100% | 99.33% | 99.23% | 17.30 minutes | 317 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.7405 → 0.01587 | 0.01001 (357) | 0.207 → 0.02676 | 0.02371 (481) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [mnist-20261023-105208-872d486e](outputs/experiment-history/run-index.md#mnist-20261023-105208-872d486e).

**Draft notes:** Legacy snapshot: activation/forward code is missing. Do not infer it from a later restored version.

### Change 61 — mnist — 2026-10-23

_Legacy mlp (module shapes saved; forward source not captured)_

**Change:** legacy MLP (module shapes saved; forward source not captured). Exact module shapes are in the saved configuration; activation and forward source were not captured. Optimizer=not recorded, lr=0.05, weight_decay=0.0001, batch=128, epochs=500, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261023-122327-464bd648 | — | — | — | — | — | 50 completed epoch rows; no final score; no completed epoch metrics were saved |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.724 → 0.04265 | 0.04129 (49) | 0.2221 → 0.03736 | 0.03457 (37) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [mnist-20261023-122327-464bd648](outputs/experiment-history/run-index.md#mnist-20261023-122327-464bd648).

**Draft notes:** Legacy snapshot: activation/forward code is missing. Do not infer it from a later restored version.

### Change 62 — mnist — 2026-10-23

_Flat residual mlp_

**Change:** flat residual MLP. Width=128, hidden=128, depth=7; recorded operations relu. Optimizer=SGD, lr=0.05, weight_decay=1e-05, batch=128, epochs=500, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261023-123641-6d74404c | — | — | — | — | — | 3 completed epoch rows; no final score |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.4734 → 0.1302 | 0.1302 (3) | 0.2188 → 0.1847 | 0.1786 (2) |

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261023-134613-19c0d4f0 | — | — | — | — | — | 24 completed epoch rows; no final score |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.4734 → 0.001914 | 0.001914 (24) | 0.2188 → 0.07113 | 0.06862 (21) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [mnist-20261023-123641-6d74404c](outputs/experiment-history/run-index.md#mnist-20261023-123641-6d74404c), [mnist-20261023-134613-19c0d4f0](outputs/experiment-history/run-index.md#mnist-20261023-134613-19c0d4f0).

### Change 63 — mnist — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=128, hidden=128, depth=7; recorded operations relu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=500, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261024-181654-70e55ef9 | 100% | 99% | 98.62% | 13.04 minutes | 372 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.3796 → 2.559e-09 | 8.389e-11 (215) | 0.1722 → 0.108 | 0.06154 (101) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [mnist-20261024-181654-70e55ef9](outputs/experiment-history/run-index.md#mnist-20261024-181654-70e55ef9).

### Change 64 — mnist — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=256, hidden=512, depth=50; recorded operations silu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=500, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261024-191704-b84eb60a | — | — | — | — | — | 2 completed epoch rows; no final score |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.6104 → 0.203 | 0.203 (2) | 0.283 → 0.1754 | 0.1754 (2) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [mnist-20261024-191704-b84eb60a](outputs/experiment-history/run-index.md#mnist-20261024-191704-b84eb60a).

### Change 65 — mnist — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=64, hidden=512, depth=3; recorded operations silu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=500, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261024-194929-07c1eda8 | — | — | — | — | — | 14 completed epoch rows; no final score |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.3918 → 0.03621 | 0.03621 (14) | 0.2342 → 0.08784 | 0.07267 (9) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [mnist-20261024-194929-07c1eda8](outputs/experiment-history/run-index.md#mnist-20261024-194929-07c1eda8).

### Change 66 — mnist — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=128, hidden=512, depth=50; recorded operations silu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=128, epochs=500, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261024-195929-3a9805da | — | — | — | — | — | 2 completed epoch rows; no final score |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.4511 → 0.2075 | 0.2075 (2) | 0.2483 → 0.1645 | 0.1645 (2) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [mnist-20261024-195929-3a9805da](outputs/experiment-history/run-index.md#mnist-20261024-195929-3a9805da).

### Change 67 — mnist — date not recorded

_500-epoch double-descent experiment_

**Change:** Journal settings: depth=6, width=256, hidden=512, batch=256, learning_rate=0.05, epochs=500, weight_decay=0.0001.

**Why:** “Today in class we discussed double descent. So, I tried training MORE with what worked.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.32% |
| Training time | Not recorded |
| Run ID | Not recorded |

“And it worked!” TODO — recover the run and describe its loss curve. Nearby CSV-only 500-epoch-looking results cannot be safely equated to this claim.

### Change 68 — mnist — date not recorded

_5000-epoch experiment_

**Change:** Journal says training was extended to 5000 epochs; exact saved setup is not linked.

**Why:** “And it worked! I tried with 5000 too but that didn't work.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Journal result | “Its performance was incredibly variable with loss so I decided to try checking whatever epoch was the winner.” |

TODO — add the actual score and time, identify the checkpoint-selection rule, and link the run. Do not assume the long CSV-only run is this experiment.

### Change 69 — mnist — date not recorded

_GELU experiment before CIFAR-10_

**Change:** Journal says all activations were changed to GELU. No exact MNIST run ID is securely linked.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result |
|---|---|
| Journal result | “Then I tried setting all of the activations to GELU. Slight regression so I am finally moving on to CIFAR.” |

TODO — identify the score, time and loss curve. Later module-only snapshots do not capture the activation function.

## M4 — Rolling-window transfer

### Change 70 — mnist — 2026-10-27

_Hierarchical 2×2 rolling-window mlp with padded full-image coverage_

**Change:** hierarchical 2×2 rolling-window MLP with padded full-image coverage. Width=512, hidden=1024, depth=2; recorded operations gelu, pad. Window MLP hidden/output=128/64; spatial sizes 28 → 27 → 14 → 7 → 4 → 2 → 1. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=500, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261027-105310-e7eed90e | — | — | — | — | — | 72 completed epoch rows; no final score |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.81 → 4.343e-06 | 4.133e-06 (70) | 0.2037 → 0.08058 | 0.05907 (11) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [mnist-20261027-105310-e7eed90e](outputs/experiment-history/run-index.md#mnist-20261027-105310-e7eed90e).

### Change 71 — mnist — 2026-10-27

_Hierarchical 2×2 rolling-window mlp with padded full-image coverage_

**Change:** hierarchical 2×2 rolling-window MLP with padded full-image coverage. Width=512, hidden=1024, depth=2; recorded operations gelu, pad. Window MLP hidden/output=128/64; spatial sizes 28 → 27 → 14 → 7 → 4 → 2 → 1. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=500, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261027-111813-3f4dc443 | — | — | — | — | — | 1 completed epoch rows; no final score |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.7939 → 0.7939 | 0.7939 (1) | 0.1913 → 0.1913 | 0.1913 (1) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [mnist-20261027-111813-3f4dc443](outputs/experiment-history/run-index.md#mnist-20261027-111813-3f4dc443).

### Change 72 — mnist — 2026-10-27

_Hierarchical 2×2 rolling-window mlp with padded full-image coverage_

**Change:** hierarchical 2×2 rolling-window MLP with padded full-image coverage. Width=512, hidden=1024, depth=2; recorded operations gelu, pad. Window MLP hidden/output=128/64; spatial sizes 28 → 27 → 14 → 7 → 4 → 2 → 1. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=20, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261027-111917-d3e67d1d | 96.2463% | 95.7333% | 95.73% | 8.33 minutes | 9 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.7939 → 2.304 | 0.2109 (8) | 0.1913 → 2.303 | 0.1592 (4) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [mnist-20261027-111917-d3e67d1d](outputs/experiment-history/run-index.md#mnist-20261027-111917-d3e67d1d).

## C1 — Flat MLP exploration

### Change 73 — cifar10 — 2026-10-23

_Transfer MNIST configuration to CIFAR-10_

**Change:** legacy MLP (module shapes saved; forward source not captured). Exact module shapes are in the saved configuration; activation and forward source were not captured. Optimizer=not recorded, lr=0.05, weight_decay=0.0001, batch=256, epochs=30, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261023-112130-b25190a4 | 99.99% | 56.4% | 54.59% | 0.50 minutes | 28 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.68 → 0.00421 | 0.00421 (30) | 1.559 → 2.498 | 1.312 (5) |

Journal interpretation: “Crazy stuff. Instantly pulls out 99.99% train accuracy? 54% test? This indicates to me that it has become so difficult to learn generalizable patterns that it learned what it could then instantly memorized.” TODO — check this interpretation against the saved configuration and validation result.

**Runs:** [cifar10-20261023-112130-b25190a4](outputs/experiment-history/run-index.md#cifar10-20261023-112130-b25190a4).

**Draft notes:** Legacy snapshot: activation/forward code is missing. Do not infer it from a later restored version. Journal says layers=6 and 500 epochs; snapshot records seven blocks and 30 epochs. Use the saved settings for this run.

### Change 74 — cifar10 — 2026-10-23

_CIFAR activation comparison (legacy metadata)_

**Change:** legacy MLP (module shapes saved; forward source not captured). Exact module shapes are in the saved configuration; activation and forward source were not captured. Optimizer=not recorded, lr=0.05, weight_decay=0.0001, batch=256, epochs=30, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** “Let's try tanh, not expecting much. At this size of network it should be pretty much interchangable.”

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261023-112407-9711758e | 91.51% | 52.84% | 52% | 0.50 minutes | 25 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.801 → 0.309 | 0.2515 (29) | 1.673 → 1.8 | 1.402 (9) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261023-112407-9711758e](outputs/experiment-history/run-index.md#cifar10-20261023-112407-9711758e).

**Draft notes:** Legacy snapshot: activation/forward code is missing. Do not infer it from a later restored version.

### Change 75 — cifar10 — 2026-10-23

_Smaller CIFAR model and batch_

**Change:** legacy MLP (module shapes saved; forward source not captured). Exact module shapes are in the saved configuration; activation and forward source were not captured. Optimizer=not recorded, lr=0.05, weight_decay=0.0001, batch=128, epochs=30, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “I applied a shrink to the residual stream to 64. Didn't do much. Did that + shrank batch size:”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261023-122642-1f758bb6 | 83.72% | 52.28% | 51.78% | 0.80 minutes | 23 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.703 → 0.3295 | 0.3295 (30) | 1.613 → 2.105 | 1.373 (10) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261023-122642-1f758bb6](outputs/experiment-history/run-index.md#cifar10-20261023-122642-1f758bb6).

**Draft notes:** Legacy snapshot: activation/forward code is missing. Do not infer it from a later restored version. Journal says residual width 64; saved module shapes show width=128 and hidden=128.

### Change 76 — cifar10 — 2026-10-23

_Record complete settings and inspect class accuracy_

**Change:** flat residual MLP. Width=128, hidden=128, depth=7; recorded operations relu. Optimizer=SGD, lr=0.05, weight_decay=1e-05, batch=128, epochs=30, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “I think we need to understand which classes its getting right and wrong.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261023-123652-808ffc8f | 68.5311% | 52.48% | 52.09% | 0.74 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.703 → 0.3152 | 0.3152 (30) | 1.606 → 2.196 | 1.38 (10) |

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261023-124537-020df882 | 68.5311% | 52.48% | 52.09% | 0.76 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.703 → 0.3152 | 0.3152 (30) | 1.606 → 2.196 | 1.38 (10) |

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261023-125414-def90612 | 68.5311% | 52.48% | 52.09% | 0.75 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.703 → 0.3152 | 0.3152 (30) | 1.606 → 2.196 | 1.38 (10) |

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261023-134453-1fe1f1ca | — | — | — | — | — | 19 completed epoch rows; no final score |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.703 → 0.6339 | 0.6339 (19) | 1.606 → 1.707 | 1.38 (10) |

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261023-134526-bd3594de | 68.5311% | 52.48% | 52.09% | 0.76 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.703 → 0.3152 | 0.3152 (30) | 1.606 → 2.196 | 1.38 (10) |

Journal interpretation: “Cat is incredibly far off but we haven't learned much here. It's learning something about all of these” TODO — check this interpretation against the saved configuration and validation result.

**Runs:** [cifar10-20261023-123652-808ffc8f](outputs/experiment-history/run-index.md#cifar10-20261023-123652-808ffc8f), [cifar10-20261023-124537-020df882](outputs/experiment-history/run-index.md#cifar10-20261023-124537-020df882), [cifar10-20261023-125414-def90612](outputs/experiment-history/run-index.md#cifar10-20261023-125414-def90612), [cifar10-20261023-134453-1fe1f1ca](outputs/experiment-history/run-index.md#cifar10-20261023-134453-1fe1f1ca), [cifar10-20261023-134526-bd3594de](outputs/experiment-history/run-index.md#cifar10-20261023-134526-bd3594de).

### Change 77 — cifar10 — 2026-10-23

_Reduce CIFAR training to 10 epochs_

**Change:** flat residual MLP. Width=128, hidden=128, depth=7; recorded operations relu. Optimizer=SGD, lr=0.05, weight_decay=1e-05, batch=128, epochs=10, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “I am changing the runs to use less epochs since performance is worsening on average after 10%. Likely the point at which it starts memorizing given thats where loss diverges.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261023-134657-9294e112 | 68.5311% | 52.48% | 52.09% | 0.25 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.703 → 0.9983 | 0.9983 (10) | 1.606 → 1.38 | 1.38 (10) |

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261023-134752-02729c78 | 68.5311% | 52.48% | 52.09% | 0.29 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.703 → 0.9983 | 0.9983 (10) | 1.606 → 1.38 | 1.38 (10) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261023-134657-9294e112](outputs/experiment-history/run-index.md#cifar10-20261023-134657-9294e112), [cifar10-20261023-134752-02729c78](outputs/experiment-history/run-index.md#cifar10-20261023-134752-02729c78).

### Change 78 — cifar10 — 2026-10-23

_Journal-only ReLU comparison_

**Change:** Journal lists a ReLU trial after tanh with identical printed results. TODO — identify a distinct run ID or confirm that a result block was copied.

**Why:** “Worse. Reverting. Let's try some more.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Measure | Result (journal) |
|---|---:|
| Train accuracy | 91.51% |
| Validation accuracy | 52.84% |
| Test accuracy | 52.00% |
| Selected epoch | 25 |
| Training time | 0.5 minutes |

TODO — verify that ReLU was actually run and describe its loss curve.

**Draft notes:** No separate saved run was securely matched; do not invent one.

### Change 79 — cifar10 — 2026-10-24

_Switch CIFAR from SGD to AdamW_

**Change:** flat residual MLP. Width=128, hidden=128, depth=7; recorded operations relu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=128, epochs=10, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “I decided to switch from SGD with weight decay to ADAMW. I saw no immediate ways forward from the previous and learned more about how ADAMW works.”

**Expected:** “We will hopefully see some really nice improvements to accuracy and keep our weight decay to avoid overfitting. Turning down learning rate to match the lowerer needs of ADAMW as well.”

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-151753-14698b08 | 68.8822% | 54.36% | 53.32% | 0.45 minutes | 8 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.677 → 0.8629 | 0.8629 (10) | 1.534 → 1.38 | 1.32 (5) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-151753-14698b08](outputs/experiment-history/run-index.md#cifar10-20261024-151753-14698b08).

### Change 80 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=128, hidden=128, depth=7; recorded operations relu. Optimizer=AdamW, lr=0.001, weight_decay=5e-05, batch=128, epochs=10, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-174927-b3358428 | 71.54% | 55.1% | 54.73% | 0.44 minutes | 9 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.674 → 0.8652 | 0.8652 (10) | 1.515 → 1.375 | 1.309 (9) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-174927-b3358428](outputs/experiment-history/run-index.md#cifar10-20261024-174927-b3358428).

### Change 81 — cifar10 — 2026-10-24

_CIFAR data augmentation_

**Change:** flat residual MLP. Width=128, hidden=128, depth=7; recorded operations relu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=128, epochs=10, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “No magic bullet of AdamW will work it seems. Let's try data augmentaton that we learned in class.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-180552-79892e5e | 58.1622% | 55.18% | 54.14% | 0.80 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.796 → 1.33 | 1.33 (10) | 1.581 → 1.262 | 1.262 (10) |

Journal interpretation: “Well, Overfitting decreased. Performance is bad.” TODO — check this interpretation against the saved configuration and validation result.

**Runs:** [cifar10-20261024-180552-79892e5e](outputs/experiment-history/run-index.md#cifar10-20261024-180552-79892e5e).

### Change 82 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=128, hidden=128, depth=7; recorded operations relu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=10, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-190549-b5f1f96d | 69.8022% | 54.7% | 54.17% | 0.25 minutes | 9 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.7 → 0.8778 | 0.8778 (10) | 1.516 → 1.368 | 1.309 (7) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-190549-b5f1f96d](outputs/experiment-history/run-index.md#cifar10-20261024-190549-b5f1f96d).

### Change 83 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=128, hidden=512, depth=7; recorded operations silu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=10, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-190721-622fa5d9 | 71.6867% | 55.64% | 55.43% | 0.25 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.681 → 0.9185 | 0.9185 (10) | 1.541 → 1.308 | 1.272 (8) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-190721-622fa5d9](outputs/experiment-history/run-index.md#cifar10-20261024-190721-622fa5d9).

### Change 84 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=128, hidden=512, depth=8; recorded operations silu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=10, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-190829-4408179b | 68.9733% | 55.9% | 55.34% | 0.27 minutes | 9 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.687 → 0.9351 | 0.9351 (10) | 1.542 → 1.319 | 1.282 (9) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-190829-4408179b](outputs/experiment-history/run-index.md#cifar10-20261024-190829-4408179b).

### Change 85 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=128, hidden=512, depth=20; recorded operations silu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=10, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-190942-5d5a11ec | 64.2911% | 54.94% | 53.88% | 0.54 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.747 → 1.087 | 1.087 (10) | 1.582 → 1.304 | 1.281 (9) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-190942-5d5a11ec](outputs/experiment-history/run-index.md#cifar10-20261024-190942-5d5a11ec).

### Change 86 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=256, hidden=512, depth=50; recorded operations silu. Optimizer=AdamW, lr=0.001, weight_decay=5e-05, batch=256, epochs=10, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-191149-9e04c0c6 | 65.2489% | 55.08% | 53.86% | 1.29 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.863 → 1.077 | 1.077 (10) | 1.665 → 1.314 | 1.314 (10) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-191149-9e04c0c6](outputs/experiment-history/run-index.md#cifar10-20261024-191149-9e04c0c6).

### Change 87 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=256, hidden=512, depth=50; recorded operations silu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=15, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-191810-0c227281 | 67.3644% | 55.76% | 54.68% | 1.98 minutes | 11 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.863 → 0.8132 | 0.8132 (15) | 1.665 → 1.376 | 1.288 (11) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-191810-0c227281](outputs/experiment-history/run-index.md#cifar10-20261024-191810-0c227281).

### Change 88 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=256, hidden=512, depth=50; recorded operations silu. Optimizer=Muon, lr=0.02, weight_decay=1e-05, batch=256, epochs=15, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-192349-50b5b789 | 71.0378% | 56.3% | 55.46% | 4.58 minutes | 5 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.661 → 0.2463 | 0.2463 (15) | 1.501 → 1.987 | 1.291 (4) |

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-193237-88e5f013 | 71.0378% | 56.3% | 55.46% | 4.61 minutes | 5 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.661 → 0.2463 | 0.2463 (15) | 1.501 → 1.987 | 1.291 (4) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-192349-50b5b789](outputs/experiment-history/run-index.md#cifar10-20261024-192349-50b5b789), [cifar10-20261024-193237-88e5f013](outputs/experiment-history/run-index.md#cifar10-20261024-193237-88e5f013).

### Change 89 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=256, hidden=512, depth=50; recorded operations silu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=15, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-193733-de010494 | 82.8289% | 56.26% | 55.18% | 4.70 minutes | 8 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.675 → 0.2738 | 0.2738 (15) | 1.501 → 1.933 | 1.285 (5) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-193733-de010494](outputs/experiment-history/run-index.md#cifar10-20261024-193733-de010494).

### Change 90 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=128, hidden=512, depth=50; recorded operations silu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=15, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-194312-d0e4074c | 74.0978% | 56.46% | 55% | 3.72 minutes | 7 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.705 → 0.3582 | 0.3582 (15) | 1.529 → 1.827 | 1.286 (5) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-194312-d0e4074c](outputs/experiment-history/run-index.md#cifar10-20261024-194312-d0e4074c).

### Change 91 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=128, hidden=512, depth=3; recorded operations silu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=15, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-194834-440d42d1 | 76.2667% | 55.92% | 54.58% | 0.37 minutes | 7 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.625 → 0.2762 | 0.2762 (15) | 1.47 → 2.183 | 1.309 (4) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-194834-440d42d1](outputs/experiment-history/run-index.md#cifar10-20261024-194834-440d42d1).

### Change 92 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=64, hidden=512, depth=3; recorded operations silu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=10, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-194948-28bf206f | 63.3222% | 54.36% | 53.7% | 0.16 minutes | 9 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.698 → 1.067 | 1.067 (10) | 1.54 → 1.326 | 1.302 (8) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-194948-28bf206f](outputs/experiment-history/run-index.md#cifar10-20261024-194948-28bf206f).

### Change 93 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=64, hidden=512, depth=50; recorded operations silu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=10, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-195018-9050bcbd | 56.7044% | 50.7% | 51.46% | 1.19 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.814 → 1.261 | 1.261 (10) | 1.678 → 1.373 | 1.373 (10) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-195018-9050bcbd](outputs/experiment-history/run-index.md#cifar10-20261024-195018-9050bcbd).

### Change 94 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=96, hidden=512, depth=50; recorded operations silu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=10, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-195211-13ca66a0 | 58.8289% | 51.92% | 51.88% | 1.20 minutes | 9 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.792 → 1.192 | 1.192 (10) | 1.642 → 1.352 | 1.337 (9) |

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-195449-e6c9f45c | 58.8289% | 51.92% | 51.88% | 1.19 minutes | 9 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.792 → 1.192 | 1.192 (10) | 1.642 → 1.352 | 1.337 (9) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-195211-13ca66a0](outputs/experiment-history/run-index.md#cifar10-20261024-195211-13ca66a0), [cifar10-20261024-195449-e6c9f45c](outputs/experiment-history/run-index.md#cifar10-20261024-195449-e6c9f45c).

### Change 95 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=128, hidden=512, depth=50; recorded operations silu. Optimizer=AdamW, lr=0.005, weight_decay=1e-05, batch=256, epochs=10, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-195653-50a070d4 | 60.0933% | 52.46% | 52.95% | 1.22 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.974 → 1.209 | 1.209 (10) | 1.793 → 1.342 | 1.342 (10) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-195653-50a070d4](outputs/experiment-history/run-index.md#cifar10-20261024-195653-50a070d4).

### Change 96 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=128, hidden=512, depth=50; recorded operations silu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=128, epochs=10, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-200016-2a3ca5fb | 60.3756% | 52.84% | 52.73% | 2.37 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.816 → 1.17 | 1.17 (10) | 1.65 → 1.355 | 1.338 (8) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-200016-2a3ca5fb](outputs/experiment-history/run-index.md#cifar10-20261024-200016-2a3ca5fb).

### Change 97 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=256, hidden=512, depth=50; recorded operations silu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=128, epochs=10, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-200407-caebc9e6 | 65.4644% | 54.84% | 53.61% | 2.55 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.824 → 1.079 | 1.079 (10) | 1.66 → 1.296 | 1.296 (10) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-200407-caebc9e6](outputs/experiment-history/run-index.md#cifar10-20261024-200407-caebc9e6).

## C2 — Happy Accident and controlled reruns

### Change 98 — cifar10 — 2026-10-24

_Happy Accident: restored configuration_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=30, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “So I wasn't liking the results I was getting so I added a feature to roll back to previous configurations. But, apparently it was bugged.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-210758-6cf0d0b4 | 74.4267% | 60.42% | 58.88% | 1.53 minutes | 29 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.784 → 0.9809 | 0.9809 (30) | 1.552 → 1.182 | 1.154 (29) |

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-214516-c1cc95e3 | 74.4267% | 60.42% | 58.88% | 1.41 minutes | 29 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.784 → 0.9809 | 0.9809 (30) | 1.552 → 1.182 | 1.154 (29) |

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-215408-cb0ff45d | 74.4267% | 60.42% | 58.88% | 1.40 minutes | 29 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.784 → 0.9809 | 0.9809 (30) | 1.552 → 1.182 | 1.154 (29) |

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-215556-2b1c3233 | — | — | — | — | — | 0 completed epoch rows; no final score; no completed epoch metrics were saved |

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-220508-08117fee | 74.4267% | 60.42% | 58.88% | 1.46 minutes | 29 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.784 → 0.9809 | 0.9809 (30) | 1.552 → 1.182 | 1.154 (29) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-210758-6cf0d0b4](outputs/experiment-history/run-index.md#cifar10-20261024-210758-6cf0d0b4), [cifar10-20261024-214516-c1cc95e3](outputs/experiment-history/run-index.md#cifar10-20261024-214516-c1cc95e3), [cifar10-20261024-215408-cb0ff45d](outputs/experiment-history/run-index.md#cifar10-20261024-215408-cb0ff45d), [cifar10-20261024-215556-2b1c3233](outputs/experiment-history/run-index.md#cifar10-20261024-215556-2b1c3233), [cifar10-20261024-220508-08117fee](outputs/experiment-history/run-index.md#cifar10-20261024-220508-08117fee).

### Change 99 — cifar10 — 2026-10-24

_Isolate width and hidden size_

**Change:** flat residual MLP. Width=128, hidden=128, depth=7; recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=30, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “So... I need to understand what happened here. I am going to look through each change one by one and theorize then test.”

**Expected:** “I imagine that a residual stream of 128 with a hidden width of 128 was bottlenecking the model.”

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-212021-44f3ab35` | 63.9867% | 57.56% | 56.68% | 1.35 minutes | 29 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.837 → 1.17 | 1.17 (30) | 1.647 → 1.214 | 1.208 (29) |

Journal interpretation: “Looks like we were a little bit right, but it only made a small difference. Reverting and moving to the next.” TODO — check this interpretation against the saved configuration and validation result.

**Runs:** [cifar10-20261024-212021-44f3ab35](outputs/experiment-history/run-index.md#cifar10-20261024-212021-44f3ab35).

### Change 100 — cifar10 — 2026-10-24

_Activation comparison: saved source uses ReLU_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations relu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=30, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** “I expect GELU is better because by definition it was an iteration upon ReLU.”

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-212806-223e71d6` | 71.7667% | 60.04% | 58.51% | 1.43 minutes | 29 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.822 → 1.035 | 1.035 (30) | 1.585 → 1.188 | 1.168 (29) |

Journal interpretation: “And so it was.” TODO — check this interpretation against the saved configuration and validation result.

**Runs:** [cifar10-20261024-212806-223e71d6](outputs/experiment-history/run-index.md#cifar10-20261024-212806-223e71d6).

**Draft notes:** Journal labels this ReLU → GELU, but the saved block and head use ReLU. The quoted expectation remains the original wording; rewrite the comparison direction correctly.

### Change 101 — cifar10 — 2026-10-24

_Batch 128 with augmentation accidentally disabled_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=128, epochs=30, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** “Batch size being smaller in theory should help with generalization. We are making more erratic jumps that speed up gradient descent.”

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-213729-7b6a5d2b` | 99.9978% | 57.38% | 55.77% | 0.85 minutes | 27 | — |
| `cifar10-20261024-222752-audit-batch-128` | 99.9978% | 57.38% | 55.77% | 0.86 minutes | 27 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| `cifar10-20261024-213729-7b6a5d2b` | 1.631 → 0.001167 | 0.001167 (30) | 1.497 → 2.62 | 1.336 (4) |
| `cifar10-20261024-222752-audit-batch-128` | 1.631 → 0.001167 | 0.001167 (30) | 1.497 → 2.62 | 1.336 (4) |

Journal interpretation: “Worse, not entirely unexpected.” TODO — check this interpretation against the saved configuration and validation result.

**Runs:** [cifar10-20261024-213729-7b6a5d2b](outputs/experiment-history/run-index.md#cifar10-20261024-213729-7b6a5d2b), [cifar10-20261024-222752-audit-batch-128](outputs/experiment-history/run-index.md#cifar10-20261024-222752-audit-batch-128).

**Draft notes:** Both batch size and augmentation changed. This does not isolate batch size.

### Change 102 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=30, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-213937-1058bdc7` | 99.9867% | 56.4% | 54.59% | 0.51 minutes | 28 | — |
| `cifar10-20261024-221742-c8c03980` | 99.9867% | 56.4% | 54.59% | 0.51 minutes | 28 | — |
| `cifar10-20261024-222752-audit-baseline` | 99.9867% | 56.4% | 54.59% | 0.52 minutes | 28 | — |
| `cifar10-20261024-223819-2016e143` | — | — | — | — | — | No final score; 11 completed epoch rows saved |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| `cifar10-20261024-213937-1058bdc7` | 1.68 → 0.00421 | 0.00421 (30) | 1.559 → 2.498 | 1.312 (5) |
| `cifar10-20261024-221742-c8c03980` | 1.68 → 0.00421 | 0.00421 (30) | 1.559 → 2.498 | 1.312 (5) |
| `cifar10-20261024-222752-audit-baseline` | 1.68 → 0.00421 | 0.00421 (30) | 1.559 → 2.498 | 1.312 (5) |
| `cifar10-20261024-223819-2016e143` | 1.68 → 0.6411 | 0.6411 (11) | 1.559 → 1.488 | 1.312 (5) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-213937-1058bdc7](outputs/experiment-history/run-index.md#cifar10-20261024-213937-1058bdc7), [cifar10-20261024-221742-c8c03980](outputs/experiment-history/run-index.md#cifar10-20261024-221742-c8c03980), [cifar10-20261024-222752-audit-baseline](outputs/experiment-history/run-index.md#cifar10-20261024-222752-audit-baseline), [cifar10-20261024-223819-2016e143](outputs/experiment-history/run-index.md#cifar10-20261024-223819-2016e143).

### Change 103 — cifar10 — 2026-10-24

_Reduce CIFAR weight decay to 0.00001_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=1e-05, batch=256, epochs=30, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “Weight decay is a big deal, I've seen it absolutely tank performance in other trials. I've also seen it save performance during MNIST where we were a hundred percent accuracy.”

**Expected:** “This is what I had expected to make the biggest difference.”

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-215848-059027aa` | 74.9% | 60.38% | 59.07% | 1.46 minutes | 29 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.784 → 0.9737 | 0.9737 (30) | 1.552 → 1.185 | 1.158 (29) |

Journal interpretation: “Incredible. We've been over regularizing. Why not? Even smaller.” TODO — check this interpretation against the saved configuration and validation result.

**Runs:** [cifar10-20261024-215848-059027aa](outputs/experiment-history/run-index.md#cifar10-20261024-215848-059027aa).

### Change 104 — cifar10 — 2026-10-24

_Zero weight decay_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=0, batch=256, epochs=30, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “Incredible. We've been over regularizing. Why not? Even smaller.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-220725-3ec48d0c` | 74.9311% | 60.36% | 59.11% | 1.41 minutes | 29 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.784 → 0.9729 | 0.9729 (30) | 1.552 → 1.185 | 1.158 (29) |

Journal interpretation: “It rounded to zero after I added one more. Going to leave regularization on.” TODO — check this interpretation against the saved configuration and validation result.

**Runs:** [cifar10-20261024-220725-3ec48d0c](outputs/experiment-history/run-index.md#cifar10-20261024-220725-3ec48d0c).

### Change 105 — cifar10 — 2026-10-24

_Augmentation-off audit: relu_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations relu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=30, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “Forgot to turn augmentation off, had to rerun all those experiments”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-222752-audit-relu` | 99.9844% | 55.52% | 54.81% | 0.53 minutes | 30 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.718 → 0.01115 | 0.01115 (30) | 1.578 → 2.306 | 1.327 (5) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-222752-audit-relu](outputs/experiment-history/run-index.md#cifar10-20261024-222752-audit-relu).

**Draft notes:** Controlled augmentation-off rerun. The historical prediction is not supplied by the later result table.

### Change 106 — cifar10 — 2026-10-24

_Augmentation-off audit: weight-decay-1e-5_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=1e-05, batch=256, epochs=30, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “Forgot to turn augmentation off, had to rerun all those experiments”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-222752-audit-weight-decay-1e-5` | 99.9978% | 55.3% | 54.75% | 0.51 minutes | 29 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.68 → 0.002158 | 0.002158 (30) | 1.559 → 2.664 | 1.313 (5) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-222752-audit-weight-decay-1e-5](outputs/experiment-history/run-index.md#cifar10-20261024-222752-audit-weight-decay-1e-5).

**Draft notes:** Controlled augmentation-off rerun. The historical prediction is not supplied by the later result table.

### Change 107 — cifar10 — 2026-10-24

_Augmentation-off audit: weight-decay-zero_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=0, batch=256, epochs=30, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “Forgot to turn augmentation off, had to rerun all those experiments”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-222752-audit-weight-decay-zero` | 99.9978% | 55.72% | 54.67% | 0.49 minutes | 29 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.68 → 0.002283 | 0.002283 (30) | 1.559 → 2.644 | 1.313 (5) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-222752-audit-weight-decay-zero](outputs/experiment-history/run-index.md#cifar10-20261024-222752-audit-weight-decay-zero).

**Draft notes:** Controlled augmentation-off rerun. The historical prediction is not supplied by the later result table.

### Change 108 — cifar10 — 2026-10-24

_Augmentation-off audit: width-hidden_

**Change:** flat residual MLP. Width=128, hidden=128, depth=7; recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=30, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “Forgot to turn augmentation off, had to rerun all those experiments”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-222752-audit-width-hidden` | 72.8311% | 51.74% | 50.29% | 0.43 minutes | 15 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.736 → 0.3564 | 0.3564 (30) | 1.6 → 2.102 | 1.367 (7) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-222752-audit-width-hidden](outputs/experiment-history/run-index.md#cifar10-20261024-222752-audit-width-hidden).

**Draft notes:** Controlled augmentation-off rerun. The historical prediction is not supplied by the later result table.

## C3 — Optimizers, width and depth

### Change 109 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-234504-6badff50` | 87.4733% | 61.48% | 60.11% | 2.83 minutes | 53 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.784 → 0.6976 | 0.6976 (60) | 1.552 → 1.319 | 1.152 (35) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-234504-6badff50](outputs/experiment-history/run-index.md#cifar10-20261024-234504-6badff50).

### Change 110 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations gelu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-234930-b4aa43ea` | 96.3133% | 61.26% | 59.01% | 3.38 minutes | 43 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.799 → 0.3329 | 0.3329 (60) | 1.552 → 1.892 | 1.181 (19) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-234930-b4aa43ea](outputs/experiment-history/run-index.md#cifar10-20261024-234930-b4aa43ea).

### Change 111 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations gelu. Optimizer=AdamW, lr=0.005, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-235432-57963b71` | 93.7622% | 59.64% | 58.11% | 3.31 minutes | 52 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.005 → 0.4989 | 0.4989 (60) | 1.748 → 1.864 | 1.194 (24) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261024-235432-57963b71](outputs/experiment-history/run-index.md#cifar10-20261024-235432-57963b71).

### Change 112 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations gelu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-000330-5cea5707` | 91.9044% | 61.48% | 59.43% | 3.39 minutes | 32 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.799 → 0.3178 | 0.3178 (60) | 1.552 → 1.843 | 1.178 (15) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-000330-5cea5707](outputs/experiment-history/run-index.md#cifar10-20261025-000330-5cea5707).

### Change 113 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations gelu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-000912-e7c42e1d` | 94.1978% | 61.68% | 59.08% | 3.53 minutes | 49 | — |
| `cifar10-20261025-002050-9941eabc` | 94.1978% | 61.68% | 59.08% | 4.45 minutes | 49 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| `cifar10-20261025-000912-e7c42e1d` | 1.833 → 0.5039 | 0.5039 (60) | 1.582 → 1.526 | 1.152 (19) |
| `cifar10-20261025-002050-9941eabc` | 1.833 → 0.5039 | 0.5039 (60) | 1.582 → 1.526 | 1.152 (19) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-000912-e7c42e1d](outputs/experiment-history/run-index.md#cifar10-20261025-000912-e7c42e1d), [cifar10-20261025-002050-9941eabc](outputs/experiment-history/run-index.md#cifar10-20261025-002050-9941eabc).

### Change 114 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations silu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=30, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-001310-684974cb` | 70.9911% | 60.26% | 58.82% | 1.79 minutes | 29 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.783 → 1.041 | 1.041 (30) | 1.556 → 1.176 | 1.147 (29) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-001310-684974cb](outputs/experiment-history/run-index.md#cifar10-20261025-001310-684974cb).

### Change 115 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations silu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=50, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-001521-3603c62e` | 78.84% | 61.28% | 59.47% | 2.97 minutes | 46 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.783 → 0.8696 | 0.8696 (50) | 1.556 → 1.178 | 1.135 (41) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-001521-3603c62e](outputs/experiment-history/run-index.md#cifar10-20261025-001521-3603c62e).

### Change 116 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** flat residual MLP. Width=256, hidden=512, depth=7; recorded operations gelu. Optimizer=Muon, lr=0.02, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-002715-5a076b1b` | 89.4933% | 61.56% | 59.25% | 6.93 minutes | 31 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.748 → 0.4276 | 0.4276 (60) | 1.511 → 1.658 | 1.152 (18) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-002715-5a076b1b](outputs/experiment-history/run-index.md#cifar10-20261025-002715-5a076b1b).

### Change 117 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** flat residual MLP. Width=512, hidden=512, depth=7; recorded operations gelu. Optimizer=Muon, lr=0.02, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-003949-24bf237f` | 93.8378% | 61.96% | 59.69% | 9.16 minutes | 38 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.763 → 0.4431 | 0.4414 (59) | 1.486 → 1.579 | 1.144 (15) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-003949-24bf237f](outputs/experiment-history/run-index.md#cifar10-20261025-003949-24bf237f).

### Change 118 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** flat residual MLP. Width=512, hidden=512, depth=7; recorded operations gelu. Optimizer=RMSprop, lr=0.001, weight_decay=1e-05, batch=256, epochs=30, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-005328-ee80b2f5` | 75.6644% | 60.24% | 57.68% | 2.25 minutes | 27 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.318 → 0.9572 | 0.9572 (30) | 1.786 → 1.228 | 1.165 (26) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-005328-ee80b2f5](outputs/experiment-history/run-index.md#cifar10-20261025-005328-ee80b2f5).

### Change 119 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** flat residual MLP. Width=512, hidden=512, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.02, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-005652-cd458538` | 97.3133% | 62.04% | 60.13% | 8.01 minutes | 57 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.752 → 0.4525 | 0.4525 (60) | 1.509 → 1.602 | 1.167 (16) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-005652-cd458538](outputs/experiment-history/run-index.md#cifar10-20261025-005652-cd458538).

### Change 120 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** flat residual MLP. Width=512, hidden=512, depth=3; recorded operations gelu. Optimizer=Muon, lr=0.02, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-115606-bd7c0564` | 95.6933% | 59.12% | 57.55% | 4.25 minutes | 27 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.659 → 0.1991 | 0.1991 (60) | 1.443 → 2.177 | 1.205 (9) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-115606-bd7c0564](outputs/experiment-history/run-index.md#cifar10-20261025-115606-bd7c0564).

### Change 121 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** flat residual MLP. Width=512, hidden=512, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-123955-7c8c8c1d` | 96.8067% | 62.26% | 60.15% | 5.91 minutes | 60 | — |
| `cifar10-20261025-210636-d9406711` | 96.8067% | 62.26% | 60.15% | 6.03 minutes | 60 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| `cifar10-20261025-123955-7c8c8c1d` | 1.762 → 0.5186 | 0.5181 (59) | 1.522 → 1.485 | 1.138 (17) |
| `cifar10-20261025-210636-d9406711` | 1.762 → 0.5186 | 0.5181 (59) | 1.522 → 1.485 | 1.138 (17) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-123955-7c8c8c1d](outputs/experiment-history/run-index.md#cifar10-20261025-123955-7c8c8c1d), [cifar10-20261025-210636-d9406711](outputs/experiment-history/run-index.md#cifar10-20261025-210636-d9406711).

### Change 122 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** flat residual MLP. Width=512, hidden=512, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.01, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-124849-bcdd1a30` | 94.88% | 61.9% | 60.26% | 6.06 minutes | 35 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.743 → 0.3943 | 0.3943 (60) | 1.469 → 1.694 | 1.161 (12) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-124849-bcdd1a30](outputs/experiment-history/run-index.md#cifar10-20261025-124849-bcdd1a30).

## C4 — Bilinear gates, pooling and dropout

### Change 123 — cifar10 — 2026-10-25

_Incomplete block construction: block forward returns none_

**Change:** incomplete block construction: block forward returns None. Width=512, hidden=512, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.01, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-125710-62ba38ce` | — | — | — | — | — | No final score; 0 completed epoch rows saved; No completed epoch metrics were saved. |
| `cifar10-20261025-125841-0e1749e2` | — | — | — | — | — | No final score; 0 completed epoch rows saved; No completed epoch metrics were saved. |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| `cifar10-20261025-125710-62ba38ce` | — | — | — | — |
| `cifar10-20261025-125841-0e1749e2` | — | — | — | — |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-125710-62ba38ce](outputs/experiment-history/run-index.md#cifar10-20261025-125710-62ba38ce), [cifar10-20261025-125841-0e1749e2](outputs/experiment-history/run-index.md#cifar10-20261025-125841-0e1749e2).

**Draft notes:** Saved block source returns None. No exception receipt was found; do not invent the exact failure message.

### Change 124 — cifar10 — 2026-10-25

_Incomplete block construction: block forward returns none_

**Change:** incomplete block construction: block forward returns None. Width=512, hidden=512, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-130442-f759bea4` | — | — | — | — | — | No final score; 0 completed epoch rows saved; No completed epoch metrics were saved. |
| `cifar10-20261025-132544-8610feee` | — | — | — | — | — | No final score; 0 completed epoch rows saved; No completed epoch metrics were saved. |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| `cifar10-20261025-130442-f759bea4` | — | — | — | — |
| `cifar10-20261025-132544-8610feee` | — | — | — | — |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-130442-f759bea4](outputs/experiment-history/run-index.md#cifar10-20261025-130442-f759bea4), [cifar10-20261025-132544-8610feee](outputs/experiment-history/run-index.md#cifar10-20261025-132544-8610feee).

**Draft notes:** Saved block source returns None. No exception receipt was found; do not invent the exact failure message.

### Change 125 — cifar10 — 2026-10-25

_Bilinear gated blocks_

**Change:** bilinear blocks without skip connections; unused extra Linear layers remain registered. Width=512, hidden=512, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “Time to experiment with some gated linear units. While it's directly supported by PyTorch, it's not supported by torch canvas. I could add the functionality, but I think it's a more interesting exploration if I just build it out of its puzzle pieces.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-135437-5b0d7d45` | 98.7511% | 60.34% | 57.66% | 5.90 minutes | 44 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.053 → 0.3892 | 0.3892 (60) | 1.771 → 1.715 | 1.201 (12) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-135437-5b0d7d45](outputs/experiment-history/run-index.md#cifar10-20261025-135437-5b0d7d45).

**Draft notes:** Historical model had no residual addition despite its class name. Record this limitation rather than calling it a residual architecture.

### Change 126 — cifar10 — 2026-10-25

_Bilinear blocks without skip connections; unused extra linear layers remain registered_

**Change:** bilinear blocks without skip connections; unused extra Linear layers remain registered. Width=512, hidden=512, depth=6; recorded operations silu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-140146-70e4347d` | — | — | — | — | — | No final score; 22 completed epoch rows saved |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.027 → 0.7744 | 0.7744 (22) | 1.754 → 1.371 | 1.211 (11) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-140146-70e4347d](outputs/experiment-history/run-index.md#cifar10-20261025-140146-70e4347d).

**Draft notes:** Historical model had no residual addition despite its class name. Record this limitation rather than calling it a residual architecture.

### Change 127 — cifar10 — 2026-10-25

_Bilinear blocks without skip connections_

**Change:** bilinear blocks without skip connections. Width=512, hidden=512, depth=6; recorded operations silu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-140404-e8234a17` | 98.7733% | 60.24% | 57.57% | 6.03 minutes | 45 | — |
| `cifar10-20261025-141647-eeafe2e6` | 98.7733% | 60.24% | 57.57% | 5.99 minutes | 45 | — |
| `cifar10-20261025-142317-26fef98f` | 98.7733% | 60.24% | 57.57% | 6.09 minutes | 45 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| `cifar10-20261025-140404-e8234a17` | 2.032 → 0.387 | 0.387 (60) | 1.764 → 1.728 | 1.243 (13) |
| `cifar10-20261025-141647-eeafe2e6` | 2.032 → 0.387 | 0.387 (60) | 1.764 → 1.728 | 1.243 (13) |
| `cifar10-20261025-142317-26fef98f` | 2.032 → 0.387 | 0.387 (60) | 1.764 → 1.728 | 1.243 (13) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-140404-e8234a17](outputs/experiment-history/run-index.md#cifar10-20261025-140404-e8234a17), [cifar10-20261025-141647-eeafe2e6](outputs/experiment-history/run-index.md#cifar10-20261025-141647-eeafe2e6), [cifar10-20261025-142317-26fef98f](outputs/experiment-history/run-index.md#cifar10-20261025-142317-26fef98f).

**Draft notes:** Historical model had no residual addition despite its class name. Record this limitation rather than calling it a residual architecture.

### Change 128 — cifar10 — 2026-10-25

_4×4 max pooling followed by bilinear residual blocks_

**Change:** 4×4 max pooling followed by bilinear residual blocks. Width=512, hidden=512, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-195130-38a08235` | — | — | — | — | — | No final score; 13 completed epoch rows saved |
| `cifar10-20261025-201234-869c059e` | 82.2622% | 57.38% | 55.73% | 6.03 minutes | 14 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| `cifar10-20261025-195130-38a08235` | 1.828 → 0.984 | 0.984 (13) | 1.59 → 1.325 | 1.276 (9) |
| `cifar10-20261025-201234-869c059e` | 1.828 → 0.3808 | 0.3808 (60) | 1.59 → 1.774 | 1.276 (9) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-195130-38a08235](outputs/experiment-history/run-index.md#cifar10-20261025-195130-38a08235), [cifar10-20261025-201234-869c059e](outputs/experiment-history/run-index.md#cifar10-20261025-201234-869c059e).

**Draft notes:** Historical model includes `nn.MaxPool2d(kernel_size=4)`, outside the assignment’s listed layer types. It was an exploration, not the final submitted design.

### Change 129 — cifar10 — 2026-10-25

_Normalized value/gate blocks with no dropout; no skip addition in this saved version_

**Change:** normalized value/gate blocks with no dropout; no skip addition in this saved version. Width=512, hidden=1024, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-213314-9c0e7949` | 97.0422% | 61.52% | 59.67% | 11.40 minutes | 54 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.188 → 0.5288 | 0.5288 (60) | 1.798 → 1.475 | 1.173 (19) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261025-213314-9c0e7949](outputs/experiment-history/run-index.md#cifar10-20261025-213314-9c0e7949).

**Draft notes:** Historical model had no residual addition despite its class name. Record this limitation rather than calling it a residual architecture.

### Change 130 — cifar10 — 2026-10-26

_Add dropout 0.1 to gated blocks_

**Change:** normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version. Width=512, hidden=1024, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “At this point I added dropout to the output of the layer so it would have a 10% chance of setting a value of the input to zero.”

**Expected:** “Intuitively this makes sense. We randomly are tossing out information and asking the model to still answer the question.”

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-131127-ba8e8b68` | 95.12% | 63.5% | 60.87% | 11.35 minutes | 52 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.303 → 0.6481 | 0.6481 (60) | 1.789 → 1.34 | 1.165 (20) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261026-131127-ba8e8b68](outputs/experiment-history/run-index.md#cifar10-20261026-131127-ba8e8b68).

**Draft notes:** Historical model had no residual addition despite its class name. Record this limitation rather than calling it a residual architecture. The journal discusses seven masks, but this saved architecture has six blocks. TODO — correct the dropout explanation.

### Change 131 — cifar10 — 2026-10-26

_Normalized value/gate blocks with 0.15 dropout; no skip addition in this saved version_

**Change:** normalized value/gate blocks with 0.15 dropout; no skip addition in this saved version. Width=512, hidden=1024, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=10, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-151046-ee81c4cb` | 60.6178% | 56.38% | 55.34% | 2.67 minutes | 10 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.39 → 1.368 | 1.368 (10) | 1.827 → 1.257 | 1.257 (10) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261026-151046-ee81c4cb](outputs/experiment-history/run-index.md#cifar10-20261026-151046-ee81c4cb).

**Draft notes:** Historical model had no residual addition despite its class name. Record this limitation rather than calling it a residual architecture.

### Change 132 — cifar10 — 2026-10-26

_Normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version_

**Change:** normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version. Width=512, hidden=1024, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=10, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-152251-5dfa7476` | 61.68% | 56.64% | 55.61% | 1.88 minutes | 10 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.303 → 1.344 | 1.344 (10) | 1.789 → 1.264 | 1.264 (10) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261026-152251-5dfa7476](outputs/experiment-history/run-index.md#cifar10-20261026-152251-5dfa7476).

**Draft notes:** Historical model had no residual addition despite its class name. Record this limitation rather than calling it a residual architecture.

### Change 133 — cifar10 — 2026-10-26

_Normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version_

**Change:** normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version. Width=512, hidden=1024, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=15, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-152521-7a9bfc86` | 66.5511% | 58.58% | 57.41% | 2.82 minutes | 14 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.303 → 1.23 | 1.23 (15) | 1.789 → 1.19 | 1.187 (14) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261026-152521-7a9bfc86](outputs/experiment-history/run-index.md#cifar10-20261026-152521-7a9bfc86).

**Draft notes:** Historical model had no residual addition despite its class name. Record this limitation rather than calling it a residual architecture.

### Change 134 — cifar10 — 2026-10-26

_Normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version_

**Change:** normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version. Width=512, hidden=1024, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=16, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-152851-db2487cc` | — | — | — | — | — | No final score; 0 completed epoch rows saved; No completed epoch metrics were saved. |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | — | — | — | — |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261026-152851-db2487cc](outputs/experiment-history/run-index.md#cifar10-20261026-152851-db2487cc).

**Draft notes:** Historical model had no residual addition despite its class name. Record this limitation rather than calling it a residual architecture.

### Change 135 — cifar10 — 2026-10-26

_Normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version_

**Change:** normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version. Width=512, hidden=1024, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=20, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-152951-e65d9ef5` | 72.88% | 60.42% | 58.7% | 3.75 minutes | 20 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.303 → 1.136 | 1.136 (20) | 1.789 → 1.165 | 1.165 (20) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261026-152951-e65d9ef5](outputs/experiment-history/run-index.md#cifar10-20261026-152951-e65d9ef5).

**Draft notes:** Historical model had no residual addition despite its class name. Record this limitation rather than calling it a residual architecture.

### Change 136 — cifar10 — 2026-10-26

_Normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version_

**Change:** normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version. Width=512, hidden=1024, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=20, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-153426-931b32ea` | 72.88% | 60.42% | 58.7% | 3.79 minutes | 20 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.303 → 1.136 | 1.136 (20) | 1.789 → 1.165 | 1.165 (20) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261026-153426-931b32ea](outputs/experiment-history/run-index.md#cifar10-20261026-153426-931b32ea).

**Draft notes:** Historical model had no residual addition despite its class name. Record this limitation rather than calling it a residual architecture.

## C5 — Hierarchical rolling windows

### Change 137 — cifar10 — 2026-10-26

_Hierarchical 2×2 rolling-window MLP_

**Change:** hierarchical 2×2 rolling-window MLP without edge padding. Width=256, hidden=512, depth=2; recorded operations gelu. Window MLP hidden/output=32/16; spatial sizes 32 → 31 → 15 → 7 → 3 → 1. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** “I'm wondering if I can create many feed forward networks. The idea being I take the32 by32 input image. Then subdivide it twice. That would then create many four by four sub images. Then run a multi-layer perceptron to create some new representation.”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-215237-2ac3a7b3` | 55.5444% | 49.98% | 48.27% | 9.91 minutes | 54 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.39 → 1.386 | 1.386 (60) | 1.874 → 1.399 | 1.386 (59) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261026-215237-2ac3a7b3](outputs/experiment-history/run-index.md#cifar10-20261026-215237-2ac3a7b3).

**Draft notes:** The journal’s earlier sketch describes 4×4 subdivisions. The actual saved implementation uses rolling 2×2 windows. Keep the idea and implementation distinct.

### Change 138 — cifar10 — 2026-10-26

_Widen rolling windows and classifier_

**Change:** hierarchical 2×2 rolling-window MLP without edge padding. Width=512, hidden=1024, depth=2; recorded operations gelu. Window MLP hidden/output=128/64; spatial sizes 32 → 31 → 15 → 7 → 3 → 1. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-221344-5d653ac6` | 71.3511% | 60.8% | 59.15% | 14.83 minutes | 57 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 3.277 → 1.052 | 1.052 (60) | 1.973 → 1.153 | 1.153 (60) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261026-221344-5d653ac6](outputs/experiment-history/run-index.md#cifar10-20261026-221344-5d653ac6).

### Change 139 — cifar10 — 2026-10-26

_Preserve full-image coverage with edge padding_

**Change:** hierarchical 2×2 rolling-window MLP with padded full-image coverage. Width=512, hidden=1024, depth=2; recorded operations gelu, pad. Window MLP hidden/output=128/64; spatial sizes 32 → 31 → 16 → 8 → 4 → 2 → 1. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=60, seed=42; augmentation=True. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-224057-085c2c3b` | 88.5156% | 76.1% | 75.45% | 12.61 minutes | 51 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 3.459 → 0.4437 | 0.4437 (60) | 1.884 → 0.7835 | 0.7223 (43) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [cifar10-20261026-224057-085c2c3b](outputs/experiment-history/run-index.md#cifar10-20261026-224057-085c2c3b).

**Draft notes:** Padding preserves previously discarded edges and adds a sixth stage. This changes coverage and depth together.

## I1 — Flat gated MLP

### Change 140 — imagenet32 — 2026-10-26

_Flat gated MLP on ImageNet32_

**Change:** normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version. Width=512, hidden=1024, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=10, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet32-20261026-195826-4b2dc142` | 18.7556% | 12.1632% | 11.106% | 42.22 minutes | 10 | — |
| `imagenet32-20261026-204057-1f232b47` | — | — | — | — | — | No final score; 4 completed epoch rows saved |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| `imagenet32-20261026-195826-4b2dc142` | 6.185 → 4.644 | 4.644 (10) | 5.829 → 4.963 | 4.963 (10) |
| `imagenet32-20261026-204057-1f232b47` | 6.185 → 5.263 | 5.263 (4) | 5.829 → 5.212 | 5.212 (4) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [imagenet32-20261026-195826-4b2dc142](outputs/experiment-history/run-index.md#imagenet32-20261026-195826-4b2dc142), [imagenet32-20261026-204057-1f232b47](outputs/experiment-history/run-index.md#imagenet32-20261026-204057-1f232b47).

**Draft notes:** Historical model had no residual addition despite its class name. Record this limitation rather than calling it a residual architecture.

### Change 141 — imagenet32 — 2026-10-26

_Normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version_

**Change:** normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version. Width=512, hidden=1024, depth=6; recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=20, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet32-20261026-210325-b9c04e03` | 28.1384% | 12.7307% | 11.648% | 114.63 minutes | 17 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 6.185 → 3.954 | 3.954 (20) | 5.829 → 5.18 | 4.947 (11) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [imagenet32-20261026-210325-b9c04e03](outputs/experiment-history/run-index.md#imagenet32-20261026-210325-b9c04e03).

**Draft notes:** Historical model had no residual addition despite its class name. Record this limitation rather than calling it a residual architecture.

## I2 — Rolling-window MLP

### Change 142 — imagenet32 — 2026-10-27

_Rolling-window MLP on ImageNet32_

**Change:** hierarchical 2×2 rolling-window MLP with padded full-image coverage. Width=512, hidden=1024, depth=2; recorded operations gelu, pad. Window MLP hidden/output=128/64; spatial sizes 32 → 31 → 16 → 8 → 4 → 2 → 1. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=10, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet32-20261027-000850-1c570f4f` | 26.4322% | 20.4939% | 18.496% | 62.01 minutes | 10 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 5.692 → 3.827 | 3.827 (10) | 5.055 → 4.117 | 4.117 (10) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [imagenet32-20261027-000850-1c570f4f](outputs/experiment-history/run-index.md#imagenet32-20261027-000850-1c570f4f).

## I3 — Continued training (analysis only)

### Change 143 — imagenet32 — 2026-10-27

_Continue ImageNet training (analysis experiment)_

**Change:** hierarchical 2×2 rolling-window MLP with padded full-image coverage. Width=512, hidden=1024, depth=2; recorded operations gelu, pad. Window MLP hidden/output=128/64; spatial sizes 32 → 31 → 16 → 8 → 4 → 2 → 1. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=100, seed=42; augmentation=False. Other optimizer, split and augmentation settings are preserved in the evidence snapshot.

**Why:** TODO — add your own reason, based on the evidence you had then.

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet32-continuation-20261027-013419-8e7bc688` | 37.1263% | 21.9809% | 20.004% | 134.60 minutes | 25 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 5.692 → 3.229 | 3.229 (25) | 5.055 → 4.106 | 4.079 (16) |

TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [imagenet32-continuation-20261027-013419-8e7bc688](outputs/experiment-history/run-index.md#imagenet32-continuation-20261027-013419-8e7bc688).

**Draft notes:** Same model as parent, continued from saved weights with optimizer/random state restarted. This is analysis evidence, not fresh-from-scratch grading performance.

## Separate branch — MLP-Mixer

### Change 144 — mnist / cifar10 / imagenet32 — 2026-10-26

_Separate MLP-Mixer implementation_

**Change:** Separate `t3/implement-google-paper` worktree: non-overlapping 4×4 linear patches; eight Mixer blocks; channel width 512; token/channel hidden widths 256/2048; pre-LayerNorm, residual token/channel mixing, mean over tokens and a linear head. Exact task defaults are implementation settings, not a saved full-dataset experiment.

**Why:** “Note2: Given that I just discovered it was Google that came up with the idea of the MLP mixer. I'm gonna take their word for it. https://arxiv.org/html/2105.01601v4”

**Expected:** TODO — no prediction located. If reconstructing it now, label it retrospective.

**Result:**

| Evidence status | Recorded result |
| --- | --- |
| Synthetic checks | Journal records synthetic checks, including Metal training. |
| Full-dataset scores | None recorded. |

TODO — describe your actual reason and expectation, cite the paper, and add a run ID only after a dataset experiment exists.

## Final harness preparation

### Change 145 — mnist / cifar10 / imagenet32 — 2026-10-27

**Change:** Freeze task-specific public submission recipes separately from the experimental dashboard. Restore the seven-block GELU MNIST model from `mnist-20261023-105208-872d486e`; retain the padded rolling-window model from `cifar10-20261026-224057-085c2c3b` and `imagenet32-20261027-000850-1c570f4f` for the other tasks. ImageNet now allows up to 25 fresh epochs. Use seed 0 in the instructor's original harness, CUDA on a high-RAM A100, and training allocations of 15/15/60 minutes. Normalize ImageNet minibatches without expanding the entire dataset into float tensors. Add optional fixed-batch diagnostics, disabled by default.

**Why:** User instruction: “Use the strongest documented validation recipes per task, within the time limit.” The ImageNet continuation used saved weights, so its 20.004% test accuracy does not demonstrate a valid fresh grading run. TODO — add your own explanation of why these earlier results support the final choices.

**Expected:** No personal prediction was recorded before this CUDA run. TODO — add your own reading, labeled retrospective. The implementation goal was a clean install and fresh training through the unchanged instructor evaluation cells; this is not a recovered personal prediction.

**Result:** The exact instructor harness passed on a high-RAM A100, seed 0, installed from commit `b5ab6e1f75de4cfc257cf85b7e2535cfa217934c`. All models trained from scratch.

| Run | Test accuracy | Validation accuracy | Training time | Selected epoch | Completed epochs | Stop reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mnist-c628ed51` | 99.150% | 99.3667% | 14.70 minutes | 270 | 382 | time_budget |
| `cifar10-32ae8cde` | 74.760% | 76.6400% | 3.82 minutes | 56 | 60 | epochs |
| `imagenet32-8856b095` | 19.960% | 22.0222% | 21.81 minutes | 23 | 25 | epochs |

| Dataset | Train loss (start to end) | Validation loss (start to end) | Minimum validation loss | Minimum epoch |
| --- | --- | --- | ---: | ---: |
| MNIST | 0.742 to 0.012 | 0.218 to 0.027 | 0.024 | 361 |
| CIFAR10 | 3.574 to 0.457 | 1.909 to 0.796 | 0.736 | 56 |
| ImageNet32 | 5.651 to 3.252 | 5.013 to 4.111 | 4.070 | 15 |

| Summary measure | Result |
| --- | ---: |
| Total training time | 40.33 minutes |

The ImageNet fixed-batch diagnostics contain eight snapshots, including initialization and selected epoch 23.

| Residual classifier branch removed | Validation accuracy drop (percentage points) |
| --- | ---: |
| First | 10.1634 |
| Second | 3.6131 |

Full validation accuracy after restoring the saved selected model exactly matched training's selected checkpoint. TODO — add your own interpretation and whether the result matched your expectation.

**Runs:** [Fresh harness evidence and run records](outputs/final-evaluation/README.md). [Report images](outputs/report-figures/README.md). The historical 20.004% continuation remains separate from this fresh 19.960% result.

## Follow-up analysis — window-stage spatial ablations — 2026-10-27

The saved epoch-23 ImageNet model was evaluated without retraining on the same
validation split. At each of its six window stages, replacing each neighborhood
with repeated copies of its top-left pixel or its mean preserved channel
mapping and shape while removing spatial information.

| Evaluation | Stage | Validation accuracy |
| --- | ---: | ---: |
| Mean-only ablation | 1 | 0.5823% |
| Mean-only ablation | 4 | 16.4515% |
| Full model | — | 22.0222% |

Weights and restored accuracy were unchanged. [All twelve
results and methods](outputs/window-stage-ablation/README.md).

TODO — add your own interpretation. These controls retain the learned MLP and
should not be described as deleting the whole stage.

### Change 146 — imagenet32 — 2026-10-28

**Change:** Add one residual MLP before the last 2×2 to 1×1 window stage. Flatten 64×2×2 to 256; pre-LayerNorm, Linear(256,512), GELU, Linear(512,256), add the original stream, and reshape. Add 263,424 parameters. Keep the final image representation, training recipe, seed 0, and split unchanged.

**Why:** The user asked, “It almost seems to imply there's more signal to be extracted with additional layers somehow?” Earlier spatial controls showed sensitivity before the final compression. The extra block tests additional processing there. TODO — add your own interpretation in your own words.

**Expected:** Recorded before training: assistant-proposed hypothesis, accepted for one experiment, that extra processing before compression may increase best validation accuracy above 22.022230%. Extra capacity may instead overfit; compare validation loss and the training/validation gap. No numerical gain was predicted. This is not a recovered personal prediction.

**Result:** One fresh seed-0 run, compared with the qualified baseline:

| Measure | Baseline | Extra block |
| --- | ---: | ---: |
| Best validation accuracy | 22.0222% | 21.1067% |
| Selected epoch / completed epochs | 23 / 25 | 25 / 25 |
| Accuracy change from baseline (percentage points) | — | -0.9156 |
| Minimum validation loss | 4.069596 (epoch 15) | 4.101746 (epoch 25) |
| Epoch-25 training loss | 3.252028 | 3.419130 |
| Training time (minutes) | 21.81 | 23.52 |

The accuracy hypothesis was not supported within this fixed 25-epoch recipe. The candidate was still improving at the cutoff; no continuation or retry was run. No test-set evaluation or default change. TODO — add your own interpretation.

**Runs:** `imagenet32-prereduction-20261028-s0`; [pre-run protocol](outputs/pre-reduction-experiment/intent.json); [results, curves, and evidence](outputs/pre-reduction-experiment/README.md).

## Next entry

### Change 147 — TODO task — TODO date

**Change:** TODO

**Why:** TODO — connect the change to an earlier result.

**Expected:** TODO — write this before running the experiment.

**Result:** TODO — accuracy, training time, loss-curve behavior and whether the prediction held.

| Measure | Result |
| --- | --- |
| Accuracy | TODO |
| Training time | TODO |
| Loss-curve behavior | TODO |
| Whether the prediction held | TODO |

**Runs:** TODO — add the saved run ID.
