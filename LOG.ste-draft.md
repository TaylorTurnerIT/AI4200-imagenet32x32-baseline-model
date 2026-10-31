# Development Log — reconstructed draft

Taylor Turner

Writing style: [80% ASD-STE100](https://github.com/JAICHANGPARK/ASD-STE100). Source: [LOG.md](LOG.md) at commit `6b7e86d`.

I reconstructed this log from [journal.md](journal.md) and saved experiments. New reasons and expectations are marked as retrospective reconstructions or estimates. They are not predictions recorded before training. Recorded measurements stay exact. Missing measurements and settings remain explicitly unknown.

## Era figures — design draft

Each task has an accuracy timeline. A dot marks each recorded result. A step line shows the best result. Era boundaries mark transitions. Downward markers show clipped low outliers. The [interactive preview](outputs/experiment-history/accuracy-preview.html) also shows full-range and log-error scales.

![MNIST experiment accuracy](outputs/experiment-history/figures/mnist-accuracy.png)

![CIFAR-10 experiment accuracy](outputs/experiment-history/figures/cifar10-accuracy.png)

![ImageNet32 experiment accuracy](outputs/experiment-history/figures/imagenet32-accuracy.png)

## Baseline reference

### Change 1 — mnist / cifar10 / imagenet32 — 2026-10-15

_Original baseline_

**Change:** Run the supplied baseline with 30 residual blocks, width 128, hidden width 256, tanh, pre-LayerNorm, Xavier initialization, and SGD. Set learning rate to 0.05 and batch to 256. Source commit: `c1f6802`. Set epochs to MNIST 10, CIFAR-10 30, and ImageNet32 10.

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

Per-epoch MNIST and CIFAR-10 loss was not preserved. The early CSV rows do not contain enough configuration detail to confirm that both runs used the unchanged baseline.

## M1 — Baseline, capacity and regularization

### Change 2 — mnist — 2026-10-19

_Learning rate 0.01_

**Change:** Keep the baseline architecture and 10 epochs. Set learning rate to 0.01 and seed to 42.

**Why:** (Retrospective reconstruction) I wanted to compare a lower learning rate with the baseline.

**Expected:** (Retrospective estimate) A lower rate might train more steadily, but could make less progress in 10 epochs.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 95.49% |
| Training time | 0.6 minutes |

The lower rate did not improve short-run accuracy. No saved curve establishes whether it made the updates smoother.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 5 (raw rows kept in the evidence snapshot).

### Change 3 — mnist — 2026-10-19

_Learning rate 0.10_

**Change:** Keep the baseline architecture and 10 epochs. Set learning rate to 0.10 and seed to 42.

**Why:** (Retrospective reconstruction) I wanted to compare a higher learning rate with the baseline.

**Expected:** (Retrospective estimate) A higher rate might improve progress within 10 epochs.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 96.74% |
| Training time | 0.5 minutes |

The higher rate improved the short-run score. That fits the reconstructed convergence idea, but the loss curve is unavailable.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 6 (raw rows kept in the evidence snapshot).

### Change 4 — mnist — 2026-10-19

_Learning rate 0.15_

**Change:** Keep the baseline architecture and 10 epochs. Set learning rate to 0.15 and seed to 42.

**Why:** (Retrospective reconstruction) I wanted to test a still higher learning rate.

**Expected:** (Retrospective estimate) A higher rate might improve progress, but could make the updates less stable.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 96.11% |
| Training time | 0.5 minutes |

Another rate increase gave a worse score than the preceding rate. A larger step was not automatically better.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 7 (raw rows kept in the evidence snapshot).

### Change 5 — mnist — 2026-10-19

_50 epochs at learning rate 0.01_

**Change:** Set epochs to 50 and learning rate to 0.01. Keep the architecture unchanged.

**Why:** (Retrospective reconstruction) I wanted to see whether more epochs would improve the low-rate run.

**Expected:** (Retrospective estimate) More epochs might improve accuracy at this lower rate.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.56% |
| Training time | 2.5 minutes |

Longer training improved the low-rate score. No saved curve shows exactly when that gain developed.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 8 (raw rows kept in the evidence snapshot).

### Change 6 — mnist — 2026-10-19

_50 epochs at learning rate 0.10_

**Change:** Set epochs to 50 and learning rate to 0.10. Keep the architecture unchanged.

**Why:** (Retrospective reconstruction) I wanted to see whether more epochs would improve the higher-rate run.

**Expected:** (Retrospective estimate) More epochs might improve accuracy before training levels off.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.12% |
| Training time | 2.4 minutes |

Longer training improved the higher-rate score, but took longer. This fits the idea that the earlier cutoff limited the result.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 9 (raw rows kept in the evidence snapshot).

### Change 7 — mnist — 2026-10-19

_50 epochs at learning rate 0.15_

**Change:** Set epochs to 50 and learning rate to 0.15. Keep the architecture unchanged.

**Why:** (Retrospective reconstruction) I wanted to compare the same longer training at a higher rate.

**Expected:** (Retrospective estimate) More epochs might improve accuracy, though the higher rate could remain less stable.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.97% |
| Training time | 2.4 minutes |
| First loss spike | 0.5566 |
| Second loss spike | 0.2268 |

“We see here lots of big jumps followed by recovery. Not as good as a=0.10.” The journal shows loss spikes followed by recovery. The final score stayed below the middle-rate comparison.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 10 (raw rows kept in the evidence snapshot).

### Change 8 — mnist — 2026-10-19

_Add a training-accuracy check_

**Change:** Record training accuracy and test accuracy for the 50-epoch learning-rate comparisons.

**Why:** “I realized I didn't record train accuracy so I can't compare. Let's understand how train accuracy is looking.”

**Expected:** (Retrospective estimate) Training accuracy would show whether the models fit the training set.

**Result:**

| Learning rate | Training accuracy | Test accuracy |
|---:|---:|---:|
| 0.05 | 100.00% | 98.12% |
| 0.10 | 99.97% | 98.10% |
| 0.15 | 99.94% | 97.97% |

| Measure | Result |
|---|---|
| Training time per setup | 2.4 minutes |

“We see that the baseline learning rate was perfect for this current configuration. All it truly needed was a few more epochs.” Training accuracy was near its ceiling while test accuracy stayed lower. High training fit had not removed the generalization gap.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 11, 12, 13 (raw rows kept in the evidence snapshot).

**Draft notes:** The 0.15 CSV row appears to encode 99.93% train, while the journal says 99.94%. Preserve this discrepancy until checked.

### Change 9 — mnist — 2026-10-19

_40-layer comparison with 25 epochs_

**Change:** The journal records layers=40, epochs=25, seed=42, and `a=0.5`.

**Why:** “After my findings with accuracy revealed that the default learning rate was a solid starting point, I'm ready to move onto layers.”

**Expected:** “Right now I assume the issue is we're losing valuable signal as the layer layers add nothing but noise to an ultimately simple problem. I expect to see some small number of layers, perhaps 10-15 perform the same or better.”

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.79% |
| Training accuracy | 99.7% |
| Training time | 1.2 minutes |

The fit remained strong, but the depth, epoch count, and possible learning-rate typo make the comparison uncertain. No full loss curve survives.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 14 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records `a=0.5` and 40 layers. No saved setup resolves whether either value is a typo; both remain as recorded.

### Change 10 — mnist — 2026-10-19

_Five layers_

**Change:** Reduce recorded depth from 40 to 5. Keep 25 epochs and the preceding setup. The learning rate remains uncertain.

**Why:** (Retrospective reconstruction) I wanted to test fewer layers after predicting a smaller model could do as well or better.

**Expected:** “I expect to see some small number of layers, perhaps 10-15 perform the same or better.”

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.89% |
| Training accuracy | 99.73% |
| Training time | 0.3 minutes |

“An incredible gain! As predicted we actually see a performance increase by decreasing the layers and a gain of 1.2 minutes” The smaller stack trained faster without losing test accuracy. The journal’s stated time gain does not match the recorded times.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 15 (raw rows kept in the evidence snapshot).

**Draft notes:** Journal’s claimed 1.2-minute gain conflicts with the recorded 1.2-to-0.3-minute change (0.9 minutes). The recorded time fell from 1.2 to 0.3 minutes, a difference of 0.9 minutes. The test score rose by 0.10 percentage points, not enough to establish that depth alone caused the change because the learning rate is uncertain.

### Change 11 — mnist — 2026-10-19

_1 layers_

**Change:** Set depth to 1. Keep the 25-epoch depth-comparison setup.

**Why:** (Retrospective reconstruction) I wanted to compare very shallow models after the five-layer result.

**Expected:** (Retrospective estimate) A shallow model might approach the five-layer result.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.73% |
| Training accuracy | 99.06% |
| Training time | 0.4 minutes |

The shallow model learned, but its score fell slightly. Reducing depth did not preserve every part of the result.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 16 (raw rows kept in the evidence snapshot).

### Change 12 — mnist — 2026-10-19

_2 layers_

**Change:** Set depth to 2. Keep the 25-epoch depth-comparison setup.

**Why:** (Retrospective reconstruction) I wanted to compare very shallow models after the five-layer result.

**Expected:** (Retrospective estimate) A shallow model might approach the five-layer result.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.77% |
| Training accuracy | 99.09% |
| Training time | 0.2 minutes |

Adding back some depth recovered a little accuracy while the model remained inexpensive to train.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 17 (raw rows kept in the evidence snapshot).

### Change 13 — mnist — 2026-10-19

_3 layers_

**Change:** Set depth to 3. Keep the 25-epoch depth-comparison setup.

**Why:** (Retrospective reconstruction) I wanted to compare very shallow models after the five-layer result.

**Expected:** (Retrospective estimate) A shallow model might approach the five-layer result.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.8% |
| Training accuracy | 99.57% |
| Training time | 0.3 minutes |

More depth improved training fit much more than the test score. The added fit was not translating equally into generalization.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 18 (raw rows kept in the evidence snapshot).

### Change 14 — mnist — 2026-10-19

_4 layers_

**Change:** Set depth to 4. Keep the 25-epoch depth-comparison setup.

**Why:** (Retrospective reconstruction) I wanted to compare very shallow models after the five-layer result.

**Expected:** (Retrospective estimate) A shallow model might approach the five-layer result.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.92% |
| Training accuracy | 99.67% |
| Training time | 0.9 minutes |
| Loss milestone | 0.0x range within approximately seven epochs |

The score improved, and the journal records an early loss reduction. The full curve is not available.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 19 (raw rows kept in the evidence snapshot).

### Change 15 — mnist — 2026-10-19

_6 layers_

**Change:** Set depth to 6. Keep the 25-epoch depth-comparison setup.

**Why:** “For now, let's try a few more until we see a performance regression.”

**Expected:** (Retrospective estimate) Adding layers might improve accuracy until depth began to hurt performance.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.03% |
| Training accuracy | 99.79% |
| Training time | 0.4 minutes |

The score rose again. This did not establish that further depth increases would continue to help.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 20 (raw rows kept in the evidence snapshot).

### Change 16 — mnist — 2026-10-19

_7 layers_

**Change:** Set depth to 7. Keep the 25-epoch depth-comparison setup.

**Why:** “For now, let's try a few more until we see a performance regression.”

**Expected:** (Retrospective estimate) Adding another layer might improve accuracy or reveal a regression.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.6% |
| Training accuracy | 99.67% |
| Training time | 0.5 minutes |

“Regression. Let's roll with 6 layers for now. Going to try a run with 50 epochs and 6 layers.” The deeper trial regressed, making the earlier shallower setup a better working choice.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 21 (raw rows kept in the evidence snapshot).

### Change 17 — mnist — 2026-10-19

_50 epochs after depth sweep_

**Change:** Set epochs to 50. The recorded depth is 6, but the actual depth was probably 7.

**Why:** (Retrospective reconstruction) I wanted to check whether longer training helped the six-layer setting.

**Expected:** (Retrospective estimate) More epochs might reduce training loss, but might not improve test accuracy.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.01% |
| Training accuracy | 100% |
| Training time | 0.9 minutes |
| Last four recorded losses (in order) | 0.0012, 0.0011, 0.0010, 0.0010 |

“Zero change in the last epoch. Clearly reached the floor of this configuration.” More epochs brought training fit to its ceiling without improving test accuracy. The saved ending loss values changed little.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 22 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 18 — mnist — 2026-10-19

_Residual width 32_

**Change:** Set width=32, hidden=256, epochs=25, and learning_rate=0.05. The recorded depth is 6, but it was probably 7.

**Why:** “My ideas here are to change the size of the residual stream (width) and the size of the hidden layers (hidden).”

**Expected:** “Perhaps a smaller residual stream is equally capable, perhaps a larger residual layer will allow more signal through.”

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 95.92% |
| Training accuracy | 96.93% |
| Training time | 0.4 minutes |

“Immediately slower training. Didn't reach 0.0# until epoch 18.” Reducing branch capacity hurt both training and test accuracy. That did not look like a simple cure for overfitting.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 23 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 19 — mnist — 2026-10-19

_Residual width 256_

**Change:** Set width=256, hidden=256, epochs=25, and learning_rate=0.05. The recorded depth is 6, but it was probably 7.

**Why:** (Retrospective reconstruction) I wanted to test whether a wider residual stream preserved more signal.

**Expected:** “perhaps a larger residual layer will allow more signal through.”

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.04% |
| Training accuracy | 99.99% |
| Training time | 0.5 minutes |
| Loss milestone | 0.0x range in six epochs |

“Incredible training accuracy but a trivial improvement in test. All we did was improve the models ability to overfit.” Widening restored strong training fit, but the test score barely exceeded the earlier good setup.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 24 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 20 — mnist — 2026-10-19

_50 epochs at width 256_

**Change:** Keep width=256 and hidden=256. Extend training from 25 to 50 epochs.

**Why:** “Let's try a 50 epoch here since I see a bit more room for training loss.”

**Expected:** (Retrospective estimate) More training might reduce training loss, but might not improve test accuracy.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.01% |
| Training accuracy | 100% |
| Training time | 0.9 minutes |

“We gained an incredible -0.03%. I'd say that was just further overfitting pushing our performance worse.” More epochs improved an already strong training fit without a test gain. This matched the journal’s concern about further memorization.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 25 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 21 — mnist — 2026-10-19

_Hidden width 512_

**Change:** Set width=256, hidden=512, epochs=50, and learning_rate=0.05. The recorded depth is 6, but it was probably 7.

**Why:** “For now, let's see what happens when we tinker with the size of the hidden layers. Perhaps reducing the models capacity will serve a similar role.”

**Expected:** (Retrospective estimate) A larger hidden layer might change capacity and improve test accuracy.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.17% |
| Training accuracy | 100% |
| Training time | 1.0 minutes |

The capacity increase brought a small test improvement while training fit remained at its ceiling.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 26 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown. Journal settings increase hidden width, although the preceding idea mentions reducing capacity. The journal says reducing capacity was the goal, but this run increased hidden width from 256 to 512. The recorded configuration conflicts with that stated intention, so this is not a clean capacity reduction. The printed result is 98.17%, while the following sentence says 98.18%.

### Change 22 — mnist — 2026-10-19

_Residual width 512_

**Change:** Set width=512, hidden=512, epochs=50, and learning_rate=0.05. The recorded depth is 6, but it was probably 7.

**Why:** (Retrospective reconstruction) I wanted to test whether more residual width improved the result after increasing hidden width.

**Expected:** (Retrospective estimate) A wider residual stream might improve test accuracy by carrying more signal.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.16% |
| Training accuracy | 100% |
| Training time | 1.1 minutes |

Another capacity increase did not improve the test score. Training accuracy alone could no longer distinguish useful changes.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 27 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 23 — mnist — 2026-10-19

_Hidden width 1024_

**Change:** Set width=512, hidden=1024, epochs=50, and learning_rate=0.05. The recorded depth is 6, but it was probably 7.

**Why:** (Retrospective reconstruction) I wanted to test whether a larger hidden layer changed the result.

**Expected:** (Retrospective estimate) A larger hidden layer might change capacity, with more training cost.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.15% |
| Training accuracy | 100% |
| Training time | 1.4 minutes |

“Crazy experiments over with no real change. This is obviously a dead end. Onto Regularization.” The larger model cost more time and still did not improve generalization. Capacity alone looked less promising.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 28 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 24 — mnist — 2026-10-19

_Weight decay 0.99_

**Change:** Restore width=256 and hidden=512. Set weight_decay=0.99, learning_rate=0.05, and epochs=50. The recorded depth is 6, but it was probably 7.

**Why:** “Weight decay is a classic from ML. Let's try to force down those worthless weights and see if we can open up some capacity for the test accuracy to use.”

**Expected:** (Retrospective estimate) Strong decay might reduce useful weight magnitude and restrict learning.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 11.35% |
| Training accuracy | 11.24% |
| Training time | 1.0 minutes |
| Loss change across 50 epochs | about 0.0001 |

“Didn't learn a thing.” Both accuracies remained near chance, and the journal notes almost no loss change. This regularization setting prevented useful learning.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 29 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown. In SGD, weight decay adds a term proportional to each parameter to the gradient. A value of 0.99 is a strong penalty coefficient; it does not mean “99% of squared magnitude allowed to affect weight decay.”

### Change 25 — mnist — 2026-10-19

_Weight decay 0.01_

**Change:** Reduce weight_decay from 0.99 to 0.01. Set epochs=25, width=256, hidden=512, and learning_rate=0.05.

**Why:** (Retrospective reconstruction) The 0.99 decay prevented useful learning. I reduced it to see whether weaker decay could recover accuracy.

**Expected:** (Retrospective estimate) Less decay might recover learning while still limiting overfitting.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 87.79% |
| Training accuracy | 86.78% |
| Training time | 0.5 minutes |

The next setting improved on the failed run, but both scores stayed below the earlier working recipe.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 30 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 26 — mnist — 2026-10-19

_Weight decay 0.0001_

**Change:** Set weight_decay=0.0001. Keep 25 epochs, width=256, hidden=512, and learning_rate=0.05.

**Why:** “Let's try suggestion from the internet, 1e-4.”

**Expected:** (Retrospective estimate) A small decay value might preserve learning while limiting overfitting.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.09% |
| Training accuracy | 99.96% |
| Training time | 0.5 minutes |
| Recorded loss epochs | 22–25 |
| Recorded losses (in order) | 0.0093, 0.0077, 0.0060, 0.0047 |

Lighter decay restored strong fit, and the saved ending losses fell. That provided a working base for a longer run.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 31 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown. The journal says only “suggestion from the internet”; it does not name a source.

### Change 27 — mnist — 2026-10-19

_50 epochs with weight decay 0.0001_

**Change:** Extend the regularized configuration to 50 epochs. Keep width=256, hidden=512, and learning_rate=0.05.

**Why:** “This is interesting, we're almost back where we started but we still have a little wiggle room with train accuracy. I still see some more downward motion in the loss as well.”

**Expected:** (Retrospective estimate) More epochs might use the remaining room in training loss and improve test accuracy.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.24% |
| Training accuracy | 100% |
| Training time | 1.0 minutes |

“The plan paid off! We closed the train accuracy to 100% again and gained 0.10 of test accuracy.” The longer run improved the test score and restored perfect training accuracy. This fits the reconstructed expectation of more useful learning at this decay.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 32 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 28 — mnist — 2026-10-19

_Weight decay 0.001_

**Change:** Increase weight_decay to 0.001. Keep 50 epochs, width=256, hidden=512, and learning_rate=0.05.

**Why:** (Retrospective reconstruction) The 0.0001 run left room for more regularization. I increased decay to test that.

**Expected:** (Retrospective estimate) More decay might leave more room between training and test accuracy.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.92% |
| Training accuracy | 99.27% |
| Training time | 1.0 minutes |

“Tried another jump in regularization and I'm starting to see a lot of oscillation. We've lost test accuracy in exchange for some train room.” Stronger decay hurt this result. The journal describes oscillation, which motivated changing the learning rate next.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 33 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 29 — mnist — 2026-10-19

_Learning rate 0.03 with decay 0.001_

**Change:** Set learning_rate=0.03. Keep weight_decay=0.001, epochs=50, width=256, and hidden=512.

**Why:** “We've lost test accuracy in exchange for some train room. Let me try a smaller learning rate.”

**Expected:** (Retrospective estimate) A lower rate might reduce oscillation while retaining some training progress.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.06% |
| Training accuracy | 99.66% |
| Training time | 1.0 minutes |

“a=0.03 worked. I saw oscillation happening and it predictably solved it.” The lower rate improved the score after the stronger-decay trial. The journal reports less oscillation, but no complete epoch artifact survives.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 34 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 30 — mnist — 2026-10-19

_Learning rate 0.01 with decay 0.001_

**Change:** Set learning_rate=0.01. Keep weight_decay=0.001, epochs=50, width=256, and hidden=512.

**Why:** (Retrospective reconstruction) The 0.03 rate improved the result. I lowered it again to check training behavior.

**Expected:** (Retrospective estimate) A lower rate might reduce oscillation but slow training.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.97% |
| Training accuracy | 99.91% |
| Training time | 1.0 minutes |

“With 0.01 we're seeing incredibly slow training with more room for training by epoch 50.” An even smaller rate improved training fit but not the test score. The journal describes slow learning with room left at the cutoff.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 35 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

## M2 — Momentum and depth corrections

### Change 31 — mnist — 2026-10-20

_SGD momentum 0.9_

**Change:** Set momentum=0.9. Keep learning_rate=0.01, weight_decay=0.001, epochs=50, width=256, and hidden=512.

**Why:** “We're already using SGD, I have no interest in trying classical GD due to time and probable worse performance. So, let's try just adding some momentum and see what happens.”

**Expected:** (Retrospective estimate) Momentum might improve progress, though it could overshoot.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 96.66% |
| Training accuracy | 98.25% |
| Training time | 1.1 minutes |

“Well that didn't work. Let's try less.” High momentum worsened both scores under this recipe. That motivated trying less momentum instead of assuming faster updates would help.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 36 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 32 — mnist — 2026-10-20

_SGD momentum 0.5_

**Change:** Reduce momentum to 0.5. Keep learning_rate=0.01, weight_decay=0.001, epochs=50, width=256, and hidden=512.

**Why:** (Retrospective reconstruction) The 0.9 momentum result was poor. I reduced momentum to test a smaller update history.

**Expected:** (Retrospective estimate) Less momentum might improve on the 0.9 result.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98% |
| Training accuracy | 100% |
| Training time | 1.1 minutes |

Less momentum recovered strong training fit and much of the test score. Generalization still left room for improvement.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 37 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 33 — mnist — 2026-10-20

_Weight decay 0.005 with momentum 0.5_

**Change:** Set weight_decay=0.005. Keep momentum=0.5, learning_rate=0.01, epochs=50, width=256, and hidden=512.

**Why:** “Well that kinda worked. Maybe more regularization?”

**Expected:** (Retrospective estimate) More decay might regularize the model, but could limit learning.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 94.08% |
| Training accuracy | 94.67% |
| Training time | 1.1 minutes |

“Huge overshoot. Let's try less momentum then bring in dampening. Could also try a smaller learning rate.” This setting hurt both training and test accuracy. The journal calls it overshoot, but missing curves limit that diagnosis.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 38 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 34 — mnist — 2026-10-20

_Momentum 0.3 with decay 0.005_

**Change:** Set momentum=0.3. Keep weight_decay=0.005, batch=256, learning_rate=0.01, epochs=50, width=256, and hidden=512.

**Why:** (Retrospective reconstruction) The stronger decay result was poor. I reduced momentum after noting overshoot.

**Expected:** (Retrospective estimate) Less momentum might reduce overshoot.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 96.65% |
| Training accuracy | 97.73% |
| Training time | 1.1 minutes |

Less momentum recovered part of the lost score, but did not restore the earlier working result.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 39 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 35 — mnist — 2026-10-20

_Batch 128 and weight decay 0.001_

**Change:** Set batch=128 and weight_decay=0.001. Keep momentum=0.3, learning_rate=0.01, epochs=50, width=256, and hidden=512.

**Why:** “Doing something else as a waiting move. Changing weight decay back down and turning down batch size so it must generalize more.”

**Expected:** (Retrospective estimate) Smaller batches might improve generalization, with longer training time.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 96.36% |
| Training accuracy | 98.41% |
| Training time | 2.0 minutes |

“Still oscillating.” Training fit improved while test accuracy fell. Batch size and weight decay both changed, so this does not isolate the smaller batch’s effect.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 42 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown. The nearby CSV-only result at 11:42 is not assigned this setup because the journal does not identify it.

### Change 36 — mnist — 2026-10-20

_Learning rate 0.001 and momentum 0.9_

**Change:** Set learning_rate=0.001 and momentum=0.9. Keep batch=256, weight_decay=0.001, epochs=50, width=256, and hidden=512.

**Why:** “https://distill.pub/2017/momentum/ This post opened my eyes. Let's experiment. Maybe our learning rate is too high? Let's try more momentum, less learning rate.”

**Expected:** (Retrospective estimate) A lower rate with more momentum might improve training behavior.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.76% |
| Training accuracy | 99.96% |
| Training time | 1.2 minutes |

“Slow training, okay results.” The combined recipe fit the training data well, but the journal calls learning slow. Several settings changed together, limiting causal conclusions.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 43 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 37 — mnist — 2026-10-20

_Batch 128 at learning rate 0.001_

**Change:** Reduce batch to 128. Keep learning_rate=0.001, momentum=0.9, weight_decay=0.001, epochs=50, width=256, and hidden=512.

**Why:** “Slow training, okay results. More batch to make more little jumps?”

**Expected:** (Retrospective estimate) A smaller batch might produce more optimizer updates per epoch and alter convergence.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.85% |
| Training accuracy | 99.93% |
| Training time | 2.0 minutes |

“Faster convergence, seeing some oscillation.” The smaller batch gave a small test gain and more updates per epoch. The journal still reports oscillation.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 44 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown. A batch of 128 gives more optimizer updates per epoch than a batch of 256. The journal’s “More batch” wording is ambiguous; the recorded setting was a smaller batch.

### Change 38 — mnist — 2026-10-20

_Learning rate 0.0001_

**Change:** Set learning_rate=0.0001. Keep batch=128, momentum=0.9, weight_decay=0.001, epochs=50, width=256, and hidden=512.

**Why:** (Retrospective reconstruction) I lowered the learning rate again after comparing momentum and batch settings.

**Expected:** (Retrospective estimate) A lower rate might improve the result, but take longer to train.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 97.94% |
| Training accuracy | 100% |
| Training time | 2.9 minutes |

The smallest rate cost more time for a small score change. Training accuracy reached its ceiling without settling the generalization question.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 45 (raw rows kept in the evidence snapshot).

**Draft notes:** The journal records depth 6, but its retrospective correction says these runs probably used 7. Exact transition time is unknown.

### Change 39 — mnist — 2026-10-20

_Accidental weight decay 0.00001_

**Change:** Set depth=7, width=256, hidden=512, batch=256, learning_rate=0.05, and epochs=50. The entered weight_decay was 0.1e-4 (=0.00001) by mistake.

**Why:** (Retrospective reconstruction) I wanted to verify the promising regularized setup at depth 7.

**Expected:** (Retrospective estimate) The corrected depth and small decay value might retain high test accuracy.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.22% |
| Training accuracy | 100% |
| Training time | 1.0 minutes |

“I actually made a mistake, I forgot to remove the 0. before the 1 in weight_decay and I got this 98.22%.” The accidental setting produced a strong score, but this is not a clean depth-and-decay comparison. The early depth transition remains uncertain.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 47 (raw rows kept in the evidence snapshot).

**Draft notes:** The preceding 12:08 CSV result also scored 98.22%, but its settings are not independently recorded. It remains in the unresolved list.

### Change 40 — mnist — 2026-10-20

_Correct decay to 0.0001_

**Change:** Keep depth=7, width=256, hidden=512, batch=256, learning_rate=0.05, and epochs=50. Set weight_decay=0.0001.

**Why:** (Retrospective reconstruction) I corrected the accidental weight decay value to the intended 0.0001.

**Expected:** (Retrospective estimate) Correcting decay to 0.0001 might keep the result near the accidental run.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.16% |
| Training accuracy | 100% |
| Training time | 0.9 minutes |

Correcting the decay slightly lowered the score. One comparison with uncertain earlier depth does not settle which setting generalizes better.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 48 (raw rows kept in the evidence snapshot).

### Change 41 — mnist — 2026-10-20

_Explicitly test depth 6_

**Change:** Change depth from 7 to 6. Keep width=256, hidden=512, batch=256, learning_rate=0.05, epochs=50, and weight_decay=0.0001.

**Why:** (Retrospective reconstruction) I wanted to compare depth 6 directly with depth 7.

**Expected:** (Retrospective estimate) Depth 6 might perform similarly to depth 7.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.24% |
| Training accuracy | 100% |
| Training time | 1.0 minutes |

The journal and later replay disagree about which depth owns the scores. I keep those observations separate instead of declaring a winning depth.

These values match the old row’s shifted layout.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 49 (raw rows kept in the evidence snapshot).

**Draft notes:** Later retrospective replay evidence in the journal associates 98.24% with seven blocks and 98.16% with six. These contemporary labels say the reverse. The historical rows report 98.24% at recorded depth 6 and 98.16% at recorded depth 7. A later replay associates those scores with the opposite depths. The original rows remain unchanged; the replay cannot establish which historical label is correct.

### Change 42 — mnist — 2026-10-20

_Unmatched legacy result at 11:42_

**Change:** Unmatched legacy result. No exact setup or run directory was saved, so this row cannot support an isolated or reproducible comparison.

**Why:** (Retrospective reconstruction) I was still trying to get more out of MNIST before moving on. The exact change behind this row is not recoverable.

**Expected:** (Retrospective estimate) A better MNIST result was the goal. Without the settings, I cannot assign a specific mechanism to this row.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 97.65% |

The CSV preserves a test score without a reproducible setup. Training time, train accuracy, and loss behavior cannot be assigned reliably.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 40 (raw rows kept in the evidence snapshot).

### Change 43 — mnist — 2026-10-20

_Unmatched legacy result at 11:44_

**Change:** Unmatched legacy result. No exact setup or run directory was saved, so this row cannot support an isolated or reproducible comparison.

**Why:** (Retrospective reconstruction) I was still trying to get more out of MNIST before moving on. The exact change behind this row is not recoverable.

**Expected:** (Retrospective estimate) A better MNIST result was the goal. Without the settings, I cannot assign a specific mechanism to this row.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 96.36% |

The CSV preserves a test score without a reproducible setup. Training time, train accuracy, and loss behavior cannot be assigned reliably.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 41 (raw rows kept in the evidence snapshot).

### Change 44 — mnist — 2026-10-20

_Unmatched legacy result at 12:08_

**Change:** Unmatched legacy result. No exact setup or run directory was saved, so this row cannot support an isolated or reproducible comparison.

**Why:** (Retrospective reconstruction) I was still trying to get more out of MNIST before moving on. The exact change behind this row is not recoverable.

**Expected:** (Retrospective estimate) A better MNIST result was the goal. Without the settings, I cannot assign a specific mechanism to this row.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.22% |

The CSV preserves a test score without a reproducible setup. Training time, train accuracy, and loss behavior cannot be assigned reliably.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 46 (raw rows kept in the evidence snapshot).

### Change 45 — mnist — 2026-10-20

_Unmatched legacy result at 12:40_

**Change:** Unmatched legacy result. No exact setup or run directory was saved, so this row cannot support an isolated or reproducible comparison.

**Why:** (Retrospective reconstruction) I was still trying to get more out of MNIST before moving on. The exact change behind this row is not recoverable.

**Expected:** (Retrospective estimate) A better MNIST result was the goal. Without the settings, I cannot assign a specific mechanism to this row.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.18% |

The CSV preserves a test score without a reproducible setup. Training time, train accuracy, and loss behavior cannot be assigned reliably.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 50 (raw rows kept in the evidence snapshot).

## M3 — Longer training, validation and augmentation

### Change 46 — mnist — 2026-10-22

_Unmatched legacy result at 14:49_

**Change:** Unmatched legacy result. No exact setup or run directory was saved, so this row cannot support an isolated or reproducible comparison.

**Why:** (Retrospective reconstruction) I was still trying to get more out of MNIST before moving on. The exact change behind this row is not recoverable.

**Expected:** (Retrospective estimate) A better MNIST result was the goal. Without the settings, I cannot assign a specific mechanism to this row.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.18% |

The CSV preserves a test score without a reproducible setup. Training time, train accuracy, and loss behavior cannot be assigned reliably.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 51 (raw rows kept in the evidence snapshot).

### Change 47 — mnist — 2026-10-22

_Unmatched legacy result at 14:54_

**Change:** Unmatched legacy result. No exact setup or run directory was saved, so this row cannot support an isolated or reproducible comparison.

**Why:** (Retrospective reconstruction) I was still trying to get more out of MNIST before moving on. The exact change behind this row is not recoverable.

**Expected:** (Retrospective estimate) A better MNIST result was the goal. Without the settings, I cannot assign a specific mechanism to this row.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.18% |

The CSV preserves a test score without a reproducible setup. Training time, train accuracy, and loss behavior cannot be assigned reliably.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 52 (raw rows kept in the evidence snapshot).

### Change 48 — mnist — 2026-10-22

_Unmatched legacy result at 15:33_

**Change:** Unmatched legacy result. No exact setup or run directory was saved, so this row cannot support an isolated or reproducible comparison.

**Why:** (Retrospective reconstruction) I was still trying to get more out of MNIST before moving on. The exact change behind this row is not recoverable.

**Expected:** (Retrospective estimate) A better MNIST result was the goal. Without the settings, I cannot assign a specific mechanism to this row.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.37% |

The CSV preserves a test score without a reproducible setup. Training time, train accuracy, and loss behavior cannot be assigned reliably.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 53 (raw rows kept in the evidence snapshot).

### Change 49 — mnist — 2026-10-22

_Unmatched legacy result at 17:30_

**Change:** Unmatched legacy result. No exact setup or run directory was saved, so this row cannot support an isolated or reproducible comparison.

**Why:** (Retrospective reconstruction) I was still trying to get more out of MNIST before moving on. The exact change behind this row is not recoverable.

**Expected:** (Retrospective estimate) A better MNIST result was the goal. Without the settings, I cannot assign a specific mechanism to this row.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.07% |

The CSV preserves a test score without a reproducible setup. Training time, train accuracy, and loss behavior cannot be assigned reliably.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 54 (raw rows kept in the evidence snapshot).

### Change 50 — mnist — 2026-10-22

_Unmatched legacy result at 17:38_

**Change:** Exact setup is not recoverable from the saved record. This result cannot support an isolated or reproducible comparison.

**Why:** (Retrospective reconstruction) I was still trying to get more out of MNIST before moving on. The exact change behind this row is not recoverable.

**Expected:** (Retrospective estimate) A better MNIST result was the goal. Without the settings, I cannot assign a specific mechanism to this row.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.24% |
| CSV warning | Column layout is shifted; printed time/train values require review |

Training time, train accuracy, and loss behavior are not recoverable from this shifted CSV row. The test accuracy alone cannot identify the setup or support a comparison.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 55 (raw rows kept in the evidence snapshot).

### Change 51 — mnist — 2026-10-22

_Unmatched legacy result at 17:48_

**Change:** Exact setup is not recoverable from the saved record. This result cannot support an isolated or reproducible comparison.

**Why:** (Retrospective reconstruction) I was still trying to get more out of MNIST before moving on. The exact change behind this row is not recoverable.

**Expected:** (Retrospective estimate) A better MNIST result was the goal. Without the settings, I cannot assign a specific mechanism to this row.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 97.86% |
| CSV warning | Column layout is shifted; printed time/train values require review |

Training time, train accuracy, and loss behavior are not recoverable from this shifted CSV row. The test accuracy alone cannot identify the setup or support a comparison.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 56 (raw rows kept in the evidence snapshot).

### Change 52 — mnist — 2026-10-22

_Unmatched legacy result at 17:49_

**Change:** Exact setup is not recoverable from the saved record. This result cannot support an isolated or reproducible comparison.

**Why:** (Retrospective reconstruction) I was still trying to get more out of MNIST before moving on. The exact change behind this row is not recoverable.

**Expected:** (Retrospective estimate) A better MNIST result was the goal. Without the settings, I cannot assign a specific mechanism to this row.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.24% |
| CSV warning | Column layout is shifted; printed time/train values require review |

Training time, train accuracy, and loss behavior are not recoverable from this shifted CSV row. The test accuracy alone cannot identify the setup or support a comparison.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 57 (raw rows kept in the evidence snapshot).

### Change 53 — mnist — 2026-10-22

_Unmatched legacy result at 21:34_

**Change:** Exact setup is not recoverable from the saved record. This result cannot support an isolated or reproducible comparison.

**Why:** (Retrospective reconstruction) I was still trying to get more out of MNIST before moving on. The exact change behind this row is not recoverable.

**Expected:** (Retrospective estimate) A better MNIST result was the goal. Without the settings, I cannot assign a specific mechanism to this row.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.09% |
| CSV warning | Column layout is shifted; printed time/train values require review |

Training time, train accuracy, and loss behavior are not recoverable from this shifted CSV row. The test accuracy alone cannot identify the setup or support a comparison.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 58 (raw rows kept in the evidence snapshot).

### Change 54 — mnist — 2026-10-22

_Unmatched legacy result at 21:47_

**Change:** Exact setup is not recoverable from the saved record. This result cannot support an isolated or reproducible comparison.

**Why:** (Retrospective reconstruction) I was still trying to get more out of MNIST before moving on. The exact change behind this row is not recoverable.

**Expected:** (Retrospective estimate) A better MNIST result was the goal. Without the settings, I cannot assign a specific mechanism to this row.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (CSV) | 98.37% |
| CSV warning | Column layout is shifted; printed time/train values require review |

Training time, train accuracy, and loss behavior are not recoverable from this shifted CSV row. The test accuracy alone cannot identify the setup or support a comparison.

**Evidence:** [journal.md](journal.md); legacy CSV line(s) 59 (raw rows kept in the evidence snapshot).

### Change 55 — mnist — 2026-10-22

_Validation-based checkpoint selection_

**Change:** Use a legacy MLP. The saved configuration contains the exact module shapes. The record does not include the activation or forward source.

- Optimizer: not recorded.
- Learning rate: 0.05.
- Weight decay: 0.0001.
- Batch size: 256.
- Epochs: 500.
- Seed: 42.
- Augmentation: not recorded.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** “When I implemented augmentation, I used Codex and it made an executive decision I was initially confused by. It created a train/test/validation split.”

**Expected:** (Retrospective estimate) Validation selection could keep a useful earlier checkpoint instead of assuming that the last epoch was best.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261022-220617-e4881f95 | 100% | 98.48% | 98.16% | 9.10 minutes | 368 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.6946 → 0.0002703 | 0.0002227 (278) | 0.3219 → 0.07032 | 0.05885 (358) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [mnist-20261022-220617-e4881f95](outputs/experiment-history/run-index.md#mnist-20261022-220617-e4881f95).

**Draft notes:** The legacy snapshot does not include activation/forward code. Do not infer it from a later restored version.

### Change 56 — mnist — 2026-10-22

_Legacy mlp (module shapes saved; forward source not captured)_

**Change:** Use a legacy MLP. The saved configuration contains the exact module shapes. The record does not include the activation or forward source.

- Optimizer: not recorded.
- Learning rate: 0.01.
- Weight decay: 0.0001.
- Batch size: 256.
- Epochs: 1000.
- Seed: 42.
- Augmentation: not recorded.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) Longer training had helped MNIST. I tried slower updates with a longer budget.

**Expected:** (Retrospective estimate) Slower updates with more epochs could improve the fit without the larger steps of the earlier recipe.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261022-230308-ed671f61 | — | — | — | — | — | 143 completed epoch rows; no final score |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.6247 → 0.000658 | 0.000658 (143) | 0.3862 → 0.09722 | 0.08053 (42) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup. No final score was saved, so the full-run expectation remains untested.

**Runs:** [mnist-20261022-230308-ed671f61](outputs/experiment-history/run-index.md#mnist-20261022-230308-ed671f61).

**Draft notes:** The legacy snapshot does not include activation/forward code. Do not infer it from a later restored version.

### Change 57 — mnist — 2026-10-22

_Legacy mlp (module shapes saved; forward source not captured)_

**Change:** Use a legacy MLP. The saved configuration contains the exact module shapes. The record does not include the activation or forward source.

- Optimizer: not recorded.
- Learning rate: 0.03.
- Weight decay: 0.0001.
- Batch size: 256.
- Epochs: 1000.
- Seed: 42.
- Augmentation: not recorded.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) The lower-rate run was slow. I tried a middle learning rate while keeping the longer budget.

**Expected:** (Retrospective estimate) A middle learning rate could make faster progress than the lower rate without losing as much control as a larger step.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261022-230552-e687600f | 100% | 98.42% | 98.06% | 18.50 minutes | 653 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.5453 → 0.000259 | 0.0002077 (341) | 0.3552 → 0.08314 | 0.06415 (615) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [mnist-20261022-230552-e687600f](outputs/experiment-history/run-index.md#mnist-20261022-230552-e687600f).

**Draft notes:** The legacy snapshot does not include activation/forward code. Do not infer it from a later restored version.

### Change 58 — mnist — 2026-10-23

_MNIST data augmentation_

**Change:** Use a legacy MLP. The saved configuration contains the exact module shapes. The record does not include the activation or forward source.

- Optimizer: not recorded.
- Learning rate: 0.05.
- Weight decay: 0.0001.
- Batch size: 256.
- Epochs: 400.
- Seed: 42.
- Augmentation: True.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** “Then, also building from what we did in class I decided to add data augmentation. Move the image around, blur it, brighten it, etc.”

**Expected:** (Retrospective estimate) Augmentation could reduce memorization of fixed digit positions. Training might become harder while validation accuracy improved.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261023-001324-8a4087e7 | 99.97% | 99.32% | 99.15% | 13.70 minutes | 315 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.095 → 0.01925 | 0.0156 (357) | 0.4271 → 0.02562 | 0.02282 (323) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [mnist-20261023-001324-8a4087e7](outputs/experiment-history/run-index.md#mnist-20261023-001324-8a4087e7).

**Draft notes:** The legacy snapshot does not include activation/forward code. Do not infer it from a later restored version.

### Change 59 — mnist — 2026-10-23

_Legacy mlp (module shapes saved; forward source not captured)_

**Change:** Use a legacy MLP. The saved configuration contains the exact module shapes. The record does not include the activation or forward source.

- Optimizer: not recorded.
- Learning rate: 0.05.
- Weight decay: 0.0001.
- Batch size: 256.
- Epochs: 500.
- Seed: 42.
- Augmentation: True.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) The augmented model needed time to learn. I gave it a longer budget and kept validation-based checkpoint selection.

**Expected:** (Retrospective estimate) More augmented training could improve the best validation score, even if the final epoch was not the best checkpoint.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261023-100855-d6864421 | 99.97% | 99.32% | 99.15% | 17.40 minutes | 315 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.095 → 0.01711 | 0.0156 (357) | 0.4271 → 0.02901 | 0.02282 (323) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [mnist-20261023-100855-d6864421](outputs/experiment-history/run-index.md#mnist-20261023-100855-d6864421).

**Draft notes:** The legacy snapshot does not include activation/forward code. Do not infer it from a later restored version.

### Change 60 — mnist — 2026-10-23

_Legacy mlp (module shapes saved; forward source not captured)_

**Change:** Use a legacy MLP. The saved configuration contains the exact module shapes. The record does not include the activation or forward source.

- Optimizer: not recorded.
- Learning rate: 0.05.
- Weight decay: 0.0001.
- Batch size: 256.
- Epochs: 500.
- Seed: 42.
- Augmentation: True.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) The longer augmented runs were finally passing the earlier MNIST results. I checked that visible recipe again before moving on.

**Expected:** (Retrospective estimate) The visible recipe could reach a similar validation level again. Missing activation and forward source limit the comparison.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261023-105208-872d486e | 100% | 99.33% | 99.23% | 17.30 minutes | 317 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.7405 → 0.01587 | 0.01001 (357) | 0.207 → 0.02676 | 0.02371 (481) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [mnist-20261023-105208-872d486e](outputs/experiment-history/run-index.md#mnist-20261023-105208-872d486e).

**Draft notes:** The legacy snapshot does not include activation/forward code. Do not infer it from a later restored version.

### Change 61 — mnist — 2026-10-23

_Legacy mlp (module shapes saved; forward source not captured)_

**Change:** Use a legacy MLP. The saved configuration contains the exact module shapes. The record does not include the activation or forward source.

- Optimizer: not recorded.
- Learning rate: 0.05.
- Weight decay: 0.0001.
- Batch size: 128.
- Epochs: 500.
- Seed: 42.
- Augmentation: True.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I tried a smaller batch with augmentation to get more updates per epoch and test the effect of noisier gradients.

**Expected:** (Retrospective estimate) Smaller batches could change generalization through more frequent updates, at the cost of longer epochs.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261023-122327-464bd648 | — | — | — | — | — | 50 completed epoch rows; no final score |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.724 → 0.04265 | 0.04129 (49) | 0.2221 → 0.03736 | 0.03457 (37) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup. No final score was saved, so the full-run expectation remains untested.

**Runs:** [mnist-20261023-122327-464bd648](outputs/experiment-history/run-index.md#mnist-20261023-122327-464bd648).

**Draft notes:** The legacy snapshot does not include activation/forward code. Do not infer it from a later restored version.

### Change 62 — mnist — 2026-10-23

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 128.
- Hidden size: 128.
- Depth: 7.
- Recorded operations: relu.
- Optimizer: SGD.
- Learning rate: 0.05.
- Weight decay: 1e-05.
- Batch size: 128.
- Epochs: 500.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I wanted a compact, fully recorded comparison after the legacy runs. I also checked the simpler path without augmentation.

**Expected:** (Retrospective estimate) A small flat model could remain competitive on MNIST. Without augmentation, it might fit the training examples more easily.

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

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup. No final score was saved, so the full-run expectation remains untested.

**Runs:** [mnist-20261023-123641-6d74404c](outputs/experiment-history/run-index.md#mnist-20261023-123641-6d74404c), [mnist-20261023-134613-19c0d4f0](outputs/experiment-history/run-index.md#mnist-20261023-134613-19c0d4f0).

### Change 63 — mnist — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 128.
- Hidden size: 128.
- Depth: 7.
- Recorded operations: relu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 500.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I wanted to see whether AdamW could train the compact model more effectively than the earlier SGD recipes.

**Expected:** (Retrospective estimate) AdamW could reach a useful fit faster, although adaptive updates alone might not improve held-out accuracy.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261024-181654-70e55ef9 | 100% | 99% | 98.62% | 13.04 minutes | 372 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.3796 → 2.559e-09 | 8.389e-11 (215) | 0.1722 → 0.108 | 0.06154 (101) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [mnist-20261024-181654-70e55ef9](outputs/experiment-history/run-index.md#mnist-20261024-181654-70e55ef9).

### Change 64 — mnist — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 256.
- Hidden size: 512.
- Depth: 50.
- Recorded operations: silu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 500.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I tried a much deeper, wider model to test whether extra capacity could improve the learned representation.

**Expected:** (Retrospective estimate) More capacity could learn a richer representation. It could also add training cost or make optimization harder.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261024-191704-b84eb60a | — | — | — | — | — | 2 completed epoch rows; no final score |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.6104 → 0.203 | 0.203 (2) | 0.283 → 0.1754 | 0.1754 (2) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup. No final score was saved, so the full-run expectation remains untested.

**Runs:** [mnist-20261024-191704-b84eb60a](outputs/experiment-history/run-index.md#mnist-20261024-191704-b84eb60a).

### Change 65 — mnist — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 64.
- Hidden size: 512.
- Depth: 3.
- Recorded operations: silu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 500.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I then tried the opposite capacity choice: a smaller residual stream and fewer blocks.

**Expected:** (Retrospective estimate) A smaller model could train faster and leave less room for memorization. Too narrow a stream could also lose useful signal.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261024-194929-07c1eda8 | — | — | — | — | — | 14 completed epoch rows; no final score |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.3918 → 0.03621 | 0.03621 (14) | 0.2342 → 0.08784 | 0.07267 (9) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup. No final score was saved, so the full-run expectation remains untested.

**Runs:** [mnist-20261024-194929-07c1eda8](outputs/experiment-history/run-index.md#mnist-20261024-194929-07c1eda8).

### Change 66 — mnist — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 128.
- Hidden size: 512.
- Depth: 50.
- Recorded operations: silu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 128.
- Epochs: 500.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I tested a deep but narrower model with smaller batches. This combined capacity and optimization changes.

**Expected:** (Retrospective estimate) Extra depth could help a compact stream process its features. The combined changes might also make optimization harder.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261024-195929-3a9805da | — | — | — | — | — | 2 completed epoch rows; no final score |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.4511 → 0.2075 | 0.2075 (2) | 0.2483 → 0.1645 | 0.1645 (2) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup. No final score was saved, so the full-run expectation remains untested.

**Runs:** [mnist-20261024-195929-3a9805da](outputs/experiment-history/run-index.md#mnist-20261024-195929-3a9805da).

### Change 67 — mnist — date not recorded

_500-epoch double-descent experiment_

**Change:** The journal records these settings.

- depth=6.
- width=256.
- hidden=512.
- batch=256.
- learning_rate=0.05.
- epochs=500.
- weight_decay=0.0001.

**Why:** “Today in class we discussed double descent. So, I tried training MORE with what worked.”

**Expected:** (Retrospective estimate) A longer run could find a better generalizing checkpoint after the earlier plateau.

**Result:**

| Measure | Result |
|---|---|
| Test accuracy (journal) | 98.32% |
| Training time | Not recorded |
| Run ID | Not recorded |

“And it worked!” The corresponding run and loss curve are not recoverable, so the claim cannot be quantified here.

### Change 68 — mnist — date not recorded

_5000-epoch experiment_

**Change:** The journal says training was extended to 5000 epochs. The exact saved setup is not linked.

**Why:** “And it worked! I tried with 5000 too but that didn't work.”

**Expected:** (Retrospective estimate) Training much longer might find another improvement, but could also worsen loss or memorization.

**Result:**

| Measure | Result |
|---|---|
| Journal result | “Its performance was incredibly variable with loss so I decided to try checking whatever epoch was the winner.” |

A score, training time, and loss curve for this claimed run are not recoverable. The journal records a qualitative result only.

### Change 69 — mnist — date not recorded

_GELU experiment before CIFAR-10_

**Change:** The journal says all activations changed to GELU. No exact MNIST run ID is linked.

**Why:** (Retrospective reconstruction) I wanted to test a smooth activation before leaving MNIST.

**Expected:** (Retrospective estimate) GELU could change how smoothly signals and gradients pass through the model. I expected a modest effect rather than a guaranteed gain.

**Result:**

| Measure | Result |
|---|---|
| Journal result | “Then I tried setting all of the activations to GELU. Slight regression so I am finally moving on to CIFAR.” |

A score, training time, and loss curve for this claimed run are not recoverable. The journal records a qualitative result only.

## M4 — Rolling-window transfer

### Change 70 — mnist — 2026-10-27

_Hierarchical 2×2 rolling-window mlp with padded full-image coverage_

**Change:** Use a hierarchical 2×2 rolling-window MLP with padded full-image coverage.

- Width: 512.
- Hidden size: 1024.
- Depth: 2.
- Recorded operations: gelu, pad.
- Window MLP hidden/output: 128/64.
- Spatial sizes: 28 → 27 → 14 → 7 → 4 → 2 → 1.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 500.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) The window approach helped on color images. I tried the same general idea on MNIST to see whether it transferred.

**Expected:** (Retrospective estimate) Local windows could capture useful digit structure, but early compression might lose detail that a flat model keeps.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261027-105310-e7eed90e | — | — | — | — | — | 72 completed epoch rows; no final score |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.81 → 4.343e-06 | 4.133e-06 (70) | 0.2037 → 0.08058 | 0.05907 (11) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup. No final score was saved, so the full-run expectation remains untested.

**Runs:** [mnist-20261027-105310-e7eed90e](outputs/experiment-history/run-index.md#mnist-20261027-105310-e7eed90e).

### Change 71 — mnist — 2026-10-27

_Hierarchical 2×2 rolling-window mlp with padded full-image coverage_

**Change:** Use a hierarchical 2×2 rolling-window MLP with padded full-image coverage.

- Width: 512.
- Hidden size: 1024.
- Depth: 2.
- Recorded operations: gelu, pad.
- Window MLP hidden/output: 128/64.
- Spatial sizes: 28 → 27 → 14 → 7 → 4 → 2 → 1.
- Optimizer: Muon.
- Learning rate: 0.03.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 500.
- Seed: 42.
- Augmentation: True.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I combined the window model with Muon and augmentation to test optimization and generalization together.

**Expected:** (Retrospective estimate) Muon could improve updates to the shared linear weights. Augmentation could reduce dependence on fixed digit positions.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261027-111813-3f4dc443 | — | — | — | — | — | 1 completed epoch rows; no final score |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.7939 → 0.7939 | 0.7939 (1) | 0.1913 → 0.1913 | 0.1913 (1) |

The saved losses show learning behavior, but they do not isolate why this setup performed differently. No final score was saved, so the full-run expectation remains untested.

**Runs:** [mnist-20261027-111813-3f4dc443](outputs/experiment-history/run-index.md#mnist-20261027-111813-3f4dc443).

### Change 72 — mnist — 2026-10-27

_Hierarchical 2×2 rolling-window mlp with padded full-image coverage_

**Change:** Use a hierarchical 2×2 rolling-window MLP with padded full-image coverage.

- Width: 512.
- Hidden size: 1024.
- Depth: 2.
- Recorded operations: gelu, pad.
- Window MLP hidden/output: 128/64.
- Spatial sizes: 28 → 27 → 14 → 7 → 4 → 2 → 1.
- Optimizer: Muon.
- Learning rate: 0.03.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 20.
- Seed: 42.
- Augmentation: True.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I shortened the budget so I could judge this expensive MNIST recipe without another long experiment.

**Expected:** (Retrospective estimate) A shorter run could reveal whether this window recipe offered a useful validation gain without spending the full long-training budget.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| mnist-20261027-111917-d3e67d1d | 96.2463% | 95.7333% | 95.73% | 8.33 minutes | 9 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 0.7939 → 2.304 | 0.2109 (8) | 0.1913 → 2.303 | 0.1592 (4) |

Both losses ended above their starting values. Longer training did not keep improving this recipe.

**Runs:** [mnist-20261027-111917-d3e67d1d](outputs/experiment-history/run-index.md#mnist-20261027-111917-d3e67d1d).

## C1 — Flat MLP exploration

### Change 73 — cifar10 — 2026-10-23

_Transfer MNIST configuration to CIFAR-10_

**Change:** Use a legacy MLP. The saved configuration contains the exact module shapes. The record does not include the activation or forward source.

- Optimizer: not recorded.
- Learning rate: 0.05.
- Weight decay: 0.0001.
- Batch size: 256.
- Epochs: 30.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) MNIST was close to solved. I used its recipe as a starting point for CIFAR-10 to see what broke.

**Expected:** (Retrospective estimate) A reasonable MNIST model could establish a CIFAR-10 reference, but the more complex images would probably expose a larger generalization gap.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261023-112130-b25190a4 | 99.99% | 56.4% | 54.59% | 0.50 minutes | 28 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.68 → 0.00421 | 0.00421 (30) | 1.559 → 2.498 | 1.312 (5) |

Journal interpretation:

“Crazy stuff. Instantly pulls out 99.99% train accuracy? 54% test? This indicates to me that it has become so difficult to learn generalizable patterns that it learned what it could then instantly memorized.”

The saved validation score is close to the adjacent baseline, while training accuracy is much higher. This supports the overfitting interpretation, but the legacy model source is incomplete.

**Runs:** [cifar10-20261023-112130-b25190a4](outputs/experiment-history/run-index.md#cifar10-20261023-112130-b25190a4).

**Draft notes:** The legacy snapshot does not include activation/forward code. Do not infer it from a later restored version. Journal says layers=6 and 500 epochs. The snapshot records seven blocks and 30 epochs. Use the saved settings for this run.

### Change 74 — cifar10 — 2026-10-23

_CIFAR activation comparison (legacy metadata)_

**Change:** Use a legacy MLP. The saved configuration contains the exact module shapes. The record does not include the activation or forward source.

- Optimizer: not recorded.
- Learning rate: 0.05.
- Weight decay: 0.0001.
- Batch size: 256.
- Epochs: 30.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) CIFAR training accuracy was high while test accuracy was much lower. I tried an activation change before adding more capacity.

**Expected:** “Let's try tanh, not expecting much. At this size of network it should be pretty much interchangable.”

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261023-112407-9711758e | 91.51% | 52.84% | 52% | 0.50 minutes | 25 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.801 → 0.309 | 0.2515 (29) | 1.673 → 1.8 | 1.402 (9) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261023-112407-9711758e](outputs/experiment-history/run-index.md#cifar10-20261023-112407-9711758e).

**Draft notes:** The legacy snapshot does not include activation/forward code. Do not infer it from a later restored version.

### Change 75 — cifar10 — 2026-10-23

_Smaller CIFAR model and batch_

**Change:** Use a legacy MLP. The saved configuration contains the exact module shapes. The record does not include the activation or forward source.

- Optimizer: not recorded.
- Learning rate: 0.05.
- Weight decay: 0.0001.
- Batch size: 128.
- Epochs: 30.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** “I applied a shrink to the residual stream to 64. Didn't do much. Did that + shrank batch size:”

**Expected:** (Retrospective estimate) Less capacity and smaller batches could reduce memorization. An overly narrow stream could also discard useful class information.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261023-122642-1f758bb6 | 83.72% | 52.28% | 51.78% | 0.80 minutes | 23 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.703 → 0.3295 | 0.3295 (30) | 1.613 → 2.105 | 1.373 (10) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261023-122642-1f758bb6](outputs/experiment-history/run-index.md#cifar10-20261023-122642-1f758bb6).

**Draft notes:** The legacy snapshot does not include activation/forward code. Do not infer it from a later restored version. Journal says residual width 64. The saved module shapes show width=128 and hidden=128.

### Change 76 — cifar10 — 2026-10-23

_Record complete settings and inspect class accuracy_

**Change:** Use a flat residual MLP.

- Width: 128.
- Hidden size: 128.
- Depth: 7.
- Recorded operations: relu.
- Optimizer: SGD.
- Learning rate: 0.05.
- Weight decay: 1e-05.
- Batch size: 128.
- Epochs: 30.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** “I think we need to understand which classes its getting right and wrong.”

**Expected:** (Retrospective estimate) Class scores could show whether the problem came from a few classes or a broader weakness in the representation.

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

Journal interpretation:

“Cat is incredibly far off but we haven't learned much here. It's learning something about all of these”

The saved validation score is close to the adjacent baseline, while training accuracy is much higher. This supports the overfitting interpretation, but the legacy model source is incomplete.

**Runs:** [cifar10-20261023-123652-808ffc8f](outputs/experiment-history/run-index.md#cifar10-20261023-123652-808ffc8f), [cifar10-20261023-124537-020df882](outputs/experiment-history/run-index.md#cifar10-20261023-124537-020df882), [cifar10-20261023-125414-def90612](outputs/experiment-history/run-index.md#cifar10-20261023-125414-def90612), [cifar10-20261023-134453-1fe1f1ca](outputs/experiment-history/run-index.md#cifar10-20261023-134453-1fe1f1ca), [cifar10-20261023-134526-bd3594de](outputs/experiment-history/run-index.md#cifar10-20261023-134526-bd3594de).

### Change 77 — cifar10 — 2026-10-23

_Reduce CIFAR training to 10 epochs_

**Change:** Use a flat residual MLP.

- Width: 128.
- Hidden size: 128.
- Depth: 7.
- Recorded operations: relu.
- Optimizer: SGD.
- Learning rate: 0.05.
- Weight decay: 1e-05.
- Batch size: 128.
- Epochs: 10.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** “I am changing the runs to use less epochs since performance is worsening on average after 10%. Likely the point at which it starts memorizing given thats where loss diverges.”

**Expected:** (Retrospective estimate) An earlier cutoff could avoid the later validation-loss rise and make comparisons faster.

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

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [cifar10-20261023-134657-9294e112](outputs/experiment-history/run-index.md#cifar10-20261023-134657-9294e112), [cifar10-20261023-134752-02729c78](outputs/experiment-history/run-index.md#cifar10-20261023-134752-02729c78).

### Change 78 — cifar10 — 2026-10-23

_Journal-only ReLU comparison_

**Change:** The journal lists a ReLU trial after tanh. The printed results are identical, and no distinct run ID is linked. A separate result is unconfirmed.

**Why:** “Worse. Reverting. Let's try some more.”

**Expected:** (Retrospective estimate) ReLU might behave differently from tanh without changing the visible model size. I expected a small effect rather than a large jump.

**Result:**

| Measure | Result (journal) |
|---|---:|
| Train accuracy | 91.51% |
| Validation accuracy | 52.84% |
| Test accuracy | 52.00% |
| Selected epoch | 25 |
| Training time | 0.5 minutes |

The printed metrics duplicate the preceding result, and no distinct run ID is linked. A separate ReLU outcome and its curve are not confirmed.

**Draft notes:** No separate saved run was securely matched. Do not invent one.

### Change 79 — cifar10 — 2026-10-24

_Switch CIFAR from SGD to AdamW_

**Change:** Use a flat residual MLP.

- Width: 128.
- Hidden size: 128.
- Depth: 7.
- Recorded operations: relu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 128.
- Epochs: 10.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** “I decided to switch from SGD with weight decay to ADAMW. I saw no immediate ways forward from the previous and learned more about how ADAMW works.”

**Expected:** “We will hopefully see some really nice improvements to accuracy and keep our weight decay to avoid overfitting. Turning down learning rate to match the lowerer needs of ADAMW as well.”

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-151753-14698b08 | 68.8822% | 54.36% | 53.32% | 0.45 minutes | 8 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.677 → 0.8629 | 0.8629 (10) | 1.534 → 1.38 | 1.32 (5) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [cifar10-20261024-151753-14698b08](outputs/experiment-history/run-index.md#cifar10-20261024-151753-14698b08).

### Change 80 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 128.
- Hidden size: 128.
- Depth: 7.
- Recorded operations: relu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 5e-05.
- Batch size: 128.
- Epochs: 10.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I increased weight decay to see whether stronger regularization could control memorization under AdamW.

**Expected:** (Retrospective estimate) Stronger decay could reduce the train/validation gap, but excessive decay could also slow useful learning.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-174927-b3358428 | 71.54% | 55.1% | 54.73% | 0.44 minutes | 9 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.674 → 0.8652 | 0.8652 (10) | 1.515 → 1.375 | 1.309 (9) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [cifar10-20261024-174927-b3358428](outputs/experiment-history/run-index.md#cifar10-20261024-174927-b3358428).

### Change 81 — cifar10 — 2026-10-24

_CIFAR data augmentation_

**Change:** Use a flat residual MLP.

- Width: 128.
- Hidden size: 128.
- Depth: 7.
- Recorded operations: relu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 128.
- Epochs: 10.
- Seed: 42.
- Augmentation: True.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** “No magic bullet of AdamW will work it seems. Let's try data augmentaton that we learned in class.”

**Expected:** (Retrospective estimate) The model would need to recognize transformed objects. That could improve held-out performance even if training accuracy fell.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-180552-79892e5e | 58.1622% | 55.18% | 54.14% | 0.80 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.796 → 1.33 | 1.33 (10) | 1.581 → 1.262 | 1.262 (10) |

The specific setup or measurement is not recoverable from the preserved evidence.

Journal interpretation: “Well, Overfitting decreased. Performance is bad.”

**Runs:** [cifar10-20261024-180552-79892e5e](outputs/experiment-history/run-index.md#cifar10-20261024-180552-79892e5e).

### Change 82 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 128.
- Hidden size: 128.
- Depth: 7.
- Recorded operations: relu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 10.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I checked a larger batch without augmentation as another control for the optimizer recipe.

**Expected:** (Retrospective estimate) Larger batches could make updates less noisy, while removing augmentation might make memorization easier.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-190549-b5f1f96d | 69.8022% | 54.7% | 54.17% | 0.25 minutes | 9 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.7 → 0.8778 | 0.8778 (10) | 1.516 → 1.368 | 1.309 (7) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [cifar10-20261024-190549-b5f1f96d](outputs/experiment-history/run-index.md#cifar10-20261024-190549-b5f1f96d).

### Change 83 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 128.
- Hidden size: 512.
- Depth: 7.
- Recorded operations: silu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 10.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) The simpler optimizer changes stalled. I widened the branches and tried SiLU to test a richer transformation.

**Expected:** (Retrospective estimate) Wider branches could learn more useful features, and SiLU could change optimization. More capacity could also widen the generalization gap.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-190721-622fa5d9 | 71.6867% | 55.64% | 55.43% | 0.25 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.681 → 0.9185 | 0.9185 (10) | 1.541 → 1.308 | 1.272 (8) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [cifar10-20261024-190721-622fa5d9](outputs/experiment-history/run-index.md#cifar10-20261024-190721-622fa5d9).

### Change 84 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 128.
- Hidden size: 512.
- Depth: 8.
- Recorded operations: silu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 10.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I added a block to see whether the wider branches benefited from another transformation.

**Expected:** (Retrospective estimate) Another block could improve the representation if the wider branches still needed more processing.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-190829-4408179b | 68.9733% | 55.9% | 55.34% | 0.27 minutes | 9 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.687 → 0.9351 | 0.9351 (10) | 1.542 → 1.319 | 1.282 (9) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [cifar10-20261024-190829-4408179b](outputs/experiment-history/run-index.md#cifar10-20261024-190829-4408179b).

### Change 85 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 128.
- Hidden size: 512.
- Depth: 20.
- Recorded operations: silu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 10.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I increased depth further to test whether more transformations helped while keeping the residual stream narrow.

**Expected:** (Retrospective estimate) More depth could help with complex features, but only if the extra blocks remained useful and trainable.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-190942-5d5a11ec | 64.2911% | 54.94% | 53.88% | 0.54 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.747 → 1.087 | 1.087 (10) | 1.582 → 1.304 | 1.281 (9) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [cifar10-20261024-190942-5d5a11ec](outputs/experiment-history/run-index.md#cifar10-20261024-190942-5d5a11ec).

### Change 86 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 256.
- Hidden size: 512.
- Depth: 50.
- Recorded operations: silu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 5e-05.
- Batch size: 256.
- Epochs: 10.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I tried a larger stream and many more blocks, with stronger decay to limit memorization.

**Expected:** (Retrospective estimate) The larger model could improve fit. Stronger decay might control overfitting, although the combined changes would not identify one cause.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-191149-9e04c0c6 | 65.2489% | 55.08% | 53.86% | 1.29 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.863 → 1.077 | 1.077 (10) | 1.665 → 1.314 | 1.314 (10) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [cifar10-20261024-191149-9e04c0c6](outputs/experiment-history/run-index.md#cifar10-20261024-191149-9e04c0c6).

### Change 87 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 256.
- Hidden size: 512.
- Depth: 50.
- Recorded operations: silu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 15.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I gave the large model more epochs and relaxed the decay to see whether it needed a less restrictive training recipe.

**Expected:** (Retrospective estimate) More time and less decay could help the large model learn, with a risk of widening the generalization gap.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-191810-0c227281 | 67.3644% | 55.76% | 54.68% | 1.98 minutes | 11 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.863 → 0.8132 | 0.8132 (15) | 1.665 → 1.376 | 1.288 (11) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [cifar10-20261024-191810-0c227281](outputs/experiment-history/run-index.md#cifar10-20261024-191810-0c227281).

### Change 88 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 256.
- Hidden size: 512.
- Depth: 50.
- Recorded operations: silu.
- Optimizer: Muon.
- Learning rate: 0.02.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 15.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I tried Muon because the model mainly uses weight matrices and AdamW had not broken through the accuracy plateau.

**Expected:** (Retrospective estimate) Muon could make more effective matrix updates and reach a better validation score within the same training budget.

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

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261024-192349-50b5b789](outputs/experiment-history/run-index.md#cifar10-20261024-192349-50b5b789), [cifar10-20261024-193237-88e5f013](outputs/experiment-history/run-index.md#cifar10-20261024-193237-88e5f013).

### Change 89 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 256.
- Hidden size: 512.
- Depth: 50.
- Recorded operations: silu.
- Optimizer: Muon.
- Learning rate: 0.03.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 15.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I raised the Muon learning rate to see whether stronger updates could make more progress within the same budget.

**Expected:** (Retrospective estimate) A higher rate could improve progress, but could also make the updates too aggressive.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-193733-de010494 | 82.8289% | 56.26% | 55.18% | 4.70 minutes | 8 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.675 → 0.2738 | 0.2738 (15) | 1.501 → 1.933 | 1.285 (5) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261024-193733-de010494](outputs/experiment-history/run-index.md#cifar10-20261024-193733-de010494).

### Change 90 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 128.
- Hidden size: 512.
- Depth: 50.
- Recorded operations: silu.
- Optimizer: Muon.
- Learning rate: 0.03.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 15.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I reduced the residual stream while keeping the deep model to test the cost of passing less information through it.

**Expected:** (Retrospective estimate) A narrower stream could reduce memorization and cost. It could also bottleneck useful image information.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-194312-d0e4074c | 74.0978% | 56.46% | 55% | 3.72 minutes | 7 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.705 → 0.3582 | 0.3582 (15) | 1.529 → 1.827 | 1.286 (5) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261024-194312-d0e4074c](outputs/experiment-history/run-index.md#cifar10-20261024-194312-d0e4074c).

### Change 91 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 128.
- Hidden size: 512.
- Depth: 3.
- Recorded operations: silu.
- Optimizer: Muon.
- Learning rate: 0.03.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 15.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I then reduced depth sharply to see whether the deep stack was adding useful work or just cost.

**Expected:** (Retrospective estimate) Fewer blocks could reduce cost without losing much accuracy if the deep stack was contributing little.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-194834-440d42d1 | 76.2667% | 55.92% | 54.58% | 0.37 minutes | 7 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.625 → 0.2762 | 0.2762 (15) | 1.47 → 2.183 | 1.309 (4) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261024-194834-440d42d1](outputs/experiment-history/run-index.md#cifar10-20261024-194834-440d42d1).

### Change 92 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 64.
- Hidden size: 512.
- Depth: 3.
- Recorded operations: silu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 10.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I combined a very narrow model with AdamW to check another compact recipe.

**Expected:** (Retrospective estimate) A compact AdamW model could provide a cheaper comparison, but changing optimizer and capacity together would limit causal conclusions.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-194948-28bf206f | 63.3222% | 54.36% | 53.7% | 0.16 minutes | 9 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.698 → 1.067 | 1.067 (10) | 1.54 → 1.326 | 1.302 (8) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [cifar10-20261024-194948-28bf206f](outputs/experiment-history/run-index.md#cifar10-20261024-194948-28bf206f).

### Change 93 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 64.
- Hidden size: 512.
- Depth: 50.
- Recorded operations: silu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 10.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I restored a deep stack while keeping the narrow stream to see whether depth could compensate for that bottleneck.

**Expected:** (Retrospective estimate) Depth could make better use of the narrow representation, provided that the stream still carried enough information.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-195018-9050bcbd | 56.7044% | 50.7% | 51.46% | 1.19 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.814 → 1.261 | 1.261 (10) | 1.678 → 1.373 | 1.373 (10) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [cifar10-20261024-195018-9050bcbd](outputs/experiment-history/run-index.md#cifar10-20261024-195018-9050bcbd).

### Change 94 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 96.
- Hidden size: 512.
- Depth: 50.
- Recorded operations: silu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 10.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I widened the stream slightly to see whether the narrow model was discarding useful information.

**Expected:** (Retrospective estimate) A slightly wider stream could preserve more useful information without the full cost of the larger models.

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

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [cifar10-20261024-195211-13ca66a0](outputs/experiment-history/run-index.md#cifar10-20261024-195211-13ca66a0), [cifar10-20261024-195449-e6c9f45c](outputs/experiment-history/run-index.md#cifar10-20261024-195449-e6c9f45c).

### Change 95 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 128.
- Hidden size: 512.
- Depth: 50.
- Recorded operations: silu.
- Optimizer: AdamW.
- Learning rate: 0.005.
- Weight decay: 1e-05.
- Batch size: 256.
- Epochs: 10.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I widened the stream again and raised the AdamW learning rate. This tested a more aggressive recipe rather than one isolated change.

**Expected:** (Retrospective estimate) A wider stream and larger step could improve progress, but the aggressive rate could destabilize training.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-195653-50a070d4 | 60.0933% | 52.46% | 52.95% | 1.22 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.974 → 1.209 | 1.209 (10) | 1.793 → 1.342 | 1.342 (10) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [cifar10-20261024-195653-50a070d4](outputs/experiment-history/run-index.md#cifar10-20261024-195653-50a070d4).

### Change 96 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 128.
- Hidden size: 512.
- Depth: 50.
- Recorded operations: silu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 128.
- Epochs: 10.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I returned to a lower rate and smaller batch after testing the aggressive update setting.

**Expected:** (Retrospective estimate) The lower rate could reduce overshoot. Smaller batches could add useful variation but also increase noise and runtime.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-200016-2a3ca5fb | 60.3756% | 52.84% | 52.73% | 2.37 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.816 → 1.17 | 1.17 (10) | 1.65 → 1.355 | 1.338 (8) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [cifar10-20261024-200016-2a3ca5fb](outputs/experiment-history/run-index.md#cifar10-20261024-200016-2a3ca5fb).

### Change 97 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP.

- Width: 256.
- Hidden size: 512.
- Depth: 50.
- Recorded operations: silu.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 1e-05.
- Batch size: 128.
- Epochs: 10.
- Seed: 42.
- Augmentation: False.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I widened the stream with the smaller batch to check whether more capacity helped that recipe.

**Expected:** (Retrospective estimate) The wider stream could preserve more information, although it could also make training fit improve faster than validation.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
|---|---:|---:|---:|---:|---:|---|
| cifar10-20261024-200407-caebc9e6 | 65.4644% | 54.84% | 53.61% | 2.55 minutes | 10 | — |

| Run | Train loss start → end | Minimum train loss (epoch) | Validation loss start → end | Minimum validation loss (epoch) |
|---|---:|---:|---:|---:|
| Same run | 1.824 → 1.079 | 1.079 (10) | 1.66 → 1.296 | 1.296 (10) |

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [cifar10-20261024-200407-caebc9e6](outputs/experiment-history/run-index.md#cifar10-20261024-200407-caebc9e6).

## C2 — Happy Accident and controlled reruns

### Change 98 — cifar10 — 2026-10-24

_Happy Accident: restored configuration_

**Change:** Use a flat residual MLP.

- Width: 256.
- Hidden size: 512.
- Depth: 7.
- Recorded operations: gelu.
- Optimizer: SGD.
- Learning rate: 0.05.
- Weight decay: 0.0001.
- Batch size: 256.
- Epochs: 30.
- Seed: 42.
- Augmentation: True.
- The evidence snapshot contains the other optimizer, split, and augmentation settings.

**Why:** “So I wasn't liking the results I was getting so I added a feature to roll back to previous configurations. But, apparently it was bugged.”

**Expected:** (Retrospective estimate) Restoring a configuration would normally reproduce its behavior. An unexpected gain would require checking the actual restored settings.

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

Both losses ended below their starting values. The recipe learned, but the final losses alone do not establish an improvement over another setup.

**Runs:** [cifar10-20261024-210758-6cf0d0b4](outputs/experiment-history/run-index.md#cifar10-20261024-210758-6cf0d0b4), [cifar10-20261024-214516-c1cc95e3](outputs/experiment-history/run-index.md#cifar10-20261024-214516-c1cc95e3), [cifar10-20261024-215408-cb0ff45d](outputs/experiment-history/run-index.md#cifar10-20261024-215408-cb0ff45d), [cifar10-20261024-215556-2b1c3233](outputs/experiment-history/run-index.md#cifar10-20261024-215556-2b1c3233), [cifar10-20261024-220508-08117fee](outputs/experiment-history/run-index.md#cifar10-20261024-220508-08117fee).

### Change 99 — cifar10 — 2026-10-24

_Isolate width and hidden size_

**Change:** Use a flat residual MLP. Width=128, hidden=128, depth=7. Recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=30, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** “So... I need to understand what happened here. I am going to look through each change one by one and theorize then test.”

**Expected:** “I imagine that a residual stream of 128 with a hidden width of 128 was bottlenecking the model.”

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-212021-44f3ab35` | 63.9867% | 57.56% | 56.68% | 1.35 minutes | 29 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.837 → 1.17 | 1.17 (30) | 1.647 → 1.214 | 1.208 (29) |

Journal interpretation: “Looks like we were a little bit right, but it only made a small difference. Reverting and moving to the next.” The smaller configuration still learned, but did not establish a gain from shrinking the model. The two widths changed together.

**Runs:** [cifar10-20261024-212021-44f3ab35](outputs/experiment-history/run-index.md#cifar10-20261024-212021-44f3ab35).

### Change 100 — cifar10 — 2026-10-24

_Activation comparison: saved source uses ReLU_

**Change:** Use a flat residual MLP. Width=256, hidden=512, depth=7. Recorded operations relu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=30, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I wanted to test the activation explanation for the restored model’s gain. The saved source later showed that my comparison label was reversed.

**Expected:** “I expect GELU is better because by definition it was an iteration upon ReLU.”

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-212806-223e71d6` | 71.7667% | 60.04% | 58.51% | 1.43 minutes | 29 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.822 → 1.035 | 1.035 (30) | 1.585 → 1.188 | 1.168 (29) |

Journal interpretation: “And so it was.” The saved source uses ReLU in the block and head. This result cannot validate the journal’s stated GELU direction.

**Runs:** [cifar10-20261024-212806-223e71d6](outputs/experiment-history/run-index.md#cifar10-20261024-212806-223e71d6).

**Draft notes:** The journal says ReLU → GELU. The saved block and head use ReLU. The original expectation remains quoted, but this saved run does not test the direction stated in the journal.

### Change 101 — cifar10 — 2026-10-24

_Batch 128 with augmentation accidentally disabled_

**Change:** Use a flat residual MLP. Width=256, hidden=512, depth=7. Recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=128, epochs=30, seed=42. Augmentation=False. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I tested a smaller batch to see whether noisier updates helped generalization. Augmentation also changed, so this was not an isolated comparison.

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

Journal interpretation: “Worse, not entirely unexpected.” Training accuracy was near its ceiling while validation stayed much lower. Batch size and augmentation both changed, so this does not isolate the batch-size effect.

**Runs:** [cifar10-20261024-213729-7b6a5d2b](outputs/experiment-history/run-index.md#cifar10-20261024-213729-7b6a5d2b), [cifar10-20261024-222752-audit-batch-128](outputs/experiment-history/run-index.md#cifar10-20261024-222752-audit-batch-128).

**Draft notes:** Batch size and augmentation both changed. This run does not isolate batch size.

### Change 102 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP. Width=256, hidden=512, depth=7. Recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=30, seed=42. Augmentation=False. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I needed an augmentation-off reference so I could separate the model’s training fit from the effect of transformed images.

**Expected:** (Retrospective estimate) Without augmentation, the model could fit the training set more easily. I expected that to expose any remaining generalization problem.

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

Without augmentation, the complete runs nearly fit the training set while validation loss rose. This made the generalization gap clear.

**Draft notes:** The partial run has no final score. This does not prove a crash.

**Runs:** [cifar10-20261024-213937-1058bdc7](outputs/experiment-history/run-index.md#cifar10-20261024-213937-1058bdc7), [cifar10-20261024-221742-c8c03980](outputs/experiment-history/run-index.md#cifar10-20261024-221742-c8c03980), [cifar10-20261024-222752-audit-baseline](outputs/experiment-history/run-index.md#cifar10-20261024-222752-audit-baseline), [cifar10-20261024-223819-2016e143](outputs/experiment-history/run-index.md#cifar10-20261024-223819-2016e143).

### Change 103 — cifar10 — 2026-10-24

_Reduce CIFAR weight decay to 0.00001_

**Change:** Use a flat residual MLP. Width=256, hidden=512, depth=7. Recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=1e-05, batch=256, epochs=30, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** “Weight decay is a big deal, I've seen it absolutely tank performance in other trials. I've also seen it save performance during MNIST where we were a hundred percent accuracy.”

**Expected:** “This is what I had expected to make the biggest difference.”

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-215848-059027aa` | 74.9% | 60.38% | 59.07% | 1.46 minutes | 29 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.784 → 0.9737 | 0.9737 (30) | 1.552 → 1.185 | 1.158 (29) |

Journal interpretation: “Incredible. We've been over regularizing. Why not? Even smaller.” The result fits the idea that strong regularization limited useful learning. Other changes in this exploratory sequence limit an isolated causal claim.

**Runs:** [cifar10-20261024-215848-059027aa](outputs/experiment-history/run-index.md#cifar10-20261024-215848-059027aa).

### Change 104 — cifar10 — 2026-10-24

_Zero weight decay_

**Change:** Use a flat residual MLP. Width=256, hidden=512, depth=7. Recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=0, batch=256, epochs=30, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** “Incredible. We've been over regularizing. Why not? Even smaller.”

**Expected:** (Retrospective estimate) Removing the remaining decay might improve fit, but could also make memorization easier.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-220725-3ec48d0c` | 74.9311% | 60.36% | 59.11% | 1.41 minutes | 29 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.784 → 0.9729 | 0.9729 (30) | 1.552 → 1.185 | 1.158 (29) |

Journal interpretation: “It rounded to zero after I added one more. Going to leave regularization on.” The zero-decay and low-decay scores stayed close. The quote describes leaving regularization on, while this saved configuration records zero decay.

**Runs:** [cifar10-20261024-220725-3ec48d0c](outputs/experiment-history/run-index.md#cifar10-20261024-220725-3ec48d0c).

### Change 105 — cifar10 — 2026-10-24

_Augmentation-off audit: relu_

**Change:** Use a flat residual MLP. Width=256, hidden=512, depth=7. Recorded operations relu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=30, seed=42. Augmentation=False. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** “Forgot to turn augmentation off, had to rerun all those experiments”

**Expected:** (Retrospective estimate) The activation comparison could look different once augmentation stopped confounding the result.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-222752-audit-relu` | 99.9844% | 55.52% | 54.81% | 0.53 minutes | 30 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.718 → 0.01115 | 0.01115 (30) | 1.578 → 2.306 | 1.327 (5) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261024-222752-audit-relu](outputs/experiment-history/run-index.md#cifar10-20261024-222752-audit-relu).

**Draft notes:** This rerun turns augmentation off. The later result table does not give the historical prediction.

### Change 106 — cifar10 — 2026-10-24

_Augmentation-off audit: weight-decay-1e-5_

**Change:** Use a flat residual MLP. Width=256, hidden=512, depth=7. Recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=1e-05, batch=256, epochs=30, seed=42. Augmentation=False. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** “Forgot to turn augmentation off, had to rerun all those experiments”

**Expected:** (Retrospective estimate) Lower decay could improve useful learning, but an augmentation-off model might also memorize the training set more strongly.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-222752-audit-weight-decay-1e-5` | 99.9978% | 55.3% | 54.75% | 0.51 minutes | 29 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.68 → 0.002158 | 0.002158 (30) | 1.559 → 2.664 | 1.313 (5) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261024-222752-audit-weight-decay-1e-5](outputs/experiment-history/run-index.md#cifar10-20261024-222752-audit-weight-decay-1e-5).

**Draft notes:** This rerun turns augmentation off. The later result table does not give the historical prediction.

### Change 107 — cifar10 — 2026-10-24

_Augmentation-off audit: weight-decay-zero_

**Change:** Use a flat residual MLP. Width=256, hidden=512, depth=7. Recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=0, batch=256, epochs=30, seed=42. Augmentation=False. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** “Forgot to turn augmentation off, had to rerun all those experiments”

**Expected:** (Retrospective estimate) Removing decay could improve training fit without necessarily improving validation accuracy.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-222752-audit-weight-decay-zero` | 99.9978% | 55.72% | 54.67% | 0.49 minutes | 29 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.68 → 0.002283 | 0.002283 (30) | 1.559 → 2.644 | 1.313 (5) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261024-222752-audit-weight-decay-zero](outputs/experiment-history/run-index.md#cifar10-20261024-222752-audit-weight-decay-zero).

**Draft notes:** This rerun turns augmentation off. The later result table does not give the historical prediction.

### Change 108 — cifar10 — 2026-10-24

_Augmentation-off audit: width-hidden_

**Change:** Use a flat residual MLP. Width=128, hidden=128, depth=7. Recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=30, seed=42. Augmentation=False. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** “Forgot to turn augmentation off, had to rerun all those experiments”

**Expected:** (Retrospective estimate) The smaller widths could reveal whether the larger model’s capacity was carrying useful information or mostly enabling memorization.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-222752-audit-width-hidden` | 72.8311% | 51.74% | 50.29% | 0.43 minutes | 15 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.736 → 0.3564 | 0.3564 (30) | 1.6 → 2.102 | 1.367 (7) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261024-222752-audit-width-hidden](outputs/experiment-history/run-index.md#cifar10-20261024-222752-audit-width-hidden).

**Draft notes:** This rerun turns augmentation off. The later result table does not give the historical prediction.

## C3 — Optimizers, width and depth

### Change 109 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP. Width=256, hidden=512, depth=7. Recorded operations gelu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=60, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I gave the audited SGD model more epochs to see whether the earlier budget was limiting its validation score.

**Expected:** (Retrospective estimate) More SGD epochs could improve validation if useful learning remained. Otherwise, they might mainly improve training fit.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-234504-6badff50` | 87.4733% | 61.48% | 60.11% | 2.83 minutes | 53 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.784 → 0.6976 | 0.6976 (60) | 1.552 → 1.319 | 1.152 (35) |

Both losses ended below their starting values. The recipe learned, but the endpoint comparison alone does not isolate an improvement over another setup.

**Runs:** [cifar10-20261024-234504-6badff50](outputs/experiment-history/run-index.md#cifar10-20261024-234504-6badff50).

### Change 110 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP. Width=256, hidden=512, depth=7. Recorded operations gelu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I returned to AdamW with the larger model to test whether adaptive updates could improve the longer-training recipe.

**Expected:** (Retrospective estimate) AdamW could reach a better fit within the longer budget, although it would not automatically solve generalization.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-234930-b4aa43ea` | 96.3133% | 61.26% | 59.01% | 3.38 minutes | 43 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.799 → 0.3329 | 0.3329 (60) | 1.552 → 1.892 | 1.181 (19) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261024-234930-b4aa43ea](outputs/experiment-history/run-index.md#cifar10-20261024-234930-b4aa43ea).

### Change 111 — cifar10 — 2026-10-24

_Flat residual mlp_

**Change:** Use a flat residual MLP. Width=256, hidden=512, depth=7. Recorded operations gelu. Optimizer=AdamW, lr=0.005, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I raised the AdamW learning rate to see whether the model could make more progress before it started memorizing.

**Expected:** (Retrospective estimate) The larger rate could speed useful learning, but might also make updates too aggressive.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261024-235432-57963b71` | 93.7622% | 59.64% | 58.11% | 3.31 minutes | 52 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.005 → 0.4989 | 0.4989 (60) | 1.748 → 1.864 | 1.194 (24) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261024-235432-57963b71](outputs/experiment-history/run-index.md#cifar10-20261024-235432-57963b71).

### Change 112 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** Use a flat residual MLP. Width=256, hidden=512, depth=7. Recorded operations gelu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I returned to the lower AdamW rate after testing the more aggressive setting.

**Expected:** (Retrospective estimate) The lower rate could improve control of the updates, at the cost of slower progress.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-000330-5cea5707` | 91.9044% | 61.48% | 59.43% | 3.39 minutes | 32 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.799 → 0.3178 | 0.3178 (60) | 1.552 → 1.843 | 1.178 (15) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261025-000330-5cea5707](outputs/experiment-history/run-index.md#cifar10-20261025-000330-5cea5707).

### Change 113 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** Use a flat residual MLP. Width=256, hidden=512, depth=7. Recorded operations gelu. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I kept exploring AdamW settings around the better rate instead of making the model larger again.

**Expected:** (Retrospective estimate) Optimizer choices around the same architecture could change convergence. I hoped to improve validation without adding more capacity.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-000912-e7c42e1d` | 94.1978% | 61.68% | 59.08% | 3.53 minutes | 49 | — |
| `cifar10-20261025-002050-9941eabc` | 94.1978% | 61.68% | 59.08% | 4.45 minutes | 49 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| `cifar10-20261025-000912-e7c42e1d` | 1.833 → 0.5039 | 0.5039 (60) | 1.582 → 1.526 | 1.152 (19) |
| `cifar10-20261025-002050-9941eabc` | 1.833 → 0.5039 | 0.5039 (60) | 1.582 → 1.526 | 1.152 (19) |

Both losses ended below their starting values. The recipe learned, but the endpoint comparison alone does not isolate an improvement over another setup.

**Runs:** [cifar10-20261025-000912-e7c42e1d](outputs/experiment-history/run-index.md#cifar10-20261025-000912-e7c42e1d), [cifar10-20261025-002050-9941eabc](outputs/experiment-history/run-index.md#cifar10-20261025-002050-9941eabc).

### Change 114 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** Use a flat residual MLP. Width=256, hidden=512, depth=7. Recorded operations silu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=30, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I tried SiLU with SGD to see whether an activation change helped the familiar training recipe.

**Expected:** (Retrospective estimate) SiLU could change signal and gradient behavior without changing the main model dimensions.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-001310-684974cb` | 70.9911% | 60.26% | 58.82% | 1.79 minutes | 29 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.783 → 1.041 | 1.041 (30) | 1.556 → 1.176 | 1.147 (29) |

Both losses ended below their starting values. The recipe learned, but the endpoint comparison alone does not isolate an improvement over another setup.

**Runs:** [cifar10-20261025-001310-684974cb](outputs/experiment-history/run-index.md#cifar10-20261025-001310-684974cb).

### Change 115 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** Use a flat residual MLP. Width=256, hidden=512, depth=7. Recorded operations silu. Optimizer=SGD, lr=0.05, weight_decay=0.0001, batch=256, epochs=50, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I extended the SiLU run to see whether it still had useful learning left at the earlier cutoff.

**Expected:** (Retrospective estimate) More epochs could help if the SiLU model still had useful learning left. Validation selection would keep an earlier checkpoint if necessary.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-001521-3603c62e` | 78.84% | 61.28% | 59.47% | 2.97 minutes | 46 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.783 → 0.8696 | 0.8696 (50) | 1.556 → 1.178 | 1.135 (41) |

Both losses ended below their starting values. The recipe learned, but the endpoint comparison alone does not isolate an improvement over another setup.

**Runs:** [cifar10-20261025-001521-3603c62e](outputs/experiment-history/run-index.md#cifar10-20261025-001521-3603c62e).

### Change 116 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** Use a flat residual MLP. Width=256, hidden=512, depth=7. Recorded operations gelu. Optimizer=Muon, lr=0.02, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I tried Muon because the model mainly uses weight matrices and the AdamW results were still near the same accuracy range.

**Expected:** (Retrospective estimate) Muon could improve the matrix updates and move the validation score beyond the earlier optimizer recipes.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-002715-5a076b1b` | 89.4933% | 61.56% | 59.25% | 6.93 minutes | 31 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.748 → 0.4276 | 0.4276 (60) | 1.511 → 1.658 | 1.152 (18) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261025-002715-5a076b1b](outputs/experiment-history/run-index.md#cifar10-20261025-002715-5a076b1b).

### Change 117 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** Use a flat residual MLP. Width=512, hidden=512, depth=7. Recorded operations gelu. Optimizer=Muon, lr=0.02, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I widened the residual stream to preserve more information between blocks while keeping the branch hidden width fixed.

**Expected:** (Retrospective estimate) A wider stream could carry more useful image information, but it could also increase cost and memorization.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-003949-24bf237f` | 93.8378% | 61.96% | 59.69% | 9.16 minutes | 38 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.763 → 0.4431 | 0.4414 (59) | 1.486 → 1.579 | 1.144 (15) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261025-003949-24bf237f](outputs/experiment-history/run-index.md#cifar10-20261025-003949-24bf237f).

### Change 118 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** Use a flat residual MLP. Width=512, hidden=512, depth=7. Recorded operations gelu. Optimizer=RMSprop, lr=0.001, weight_decay=1e-05, batch=256, epochs=30, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I tried RMSprop as another adaptive optimizer recipe for the wider model.

**Expected:** (Retrospective estimate) RMSprop could control differently scaled gradients more effectively. The optimizer and budget changes would limit an isolated comparison.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-005328-ee80b2f5` | 75.6644% | 60.24% | 57.68% | 2.25 minutes | 27 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.318 → 0.9572 | 0.9572 (30) | 1.786 → 1.228 | 1.165 (26) |

Both losses ended below their starting values. The recipe learned, but the endpoint comparison alone does not isolate an improvement over another setup.

**Runs:** [cifar10-20261025-005328-ee80b2f5](outputs/experiment-history/run-index.md#cifar10-20261025-005328-ee80b2f5).

### Change 119 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** Use a flat residual MLP. Width=512, hidden=512, depth=6. Recorded operations gelu. Optimizer=Muon, lr=0.02, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I reduced depth slightly to check whether every block was useful in the wider model.

**Expected:** (Retrospective estimate) Slightly less depth could remove unnecessary work without losing useful transformations.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-005652-cd458538` | 97.3133% | 62.04% | 60.13% | 8.01 minutes | 57 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.752 → 0.4525 | 0.4525 (60) | 1.509 → 1.602 | 1.167 (16) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261025-005652-cd458538](outputs/experiment-history/run-index.md#cifar10-20261025-005652-cd458538).

### Change 120 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** Use a flat residual MLP. Width=512, hidden=512, depth=3. Recorded operations gelu. Optimizer=Muon, lr=0.02, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I reduced depth further to test whether a shorter model could keep the accuracy while doing less work.

**Expected:** (Retrospective estimate) A much shorter stack could reduce cost, but it might also lose transformations needed for the more complex images.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-115606-bd7c0564` | 95.6933% | 59.12% | 57.55% | 4.25 minutes | 27 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.659 → 0.1991 | 0.1991 (60) | 1.443 → 2.177 | 1.205 (9) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261025-115606-bd7c0564](outputs/experiment-history/run-index.md#cifar10-20261025-115606-bd7c0564).

### Change 121 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** Use a flat residual MLP. Width=512, hidden=512, depth=6. Recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I restored the deeper configuration and raised the Muon rate to test a more aggressive recipe.

**Expected:** (Retrospective estimate) The deeper stack and higher rate could improve progress together. This would test a recipe rather than one independent factor.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-123955-7c8c8c1d` | 96.8067% | 62.26% | 60.15% | 5.91 minutes | 60 | — |
| `cifar10-20261025-210636-d9406711` | 96.8067% | 62.26% | 60.15% | 6.03 minutes | 60 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| `cifar10-20261025-123955-7c8c8c1d` | 1.762 → 0.5186 | 0.5181 (59) | 1.522 → 1.485 | 1.138 (17) |
| `cifar10-20261025-210636-d9406711` | 1.762 → 0.5186 | 0.5181 (59) | 1.522 → 1.485 | 1.138 (17) |

Both losses ended below their starting values. The recipe learned, but the endpoint comparison alone does not isolate an improvement over another setup.

**Runs:** [cifar10-20261025-123955-7c8c8c1d](outputs/experiment-history/run-index.md#cifar10-20261025-123955-7c8c8c1d), [cifar10-20261025-210636-d9406711](outputs/experiment-history/run-index.md#cifar10-20261025-210636-d9406711).

### Change 122 — cifar10 — 2026-10-25

_Flat residual mlp_

**Change:** Use a flat residual MLP. Width=512, hidden=512, depth=6. Recorded operations gelu. Optimizer=Muon, lr=0.01, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I lowered the Muon rate to see whether smaller updates helped the validation score.

**Expected:** (Retrospective estimate) A smaller rate could reduce overly aggressive updates, although the model might need longer to learn.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-124849-bcdd1a30` | 94.88% | 61.9% | 60.26% | 6.06 minutes | 35 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 1.743 → 0.3943 | 0.3943 (60) | 1.469 → 1.694 | 1.161 (12) |

Training loss ended lower while validation loss ended higher. Better training fit was not translating into better held-out loss.

**Runs:** [cifar10-20261025-124849-bcdd1a30](outputs/experiment-history/run-index.md#cifar10-20261025-124849-bcdd1a30).

## C4 — Bilinear gates, pooling and dropout

### Change 123 — cifar10 — 2026-10-25

_Incomplete block construction: block forward returns none_

**Change:** The block construction is incomplete. Its forward method returns None. Width=512, hidden=512, depth=6.

Recorded operations gelu. Optimizer=Muon, lr=0.01, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True.

The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I was rebuilding the block for the gating experiments. The stored version still has an incomplete forward method.

**Expected:** (Retrospective estimate) The intended block needed to return a usable tensor and pass gradients before I could judge its accuracy.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-125710-62ba38ce` | — | — | — | — | — | No final score; 0 completed epoch rows saved; No completed epoch metrics were saved. |
| `cifar10-20261025-125841-0e1749e2` | — | — | — | — | — | No final score; 0 completed epoch rows saved; No completed epoch metrics were saved. |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| `cifar10-20261025-125710-62ba38ce` | — | — | — | — |
| `cifar10-20261025-125841-0e1749e2` | — | — | — | — |

No final measurement is available for this attempt. The saved record does not establish whether the reconstructed expectation was met.

**Runs:** [cifar10-20261025-125710-62ba38ce](outputs/experiment-history/run-index.md#cifar10-20261025-125710-62ba38ce), [cifar10-20261025-125841-0e1749e2](outputs/experiment-history/run-index.md#cifar10-20261025-125841-0e1749e2).

**Draft notes:** The saved block source returns None. The record has no exception receipt. Do not infer the failure message. These records do not prove a crash.

### Change 124 — cifar10 — 2026-10-25

_Incomplete block construction: block forward returns none_

**Change:** The block construction is incomplete. Its forward method returns None. Width=512, hidden=512, depth=6.

Recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True.

The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I kept working on the block construction with another optimizer setting. The saved forward method was still incomplete.

**Expected:** (Retrospective estimate) Completing the block would make a training comparison possible. The saved incomplete method cannot establish an accuracy outcome.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-130442-f759bea4` | — | — | — | — | — | No final score; 0 completed epoch rows saved; No completed epoch metrics were saved. |
| `cifar10-20261025-132544-8610feee` | — | — | — | — | — | No final score; 0 completed epoch rows saved; No completed epoch metrics were saved. |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| `cifar10-20261025-130442-f759bea4` | — | — | — | — |
| `cifar10-20261025-132544-8610feee` | — | — | — | — |

No final measurement is available for this attempt. The saved record does not establish whether the reconstructed expectation was met.

**Runs:** [cifar10-20261025-130442-f759bea4](outputs/experiment-history/run-index.md#cifar10-20261025-130442-f759bea4), [cifar10-20261025-132544-8610feee](outputs/experiment-history/run-index.md#cifar10-20261025-132544-8610feee).

**Draft notes:** The saved block source returns None. The record has no exception receipt. Do not infer the failure message. These records do not prove a crash.

### Change 125 — cifar10 — 2026-10-25

_Bilinear gated blocks_

**Change:** Use bilinear blocks without skip connections. Unused extra Linear layers remain registered. Width=512, hidden=512, depth=6.

Recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True.

The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** “Time to experiment with some gated linear units. While it's directly supported by PyTorch, it's not supported by torch canvas. I could add the functionality, but I think it's a more interesting exploration if I just build it out of its puzzle pieces.”

**Expected:** (Retrospective estimate) Multiplying learned projections could capture feature interactions that a plain activation missed. It might also make feature scaling harder.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-135437-5b0d7d45` | 98.7511% | 60.34% | 57.66% | 5.90 minutes | 44 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.053 → 0.3892 | 0.3892 (60) | 1.771 → 1.715 | 1.201 (12) |

The bilinear design learned, but it did not produce the jump I hoped for. Training and validation still separated.

**Runs:** [cifar10-20261025-135437-5b0d7d45](outputs/experiment-history/run-index.md#cifar10-20261025-135437-5b0d7d45).

**Draft notes:** The saved model has no residual addition. Its class name does not prove one. Do not call it a residual architecture.

### Change 126 — cifar10 — 2026-10-25

_Bilinear blocks without skip connections; unused extra linear layers remain registered_

**Change:** Use bilinear blocks without skip connections. Unused extra Linear layers remain registered. Width=512, hidden=512, depth=6.

Recorded operations silu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True.

The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I made another attempt with the bilinear block to check how it trained before changing the design further.

**Expected:** (Retrospective estimate) Another attempt could show whether the bilinear design learned reliably under this recipe.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-140146-70e4347d` | — | — | — | — | — | No final score; 22 completed epoch rows saved |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.027 → 0.7744 | 0.7744 (22) | 1.754 → 1.371 | 1.211 (11) |

Both losses ended below their starting values. The recipe learned, but the endpoint comparison alone does not isolate an improvement over another setup.

**Runs:** [cifar10-20261025-140146-70e4347d](outputs/experiment-history/run-index.md#cifar10-20261025-140146-70e4347d).

**Draft notes:** The partial run does not prove a crash. The saved model has no residual addition. Its class name does not prove one. Do not call it a residual architecture.

### Change 127 — cifar10 — 2026-10-25

_Bilinear blocks without skip connections_

**Change:** Use bilinear blocks without skip connections. Width=512, hidden=512, depth=6. Recorded operations silu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I removed unused registered layers and tried SiLU in the head to make the implementation and comparison simpler.

**Expected:** (Retrospective estimate) Removing unused parameters would simplify the model. Changing the head activation might also affect optimization, so the changes would not be fully isolated.

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

The simpler implementation stayed near the other bilinear results. It did not establish a clear gain from either the cleanup or head activation.

**Runs:** [cifar10-20261025-140404-e8234a17](outputs/experiment-history/run-index.md#cifar10-20261025-140404-e8234a17), [cifar10-20261025-141647-eeafe2e6](outputs/experiment-history/run-index.md#cifar10-20261025-141647-eeafe2e6), [cifar10-20261025-142317-26fef98f](outputs/experiment-history/run-index.md#cifar10-20261025-142317-26fef98f).

**Draft notes:** The saved model has no residual addition. Its class name does not prove one. Do not call it a residual architecture.

### Change 128 — cifar10 — 2026-10-25

_4×4 max pooling followed by bilinear residual blocks_

**Change:** Use 4×4 max pooling, followed by bilinear residual blocks. Width=512, hidden=512, depth=6. Recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I tried reducing the image before the bilinear model to test whether a smaller spatial representation made the task easier.

**Expected:** (Retrospective estimate) Spatial reduction could lower cost and discourage memorization, but it might discard useful image detail.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-195130-38a08235` | — | — | — | — | — | No final score; 13 completed epoch rows saved |
| `cifar10-20261025-201234-869c059e` | 82.2622% | 57.38% | 55.73% | 6.03 minutes | 14 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| `cifar10-20261025-195130-38a08235` | 1.828 → 0.984 | 0.984 (13) | 1.59 → 1.325 | 1.276 (9) |
| `cifar10-20261025-201234-869c059e` | 1.828 → 0.3808 | 0.3808 (60) | 1.59 → 1.774 | 1.276 (9) |

Both losses ended below their starting values. The recipe learned, but the endpoint comparison alone does not isolate an improvement over another setup.

**Runs:** [cifar10-20261025-195130-38a08235](outputs/experiment-history/run-index.md#cifar10-20261025-195130-38a08235), [cifar10-20261025-201234-869c059e](outputs/experiment-history/run-index.md#cifar10-20261025-201234-869c059e).

**Draft notes:** The partial run does not prove a crash. The saved model uses `nn.MaxPool2d(kernel_size=4)`. The assignment does not list this layer type. This was an exploration, not the final submission.

### Change 129 — cifar10 — 2026-10-25

_Normalized value/gate blocks with no dropout; no skip addition in this saved version_

**Change:** Use normalized value/gate blocks with no dropout. The saved version has no skip addition. Width=512, hidden=1024, depth=6.

Recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True.

The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I normalized the value and gate paths to control their scales before multiplying them.

**Expected:** (Retrospective estimate) Normalizing both paths could keep either branch from dominating the product and improve optimization.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261025-213314-9c0e7949` | 97.0422% | 61.52% | 59.67% | 11.40 minutes | 54 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.188 → 0.5288 | 0.5288 (60) | 1.798 → 1.475 | 1.173 (19) |

Normalizing the gate paths produced a working model, but the validation score stayed near the earlier flat-model range. This saved version also lacked residual connections.

**Runs:** [cifar10-20261025-213314-9c0e7949](outputs/experiment-history/run-index.md#cifar10-20261025-213314-9c0e7949).

**Draft notes:** The saved model has no residual addition. Its class name does not prove one. Do not call it a residual architecture.

### Change 130 — cifar10 — 2026-10-26

_Add dropout 0.1 to gated blocks_

**Change:** Use normalized value/gate blocks with 0.1 dropout. The saved version has no skip addition. Width=512, hidden=1024, depth=6.

Recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=60, seed=42. Augmentation=True.

The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** “At this point I added dropout to the output of the layer so it would have a 10% chance of setting a value of the input to zero.”

**Expected:** “Intuitively this makes sense. We randomly are tossing out information and asking the model to still answer the question.”

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-131127-ba8e8b68` | 95.12% | 63.5% | 60.87% | 11.35 minutes | 52 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.303 → 0.6481 | 0.6481 (60) | 1.789 → 1.34 | 1.165 (20) |

Both losses ended below their starting values. The recipe learned, but the endpoint comparison alone does not isolate an improvement over another setup.

**Runs:** [cifar10-20261026-131127-ba8e8b68](outputs/experiment-history/run-index.md#cifar10-20261026-131127-ba8e8b68).

**Draft notes:** The saved model has no residual addition. Its class name does not prove one. Do not call it a residual architecture. The journal describes dropout on layer outputs and uses seven masks in its probability example. This saved model has six blocks and no skip addition. For six independent masks, the keep probability along a designated path is 0.9^6. This is a mask calculation, not a measure of retained image information. The journal’s seven-mask example does not describe this saved depth.

### Change 131 — cifar10 — 2026-10-26

_Normalized value/gate blocks with 0.15 dropout; no skip addition in this saved version_

**Change:** Use normalized value/gate blocks with 0.15 dropout. The saved version has no skip addition. Width=512, hidden=1024, depth=6.

Recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=10, seed=42. Augmentation=True.

The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I increased dropout to see whether stronger regularization helped, while using a shorter trial budget.

**Expected:** (Retrospective estimate) Stronger dropout could improve generalization while making training harder. The shorter budget might also limit the fit.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-151046-ee81c4cb` | 60.6178% | 56.38% | 55.34% | 2.67 minutes | 10 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.39 → 1.368 | 1.368 (10) | 1.827 → 1.257 | 1.257 (10) |

The shorter budget prevents a direct comparison with the earlier long dropout run. The result does not isolate the effect of stronger dropout.

**Runs:** [cifar10-20261026-151046-ee81c4cb](outputs/experiment-history/run-index.md#cifar10-20261026-151046-ee81c4cb).

**Draft notes:** The saved model has no residual addition. Its class name does not prove one. Do not call it a residual architecture.

### Change 132 — cifar10 — 2026-10-26

_Normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version_

**Change:** Use normalized value/gate blocks with 0.1 dropout. The saved version has no skip addition. Width=512, hidden=1024, depth=6.

Recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=10, seed=42. Augmentation=True.

The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I returned to the lower dropout rate while keeping the short budget for a more direct comparison.

**Expected:** (Retrospective estimate) Lower dropout could restore useful learning if the stronger setting removed too much signal.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-152251-5dfa7476` | 61.68% | 56.64% | 55.61% | 1.88 minutes | 10 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.303 → 1.344 | 1.344 (10) | 1.789 → 1.264 | 1.264 (10) |

Both losses ended below their starting values. The recipe learned, but the endpoint comparison alone does not isolate an improvement over another setup.

**Runs:** [cifar10-20261026-152251-5dfa7476](outputs/experiment-history/run-index.md#cifar10-20261026-152251-5dfa7476).

**Draft notes:** The saved model has no residual addition. Its class name does not prove one. Do not call it a residual architecture.

### Change 133 — cifar10 — 2026-10-26

_Normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version_

**Change:** Use normalized value/gate blocks with 0.1 dropout. The saved version has no skip addition. Width=512, hidden=1024, depth=6.

Recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=15, seed=42. Augmentation=True.

The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I gave the lower-dropout recipe a little more time to see whether the short runs were stopping too early.

**Expected:** (Retrospective estimate) A modest extension could improve the best validation score without committing to another long run.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-152521-7a9bfc86` | 66.5511% | 58.58% | 57.41% | 2.82 minutes | 14 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.303 → 1.23 | 1.23 (15) | 1.789 → 1.19 | 1.187 (14) |

Both losses ended below their starting values. The recipe learned, but the endpoint comparison alone does not isolate an improvement over another setup.

**Runs:** [cifar10-20261026-152521-7a9bfc86](outputs/experiment-history/run-index.md#cifar10-20261026-152521-7a9bfc86).

**Draft notes:** The saved model has no residual addition. Its class name does not prove one. Do not call it a residual architecture.

### Change 134 — cifar10 — 2026-10-26

_Normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version_

**Change:** Use normalized value/gate blocks with 0.1 dropout. The saved version has no skip addition. Width=512, hidden=1024, depth=6.

Recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=16, seed=42. Augmentation=True.

The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I made another short-budget attempt near this recipe to check its behavior after the source edits.

**Expected:** (Retrospective estimate) A nearby budget could show whether useful learning continued, but an unfinished run would not provide a final accuracy comparison.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-152851-db2487cc` | — | — | — | — | — | No final score; 0 completed epoch rows saved; No completed epoch metrics were saved. |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | — | — | — | — |

No final measurement is available for this attempt. The saved record does not establish whether the reconstructed expectation was met.

**Runs:** [cifar10-20261026-152851-db2487cc](outputs/experiment-history/run-index.md#cifar10-20261026-152851-db2487cc).

**Draft notes:** This record does not prove a crash. The saved model has no residual addition. Its class name does not prove one. Do not call it a residual architecture.

### Change 135 — cifar10 — 2026-10-26

_Normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version_

**Change:** Use normalized value/gate blocks with 0.1 dropout. The saved version has no skip addition. Width=512, hidden=1024, depth=6.

Recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=20, seed=42. Augmentation=True.

The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I extended the training budget again to see whether validation still improved before memorization took over.

**Expected:** (Retrospective estimate) More epochs could help if the model still learned useful features. Validation loss would show whether the later fit stopped generalizing.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-152951-e65d9ef5` | 72.88% | 60.42% | 58.7% | 3.75 minutes | 20 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.303 → 1.136 | 1.136 (20) | 1.789 → 1.165 | 1.165 (20) |

Both losses ended below their starting values. The recipe learned, but the endpoint comparison alone does not isolate an improvement over another setup.

**Runs:** [cifar10-20261026-152951-e65d9ef5](outputs/experiment-history/run-index.md#cifar10-20261026-152951-e65d9ef5).

**Draft notes:** The saved model has no residual addition. Its class name does not prove one. Do not call it a residual architecture.

### Change 136 — cifar10 — 2026-10-26

_Normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version_

**Change:** Use normalized value/gate blocks with 0.1 dropout. The saved version has no skip addition. Width=512, hidden=1024, depth=6.

Recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=20, seed=42. Augmentation=True.

The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I checked the same visible settings again. This tested consistency rather than a new conceptual architecture.

**Expected:** (Retrospective estimate) The same visible settings could give a similar result. Agreement would support consistency, not a new architectural gain.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-153426-931b32ea` | 72.88% | 60.42% | 58.7% | 3.79 minutes | 20 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.303 → 1.136 | 1.136 (20) | 1.789 → 1.165 | 1.165 (20) |

The visible settings and validation outcome agree with the preceding run. This supports consistency without establishing a new improvement.

**Runs:** [cifar10-20261026-153426-931b32ea](outputs/experiment-history/run-index.md#cifar10-20261026-153426-931b32ea).

**Draft notes:** The saved model has no residual addition. Its class name does not prove one. Do not call it a residual architecture.

## C5 — Hierarchical rolling windows

### Change 137 — cifar10 — 2026-10-26

_Hierarchical 2×2 rolling-window MLP_

**Change:** Use a hierarchical 2×2 rolling-window MLP without edge padding. Width=256, hidden=512, depth=2. Recorded operations gelu.

Window MLP hidden/output=32/16. Spatial sizes 32 → 31 → 15 → 7 → 3 → 1. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=60, seed=42.

Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** “I'm wondering if I can create many feed forward networks. The idea being I take the32 by32 input image. Then subdivide it twice. That would then create many four by four sub images. Then run a multi-layer perceptron to create some new representation.”

**Expected:** (Retrospective estimate) Processing local neighborhoods with shared MLPs could learn image structure that a flat stream missed, without using a convolution layer.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-215237-2ac3a7b3` | 55.5444% | 49.98% | 48.27% | 9.91 minutes | 54 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 2.39 → 1.386 | 1.386 (60) | 1.874 → 1.399 | 1.386 (59) |

The first narrow window model did not beat the better flat models. Its limited training fit made more window and classifier capacity worth testing.

**Runs:** [cifar10-20261026-215237-2ac3a7b3](outputs/experiment-history/run-index.md#cifar10-20261026-215237-2ac3a7b3).

**Draft notes:** The journal sketch uses 4×4 subdivisions. The saved implementation uses rolling 2×2 windows. Keep the idea separate from the implementation.

### Change 138 — cifar10 — 2026-10-26

_Widen rolling windows and classifier_

**Change:** Use a hierarchical 2×2 rolling-window MLP without edge padding. Width=512, hidden=1024, depth=2. Recorded operations gelu.

Window MLP hidden/output=128/64. Spatial sizes 32 → 31 → 15 → 7 → 3 → 1. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=60, seed=42.

Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) The first window model had limited training and validation accuracy. I thought the small window features and classifier might be bottlenecks.

**Expected:** (Retrospective estimate) Wider window features and a larger classifier could preserve more useful information, with a risk of higher cost or memorization.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-221344-5d653ac6` | 71.3511% | 60.8% | 59.15% | 14.83 minutes | 57 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 3.277 → 1.052 | 1.052 (60) | 1.973 → 1.153 | 1.153 (60) |

The combined widening improved validation over the narrow window model. This run does not identify which width increase mattered most.

**Runs:** [cifar10-20261026-221344-5d653ac6](outputs/experiment-history/run-index.md#cifar10-20261026-221344-5d653ac6).

### Change 139 — cifar10 — 2026-10-26

_Preserve full-image coverage with edge padding_

**Change:** Use a hierarchical 2×2 rolling-window MLP with padded full-image coverage. Width=512, hidden=1024, depth=2. Recorded operations gelu, pad.

Window MLP hidden/output=128/64. Spatial sizes 32 → 31 → 16 → 8 → 4 → 2 → 1. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=60, seed=42.

Augmentation=True. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) The coverage check showed that the unpadded hierarchy ignored image edges. I changed the padding so those pixels could contribute.

**Expected:** (Retrospective estimate) Keeping image edges could improve recognition. The extra reduction stage meant that coverage and depth would change together.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-20261026-224057-085c2c3b` | 88.5156% | 76.1% | 75.45% | 12.61 minutes | 51 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 3.459 → 0.4437 | 0.4437 (60) | 1.884 → 0.7835 | 0.7223 (43) |

The full-coverage version produced the strongest gain in this window sequence. Coverage and reduction depth changed together, so their effects are not isolated.

**Runs:** [cifar10-20261026-224057-085c2c3b](outputs/experiment-history/run-index.md#cifar10-20261026-224057-085c2c3b).

**Draft notes:** Padding preserves discarded edges and adds a sixth stage. The change affects both coverage and depth.

## I1 — Flat gated MLP

### Change 140 — imagenet32 — 2026-10-26

_Flat gated MLP on ImageNet32_

**Change:** Use normalized value/gate blocks with 0.1 dropout. The saved version has no skip addition. Width=512, hidden=1024, depth=6.

Recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=10, seed=42. Augmentation=False.

The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) I moved the gated model to ImageNet32 to see how it handled many more classes and more varied images.

**Expected:** (Retrospective estimate) The larger classification problem would probably learn more slowly than CIFAR-10 and expose limits of the flat representation.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet32-20261026-195826-4b2dc142` | 18.7556% | 12.1632% | 11.106% | 42.22 minutes | 10 | — |
| `imagenet32-20261026-204057-1f232b47` | — | — | — | — | — | No final score; 4 completed epoch rows saved |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| `imagenet32-20261026-195826-4b2dc142` | 6.185 → 4.644 | 4.644 (10) | 5.829 → 4.963 | 4.963 (10) |
| `imagenet32-20261026-204057-1f232b47` | 6.185 → 5.263 | 5.263 (4) | 5.829 → 5.212 | 5.212 (4) |

Both losses fell during this short run. The many-class task remained difficult, and this saved architecture lacked residual connections.

**Runs:** [imagenet32-20261026-195826-4b2dc142](outputs/experiment-history/run-index.md#imagenet32-20261026-195826-4b2dc142), [imagenet32-20261026-204057-1f232b47](outputs/experiment-history/run-index.md#imagenet32-20261026-204057-1f232b47).

**Draft notes:** The partial run does not prove a crash. The saved model has no residual addition. Its class name does not prove one. Do not call it a residual architecture.

### Change 141 — imagenet32 — 2026-10-26

_Normalized value/gate blocks with 0.1 dropout; no skip addition in this saved version_

**Change:** Use normalized value/gate blocks with 0.1 dropout. The saved version has no skip addition. Width=512, hidden=1024, depth=6.

Recorded operations gelu. Optimizer=Muon, lr=0.03, weight_decay=1e-05, batch=256, epochs=20, seed=42. Augmentation=False.

The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) The shorter ImageNet run was still learning at its cutoff. I gave the same broad recipe more epochs.

**Expected:** (Retrospective estimate) More epochs could improve validation while both losses still fell. Later memorization could instead widen the generalization gap.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet32-20261026-210325-b9c04e03` | 28.1384% | 12.7307% | 11.648% | 114.63 minutes | 17 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 6.185 → 3.954 | 3.954 (20) | 5.829 → 5.18 | 4.947 (11) |

More training reduced the training loss, but validation loss finished above its earlier minimum. Extra fitting was not the same as continued generalization.

**Runs:** [imagenet32-20261026-210325-b9c04e03](outputs/experiment-history/run-index.md#imagenet32-20261026-210325-b9c04e03).

**Draft notes:** The saved model has no residual addition. Its class name does not prove one. Do not call it a residual architecture.

## I2 — Rolling-window MLP

### Change 142 — imagenet32 — 2026-10-27

_Rolling-window MLP on ImageNet32_

**Change:** Use a hierarchical 2×2 rolling-window MLP with padded full-image coverage. Width=512, hidden=1024, depth=2. Recorded operations gelu, pad.

Window MLP hidden/output=128/64. Spatial sizes 32 → 31 → 16 → 8 → 4 → 2 → 1. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=10, seed=42.

Augmentation=False. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) The window model had improved CIFAR-10. I tried that spatial hierarchy on ImageNet32 instead of feeding every pixel into a flat model.

**Expected:** (Retrospective estimate) The local hierarchy could help with the harder images, but it would still need to distinguish many classes.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet32-20261027-000850-1c570f4f` | 26.4322% | 20.4939% | 18.496% | 62.01 minutes | 10 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 5.692 → 3.827 | 3.827 (10) | 5.055 → 4.117 | 4.117 (10) |

The window recipe improved validation over the earlier flat gated recipe. Architecture and training choices changed together, so this is a recipe comparison.

**Runs:** [imagenet32-20261027-000850-1c570f4f](outputs/experiment-history/run-index.md#imagenet32-20261027-000850-1c570f4f).

## I3 — Continued training (analysis only)

### Change 143 — imagenet32 — 2026-10-27

_Continue ImageNet training (analysis experiment)_

**Change:** Use a hierarchical 2×2 rolling-window MLP with padded full-image coverage. Width=512, hidden=1024, depth=2. Recorded operations gelu, pad.

Window MLP hidden/output=128/64. Spatial sizes 32 → 31 → 16 → 8 → 4 → 2 → 1. Optimizer=AdamW, lr=0.001, weight_decay=1e-05, batch=256, epochs=100, seed=42.

Augmentation=False. The evidence snapshot holds the other optimizer, split, and augmentation settings.

**Why:** (Retrospective reconstruction) The first ImageNet window run still improved near its cutoff. I continued its weights to examine what more training could do.

**Expected:** (Retrospective estimate) More training could improve validation while useful learning remained. The restarted optimizer state would make this different from an uninterrupted run.

**Result:**

| Run | Train accuracy | Validation accuracy | Test accuracy | Training time | Selected epoch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet32-continuation-20261027-013419-8e7bc688` | 37.1263% | 21.9809% | 20.004% | 134.60 minutes | 25 | — |

| Run | Train loss (start → end) | Minimum train loss (epoch) | Validation loss (start → end) | Minimum validation loss (epoch) |
| --- | --- | --- | --- | --- |
| Same run | 5.692 → 3.229 | 3.229 (25) | 5.055 → 4.106 | 4.079 (16) |

Validation accuracy improved over the first window run, while later validation loss was above its minimum. Continued weights make this analysis evidence, not fresh qualification.

**Runs:** [imagenet32-continuation-20261027-013419-8e7bc688](outputs/experiment-history/run-index.md#imagenet32-continuation-20261027-013419-8e7bc688).

**Draft notes:** The run used the parent model’s saved weights. It restarted the optimizer and random state. Treat this as analysis evidence. It does not qualify as fresh-from-scratch grading performance.

## Separate branch — MLP-Mixer

### Change 144 — mnist / cifar10 / imagenet32 — 2026-10-26

_Separate MLP-Mixer implementation_

**Change:** The separate `t3/implement-google-paper` worktree uses non-overlapping 4×4 linear patches. The model has eight Mixer blocks and channel width 512. Use token/channel hidden widths of 256/2048. Apply pre-LayerNorm and residual token/channel mixing. Use the mean over tokens and a linear head. Exact task defaults are implementation settings, not a saved full-dataset experiment.

**Why:** “Note2: Given that I just discovered it was Google that came up with the idea of the MLP mixer. I'm gonna take their word for it. https://arxiv.org/html/2105.01601v4”

**Expected:** (Retrospective estimate) Patch and channel mixing could improve spatial feature processing with MLP layers. Synthetic checks would test the implementation before any full-data accuracy claim.

**Result:**

| Evidence status | Recorded result |
| --- | --- |
| Synthetic checks | Journal records synthetic checks, including Metal training. |
| Full-dataset scores | None recorded. |

Synthetic checks are recorded, including Metal training. No full-dataset experiment or run ID is recorded, so model accuracy and training time remain unknown.

## Final harness preparation

### Change 145 — mnist / cifar10 / imagenet32 — 2026-10-27

**Change:** Freeze task-specific public submission recipes. Keep them separate from the experimental dashboard. Restore the seven-block GELU MNIST model from `mnist-20261023-105208-872d486e`.

Use the padded rolling-window models from `cifar10-20261026-224057-085c2c3b` and `imagenet32-20261027-000850-1c570f4f` for the other tasks. ImageNet allows up to 25 fresh epochs. Use seed 0 in the instructor's original harness.

Use CUDA on a high-RAM A100. Set training allocations to 15/15/60 minutes. Normalize ImageNet minibatches.

Do not expand the entire dataset into float tensors. Add optional fixed-batch diagnostics. Keep them disabled by default.

**Why:** (Retrospective reconstruction) “Use the strongest documented validation recipes per task, within the time limit.” I wanted one reproducible grading path for each task. The continued ImageNet weights did not demonstrate a fresh submission.

**Expected:** (Retrospective estimate) A clean installation could run the selected recipes from scratch within their budgets and return valid probabilities. Historical scores were references, not guaranteed outcomes.

**Result:** The exact instructor harness passed on a high-RAM A100 with seed 0. The installed package used commit `b5ab6e1f75de4cfc257cf85b7e2535cfa217934c`. All models trained from scratch.

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

The ImageNet fixed-batch diagnostics contain eight snapshots. They include initialization and selected epoch 23.

| Residual classifier branch removed | Validation accuracy drop (percentage points) |
| --- | ---: |
| First | 10.1634 |
| Second | 3.6131 |

Full validation accuracy after restoring the saved selected model exactly matched training's selected checkpoint. The fresh ImageNet result remains separate from the higher continued-weight score. Only the fresh run demonstrates the grading recipe.

**Runs:** [Fresh harness evidence and run records](outputs/final-evaluation/README.md). [Report images](outputs/report-figures/README.md). Keep the historical 20.004% continuation separate from the fresh 19.960% result.

## Follow-up analysis — window-stage spatial ablations — 2026-10-27

The evaluation used the saved epoch-23 ImageNet model. It did not retrain the model. It used the same validation split. At each of six window stages, the evaluation replaced each neighborhood with repeated copies of its top-left pixel or its mean. These replacements kept the channel mapping and shape. They removed spatial information.

| Evaluation | Stage | Validation accuracy |
| --- | ---: | ---: |
| Mean-only ablation | 1 | 0.5823% |
| Mean-only ablation | 4 | 16.4515% |
| Full model | — | 22.0222% |

The controls did not change the weights or the restored accuracy. [All twelve
results and methods](outputs/window-stage-ablation/README.md).

The mean-only controls reduce accuracy sharply at stage 1 and less at stage 4. These fixed-weight interventions show that spatial information at both stages matters; they do not measure retrained performance or remove a whole stage.

### Change 146 — imagenet32 — 2026-10-28

**Change:** Add one residual MLP before the last 2×2 to 1×1 window stage. Flatten 64×2×2 to 256. Apply pre-LayerNorm, Linear(256,512), GELU, and Linear(512,256).

Add the original stream. Then reshape. Add 263,424 parameters.

Keep the final image representation, training recipe, seed 0, and split unchanged.

**Why:** (Retrospective reconstruction) “It almost seems to imply there's more signal to be extracted with additional layers somehow?” The spatial controls made me wonder whether another block before compression could help. This experiment tested that idea.

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

The extra block completed all 25 epochs. Its best validation accuracy was lower. Both training loss at epoch 25 and minimum validation loss were higher. It did not support the accuracy hypothesis within this recipe; the candidate was still improving at cutoff. No continuation or retry ran, and no test-set evaluation or default change occurred. This result does not show that depth never helps.

**Runs:** `imagenet32-prereduction-20261028-s0`; [pre-run protocol](outputs/pre-reduction-experiment/intent.json); [results, curves, and evidence](outputs/pre-reduction-experiment/README.md).
