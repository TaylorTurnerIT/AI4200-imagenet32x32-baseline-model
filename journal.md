# Development Journal

by Taylor Turner

# MNIST 

## Baseline

Stochastic Gradient Descent
a = 0.05
epochs = 10
seed = 42 (if not specified, assume 42)
mnist: 95.65% test accuracy · 0.5 minutes · mps

> For now, I am only focused on the MNIST. Once I have achieved a perfect score on that dataset, I will move onto the next.

---

## Manual Hyperparameter Tuning

> I began by leaving the model architecture unchanged, only changing hyperparameters by hand to understand their impacts. 

### Learning Rate

**a = 0.05** (Baseline)
mnist: 95.65% test accuracy · 0.5 minutes · mps

**a = 0.01**
epochs = 10
seed = 42
mnist: 95.49% test accuracy · 0.6 minutes · mps

**a = 0.10**
epochs = 10
seed = 42
__**mnist: 96.74% test accuracy · 0.5 minutes · mps**__

**a = 0.15**
epochs = 10
seed = 42
mnist: 96.11% test accuracy · 0.5 minutes · mps

### Let's try more epochs.

**a = 0.01**
epochs = 50
seed = 42
mnist: 97.56% test accuracy · 2.5 minutes · mps

**a = 0.10**
epochs = 50
seed = 42
mnist: 98.12% test accuracy · 2.4 minutes · mps

**a = 0.15**
epochs = 50
seed = 42
mnist: 97.97% test accuracy · 2.4 minutes · mps

> We see here lots of big jumps followed by recovery. Not as good as a=0.10.

...
0.0951
0.0790
**0.5566**
0.0905
0.0786
0.0611
0.0679
0.0467
0.0431
**0.2268**
0.0558
0.0369
0.0308
0.0270
...

> At this stage I don't see much point in adding more learning rates or more epochs. They're leveling out at very low loss, but not ~100% yet. The jump to single digits happens at epoch 25 so we see diminishing returns from that point.

---

### Train accuracy check

> I realized I didn't record train accuracy so I can't compare. Let's understand how train accuracy is looking.

a=0.05
mnist: 98.12% test accuracy · 100.00% train accuracy · 2.4 minutes · mps

a=0.10
mnist: 98.10% test accuracy · 99.97% train accuracy · 2.4 minutes · mps

a=0.15
mnist: 97.97% test accuracy · 99.94% train accuracy · 2.4 minutes · mps

> We see that the baseline learning rate was perfect for this current configuration. All it truly needed was a few more epochs. Without those epochs, it looked like it needed more help than it really did. Better test and train accuracy for baseline w/ 50 epochs. Time to move on.

---

### Layer Tinkering

> After my findings with accuracy revealed that the default learning rate was a solid starting point, I'm ready to move onto layers. For this, I intend to begin first by scaling down the layers. Right now I assume the issue is we're losing valuable signal as the layer layers add nothing but noise to an ultimately simple problem. I expect to see some small number of layers, perhaps 10-15 perform the same or better.

> I am changing my epochs to 25 for speed of training.

### Current Model
a=0.5
seed=42
layers=40
epochs=25

mnist: 97.79% test accuracy · 99.70% train accuracy · 1.2 minutes · mps

> Note: Insane that we only lose 0.30% on train with 25 less epochs

others kept the same
layers=5
mnist: 97.89% test accuracy · 99.73% train accuracy · 0.3 minutes · mps

> An incredible gain! As predicted we actually see a performance increase by decreasing the layers and a gain of 1.2 minutes

Let's try 1-4 as well.

layers=1
mnist: 97.73% test accuracy · 99.06% train accuracy · 0.4 minutes · mps

layers=2
mnist: 97.77% test accuracy · 99.09% train accuracy · 0.2 minutes · mps

layers=3
mnist: 97.80% test accuracy · 99.56% train accuracy · 0.3 minutes · mps

layers=4
mnist: 97.92% test accuracy · 99.67% train accuracy · 0.9 minutes · mps

layers=5
mnist: 97.89% test accuracy · 99.73% train accuracy · 0.3 minutes · mps

> These results are pretty incredible. 5 layers was a solid guess. My initial thoughts are that we've given a lot of opportunity across five layers of 256 hidden neurons we've given a massive amount of capabity to transform the residual stream. Even still, that capacity is still making ~0.10% improvements to performance. Layers are important, but they aren't the whole story. What I am seeing as the trainings go on is that the epochs are dropping significantly faster. We're reaching single digits (0.0#) within approximately seven epochs 

> For now, let's try a few more until we see a performance regression.

layers=6
mnist: 98.03% test accuracy · 99.78% train accuracy · 0.4 minutes · mps

layers=7
mnist: 97.60% test accuracy · 99.67% train accuracy · 0.5 minutes · mps

> Regression. Let's roll with 6 layers for now. Going to try a run with 50 epochs and 6 layers.

> Depth correction (retrospective): The runs below, through the optimizer experiments and before the later depth-discovery note, probably used `depth=7` despite recording `layers=6`. I likely forgot to reset depth after the seven-layer test. Git confirms seven in the October 19 commits, but the exact change time is uncertain. Replaying the later regularized configuration reproduced 98.24% with seven blocks and 98.16% with six.

layers=6 # Probably 7; see depth note above.
epochs=50
mnist: 98.01% test accuracy · 100.00% train accuracy · 0.9 minutes · mps

...

epoch 47: train loss 0.0012

epoch 48: train loss 0.0011

epoch 49: train loss 0.0010

epoch 50: train loss 0.0010

> Zero change in the last epoch. Clearly reached the floor of this configuration. 

> 100% train and 98.01% test shows only 0.09% regression over 50 layers. Nevertheless its still a regression, we need more ways to improve performance. Something is missing and my next target is the size of the hidden layers.

### Changing Sizes

> My ideas here are to change the size of the residual stream (width) and the size of the hidden layers (hidden).

> Let's start by changing the residual stream. I don't have immediate intuition as to what it will change. At a size of 128 we're storing 128 dimensions worth of signal out of the input 768 pixels. Perhaps a smaller residual stream is equally capable, perhaps a larger residual layer will allow more signal through. If the latter is the case, I could see a case for more layers once more.

Baseline
width=128
mnist: 98.03% test accuracy · 99.78% train accuracy · 0.4 minutes · mps


layers=6 # Probably 7; see depth note above.
epochs=25
a=0.05
width=32
hidden=256

mnist: 95.92% test accuracy · 96.93% train accuracy · 0.4 minutes · mps

> Immediately slower training. Didn't reach 0.0# until epoch 18.

layers=6 # Probably 7; see depth note above.
epochs=25
a=0.05
width=256
hidden=256
mnist: 98.04% test accuracy · 99.99% train accuracy · 0.5 minutes · mps

> Recovered training speed, 6 epochs to hit 0.0#%. Incredible training accuracy but a trivial improvement in test. All we did was improve the models ability to overfit. This looks like a dead end. Let's try a 50 epoch here since I see a bit more room for training loss.


layers=6 # Probably 7; see depth note above.
epochs=50
a=0.05
width=256
hidden=256

mnist: 98.01% test accuracy · 100.00% train accuracy · 0.9 minutes · mps

> We gained an incredible -0.03%. I'd say that was just further overfitting pushing our performance worse. Having reached 100% at a low epoch, we could be in a good place for some regularization. For now, let's see what happens when we tinker with the size of the hidden layers. Perhaps reducing the models capacity will serve a similar role.

layers=6 # Probably 7; see depth note above.
epochs=50
a=0.05
width=256
hidden=512

mnist: 98.17% test accuracy · 100.00% train accuracy · 1.0 minutes · mps

> New record! 98.18 with almost no wiggle room. I forgot to turn down epochs so we've got quite a bit of training happening here. 

> We've given the model a huge amount of per-layer capacity and residual stream capacity that it has used to memorize the train accuracy but still find some generalization. At this stage I'd like to do a wild experiment then move onto regularization. The idea being we're not able to learn anything new once we've memorized training so there's no more room left to gain test accuracy.

layers=6 # Probably 7; see depth note above.
epochs=50
a=0.05
width=512
hidden=512

mnist: 98.16% test accuracy · 100.00% train accuracy · 1.1 minutes · mps

layers=6 # Probably 7; see depth note above.
epochs=50
a=0.05
width=512
hidden=1024

mnist: 98.15% test accuracy · 100.00% train accuracy · 1.4 minutes · mps

> Crazy experiments over with no real change. This is obviously a dead end. Onto Regularization.

## Regularization

### L2 Regularization

> Weight decay is a classic from ML. Let's try to force down those worthless weights and see if we can open up some capacity for the test accuracy to use.

> Implementing using simple pytorch optimizer parameter. (weight_decay=WEIGHT_DECAY)

layers=6 # Probably 7; see depth note above.
width=256
hidden=512
a=0.05
epochs=50
weight_decay=0.99

mnist: 11.35% test accuracy · 11.24% train accuracy · 1.0 minutes · mps

> Didn't learn a thing. Only moved by 0.0001 across 50 epochs. 0.99 for lamda means that 99% of the squared magnitude is allowed to affect the weight decay, so much regularization that it can't train at all.

layers=6 # Probably 7; see depth note above.
width=256
hidden=512
a=0.05
epochs=25
weight_decay=0.01

mnist: 87.79% test accuracy · 86.78% train accuracy · 0.5 minutes · mps

> Still a big decay lambda. Good to see that the theory held. Let's try suggestion from the internet, 1e-4.

layers=6 # Probably 7; see depth note above.
width=256
hidden=512
a=0.05
epochs=25
weight_decay=1e-4

mnist: 98.09% test accuracy · 99.96% train accuracy · 0.5 minutes · mps

...
epoch 22: train loss 0.0093

epoch 23: train loss 0.0077

epoch 24: train loss 0.0060

epoch 25: train loss 0.0047

> This is interesting, we're almost back where we started but we still have a little wiggle room with train accuracy. I still see some more downward motion in the loss as well. Let's try 50 epochs.

layers=6 # Probably 7; see depth note above.
width=256
hidden=512
a=0.05
epochs=50
weight_decay=1e-4

mnist: 98.24% test accuracy · 100.00% train accuracy · 1.0 minutes · mps

> The plan paid off! We closed the train accuracy to 100% again and gained 0.10 of test accuracy. There's more regularization where that came from.

layers=6 # Probably 7; see depth note above.
width=256
hidden=512
a=0.05
epochs=50
weight_decay=0.001

mnist: 97.92% test accuracy · 99.27% train accuracy · 1.0 minutes · mps

> Tried another jump in regularization and I'm starting to see a lot of oscillation. We've lost test accuracy in exchange for some train room. Let me try a smaller learning rate.

layers=6 # Probably 7; see depth note above.
width=256
hidden=512
a=0.03
epochs=50
weight_decay=0.001

mnist: 98.06% test accuracy · 99.66% train accuracy · 1.0 minutes · mps

layers=6 # Probably 7; see depth note above.
width=256
hidden=512
a=0.01
epochs=50
weight_decay=0.001

mnist: 97.97% test accuracy · 99.91% train accuracy · 1.0 minutes · mps

> a=0.03 worked. I saw oscillation happening and it predictably solved it. With 0.01 we're seeing incredibly slow training with more room for training by epoch 50. Even if there's room there, I would rather explore another path than wait for an endless training time. Especially with only 0.09% training accuracy left.

> It's not immediately clear to me where to go from here, but I'm starting to get curious about using a fresh optimizer configuration. Dropout comes to mind for more regularization as well if we need it.

## Optimizer Tuning

> We're already using SGD, I have no interest in trying classical GD due to time and probable worse performance. So, let's try just adding some momentum and see what happens.

> I also saw a parameter "dampening" which decreases the impact of the current gradient to help prevent overshooting. We can try that if we see overshoot. 

layers=6 # Probably 7; see depth note above.
width=256
hidden=512
a=0.01
epochs=50
weight_decay=0.001
momentum=0.9

mnist: 96.66% test accuracy · 98.25% train accuracy · 1.1 minutes · mps

> Well that didn't work. Let's try less.

layers=6 # Probably 7; see depth note above.
width=256
hidden=512
a=0.01
epochs=50
weight_decay=0.001
momentum=0.5

mnist: 98.00% test accuracy · 100.00% train accuracy · 1.1 minutes · mps

> Well that kinda worked. Maybe more regularization?

layers=6 # Probably 7; see depth note above.
width=256
hidden=512
a=0.01
epochs=50
weight_decay=0.005
momentum=0.5

mnist: 94.08% test accuracy · 94.67% train accuracy · 1.1 minutes · mps

> Huge overshoot. Let's try less momentum then bring in dampening. Could also try a smaller learning rate.

layers=6 # Probably 7; see depth note above.
width=256
hidden=512
batch=256
a=0.01
epochs=50
weight_decay=0.005
momentum=0.3

mnist: 96.65% test accuracy · 97.73% train accuracy · 1.1 minutes · mps

> Doing something else as a waiting move. Changing weight decay back down and turning down batch size so it must generalize more.

layers=6 # Probably 7; see depth note above.
width=256
hidden=512
batch=128
a=0.01
epochs=50
weight_decay=0.001
momentum=0.3

mnist: 96.36% test accuracy · 98.41% train accuracy · 2.0 minutes · mps

> Still oscillating. Let's bring in dampening with previous batch size and new batch size. Odd that we overfit more with smaller batches.

> https://distill.pub/2017/momentum/ This post opened my eyes. Let's experiment. Maybe our learning rate is too high? Let's try more momentum, less learning rate.

layers=6 # Probably 7; see depth note above.
width=256
hidden=512
batch=256
a=0.001
epochs=50
weight_decay=0.001
momentum=0.9

mnist: 97.76% test accuracy · 99.96% train accuracy · 1.2 minutes · mps

> Slow training, okay results. More batch to make more little jumps?

layers=6 # Probably 7; see depth note above.
width=256
hidden=512
batch=128
a=0.001
epochs=50
weight_decay=0.001
momentum=0.9

mnist: 97.85% test accuracy · 99.94% train accuracy · 2.0 minutes · mps

> Faster convergence, seeing some oscillation. 

layers=6 # Probably 7; see depth note above.
width=256
hidden=512
batch=128
a=0.0001
epochs=50
weight_decay=0.001
momentum=0.9

mnist: 97.94% test accuracy · 100.00% train accuracy · 2.9 minutes · mps

> I'm starting to think that I have focused too much on MNIST. I think it's time to accept my high score and call it. Good to move on and come back later with future findings.

> Note: at some point I set depth=7. No idea when that happens but almost every experiment has now been thrown into question. We found minimal changes but not zero.

layers=7
width=256
hidden=512
batch=256
a=0.05
epochs=50
weight_decay=0.1e-4 # accident, 0.1e-4 == 1e-5 != 1e-4

mnist: 98.22% test accuracy · 100.00% train accuracy · 1.0 minutes · mps

> I actually made a mistake, I forgot to remove the 0. before the 1 in weight_decay and I got this 98.22%. 

layers=7
width=256
hidden=512
batch=256
a=0.05
epochs=50
weight_decay=1e-4

mnist: 98.16% test accuracy · 100.00% train accuracy · 0.9 minutes · mps

layers=6
width=256
hidden=512
batch=256
a=0.05
epochs=50
weight_decay=1e-4

mnist: 98.24% test accuracy · 100.00% train accuracy · 1.0 minutes · mps

> Today in class we discussed double descent. So, I tried training MORE with what worked. 

layers=6
width=256
hidden=512
batch=256
a=0.05
epochs=500
weight_decay=1e-4

mnist: 98.32% test accuracy

> And it worked! I tried with 5000 too but that didn't work. Its performance was incredibly variable with loss so I decided to try checking whatever epoch was the winner. That epoch would have the perfect amount of training.

> Then, also building from what we did in class I decided to add data augmentation. Move the image around, blur it, brighten it, etc. 

> When I implemented augmentation, I used Codex and it made an executive decision I was initially confused by. It created a train/test/validation split. The issue it spotted and I didn't was that I was leaking information into the test set. By choosing the ideal epoch for the training set I wasn't leaving it independent as I was tuning the model to it. This was an important realization. As a competition, the flaw is that we're trying to maximize performance on data we have. Thus, it is incredibly liable to data leakage by using test to tune.

> So, continuing with validation I had it choose the best epoch for the validation set. In one run, I achieved 99.32% accuracy after 384 epochs. Just when I was thinking about switching to CIFAR, I passed the 99% barrier twice. Then I tried setting all of the activations to GELU. Slight regression so I am finally moving on to CIFAR.

# CIFAR

## Baseline (inherited from MNIST)

layers=6
width=256
hidden=512
batch=256
a=0.05
epochs=500 (keep best)
weight_decay=1e-4
activation_function (all): GELU

cifar10: 54.59% test accuracy · 99.99% train accuracy · 0.5 minutes · mps
Best epoch: 28 · validation accuracy: 56.40%

> Crazy stuff. Instantly pulls out 99.99% train accuracy? 54% test? This indicates to me that it has become so difficult to learn generalizable patterns that it learned what it could then instantly memorized. Very strange this didn't happen with MNIST. Perhaps there was more generalizable ideas here. Let's try tanh, not expecting much. At this size of network it should be pretty much interchangable. But at this overfitting I feel it's a bad idea to scale up further.

layers=6
width=256
hidden=512
batch=256
a=0.05
epochs=500 (keep best)
weight_decay=1e-4
activation_function (all): tanh

cifar10: 52.00% test accuracy · 91.51% train accuracy · 0.5 minutes · mps
Best epoch: 25 · validation accuracy: 52.84%

> Worse. Reverting. Let's try some more.

layers=6
width=256
hidden=512
batch=256
a=0.05
epochs=500 (keep best)
weight_decay=1e-4
activation_function (all): ReLU

cifar10: 52.00% test accuracy · 91.51% train accuracy · 0.5 minutes · mps
Best epoch: 25 · validation accuracy: 52.84%

> Clearly there's more to the story of CIFAR than there was with MNIST.

> I applied a shrink to the residual stream to 64. Didn't do much. Did that + shrank batch size:

cifar10: 51.78% test accuracy · 83.72% train accuracy · 0.8 minutes · mps
Best epoch: 23 · validation accuracy: 52.28%

> I made a new format so I can stop losing track of details. Tried 10x the amount of weight decay. Hurt train accuracy a lot in the long run. The strange thing that I can't get over is this ~52% accuracy. Why would the model even while hurting for train accuracy so badly remain firm on this 52%? If there was only two classes, it would be obvious that it was randomly guessing. But given ten options, it gets half of them wrong. I think we need to understand which classes its getting right and wrong.

```
Run: runs/cifar10-20261023-123652-808ffc8f
Time: 2026-10-23 12:37:37
Task: cifar10
Train accuracy:      68.53% (45000 clean examples)
Validation accuracy: 52.48% (5000 examples)
Test accuracy:       52.09% (10000 examples)
Best epoch: 10 / 30 completed (30 configured)
Training time: 0.74 minutes
All final scores use the best validation checkpoint. Train excludes the holdout.

Hyperparameters and architecture:
{
  "task": "cifar10",
  "seed": 42,
  "device": "mps",
  "epochs": 30,
  "batch_size": 128,
  "learning_rate": 0.05,
  "weight_decay": 1e-05,
  "momentum": 0.0,
  "dampening": 0,
  "optimizer": "SGD",
  "loss": "CrossEntropyLoss()",
  "learning_rate_schedule": "constant",
  "shuffle": true,
  "architecture": {
    "width": 128,
    "hidden": 128,
    "depth": 7,
    "functional_operations": [
      "relu"
    ],
    "in_features": 3072,
    "num_classes": 10,
    "normalization": [
      "LayerNorm"
    ],
    "parameters": 644362
  },
  "validation_fraction": 0.1,
  "training_examples": 45000,
  "validation_examples": 5000,
  "selection_metric": "validation_accuracy",
  "tie_break": "earliest_epoch",
  "augmentation": {
    "enabled": false
  }
}
```

test	0	airplane	  611	1000	61.1 (vs 74.83 test)
test	1	automobile	632	1000	63.2 (vs 80.33 test)
test	2	bird	      345	1000	34.5 (49.67 test)
test	3	cat	        296	1000	29.6 (46.09 test)
test	4	deer	      470	1000	47   (67.25)
test	5	dog	        491	1000	49.1 (66.16
test	6	frog	      583	1000	58.  (74.6)
test	7	horse	      578	1000	57.8 (76.71)
test	8	ship	      658	1000	65.8 (76.26)
test	9	truck	      545	1000	54.5 (73.37)

> Cat is incredibly far off but we haven't learned much here. It's learning something about all of these

# Planning

> Everything below this point is ideas and planning, not necessarily implemented.

### Dropout with bootstrapping

### Oscillation detection and mean.
