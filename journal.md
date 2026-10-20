# Development Journal

by Taylor Turner

## Baseline

Stochastic Gradient Descent
a = 0.05
epochs = 10
seed = 42 (if not specified, assume 42)
mnist: 95.65% test accuracy · 0.5 minutes · mps

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

> At this stage I don't see much point in adding more learning rates or more epochs. They're leveling out at very low loss, but not ~100% yet.

### Train accuracy check

> I realized I didn't record train accuracy so I can't compare. Let's understand how train accuracy is looking.

a=0.05
mnist: 98.12% test accuracy · 100.00% train accuracy · 2.4 minutes · mps

a=0.10
mnist: 98.10% test accuracy · 99.97% train accuracy · 2.4 minutes · mps

a=0.15
mnist: 97.97% test accuracy · 99.94% train accuracy · 2.4 minutes · mps

> We see that the baseline learning rate was perfect for this current configuration. All it truly needed was a few more epochs. Without those epochs, it looked like it needed more help than it really did. Better test and train accuracy for baseline w/ 50 epochs. Time to move on.

> I am thinking the next steps are to investigate the gradient norm over layers. Then, tinker with the number of layers. We can think about changing the shape once it is understood if the gradient is stable.

# Planning

> Everything below this point is ideas and planning, not necessarily implemented.

### Dropout with bootstrapping

### Oscillation detection and mean.
