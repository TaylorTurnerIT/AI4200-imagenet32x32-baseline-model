# Development Journal

by Taylor Turner

## Baseline

a = 0.05
epochs = 10
seed = 42 (if not specified, assume 42)
mnist: 95.65% test accuracy · 0.5 minutes · mps

## Manual Hyperparameter Tuning

> I began by leaving the model architecture unchanged, only changing hyperparameters by hand to understand their impacts.

### Learning Rate

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

# Planning

> Everything below this point is ideas and planning, not necessarily implemented.

### Dropout with bootstrapping

### Oscillation detection and mean.
