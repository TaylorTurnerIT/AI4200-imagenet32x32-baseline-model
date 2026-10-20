# Development Journal

by Taylor Turner

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

layers=6
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


layers=6
epochs=25
a=0.05
width=32
hidden=256

mnist: 95.92% test accuracy · 96.93% train accuracy · 0.4 minutes · mps

> Immediately slower training. Didn't reach 0.0# until epoch 18.

layers=6
epochs=25
a=0.05
width=256
hidden=256
mnist: 98.04% test accuracy · 99.99% train accuracy · 0.5 minutes · mps

> Recovered training speed, 6 epochs to hit 0.0#%. Incredible training accuracy but a trivial improvement in test. All we did was improve the models ability to overfit. This looks like a dead end. Let's try a 50 epoch here since I see a bit more room for training loss.


layers=6
epochs=50
a=0.05
width=256
hidden=256

mnist: 98.01% test accuracy · 100.00% train accuracy · 0.9 minutes · mps

> We gained an incredible -0.03%. I'd say that was just further overfitting pushing our performance worse. Having reached 100% at a low epoch, we could be in a good place for some regularization. For now, let's see what happens when we tinker with the size of the hidden layers. Perhaps reducing the models capacity will serve a similar role.

layers=6
epochs=50
a=0.05
width=256
hidden=512

mnist: 98.17% test accuracy · 100.00% train accuracy · 1.0 minutes · mps

> New record! 98.18 with almost no wiggle room. I forgot to turn down epochs so we've got quite a bit of training happening here.

---

> I am thinking the next steps are to investigate the gradient norm over layers. Then, tinker with the number of layers. We can think about changing the shape once it is understood if the gradient is stable.

# Planning

> Everything below this point is ideas and planning, not necessarily implemented.

### Dropout with bootstrapping

### Oscillation detection and mean.
