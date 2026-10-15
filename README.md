# AI 4200 Project 1: Residual MLP baseline

The instructor baseline. Fork this repo as the starting point for your submission.

## The assignment

Your job is to build, train, and analyze a residual MLP on three image classification benchmarks: **MNIST**, **CIFAR-10**, and **ImageNet 32×32**. This repo is a deliberately plain starting point. It is meant to be beaten.

**What you can change.** You can change anything about how the network is built and trained, as long as it stays an MLP made from these four pieces:

- **Linear layers.** Any depth and any width.
- **Activations.** ReLU, tanh, or anything else.
- **Normalization.** Batch norm, layer norm, or anything else.
- **Residual connections.**

You can also experiment freely with:

- **Training.** Loss function, optimizer, learning-rate schedule, regularization (dropout, weight decay, and so on), and initialization.
- **Data handling.** Data augmentation, test-time augmentation, and creative ways of feeding images into an MLP, such as splitting an image into patches or mixing information across positions. You must be able to explain any of these in your report.

**What you cannot do.**

- **No new layer types.** No convolutions, recurrent layers, or other modality-specific layers.
- **No ensembles.** Each submission is one model.
- **No pretrained weights and no distillation.** Your model must be trained from the data directly.
- **No extra data.** Use only the data each benchmark provides.
- **No saved weights.** Your code must build and train the model from scratch when `train()` is called. It may not load saved weights.

**Document every run.** The core of the grade is your development process. For every training run, record:

- **What you changed** and why.
- **What happened**: the accuracy and the loss curves.
- **Your reading of the result.**
- **What you will try next**, based on that reading.

A run that fails but is well analyzed is worth more than a lucky one that isn't explained.

**How you submit.** Submit a public GitHub repository that installs with:

```bash
pip install git+https://github.com/<you>/<your-repo>.git
```

Your package must expose the two functions in `train.py`, with their names and arguments unchanged:

```python
model = train(task, X_train, y_train, seed=0, device="cuda")   # build and train from scratch
probs = predict(model, X_test)                                 # (N, num_classes) probabilities
```

The course notebook downloads the data, installs your package, calls these two functions for each task, and reports test accuracy. Do your development in your own environment. The notebook is only the test harness.

**Leaderboard.** Test accuracies go on a class leaderboard. The leaderboard does not count toward your grade. The single top-performing model earns extra credit: **3 points** if its author is in the 4000-level section, or **2 points** if in the 5000-level section.

## Install

### Local development with uv and marimo

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
./run
```

This installs Python 3.12 and the locked dependencies into `.venv`, installs this
package in editable mode, and opens `app.py` in marimo. Select a task and click
**Train and evaluate**. The first run downloads the dataset; later runs reuse
`data/`. The app prefers CUDA, then Apple Silicon MPS, then CPU. You can select
CPU manually. The status panel verifies an actual Metal forward/backward operation
and shows the Python interpreter. Select **Apple Metal (MPS)** to train on this
Mac's GPU. CUDA is NVIDIA-specific and is not required on macOS.
Results append to `results.csv`; the app never posts to the leaderboard.

The app defaults to MNIST. CIFAR-10 and ImageNet 32×32 are available in the task
selector. ImageNet requires several GB of disk space and substantially more RAM
during training: the baseline converts the full training array to float32.
Dataset downloads require internet access.

To prepare the environment without opening marimo:

```bash
uv sync --locked
```

To launch the app without the editor:

```bash
uv run --locked marimo run app.py
```

In VS Code, install the recommended Python and marimo extensions and select
`.venv/bin/python` as the interpreter. Open `app.py` with the marimo extension.
Restart the marimo session after editing imported model or training code to ensure
it uses your latest changes.

With [just](https://github.com/casey/just) installed, use these shortcuts:

```bash
just model       # Open model.py in Zed.
just train       # Open train.py in Zed.
just app         # Open the marimo notebook editor in your browser.
just app-source  # Open app.py as Python source in Zed.
```

Run `just` to list commands. Pass another macOS app name to a source shortcut,
for example `just model "Visual Studio Code"`. `just marimo` aliases `just app`;
extra marimo options work too, such as `just app --headless --port 2718`.

Marimo's notebook editor shows the cells in `app.py`. Edit the ordinary Python
modules `model.py` and `train.py` in your code editor. Restart the marimo session
after changing those modules. `just app` watches edits to `app.py`, so that file
can also be edited in your code editor while marimo is open.

`project1_harness.ipynb` is the grading notebook. Its setup cell's `PACKAGE_URL`
points to `git+https://github.com/TaylorTurnerIT/AI4200-imagenet32x32-baseline-model.git`.
Change that value if your submission repository changes. Set `SECTION` to your
course section before submitting; leave `LEADERBOARD_ALIAS` empty to skip posting.
The notebook also selects MPS on Macs when CUDA is unavailable.
The marimo app uses
the same dataset sources, cache format, and required `train()` / `predict()`
interface. It imports your local package instead of reinstalling the instructor's
GitHub package. Generated datasets, results, and virtual environments are ignored
by Git.

### Submission installation

```bash
pip install git+https://github.com/JesseTNRoberts/AI4200-imagenet32x32-baseline-model.git
```

## Files

- `src/residual_mlp_baseline/model.py`: the residual MLP. One class for all three tasks; only the input size and number of classes change.
- `src/residual_mlp_baseline/train.py`: the required `train` and `predict` functions.
- `app.py`: the local marimo development app.
- `project1_harness.ipynb`: the test harness. It downloads and caches the datasets, installs a package from GitHub, and reports test accuracy.

## Use

```python
from residual_mlp_baseline import train, predict

model = train("cifar10", X_train, y_train, seed=0, device="cuda")
probs = predict(model, X_test)
accuracy = (probs.argmax(1) == y_test).float().mean()
```

`task` is `"mnist"`, `"cifar10"`, or `"imagenet32"`.
