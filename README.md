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
package in editable mode, and opens the single notebook,
`src/residual_mlp_baseline/notebook.py`. Select a task using the controls at the
top, then click **Train and evaluate**. The first run downloads the dataset; later runs reuse
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
uv run --locked marimo run src/residual_mlp_baseline/notebook.py
```

In VS Code, install the recommended Python and marimo extensions and select
`.venv/bin/python` as the interpreter. Open
`src/residual_mlp_baseline/notebook.py` with the marimo extension. Model classes,
training functions, dataset loading, and evaluation live together in this file.
The package and grading harness import these same definitions.

With [just](https://github.com/casey/just) installed, use these shortcuts:

```bash
just model       # Open the consolidated notebook source in Zed.
just train       # Open the same notebook source in Zed.
just app         # Open the consolidated marimo notebook.
just app-source  # Open the notebook source in Zed.
```

Run `just` to list commands. Pass another macOS app name to a source shortcut,
for example `just model "Visual Studio Code"`. `just marimo` aliases `just app`;
extra marimo options work too, such as `just app --headless --port 2718`.

The **Train and evaluate** button, task, device, and seed controls appear first,
with run results directly below. Edit imports and settings in the setup cell,
and model classes and training functions in their cells below. Training starts
only when you click the button. `just app` also watches files saved from an
external editor.

`project1_harness.ipynb` is the grading notebook. Its setup cell's `PACKAGE_URL`
points to `git+https://github.com/TaylorTurnerIT/AI4200-imagenet32x32-baseline-model.git`.
Change that value if your submission repository changes. Set `SECTION` to your
course section before submitting; leave `LEADERBOARD_ALIAS` empty to skip posting.
The notebook also selects MPS on Macs when CUDA is unavailable.
The marimo app uses the same dataset sources, cache format, and required
`train()` / `predict()` interface. It uses the local model and training definitions
instead of reinstalling the instructor's GitHub package. Generated datasets,
results, and virtual environments are ignored by Git.

### Submission installation

```bash
pip install git+https://github.com/JesseTNRoberts/AI4200-imagenet32x32-baseline-model.git
```

## Files

- `src/residual_mlp_baseline/notebook.py`: the single marimo notebook containing the model, training, prediction, and local evaluation controls.
- `src/residual_mlp_baseline/model.py`: compatibility imports for the model classes and task sizes.
- `src/residual_mlp_baseline/train.py`: compatibility imports for the required `train` and `predict` functions.
- `project1_harness.ipynb`: the test harness. It downloads and caches the datasets, installs a package from GitHub, and reports test accuracy.

## Use

```python
from residual_mlp_baseline import train, predict

model = train("cifar10", X_train, y_train, seed=0, device="cuda")
probs = predict(model, X_test)
accuracy = (probs.argmax(1) == y_test).float().mean()
```

`task` is `"mnist"`, `"cifar10"`, or `"imagenet32"`.
