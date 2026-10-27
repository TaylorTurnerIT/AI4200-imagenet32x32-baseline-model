# AI 4200 Project 1: Residual MLP baseline

The project started from the instructor baseline. See [SUBMISSION.md](SUBMISSION.md)
for the frozen task-specific grading recipes and exact-harness reproduction.
The [development log draft](LOG.md) and [report figures](outputs/report-figures/)
preserve the experimental history.

## Rolling 2×2 window experiment

The model uses six hand-built window MLP stages for 32×32 images:
`32 → 31 → 16 → 8 → 4 → 2 → 1`. The first stage slides by one pixel; later stages
slide by two. MNIST shrinks `28 → 27 → 14 → 7 → 4 → 2 → 1`. Stride-2 stages
append one zero row or column at the bottom or right when needed, so unmatched
edges remain connected to the final prediction.

Each stage uses `Tensor.unfold` and reshaping to gather 2×2 windows, then applies
one shared `Linear(4*C, 128) → GELU → Linear(128, 64) → GELU` MLP at every position.
Different stages learn separate weights. The output becomes the next stage's
64-channel feature map. The final 64-value vector feeds a width-512 classifier
with two residual blocks and a linear class head. There are no convolution or
pooling layers. Sharing a local linear map has the same mathematical structure
as a convolution; the implementation explicitly gathers windows and uses linear
layers, with a nonlinear MLP at each position.

Edit `window_hidden`, `window_features`, and `shrinking_stride` in `ResidualMLP`
to try variants. `shrinking_stride=1` keeps every stage overlapping and shrinks
one pixel per stage. Saved runs record window shapes, strides, and MLP dimensions
alongside model source. Loading an older run restores its older model source.

Each stage is an explicit `RollingWindowBlock` for Torch Canvas inspection.
The refactor preserves checkpoint keys, seeded weights, logits, and gradients;
new runs save its source alongside the residual block and model source.

Run `./run` in this worktree. AdamW is the experiment default; existing CIFAR-10
augmentation settings remain enabled. Automatic leaderboard posting is disabled.
Training starts only when **Train and evaluate** is pressed. Results and run
artifacts stay in this worktree.

To train ImageNet32 from the existing `data/imagenet32/` cache without opening
the dashboard, run `uv run --locked python scripts/run_experiment.py --task imagenet32 --submit`.
This uses the task's configured epoch count (10 by default), seed 42, and Apple
Metal. It saves the same scores, checkpoint, class metrics, and curves under
`runs/`, then submits the selected checkpoint's test score regardless of accuracy.
Omit `--submit` to keep the result local; `--epochs`, `--seed`, and `--device`
override their run settings without changing notebook defaults.

For analysis after a run finishes, `python scripts/continue_experiment.py
runs/<parent-run> --submit` continues its selected weights and exact saved split
in a separate run directory. A weights-only parent restarts optimizer and random
state; this is recorded and is not an uninterrupted training trajectory. Every
new epoch saves `last.pt` with optimizer and random state, allowing an exact
epoch-boundary resume with `--checkpoint last.pt`. `best.pt` remains the checkpoint
with highest validation accuracy. The grading `train()` function still starts
from scratch.

The continuation stops when three consecutive checks show a three-epoch mean
validation loss more than 1% above its minimum and above the preceding window,
while mean training loss falls. It waits six new epochs before its first check,
and records the loss minimum and confirmation epoch in `divergence.json`. The
default guard is 100 total epochs; reaching it without confirmation is reported
as `epoch_guard`, not divergence. Change it with `--max-epochs`.

The first run reached 55.54% clean training, 49.98% validation, and 48.27% test
accuracy. The wider configuration doubles classifier width from 256 to 512 and
block hidden width from 512 to 1024, and widens window outputs from 16 to 64
with window hidden width 128 instead of 32. Strides and training settings stay
fixed. It reached 71.35% clean training, 60.80% validation, and 59.15% test accuracy.

Both runs used the original unpadded shrinking stages. A coverage check found
that their final vector saw only the top-left 17×17 pixels of each 32×32 input:
changing every pixel outside that region left saved-model logits unchanged.
The edge-padding correction keeps the full image connected and adds one window
stage. It reached 88.52% clean training, 76.10% validation, and 75.45% test
accuracy, selecting epoch 51 of 60. The leaderboard confirmed the 75.45% test
submission as TaylorT, section 5000. Archived runs retain their original source,
weights, and metrics.

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
Results append to `results.csv`. With `LEADERBOARD_ENABLED = True`, each completed
app run posts its test accuracy, training duration, and repository URL to the
unofficial class leaderboard as **TaylorT**, section **5000**. Posting is disabled
by default. The result panel reports whether
the form confirmed the submission. Local results are saved before posting;
network failures do not discard them.

Training holds out a seeded random 10% of the supplied training examples for
validation. Each epoch prints training loss and accuracy plus validation loss
and accuracy. Training accuracy counts predictions already computed during
training; validation scores use the weights at the end of the epoch. The model
returned by `train()` uses the epoch with the highest validation accuracy,
keeping the earliest epoch on ties. The app evaluates test accuracy only once,
after selecting that checkpoint. Final training accuracy excludes the holdout.

Each app run creates a unique directory under `runs/` containing:

- `metrics.csv`: epoch, train/validation loss and accuracy, and elapsed seconds.
  Accuracies are fractions from 0 to 1; rows are saved after every epoch.
- `best.pt`: best weights, epoch, validation accuracy, configuration, and exact
  training/validation indices. This is an analysis artifact; `train()` always
  starts from scratch and never loads weights from disk.
- `config.json`: task, seed, model architecture and source, split sizes, and training settings.
- `summary.txt`: the same copyable text shown in the final output, with clean
  train, validation, and test accuracy plus the hyperparameter breakdown.
- `summary.json`: full-precision final scores and the configuration captured
  before training, including model source and per-digit augmentation settings.
  Also includes per-class accuracy for each split.
- `per_class_accuracy.md`: a Markdown table with one row per class and train,
  validation, and test accuracy columns. Scores are percentages from the selected
  checkpoint on clean images. Classes absent from a split show `N/A`.
- `curves.png`: loss and accuracy plots with the selected epoch marked, also
  shown in the app after training completes.
- `leaderboard-submission.json`: submitted fields and confirmation/error status.
  `leaderboard-confirmation.html` stores the server response when available.
  Existing attempts are not retried automatically, avoiding duplicate posts
  when a response is lost. Separate completed runs each receive a submission.

The final output includes a text block you can copy into your experiment journal.
The summary in `results.csv` includes train, validation, and test accuracy,
best epoch, device, seed, configured and completed epochs, optimizer settings,
actual model dimensions and functional operations, split sizes, augmentation,
and the full configuration as JSON. The existing `accuracy` column remains test
accuracy. Older summary rows remain available, with new columns blank; settings
that were not recorded are not inferred from the current model.
Calls through the grading interface return the same selected model and expose
metrics as `model.training_history`, but do not write run files.

The **Per-class accuracy** output is a Markdown table with one row per class:
Class, Train, Validation, and Test. The same metric handles MNIST digits,
CIFAR-10 class names, and ImageNet32 class IDs 0–999. Each `results.csv` row links
to the saved Markdown table; `summary.txt` includes the table for copying into
your journal. `summary.json` records the same columns as fractions from 0 to 1.
Test scores reuse existing predictions; no additional test evaluation is performed.

The default model uses the rolling-window stages described above, backbone
width 512, and two gated residual blocks with hidden width 1024 and dropout 0.1.
Training uses batch size 256 and AdamW with learning rate 0.001, weight decay
`1e-5`, and AMSGrad. CIFAR-10 uses 60 epochs with augmentation enabled; ImageNet32
uses 10 epochs with augmentation disabled. The dashboard starts with seed 42
and selects the best validation checkpoint. Model depth follows the `depth`
constructor argument.

### Reloading previous experiments

Loading restores the saved augmentation settings, including whether augmentation
is enabled. The visible switch, **Next run** status, preview, and training all
read the same controls. Training stops if the visible switch and control state
disagree. Changing the switch after loading changes the next run.

`./run` enables file watching, and the project sets
`runtime.watcher_on_save = "autorun"` so loading settings updates live cells.
`runtime.auto_reload` controls imported modules and does not enable this behavior.
After changing the project configuration, restart an already-running Marimo
server once with `./run`; Marimo caches project configuration at startup.

The **Saved runs** panel lists saved configurations for the selected dataset,
with validation and test scores. Pick a run and click **Load selected settings**,
or click **Load best settings**. **Best by** defaults to validation accuracy;
choose test accuracy to load the highest test-scoring run. Only completed runs
with a recorded score qualify for the shortcut. Ties use the earliest run ID.
Click **Refresh saved runs** after completing another experiment.

Loading updates the notebook's model classes, dimensions, optimizer preset,
augmentation settings, batch size, epochs, validation fraction, seed, and device
preference. The notebook reloads, and Torch Canvas can follow the restored model
code. If the original device is unavailable, the dashboard uses an available
device. Loading does not start training or load checkpoint weights; press
**Run** to train a new model from scratch. New runs record their parent run in
`restored_from`.

Recent runs restore their saved model source. Older runs can recover dimensions
from the saved model representation, including unrolled residual blocks.
Unrecorded activation, optimizer, or augmentation values require an explicit
choice under **Older runs with missing settings**. Recorded values take
precedence. Runs without `config.json` cannot be reconstructed from a score
alone. Unrelated notebook text is preserved; loading stops if the file changes
during restoration.

### Optimizer selection

Choose **SGD**, **Adam**, **AdamW**, **Muon**, **RMSprop**, or **Adagrad** in the
**Optimizer** panel before pressing **Run**. Each optimizer has its own controls
and learning rate preset; edits persist when switching optimizers in the app
session. AdamW is the default for the rolling-window experiment. The selected
settings apply to whichever dataset you run.

| Optimizer | Initial learning rate | Additional controls |
| --- | ---: | --- |
| SGD | 0.05 | Momentum, dampening, Nesterov |
| Adam / AdamW | 0.001 | Betas, epsilon, AMSGrad |
| Muon | 0.02 | Momentum, Nesterov, Newton–Schulz steps, epsilon, learning rate scaling, auxiliary AdamW settings |
| RMSprop | 0.001 | Alpha, momentum, epsilon, centered |
| Adagrad | 0.01 | Learning rate decay, initial accumulator, epsilon |

All presets expose weight decay. SGD starts at `1e-4`; other presets start at
`1e-5`. These are starting points for experiments. SGD's Nesterov option requires positive momentum and zero dampening.

Muon uses native `torch.optim.Muon` and is offered only when the installed
PyTorch provides it. It updates hidden 2D weight matrices. A separate AdamW
optimizer updates the input projection, final output layer, biases, and
normalization parameters; its learning rate, weight decay, betas, and epsilon
have separate controls. Both optimizers step from the same backward pass.
Muon starts with the `original` learning rate scaling; `match_rms_adamw` is
also selectable.

Each completed run records the optimizer name, selected settings, effective
parameter-group settings, and parameter assignments in its configuration and
checkpoint. `results.csv` includes optimizer settings and groups as JSON;
the copyable summary includes their settings and parameter counts.

For calls outside the dashboard, set `OPTIMIZER_NAME` and `OPTIMIZER_DEFAULTS`
in the setup cell. The grading interface `train(task, X_train, y_train, seed,
device)` keeps its existing signature and uses that default. For an individual
programmatic experiment:

```python
from residual_mlp_baseline.notebook import optimizer_config, train_with_metrics

settings = optimizer_config("SGD")
settings.update(lr=0.02, momentum=0.9)
model = train_with_metrics("mnist", X_train, y_train, device="cpu",
                           optimizer_settings=settings)
```

### Dataset-specific augmentation

The augmentation panel follows the dashboard's **Task** selection. MNIST,
CIFAR-10, and ImageNet32 each have an independent toggle (off by default) and
settings. Switching tasks preserves edits for each dataset during the app
session. Press **Run** to train with the selected dataset's configuration.

MNIST keeps one settings tab per digit, 0–9, with rotation, translation, scale,
horizontal shear, brightness, Gaussian noise, and square erasing. No flips are
available for digits. CIFAR-10 and ImageNet32 each use a shared RGB preset across
all classes, with the same transforms plus horizontal flips. Their initial
settings differ: CIFAR-10 uses up to 4 pixels of translation and 10% brightness
variation; ImageNet32 uses 2 pixels and 20%. Both use translation probability
0.5, flip probability 0.5, and brightness probability 0.25. Other RGB transforms
start with probability 0. Enable the dataset's toggle to apply its preset.

Each category has probability and maximum amount controls. Either set to zero
disables it. Rotation and shear use degrees; translation uses pixels per axis.
Scale and brightness sample within `1 ± amount`. Noise samples a standard
deviation up to the amount on the 0–1 intensity scale. Erasing samples a square
up to the chosen fraction of image area. Flip amount 1 enables flipping at its
selected probability. Transforms sample fresh values for each training image.

The **Class preview** picker shows a labeled thumbnail for each class in the
selected dataset. Select a row to compare up to six original and augmented
examples. Search by class name or ID, and page through large datasets such as
ImageNet32. Classes without cached examples show **Not cached**. MNIST uses digit
labels, CIFAR-10 uses class names, and ImageNet32 uses numeric class IDs.

Previews use cached training images and start no downloads. After downloading a
dataset for the first time, click **Refresh cached previews**. Change the preview
seed for fresh augmentation. The preview uses the same configuration as **Run**,
including the dataset's toggle. Only training batches are augmented; validation,
final training accuracy, and test prediction use clean images.

The complete selected configuration is saved in `config.json`, `best.pt`,
`summary.json`, and `results.csv`. Edit `AUGMENTATION_ENABLED` in the setup cell
and `augmentation_categories(task)` to change dataset defaults shared by the app
and grading entry point. `augmentation_config(task)` defines defaults and
per-class overrides; `ImageAugmenter` contains the shared PyTorch transforms.

The app defaults to CIFAR-10. MNIST and ImageNet 32×32 are available in the task
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
Change that value if your submission repository changes. The harness defaults to
`LEADERBOARD_ALIAS = "TaylorT"` and `SECTION = "5000"`; set its alias to an empty
string to disable harness submissions. The app and harness submit their own
completed evaluations; calling the package's `train()` alone does not post.
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
