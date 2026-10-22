"""Model, training, and local evaluation in one editable marimo notebook."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import csv
    import os
    import platform
    import sys
    import time
    import zipfile
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Each task differs only in its input size and its number of classes.
    # The same model class is used for all three.
    TASKS = {
        "mnist":      {"in_features": 28 * 28,     "num_classes": 10},    # 28x28 grayscale
        "cifar10":    {"in_features": 3 * 32 * 32, "num_classes": 10},    # 32x32 color
        "imagenet32": {"in_features": 3 * 32 * 32, "num_classes": 1000},  # 32x32 color
    }


    # Baseline settings. Deliberately plain so they are easy to beat:
    # plain SGD, constant learning rate, no dropout, no schedule, no augmentation.
    LEARNING_RATE = 0.05
    BATCH_SIZE = 256
    EPOCHS = {"mnist": 100, "cifar10": 30, "imagenet32": 10}
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.0
    DAMPENING = 0.0


@app.cell
def _(available_devices):
    task = mo.ui.dropdown(["mnist", "cifar10", "imagenet32"], value="mnist", label="Task")
    device = mo.ui.dropdown(available_devices, value=list(available_devices)[-1], label="Device")
    seed = mo.ui.number(start=0, stop=2**32 - 1, value=42, label="Seed")
    run_training = mo.ui.run_button(label="Train and evaluate")
    mo.hstack([task, device, seed, run_training], justify="start")
    return device, run_training, seed, task


@app.cell
def _(
    NUM_CLASSES,
    RESULTS_FILE,
    device,
    load_task,
    predict,
    run_training,
    seed,
    task,
    train,
):
    mo.stop(not run_training.value, mo.md("Ready. Select a task and start training."))

    def _accuracy_on(model, X, y, chunk=1024):
        """Accuracy of the model on (X, y), scored in chunks to save memory."""
        correct = 0
        for start in range(0, len(X), chunk):
            probs = torch.as_tensor(predict(model, X[start:start + chunk])).float().cpu()
            correct += int((probs.argmax(1).numpy() == np.asarray(y[start:start + chunk])).sum())
        return correct / len(X)

    with mo.status.spinner(title=f"Loading {task.value} and training on {device.value}..."):
        _X_train, _y_train, _X_test, _y_test = load_task(task.value)
        _start = time.perf_counter()
        _model = train(task.value, _X_train, _y_train, seed=int(seed.value), device=device.value)
        _minutes = (time.perf_counter() - _start) / 60
        _probs = torch.as_tensor(predict(_model, _X_test)).float().cpu()
        assert _probs.shape == (len(_y_test), NUM_CLASSES[task.value]), "Unexpected prediction shape"
        assert torch.isfinite(_probs).all(), "Predictions contain non-finite values"
        assert (_probs >= 0).all(), "Predictions contain negative probabilities"
        assert torch.allclose(_probs.sum(1), torch.ones(len(_probs)), atol=1e-3), "Probabilities must sum to 1"
        _accuracy = float(np.mean(_probs.argmax(1).numpy() == _y_test))
        _train_accuracy = _accuracy_on(_model, _X_train, _y_train)

    _header = ["time", "package_url", "import_name", "task", "accuracy", "train_accuracy", "train_minutes", "seed"]
    _new_file = not RESULTS_FILE.exists() or RESULTS_FILE.stat().st_size == 0
    if not _new_file:
        # Older results.csv files have no train_accuracy column; rewrite them so
        # every row has the same number of columns.
        with RESULTS_FILE.open(newline="") as _file:
            _old_rows = list(csv.reader(_file))
        _old_header = _old_rows[0] if _old_rows else []
        if "train_accuracy" not in _old_header:
            _missing = [c for c in _header if c not in _old_header]
            with RESULTS_FILE.open("w", newline="") as _file:
                _writer = csv.writer(_file)
                _writer.writerow(_old_header + _missing)
                for _row in _old_rows[1:]:
                    _writer.writerow(_row + [""] * len(_missing))
    with RESULTS_FILE.open("a", newline="") as _file:
        _writer = csv.writer(_file)
        if _new_file:
            _writer.writerow(_header)
        _writer.writerow([time.strftime("%Y-%m-%d %H:%M"), str(RESULTS_FILE.parent),
                          "residual_mlp_baseline", task.value, f"{_accuracy:.4f}",
                          f"{_train_accuracy:.4f}", f"{_minutes:.1f}", int(seed.value)])

    del _model, _X_train, _y_train, _X_test, _y_test, _probs
    if device.value == "cuda":
        torch.cuda.empty_cache()
    elif device.value == "mps":
        torch.mps.empty_cache()

    mo.md(
        f"**{task.value}: {_accuracy:.2%} test accuracy** · "
        f"**{_train_accuracy:.2%} train accuracy** · {_minutes:.1f} minutes · {device.value}  \n"
        f"Train-test gap: {(_train_accuracy - _accuracy):.2%}  \nSaved to `results.csv`."
    )
    return


@app.cell
def _():
    mo.md("""
    # Residual MLP development

    Edit the model and training functions below, then use the controls above.
    Train your local code on MNIST, CIFAR-10, or ImageNet 32×32.
    Choose a task, then click **Train and evaluate**. The first run downloads
    the dataset; later runs reuse `data/`. Results append to `results.csv`.
    No leaderboard results are posted.

    ImageNet training needs substantial RAM because the baseline converts the
    entire training set to float32. Start with MNIST to check your environment.
    """)
    return


@app.cell
def _():
    available_devices = {"CPU": "cpu"}
    _metal_status = "Unavailable"
    if torch.backends.mps.is_available():
        # Verify real Metal execution before offering MPS as a training device.
        _probe = torch.ones((4, 4), device="mps", requires_grad=True)
        (_probe @ _probe).sum().backward()
        torch.mps.synchronize()
        del _probe
        available_devices["Apple Metal (MPS)"] = "mps"
        _metal_status = "Ready — GPU forward/backward check passed"
    if torch.cuda.is_available():
        available_devices["NVIDIA CUDA"] = "cuda"
    mo.md(f"""
    **Apple Metal:** {_metal_status}  
    **PyTorch:** {torch.__version__} · **Architecture:** {platform.machine()}  
    **Python:** `{sys.executable}`

    PyTorch calls Apple's Metal backend **MPS**. CUDA is NVIDIA-specific;
    an unavailable CUDA backend does not mean Metal is unavailable.
    """)
    return (available_devices,)


@app.cell
def _():
    mo.md("""
    ## Model and training

    Edit task sizes and training settings in the setup cell. The classes and
    functions below are also used by the package and grading harness.
    Keep the `train(task, X_train, y_train, seed=0, device="cuda")` and
    `predict(model, X)` interfaces unchanged for grading.
    """)
    return


@app.class_definition
class Block(nn.Module):
    """One residual block:  x + Linear(tanh(Linear(LayerNorm(x))))

    The block expands to `hidden` features, applies tanh, and projects back to
    `width` features so its output can be added to its input (the skip connection).
    """

    def __init__(self, width, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(width)       # pre-norm: normalize before the layers
        self.fc1 = nn.Linear(width, hidden)   # expand:  width  -> hidden
        self.fc2 = nn.Linear(hidden, width)   # project: hidden -> width

    def forward(self, x):
        h = self.fc2(torch.tanh(self.fc1(self.norm(x))))
        return x + h


@app.class_definition
class ResidualMLP(nn.Module):
    """Baseline residual MLP.

    input -> Linear(in_features -> width)            project into the backbone
          -> depth x Block                           residual backbone, stays `width` wide
          -> LayerNorm -> Linear -> tanh -> Linear   head: a block without the skip
          -> logits (num_classes)

    The head has no skip connection because its output size (num_classes)
    differs from its input size (width), so the two cannot be added.
    """

    def __init__(self, in_features, num_classes, width=256, hidden=512, depth=7):
        super().__init__()

        # Project the flattened image down to the backbone width.
        self.input_layer = nn.Linear(in_features, width)

        # The residual backbone: `depth` identical blocks.
        self.blocks = nn.ModuleList([Block(width, hidden) for _ in range(depth)])

        # The head: same layers as a block, but the last layer outputs class scores.
        self.head_norm = nn.LayerNorm(width)
        self.head_fc1 = nn.Linear(width, hidden)
        self.head_fc2 = nn.Linear(hidden, num_classes)

        # Xavier (Glorot) initialization for every linear layer, zero biases.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Training forward pass. Returns raw logits (no softmax).

        nn.CrossEntropyLoss applies the softmax itself, so training uses logits.
        """
        x = x.flatten(start_dim=1)            # (N, C, H, W) -> (N, C*H*W)
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.head_fc2(torch.tanh(self.head_fc1(self.head_norm(x))))
        return x

    def predict(self, x):
        """Inference pass, never used during training. Returns class probabilities."""
        self.eval()                           # switch layers to inference behavior
        with torch.no_grad():                 # no gradients needed at inference
            return torch.softmax(self.forward(x), dim=1)



@app.function
def prepare(X):
    """Convert uint8 images (0-255) to floats in roughly [-1, 1]."""
    return (torch.as_tensor(X).float() / 255.0 - 0.5) / 0.5


@app.function
def train(task, X_train, y_train, seed=0, device="cuda"):
    """Build a new model and train it from scratch. Returns the trained model."""
    torch.manual_seed(seed)                   # makes the run reproducible

    # 1. Build the model for this task. Only input size and class count differ.
    model = ResidualMLP(**TASKS[task]).to(device)

    # 2. Wrap the data so we can loop over shuffled mini-batches.
    dataset = TensorDataset(prepare(X_train), torch.as_tensor(y_train).long())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Choose the loss and the optimizer.
    loss_fn = nn.CrossEntropyLoss()           # expects raw logits
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)

    # 4. The training loop.
    model.train()
    for epoch in range(EPOCHS[task]):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)           # forward pass
            loss = loss_fn(logits, y_batch)   # how wrong are we?

            optimizer.zero_grad()             # clear old gradients
            loss.backward()                   # backward pass: compute gradients
            optimizer.step()                  # update the weights

            total_loss += loss.item() * len(X_batch)

        print(f"epoch {epoch + 1}: train loss {total_loss / len(dataset):.4f}")

    return model


@app.function
def predict(model, X):
    """Return class probabilities for X, one row per example, on the CPU."""
    device = next(model.parameters()).device
    X = prepare(X)
    # Predict in chunks so a large test set does not run out of GPU memory.
    chunks = [model.predict(X[i:i + 1024].to(device)).cpu() for i in range(0, len(X), 1024)]
    return torch.cat(chunks)



@app.cell
def _():
    # Resolve caches relative to the app, even if launched from another directory.
    DATA_DIR = str(Path(__file__).resolve().parents[2] / "data")
    RESULTS_FILE = Path(__file__).resolve().parents[2] / "results.csv"
    return DATA_DIR, RESULTS_FILE


@app.cell
def _(DATA_DIR):
    # For each task: Hugging Face dataset name, image column, train split, test split.
    SOURCES = {
        "mnist":      ("ylecun/mnist",                     "image", "train", "test"),
        "cifar10":    ("uoft-cs/cifar10",                  "img",   "train", "test"),
        "imagenet32": ("benjamin-paine/imagenet-1k-32x32", "image", "train", "validation"),
    }
    NUM_CLASSES = {"mnist": 10, "cifar10": 10, "imagenet32": 1000}


    def save_split(split, image_column, mode, folder, name):
        """Write one Hugging Face split to X_<name>.npy (N, C, H, W) and y_<name>.npy.

        The images are written straight into a file on disk, 10,000 at a time, so
        even ImageNet (about 4 GB) never has to fit in memory all at once.
        """
        n = len(split)
        height, width = np.asarray(split[0][image_column]).shape[:2]
        channels = 1 if mode == "L" else 3
        X = np.lib.format.open_memmap(os.path.join(folder, f"X_{name}.npy"), mode="w+",
                                      dtype=np.uint8, shape=(n, channels, height, width))
        y = np.empty(n, dtype=np.int64)
        for start in range(0, n, 10_000):
            chunk = split[start:start + 10_000]
            # mode is "L" (grayscale) or "RGB"; convert() fixes any odd image formats.
            images = np.stack([np.asarray(img.convert(mode)) for img in chunk[image_column]])
            stop = start + len(images)
            X[start:stop] = images[:, None] if mode == "L" else images.transpose(0, 3, 1, 2)
            y[start:stop] = chunk["label"]
            print(f"  {name}: {stop:,} / {n:,} images", end="\r")
        print()
        X.flush()
        np.save(os.path.join(folder, f"y_{name}.npy"), y)


    def load_task(task):
        """Return X_train, y_train, X_test, y_test, downloading only if there is no saved copy.

        The arrays are memory-mapped: they are read from disk as they are used
        instead of all being loaded into RAM up front.
        """
        folder = os.path.join(DATA_DIR, task)
        old_file = os.path.join(DATA_DIR, f"{task}.npz")   # saved by an earlier version of this notebook

        if not os.path.isdir(folder) and os.path.exists(old_file):
            # Unpack the earlier single-file copy into the folder format; no download needed.
            print(f"{task}: unpacking {old_file} (one time only; you can delete it afterwards)")
            temp_folder = folder + ".partial"
            os.makedirs(temp_folder, exist_ok=True)
            # An .npz file is a zip of .npy files, so extracting it streams them to disk
            # without loading the arrays into memory.
            with zipfile.ZipFile(old_file) as old:
                old.extractall(temp_folder)
            os.replace(temp_folder, folder)

        if not os.path.isdir(folder):                    # no saved copy yet: download it
            from datasets import load_dataset
            name, image_column, train_split, test_split = SOURCES[task]
            mode = "L" if task == "mnist" else "RGB"
            print(f"{task}: getting {name} from Hugging Face (first run only; reuses its download cache if present)")

            # Write into a temporary folder, then rename it. An interrupted run then
            # never leaves a half-written folder that looks like a finished one.
            temp_folder = folder + ".partial"
            os.makedirs(temp_folder, exist_ok=True)
            save_split(load_dataset(name, split=train_split), image_column, mode, temp_folder, "train")
            save_split(load_dataset(name, split=test_split), image_column, mode, temp_folder, "test")
            os.replace(temp_folder, folder)

        # mmap_mode="c" (copy-on-write): read from disk on demand, and any changes
        # your code makes to the arrays stay in memory and never touch the files.
        return tuple(np.load(os.path.join(folder, f"{part}.npy"), mmap_mode="c")
                     for part in ["X_train", "y_train", "X_test", "y_test"])

    return NUM_CLASSES, load_task


if __name__ == "__main__":
    app.run()
