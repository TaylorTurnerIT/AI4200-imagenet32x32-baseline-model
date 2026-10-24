"""Model, training, and local evaluation in one editable marimo notebook."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import ast
    import csv
    import inspect
    import json
    import os
    import platform
    import sys
    import time
    import textwrap
    import zipfile
    from uuid import uuid4
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset, TensorDataset

    # Each task differs only in its input size and its number of classes.
    # The same model class is used for all three.
    TASKS = {
        "mnist":      {"in_features": 28 * 28,     "num_classes": 10},    # 28x28 grayscale
        "cifar10":    {"in_features": 3 * 32 * 32, "num_classes": 10},    # 32x32 color
        "imagenet32": {"in_features": 3 * 32 * 32, "num_classes": 1000},  # 32x32 color
    }

    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    EPOCHS = {"mnist": 500, "cifar10": 10, "imagenet32": 10}
    WEIGHT_DECAY = 1e-5
    MOMENTUM = 0.0
    DAMPENING = 0.0
    VALIDATION_FRACTION = 0.1

    OPTIMIZER_NAME = "AdamW"
    OPTIMIZER_DEFAULTS = {
        "SGD": {"lr": 0.05, "weight_decay": WEIGHT_DECAY, "momentum": MOMENTUM,
                "dampening": DAMPENING, "nesterov": False},
        "Adam": {"lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                 "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "amsgrad": False},
        "AdamW": {"lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                  "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "amsgrad": False},
        "Muon": {"lr": 0.02, "weight_decay": WEIGHT_DECAY, "momentum": 0.95,
                 "nesterov": True, "ns_steps": 5, "eps": 1e-7, "adjust_lr_fn": "original",
                 "auxiliary_lr": LEARNING_RATE, "auxiliary_weight_decay": WEIGHT_DECAY,
                 "auxiliary_beta1": 0.9, "auxiliary_beta2": 0.999, "auxiliary_eps": 1e-8},
        "RMSprop": {"lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                    "momentum": 0.0, "alpha": 0.99, "eps": 1e-8, "centered": False},
        "Adagrad": {"lr": 0.01, "weight_decay": WEIGHT_DECAY, "lr_decay": 0.0,
                    "initial_accumulator_value": 0.0, "eps": 1e-10},
    }

    # Post each completed app run's test result to the unofficial class leaderboard.
    LEADERBOARD_ENABLED = True
    LEADERBOARD_ALIAS = "TaylorT"
    LEADERBOARD_SECTION = "5000"
    LEADERBOARD_REPOSITORY = "git+https://github.com/TaylorTurnerIT/AI4200-imagenet32x32-baseline-model.git"
    LEADERBOARD_FORM = "https://docs.google.com/forms/d/e/1FAIpQLScZjEdbLFtn1CNvsfwYEbM4YGRxdOU5cyJ7RANM7MrxT0a4oA/viewform"

    # Dataset-specific augmentation defaults. Amounts are maxima, sampled per image.
    # No MNIST flips: they can change a digit's identity.
    AUGMENTATION_ENABLED = {"mnist": False, "cifar10": False, "imagenet32": False}
    AUGMENTATION_CATEGORIES = {
        # name: (label, unit, maximum slider value, step, default probability, default amount)
        "rotation": ("Rotation", "degrees", 30.0, 1.0, 0.5, 10.0),
        "translation": ("Translation", "pixels per axis", 7.0, 0.5, 0.5, 2.0),
        "scale": ("Scale", "fraction around 1", 0.4, 0.01, 0.5, 0.1),
        "shear": ("Horizontal shear", "degrees", 25.0, 1.0, 0.25, 5.0),
        "brightness": ("Brightness", "fraction around 1", 0.5, 0.01, 0.25, 0.1),
        "noise": ("Gaussian noise", "std on 0–1 intensity", 0.3, 0.01, 0.25, 0.05),
        "erasing": ("Random square erasing", "fraction of image area", 0.25, 0.01, 0.1, 0.05),
    }


@app.cell(hide_code=True)
def _(available_devices):
    task = mo.ui.dropdown(["mnist", "cifar10", "imagenet32"], value="mnist", label="Task")
    device = mo.ui.dropdown(available_devices, value=list(available_devices)[-1], label="Device")
    seed = mo.ui.number(start=0, stop=2**32 - 1, value=42, label="Seed")
    run_training = mo.ui.run_button(label="Run")
    mo.vstack([
        mo.hstack([task, device, seed, run_training], justify="start"),
        mo.md(f"Completed runs post test accuracy as **{LEADERBOARD_ALIAS}**, section **{LEADERBOARD_SECTION}**."
              if LEADERBOARD_ENABLED else "Leaderboard posting disabled."),
    ])
    return device, run_training, seed, task


@app.cell(hide_code=True)
def _():
    optimizer_choice = mo.ui.dropdown(available_optimizer_names(), value=OPTIMIZER_NAME, label="Optimizer")

    def _optimizer_control(key, value):
        _labels = {
            "lr": "Learning rate", "weight_decay": "Weight decay", "eps": "Epsilon",
            "beta1": "Beta 1", "beta2": "Beta 2", "lr_decay": "Learning rate decay",
            "ns_steps": "Newton–Schulz steps", "adjust_lr_fn": "Muon learning rate scaling",
        }
        _base_key = key.removeprefix("auxiliary_")
        _label = _labels.get(_base_key, _base_key.replace("_", " ").capitalize())
        if key.startswith("auxiliary_"):
            _label = f"AdamW: {_label}"
        if isinstance(value, bool):
            return mo.ui.switch(value=value, label=_label)
        if key == "adjust_lr_fn":
            return mo.ui.dropdown(["original", "match_rms_adamw"], value=value, label=_label)
        if key == "ns_steps":
            return mo.ui.number(start=1, stop=50, step=1, value=value, label=_label)
        _stop = 0.999999 if _base_key in ("beta1", "beta2") else None
        _step = (value if _base_key == "eps" else
                 1e-5 if _base_key == "weight_decay" else
                 1e-4 if _base_key == "lr" else 0.001)
        return mo.ui.number(start=0, stop=_stop, step=_step, value=value, label=_label)

    # Keep each optimizer's edited values while switching the dropdown.
    optimizer_controls = mo.ui.dictionary({
        name: mo.ui.dictionary({key: _optimizer_control(key, value)
                               for key, value in OPTIMIZER_DEFAULTS[name].items()})
        for name in available_optimizer_names()
    })
    return optimizer_choice, optimizer_controls


@app.cell(hide_code=True)
def _(optimizer_choice, optimizer_controls):
    _note = ("Muon updates hidden matrices; AdamW updates input/output weights, biases, and normalization parameters."
             if optimizer_choice.value == "Muon" else "")
    mo.vstack([
        mo.md("## Optimizer\nChoose an optimizer and adjust its settings before pressing **Run**."),
        optimizer_choice, optimizer_controls[optimizer_choice.value], mo.md(_note),
        mo.md("Muon requires a PyTorch build with `torch.optim.Muon`."
              if "Muon" not in available_optimizer_names() else ""),
    ])
    return


@app.cell(hide_code=True)
def _():
    # Construct all controls once so task switches preserve each dataset's edits.
    def _settings_controls(settings, categories):
        return mo.ui.dictionary({
            category: mo.ui.dictionary({
                "probability": mo.ui.slider(0.0, 1.0, step=0.05,
                    value=settings[category]["probability"], show_value=True,
                    label="Probability per image"),
                "amount": mo.ui.slider(0.0, spec[2], step=spec[3],
                    value=settings[category]["amount"], show_value=True,
                    label=f"Maximum amount ({spec[1]})"),
            }) for category, spec in categories.items()
        })

    augmentation_controls = mo.ui.dictionary({
        name: mo.ui.dictionary({
            "enabled": mo.ui.switch(value=augmentation_config(name)["enabled"],
                                    label=f"Enable {name} augmentation"),
            "default": _settings_controls(augmentation_config(name)["default"], augmentation_categories(name)),
            "classes": mo.ui.dictionary({
                class_id: _settings_controls(settings, augmentation_categories(name))
                for class_id, settings in augmentation_config(name)["classes"].items()
            }),
        }) for name in TASKS
    })
    return (augmentation_controls,)


@app.cell(hide_code=True)
def _():
    preview_seed = mo.ui.number(start=0, stop=2**32 - 1, value=42, label="Preview seed")
    preview_refresh = mo.ui.button(label="Refresh cached previews")
    return preview_refresh, preview_seed


@app.cell(hide_code=True)
def _(DATA_DIR, preview_refresh, task):
    _refresh = preview_refresh.value
    _folder = Path(DATA_DIR) / task.value
    preview_images = None
    preview_labels = np.empty(0, dtype=np.int64)
    if (_folder / "X_train.npy").exists() and (_folder / "y_train.npy").exists():
        preview_images = np.load(_folder / "X_train.npy", mmap_mode="c")
        preview_labels = np.load(_folder / "y_train.npy", mmap_mode="c")
    _initial = int(preview_labels.min()) if len(preview_labels) else 0
    preview_class = mo.ui.table(
        class_preview_rows(task.value, preview_images, preview_labels),
        selection="single", initial_selection=[_initial], pagination=True, page_size=10,
        hidden_columns=["class_id"], show_column_summaries=False, show_data_types=False,
        show_download=False, label="Select a class to preview", max_height=550,
    )
    return preview_class, preview_images, preview_labels


@app.cell(hide_code=True)
def _(
    augmentation_controls,
    preview_class,
    preview_refresh,
    preview_seed,
    task,
):
    _controls = augmentation_controls[task.value]
    _categories = augmentation_categories(task.value)

    def _panel(settings):
        return mo.vstack([
            mo.vstack([mo.md(f"**{spec[0]}**"), settings[category].hstack()])
            for category, spec in _categories.items()
        ])

    _settings = (mo.ui.tabs({f"Digit {digit}": _panel(_controls["classes"][str(digit)])
                            for digit in range(10)})
                 if task.value == "mnist" else _panel(_controls["default"]))
    mo.vstack([
        mo.md(f"""
        ## {task.value} augmentation

        {"Each digit has its own settings. No flips are applied." if task.value == "mnist" else
         "Settings apply to every class in this dataset. Horizontal flips are available."}
        Each category has an independent probability and maximum strength.
        Probability 0 or amount 0 disables a category. Geometry and brightness
        sample within ±amount; noise samples a standard deviation up to its amount.
        Erasing removes a square up to the chosen area. Flip amount 1 enables
        flipping at the selected probability.

        Changes apply on the next **Run** for the selected dataset. Switching
        datasets preserves each dataset's settings for this app session.
        Only training batches are augmented; final scoring uses clean images.
        """),
        _controls["enabled"], _settings,
        mo.md("### Class preview\nSelect a row to compare original and augmented examples. "
              "Search or page through classes. Previews use cached training images; "
              "after the first dataset download, click **Refresh cached previews**."),
        preview_refresh, preview_class, preview_seed,
    ])
    return


@app.cell(hide_code=True)
def _(
    augmentation_controls,
    preview_class,
    preview_images,
    preview_labels,
    preview_seed,
    task,
):
    mo.stop(preview_images is None,
            mo.md(f"Run {task.value} once, then refresh cached previews. No preview downloads are started."))
    mo.stop(not preview_class.value, mo.md("Select a class in the preview table."))
    _selected = preview_class.value[0]
    _images, _labels = preview_images, preview_labels
    _indices = np.flatnonzero(_labels == int(_selected["class_id"]))[:6]
    mo.stop(len(_indices) == 0, mo.md("No cached examples for this class."))
    _original = prepare(_images[_indices])
    _augmenter = ImageAugmenter(task.value, selected_augmentation(task.value, augmentation_controls.value),
                               seed=int(preview_seed.value))
    _augmented = _augmenter(_original, torch.as_tensor(_labels[_indices]).long())
    import matplotlib.pyplot as _plt

    _figure, _axes = _plt.subplots(2, len(_indices), figsize=(9, 3), squeeze=False)
    for _column in range(len(_indices)):
        for _row, _batch in enumerate([_original, _augmented]):
            _image = ((_batch[_column] + 1) / 2).clamp(0, 1)
            if _image.shape[0] == 1:
                _axes[_row, _column].imshow(_image[0].numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                _axes[_row, _column].imshow(_image.permute(1, 2, 0).numpy())
            _axes[_row, _column].set_xticks([])
            _axes[_row, _column].set_yticks([])
    _axes[0, 0].set_ylabel("Original")
    _axes[1, 0].set_ylabel("Augmented")
    _figure.tight_layout()
    _plt.close(_figure)
    mo.vstack([mo.md(f"**{_selected['Class']}** · Original and augmented samples. "
                     "Change the seed for fresh augmentation. The dataset's toggle controls this preview."), _figure])
    return


@app.cell(hide_code=True)
def _(
    NUM_CLASSES,
    RESULTS_FILE,
    augmentation_controls,
    device,
    load_task,
    optimizer_choice,
    optimizer_controls,
    run_training,
    seed,
    task,
):
    mo.stop(not run_training.value, mo.md("Ready. Select a task and start training."))

    with mo.status.spinner(title=f"Loading {task.value} and training on {device.value}..."):
        _X_train, _y_train, _X_test, _y_test = load_task(task.value)
        _start = time.perf_counter()
        _run_dir = RESULTS_FILE.parent / "runs" / f"{task.value}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        _model = train_with_metrics(task.value, _X_train, _y_train, seed=int(seed.value),
                                    device=device.value, run_dir=_run_dir,
                                    augmentation=selected_augmentation(task.value, augmentation_controls.value),
                                    optimizer_settings=selected_optimizer(optimizer_choice.value, optimizer_controls.value))
        _minutes = (time.perf_counter() - _start) / 60
        _probs = torch.as_tensor(predict(_model, _X_test)).float().cpu()
        assert _probs.shape == (len(_y_test), NUM_CLASSES[task.value]), "Unexpected prediction shape"
        assert torch.isfinite(_probs).all(), "Predictions contain non-finite values"
        assert (_probs >= 0).all(), "Predictions contain negative probabilities"
        assert torch.allclose(_probs.sum(1), torch.ones(len(_probs)), atol=1e-3), "Probabilities must sum to 1"
        _accuracy = float(np.mean(_probs.argmax(1).numpy() == _y_test))
        _class_count = NUM_CLASSES[task.value]
        _test_classes = per_class_accuracy(task.value, _y_test, _probs.argmax(1).numpy(), _class_count)
        _train_classes = evaluate_per_class(
            _model, task.value, _X_train, _y_train, _model.training_indices.numpy(), _class_count)
        _validation_classes = evaluate_per_class(
            _model, task.value, _X_train, _y_train, _model.validation_indices.numpy(), _class_count)
        _train_accuracy = sum(row["correct"] for row in _train_classes) / len(_model.training_indices)
        _class_rows = []
        for _split, _rows in (("train", _train_classes), ("validation", _validation_classes), ("test", _test_classes)):
            for _row in _rows:
                _class_rows.append(dict(split=_split, **_row))
        _best_epoch = _model.best_epoch
        _validation_accuracy = _model.best_validation_accuracy
        _history = _model.training_history

    _summary = save_run_result(
        RESULTS_FILE, _run_dir, _model, _accuracy, _train_accuracy, _minutes,
        test_examples=len(_y_test), per_class=_class_rows,
    )
    _summary_text, _, _class_markdown = _summary.partition("\n\nPer-class accuracy:\n\n")

    import matplotlib.pyplot as plt

    _figure, _axes = plt.subplots(1, 2, figsize=(10, 3.5))
    _epochs = [row["epoch"] for row in _history]
    for _name, _label in [("train", "Train"), ("validation", "Validation")]:
        _axes[0].plot(_epochs, [row[f"{_name}_loss"] for row in _history], label=_label)
        _axes[1].plot(_epochs, [100 * row[f"{_name}_accuracy"] for row in _history], label=_label)
    _axes[0].set_ylabel("Cross-entropy loss")
    _axes[1].set_ylabel("Accuracy (%)")
    for _axis in _axes:
        _axis.set_xlabel("Epoch")
        _axis.axvline(_best_epoch, color="gray", linestyle="--", label="Best epoch")
        _axis.legend()
    _figure.tight_layout()
    _figure.savefig(_run_dir / "curves.png", dpi=150)
    plt.close(_figure)

    del _model, _X_train, _y_train, _X_test, _y_test, _probs
    if device.value == "cuda":
        torch.cuda.empty_cache()
    elif device.value == "mps":
        torch.mps.empty_cache()

    with mo.status.spinner(title="Submitting test accuracy to the leaderboard..."):
        _leaderboard_status = submit_run_to_leaderboard(task.value, _accuracy, _minutes, _run_dir)

    mo.vstack([
        mo.md(f"```text\n{_summary_text}\n```"),
        mo.md(
            f"Saved to `results.csv` and `{_run_dir}/summary.txt` / `summary.json`.  \n"
            f"{_leaderboard_status}  \n"
            "Epoch train accuracy uses training batches (augmented when enabled); validation uses clean images and each epoch's final weights."
        ),
        mo.md(f"### Per-class accuracy\n\n```markdown\n{_class_markdown}\n```"),
        _figure,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Residual MLP development

    Edit the model and training functions below, then use the controls above.
    Train your local code on MNIST, CIFAR-10, or ImageNet 32×32.
    Choose a task, then click **Run**. The first run downloads
    the dataset; later runs reuse `data/`. Results append to `results.csv`.
    Each run holds out 10% of training examples for validation, records metrics
    every epoch, and evaluates the best validation checkpoint on the test set once.
    Curves and checkpoints save under `runs/`. Each completed run automatically
    posts its test accuracy to the unofficial class leaderboard. Submission status
    appears with the result; a network failure does not discard local results.

    ImageNet training needs substantial RAM because the baseline converts the
    entire training set to float32. Start with MNIST to check your environment.
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
    def __init__(self, width=256, hidden=512):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, x):
        n = self.norm(x)
        a = F.relu(self.fc1(n))
        h = self.fc2(a)
        return x + h


@app.class_definition
class ResidualMLP(nn.Module):
    """Baseline residual MLP.

    input -> Linear(in_features -> width)            project into the backbone
          -> depth x Block                           residual backbone, stays `width` wide
          -> LayerNorm -> Linear -> GELU -> Linear   head: a block without the skip
          -> logits (num_classes)

    The head has no skip connection because its output size (num_classes)
    differs from its input size (width), so the two cannot be added.
    """

    def __init__(self, in_features, num_classes, width=128, hidden=128, depth=7):
        super().__init__()

        # Project the flattened image down to the backbone width.
        self.input_layer = nn.Linear(in_features, width)
        self.blocks = nn.ModuleList([Block(width, hidden) for _ in range(7)])
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
        x = x.flatten(start_dim=1)
        x = self.input_layer(x)
        for blk in self.blocks:
            x = blk(x)
        n = self.head_norm(x)
        a = F.relu(self.head_fc1(n))
        return self.head_fc2(a)

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
def dataset_class_names(task):
    """Names in the dataset's label order; ImageNet32 retains its numeric IDs."""
    names = {
        "mnist": [str(digit) for digit in range(10)],
        "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    }.get(task)
    return names if names is not None else [f"class {class_id}" for class_id in range(TASKS[task]["num_classes"])]


@app.function
def class_preview_rows(task, images, labels):
    """One cached thumbnail per class, without scanning the full image array."""
    class_ids, indices = np.unique(np.asarray(labels), return_index=True)
    first = dict(zip(class_ids.tolist(), indices.tolist()))
    rows = []
    for class_id, name in enumerate(dataset_class_names(task)):
        thumbnail = "Not cached"
        if class_id in first:
            image = np.asarray(images[first[class_id]])
            image = image[0] if image.shape[0] == 1 else image.transpose(1, 2, 0)
            thumbnail = mo.image(image, alt=f"{name} example", width=56, height=56,
                                 style={"image-rendering": "pixelated"})
        rows.append({"class_id": class_id, "Class": name, "Preview": thumbnail})
    return rows


@app.function
def per_class_accuracy(task, labels, predictions, num_classes=None):
    """Count correct predictions for each true class, including absent classes."""
    num_classes = TASKS[task]["num_classes"] if num_classes is None else num_classes
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.ndim != 1 or predictions.shape != labels.shape:
        raise ValueError("Labels and predictions must be matching one-dimensional arrays")
    for values in (labels, predictions):
        if values.size and (not np.issubdtype(values.dtype, np.integer)
                            or np.any(values < 0) or np.any(values >= num_classes)):
            raise ValueError("Class IDs must be integers between 0 and num_classes - 1")
    labels, predictions = labels.astype(np.int64), predictions.astype(np.int64)
    totals = np.bincount(labels, minlength=num_classes)
    correct = np.bincount(labels[labels == predictions], minlength=num_classes)
    names = dataset_class_names(task)
    return [{"class_id": class_id,
             "class_name": names[class_id],
             "correct": int(correct[class_id]), "total": int(totals[class_id]),
             "accuracy": float(correct[class_id] / totals[class_id]) if totals[class_id] else None}
            for class_id in range(num_classes)]


@app.function
def evaluate_per_class(model, task, images, labels, indices, num_classes=None, chunk=1024):
    """Score a clean split in bounded batches with the shared class metric."""
    predictions = []
    for start in range(0, len(indices), chunk):
        batch = indices[start:start + chunk]
        probabilities = torch.as_tensor(predict(model, images[batch])).cpu()
        predictions.append(probabilities.argmax(1).numpy())
    predicted = np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64)
    return per_class_accuracy(task, np.asarray(labels)[indices], predicted, num_classes)


@app.function
def class_accuracy_columns(per_class):
    """Group split metrics into one row per class, without data counts."""
    classes = {}
    for row in per_class:
        entry = classes.setdefault(row["class_id"], {
            "class_id": row["class_id"], "class_name": row["class_name"],
            "train_accuracy": None, "validation_accuracy": None, "test_accuracy": None,
        })
        entry[f"{row['split']}_accuracy"] = row["accuracy"]
    return [classes[class_id] for class_id in sorted(classes)]


@app.function
def format_class_accuracy(per_class):
    """Render each class's train, validation, and test accuracy as Markdown."""
    lines = ["| Class | Train | Validation | Test |", "| --- | ---: | ---: | ---: |"]
    for row in class_accuracy_columns(per_class):
        name = row["class_name"].replace("|", "&#124;").replace("\n", " ")
        scores = ["N/A" if row[f"{split}_accuracy"] is None else f"{row[f'{split}_accuracy']:.2%}"
                  for split in ("train", "validation", "test")]
        lines.append("| " + " | ".join([name] + scores) + " |")
    return "\n".join(lines)


@app.function
def save_run_result(results_file, run_dir, model, test_accuracy, train_accuracy, train_minutes, test_examples,
                    per_class=None):
    """Save a completed run's scores and the settings captured before training."""
    config = model.run_config
    result = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "package_url": str(results_file.parent), "import_name": "residual_mlp_baseline",
        "task": config["task"], "accuracy": f"{test_accuracy:.4f}",
        "test_accuracy": f"{test_accuracy:.4f}",
        "train_accuracy": f"{train_accuracy:.4f}", "train_minutes": f"{train_minutes:.1f}",
        "seed": config["seed"], "best_epoch": model.best_epoch,
        "validation_accuracy": f"{model.best_validation_accuracy:.4f}",
        "run_dir": str(run_dir.relative_to(results_file.parent)),
        "device": config["device"], "epochs_completed": len(model.training_history),
        **{key: config[key] for key in (
            "epochs", "batch_size", "learning_rate", "weight_decay", "momentum",
            "dampening", "optimizer", "loss", "learning_rate_schedule", "shuffle",
            "validation_fraction", "training_examples", "validation_examples",
        )},
        "test_examples": test_examples,
        "betas": config.get("betas"), "eps": config.get("eps"),
        "optimizer_settings": json.dumps(config.get("optimizer_settings"), sort_keys=True),
        "optimizer_groups": json.dumps(config.get("optimizer_groups"), sort_keys=True),
        **config["architecture"],
        "augmentation": json.dumps(config["augmentation"], sort_keys=True),
        "model": config["model"],
        "hyperparameters": json.dumps(config, sort_keys=True),
    }
    if per_class is not None:
        result["per_class_accuracy_file"] = str((run_dir / "per_class_accuracy.md").relative_to(results_file.parent))
        (run_dir / "per_class_accuracy.md").write_text(format_class_accuracy(per_class) + "\n")
    # Keep historical rows and unknown columns; unavailable old settings stay blank.
    old_rows, header = [], []
    if results_file.exists() and results_file.stat().st_size:
        with results_file.open(newline="") as file:
            reader = csv.DictReader(file)
            header, old_rows = reader.fieldnames or [], list(reader)
    new_header = header + [key for key in result if key not in header]
    if new_header != header:
        with results_file.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=new_header)
            writer.writeheader()
            writer.writerows(old_rows)
    with results_file.open("a", newline="") as file:
        csv.DictWriter(file, fieldnames=new_header).writerow(result)

    # Source captures include functional activations omitted by PyTorch's repr.
    settings = {key: value for key, value in config.items()
                if key not in ("model", "model_source", "block_source", "train_accuracy_kind")}
    if "optimizer_groups" in settings:
        settings["optimizer_groups"] = [{key: value for key, value in group.items() if key != "parameter_names"}
                                         for group in settings["optimizer_groups"]]
    if not config["augmentation"]["enabled"]:
        settings["augmentation"] = {"enabled": False}
    summary = (
        f"Run: {result['run_dir']}\nTime: {result['time']}\nTask: {config['task']}\n"
        f"Train accuracy:      {train_accuracy:.2%} ({config['training_examples']} clean examples)\n"
        f"Validation accuracy: {model.best_validation_accuracy:.2%} ({config['validation_examples']} examples)\n"
        f"Test accuracy:       {test_accuracy:.2%} ({test_examples} examples)\n"
        f"Best epoch: {model.best_epoch} / {len(model.training_history)} completed "
        f"({config['epochs']} configured)\nTraining time: {train_minutes:.2f} minutes\n"
        "All final scores use the best validation checkpoint. Train excludes the holdout.\n\n"
        "Hyperparameters and architecture:\n" + json.dumps(settings, indent=2)
    )
    if per_class is not None:
        summary += "\n\nPer-class accuracy:\n\n" + format_class_accuracy(per_class)
    (run_dir / "summary.txt").write_text(summary + "\n")
    (run_dir / "summary.json").write_text(json.dumps({
        "result": result,
        "performance": {"train_accuracy": train_accuracy,
                        "validation_accuracy": model.best_validation_accuracy,
                        "test_accuracy": test_accuracy, "train_minutes": train_minutes},
        "hyperparameters": config,
        **({"per_class_accuracy": class_accuracy_columns(per_class)} if per_class is not None else {}),
    }, indent=2) + "\n")
    return summary


@app.function
def submit_run_to_leaderboard(task, accuracy, train_minutes, run_dir):
    """Post one completed run, preserving a receipt and avoiding duplicate attempts.

    Network failures leave local training results intact. An ambiguous POST is
    never retried automatically because the form has no idempotency key.
    """
    import re
    import urllib.parse
    import urllib.request

    if not LEADERBOARD_ENABLED:
        return "Leaderboard posting disabled."
    run_dir = Path(run_dir)
    receipt_path = run_dir / "leaderboard-submission.json"
    try:
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text())
            if previous.get("confirmed"):
                return f"Leaderboard submission confirmed as {previous['answers']['alias']} (already submitted)."
            return "Leaderboard submission not confirmed; an earlier attempt exists. Check the run's receipt before retrying."
        if task not in TASKS or not 0 <= accuracy <= 1 or not 0 <= train_minutes < float("inf"):
            raise ValueError("Invalid task, test accuracy, or training duration")
        if not LEADERBOARD_ALIAS.strip() or LEADERBOARD_SECTION not in ("4000", "5000"):
            raise ValueError("Set a leaderboard alias and section 4000 or 5000")
        answers = {
            "alias": LEADERBOARD_ALIAS.strip(), "section": LEADERBOARD_SECTION,
            "task": task, "accuracy": f"{accuracy:.4f}",
            "train minutes": f"{train_minutes:.1f}", "repository": LEADERBOARD_REPOSITORY,
        }
        receipt = {"answers": answers, "confirmed": False, "status": "preparing",
                   "submitted_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        with urllib.request.urlopen(LEADERBOARD_FORM, timeout=20) as response:
            page = response.read().decode("utf-8")
        match = re.search(r"FB_PUBLIC_LOAD_DATA_ = (.*?);\s*</script>", page, re.S)
        if match is None:
            raise ValueError("Could not read leaderboard form fields")
        form = json.loads(match.group(1))
        fields = {question[1].strip().lower(): f"entry.{question[4][0][0]}"
                  for question in form[1][1] if question[4]}
        if not set(answers).issubset(fields):
            raise ValueError("Leaderboard form questions changed")
        payload = urllib.parse.urlencode({fields[name]: value for name, value in answers.items()}).encode()
        receipt["status"] = "posting"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        address = LEADERBOARD_FORM.partition("?")[0].replace("/viewform", "/formResponse")
        request = urllib.request.Request(address, data=payload,
                    headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(request, timeout=20) as response:
            confirmation = response.read().decode("utf-8")
            receipt["http_status"] = response.status
        (run_dir / "leaderboard-confirmation.html").write_text(confirmation)
        receipt["confirmed"] = "Your response has been recorded" in confirmation
        receipt["status"] = "confirmed" if receipt["confirmed"] else "unconfirmed"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        if receipt["confirmed"]:
            return f"Leaderboard submission confirmed as {answers['alias']}, section {answers['section']}."
        return "Leaderboard submission not confirmed. Local results and the server response are saved in the run directory."
    except Exception as error:
        if "receipt" in locals():
            receipt["status"] = "unconfirmed" if receipt["status"] == "posting" else "failed"
            receipt["error"] = str(error)
            try:
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            except OSError:
                pass
        return f"Leaderboard submission not confirmed ({error}). Local results are saved; no automatic retry was sent."


@app.function
def available_optimizer_names():
    """Offer supported presets without requiring optional optimizer packages."""
    return [name for name in OPTIMIZER_DEFAULTS if hasattr(torch.optim, name)]


@app.function
def optimizer_config(name=None):
    """A fresh preset shared by the dashboard and grading entry point."""
    name = OPTIMIZER_NAME if name is None else name
    if name not in OPTIMIZER_DEFAULTS:
        raise ValueError(f"Unknown optimizer: {name}")
    if name not in available_optimizer_names():
        raise ValueError(f"{name} requires a PyTorch build providing torch.optim.{name}")
    return {"name": name, **json.loads(json.dumps(OPTIMIZER_DEFAULTS[name]))}


@app.function
def selected_optimizer(name, configurations):
    """Snapshot the selected optimizer's controls without mixing presets."""
    return {**optimizer_config(name), **json.loads(json.dumps(configurations[name])), "name": name}


@app.function
def build_optimizers(model, settings=None):
    """Build optimizers and record effective settings plus parameter assignments."""
    settings = {} if settings is None else json.loads(json.dumps(settings))
    config = optimizer_config(settings.get("name"))
    if set(settings) - set(config):
        raise ValueError(f"Unknown settings for {config['name']}: {sorted(set(settings) - set(config))}")
    config.update(settings)
    name = config["name"]
    kwargs = {key: value for key, value in config.items() if key != "name"}
    named = [(key, parameter) for key, parameter in model.named_parameters() if parameter.requires_grad]
    if name == "Muon":
        matrices = [parameter for key, parameter in named if parameter.ndim == 2
                    and not key.startswith(("input_layer.", "head_fc2."))]
        matrix_ids = {id(parameter) for parameter in matrices}
        other = [parameter for _, parameter in named if id(parameter) not in matrix_ids]
        if not matrices:
            raise ValueError("Muon needs at least one hidden 2D weight matrix")
        auxiliary = {key.removeprefix("auxiliary_"): kwargs.pop(key)
                     for key in list(kwargs) if key.startswith("auxiliary_")}
        auxiliary["betas"] = (auxiliary.pop("beta1"), auxiliary.pop("beta2"))
        optimizers = [torch.optim.Muon(matrices, **kwargs)]
        if other:
            optimizers.append(torch.optim.AdamW(other, **auxiliary))
    else:
        if name in ("Adam", "AdamW"):
            kwargs["betas"] = (kwargs.pop("beta1"), kwargs.pop("beta2"))
        optimizers = [getattr(torch.optim, name)([parameter for _, parameter in named], **kwargs)]
    parameter_names = {id(parameter): key for key, parameter in named}
    groups = []
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            groups.append({"optimizer": type(optimizer).__name__,
                           **{key: value for key, value in group.items() if key != "params"},
                           "parameter_names": [parameter_names[id(parameter)] for parameter in group["params"]],
                           "parameter_count": sum(parameter.numel() for parameter in group["params"])})
    # Normalize tuples so in-memory configuration agrees with saved JSON.
    return optimizers, config, json.loads(json.dumps(groups))


@app.function
def augmentation_categories(task):
    """Slider ranges and distinct defaults for grayscale digits and RGB images."""
    if task not in TASKS:
        raise ValueError(f"Unknown augmentation task: {task}")
    categories = dict(AUGMENTATION_CATEGORIES)
    if task != "mnist":
        # Color datasets share transforms, but keep independent strengths.
        defaults = {
            "rotation": (0.0, 10.0), "translation": (0.5, 4.0 if task == "cifar10" else 2.0),
            "scale": (0.0, 0.1), "shear": (0.0, 5.0),
            "brightness": (0.25, 0.1 if task == "cifar10" else 0.2),
            "noise": (0.0, 0.05), "erasing": (0.0, 0.05),
        }
        categories = {name: spec[:4] + defaults[name] for name, spec in categories.items()}
        categories["horizontal_flip"] = ("Horizontal flip", "1 = enabled", 1.0, 1.0, 0.5, 1.0)
    return categories


@app.function
def augmentation_config(task):
    """Dataset-tagged defaults shared by dashboard controls and training."""
    settings = {name: {"probability": spec[4], "amount": spec[5]}
                for name, spec in augmentation_categories(task).items()}
    return {
        "task": task, "enabled": AUGMENTATION_ENABLED[task], "default": settings,
        "classes": {str(digit): json.loads(json.dumps(settings)) for digit in range(10)}
                   if task == "mnist" else {},
    }


@app.function
def selected_augmentation(task, configurations):
    """Snapshot only the selected dataset's controls for a run or preview."""
    config = json.loads(json.dumps(configurations[task]))
    if config.get("task", task) != task:
        raise ValueError("Augmentation configuration does not match the selected dataset")
    config["task"] = task
    return config


@app.class_definition
class ImageAugmenter:
    """Fresh dataset-aware transforms on normalized grayscale or RGB CPU batches.

    Uses a private RNG so augmentation does not alter data shuffling or the
    holdout split. Affine transforms share one resampling pass. Black fill is
    -1 in the model's normalized input range. Source images are never mutated.
    """

    def __init__(self, task, config=None, seed=0):
        self.task = task
        self.config = json.loads(json.dumps(augmentation_config(task) if config is None else config))
        if self.config.get("task") != task:
            raise ValueError("Augmentation configuration does not match the selected dataset")
        if not isinstance(self.config.get("enabled"), bool):
            raise ValueError("Augmentation enabled must be a boolean")
        self.num_classes = TASKS[task]["num_classes"]
        self.channels = 1 if task == "mnist" else 3
        categories = augmentation_categories(task)
        self.names = list(categories)
        overrides = self.config["classes"]
        valid_classes = {str(i) for i in range(self.num_classes)}
        if any(class_id not in valid_classes for class_id in overrides):
            raise ValueError("Augmentation override has an invalid class ID")
        values = []
        for class_id in range(self.num_classes):
            row = []
            class_settings = overrides.get(str(class_id), self.config["default"])
            for name, spec in categories.items():
                settings = class_settings[name]
                probability, amount = settings["probability"], settings["amount"]
                if not 0 <= probability <= 1 or not 0 <= amount <= spec[2]:
                    raise ValueError(f"Invalid {name} settings for class {class_id}")
                row.append([probability, amount])
            values.append(row)
        self.parameters = torch.tensor(values, dtype=torch.float32)
        self.generator = torch.Generator().manual_seed(seed)
        self.active = self.config["enabled"] and bool(
            ((self.parameters[:, :, 0] > 0) & (self.parameters[:, :, 1] > 0)).any())

    def __call__(self, images, labels):
        if not self.active:
            return images
        if images.device.type != "cpu" or images.ndim != 4 or images.shape[1] != self.channels:
            raise ValueError(f"{self.task} augmentation expects normalized NCHW CPU images with {self.channels} channels")
        if labels.ndim != 1 or len(labels) != len(images) or bool(((labels < 0) | (labels >= self.num_classes)).any()):
            raise ValueError(f"Augmentation expects one label per image, from 0 to {self.num_classes - 1}")
        count, _, height, width = images.shape
        parameters = self.parameters[labels]
        gates = torch.rand((count, len(self.names)), generator=self.generator) < parameters[:, :, 0]
        strengths = parameters[:, :, 1] * gates
        amounts = {name: strengths[:, i] for i, name in enumerate(self.names)}

        def signed(name):
            return (2 * torch.rand(count, generator=self.generator) - 1) * amounts[name]

        # Invert the sampled forward rotation/shear/scale/translation for affine_grid.
        angle = torch.deg2rad(signed("rotation"))
        shear = torch.tan(torch.deg2rad(signed("shear")))
        scale = 1 + signed("scale")
        cosine, sine = angle.cos(), angle.sin()
        a = (cosine + sine * shear) / scale
        b = (sine - cosine * shear) / scale
        c, d = -sine / scale, cosine / scale
        tx = signed("translation") * (2 / width)
        ty = signed("translation") * (2 / height)
        theta = torch.stack([a, b, -a * tx - b * ty, c, d, -c * tx - d * ty], dim=1).reshape(-1, 2, 3)
        geometric = sum(amounts[name] for name in ("rotation", "translation", "scale", "shear")) > 0
        output = images
        if bool(geometric.any()):
            grid = F.affine_grid(theta, images.shape, align_corners=False)
            warped = F.grid_sample(images + 1, grid, align_corners=False, padding_mode="zeros") - 1
            output = torch.where(geometric[:, None, None, None], warped, images)
        if "horizontal_flip" in amounts:
            flip = amounts["horizontal_flip"] > 0
            output = torch.where(flip[:, None, None, None], output.flip(-1), output)
        brightness = signed("brightness")[:, None, None, None]
        output = (output + 1) * (1 + brightness) - 1
        noise_std = torch.rand(count, generator=self.generator) * amounts["noise"]
        output = output + 2 * noise_std[:, None, None, None] * torch.randn(images.shape, generator=self.generator)
        output = output.clamp(-1, 1)

        area = torch.rand(count, generator=self.generator) * amounts["erasing"]
        side = (area.sqrt() * min(height, width)).floor().long()
        top = (torch.rand(count, generator=self.generator) * (height - side + 1)).floor().long()
        left = (torch.rand(count, generator=self.generator) * (width - side + 1)).floor().long()
        rows = torch.arange(height)[None, :, None]
        cols = torch.arange(width)[None, None, :]
        mask = ((rows >= top[:, None, None]) & (rows < (top + side)[:, None, None]) &
                (cols >= left[:, None, None]) & (cols < (left + side)[:, None, None]))
        output = output.masked_fill(mask[:, None], -1)
        # Preserve exact inputs for classes whose categories are all disabled.
        changed = (strengths > 0).any(dim=1)
        return torch.where(changed[:, None, None, None], output, images)


@app.function
def train(task, X_train, y_train, seed=0, device="cuda"):
    """Train from scratch and return the best validation checkpoint's model."""
    return train_with_metrics(task, X_train, y_train, seed=seed, device=device)


@app.function
def train_with_metrics(task, X_train, y_train, seed=0, device="cuda", run_dir=None, augmentation=None,
                       optimizer_settings=None):
    """Train with a seeded holdout; optionally persist each epoch and best weights.

    No test data enters training or checkpoint selection. Tied validation
    accuracies keep the earlier epoch. The grading entry point remains train().
    """
    torch.manual_seed(seed)                   # makes the run reproducible
    if len(X_train) != len(y_train) or len(X_train) < 2:
        raise ValueError("Training needs at least two examples and matching labels")
    if not 0 < VALIDATION_FRACTION < 1 or EPOCHS[task] < 1:
        raise ValueError("Use a validation fraction between 0 and 1 and at least one epoch")

    augmentation = augmentation_config(task) if augmentation is None else augmentation
    augmenter = ImageAugmenter(task, augmentation, seed=seed + 1)

    # 1. Build the model for this task. Only input size and class count differ.
    model = ResidualMLP(**TASKS[task]).to(device)

    # 2. Wrap the data so we can loop over shuffled mini-batches.
    dataset = TensorDataset(prepare(X_train), torch.as_tensor(y_train).long())
    split = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    validation_size = min(len(dataset) - 1, max(1, int(len(dataset) * VALIDATION_FRACTION)))
    validation_indices, training_indices = split[:validation_size], split[validation_size:]
    loader = DataLoader(Subset(dataset, training_indices.tolist()), batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(Subset(dataset, validation_indices.tolist()), batch_size=BATCH_SIZE)

    # 3. Choose the loss and the optimizer.
    loss_fn = nn.CrossEntropyLoss()           # expects raw logits
    optimizers, optimizer_settings, optimizer_groups = build_optimizers(model, optimizer_settings)
    functional_operations = set()
    for cls in (type(model), Block):
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(cls)))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "F"):
                functional_operations.add(node.func.attr)
    metadata = {
        "task": task, "seed": seed, "device": str(device), "epochs": EPOCHS[task],
        "batch_size": BATCH_SIZE, "learning_rate": optimizer_groups[0]["lr"],
        "weight_decay": optimizer_groups[0]["weight_decay"],
        "momentum": optimizer_groups[0].get("momentum"),
        "dampening": optimizer_groups[0].get("dampening"),
        "betas": optimizer_groups[0].get("betas"),
        "eps": optimizer_groups[0].get("eps"),
        "optimizer": optimizer_settings["name"], "loss": repr(loss_fn),
        "optimizer_settings": optimizer_settings, "optimizer_groups": optimizer_groups,
        "learning_rate_schedule": "constant", "shuffle": True,
        "architecture": {
            "width": model.input_layer.out_features,
            "hidden": model.head_fc1.out_features,
            "depth": len(model.blocks),
            "functional_operations": sorted(functional_operations),
            "in_features": model.input_layer.in_features,
            "num_classes": model.head_fc2.out_features,
            "normalization": sorted({type(layer).__name__ for layer in model.modules()
                                     if "Norm" in type(layer).__name__}),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "model_source": inspect.getsource(type(model)),
        "block_source": inspect.getsource(Block),
        "validation_fraction": VALIDATION_FRACTION,
        "training_examples": len(training_indices), "validation_examples": validation_size,
        "selection_metric": "validation_accuracy", "tie_break": "earliest_epoch",
        "model": repr(model),
        "augmentation": augmenter.config,
        "train_accuracy_kind": "training-batch predictions (augmented when enabled), before each batch update",
    }
    fields = ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "elapsed_seconds"]
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with (run_dir / "metrics.csv").open("w", newline="") as file:
            csv.DictWriter(file, fieldnames=fields).writeheader()

    # 4. The training loop.
    history = []
    best_accuracy, best_epoch, best_state = -1.0, 0, None
    started = time.perf_counter()
    for epoch in range(EPOCHS[task]):
        model.train()
        total_loss = torch.zeros((), device=device)
        correct = torch.zeros((), dtype=torch.long, device=device)
        for X_batch, y_batch in loader:
            X_batch = augmenter(X_batch, y_batch)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)           # forward pass
            loss = loss_fn(logits, y_batch)   # how wrong are we?

            for optimizer in optimizers:
                optimizer.zero_grad()             # clear old gradients
            loss.backward()                   # backward pass: compute gradients
            for optimizer in optimizers:
                optimizer.step()                  # update the weights

            total_loss += loss.detach() * len(X_batch)
            correct += (logits.detach().argmax(1) == y_batch).sum()

        model.eval()
        validation_loss = torch.zeros((), device=device)
        validation_correct = torch.zeros((), dtype=torch.long, device=device)
        with torch.inference_mode():
            for X_batch, y_batch in validation_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                validation_loss += loss_fn(logits, y_batch) * len(X_batch)
                validation_correct += (logits.argmax(1) == y_batch).sum()
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss.item() / len(training_indices),
            "train_accuracy": correct.item() / len(training_indices),
            "validation_loss": validation_loss.item() / validation_size,
            "validation_accuracy": validation_correct.item() / validation_size,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        if row["validation_accuracy"] > best_accuracy:
            best_accuracy, best_epoch = row["validation_accuracy"], epoch + 1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if run_dir is not None:
                torch.save({"state_dict": best_state, "epoch": best_epoch,
                            "validation_accuracy": best_accuracy, "config": metadata,
                            "training_indices": training_indices, "validation_indices": validation_indices},
                           run_dir / "best.pt.partial")
                (run_dir / "best.pt.partial").replace(run_dir / "best.pt")
        if run_dir is not None:
            with (run_dir / "metrics.csv").open("a", newline="") as file:
                csv.DictWriter(file, fieldnames=fields).writerow(row)
        print(f"epoch {epoch + 1}: train loss {row['train_loss']:.4f}, "
              f"train accuracy {row['train_accuracy']:.2%}, "
              f"validation loss {row['validation_loss']:.4f}, "
              f"validation accuracy {row['validation_accuracy']:.2%} (best epoch {best_epoch})",
              flush=True)

    # Restore weights from this run's memory, never from a previous run's saved file.
    model.load_state_dict(best_state)
    model.eval()
    model.run_config = metadata
    model.training_history = history
    model.best_epoch = best_epoch
    model.best_validation_accuracy = best_accuracy
    model.training_indices = training_indices
    model.validation_indices = validation_indices
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
