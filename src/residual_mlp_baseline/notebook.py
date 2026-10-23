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


    # Baseline settings. Deliberately plain so they are easy to beat:
    # plain SGD, constant learning rate, no dropout, no schedule, no augmentation.
    LEARNING_RATE = 0.05
    BATCH_SIZE = 128
    EPOCHS = {"mnist": 500, "cifar10": 30, "imagenet32": 10}
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.0
    DAMPENING = 0.0
    VALIDATION_FRACTION = 0.1

    # Post each completed app run's test result to the unofficial class leaderboard.
    LEADERBOARD_ENABLED = True
    LEADERBOARD_ALIAS = "TaylorT"
    LEADERBOARD_SECTION = "5000"
    LEADERBOARD_REPOSITORY = "git+https://github.com/TaylorTurnerIT/AI4200-imagenet32x32-baseline-model.git"
    LEADERBOARD_FORM = "https://docs.google.com/forms/d/e/1FAIpQLScZjEdbLFtn1CNvsfwYEbM4YGRxdOU5cyJ7RANM7MrxT0a4oA/viewform"

    # MNIST-only augmentation defaults. Amounts are maxima, sampled per image.
    # No flips: they can change a digit's identity.
    DIGIT_AUGMENTATION_ENABLED = False
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
    augmentation_enabled = mo.ui.switch(value=DIGIT_AUGMENTATION_ENABLED, label="Enable MNIST augmentation")
    preview_digit = mo.ui.dropdown([str(digit) for digit in range(10)], value="0", label="Preview digit")
    preview_seed = mo.ui.number(start=0, stop=2**32 - 1, value=42, label="Preview seed")
    _defaults = digit_augmentation_config()["digits"]
    augmentation_digits = mo.ui.dictionary({
        str(digit): mo.ui.dictionary({
            category: mo.ui.dictionary({
                "probability": mo.ui.slider(0.0, 1.0, step=0.05,
                    value=_defaults[str(digit)][category]["probability"], show_value=True,
                    label="Probability per image"),
                "amount": mo.ui.slider(0.0, spec[2], step=spec[3],
                    value=_defaults[str(digit)][category]["amount"], show_value=True,
                    label=f"Maximum amount ({spec[1]})"),
            }) for category, spec in AUGMENTATION_CATEGORIES.items()
        }) for digit in range(10)
    })
    mo.vstack([
        mo.md("""
        ## Digit augmentation

        Enable augmentation, then choose a digit tab to tune its categories.
        Each category has an independent probability and maximum strength.
        Probability 0 or amount 0 disables that category for that digit.
        Geometry and brightness sample within ±amount; noise samples a standard
        deviation up to its amount. Erasing removes a square up to the chosen area.
        Changes take effect on the next **Run**. Augmentation applies only to MNIST
        training batches; validation and test images stay unchanged.
        Large transforms can make digits ambiguous. No flips are applied.
        """),
        augmentation_enabled,
        mo.ui.tabs({f"Digit {digit}": mo.vstack([
            mo.vstack([mo.md(f"**{spec[0]}**"),
                       augmentation_digits[str(digit)][category].hstack()])
            for category, spec in AUGMENTATION_CATEGORIES.items()
        ]) for digit in range(10)}),
        mo.hstack([preview_digit, preview_seed], justify="start"),
    ])
    return (
        augmentation_digits,
        augmentation_enabled,
        preview_digit,
        preview_seed,
    )


@app.cell(hide_code=True)
def _(
    DATA_DIR,
    augmentation_digits,
    augmentation_enabled,
    preview_digit,
    preview_seed,
):
    _folder = Path(DATA_DIR) / "mnist"
    mo.stop(not (_folder / "X_train.npy").exists() or not (_folder / "y_train.npy").exists(),
            mo.md("Run MNIST once to cache images for the augmentation preview. No preview downloads are started."))
    _images = np.load(_folder / "X_train.npy", mmap_mode="c")
    _labels = np.load(_folder / "y_train.npy", mmap_mode="c")
    _indices = np.flatnonzero(_labels == int(preview_digit.value))[:6]
    mo.stop(len(_indices) == 0, mo.md("No cached examples for this digit."))
    _original = prepare(_images[_indices])
    _augmenter = DigitAugmenter({"enabled": augmentation_enabled.value,
                                "digits": augmentation_digits.value}, seed=int(preview_seed.value))
    _augmented = _augmenter(_original, torch.as_tensor(_labels[_indices]).long())
    import matplotlib.pyplot as _plt

    _figure, _axes = _plt.subplots(2, len(_indices), figsize=(9, 3), squeeze=False)
    for _column in range(len(_indices)):
        for _row, _batch in enumerate([_original, _augmented]):
            _axes[_row, _column].imshow(_batch[_column, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
            _axes[_row, _column].set_xticks([])
            _axes[_row, _column].set_yticks([])
    _axes[0, 0].set_ylabel("Original")
    _axes[1, 0].set_ylabel("Augmented")
    _figure.tight_layout()
    _plt.close(_figure)
    mo.vstack([mo.md("Change the preview seed for fresh samples. The master toggle also controls this preview."), _figure])
    return


@app.cell(hide_code=True)
def _(
    NUM_CLASSES,
    RESULTS_FILE,
    augmentation_digits,
    augmentation_enabled,
    device,
    load_task,
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
                                    augmentation={"enabled": augmentation_enabled.value,
                                                  "digits": augmentation_digits.value})
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
        _class_rows = [dict(split=split, **row)
                       for split, rows in (("train", _train_classes), ("validation", _validation_classes),
                                           ("test", _test_classes)) for row in rows]
        _best_epoch = _model.best_epoch
        _validation_accuracy = _model.best_validation_accuracy
        _history = _model.training_history

    _summary = save_run_result(
        RESULTS_FILE, _run_dir, _model, _accuracy, _train_accuracy, _minutes,
        test_examples=len(_y_test), per_class=_class_rows,
    )

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
        mo.md(f"```text\n{_summary}\n```"),
        mo.md(
            f"Saved to `results.csv` and `{_run_dir}/summary.txt` / `summary.json`.  \n"
            f"{_leaderboard_status}  \n"
            "Epoch train accuracy uses training batches (augmented when enabled); validation uses clean images and each epoch's final weights."
        ),
        mo.md("### Per-class accuracy\n"
              "Correct predictions / examples with that true class (recall). "
              "All splits use clean images and the selected checkpoint. "
              "Classes absent from a split show a blank accuracy. "
              "Search, sort, or download the table as CSV."),
        mo.ui.table(
            [{**{key: value for key, value in row.items() if key != "accuracy"},
              "accuracy_percent": None if row["accuracy"] is None else round(100 * row["accuracy"], 2)}
             for row in _class_rows],
            pagination=True, page_size=20, selection=None, label="Per-class accuracy",
        ),
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
    names = {
        "mnist": [str(digit) for digit in range(10)],
        "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    }.get(task)
    return [{"class_id": class_id,
             "class_name": names[class_id] if names is not None else f"class {class_id}",
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
        **config["architecture"],
        "augmentation": json.dumps(config["augmentation"], sort_keys=True),
        "model": config["model"],
        "hyperparameters": json.dumps(config, sort_keys=True),
    }
    if per_class is not None:
        result["per_class_accuracy_file"] = str((run_dir / "per_class_accuracy.csv").relative_to(results_file.parent))
        with (run_dir / "per_class_accuracy.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["split", "class_id", "class_name", "correct", "total", "accuracy"])
            writer.writeheader()
            writer.writerows(per_class)
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
        summary += f"\n\nPer-class accuracy (all splits): {result['per_class_accuracy_file']}"
    (run_dir / "summary.txt").write_text(summary + "\n")
    (run_dir / "summary.json").write_text(json.dumps({
        "result": result,
        "performance": {"train_accuracy": train_accuracy,
                        "validation_accuracy": model.best_validation_accuracy,
                        "test_accuracy": test_accuracy, "train_minutes": train_minutes},
        "hyperparameters": config,
        **({"per_class_accuracy": per_class} if per_class is not None else {}),
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
def digit_augmentation_config():
    """Default settings shared by the notebook controls and grading entry point."""
    return {
        "enabled": DIGIT_AUGMENTATION_ENABLED,
        "digits": {str(digit): {
            name: {"probability": spec[4], "amount": spec[5]}
            for name, spec in AUGMENTATION_CATEGORIES.items()
        } for digit in range(10)},
    }


@app.class_definition
class DigitAugmenter:
    """Fresh, label-specific MNIST transforms on normalized CPU training batches.

    Uses a private RNG so augmentation does not alter data shuffling or the
    holdout split. Affine transforms share one resampling pass. Black fill is
    -1 in the model's normalized input range. Source images are never mutated.
    """

    def __init__(self, config, seed):
        self.config = json.loads(json.dumps(config))
        if not isinstance(self.config.get("enabled"), bool):
            raise ValueError("Augmentation enabled must be a boolean")
        self.names = list(AUGMENTATION_CATEGORIES)
        values = []
        for digit in range(10):
            row = []
            for name, spec in AUGMENTATION_CATEGORIES.items():
                settings = self.config["digits"][str(digit)][name]
                probability, amount = settings["probability"], settings["amount"]
                if not 0 <= probability <= 1 or not 0 <= amount <= spec[2]:
                    raise ValueError(f"Invalid {name} settings for digit {digit}")
                row.append([probability, amount])
            values.append(row)
        self.parameters = torch.tensor(values, dtype=torch.float32)
        self.generator = torch.Generator().manual_seed(seed)
        self.active = self.config["enabled"] and bool(
            ((self.parameters[:, :, 0] > 0) & (self.parameters[:, :, 1] > 0)).any())

    def __call__(self, images, labels):
        if not self.active:
            return images
        if images.device.type != "cpu" or images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("Digit augmentation expects normalized NCHW grayscale CPU images")
        if bool(((labels < 0) | (labels > 9)).any()):
            raise ValueError("Digit augmentation expects labels from 0 to 9")
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
        # Preserve exact inputs for digits whose categories are all disabled.
        changed = (strengths > 0).any(dim=1)
        return torch.where(changed[:, None, None, None], output, images)


@app.function
def train(task, X_train, y_train, seed=0, device="cuda"):
    """Train from scratch and return the best validation checkpoint's model."""
    return train_with_metrics(task, X_train, y_train, seed=seed, device=device)


@app.function
def train_with_metrics(task, X_train, y_train, seed=0, device="cuda", run_dir=None, augmentation=None):
    """Train with a seeded holdout; optionally persist each epoch and best weights.

    No test data enters training or checkpoint selection. Tied validation
    accuracies keep the earlier epoch. The grading entry point remains train().
    """
    torch.manual_seed(seed)                   # makes the run reproducible
    if len(X_train) != len(y_train) or len(X_train) < 2:
        raise ValueError("Training needs at least two examples and matching labels")
    if not 0 < VALIDATION_FRACTION < 1 or EPOCHS[task] < 1:
        raise ValueError("Use a validation fraction between 0 and 1 and at least one epoch")

    augmentation = digit_augmentation_config() if augmentation is None else augmentation
    augmenter = DigitAugmenter(augmentation, seed=seed + 1)
    if task != "mnist":
        augmenter.active = False
        augmenter.config["enabled"] = False

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
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    metadata = {
        "task": task, "seed": seed, "device": str(device), "epochs": EPOCHS[task],
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
        "momentum": optimizer.param_groups[0]["momentum"],
        "dampening": optimizer.param_groups[0]["dampening"],
        "optimizer": type(optimizer).__name__, "loss": repr(loss_fn),
        "learning_rate_schedule": "constant", "shuffle": True,
        "architecture": {
            "width": model.input_layer.out_features,
            "hidden": model.head_fc1.out_features,
            "depth": len(model.blocks),
            "functional_operations": sorted({
                node.func.attr for cls in (type(model), Block)
                for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(cls))))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "F"
            }),
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

            optimizer.zero_grad()             # clear old gradients
            loss.backward()                   # backward pass: compute gradients
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
