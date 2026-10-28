"""ImageNet32 models, training controls, and dataset-specific saved history."""

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

    NOTEBOOK_TASK = 'imagenet32'
    MAX_TRAIN_SECONDS = 3600
    DEFAULT_TASK = 'imagenet32'
    DEFAULT_SEED = 0
    DEFAULT_DEVICE = 'cuda'
    RESTORED_FROM = {'run_id': 'imagenet32-8856b095',
     'task': 'imagenet32',
     'legacy_activation': None,
     'legacy_optimizer': None,
     'legacy_augmentation': None}
    AUGMENTATION_CONFIG_OVERRIDES = {'cifar10': {'enabled': True,
                 'default': {'rotation': {'probability': 0.45, 'amount': 10.0},
                             'translation': {'probability': 0.5, 'amount': 4.0},
                             'scale': {'probability': 0.55, 'amount': 0.1},
                             'shear': {'probability': 0.5, 'amount': 5.0},
                             'brightness': {'probability': 0.45, 'amount': 0.1},
                             'noise': {'probability': 0.5, 'amount': 0.05},
                             'erasing': {'probability': 0.5, 'amount': 0.05},
                             'horizontal_flip': {'probability': 0.5, 'amount': 1.0}},
                 'classes': {},
                 'task': 'cifar10'},
     'imagenet32': {'task': 'imagenet32',
                    'enabled': False,
                    'default': {'rotation': {'probability': 0.5, 'amount': 10.0},
                                'translation': {'probability': 0.5, 'amount': 2.0},
                                'scale': {'probability': 0.5, 'amount': 0.1},
                                'shear': {'probability': 0.25, 'amount': 5.0},
                                'brightness': {'probability': 0.25, 'amount': 0.1},
                                'noise': {'probability': 0.25, 'amount': 0.05},
                                'erasing': {'probability': 0.1, 'amount': 0.05},
                                'horizontal_flip': {'probability': 0.5, 'amount': 1.0}},
                    'classes': {}}}

    LEARNING_RATE = 0.001
    BATCH_SIZE = 256
    EPOCHS = {'mnist': 500, 'cifar10': 60, 'imagenet32': 25}
    WEIGHT_DECAY = 1e-05
    MOMENTUM = 0.0
    DAMPENING = 0.0
    VALIDATION_FRACTION = 0.1

    OPTIMIZER_NAME = 'AdamW'
    OPTIMIZER_DEFAULTS = {'SGD': {'lr': 0.05, 'weight_decay': 0.0001, 'momentum': 0.0, 'dampening': 0.0, 'nesterov': False},
     'Adam': {'lr': 0.001,
              'weight_decay': 1e-05,
              'beta1': 0.9,
              'beta2': 0.999,
              'eps': 1e-08,
              'amsgrad': False},
     'AdamW': {'lr': 0.001,
               'weight_decay': 1e-05,
               'beta1': 0.9,
               'beta2': 0.999,
               'eps': 1e-08,
               'amsgrad': True},
     'Muon': {'lr': 0.03,
              'weight_decay': 1e-05,
              'momentum': 0.95,
              'nesterov': True,
              'ns_steps': 5,
              'eps': 1e-07,
              'adjust_lr_fn': 'original',
              'auxiliary_lr': 0.001,
              'auxiliary_weight_decay': 1e-05,
              'auxiliary_beta1': 0.9,
              'auxiliary_beta2': 0.999,
              'auxiliary_eps': 1e-08},
     'RMSprop': {'lr': 0.001,
                 'weight_decay': 1e-05,
                 'momentum': 0.0,
                 'alpha': 0.99,
                 'eps': 1e-08,
                 'centered': False},
     'Adagrad': {'lr': 0.01,
                 'weight_decay': 1e-05,
                 'lr_decay': 0.0,
                 'initial_accumulator_value': 0.0,
                 'eps': 1e-10}}

    # Post each completed app run's test result to the unofficial class leaderboard.
    LEADERBOARD_ENABLED = False
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
    _augmenter = ImageAugmenter(task.value, effective_augmentation(task.value, augmentation_controls),
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
def _():
    saved_runs_refresh = mo.ui.button(label="Refresh saved runs")
    load_saved = mo.ui.run_button(label="Load selected settings")
    load_best = mo.ui.run_button(label="Load best settings")
    best_metric = mo.ui.dropdown({"Validation accuracy": "validation_accuracy", "Test accuracy": "test_accuracy"},
                                 value="Validation accuracy", label="Best by")
    legacy_activation = mo.ui.dropdown(["gelu", "relu", "silu", "tanh", "sigmoid"],
                                       value=None, allow_select_none=True, label="Legacy activation (if missing)")
    legacy_optimizer = mo.ui.dropdown(available_optimizer_names(), value=None, allow_select_none=True,
                                      label="Legacy optimizer (if missing)")
    legacy_augmentation = mo.ui.dropdown({"Off": "off", "Current defaults": "current"},
                                         value=None, allow_select_none=True,
                                         label="Legacy augmentation (if missing)")
    return (
        best_metric,
        legacy_activation,
        legacy_augmentation,
        legacy_optimizer,
        load_best,
        load_saved,
        saved_runs_refresh,
    )


@app.cell(hide_code=True)
def _(RESULTS_FILE, saved_runs_refresh, task):
    _refresh = saved_runs_refresh.value
    saved_run_catalog = list_saved_runs(RESULTS_FILE, task.value, include_archived=True)
    _options = {}
    for _entry in reversed(saved_run_catalog):
        _validation = "N/A" if _entry["validation_accuracy"] is None else f"{_entry['validation_accuracy']:.2%}"
        _test = "N/A" if _entry["test_accuracy"] is None else f"{_entry['test_accuracy']:.2%}"
        _status = "" if _entry["completed"] else " · unfinished"
        _options[f"{_entry['id']} · validation {_validation} · test {_test}{_status}"] = _entry["id"]
    saved_run_choice = mo.ui.dropdown(_options, value=next(iter(_options), None), searchable=True,
                                      label="Saved run", full_width=True)
    return saved_run_catalog, saved_run_choice


@app.cell(hide_code=True)
def _(
    best_metric,
    legacy_activation,
    legacy_augmentation,
    legacy_optimizer,
    load_best,
    load_saved,
    saved_run_catalog,
    saved_run_choice,
    saved_runs_refresh,
):
    _best = best_saved_run(saved_run_catalog, best_metric.value)
    _best_note = f"Best completed run: **{_best['id']}** ({_best[best_metric.value]:.2%})." if _best else "No scored completed run available."
    mo.vstack([
        mo.md("## Saved runs\nLoad a previous run's model code, optimizer, augmentation, seed, and training settings. "
              "Loading updates this notebook on disk. The next **Run** trains a fresh model."),
        saved_runs_refresh, saved_run_choice,
        mo.hstack([load_saved, best_metric, load_best], justify="start"), mo.md(_best_note),
        mo.accordion({"Older runs with missing settings": mo.vstack([
            mo.md("Older snapshots may lack activation code, optimizer names, or augmentation settings. "
                  "Choose the missing values here before loading; recorded settings take precedence."),
            legacy_activation, legacy_optimizer, legacy_augmentation,
        ])}),
    ])
    return


@app.cell(hide_code=True)
def _():
    task = mo.ui.dropdown([NOTEBOOK_TASK], value=NOTEBOOK_TASK, label="Dataset")
    _available = {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
    device = (DEFAULT_DEVICE if _available.get(DEFAULT_DEVICE) else
              "cuda" if _available["cuda"] else "mps" if _available["mps"] else "cpu")
    seed = mo.ui.number(start=0, stop=2**32 - 1, value=DEFAULT_SEED, label="Seed")
    run_training = mo.ui.run_button(label="Run")
    return device, run_training, seed, task


@app.cell(hide_code=True)
def _():
    get_loaded_epochs, set_loaded_epochs = mo.state(None)
    return get_loaded_epochs, set_loaded_epochs


@app.cell(hide_code=True)
def _(get_loaded_epochs, task):
    _loaded = get_loaded_epochs()
    _epochs = (_loaded["epochs"] if _loaded and _loaded["task"] == task.value else EPOCHS[task.value])
    epoch_count = mo.ui.number(start=1, step=1, value=_epochs, label="Epochs")
    time_limit = mo.ui.number(start=0, step=1, value=(MAX_TRAIN_SECONDS or 0) / 60,
                              label="Time limit (minutes; 0 = unlimited)")
    return epoch_count, time_limit


@app.cell(hide_code=True)
def _(epoch_count, run_training, seed, task, time_limit):
    mo.vstack([
        mo.hstack([task, run_training], justify="start", wrap=True),
        mo.hstack([seed, epoch_count, time_limit], justify="start", wrap=True),
        mo.md(f"Completed runs post test accuracy as **{LEADERBOARD_ALIAS}**, section **{LEADERBOARD_SECTION}**."
              if LEADERBOARD_ENABLED else "Leaderboard posting disabled."),
    ])
    return


@app.cell(hide_code=True)
def _(
    best_metric,
    legacy_activation,
    legacy_augmentation,
    legacy_optimizer,
    load_best,
    load_saved,
    saved_run_catalog,
    saved_run_choice,
    set_loaded_epochs,
):
    mo.stop(not (load_saved.value or load_best.value))
    _entry = best_saved_run(saved_run_catalog, best_metric.value) if load_best.value else next(
        (entry for entry in saved_run_catalog if entry["id"] == saved_run_choice.value), None)
    mo.stop(_entry is None, mo.md("Choose a saved run with available settings."))
    try:
        _loaded = apply_saved_run(Path(__file__), _entry, legacy_activation.value, legacy_optimizer.value,
                                  legacy_augmentation.value)
    except (ValueError, OSError, SyntaxError, StopIteration, KeyError, TypeError) as _error:
        mo.stop(True, mo.md(f"Could not load settings: {_error}"))
    # A repeated load may leave the source unchanged, so explicitly reset the control.
    set_loaded_epochs({"task": _entry["config"]["task"], "epochs": _entry["config"]["epochs"]})
    mo.md(f"Loaded **{_loaded}**. The notebook will reload with its settings. Press **Run** when ready.")
    return


@app.cell(hide_code=True)
def _(augmentation_controls, epoch_count, optimizer_choice, task):
    _augmentation = effective_augmentation(task.value, augmentation_controls)
    mo.md(f"**Next run:** {task.value} · {optimizer_choice.value} · {int(epoch_count.value)} epochs · "
          f"augmentation **{'ON' if _augmentation['enabled'] else 'OFF'}**. "
          "Loading a saved run replaces settings; check this status before pressing **Run**.")
    return


@app.cell(hide_code=True)
def _(
    NUM_CLASSES,
    RESULTS_FILE,
    augmentation_controls,
    device,
    epoch_count,
    load_task,
    optimizer_choice,
    optimizer_controls,
    run_training,
    seed,
    task,
    time_limit,
):
    mo.stop(not run_training.value, mo.md("Ready. Select a task and start training."))

    with mo.status.spinner(title=f"Loading {task.value} and training on Apple Metal (MPS)..."):
        _X_train, _y_train, _X_test, _y_test = load_task(task.value)
        _start = time.perf_counter()
        _run_dir = RESULTS_FILE.parent / "runs" / f"{task.value}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        _model = train_with_metrics(task.value, _X_train, _y_train, seed=int(seed.value),
                                    device=device, run_dir=_run_dir, epochs=int(epoch_count.value),
                                    augmentation=effective_augmentation(task.value, augmentation_controls),
                                    optimizer_settings=selected_optimizer(optimizer_choice.value, optimizer_controls.value),
                                    max_seconds=float(time_limit.value) * 60 or None)
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
    if device == "mps":
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
    # ImageNet32 development

    Edit the model and training functions below, then use the controls above.
    This notebook owns ImageNet32 model settings and history. Other dataset notebooks are independent.
    Default preset: `imagenet32-8856b095` · validation 22.0222%.
    These settings train fresh weights. Click **Run** to start. The first run downloads
    the dataset; later runs reuse `data/`. Results append to `results.csv`.
    Each run holds out 10% of training examples for validation, records metrics
    every epoch, and evaluates the best validation checkpoint on the test set once.
    Curves and checkpoints save under `runs/`. Archived settings and metrics remain
    available even without local checkpoint files. Leaderboard posting is off by default.

    ImageNet normalizes one minibatch at a time. Training never starts just by opening this notebook.
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
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Model and training

    Edit task sizes and training settings in the setup cell. The classes and
    functions below drive this experimental notebook. The frozen grading path lives in `submission.py`.
    Keep the `train(task, X_train, y_train, seed=0, device="cuda")` and
    `predict(model, X)` interfaces unchanged for grading.
    """)
    return


@app.class_definition
class Block19(nn.Module):
    def __init__(self, width=512, h1=256, hidden=1024, dropout=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(width)
        self.value_up = nn.Linear(width, hidden)
        self.value_norm = nn.LayerNorm(hidden)
        self.gate_up = nn.Linear(width, hidden)
        self.gate_norm = nn.LayerNorm(hidden)
        self.down = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        normalized = self.input_norm(x)
        h = self.value_up(normalized)
        value = self.value_norm(h)
        a = F.gelu(self.gate_up(normalized))
        gate = self.gate_norm(a)
        gated = value * gate
        return x + self.dropout(self.down(gated))


@app.class_definition
class RollingWindowBlock(nn.Sequential):
    """Gather padded 2x2 windows, apply a shared MLP, and return an NCHW map.

    Numeric module names preserve existing window_mlps checkpoint keys.
    """

    def __init__(self, channels=3, window_hidden=128, window_features=64, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.add_module("0", nn.Linear(channels * 4, window_hidden))
        self.add_module("1", nn.GELU())
        self.add_module("2", nn.Linear(window_hidden, window_features))
        self.add_module("3", nn.GELU())

    def extract_windows(self, x):
        """Return (batch, output rows, output columns, C*2*2)."""
        x = F.pad(x, (0, self.padding, 0, self.padding))
        x = x.unfold(2, 2, self.stride).unfold(3, 2, self.stride)
        return x.permute(0, 2, 3, 1, 4, 5).flatten(3)

    def forward(self, x):
        x = self.extract_windows(x)
        x = self[0](x)
        x = self[1](x)
        x = self[2](x)
        x = self[3](x)
        return x.permute(0, 3, 1, 2)


@app.class_definition
class ResidualMLP(nn.Module):
    # torch-canvas input x: e, channels, 32, 32
    # torch-canvas sizes: in_features=3072, num_classes=1000
    """Hierarchical shared window MLPs, followed by a residual classifier.

    input -> sliding 2x2 windows, first stride 1, later shrinking_stride
          -> shared Linear -> GELU -> Linear -> GELU per window
          -> restore feature map and repeat until the spatial size is 1x1
          -> flatten final features -> Linear into the backbone
          -> depth x Block                           residual backbone, stays `width` wide
          -> Linear(width -> num_classes)            classification head
          -> logits (num_classes)

    The head has no skip connection because its output size (num_classes)
    differs from its input size (width), so the two cannot be added.
    """

    def __init__(self, in_features, num_classes, width=512, depth=2, hidden=1024,
                 dropout=0.1, window_hidden=128, window_features=64, shrinking_stride=2):
        super().__init__()

        if in_features != 784 and in_features != 3072:
            raise ValueError("Expected MNIST (784) or RGB 32x32 (3072) input features")
        if shrinking_stride != 1 and shrinking_stride != 2:
            raise ValueError("shrinking_stride must be 1 or 2")
        self.in_features = in_features
        channels = 1 if in_features == 784 else 3
        height = 28 if in_features == 784 else 32
        image_width = height
        self.image_shape = (channels, height, image_width)
        self.window_size = 2
        self.shrinking_stride = shrinking_stride
        self.window_shapes = [self.image_shape]
        self.window_strides = []
        self.window_padding = []
        self.window_mlps = nn.ModuleList()
        stages = height - 1 if shrinking_stride == 1 else 6
        for stage in range(stages):
            stride = 1 if stage == 0 else shrinking_stride
            self.window_strides.append(stride)
            # Keep the final row/column when a stride-2 stage has odd size.
            pad_height, pad_width = height % stride, image_width % stride
            self.window_padding.append((pad_height, pad_width))
            self.window_mlps.append(RollingWindowBlock(
                channels, window_hidden, window_features, stride, pad_height))
            height = (height + pad_height - self.window_size) // stride + 1
            image_width = (image_width + pad_width - self.window_size) // stride + 1
            channels = window_features
            self.window_shapes.append((channels, height, image_width))

        # Project the flattened window features down to the backbone width.
        self.input_layer = nn.Linear(channels * height * image_width, width)
        self.blocks = nn.ModuleList([Block19(width, hidden=hidden, dropout=dropout) for _ in range(depth)])
        self.head_fc2 = nn.Linear(width, num_classes)

        # Xavier (Glorot) initialization for every linear layer, zero biases.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def extract_windows(self, x, stage=0):
        """Return (batch, positions, C*2*2), ordered by row, column, channel, dy, dx."""
        if x.ndim != 4 or tuple(x.shape[1:]) != self.window_shapes[stage]:
            raise ValueError(f"Expected NCHW feature map (N, C, H, W) with C,H,W={self.window_shapes[stage]}")
        # Tensor.unfold only creates sliding views; it performs no learned operation.
        return self.window_mlps[stage].extract_windows(x).flatten(1, 2)

    def forward(self, x):
        """Training forward pass. Returns raw logits (no softmax).

        nn.CrossEntropyLoss applies the softmax itself, so training uses logits.
        """
        if x.ndim != 4 or tuple(x.shape[1:]) != self.image_shape:
            raise ValueError(f"Expected NCHW images with C,H,W={self.image_shape}")
        for window_mlp in self.window_mlps:
            x = window_mlp(x)
        x = self.input_layer(x.flatten(1))
        for blk in self.blocks:
            x = blk(x)
        return self.head_fc2(x)

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
        "restored_from_run": (config.get("restored_from") or {}).get("run_id", ""),
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
                if key not in ("model", "model_source", "block_source", "support_sources", "train_accuracy_kind")}
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
def list_saved_runs(results_file, task, include_archived=False):
    """Read local configuration snapshots and scores without loading weights."""
    results_file = Path(results_file)
    rows = {}
    if results_file.exists():
        with results_file.open(newline="") as file:
            for row in csv.DictReader(file):
                if row.get("task") == task and row.get("run_dir"):
                    rows[Path(row["run_dir"]).name] = row
    entries = []
    for path in sorted((results_file.parent / "runs").glob("*/config.json")):
        try:
            config = json.loads(path.read_text())
            if config.get("task") != task:
                continue
            summary = {}
            summary_path = path.parent / "summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                except (OSError, ValueError):
                    summary = {}
            row = rows.get(path.parent.name, {})
            performance = summary.get("performance", {})
            validation = performance.get("validation_accuracy", row.get("validation_accuracy"))
            test = performance.get("test_accuracy", row.get("test_accuracy") or row.get("accuracy"))
            if validation in (None, "") and (path.parent / "metrics.csv").exists():
                with (path.parent / "metrics.csv").open(newline="") as file:
                    scores = []
                    for epoch in csv.DictReader(file):
                        score = recorded_accuracy(epoch.get("validation_accuracy"))
                        if score is not None:
                            scores.append(score)
                    validation = max(scores) if scores else None
            entries.append({"id": path.parent.name, "config": config,
                            "validation_accuracy": recorded_accuracy(validation),
                            "test_accuracy": recorded_accuracy(test),
                            "completed": bool(row or summary.get("result")),
                            "time": row.get("time", path.parent.name)})
        except (OSError, ValueError, TypeError, AttributeError):
            # An interrupted or malformed run must not hide other saved runs.
            continue
    if include_archived:
        from residual_mlp_baseline.saved_history import merge_archived_runs
        entries = merge_archived_runs(task, entries)
    return entries


@app.function
def recorded_accuracy(value):
    """Return a valid saved score, keeping missing scores distinct from zero."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if 0 <= value <= 1 else None


@app.function
def best_saved_run(entries, metric="validation_accuracy"):
    """Rank completed runs by the chosen score; ties keep the earliest run ID."""
    if metric not in ("validation_accuracy", "test_accuracy"):
        raise ValueError("Choose validation or test accuracy")
    eligible = []
    for entry in sorted(entries, key=lambda row: row["id"]):
        if entry["completed"] and entry.get(metric) is not None:
            eligible.append(entry)
    return max(eligible, key=lambda entry: entry[metric]) if eligible else None


@app.function
def replace_source_span(source, node, replacement, include_decorators=False):
    """Patch one AST span while preserving unrelated notebook text and comments."""
    lines = source.splitlines(keepends=True)
    start_line = node.lineno
    start_column = node.col_offset
    if include_decorators and node.decorator_list:
        start_line = node.decorator_list[0].lineno
        start_column = node.decorator_list[0].col_offset - 1
    start = sum(len(line) for line in lines[:start_line - 1]) + start_column
    end = sum(len(line) for line in lines[:node.end_lineno - 1]) + node.end_col_offset
    return source[:start] + replacement + source[end:]


@app.function
def setup_value(source, name):
    """Read a literal setup value for preservation during a preset load."""
    setup = next(node for node in ast.parse(source).body if isinstance(node, ast.With))
    node = next(node for node in setup.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
    return ast.literal_eval(node.value)


@app.function
def patch_setup_value(source, name, value):
    """Replace a setup assignment's value, retaining its comments and location."""
    import pprint

    setup = next(node for node in ast.parse(source).body if isinstance(node, ast.With))
    node = next(node for node in setup.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
    literal = pprint.pformat(value, width=100, sort_dicts=False)
    literal = literal.replace("\n", "\n" + " " * node.col_offset)
    return replace_source_span(source, node.value, literal)


@app.function
def restored_architecture(config):
    """Recover dimensions from explicit metadata or an older model repr."""
    import re

    if config.get("architecture"):
        architecture = dict(config["architecture"])
    else:
        model = config.get("model", "")
        projection = re.search(r"\(input_layer\): Linear\(in_features=(\d+), out_features=(\d+)", model)
        head = re.search(r"\(head_fc1\): Linear\(in_features=(\d+), out_features=(\d+)", model)
        repeated = re.search(r"(\d+) x \w+\(", model)
        depth = int(repeated.group(1)) if repeated else len(re.findall(r"\(\d+\): \w+\(", model))
        if not depth:
            depth = len(re.findall(r"\(fc1(?:_\d+)?\): Linear\(", model))
        if not projection or not head or not depth:
            raise ValueError("This run did not save enough architecture information to restore")
        architecture = {"width": int(projection.group(2)), "hidden": int(head.group(2)), "depth": depth}
    for key in ("width", "hidden", "depth"):
        if not isinstance(architecture.get(key), int) or architecture[key] < 1:
            raise ValueError(f"Invalid saved architecture: {key}")
    return architecture


@app.function
def class_snapshot(source):
    """Extract a single saved class without evaluating the saved Python."""
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
        raise ValueError("Saved model source must contain exactly one class")
    node = tree.body[0]
    return node.name, "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])


@app.function
def configure_class_snapshot(source, architecture, activation=None):
    """Set actual dimensions, and normalize historical hardcoded block counts."""
    tree = ast.parse(source)
    cls = tree.body[0]
    constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    replacements = []
    defaults = constructor.args.defaults
    arguments = constructor.args.args[len(constructor.args.args) - len(defaults):]
    for argument, default in zip(arguments, defaults):
        if argument.arg in architecture:
            replacements.append((default, str(architecture[argument.arg])))
    for node in ast.walk(constructor):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute) and target.attr == "blocks"
                                               for target in node.targets):
            for child in ast.walk(node.value):
                if (isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "range"
                        and len(child.args) == 1 and isinstance(child.args[0], ast.Constant)):
                    replacements.append((child.args[0], "depth"))
    if activation is not None:
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and method.name == "forward":
                for node in ast.walk(method):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and isinstance(node.func.value, ast.Name) and node.func.value.id == "F"
                            and node.func.attr in ("relu", "gelu", "silu", "tanh", "sigmoid")):
                        replacements.append((node.func, f"F.{activation}"))
    replacements.sort(key=lambda edit: (edit[0].lineno, edit[0].col_offset), reverse=True)
    for node, replacement in replacements:
        source = replace_source_span(source, node, replacement)
    return source


@app.function
def restored_optimizer(config, legacy_optimizer=None):
    """Use recorded optimizer settings; require a choice if its name was not saved."""
    if config.get("optimizer_settings"):
        recorded = dict(config["optimizer_settings"])
        settings = {**optimizer_config(recorded["name"]), **recorded}
        validate_restored_optimizer(settings)
        return settings
    name = config.get("optimizer") or legacy_optimizer
    if not name:
        raise ValueError("This older run did not record its optimizer. Choose a legacy optimizer below.")
    settings = optimizer_config(name)
    mapping = {"learning_rate": "lr", "weight_decay": "weight_decay", "momentum": "momentum",
               "dampening": "dampening", "eps": "eps"}
    for recorded, key in mapping.items():
        if config.get(recorded) is not None and key in settings:
            settings[key] = config[recorded]
    if config.get("betas") and "beta1" in settings:
        settings["beta1"], settings["beta2"] = config["betas"]
    validate_restored_optimizer(settings)
    return settings


@app.function
def validate_restored_optimizer(settings):
    """Reject unusable settings before updating the notebook, preserving RNG state."""
    for key, value in settings.items():
        if isinstance(value, (int, float)) and not np.isfinite(value):
            raise ValueError(f"Saved optimizer setting {key} must be finite")
    with torch.random.fork_rng(devices=[]):
        build_optimizers(nn.Linear(1, 1), settings)


@app.function
def restored_augmentation(config, legacy_augmentation=None):
    """Translate legacy digit controls to the dataset-aware configuration schema."""
    task = config["task"]
    saved = config.get("augmentation")
    if saved is None:
        if legacy_augmentation not in ("off", "current"):
            raise ValueError("This older run did not record augmentation. Choose a legacy augmentation setting below.")
        normalized = augmentation_config(task)
        normalized["enabled"] = legacy_augmentation == "current"
        return normalized
    if "default" in saved and "classes" in saved:
        normalized = dict(saved)
        normalized["task"] = task
    else:
        normalized = augmentation_config(task)
        normalized["enabled"] = saved["enabled"]
        if task == "mnist" and "digits" in saved:
            normalized["classes"] = saved["digits"]
        elif task != "mnist" and saved["enabled"]:
            raise ValueError("This older run did not record a supported RGB augmentation configuration")
    ImageAugmenter(task, normalized)
    return normalized


@app.function
def torch_canvas_model_metadata(source, task):
    """Add display-only dataset dimensions without changing executable model code."""
    import io
    import re
    import tokenize

    shapes = {"mnist": "e, 1, 28, 28", "cifar10": "e, 3, 32, 32", "imagenet32": "e, 3, 32, 32"}
    if "RollingWindowBlock(" in source:
        shapes = {task: shape.replace(", 1,", ", channels,").replace(", 3,", ", channels,")
                  for task, shape in shapes.items()}
    if task not in shapes or task not in TASKS:
        raise ValueError("Unknown dataset for model display metadata")
    original = ast.parse(source)
    model = original.body[0]
    forward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    input_name = forward.args.args[1].arg
    lines = source.splitlines()
    metadata = re.compile(r"^#\s*torch-canvas (?:input\s+\w+|sizes)\s*:")
    remove = {token.start[0] - 1 for token in tokenize.generate_tokens(io.StringIO(source).readline)
              if token.type == tokenize.COMMENT and token.start[1] == 4 and metadata.match(token.string)}
    lines = [line for index, line in enumerate(lines) if index not in remove]
    model = ast.parse("\n".join(lines)).body[0]
    sizes = TASKS[task]
    lines[model.body[0].lineno - 1:model.body[0].lineno - 1] = [
        f"    # torch-canvas input {input_name}: {shapes[task]}",
        f"    # torch-canvas sizes: in_features={sizes['in_features']}, num_classes={sizes['num_classes']}",
    ]
    result = "\n".join(lines)
    if ast.dump(ast.parse(result)) != ast.dump(original):
        raise ValueError("Display metadata must not change executable model code")
    return result


@app.function
def restore_run_source(source, entry, legacy_activation=None, legacy_optimizer=None, legacy_augmentation=None):
    """Restore architecture and experiment defaults, preserving the rest of the notebook."""
    config = entry["config"]
    task = config["task"]
    if task not in TASKS:
        raise ValueError("Unknown saved dataset")
    setup_names = {target.id for node in ast.parse(source).body
                   if isinstance(node, ast.With) for child in node.body if isinstance(child, ast.Assign)
                   for target in child.targets if isinstance(target, ast.Name)}
    if "NOTEBOOK_TASK" in setup_names and setup_value(source, "NOTEBOOK_TASK") != task:
        raise ValueError("This notebook only loads settings for its own dataset")
    architecture = restored_architecture(config)
    tree = ast.parse(source)
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    model = next(node for node in classes if node.name == "ResidualMLP")
    # Marimo may wrap a class in a cell when an edit introduces a dependency.
    # Replace the entire wrapper so stale cell arguments cannot survive loading.
    model_span = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and model in node.body), model)
    support_names = set(config.get("support_sources", {})) | {"RollingWindowBlock"}
    support_nodes = [node for node in classes if node.name in support_names]
    blocks = [node for node in classes if node.name not in ("ResidualMLP", "ImageAugmenter") and node not in support_nodes]
    if len(blocks) != 1:
        raise ValueError("The current notebook must have one residual block class")
    block = blocks[0]
    if config.get("model_source") and config.get("block_source"):
        model_name, model_source = class_snapshot(config["model_source"])
        block_name, block_source = class_snapshot(config["block_source"])
        if model_name != "ResidualMLP":
            # Grading runs can save task-specific class names. Keep the editable
            # dashboard's public class name while preserving its implementation.
            model_source = model_source.replace(f"class {model_name}(", "class ResidualMLP(", 1)
        model_source = configure_class_snapshot(model_source, architecture)
        block_source = configure_class_snapshot(block_source, architecture)
    else:
        if legacy_activation not in ("gelu", "relu", "silu", "tanh", "sigmoid"):
            raise ValueError("This older run did not record activation code. Choose a legacy activation below.")
        model_source = textwrap.dedent("\n".join(source.splitlines()[model.lineno - 1:model.end_lineno]))
        block_source = ast.get_source_segment(source, block)
        model_source = configure_class_snapshot(model_source, architecture, legacy_activation)
        block_source = configure_class_snapshot(block_source, architecture, legacy_activation)
    model_source = torch_canvas_model_metadata(model_source, task)
    support_sources = dict(config.get("support_sources", {}))
    # Older snapshots predate support_sources; retain a current helper only when
    # the restored model actually references it (including legacy dimension-only runs).
    references = {node.id for node in ast.walk(ast.parse(model_source)) if isinstance(node, ast.Name)}
    for node in support_nodes:
        if node.name in references and node.name not in support_sources:
            support_sources[node.name] = ast.get_source_segment(source, node)
    helpers = {}
    for name, snapshot in support_sources.items():
        actual_name, helper_source = class_snapshot(snapshot)
        if actual_name != name or name in ("ResidualMLP", block_name if config.get("block_source") else block.name):
            raise ValueError("Invalid saved support class")
        helpers[name] = "@app.class_definition\n" + helper_source
    current_helpers = {node.name: node for node in support_nodes}
    missing_helpers = [text for name, text in helpers.items() if name not in current_helpers]
    decorated_model = "\n\n\n".join(missing_helpers + ["@app.class_definition\n" + model_source])
    replacements = [(model_span, decorated_model), (block, "@app.class_definition\n" + block_source)]
    replacements.extend((node, helpers.get(node.name, "")) for node in support_nodes)
    replacements.sort(key=lambda edit: edit[0].lineno, reverse=True)
    for node, replacement in replacements:
        source = replace_source_span(source, node, replacement, include_decorators=True)
    settings = restored_optimizer(config, legacy_optimizer)
    augmentation = restored_augmentation(config, legacy_augmentation)
    for key in ("batch_size", "epochs", "seed"):
        if not isinstance(config.get(key), int) or config[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid or missing saved {key}")
    if config["seed"] > 2**32 - 1:
        raise ValueError("Saved seed is outside the dashboard's supported range")
    if not 0 < config.get("validation_fraction", 0) < 1:
        raise ValueError("Invalid saved validation fraction")
    epochs = setup_value(source, "EPOCHS")
    epochs[task] = config["epochs"]
    presets = dict(OPTIMIZER_DEFAULTS)
    presets[settings["name"]] = {key: value for key, value in settings.items() if key != "name"}
    overrides = setup_value(source, "AUGMENTATION_CONFIG_OVERRIDES")
    overrides[task] = augmentation
    values = {"DEFAULT_TASK": task, "DEFAULT_SEED": config["seed"],
              "DEFAULT_DEVICE": config.get("device"), "BATCH_SIZE": config["batch_size"],
              "EPOCHS": epochs, "VALIDATION_FRACTION": config["validation_fraction"],
              "LEARNING_RATE": settings["lr"], "WEIGHT_DECAY": settings["weight_decay"],
              "OPTIMIZER_NAME": settings["name"], "OPTIMIZER_DEFAULTS": presets,
              "AUGMENTATION_CONFIG_OVERRIDES": overrides,
              "RESTORED_FROM": {"run_id": entry["id"], "task": task,
                                "legacy_activation": legacy_activation, "legacy_optimizer": legacy_optimizer,
                                "legacy_augmentation": legacy_augmentation}}
    if "MAX_TRAIN_SECONDS" in setup_names:
        limit = config.get("max_seconds")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float))
                                  or not 0 < limit < float("inf")):
            raise ValueError("Invalid saved training time limit")
        values["MAX_TRAIN_SECONDS"] = limit
    for name, value in values.items():
        source = patch_setup_value(source, name, value)
    ast.parse(source)
    return source


@app.function
def apply_saved_run(notebook_path, entry, legacy_activation=None, legacy_optimizer=None, legacy_augmentation=None):
    """Atomically apply a preset without reading checkpoint weights or running training."""
    import tempfile

    notebook_path = Path(notebook_path)
    before = notebook_path.read_text()
    restored = restore_run_source(before, entry, legacy_activation, legacy_optimizer, legacy_augmentation)
    if notebook_path.read_text() != before:
        raise ValueError("The notebook changed during loading. Refresh saved runs and try again.")
    if restored == before:
        return entry["id"]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=notebook_path.parent,
                                         prefix=".notebook-restore-", suffix=".py", delete=False) as file:
            temporary = Path(file.name)
            file.write(restored)
        temporary.chmod(notebook_path.stat().st_mode & 0o777)
        if notebook_path.read_text() != before:
            raise ValueError("The notebook changed during loading. Refresh saved runs and try again.")
        temporary.replace(notebook_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return entry["id"]


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
    if task in AUGMENTATION_CONFIG_OVERRIDES:
        return json.loads(json.dumps(AUGMENTATION_CONFIG_OVERRIDES[task]))
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


@app.function
def effective_augmentation(task, controls):
    """Use the same controls for display, preview, and training; reject stale UI state."""
    config = selected_augmentation(task, controls.value)
    if config["enabled"] != controls[task]["enabled"].value:
        raise ValueError("Augmentation controls disagree. Reload the notebook before training.")
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
def model_architecture(model):
    """Describe actual layers, including Canvas models with a single-layer head.

    Preserve the historical meaning of hidden (head expansion) when present.
    Otherwise record the first block's expansion, or backbone width when the
    model has no expanded linear layer.
    """
    head = getattr(model, "head_fc1", None)
    block_linears = [layer for layer in model.blocks[0].modules() if isinstance(layer, nn.Linear)] if len(model.blocks) else []
    hidden = head.out_features if isinstance(head, nn.Linear) else (
        block_linears[0].out_features if block_linears else model.input_layer.out_features)
    architecture = {
        "width": model.input_layer.out_features,
        "hidden": hidden,
        "depth": len(model.blocks),
        "in_features": getattr(model, "in_features", model.input_layer.in_features),
        "num_classes": model.head_fc2.out_features,
        "normalization": sorted({type(layer).__name__ for layer in model.modules()
                                 if "Norm" in type(layer).__name__}),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    if hasattr(model, "window_mlps"):
        architecture.update(
            window_size=model.window_size, shrinking_stride=model.shrinking_stride,
            window_strides=model.window_strides, window_shapes=model.window_shapes,
            window_hidden=model.window_mlps[0][0].out_features,
            window_features=model.window_mlps[0][2].out_features,
            window_order="row, column, channel, dy, dx", window_padding=getattr(model, "window_padding", 0),
            window_weight_sharing="within each stage", window_aggregation="hierarchical",
        )
    return architecture


@app.function
def train(task, X_train, y_train, seed=0, device="cuda"):
    """Train from scratch and return the best validation checkpoint's model."""
    return train_with_metrics(task, X_train, y_train, seed=seed, device=device)


@app.function
def train_with_metrics(task, X_train, y_train, seed=0, device="cuda", run_dir=None, augmentation=None,
                       optimizer_settings=None, epochs=None, model_factory=None, batch_size=None,
                       max_seconds=None):
    """Train with a seeded holdout; optionally persist each epoch and best weights.

    No test data enters training or checkpoint selection. Tied validation
    accuracies keep the earlier epoch. The grading entry point remains train().
    """
    if task != NOTEBOOK_TASK:
        raise ValueError("Use the dedicated notebook for that dataset")
    call_started = time.perf_counter()
    if max_seconds is not None and (not isinstance(max_seconds, (int, float))
                                    or isinstance(max_seconds, bool) or not np.isfinite(max_seconds)
                                    or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive")
    deadline = (call_started + max_seconds - min(30.0, max_seconds * 0.02)
                if max_seconds is not None else None)
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    torch.manual_seed(seed)                   # makes the run reproducible
    if len(X_train) != len(y_train) or len(X_train) < 2:
        raise ValueError("Training needs at least two examples and matching labels")
    epochs = EPOCHS[task] if epochs is None else epochs
    if (not 0 < VALIDATION_FRACTION < 1 or not isinstance(epochs, int)
            or isinstance(epochs, bool) or epochs < 1):
        raise ValueError("Use a validation fraction between 0 and 1 and at least one epoch")

    augmentation = augmentation_config(task) if augmentation is None else augmentation
    augmenter = ImageAugmenter(task, augmentation, seed=seed + 1)

    # 1. Build the model for this task. Only input size and class count differ.
    factory = ResidualMLP if model_factory is None else model_factory
    model = factory(**TASKS[task]).to(device)

    # 2. Wrap the data so we can loop over shuffled mini-batches.
    # Keep ImageNet's memory-mapped uint8 storage; normalize one batch at a time.
    streaming_normalization = task == "imagenet32"
    dataset = TensorDataset(torch.as_tensor(X_train) if streaming_normalization else prepare(X_train),
                            torch.as_tensor(y_train).long())
    split = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    validation_size = min(len(dataset) - 1, max(1, int(len(dataset) * VALIDATION_FRACTION)))
    validation_indices, training_indices = split[:validation_size], split[validation_size:]
    loader = DataLoader(Subset(dataset, training_indices.tolist()), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(Subset(dataset, validation_indices.tolist()), batch_size=batch_size)

    # 3. Choose the loss and the optimizer.
    loss_fn = nn.CrossEntropyLoss()           # expects raw logits
    optimizers, optimizer_settings, optimizer_groups = build_optimizers(model, optimizer_settings)
    functional_operations = set()
    block_cls = type(model.blocks[0]) if len(model.blocks) else None
    model_types = [type(model)]
    if block_cls is not None:
        model_types.append(block_cls)
    support_types = list(dict.fromkeys(type(layer) for layer in model.modules()
                                     if type(layer).__module__ == type(model).__module__
                                     and type(layer) not in model_types))
    model_types.extend(support_types)
    for cls in model_types:
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(cls)))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "F"):
                functional_operations.add(node.func.attr)
    metadata = {
        "task": task, "seed": seed, "device": str(device), "epochs": epochs,
        "restored_from": RESTORED_FROM if RESTORED_FROM and RESTORED_FROM["task"] == task else None,
        "batch_size": batch_size, "learning_rate": optimizer_groups[0]["lr"],
        "max_seconds": max_seconds,
        "streaming_normalization": streaming_normalization,
        "weight_decay": optimizer_groups[0]["weight_decay"],
        "momentum": optimizer_groups[0].get("momentum"),
        "dampening": optimizer_groups[0].get("dampening"),
        "betas": optimizer_groups[0].get("betas"),
        "eps": optimizer_groups[0].get("eps"),
        "optimizer": optimizer_settings["name"], "loss": repr(loss_fn),
        "optimizer_settings": optimizer_settings, "optimizer_groups": optimizer_groups,
        "learning_rate_schedule": "constant", "shuffle": True,
        "architecture": {
            **model_architecture(model),
            "functional_operations": sorted(functional_operations),
        },
        "model_source": inspect.getsource(type(model)),
        "block_source": inspect.getsource(block_cls) if block_cls else None,
        "support_sources": {cls.__name__: inspect.getsource(cls) for cls in support_types},
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
    stop_reason = "epochs"
    for epoch in range(epochs):
        if deadline is not None and time.perf_counter() >= deadline:
            stop_reason = "time_budget"
            break
        model.train()
        total_loss = torch.zeros((), device=device)
        correct = torch.zeros((), dtype=torch.long, device=device)
        for X_batch, y_batch in loader:
            if deadline is not None and time.perf_counter() >= deadline:
                stop_reason = "time_budget"
                break
            if streaming_normalization:
                X_batch = prepare(X_batch)
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

        if stop_reason == "time_budget":
            break

        model.eval()
        validation_loss = torch.zeros((), device=device)
        validation_correct = torch.zeros((), dtype=torch.long, device=device)
        with torch.inference_mode():
            for X_batch, y_batch in validation_loader:
                if deadline is not None and time.perf_counter() >= deadline:
                    stop_reason = "time_budget"
                    break
                if streaming_normalization:
                    X_batch = prepare(X_batch)
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                validation_loss += loss_fn(logits, y_batch) * len(X_batch)
                validation_correct += (logits.argmax(1) == y_batch).sum()
        if stop_reason == "time_budget":
            break
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
    if best_state is None:
        raise RuntimeError("Training time budget ended before one complete training/validation epoch")
    model.load_state_dict(best_state)
    model.eval()
    model.run_config = metadata
    model.training_history = history
    model.best_epoch = best_epoch
    model.best_validation_accuracy = best_accuracy
    model.training_indices = training_indices
    model.validation_indices = validation_indices
    model.stop_reason = stop_reason
    model.training_seconds = time.perf_counter() - call_started
    if run_dir is not None:
        (run_dir / "training-receipt.json").write_text(json.dumps({
            "task": task, "seed": seed, "stop_reason": stop_reason,
            "epochs_completed": len(history), "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
            "training_seconds": model.training_seconds, "max_seconds": max_seconds,
        }, indent=2) + "\n")
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
    _project = next((parent for parent in Path(__file__).resolve().parents
                     if (parent / "pyproject.toml").exists()), Path.cwd())
    DATA_DIR = str(_project / "data")
    RESULTS_FILE = _project / "results.csv"
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
