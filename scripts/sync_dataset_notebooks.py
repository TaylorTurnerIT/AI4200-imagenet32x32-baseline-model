"""Explicitly refresh portable histories and notebook presets from saved runs.

This regenerates the three editable dataset notebooks. Commit notebook edits
before using it; ordinary notebook startup never rewrites source or defaults.
"""

import argparse
import csv
import json
from pathlib import Path

from residual_mlp_baseline import notebook


def dataset_source(template, task, entry):
    source = template.replace("    DEFAULT_TASK =", f"    NOTEBOOK_TASK = {task!r}\n"
                              "    MAX_TRAIN_SECONDS = None\n    DEFAULT_TASK =", 1)
    source = notebook.restore_run_source(source, entry)
    source = source.replace('mo.ui.dropdown(["mnist", "cifar10", "imagenet32"], value=DEFAULT_TASK, label="Task")',
                            'mo.ui.dropdown([NOTEBOOK_TASK], value=NOTEBOOK_TASK, label="Dataset")')
    source = source.replace('    device = "mps"',
                            '    _available = {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}\n'
                            '    device = (DEFAULT_DEVICE if _available.get(DEFAULT_DEVICE) else\n'
                            '              "cuda" if _available["cuda"] else "mps" if _available["mps"] else "cpu")', 1)
    source = source.replace('list_saved_runs(RESULTS_FILE, task.value)',
                            'list_saved_runs(RESULTS_FILE, task.value, include_archived=True)', 1)
    source = source.replace('    return (epoch_count,)',
                            '    time_limit = mo.ui.number(start=0, step=1, value=(MAX_TRAIN_SECONDS or 0) / 60,\n'
                            '                              label="Time limit (minutes; 0 = unlimited)")\n'
                            '    return epoch_count, time_limit', 1)
    source = source.replace('def _(epoch_count, run_training, seed, task):',
                            'def _(epoch_count, run_training, seed, task, time_limit):', 1)
    source = source.replace('mo.hstack([seed, epoch_count], justify="start", wrap=True),',
                            'mo.hstack([seed, epoch_count, time_limit], justify="start", wrap=True),', 1)
    source = source.replace('    run_training,\n    seed,\n    task,\n):',
                            '    run_training,\n    seed,\n    task,\n    time_limit,\n):', 1)
    source = source.replace('optimizer_settings=selected_optimizer(optimizer_choice.value, optimizer_controls.value))',
                            'optimizer_settings=selected_optimizer(optimizer_choice.value, optimizer_controls.value),\n'
                            '                                    max_seconds=float(time_limit.value) * 60 or None)', 1)
    source = source.replace('    call_started = time.perf_counter()',
                            '    if task != NOTEBOOK_TASK:\n'
                            '        raise ValueError("Use the dedicated notebook for that dataset")\n'
                            '    call_started = time.perf_counter()', 1)
    source = source.replace('    DATA_DIR = str(Path(__file__).resolve().parents[2] / "data")\n'
                            '    RESULTS_FILE = Path(__file__).resolve().parents[2] / "results.csv"',
                            '    _project = next((parent for parent in Path(__file__).resolve().parents\n'
                            '                     if (parent / "pyproject.toml").exists()), Path.cwd())\n'
                            '    DATA_DIR = str(_project / "data")\n'
                            '    RESULTS_FILE = _project / "results.csv"', 1)
    title = {"mnist": "MNIST", "cifar10": "CIFAR-10", "imagenet32": "ImageNet32"}[task]
    source = source.replace('# Residual MLP development', f'# {title} development', 1)
    source = source.replace('Train your local code on MNIST, CIFAR-10, or ImageNet 32×32.',
                            f'This notebook owns {title} model settings and history. Other dataset notebooks are independent.', 1)
    source = source.replace('Choose a task, then click **Run**.',
                            f'Default preset: `{entry["id"]}` · validation {entry["validation_accuracy"]:.4%}.\n'
                            '    These settings train fresh weights. Click **Run** to start.', 1)
    source = source.replace('Curves and checkpoints save under `runs/`. Each completed run automatically\n'
                            '    posts its test accuracy to the unofficial class leaderboard. Submission status\n'
                            '    appears with the result; a network failure does not discard local results.',
                            'Curves and checkpoints save under `runs/`. Archived settings and metrics remain\n'
                            '    available even without local checkpoint files. Leaderboard posting is off by default.', 1)
    source = source.replace('    ImageNet training needs substantial RAM because the baseline converts the\n'
                            '    entire training set to float32. Start with MNIST to check your environment.',
                            '    ImageNet normalizes one minibatch at a time. Training never starts just by opening this notebook.', 1)
    source = source.replace('functions below are also used by the package and grading harness.',
                            'functions below drive this experimental notebook. The frozen grading path lives in `submission.py`.', 1)
    source = source.replace(f'"""Model, training, and local evaluation in one editable marimo notebook."""',
                            f'"""{title} models, training controls, and dataset-specific saved history."""', 1)
    return source


def sync(source_root, output_root):
    package = output_root / "src" / "residual_mlp_baseline"
    history_dir = package / "histories"
    notebooks = package / "notebooks"
    history_dir.mkdir(exist_ok=True)
    notebooks.mkdir(exist_ok=True)
    (notebooks / "__init__.py").write_text('"""Independent editable Marimo notebooks, one per dataset."""\n')
    template = (package / "notebook.py").read_text()
    rows = list(csv.DictReader((source_root / "results.csv").open())) if (source_root / "results.csv").exists() else []
    for task in notebook.TASKS:
        entries = notebook.list_saved_runs(source_root / "results.csv", task)
        best = notebook.best_saved_run(entries)
        if best is None:
            raise ValueError(f"No completed validation-scored preset for {task}")
        for entry in entries:
            path = source_root / "runs" / entry["id"] / "metrics.csv"
            entry["metrics"] = list(csv.DictReader(path.open())) if path.exists() else []
            entry["archive_origin"] = str(path.parent)
            receipt = path.parent / "training-receipt.json"
            if receipt.exists():
                entry["training_receipt"] = json.loads(receipt.read_text())
        archive = {"task": task, "selection_metric": "validation_accuracy", "default_run_id": best["id"],
                   "entries": entries, "legacy_rows": [row for row in rows if row.get("task") == task]}
        (history_dir / f"{task}.json").write_text(json.dumps(archive, indent=2) + "\n")
        (notebooks / f"{task}.py").write_text(dataset_source(template, task, best))
        print(f"{task}: {len(entries)} saved configurations; default {best['id']} ({best['validation_accuracy']:.4%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    options = parser.parse_args()
    sync(options.source_root.resolve(), options.output_root.resolve())
