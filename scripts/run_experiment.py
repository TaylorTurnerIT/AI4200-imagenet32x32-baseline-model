"""Run a cached dataset through the notebook's training and reporting pipeline."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

import numpy as np
import torch

from residual_mlp_baseline import notebook


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=notebook.TASKS, default="imagenet32")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    folder = root / "data" / args.task
    arrays = [np.load(folder / f"{name}.npy", mmap_mode="c")
              for name in ("X_train", "y_train", "X_test", "y_test")]
    X_train, y_train, X_test, y_test = arrays
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple Metal (MPS) is unavailable")
        probe = torch.ones(1, device="mps", requires_grad=True)
        probe.square().sum().backward()
        torch.mps.synchronize()

    run_dir = root / "runs" / f"{args.task}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Training {len(y_train):,} images; test {len(y_test):,}; device {args.device}", flush=True)
    started = time.perf_counter()
    model = notebook.train_with_metrics(
        args.task, X_train, y_train, seed=args.seed, device=args.device,
        run_dir=run_dir, epochs=args.epochs,
    )
    minutes = (time.perf_counter() - started) / 60
    provenance = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "notebook_path": notebook.__file__,
        "argv": vars(args),
        "torch_version": torch.__version__,
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    finish_run(root, run_dir, model, X_train, y_train, X_test, y_test, minutes, args.submit)


def finish_run(root, run_dir, model, X_train, y_train, X_test, y_test, minutes, submit=False):
    """Evaluate and report a selected checkpoint, then optionally submit once."""
    task = model.run_config["task"]

    # Keep evaluation bounded and reuse the same predictions for class metrics.
    print("Evaluating selected validation checkpoint on test images...", flush=True)
    predictions = []
    for start in range(0, len(y_test), notebook.BATCH_SIZE):
        probabilities = notebook.predict(model, X_test[start:start + notebook.BATCH_SIZE])
        expected_size = min(notebook.BATCH_SIZE, len(y_test) - start)
        if probabilities.shape != (expected_size, notebook.TASKS[task]["num_classes"]):
            raise ValueError("Unexpected prediction shape")
        if not torch.isfinite(probabilities).all() or not (probabilities >= 0).all():
            raise ValueError("Invalid test probabilities")
        torch.testing.assert_close(probabilities.sum(1), torch.ones(len(probabilities)), atol=1e-3, rtol=0)
        predictions.append(probabilities.argmax(1).numpy())
    predicted = np.concatenate(predictions)
    accuracy = float(np.mean(predicted == y_test))
    test_rows = notebook.per_class_accuracy(task, y_test, predicted)
    print(f"Test accuracy: {accuracy:.4%}. Scoring clean training and validation splits...", flush=True)
    train_rows = notebook.evaluate_per_class(
        model, task, X_train, y_train, model.training_indices.numpy(), chunk=notebook.BATCH_SIZE)
    validation_rows = notebook.evaluate_per_class(
        model, task, X_train, y_train, model.validation_indices.numpy(), chunk=notebook.BATCH_SIZE)
    train_accuracy = sum(row["correct"] for row in train_rows) / len(model.training_indices)
    per_class = [dict(split=split, **row)
                 for split, rows in (("train", train_rows), ("validation", validation_rows), ("test", test_rows))
                 for row in rows]
    notebook.save_run_result(root / "results.csv", run_dir, model, accuracy, train_accuracy,
                             minutes, len(y_test), per_class=per_class)
    if submit:
        notebook.LEADERBOARD_ENABLED = True
        print(notebook.submit_run_to_leaderboard(task, accuracy, minutes, run_dir), flush=True)

    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    epochs = [row["epoch"] for row in model.training_history]
    for split in ("train", "validation"):
        axes[0].plot(epochs, [row[f"{split}_loss"] for row in model.training_history], label=split)
        axes[1].plot(epochs, [100 * row[f"{split}_accuracy"] for row in model.training_history], label=split)
    for axis, label in zip(axes, ("Cross-entropy loss", "Accuracy (%)")):
        axis.set(xlabel="Epoch", ylabel=label)
        axis.axvline(model.best_epoch, color="gray", linestyle="--", label="Selected epoch")
        axis.legend()
    figure.tight_layout()
    figure.savefig(run_dir / "curves.png", dpi=150)
    plt.close(figure)
    print(f"Completed: {run_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
