"""Continue an analysis checkpoint while measuring sustained overfitting.

The grading train() interface still always trains from scratch. Original run
artifacts are read-only; continuation artifacts go into a separate directory.
"""

import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import time
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from residual_mlp_baseline import notebook

try:
    from .run_experiment import finish_run
except ImportError:
    from run_experiment import finish_run


def divergence_status(history, start_epoch, tolerance=0.01, patience=3):
    """Require three successive regressions of three-epoch mean validation loss.

    Each candidate must exceed the best observed validation loss by 1%, exceed
    the preceding three-epoch mean, and coincide with falling mean train loss.
    Six new epochs are required before the first candidate, allowing optimizer
    restart transients to settle. Flat validation loss is not divergence.
    """
    if not 0 <= tolerance < 1 or patience < 1:
        raise ValueError("Invalid divergence thresholds")
    for row in history:
        if not all(math.isfinite(row[key]) for key in ("train_loss", "validation_loss")):
            raise ValueError("Non-finite training or validation loss")
    continued = [row for row in history if row["epoch"] > start_epoch]
    streak, first_candidate = 0, None
    for end in range(6, len(continued) + 1):
        previous, recent = continued[end - 6:end - 3], continued[end - 3:end]
        mean = lambda rows, key: sum(row[key] for row in rows) / len(rows)
        best = min(row["validation_loss"] for row in history if row["epoch"] <= recent[-1]["epoch"])
        candidate = (mean(recent, "validation_loss") > best * (1 + tolerance)
                     and mean(recent, "validation_loss") > mean(previous, "validation_loss")
                     and mean(recent, "train_loss") < mean(previous, "train_loss"))
        if candidate:
            streak += 1
            if first_candidate is None:
                first_candidate = recent[-1]["epoch"]
        else:
            streak, first_candidate = 0, None
        if streak >= patience:
            break
    minimum = min(history, key=lambda row: row["validation_loss"])
    return {"confirmed": streak >= patience, "candidate_streak": streak,
            "first_candidate_epoch": first_candidate,
            "minimum_validation_loss_epoch": minimum["epoch"],
            "minimum_validation_loss": minimum["validation_loss"],
            "confirmation_epoch": recent[-1]["epoch"] if streak >= patience else None,
            "relative_loss_threshold": tolerance, "patience": patience,
            "window_epochs": 3, "continuation_start_epoch": start_epoch}


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path, help="Parent run containing best.pt and metrics.csv")
    parser.add_argument("--checkpoint", choices=("best.pt", "last.pt"), default="best.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-epochs", type=int, default=100, help="Total epoch guard, including parent epochs")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    parent = args.parent.resolve()
    checkpoint_path = parent / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    start_epoch = checkpoint["epoch"]
    if args.max_epochs <= start_epoch:
        raise ValueError("The total epoch guard must exceed the parent checkpoint epoch")
    config = copy.deepcopy(checkpoint["config"])
    task, seed = config["task"], config["seed"]
    sources = {"model_source": notebook.ResidualMLP, "block_source": notebook.Block19}
    sources.update({name: getattr(notebook, name) for name in config.get("support_sources", {})})
    for name, cls in sources.items():
        expected = config.get(name, config.get("support_sources", {}).get(name))
        if inspect.getsource(cls) != expected:
            raise ValueError(f"Current {name} differs from the saved model; refusing architecture drift")
    if config["augmentation"]["enabled"]:
        raise ValueError("This continuation currently requires augmentation disabled")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple Metal is unavailable")

    torch.manual_seed(seed)
    model = notebook.ResidualMLP(**notebook.TASKS[task]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizers, settings, groups = notebook.build_optimizers(model, config["optimizer_settings"])
    original_state = bool(checkpoint.get("optimizer_states"))
    if original_state:
        for optimizer, state in zip(optimizers, checkpoint["optimizer_states"], strict=True):
            optimizer.load_state_dict(state)

    folder = root / "data" / task
    X_train, y_train, X_test, y_test = [np.load(folder / f"{name}.npy", mmap_mode="c")
                                      for name in ("X_train", "y_train", "X_test", "y_test")]
    training_indices = checkpoint["training_indices"]
    validation_indices = checkpoint["validation_indices"]
    joined = torch.cat((training_indices, validation_indices))
    if not torch.equal(joined.sort().values, torch.arange(len(y_train))):
        raise ValueError("Parent split must cover the training dataset exactly once")
    dataset = TensorDataset(notebook.prepare(X_train), torch.as_tensor(y_train).long())
    batch_size = config["batch_size"]
    notebook.BATCH_SIZE = batch_size
    loader = DataLoader(Subset(dataset, training_indices.tolist()), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(Subset(dataset, validation_indices.tolist()), batch_size=batch_size)
    with (parent / "metrics.csv").open() as file:
        history = [{key: int(value) if key == "epoch" else float(value) for key, value in row.items()}
                   for row in csv.DictReader(file) if int(row["epoch"]) <= start_epoch]
    if not history or history[-1]["epoch"] != start_epoch:
        raise ValueError("Missing parent checkpoint metrics")
    if "cpu_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["cpu_rng_state"])
        if args.device == "mps":
            torch.mps.set_rng_state(checkpoint["mps_rng_state"])

    origin_epoch = config.get("continuation", {}).get("origin_epoch", start_epoch)
    config.update(device=args.device, epochs=args.max_epochs, optimizer_settings=settings, optimizer_groups=groups)
    config["continuation"] = {
        "parent_run": str(parent), "parent_epoch": start_epoch, "origin_epoch": origin_epoch,
        "parent_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "optimizer_state_restored": original_state, "rng_state_restored": "cpu_rng_state" in checkpoint,
        "optimizer_note": "Restored saved state" if original_state else "Fresh AdamW state; parent saved weights only",
        "criterion": "Three successive 3-epoch mean validation-loss regressions >1% above minimum while train loss falls",
    }
    run_dir = root / "runs" / f"{task}-continuation-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (run_dir / "provenance.json").write_text(json.dumps({
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "notebook_path": notebook.__file__, "argv": {**vars(args), "parent": str(parent)},
        "torch_version": torch.__version__,
    }, indent=2) + "\n")
    fields = list(history[0])
    with (run_dir / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    selected = torch.load(parent / "best.pt", map_location="cpu", weights_only=False)
    if selected["epoch"] > start_epoch:
        raise ValueError("Selected checkpoint cannot be later than continuation start")
    best_accuracy = selected["validation_accuracy"]
    best_epoch = selected["epoch"]
    best_state = {name: tensor.cpu().clone() for name, tensor in selected["state_dict"].items()}

    def payload(epoch, state):
        return {"state_dict": state, "epoch": epoch, "validation_accuracy": best_accuracy,
                "config": config, "training_indices": training_indices, "validation_indices": validation_indices}

    atomic_checkpoint(run_dir / "best.pt", payload(best_epoch, best_state))
    loss_fn = nn.CrossEntropyLoss()
    prior_seconds = history[-1]["elapsed_seconds"]
    started = time.perf_counter()
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Continuing epoch {start_epoch}; optimizer state restored: {original_state}", flush=True)
    for epoch in range(start_epoch + 1, args.max_epochs + 1):
        model.train()
        total_loss = torch.zeros((), device=args.device)
        correct = torch.zeros((), dtype=torch.long, device=args.device)
        for batch, (images, labels) in enumerate(loader, 1):
            images, labels = images.to(args.device), labels.to(args.device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            total_loss += loss.detach() * len(images)
            correct += (logits.detach().argmax(1) == labels).sum()
            if batch % 500 == 0:
                print(f"epoch {epoch}: batch {batch}/{len(loader)}", flush=True)
        model.eval()
        validation_loss = torch.zeros((), device=args.device)
        validation_correct = torch.zeros((), dtype=torch.long, device=args.device)
        with torch.inference_mode():
            for images, labels in validation_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                logits = model(images)
                validation_loss += loss_fn(logits, labels) * len(images)
                validation_correct += (logits.argmax(1) == labels).sum()
        row = {"epoch": epoch, "train_loss": total_loss.item() / len(training_indices),
               "train_accuracy": correct.item() / len(training_indices),
               "validation_loss": validation_loss.item() / len(validation_indices),
               "validation_accuracy": validation_correct.item() / len(validation_indices),
               "elapsed_seconds": prior_seconds + time.perf_counter() - started}
        if not all(math.isfinite(value) for value in row.values()):
            raise RuntimeError("Non-finite epoch metrics; previous checkpoints retained")
        history.append(row)
        if row["validation_accuracy"] > best_accuracy:
            best_accuracy, best_epoch = row["validation_accuracy"], epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            atomic_checkpoint(run_dir / "best.pt", payload(best_epoch, best_state))
        state = payload(epoch, {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()})
        state.update(validation_accuracy=row["validation_accuracy"],
                     optimizer_states=[optimizer.state_dict() for optimizer in optimizers],
                     cpu_rng_state=torch.get_rng_state())
        if args.device == "mps":
            state["mps_rng_state"] = torch.mps.get_rng_state()
        atomic_checkpoint(run_dir / "last.pt", state)
        with (run_dir / "metrics.csv").open("a", newline="") as file:
            csv.DictWriter(file, fieldnames=fields).writerow(row)
        status = divergence_status(history, origin_epoch)
        (run_dir / "divergence.json").write_text(json.dumps(status, indent=2) + "\n")
        print(f"epoch {epoch}: train loss {row['train_loss']:.4f}, train accuracy {row['train_accuracy']:.2%}, "
              f"validation loss {row['validation_loss']:.4f}, validation accuracy {row['validation_accuracy']:.2%} "
              f"(best epoch {best_epoch}; divergence streak {status['candidate_streak']}/3)", flush=True)
        if status["confirmed"]:
            break
    status["stop_reason"] = "sustained_divergence" if status["confirmed"] else "epoch_guard"
    (run_dir / "divergence.json").write_text(json.dumps(status, indent=2) + "\n")
    model.load_state_dict(best_state)
    model.run_config, model.training_history = config, history
    model.best_epoch, model.best_validation_accuracy = best_epoch, best_accuracy
    model.training_indices, model.validation_indices = training_indices, validation_indices
    del dataset, loader, validation_loader
    finish_run(root, run_dir, model, X_train, y_train, X_test, y_test, history[-1]["elapsed_seconds"] / 60, args.submit)


if __name__ == "__main__":
    main()
