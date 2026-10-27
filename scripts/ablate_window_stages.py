"""Ablate spatial mixing in saved window MLPs without retraining.

A window stage changes both spatial size and channels, so it has no identity
bypass. These controls retain its learned channel mapper and exact geometry:
``single_pixel`` repeats the top-left pixel over all four window positions;
``mean_only`` repeats the window's spatial mean over those positions.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from analyze_submission import accuracy
from residual_mlp_baseline import training_core as core


MODES = ("single_pixel", "mean_only")


def controlled_window_output(stage, images, mode):
    """Replace spatial information while keeping the trained stage's layers."""
    if mode not in MODES:
        raise ValueError(f"Unknown window ablation: {mode}")
    windows = stage.extract_windows(images)
    pixels = windows.reshape(*windows.shape[:-1], images.shape[1], 4)
    representative = (pixels[..., :1] if mode == "single_pixel"
                      else pixels.mean(dim=-1, keepdim=True))
    controlled = representative.expand_as(pixels).reshape_as(windows)
    for layer in stage:
        controlled = layer(controlled)
    return controlled.permute(0, 3, 1, 2)


@contextmanager
def spatial_control(stage, mode):
    """Restore the original stage even if evaluation raises an exception.

    The hook recomputes the controlled stage after its ordinary forward. These
    stages contain only Linear/GELU modules, with no running statistics or RNG.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown window ablation: {mode}")
    handle = stage.register_forward_hook(
        lambda module, inputs, output: controlled_window_output(module, inputs[0], mode))
    try:
        yield
    finally:
        handle.remove()


def weight_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_receipt(path, receipt):
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


def plot_receipt(receipt, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = {"single_pixel": "Single pixel (top-left)", "mean_only": "Window mean only"}
    for mode in MODES:
        rows = [row for row in receipt["ablations"] if row["control"] == mode]
        depths = [row["stage"] for row in rows]
        axes[0].plot(depths, [100 * row["accuracy"] for row in rows], "o-", label=labels[mode])
        axes[1].plot(depths, [row["accuracy_drop_percentage_points"] for row in rows], "o-", label=labels[mode])
    axes[0].axhline(100 * receipt["full_validation_accuracy"], color="gray", linestyle="--", label="Full model")
    axes[0].set(ylabel="Validation accuracy (%)", title="Replace spatial information at one stage")
    axes[1].axhline(0, color="gray", linewidth=0.7)
    axes[1].set(ylabel="Accuracy drop (percentage points)", title="Drop from the same-backend full model")
    for axis in axes:
        axis.set(xlabel="Window stage depth", xticks=list(range(1, 7)))
        axis.grid(alpha=0.18)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "window-stage-ablation.png", dpi=300)
    vector = output / "window-stage-ablation.svg"
    figure.savefig(vector)
    vector.write_text("\n".join(line.rstrip() for line in vector.read_text().splitlines()) + "\n")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["config"]["task"] != "imagenet32":
        raise ValueError("Expected the selected ImageNet32 checkpoint")
    model = core.ResidualMLP(**core.TASKS["imagenet32"]).to(args.device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    images = np.load(args.data / "X_train.npy", mmap_mode="c")
    labels = np.load(args.data / "y_train.npy", mmap_mode="c")
    indices = checkpoint["validation_indices"].numpy()
    before = weight_digest(model)
    started = time.perf_counter()
    full = accuracy(model, images, labels, indices)
    reference = checkpoint["validation_accuracy"]
    if abs(full - reference) > 0.001:
        raise ValueError(f"Restored validation accuracy {full:.8f} differs from checkpoint {reference:.8f}")
    receipt = {
        "analysis": "Window spatial-information controls; learned channel mappers retained",
        "status": "running", "completed_utc": None,
        "run_id": args.run.name, "selected_epoch": checkpoint["epoch"],
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "weight_sha256_before": before, "device": args.device,
        "python": platform.python_version(), "torch": torch.__version__,
        "validation_examples": len(indices),
        "validation_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "full_validation_accuracy": full, "checkpoint_cuda_validation_accuracy": reference,
        "baseline_difference_percentage_points": 100 * (full - reference),
        "controls": {
            "single_pixel": "Repeat the top-left pixel at all four window positions before the unchanged learned MLP",
            "mean_only": "Repeat the padded window's spatial mean at all four positions before the unchanged learned MLP",
        },
        "intervention_note": "One window stage at a time; original padding, strides, output shapes and learned weights retained. These are spatial-information ablations, not complete stage deletion.",
        "ablations": [],
    }
    result = args.output / "window-stage-ablation.json"
    write_receipt(result, receipt)
    print(f"full validation: {full:.6%}; {len(indices)} examples", flush=True)
    for index, stage in enumerate(model.window_mlps):
        for mode in MODES:
            with spatial_control(stage, mode):
                score = accuracy(model, images, labels, indices)
            row = {"stage": index + 1, "control": mode, "accuracy": score,
                   "accuracy_drop_percentage_points": 100 * (full - score),
                   "input_shape": list(model.window_shapes[index]),
                   "output_shape": list(model.window_shapes[index + 1]),
                   "stride": stage.stride, "padding": stage.padding}
            receipt["ablations"].append(row)
            write_receipt(result, receipt)
            print(f"stage {index + 1}, {mode}: {score:.6%}; drop {row['accuracy_drop_percentage_points']:.4f} points", flush=True)
    receipt["weight_sha256_after"] = weight_digest(model)
    assert receipt["weight_sha256_after"] == before, "Ablation changed model weights"
    receipt["restored_validation_accuracy"] = accuracy(model, images, labels, indices)
    assert receipt["restored_validation_accuracy"] == full, "Hooks were not fully restored"
    receipt["status"] = "complete"
    receipt["elapsed_seconds"] = time.perf_counter() - started
    receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_receipt(result, receipt)
    plot_receipt(receipt, args.output)


if __name__ == "__main__":
    main()
