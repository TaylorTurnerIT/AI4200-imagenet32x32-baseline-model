"""Analyze a completed ImageNet harness run; never called by graded train()."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from residual_mlp_baseline import training_core as core


@torch.inference_mode()
def accuracy(model, images, labels, indices):
    correct = 0
    for start in range(0, len(indices), 256):
        selected = indices[start:start + 256]
        predicted = core.predict(model, images[selected]).argmax(1).numpy()
        correct += int(np.sum(predicted == labels[selected]))
    return correct / len(indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["config"]["task"] != "imagenet32":
        raise ValueError("This analysis expects the final ImageNet32 model")
    model = core.ResidualMLP(**core.TASKS["imagenet32"]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    images = np.load(args.data / "X_train.npy", mmap_mode="c")
    labels = np.load(args.data / "y_train.npy", mmap_mode="c")
    indices = checkpoint["validation_indices"].numpy()
    full = accuracy(model, images, labels, indices)
    ablations = []
    for index, block in enumerate(model.blocks):
        handle = block.register_forward_hook(lambda module, inputs, output: inputs[0])
        try:
            ablated = accuracy(model, images, labels, indices)
        finally:
            handle.remove()
        ablations.append({"block": index, "accuracy": ablated,
                          "accuracy_drop_percentage_points": 100 * (full - ablated)})
        print(f"block {index}: full {full:.4%}, ablated {ablated:.4%}", flush=True)
    rows = json.loads((args.probe / "training-dynamics.json").read_text())
    selected = next(row for row in reversed(rows) if row["stage"] == "selected_checkpoint")
    receipt = {"task": "imagenet32", "selected_epoch": checkpoint["epoch"],
               "validation_examples": len(indices), "full_validation_accuracy": full,
               "ablations": ablations, "selected_branch_to_stream_ratio": selected["branch_to_stream_ratio"],
               "measurement_note": "Eval-mode fixed training batch; update ratios use copied optimizer state on a probe copy."}
    (args.output / "ablation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([r["block"] + 1 for r in ablations],
              [r["accuracy_drop_percentage_points"] for r in ablations], "o-")
    axis.set(xlabel="Residual block depth", ylabel="Validation accuracy drop (percentage points)",
             xticks=list(range(1, len(ablations) + 1)), title="Remove one residual branch without retraining")
    axis.axhline(0, color="gray", linewidth=0.7)
    figure.tight_layout()
    figure.savefig(args.output / "ablation-depth.png", dpi=300)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, key, title in zip(axes.flat,
            ["activation_variance", "branch_to_stream_ratio", "rms_gradient", "update_to_weight_ratio"],
            ["Activation variance", "Branch-to-stream RMS ratio", "RMS weight gradient", "Update-to-weight RMS ratio"]):
        for row in rows:
            values = list(row[key].values())
            axis.plot(range(1, len(values) + 1), np.maximum(values, 1e-12),
                      label=f"epoch {row['epoch']} ({row['stage']})")
        axis.set(title=title, xlabel="Layer / block in depth order", yscale="log")
    axes[0, 0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(args.output / "training-dynamics.png", dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()
