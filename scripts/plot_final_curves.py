"""Compare an earlier saved CIFAR-10 experiment with the final harness run."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("outputs/experiment-history/all-runs.json"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_id = "cifar10-20261023-112130-b25190a4"
    source = next(row for row in json.loads(args.history.read_text())["runs"] if row["id"] == source_id)
    with (args.run / "metrics.csv").open() as file:
        final = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for rows, label, color in [(source["metrics"], "Earlier saved MLP", "#7b6b97"),
                               (final, "Final harness", "#217b72")]:
        epochs = [row["epoch"] for row in rows]
        for axis, metric, scale in [(axes[0], "loss", 1), (axes[1], "accuracy", 100)]:
            for split, style in [("train", "--"), ("validation", "-")]:
                axis.plot(epochs, [scale * row[f"{split}_{metric}"] for row in rows],
                          style, color=color, label=f"{label}: {split}", linewidth=1.5)
    axes[0].set(xlabel="Completed epoch", ylabel="Cross-entropy loss", title="CIFAR-10 learning curves")
    axes[1].set(xlabel="Completed epoch", ylabel="Accuracy (%)", title="Training and validation")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    vector = args.output.with_suffix(".svg")
    figure.savefig(vector)
    vector.write_text("\n".join(line.rstrip() for line in vector.read_text().splitlines()) + "\n")
    plt.close(figure)
    args.output.with_suffix(".json").write_text(json.dumps({
        "earlier_run": source_id, "final_run": args.run.name,
        "note": "Different recipes, seeds and hardware; this comparison does not isolate one change. Training accuracy is measured on training minibatches, augmented when enabled.",
        "earlier_metrics": source["metrics"], "final_metrics": final,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
