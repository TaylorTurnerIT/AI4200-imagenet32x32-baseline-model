"""Export accuracy-only experiment timelines and an interactive scale preview.

    uv run --locked python scripts/plot_experiment_accuracy.py
    uv run --locked python scripts/plot_experiment_accuracy.py --scale log-error

Read-only with respect to the evidence snapshot and LOG.md. The prior detailed
training-loss figures remain available through plot_development_eras.py.
"""

import argparse
import json
import math
import os
from pathlib import Path

from plot_development_eras import finite, observations


def chart_data(snapshot, config):
    tasks = {}
    view = config["accuracy_view"]
    for task, settings in config["tasks"].items():
        points = observations(snapshot, task, config)
        for metric in ["test", "validation"]:
            best = None
            for point in points:
                value = point[metric]
                if finite(value):
                    best = max(best, value) if best is not None else value
                point[f"best_{metric}"] = best
        tasks[task] = {
            "points": points, "eras": settings["eras"],
            "range": view["ranges"][task],
            "milestones": view["milestones"][task],
        }
    return {"tasks": tasks, "defaults": {"metric": view["metric"], "scale": view["scale"]}}


def limits(values, configured, scale):
    if scale == "clipped":
        return tuple(configured)
    if scale == "log-error":
        smallest_error = max(1e-4, 100 - max(values))
        largest_error = max(1e-4, 100 - min(values))
        padding = 10 ** max(0.03, math.log10(largest_error / smallest_error) * 0.08)
        return max(0, 100 - largest_error * padding), 100 - smallest_error / padding
    span = max(values) - min(values)
    padding = max(0.15, span * 0.08)
    return max(0, min(values) - padding), min(99.9999, max(values) + padding)


def export_figures(data, output, metric, scale, dpi=300):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    blue, gold = "#246BDE", "#D18B00"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#CCD1D8"})
    for task, task_data in data["tasks"].items():
        points = task_data["points"]
        scored = [p for p in points if finite(p[metric])]
        values = [100 * p[metric] for p in scored]
        low, high = limits(values, task_data["range"], scale)
        clipped = [p for p in scored if 100 * p[metric] < low]
        fig = plt.figure(figsize=(11.6, 5.2))
        ax = fig.add_axes([0.075, 0.28, 0.895, 0.53])
        if scale == "log-error":
            ax.set_yscale("function", functions=(
                lambda accuracy: -np.log10(np.maximum(100 - accuracy, 1e-6)),
                lambda error: 100 - 10.0 ** (-error),
            ))
            candidates = (MaxNLocator(nbins=6).tick_values(low, high) if max(values) < 90 else
                          [0, 10, 25, 50, 75, 90, 95, 97, 98, 99, 99.5, 99.9])
            ticks = [v for v in candidates
                     if low <= v <= high]
            transform = lambda value: -math.log10(max(100 - value, 1e-6))
            axis_height = fig.get_figheight() * ax.get_position().height * 72
            transformed_span = transform(high) - transform(low)
            spaced_ticks = []
            for tick in ticks:
                if not spaced_ticks or ((transform(tick) - transform(spaced_ticks[-1]))
                                        / transformed_span * axis_height >= 14):
                    spaced_ticks.append(tick)
            ax.set_yticks(spaced_ticks)
        ax.set(xlim=(0.5, len(points) + 0.5), ylim=(low, high),
               xlabel="Experiment attempt", ylabel="Accuracy (%)")
        if len(points) <= 8:
            ax.set_xticks([p["x"] for p in points])
        ax.grid(axis="y", color="#E7EAF0", linewidth=0.7)

        def draw_y(value):
            return min(high, max(low, value * 100))

        ax.plot([p["x"] for p in scored], [draw_y(p[metric]) for p in scored],
                color=gold, linewidth=1, linestyle=(0, (2, 3)), marker="o",
                markersize=3.6, zorder=3)
        best_points = [p for p in points if finite(p[f"best_{metric}"])]
        ax.step([p["x"] for p in best_points],
                [draw_y(p[f"best_{metric}"]) for p in best_points], where="post",
                color=blue, linewidth=2.5, zorder=4)
        records = [p for i, p in enumerate(scored)
                   if i == 0 or p[metric] > max(q[metric] for q in scored[:i])]
        ax.scatter([p["x"] for p in records], [draw_y(p[metric]) for p in records],
                   color=blue, s=30, zorder=5)
        ax.scatter([p["x"] for p in clipped], [low] * len(clipped), marker="v",
                   color=gold, s=44, zorder=6, clip_on=False)
        eras = {era["id"]: era for era in task_data["eras"]}
        spans = []
        for point in points:
            if not spans or spans[-1][0] != point["era"]:
                spans.append([point["era"], point["x"] - 0.5, point["x"] + 0.5])
            else:
                spans[-1][2] = point["x"] + 0.5
        for era_id, left, right in spans:
            if left > 0.5:
                ax.axvline(left, color="#B4BDC9", linewidth=0.8, linestyle=(0, (2, 4)))
            short = eras[era_id]["short"]
            if len(points) > 20 and right - left < 5:
                short = era_id.split(" ", 1)[0]
            ax.text((left + right) / 2, 1.04, short, transform=ax.get_xaxis_transform(),
                    fontsize=9, color="#677180", ha="center", va="bottom")
        by_id = {p["id"]: p for p in scored}
        notes = []
        for milestone in task_data["milestones"]:
            point = by_id.get(milestone["id"])
            if point is None:
                continue
            number = len(notes) + 1
            y = draw_y(point[metric])
            ax.annotate(str(number), (point["x"], y), xytext=(7, -18 if y > high - (high - low) * 0.12 else 16),
                        textcoords="offset points", ha="center", va="center", fontsize=9,
                        bbox={"boxstyle": "circle,pad=0.25", "fc": "white", "ec": "#C7CDD5", "lw": 0.7},
                        zorder=8)
            notes.append(f"{number}  {milestone['label']} · {100 * point[metric]:.3f}".rstrip("0").rstrip(".") + "%")
        title = {"mnist": "MNIST", "cifar10": "CIFAR-10", "imagenet32": "ImageNet32"}[task]
        best = max(values)
        fig.text(0.075, 0.94, title, fontsize=20)
        label = "Historical test accuracy" if metric == "test" else "Validation accuracy"
        mode = {"clipped": "clipped linear", "full": "full linear", "log-error": "log error spacing"}[scale]
        fig.text(0.075, 0.885, f"{label} · {mode}", fontsize=10, color="#677180")
        fig.text(0.97, 0.94, f"Best observed {best:.3f}".rstrip("0").rstrip(".") + "%",
                 ha="right", color=blue, fontsize=12)
        fig.legend(handles=[Line2D([0], [0], color=gold, linestyle="--", marker="o", markersize=4, label="Each experiment"),
                            Line2D([0], [0], color=blue, linewidth=2.5, label="Best observed so far")],
                   loc="upper left", bbox_to_anchor=(0.07, 0.17), ncol=2, frameon=False, fontsize=10)
        fig.text(0.075, 0.065, "    ·    ".join(notes), fontsize=9, color="#505A67")
        footer = f"{len(scored)} recorded results"
        if clipped:
            footer += f" · {len(clipped)} results below {low:g}% shown as downward triangles"
        if task == "imagenet32":
            footer += " · last result continues saved weights"
        fig.text(0.075, 0.018, footer, fontsize=8.5, color="#677180")
        for suffix in ["png", "svg"]:
            fig.savefig(output / f"{task}-accuracy.{suffix}", dpi=dpi, facecolor="white")
        plt.close(fig)
        print(f"{task}: {len(scored)} results, {len(clipped)} clipped; raw best={best:.4f}%")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=root / "outputs/experiment-history/all-runs.json")
    parser.add_argument("--eras", type=Path, default=root / "outputs/experiment-history/eras.json")
    parser.add_argument("--output", type=Path, default=root / "outputs/experiment-history/figures")
    parser.add_argument("--metric", choices=["test", "validation"], default="test")
    parser.add_argument("--scale", choices=["clipped", "full", "log-error"], default="clipped")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Raster export resolution (default: 300 DPI).")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(args.inventory.read_text())
    config = json.loads(args.eras.read_text())
    data = chart_data(snapshot, config)
    data["defaults"] = {"metric": args.metric, "scale": args.scale}
    export_figures(data, args.output, args.metric, args.scale, dpi=args.dpi)
    (args.output / "accuracy-data.json").write_text(json.dumps(data, indent=2) + "\n")
    template = root / "outputs/experiment-history/accuracy-template.html"
    html = template.read_text().replace("__ACCURACY_DATA__", json.dumps(data).replace("</", "<\\/"))
    (root / "outputs/experiment-history/accuracy-preview.html").write_text(html)


if __name__ == "__main__":
    main()
