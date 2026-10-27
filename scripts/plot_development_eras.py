"""Render draft development figures from the portable experiment snapshot.

Run from any directory:
    python scripts/plot_development_eras.py

Edit outputs/experiment-history/eras.json to change era labels, representative
runs or annotations. Refresh all-runs.json when incorporating later experiments.
This script reads historical evidence only; it never trains or loads weights.
"""

import argparse
import json
import math
import os
from pathlib import Path


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def observations(snapshot, task, era_config):
    """Retain every attempt, including unfinished runs and old CSV-only results."""
    points = []
    for legacy in snapshot["legacy_rows"]:
        row = legacy["raw"]
        if row["task"] != task:
            continue
        points.append({
            "id": f"legacy-csv-line-{legacy['csv_line']}",
            "time": row["time"],
            "test": float(row["accuracy"]),
            "validation": None,
            "completed": True,
        })
    for run in snapshot["runs"]:
        if run["task"] != task:
            continue
        points.append({
            "id": run["id"], "time": run["time"],
            "test": run["performance"].get("test_accuracy"),
            "validation": run["performance"].get("validation_accuracy"),
            "completed": run["completed"],
            "era": run["era"],
            "cohort_order": run["result"].get("time", "") if "-audit-" in run["id"] else "",
        })
    points.sort(key=lambda point: (point["time"], point.get("cohort_order", ""), point["id"]))
    for index, point in enumerate(points, 1):
        point["x"] = index
        if point["id"] in era_config.get("run_overrides", {}):
            point["era"] = era_config["run_overrides"][point["id"]]
        elif "era" not in point:
            candidates = [era for era in era_config["tasks"][task]["eras"]
                          if era["start"] <= point["time"]]
            point["era"] = candidates[-1]["id"]
    return points


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path,
                        default=root / "outputs/experiment-history/all-runs.json")
    parser.add_argument("--eras", type=Path,
                        default=root / "outputs/experiment-history/eras.json")
    parser.add_argument("--output", type=Path,
                        default=root / "outputs/experiment-history/figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / ".matplotlib"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    snapshot = json.loads(args.inventory.read_text())
    config = json.loads(args.eras.read_text())
    runs = {run["id"]: run for run in snapshot["runs"]}
    # Export figures use a fixed, color-blind-friendly print palette.
    blue, gray = "#0072B2", "#686868"
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#30343A", "text.color": "#20252B",
        "xtick.color": "#525861", "ytick.color": "#525861",
        "axes.edgecolor": "#D3D8DD", "grid.color": "#E4E8EC",
        "axes.titleweight": "normal", "savefig.facecolor": "white",
        "svg.fonttype": "none",
    })
    exported = {}
    for task, task_config in config["tasks"].items():
        points = observations(snapshot, task, config)
        by_id = {point["id"]: point for point in points}
        era_lookup = {era["id"]: era for era in task_config["eras"]}
        fig = plt.figure(figsize=(10.5, 10.5))
        accuracy = fig.add_axes([0.095, 0.59, 0.875, 0.25])
        loss = fig.add_axes([0.095, 0.22, 0.875, 0.25])
        fig.suptitle(task_config["title"], x=0.085, y=0.975,
                     ha="left", fontsize=18, fontweight="normal")
        fig.text(0.085, 0.936, "Development eras · design draft · evidence through October 27, 2026",
                 color="#626A73", fontsize=11)
        fig.text(0.085, 0.900, task_config["subtitle"], fontsize=10.5)

        # Era boundaries follow chronological attempts, not artificial equal widths.
        spans = []
        for point in points:
            if not spans or spans[-1][0] != point["era"]:
                spans.append([point["era"], point["x"] - 0.5, point["x"] + 0.5])
            else:
                spans[-1][2] = point["x"] + 0.5
        for index, (era_id, left, right) in enumerate(spans):
            accuracy.axvspan(left, right, color="#34495E",
                            alpha=0.025 if index % 2 == 0 else 0.075, zorder=0)
            if index:
                accuracy.axvline(left, color="#CDD3DA", linewidth=0.8, zorder=1)
            short = era_lookup[era_id]["short"]
            if len(points) > 20 and right - left < 5:
                short = era_id.split(" ", 1)[0]
            accuracy.text((left + right) / 2, 1.035, short,
                          transform=accuracy.get_xaxis_transform(), ha="center",
                          va="bottom", fontsize=9)

        x = [point["x"] for point in points]
        tests = [100 * point["test"] if finite(point["test"]) else math.nan for point in points]
        vals = [100 * point["validation"] if finite(point["validation"]) else math.nan for point in points]
        accuracy.plot(x, tests, color=gray, linestyle=(0, (3, 2)), linewidth=1.15,
                      marker="o", markersize=3, label="Historical test")
        accuracy.plot(x, vals, color=blue, linewidth=1.7, marker="o",
                      markersize=4, label="Validation")
        unfinished = [point["x"] for point in points if not point["completed"]]
        accuracy.scatter(unfinished, [-0.14] * len(unfinished), marker="x", s=24,
                         color=gray, transform=accuracy.get_xaxis_transform(), clip_on=False)
        accuracy.set(xlim=(0.4, len(points) + 0.6), ylabel="Accuracy (%)",
                     xlabel="Experiment attempts in chronological order (including unfinished attempts)")
        accuracy.xaxis.labelpad = 20
        accuracy.grid(axis="y", linewidth=0.7)
        observed = [v for v in tests + vals if math.isfinite(v)]
        padding = max(1.5, (max(observed) - min(observed)) * 0.12)
        accuracy.set_ylim(max(0, min(observed) - padding), min(100.6, max(observed) + padding))
        if len(points) <= 8:
            accuracy.set_xticks(x)
        for annotation in task_config["annotations"]:
            point = by_id[annotation["id"]]
            value = point.get(annotation.get("metric", "test"))
            assert finite(value), annotation
            accuracy.annotate(annotation["label"], (point["x"], value * 100),
                              xytext=annotation["offset"], textcoords="offset points",
                              fontsize=9, ha=annotation.get("align", "left"),
                              arrowprops={"arrowstyle": "-", "color": "#89939D", "lw": 0.7})

        if task == "mnist":
            # Show the entire failure range and a conventional linear zoom; no broken axis.
            zoom = accuracy.inset_axes([0.49, 0.13, 0.47, 0.60])
            zoom.plot(x, tests, color=gray, linestyle="--", linewidth=0.9, marker="o", markersize=2)
            zoom.plot(x, vals, color=blue, linewidth=1.1, marker="o", markersize=2)
            zoom.set(xlim=accuracy.get_xlim(), ylim=(95, 100), yticks=[95, 97, 99],
                     title="Detail: 95–100% accuracy")
            zoom.set_title("Detail: 95–100% accuracy", fontsize=9)
            zoom.tick_params(labelsize=8)
            zoom.grid(axis="y", linewidth=0.5)

        # A fresh continuation optimizer creates a boundary: show only new epochs.
        representative_manifest = []
        handles = []
        for index, representative in enumerate(task_config["representatives"]):
            run = runs[representative["id"]]
            assert run["task"] == task
            rows = run["metrics"]
            origin = run["config"].get("continuation", {}).get("origin_epoch")
            if origin is not None:
                rows = [row for row in rows if row.get("epoch", 0) > origin]
            assert rows, representative
            progress = [100 * (i + 1) / len(rows) for i in range(len(rows))]
            color = colors[index % len(colors)]
            for column, style in [("train_loss", "-"), ("validation_loss", "--")]:
                values = [row.get(column) for row in rows]
                assert all(finite(value) and value > 0 for value in values), representative
                loss.plot(progress, values, color=color, linestyle=style,
                          linewidth=1.45, alpha=0.9)
            label = representative["label"]
            handles.append(Line2D([0], [0], color=color, lw=2, label=label))
            representative_manifest.append({"id": run["id"], "label": label,
                                             "epochs": [row["epoch"] for row in rows]})
        loss.set(xlabel="Training progress (% of recorded epochs within each displayed run)",
                 ylabel="Recorded loss (log scale)", xlim=(0, 101))
        loss.set_yscale("log")
        loss.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        loss.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        loss.yaxis.set_minor_formatter(NullFormatter())
        loss.grid(axis="y", which="major", linewidth=0.6)
        loss.set_title("Representative loss curves · solid: training · dashed: validation",
                       loc="left", fontsize=11, pad=11)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.535, 0.125),
                   ncol=2, frameon=False, fontsize=9)
        fig.legend(handles=[Line2D([0], [0], color=blue, lw=1.7, marker="o", markersize=4,
                                        label="Validation"),
                                  Line2D([0], [0], color=gray, linestyle="--", marker="o", markersize=3,
                                         label="Historical test"),
                                  Line2D([0], [0], color=gray, linestyle="None", marker="x",
                                         label="Unfinished; no final score")],
                        loc="upper left", bbox_to_anchor=(0.095, 0.515), ncol=3,
                        frameon=False, fontsize=9, borderaxespad=0)
        fig.text(0.085, 0.046, task_config["footnote"], fontsize=9, color="#626A73")
        fig.text(0.085, 0.025,
                 "Test scores are historical observations. Era comparisons change multiple settings; they are not controlled ablations.",
                 fontsize=9, color="#626A73")
        for extension in ["png", "svg"]:
            fig.savefig(args.output / f"{task}-eras.{extension}", dpi=180)
        plt.close(fig)
        exported[task] = {"attempts": points, "representatives": representative_manifest}
        print(f"{task}: {len(points)} attempts; {len(unfinished)} unfinished; "
              f"{len(representative_manifest)} representative loss curves")
    (args.output / "plotted-data.json").write_text(json.dumps(exported, indent=2) + "\n")


if __name__ == "__main__":
    main()
