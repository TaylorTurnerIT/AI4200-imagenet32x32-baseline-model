"""Plot the frozen baseline and single pre-reduction candidate's learning curves."""

import argparse
import csv
import json
import os
from pathlib import Path


def load_metrics(path):
    with path.open() as file:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=Path('outputs/final-evaluation/run-records/imagenet32-8856b095/metrics.csv'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    baseline = load_metrics(args.baseline)
    candidate = load_metrics(args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(args.output.parent / '.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for rows, name, color in [(baseline, 'Baseline', '#22756f'),
                              (candidate, 'Extra block', '#c77c20')]:
        epochs = [row['epoch'] for row in rows]
        for axis, metric, scale in [(axes[0], 'loss', 1), (axes[1], 'accuracy', 100)]:
            for split, style in [('train', '--'), ('validation', '-')]:
                axis.plot(epochs, [scale * row[f'{split}_{metric}'] for row in rows],
                          style, color=color, label=f'{name}: {split}', linewidth=1.5)
        selected = max(rows, key=lambda row: row['validation_accuracy'])
        axes[1].scatter(selected['epoch'], 100*selected['validation_accuracy'], color=color, s=35, zorder=4)
    axes[0].set(xlabel='Completed epoch', ylabel='Cross-entropy loss', title='ImageNet32: same seed and recipe')
    axes[1].set(xlabel='Completed epoch', ylabel='Accuracy (%)', title='One extra block before final reduction')
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    vector = args.output.with_suffix('.svg')
    figure.savefig(vector)
    vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines()) + '\n')
    plt.close(figure)
    summary_path = args.candidate.parent / 'summary.json'
    candidate_id = (json.loads(summary_path.read_text())['run_id']
                    if summary_path.exists() else args.candidate.parent.name)
    args.output.with_suffix('.json').write_text(json.dumps({
        'baseline_run': 'imagenet32-8856b095',
        'candidate_run': candidate_id,
        'note': 'One fresh candidate attempt, seed 0. Added pre-reduction capacity; remaining recipe and training-data split unchanged. Accuracy is validation or training-minibatch accuracy, not test accuracy.',
        'baseline': baseline, 'candidate': candidate,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
