"""Execute the one authorized fresh ImageNet candidate run, without test scoring."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import tarfile
import time
import traceback

import numpy as np
import torch

from pre_reduction_model import CandidateProbe, ResidualMLP
from residual_mlp_baseline import submission, training_core as core


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reserve_attempt(output, protocol):
    """Directory creation reserves the one attempt; never remove or reuse it."""
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "reserved", "allowed_attempts": 1,
               "reserved_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol}
    save_json(output / "attempt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    runtime = json.loads(args.runtime.read_text())
    assert runtime['training_data_matches_reference'] and runtime['prepared_without_training']
    assert torch.__version__ == '2.11.0+cu130' and torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == 'NVIDIA A100-SXM4-80GB'
    assert protocol['seed'] == 0 and protocol['allowed_attempts'] == 1
    assert protocol['epochs'] == 25 and protocol['max_seconds'] == 3600
    recipe = submission.RECIPES['imagenet32']
    assert recipe['optimizer'] == protocol['optimizer']
    assert submission.augmentation_for('imagenet32') == protocol['augmentation']
    images = np.load(args.data / 'X_train.npy', mmap_mode='c')
    labels = np.load(args.data / 'y_train.npy', mmap_mode='c')
    assert images.shape == (1281167, 3, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (1281167,) and labels.dtype == np.int64
    package = Path(inspect.getsourcefile(core)).parent
    for name, digest in protocol['baseline_source_sha256'].items():
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == digest, name
    for name, digest in protocol['experiment_source_sha256'].items():
        assert hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() == digest, name
    attempt = reserve_attempt(args.output, protocol)
    run = args.output / protocol['run_id']
    archive = args.output.with_name(args.output.name + '-evidence.tar.gz')
    try:
        shutil.copy2(args.protocol, args.output / 'protocol.json')
        shutil.copy2(args.runtime, args.output / 'runtime.json')
        sources = args.output / 'source'
        sources.mkdir()
        for file in (Path(__file__), Path(inspect.getsourcefile(ResidualMLP)),
                     Path(inspect.getsourcefile(core)), Path(inspect.getsourcefile(submission))):
            shutil.copy2(file, sources / file.name)
        attempt['status'] = 'running'
        attempt['started_utc'] = datetime.now(timezone.utc).isoformat()
        save_json(args.output / 'attempt.json', attempt)
        print(f"ONE_ATTEMPT_STARTED: {protocol['run_id']}", flush=True)
        observer = CandidateProbe(images, labels, 0, args.output / 'diagnostics')
        started = time.perf_counter()
        model = core.train_with_metrics(
            'imagenet32', images, labels, seed=0, device='cuda', run_dir=run,
            augmentation=protocol['augmentation'], optimizer_settings=protocol['optimizer'],
            epochs=25, model_factory=ResidualMLP, batch_size=256, max_seconds=3600,
            observer=observer,
        )
        seconds = time.perf_counter() - started
        history = model.training_history
        selected = next(row for row in history if row['epoch'] == model.best_epoch)
        minimum = min(history, key=lambda row: row['validation_loss'])
        baseline = protocol['baseline']
        summary = {
            'run_id': protocol['run_id'], 'task': 'imagenet32', 'seed': 0,
            'fresh_weights': True, 'test_set_evaluated': False,
            'selected_epoch': model.best_epoch, 'epochs_completed': len(history),
            'stop_reason': model.stop_reason, 'training_seconds': seconds,
            'best_validation_accuracy': model.best_validation_accuracy,
            'selected_metrics': selected, 'last_metrics': history[-1],
            'minimum_validation_loss': minimum['validation_loss'],
            'minimum_validation_loss_epoch': minimum['epoch'],
            'parameters': sum(parameter.numel() for parameter in model.parameters()),
            'added_parameters': 263424,
            'baseline_run_id': baseline['run_id'],
            'baseline_best_validation_accuracy': baseline['best_validation_accuracy'],
            'validation_gain_percentage_points': 100 * (model.best_validation_accuracy - baseline['best_validation_accuracy']),
            'baseline_minimum_validation_loss': baseline['minimum_validation_loss'],
            'baseline_training_seconds': baseline['training_seconds'],
            'primary_success': model.best_validation_accuracy > baseline['best_validation_accuracy'],
            'diagnostics_snapshots': len(observer.rows),
            'source_commit': protocol['source_commit'],
            'installed_baseline_origin': json.loads(importlib.metadata.distribution('residual-mlp-baseline').read_text('direct_url.json')),
            'comparison_note': 'One seed and one fresh candidate attempt; additional capacity is the architectural intervention. No automatic promotion.',
        }
        save_json(run / 'summary.json', summary)
        attempt['status'] = 'complete'
        attempt['summary'] = summary
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException:
        attempt['status'] = 'failed'
        attempt['failure'] = traceback.format_exc()
        (args.output / 'failure.txt').write_text(attempt['failure'])
        raise
    finally:
        attempt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save_json(args.output / 'attempt.json', attempt)
        with tarfile.open(archive, 'w:gz') as bundle:
            bundle.add(args.output, arcname=args.output.name)
        print(f"ONE_ATTEMPT_EVIDENCE: {archive}; status={attempt['status']}", flush=True)


if __name__ == '__main__':
    main()
