import hashlib
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import numpy as np
import torch

PIN = 'b5ab6e1f75de4cfc257cf85b7e2535cfa217934c'
root = Path('/content')
dist = importlib.metadata.distribution('residual-mlp-baseline')
origin = json.loads(dist.read_text('direct_url.json'))
assert origin['vcs_info']['commit_id'] == PIN, origin
assert DEVICE == 'cuda' and SEED == 0 and LEADERBOARD_ALIAS == ''
assert [r[0] for r in results] == ['mnist', 'cifar10', 'imagenet32']
import residual_mlp_baseline
package = Path(residual_mlp_baseline.__file__).parent
receipt = {
    'submission_commit': PIN, 'installed_origin': origin,
    'completed_utc': datetime.now(timezone.utc).isoformat(),
    'installed_package': str(package), 'python': platform.python_version(),
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0), 'seed': SEED, 'device': DEVICE,
    'leaderboard_alias': LEADERBOARD_ALIAS, 'section': SECTION,
    'harness_output_assertions': 'passed for all three tasks',
    'source_sha256': {name: hashlib.sha256((package/name).read_bytes()).hexdigest()
                      for name in ('submission.py', 'training_core.py', 'analysis.py', 'train.py', '__init__.py')},
    'results': [],
}
for task, accuracy, minutes in results:
    runs = list((root/'ai4200-evidence').glob(task+'-*'))
    assert len(runs) == 1, runs
    run = runs[0]
    row = {'task': task, 'test_accuracy': float(accuracy), 'train_minutes': float(minutes),
           'run_id': run.name, 'training_receipt': json.loads((run/'training-receipt.json').read_text()),
           'recipe': json.loads((run/'recipe.json').read_text())}
    receipt['results'].append(row)
receipt['total_train_minutes'] = sum(r['train_minutes'] for r in receipt['results'])
receipt['within_90_minutes'] = receipt['total_train_minutes'] <= 90
(root/'ai4200-qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2), flush=True)
run = next((root/'ai4200-evidence').glob('imagenet32-*'))
probe = next((root/'ai4200-analysis').glob('imagenet32-*'))
command = [sys.executable, '/content/analyze_submission.py', '--run', str(run),
           '--data', '/content/data/imagenet32', '--probe', str(probe),
           '--output', '/content/ai4200-final-analysis', '--device', 'cuda']
analysis = subprocess.run(command, check=False)
receipt['analysis_exit_code'] = analysis.returncode
if analysis.returncode == 0:
    ablation = json.loads((root/'ai4200-final-analysis/ablation.json').read_text())
    expected = next(row for row in receipt['results'] if row['task'] == 'imagenet32')['training_receipt']['best_validation_accuracy']
    receipt['analysis_validation_matches_selected_checkpoint'] = abs(ablation['full_validation_accuracy'] - expected) < 1e-12
(root/'ai4200-qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
with tarfile.open(root/'ai4200-final-evidence.tar.gz','w:gz') as archive:
    for name in ('ai4200-evidence','ai4200-analysis','ai4200-final-analysis',
                 'ai4200-runtime.json','ai4200-qualification.json','results.csv'):
        path = root/name
        if path.exists(): archive.add(path,arcname=name)
assert analysis.returncode == 0, 'Analysis failed; partial evidence archive preserved'
assert receipt['analysis_validation_matches_selected_checkpoint'], 'Analysis validation differs from selected checkpoint; evidence preserved'
print('FINAL_EVIDENCE_READY', flush=True)
