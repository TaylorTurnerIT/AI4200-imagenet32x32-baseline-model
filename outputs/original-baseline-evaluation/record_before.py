"""Observe the original harness without changing its model or training recipe."""

import builtins
import csv as evidence_csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import shutil

_baseline_out = Path('/content/original-baseline-evidence')
_baseline_out.mkdir(exist_ok=True)
_baseline_protocol = BASELINE_PROTOCOL
assert DEVICE == 'cuda' and SEED == 0
assert TASKS_TO_RUN == ['imagenet32'] and LEADERBOARD_ALIAS == ''
_baseline_package = Path(inspect.getfile(student)).parent
_baseline_origin = json.loads(importlib.metadata.distribution('residual-mlp-baseline').read_text('direct_url.json'))
assert _baseline_origin['vcs_info']['commit_id'] == _baseline_protocol['source_commit']
_baseline_source = {}
(_baseline_out/'source').mkdir(exist_ok=True)
for _name, _expected in _baseline_protocol['package_source_sha256'].items():
    _raw = (_baseline_package/_name).read_bytes()
    assert hashlib.sha256(_raw).hexdigest() == _expected, _name
    (_baseline_out/'source'/_name).write_bytes(_raw)
    _baseline_source[_name] = _expected
assert student.train.__module__ == 'residual_mlp_baseline.train'
_baseline_original_train = student.train
_baseline_original_predict = student.predict

_baseline_data = {}
for _part in ['X_train', 'y_train', 'X_test', 'y_test']:
    _array = np.load(Path(DATA_DIR)/'imagenet32'/f'{_part}.npy', mmap_mode='r')
    _baseline_data[_part] = {
        'shape': list(_array.shape), 'dtype': str(_array.dtype),
        'array_sha256': hashlib.sha256(memoryview(_array)).hexdigest(),
    }
    if _part in _baseline_protocol['expected_training_data']:
        assert _baseline_data[_part] == _baseline_protocol['expected_training_data'][_part], _part
    del _array
(_baseline_out/'data.json').write_text(json.dumps(_baseline_data, indent=2)+'\n')
(_baseline_out/'protocol.json').write_text(json.dumps(_baseline_protocol, indent=2)+'\n')
shutil.copyfile('/content/original-baseline-runtime.json', _baseline_out/'runtime.json')
with (_baseline_out/'attempt.json').open('x') as _f:
    json.dump({'run_id': _baseline_protocol['run_id'], 'status': 'running',
               'started_utc': datetime.now(timezone.utc).isoformat()}, _f, indent=2)
with (_baseline_out/'metrics.csv').open('x', newline='') as _f:
    evidence_csv.writer(_f).writerow(['epoch', 'train_loss_printed'])

# The original trainer prints one rounded loss per epoch. Mirror those values;
# do not add forward/backward passes or alter the trainer's function.
_baseline_epoch_rows = []
_baseline_original_print = builtins.print
def _baseline_observe_print(*args, **kwargs):
    _baseline_original_print(*args, **kwargs)
    if len(args) == 1 and isinstance(args[0], str):
        _match = re.fullmatch(r'epoch (\d+): train loss ([0-9.]+)', args[0])
        if _match:
            _row = (int(_match[1]), _match[2])
            _baseline_epoch_rows.append(_row)
            with (_baseline_out/'metrics.csv').open('a', newline='') as _f:
                evidence_csv.writer(_f).writerow(_row)
builtins.print = _baseline_observe_print

def _baseline_archive_predict(model, X):
    # This runs after the harness stops its training timer. Retain the final
    # weights; the original predict still performs the only test forward pass.
    builtins.print = _baseline_original_print
    assert student.train is _baseline_original_train
    assert len(_baseline_epoch_rows) == 10
    assert len(X) == 50_000
    _configuration = {
        'run_id': _baseline_protocol['run_id'], 'task': 'imagenet32',
        'source_commit': _baseline_protocol['source_commit'],
        'recipe': _baseline_protocol['recipe'],
        'model_repr': repr(model),
        'parameters': sum(p.numel() for p in model.parameters()),
    }
    (_baseline_out/'config.json').write_text(json.dumps(_configuration, indent=2)+'\n')
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'config': _configuration}, _baseline_out/'final.pt')
    _probs = _baseline_original_predict(model, X)
    assert torch.isfinite(_probs).all() and (_probs >= 0).all()
    torch.save(_probs.argmax(dim=1), _baseline_out/'test-predictions.pt')
    student.predict = _baseline_original_predict
    return _probs
student.predict = _baseline_archive_predict
print('Original baseline verified; training has not started.')
