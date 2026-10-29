"""Archive the results after the unchanged instructor evaluation cells finish."""

import tarfile

assert builtins.print is _baseline_original_print
assert student.train is _baseline_original_train
assert student.predict is _baseline_original_predict
assert len(results) == 1 and results[0][0] == 'imagenet32'
assert [row[0] for row in _baseline_epoch_rows] == list(range(1, 11))
_task, _accuracy, _minutes = results[0]
_summary = {
    'run_id': _baseline_protocol['run_id'], 'task': _task,
    'source_commit': _baseline_protocol['source_commit'],
    'installed_origin': _baseline_origin, 'source_sha256': _baseline_source,
    'seed': SEED, 'device': DEVICE, 'fresh_weights': True,
    'test_accuracy': float(_accuracy), 'test_correct': int(round(float(_accuracy)*50_000)),
    'test_examples': 50_000, 'training_minutes': float(_minutes),
    'epochs_completed': 10, 'checkpoint': 'final epoch; no validation selection',
    'training_examples': 1_281_167, 'validation_split': None,
    'harness_probability_checks': 'passed',
    'additional_finite_nonnegative_check': 'passed',
    'leaderboard_alias': LEADERBOARD_ALIAS,
    'epoch_loss_precision': 'four decimals, as printed by original trainer',
    'initial_epoch_train_loss_printed': _baseline_epoch_rows[0][1],
    'final_epoch_train_loss_printed': _baseline_epoch_rows[-1][1],
    'recovered_after_development': True,
    'completed_utc': datetime.now(timezone.utc).isoformat(),
}
(_baseline_out/'summary.json').write_text(json.dumps(_summary, indent=2)+'\n')
(_baseline_out/'attempt.json').write_text(json.dumps({
    'run_id': _summary['run_id'], 'status': 'complete',
    'completed_utc': _summary['completed_utc'], 'epochs_completed': 10,
}, indent=2)+'\n')
_hashes = {str(p.relative_to(_baseline_out)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(_baseline_out.rglob('*')) if p.is_file()}
(_baseline_out/'artifact-sha256.json').write_text(json.dumps(_hashes, indent=2)+'\n')
with tarfile.open('/content/original-baseline-evidence.tar.gz', 'w:gz') as _archive:
    _archive.add(_baseline_out, arcname=_baseline_out.name)
print(json.dumps(_summary, indent=2))
print('ORIGINAL_BASELINE_EVIDENCE_READY')
