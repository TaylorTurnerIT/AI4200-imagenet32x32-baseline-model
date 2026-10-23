"""Leaderboard submission checks; all network calls are mocked."""

import ast
import csv
import io
import json
import tempfile
import unittest
import urllib.error
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import torch

from residual_mlp_baseline import notebook


class Response(io.BytesIO):
    status = 200


def form_page():
    titles = ['Alias', 'Section', 'Task', 'Accuracy', 'Train Minutes', 'Repository']
    questions = [[None, title, None, 0, [[number]]] for number, title in enumerate(titles)]
    data = [None, [None, questions]]
    return Response(f'<script>FB_PUBLIC_LOAD_DATA_ = {json.dumps(data)};</script>'.encode())


class LeaderboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.enabled = patch.object(notebook, 'LEADERBOARD_ENABLED', True)
        self.enabled.start()
        self.addCleanup(self.enabled.stop)

    def receipt(self):
        return json.loads((self.run / 'leaderboard-submission.json').read_text())

    def test_confirmed_submission_sends_expected_score_and_is_not_duplicated(self):
        with patch('urllib.request.urlopen', side_effect=[form_page(), Response(b'Your response has been recorded')]) as network:
            status = notebook.submit_run_to_leaderboard('mnist', .9915, 13.7, self.run)
            self.assertIn('confirmed as TaylorT, section 5000', status)
            request = network.call_args_list[1].args[0]
            payload = urllib.parse.parse_qs(request.data.decode())
            self.assertEqual(payload['entry.0'], ['TaylorT'])
            self.assertEqual(payload['entry.1'], ['5000'])
            self.assertEqual(payload['entry.2'], ['mnist'])
            self.assertEqual(payload['entry.3'], ['0.9915'])
            self.assertEqual(payload['entry.4'], ['13.7'])
            self.assertEqual(payload['entry.5'], [notebook.LEADERBOARD_REPOSITORY])
            self.assertTrue(request.full_url.endswith('/formResponse'))
            self.assertTrue(self.receipt()['confirmed'])
            self.assertTrue((self.run / 'leaderboard-confirmation.html').exists())
            again = notebook.submit_run_to_leaderboard('mnist', .9915, 13.7, self.run)
            self.assertIn('already submitted', again)
            self.assertEqual(network.call_count, 2)

    def test_network_failure_preserves_local_result(self):
        (self.run / 'metrics.csv').write_text('epoch,train_loss\n1,0.1\n')
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError('offline')) as network:
            status = notebook.submit_run_to_leaderboard('mnist', .9, 1, self.run)
            self.assertIn('not confirmed', status)
            self.assertEqual(self.receipt()['status'], 'failed')
            self.assertIn('1,0.1', (self.run / 'metrics.csv').read_text())
            self.assertEqual(network.call_count, 1)

    def test_ambiguous_post_is_not_automatically_retried(self):
        with patch('urllib.request.urlopen', side_effect=[form_page(), TimeoutError('response lost')]) as network:
            status = notebook.submit_run_to_leaderboard('mnist', .9, 1, self.run)
            self.assertIn('not confirmed', status)
            self.assertEqual(self.receipt()['status'], 'unconfirmed')
            notebook.submit_run_to_leaderboard('mnist', .9, 1, self.run)
            self.assertEqual(network.call_count, 2)

    def test_http_success_without_confirmation_is_not_reported_as_submitted(self):
        with patch('urllib.request.urlopen', side_effect=[form_page(), Response(b'Please correct the form')]):
            status = notebook.submit_run_to_leaderboard('mnist', .9, 1, self.run)
            self.assertIn('not confirmed', status)
            self.assertFalse(self.receipt()['confirmed'])

    def test_disabled_setting_does_not_contact_leaderboard(self):
        with patch.object(notebook, 'LEADERBOARD_ENABLED', False), patch('urllib.request.urlopen') as network:
            self.assertIn('disabled', notebook.submit_run_to_leaderboard('mnist', .9, 1, self.run))
            network.assert_not_called()
            self.assertFalse((self.run / 'leaderboard-submission.json').exists())

    def test_changed_form_is_not_submitted(self):
        with patch('urllib.request.urlopen', return_value=Response(b'<html>Different form</html>')) as network:
            self.assertIn('not confirmed', notebook.submit_run_to_leaderboard('mnist', .9, 1, self.run))
            self.assertEqual(network.call_count, 1)
            self.assertEqual(self.receipt()['status'], 'failed')

    def test_result_cell_posts_test_accuracy_after_local_artifacts_are_saved(self):
        source = Path(notebook.__file__).read_text()
        cell = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.args.args
                    and node.args.args[0].arg == 'NUM_CLASSES')
        cell.decorator_list = []
        results = self.run / 'results.csv'
        results.write_text('time,accuracy,old_note\n2026-01-01,0.5000,preserve me\n')
        history = [dict(epoch=1, train_loss=.1, train_accuracy=1., validation_loss=.2, validation_accuracy=.9)]
        model = SimpleNamespace(training_indices=torch.arange(5), validation_indices=torch.tensor([5]),
                                training_history=history,
                                best_epoch=1, best_validation_accuracy=.9,
                                run_config=dict(
                                    task='mnist', seed=42, device='cpu', epochs=2,
                                    batch_size=128, learning_rate=.05, weight_decay=.0001,
                                    momentum=0., dampening=0., optimizer='SGD',
                                    loss='CrossEntropyLoss()', learning_rate_schedule='constant', shuffle=True,
                                    validation_fraction=.1, training_examples=5, validation_examples=1,
                                    architecture=dict(width=128, hidden=128, depth=7, functional_operations=['relu']),
                                    augmentation={'enabled': False, 'digits': {}}, model='ResidualMLP()',
                                    model_source='class ResidualMLP: pass', block_source='class Block: pass',
                                ))

        def fake_train(*args, run_dir, **kwargs):
            run_dir.mkdir(parents=True)
            return model

        def check_submission(task, accuracy, minutes, run_dir):
            self.assertEqual(task, 'mnist')
            self.assertEqual(accuracy, .75)
            self.assertGreaterEqual(minutes, 0)
            with results.open() as file:
                rows = list(csv.DictReader(file))
                self.assertEqual(rows[0]['old_note'], 'preserve me')
                self.assertEqual(rows[0]['batch_size'], '')
                self.assertEqual(rows[1]['accuracy'], '0.7500')
                self.assertEqual(rows[1]['test_accuracy'], '0.7500')
                self.assertEqual(rows[1]['depth'], '7')
                self.assertEqual(json.loads(rows[1]['hyperparameters']), model.run_config)
            summary = (run_dir / 'summary.txt').read_text()
            self.assertIn('Test accuracy:       75.00%', summary)
            self.assertIn('Train accuracy:      100.00%', summary)
            self.assertIn('Validation accuracy: 90.00%', summary)
            self.assertIn('"batch_size": 128', summary)
            self.assertIn('relu', summary)
            saved = json.loads((run_dir / 'summary.json').read_text())
            self.assertEqual(saved['performance']['test_accuracy'], .75)
            self.assertEqual(saved['hyperparameters'], model.run_config)
            per_class = saved['per_class_accuracy']
            self.assertEqual(per_class, [
                dict(class_id=0, class_name='0', train_accuracy=1., validation_accuracy=1., test_accuracy=1.),
                dict(class_id=1, class_name='1', train_accuracy=None, validation_accuracy=None, test_accuracy=0.),
            ])
            markdown = (run_dir / 'per_class_accuracy.md').read_text()
            self.assertIn('| Class | Train | Validation | Test |', markdown)
            self.assertIn('| 0 | 100.00% | 100.00% | 100.00% |', markdown)
            self.assertIn('| 1 | N/A | N/A | 0.00% |', markdown)
            self.assertIn(markdown.strip(), summary)
            self.assertFalse((run_dir / 'per_class_accuracy.csv').exists())
            self.assertEqual(rows[1]['per_class_accuracy_file'],
                             str((run_dir / 'per_class_accuracy.md').relative_to(self.run)))
            self.assertTrue((run_dir / 'curves.png').exists())
            return 'Leaderboard submission confirmed as TaylorT, section 5000.'

        displayed = []
        namespace = dict(vars(notebook))
        namespace.update(
            mo=SimpleNamespace(stop=lambda *args: None, md=lambda value: value,
                               vstack=lambda value: displayed.extend(value),
                               status=SimpleNamespace(spinner=lambda **kwargs: nullcontext())),
            train_with_metrics=fake_train,
            predict=lambda model, images: torch.tensor([[1., 0.]]).repeat(len(images), 1),
        )
        with patch.dict('os.environ', {'MPLBACKEND': 'Agg'}):
            exec(compile(ast.Module(body=[cell], type_ignores=[]), notebook.__file__, 'exec'), namespace)
            with patch.dict(namespace, submit_run_to_leaderboard=check_submission), \
                    patch.object(notebook, 'predict', namespace['predict']):
                namespace['_'](
                    NUM_CLASSES={'mnist': 2}, RESULTS_FILE=results,
                    augmentation_digits=SimpleNamespace(value={}), augmentation_enabled=SimpleNamespace(value=False),
                    device=SimpleNamespace(value='cpu'), run_training=SimpleNamespace(value=True),
                    seed=SimpleNamespace(value=42), task=SimpleNamespace(value='mnist'),
                    load_task=lambda task: (np.zeros((6, 1, 28, 28)), np.zeros(6, dtype=int),
                                            np.zeros((4, 1, 28, 28)), np.array([0, 0, 0, 1])),
                )

        summary_path = next(self.run.glob('runs/*/summary.txt'))
        summary_text, _, markdown = summary_path.read_text().partition('\n\nPer-class accuracy:\n\n')
        self.assertIn(summary_text, displayed[0])
        self.assertIn(markdown.strip(), displayed[2])
        self.assertNotIn('| Class |', displayed[0])
        self.assertIn('| Class | Train | Validation | Test |', displayed[2])
        self.assertIn('| 1 | N/A | N/A | 0.00% |', displayed[2])


if __name__ == '__main__':
    unittest.main()
