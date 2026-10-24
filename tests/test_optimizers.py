"""Optimizer selection, parameter ownership, training, and run provenance."""

import ast
import csv
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from residual_mlp_baseline import notebook


class OptimizerTests(unittest.TestCase):
    def test_all_optimizers_train_each_dataset_and_save_effective_settings(self):
        for name in notebook.available_optimizer_names():
            for task in notebook.TASKS:
                with self.subTest(optimizer=name, task=task), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    shape = (1, 28, 28) if task == 'mnist' else (3, 32, 32)
                    images = np.random.default_rng(42).integers(0, 256, (12, *shape), dtype=np.uint8)
                    labels = np.arange(12) % 3
                    config = notebook.optimizer_config(name)
                    config['lr'] *= .5
                    with patch.dict(notebook.EPOCHS, {task: 1}), \
                            patch.object(notebook, 'ResidualMLP', partial(notebook.ResidualMLP, width=8, hidden=16)):
                        model = notebook.train_with_metrics(task, images, labels, device='cpu', run_dir=root/'run',
                                                            optimizer_settings=config)
                    saved = json.loads((root/'run'/'config.json').read_text())
                    self.assertEqual(saved['optimizer'], name)
                    self.assertEqual(saved['optimizer_settings'], config)
                    self.assertEqual(saved['learning_rate'], config['lr'])
                    self.assertEqual(saved['optimizer_groups'][0]['lr'], config['lr'])
                    self.assertEqual(saved['optimizer_groups'][0]['optimizer'], name)
                    self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
                    self.assertTrue(np.isfinite(model.training_history[0]['train_loss']))
                    names = [key for group in saved['optimizer_groups'] for key in group['parameter_names']]
                    self.assertEqual(len(names), len(set(names)))
                    self.assertEqual(set(names), set(dict(model.named_parameters())))
                    checkpoint = torch.load(root/'run'/'best.pt', weights_only=True)
                    self.assertEqual(checkpoint['config']['optimizer_groups'], saved['optimizer_groups'])
                    summary = notebook.save_run_result(root/'results.csv', root/'run', model, 0., 0., .1, 4)
                    self.assertIn(f'"optimizer": "{name}"', summary)
                    with (root/'results.csv').open() as file:
                        row = next(csv.DictReader(file))
                    self.assertEqual(json.loads(row['optimizer_settings']), config)
                    self.assertEqual(json.loads(row['optimizer_groups']), saved['optimizer_groups'])

    @unittest.skipUnless(hasattr(torch.optim, 'Muon'), 'Native Muon unavailable')
    def test_muon_splits_matrices_and_other_parameters_without_overlap(self):
        model = notebook.ResidualMLP(in_features=784, num_classes=10, width=8, hidden=16)
        optimizers, config, groups = notebook.build_optimizers(model, {'name': 'Muon'})
        self.assertEqual([type(optimizer).__name__ for optimizer in optimizers], ['Muon', 'AdamW'])
        primary, auxiliary = map(lambda group: set(group['parameter_names']), groups)
        self.assertFalse(primary & auxiliary)
        self.assertTrue(primary)
        self.assertIn('input_layer.weight', auxiliary)
        self.assertIn('head_fc2.weight', auxiliary)
        self.assertIn('head_norm.weight', auxiliary)
        self.assertIn('blocks.0.fc1.weight', primary)
        self.assertIn('head_fc1.weight', primary)
        self.assertTrue(all(dict(model.named_parameters())[key].ndim == 2 for key in primary))
        self.assertEqual(groups[1]['lr'], config['auxiliary_lr'])
        self.assertEqual(groups[1]['betas'], [config['auxiliary_beta1'], config['auxiliary_beta2']])
        before = {key: parameter.detach().clone() for key, parameter in model.named_parameters()}
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        for optimizer in optimizers:
            optimizer.step()
        self.assertTrue(all(not torch.equal(parameter, before[key]) for key, parameter in model.named_parameters()))

    def test_missing_muon_is_not_offered_or_silently_replaced(self):
        with patch.object(torch.optim, 'Muon', create=True):
            del torch.optim.Muon
            self.assertNotIn('Muon', notebook.available_optimizer_names())
            with self.assertRaisesRegex(ValueError, 'requires a PyTorch build'):
                notebook.optimizer_config('Muon')

    def test_bad_optimizer_names_and_settings_fail(self):
        model = torch.nn.Linear(3, 2)
        with self.assertRaisesRegex(ValueError, 'Unknown optimizer'):
            notebook.build_optimizers(model, {'name': 'Typo'})
        with self.assertRaisesRegex(ValueError, 'Unknown settings'):
            notebook.build_optimizers(model, {'name': 'Adam', 'momemtum': .5})
        with self.assertRaises(ValueError):
            notebook.build_optimizers(model, {'name': 'SGD', 'nesterov': True, 'momentum': 0.})

    def test_dashboard_controls_build_and_switch_without_mixing_optimizer_presets(self):
        nodes = ast.parse(Path(notebook.__file__).read_text()).body
        create = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == '_'
                      and any(isinstance(child, ast.Assign) and any(isinstance(target, ast.Name)
                      and target.id == 'optimizer_controls' for target in child.targets) for child in node.body))
        render = next(node for node in nodes if isinstance(node, ast.FunctionDef) and
                      [arg.arg for arg in node.args.args] == ['optimizer_choice', 'optimizer_controls'])
        namespace = dict(vars(notebook))
        create.decorator_list = []
        exec(compile(ast.Module(body=[create], type_ignores=[]), notebook.__file__, 'exec'), namespace)
        choice, controls = namespace['_']()
        self.assertEqual(choice.value, notebook.OPTIMIZER_NAME)
        render.decorator_list = []
        exec(compile(ast.Module(body=[render], type_ignores=[]), notebook.__file__, 'exec'), namespace)
        configs = controls.value
        for name in notebook.available_optimizer_names():
            namespace['_'](SimpleNamespace(value=name), controls)
            self.assertEqual(notebook.selected_optimizer(name, configs), notebook.optimizer_config(name))
        configs['SGD']['lr'] = .0123
        selected = notebook.selected_optimizer('SGD', configs)
        self.assertEqual(selected['lr'], .0123)
        self.assertEqual(notebook.selected_optimizer('AdamW', configs)['lr'], notebook.LEARNING_RATE)
        selected['lr'] = .9
        self.assertEqual(configs['SGD']['lr'], .0123)


if __name__ == '__main__':
    unittest.main()
