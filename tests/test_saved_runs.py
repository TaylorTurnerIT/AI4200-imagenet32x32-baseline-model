"""Saved-run ranking and architecture/settings restoration without saved weights."""

import ast
import csv
import inspect
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from residual_mlp_baseline import notebook


class SavedRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = Path(notebook.__file__).read_text()
        model = notebook.ResidualMLP(**notebook.TASKS['cifar10'], width=8, depth=3)
        self.config = {
            'task': 'cifar10', 'seed': 7, 'device': 'cpu', 'epochs': 2, 'batch_size': 4,
            'learning_rate': .002, 'weight_decay': .0002, 'validation_fraction': .2,
            'optimizer': 'AdamW', 'optimizer_settings': {**notebook.optimizer_config('AdamW'),
                                                       'lr': .002, 'weight_decay': .0002},
            'architecture': {'width': 8, 'hidden': 16, 'depth': 3},
            'model': repr(model),
            'model_source': inspect.getsource(notebook.ResidualMLP).replace('F.gelu', 'F.silu').replace('range(depth)', 'range(3)'),
            'block_source': inspect.getsource(type(model.blocks[0])).replace('F.gelu', 'F.silu'),
            'augmentation': notebook.augmentation_config('cifar10'),
        }
        self.config['augmentation']['enabled'] = True

    def save(self, name, validation=None, test=None, completed=True, task='cifar10'):
        run = self.root / 'runs' / name
        run.mkdir(parents=True)
        config = {**self.config, 'task': task}
        (run / 'config.json').write_text(json.dumps(config))
        with (run / 'metrics.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['epoch', 'validation_accuracy'])
            writer.writeheader()
            writer.writerow({'epoch': 1, 'validation_accuracy': validation})
        if completed:
            results = self.root / 'results.csv'
            existed = results.exists()
            with results.open('a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['task', 'run_dir', 'accuracy', 'validation_accuracy'])
                if not existed:
                    writer.writeheader()
                writer.writerow(dict(task=task, run_dir=f'runs/{name}', accuracy=test, validation_accuracy=validation))
        return {'id': name, 'config': config}

    def test_catalog_filters_dataset_and_best_excludes_unfinished_missing_and_nonfinite_scores(self):
        self.save('cifar10-001', .8, .7)
        self.save('cifar10-002', .7, .9)
        self.save('cifar10-003', .8, .6)
        self.save('cifar10-unfinished', 1., 1., completed=False)
        self.save('cifar10-missing', None, None)
        self.save('cifar10-invalid', 'nan', 'inf')
        self.save('mnist-001', .99, .99, task='mnist')
        bad = self.root / 'runs' / 'broken'
        bad.mkdir()
        (bad / 'config.json').write_text('{invalid')
        (self.root / 'runs' / 'cifar10-001' / 'summary.json').write_text('{broken summary')
        entries = notebook.list_saved_runs(self.root / 'results.csv', 'cifar10')
        self.assertEqual(len(entries), 6)
        self.assertEqual(notebook.best_saved_run(entries)['id'], 'cifar10-001')
        self.assertEqual(notebook.best_saved_run(entries, 'test_accuracy')['id'], 'cifar10-002')
        self.assertIsNone(notebook.best_saved_run([], 'test_accuracy'))
        zero = self.save('cifar10-zero', 0., 0.)
        entries = notebook.list_saved_runs(self.root / 'results.csv', 'cifar10')
        self.assertEqual(next(row for row in entries if row['id'] == zero['id'])['test_accuracy'], 0.)

    def test_display_metadata_preserves_executable_model_and_required_signature(self):
        source = notebook.class_snapshot(self.config['model_source'])[1]
        for task, shape, features, classes in (
            ('mnist', 'e, 1, 28, 28', 784, 10),
            ('cifar10', 'e, 3, 32, 32', 3072, 10),
            ('imagenet32', 'e, 3, 32, 32', 3072, 1000),
        ):
            with self.subTest(task=task):
                annotated = notebook.torch_canvas_model_metadata(source, task)
                self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(annotated)))
                self.assertIn(f'# torch-canvas input x: {shape}', annotated)
                self.assertIn(f'# torch-canvas sizes: in_features={features}, num_classes={classes}', annotated)
                self.assertEqual(notebook.torch_canvas_model_metadata(annotated, task), annotated)
        pooled = source.replace('self.input_layer = nn.Linear(in_features, width)',
                                'self.pool = nn.MaxPool2d(4)\n        self.input_layer = nn.Linear(in_features // 16, width)')
        pooled = pooled.replace('x = x.flatten(start_dim=1)', 'x = self.pool(x).flatten(start_dim=1)')
        annotated = notebook.torch_canvas_model_metadata(pooled, 'cifar10')
        self.assertEqual(ast.dump(ast.parse(pooled)), ast.dump(ast.parse(annotated)))
        self.assertIn('self.pool', annotated)

    def test_loading_display_metadata_does_not_mutate_archived_configuration(self):
        entry = self.save('display-only', .8, .7)
        config_path = self.root / 'runs' / entry['id'] / 'config.json'
        before = config_path.read_bytes()
        entry_before = json.dumps(entry, sort_keys=True)
        restored = notebook.restore_run_source(self.source, entry)
        self.assertEqual(config_path.read_bytes(), before)
        self.assertEqual(json.dumps(entry, sort_keys=True), entry_before)
        self.assertIn('# torch-canvas sizes: in_features=3072, num_classes=10', restored)

    def test_loading_restores_model_and_settings_and_fresh_training_records_parent(self):
        entry = self.save('cifar10-parent', .8, .7)
        target = self.root / 'notebook.py'
        target.write_text('# retain unrelated user text\n' + self.source)
        target.chmod(0o640)
        with patch('torch.load', side_effect=AssertionError('Must not load saved weights')):
            notebook.apply_saved_run(target, entry)
        restored = target.read_text()
        modified = target.stat().st_mtime_ns
        notebook.apply_saved_run(target, entry)
        self.assertEqual(target.stat().st_mtime_ns, modified)
        self.assertTrue(restored.startswith('# retain unrelated user text\n'))
        self.assertEqual(target.stat().st_mode & 0o777, 0o640)
        self.assertFalse(list(self.root.glob('.notebook-restore-*')))
        module_name = f'saved_fixture_{target.stem}_{id(self)}'
        spec = importlib.util.spec_from_file_location(module_name, target)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)
        spec.loader.exec_module(module)
        namespace = vars(module)
        self.assertEqual(namespace['DEFAULT_TASK'], 'cifar10')
        self.assertEqual(namespace['DEFAULT_SEED'], 7)
        self.assertEqual(namespace['DEFAULT_DEVICE'], 'cpu')
        self.assertEqual(namespace['BATCH_SIZE'], 4)
        self.assertEqual(namespace['EPOCHS']['cifar10'], 2)
        self.assertEqual(namespace['OPTIMIZER_NAME'], 'AdamW')
        self.assertEqual(namespace['optimizer_config'](), self.config['optimizer_settings'])
        self.assertEqual(namespace['augmentation_config']('cifar10'), self.config['augmentation'])
        model = namespace['ResidualMLP'](**notebook.TASKS['cifar10'])
        self.assertEqual(len(model.blocks), 3)
        self.assertEqual(len(namespace['ResidualMLP'](**notebook.TASKS['cifar10'], depth=2).blocks), 2)
        self.assertEqual(model.input_layer.out_features, 8)
        self.assertEqual(namespace['model_architecture'](model)['hidden'], 16)
        self.assertIn('F.silu', inspect.getsource(type(model.blocks[0]).forward))
        images = np.zeros((12, 3, 32, 32), dtype=np.uint8)
        trained = namespace['train_with_metrics']('cifar10', images, np.arange(12) % 3, device='cpu')
        self.assertEqual(trained.run_config['restored_from']['run_id'], entry['id'])
        self.assertEqual(trained.run_config['optimizer_settings'], self.config['optimizer_settings'])
        self.assertEqual(len(trained.training_history), 2)
        self.assertEqual(trained.run_config['architecture']['functional_operations'], ['silu'])

    def test_loaded_augmentation_matches_visible_switch_and_training_for_on_and_off(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                config = {**self.config, 'augmentation': {**self.config['augmentation'], 'enabled': enabled}}
                entry = {'id': f'augmentation-{enabled}', 'config': config}
                target = self.root / f'notebook_{enabled}.py'
                target.write_text(self.source)
                notebook.apply_saved_run(target, entry)
                module_name = f'loaded_augmentation_{enabled}_{id(self)}'
                spec = importlib.util.spec_from_file_location(module_name, target)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                self.addCleanup(sys.modules.pop, module_name, None)
                spec.loader.exec_module(module)
                cells = [node for node in ast.parse(target.read_text()).body
                         if isinstance(node, ast.FunctionDef) and node.name == '_']
                controls_cell = next(node for node in cells if any(
                    isinstance(child, ast.Assign) and any(
                        isinstance(output, ast.Name) and output.id == 'augmentation_controls'
                        for output in child.targets) for child in node.body))
                controls_cell.decorator_list = []
                (widgets,) = create_function(controls_cell, module)()
                visible = widgets['cifar10']['enabled'].value
                self.assertEqual(visible, enabled)
                effective = module.effective_augmentation('cifar10', widgets)
                self.assertEqual(effective, config['augmentation'])
                images = np.zeros((12, 3, 32, 32), dtype=np.uint8)
                for switch_value in (enabled, not enabled):
                    update = json.loads(json.dumps(widgets._value_frontend))
                    update['cifar10']['enabled'] = switch_value
                    widgets._update(update)
                    self.assertEqual(widgets['cifar10']['enabled'].value, switch_value)
                    effective = module.effective_augmentation('cifar10', widgets)
                    run_dir = self.root / f'run-{enabled}-{switch_value}'
                    model = module.train_with_metrics('cifar10', images, np.arange(12) % 3,
                                                      device='cpu', run_dir=run_dir, augmentation=effective)
                    saved = json.loads((run_dir / 'config.json').read_text())
                    self.assertEqual(saved['augmentation']['enabled'], switch_value)
                    self.assertEqual(model.run_config['augmentation'], effective)
                widgets.value['cifar10']['enabled'] = not widgets['cifar10']['enabled'].value
                with self.assertRaisesRegex(ValueError, 'controls disagree'):
                    module.effective_augmentation('cifar10', widgets)

    def test_legacy_dimensions_and_unrecorded_fields_require_explicit_choices(self):
        config = dict(self.config)
        config['model'] = """ResidualMLP(
          (input_layer): Linear(in_features=3072, out_features=8, bias=True)
          (blocks): ModuleList(3 x Block())
          (head_fc1): Linear(in_features=8, out_features=16, bias=True)
        )"""
        for key in ('architecture', 'model_source', 'block_source', 'optimizer', 'optimizer_settings', 'augmentation'):
            config.pop(key)
        entry = {'id': 'legacy', 'config': config}
        with self.assertRaisesRegex(ValueError, 'activation'):
            notebook.restore_run_source(self.source, entry)
        with self.assertRaisesRegex(ValueError, 'optimizer'):
            notebook.restore_run_source(self.source, entry, legacy_activation='gelu')
        with self.assertRaisesRegex(ValueError, 'augmentation'):
            notebook.restore_run_source(self.source, entry, legacy_activation='gelu', legacy_optimizer='SGD')
        restored = notebook.restore_run_source(self.source, entry, legacy_activation='gelu',
                                               legacy_optimizer='SGD', legacy_augmentation='off')
        target = self.root / 'legacy.py'
        target.write_text(restored)
        module_name = f'saved_fixture_{target.stem}_{id(self)}'
        spec = importlib.util.spec_from_file_location(module_name, target)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)
        spec.loader.exec_module(module)
        namespace = vars(module)
        model = namespace['ResidualMLP'](**notebook.TASKS['cifar10'])
        self.assertEqual((model.input_layer.out_features, namespace['model_architecture'](model)['hidden'],
                          len(model.blocks)), (8, 16, 3))
        self.assertEqual(namespace['optimizer_config']()['lr'], .002)
        self.assertFalse(namespace['augmentation_config']('cifar10')['enabled'])
        self.assertIn('F.gelu', inspect.getsource(type(model.blocks[0]).forward))

    def test_legacy_unrolled_residual_blocks_are_counted(self):
        config = {'model': """ResidualMLP(
          (input_layer): Linear(in_features=3072, out_features=8, bias=True)
          (fc1): Linear(in_features=8, out_features=16, bias=True)
          (fc2): Linear(in_features=16, out_features=8, bias=True)
          (fc1_2): Linear(in_features=8, out_features=16, bias=True)
          (fc2_2): Linear(in_features=16, out_features=8, bias=True)
          (fc1_3): Linear(in_features=8, out_features=16, bias=True)
          (fc2_3): Linear(in_features=16, out_features=8, bias=True)
          (head_fc1): Linear(in_features=8, out_features=16, bias=True)
        )"""}
        self.assertEqual(notebook.restored_architecture(config), {'width': 8, 'hidden': 16, 'depth': 3})

    def test_failed_or_concurrent_load_preserves_existing_source_and_cleans_only_its_temp(self):
        target = self.root / 'notebook.py'
        target.write_text(self.source)
        config = {**self.config, 'architecture': {'width': 0, 'hidden': 16, 'depth': 3}}
        with self.assertRaises(ValueError):
            notebook.apply_saved_run(target, {'id': 'invalid', 'config': config})
        self.assertEqual(target.read_text(), self.source)
        invalid = {**self.config, 'optimizer_settings': {**self.config['optimizer_settings'], 'lr': float('nan')}}
        with self.assertRaisesRegex(ValueError, 'finite'):
            notebook.apply_saved_run(target, {'id': 'invalid', 'config': invalid})
        self.assertEqual(target.read_text(), self.source)

        def concurrent_edit(*args, **kwargs):
            target.write_text('# concurrent user edit\n' + self.source)
            return self.source

        with patch.object(notebook, 'restore_run_source', side_effect=concurrent_edit):
            with self.assertRaisesRegex(ValueError, 'changed during loading'):
                notebook.apply_saved_run(target, {'id': 'parent', 'config': self.config})
        self.assertTrue(target.read_text().startswith('# concurrent user edit\n'))
        self.assertFalse(list(self.root.glob('.notebook-restore-*')))

    def test_dashboard_defines_legacy_augmentation_control(self):
        cells = [node for node in ast.parse(self.source).body
                 if isinstance(node, ast.FunctionDef) and node.name == '_']
        controls = next(node for node in cells if any(
            isinstance(child, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == 'legacy_activation'
                for target in child.targets)
            for child in node.body))
        controls.decorator_list = []
        values = create_function(controls, notebook)()
        outputs = dict(zip((item.id for item in controls.body[-1].value.elts), values))
        augmentation = outputs['legacy_augmentation']
        self.assertIsNone(augmentation.value)
        self.assertEqual(set(augmentation.options.values()), {'off', 'current'})

    def test_dashboard_selected_and_best_buttons_route_to_the_correct_run(self):
        first = self.save('cifar10-001', .8, .7)
        second = self.save('cifar10-002', .7, .9)
        cells = [node for node in ast.parse(self.source).body if isinstance(node, ast.FunctionDef) and node.name == '_']
        create = next(node for node in cells if any(isinstance(child, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'saved_run_choice' for target in child.targets)
            for child in node.body))
        action = next(node for node in cells if any(isinstance(child, ast.Expr) and isinstance(child.value, ast.Call)
            and isinstance(child.value.func, ast.Attribute) and child.value.func.attr == 'stop'
            for child in node.body) and 'saved_run_choice' in [arg.arg for arg in node.args.args])
        namespace = dict(vars(notebook))
        create.decorator_list = []
        exec(compile(ast.Module(body=[create], type_ignores=[]), notebook.__file__, 'exec'), namespace)
        catalog, choice = namespace['_'](self.root / 'results.csv', SimpleNamespace(value=0), SimpleNamespace(value='cifar10'))
        self.assertEqual(choice.value, second['id'])
        calls = []
        namespace['apply_saved_run'] = lambda path, entry, *args: calls.append(entry['id']) or entry['id']
        action.decorator_list = []
        exec(compile(ast.Module(body=[action], type_ignores=[]), notebook.__file__, 'exec'), namespace)
        values = dict(legacy_activation=SimpleNamespace(value=None), legacy_optimizer=SimpleNamespace(value=None),
                      legacy_augmentation=SimpleNamespace(value=None), saved_run_catalog=catalog, saved_run_choice=choice,
                      set_loaded_epochs=lambda value: None)
        for metric, expected in [('validation_accuracy', first['id']), ('test_accuracy', second['id'])]:
            namespace['_'](best_metric=SimpleNamespace(value=metric), load_best=SimpleNamespace(value=True),
                           load_saved=SimpleNamespace(value=False), **values)
            self.assertEqual(calls[-1], expected)
        namespace['_'](best_metric=SimpleNamespace(value='validation_accuracy'), load_best=SimpleNamespace(value=False),
                       load_saved=SimpleNamespace(value=True), **values)
        self.assertEqual(calls[-1], second['id'])
        empty, empty_choice = create_function(create, notebook)(self.root / 'missing.csv', SimpleNamespace(value=0),
                                                                SimpleNamespace(value='imagenet32'))
        self.assertEqual(empty, [])
        self.assertIsNone(empty_choice.value)


def create_function(cell, module):
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[cell], type_ignores=[]), module.__file__, 'exec'), namespace)
    return namespace['_']


if __name__ == '__main__':
    unittest.main()
