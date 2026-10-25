"""Checks for label-specific augmentation and its training integration.

Run with: python -m unittest discover -s tests
"""

import ast
import csv
import json
import tempfile
import unittest
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from residual_mlp_baseline import notebook as notebook


def settings(enabled=True):
    config = notebook.augmentation_config('mnist')
    config['enabled'] = enabled
    for digit in config['classes'].values():
        for category in digit.values():
            category['probability'] = 0.0
    return config


class DigitAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.images = torch.linspace(-1, 1, 20 * 28 * 28).reshape(20, 1, 28, 28)
        self.labels = torch.arange(20) % 10

    def test_master_off_and_zero_categories_are_exact_noops(self):
        configs = [settings(), notebook.augmentation_config('mnist')]
        configs[1]['enabled'] = False
        zero_amounts = notebook.augmentation_config('mnist')
        zero_amounts['enabled'] = True
        for digit in zero_amounts['classes'].values():
            for category in digit.values():
                category['amount'] = 0.0
        configs.append(zero_amounts)
        for config in configs:
            with self.subTest(config=config):
                output = notebook.ImageAugmenter('mnist', config, seed=42)(self.images, self.labels)
                self.assertTrue(torch.equal(output, self.images))

    def test_every_category_changes_only_its_selected_digit(self):
        original = self.images.clone()
        for category, spec in notebook.AUGMENTATION_CATEGORIES.items():
            with self.subTest(category=category):
                config = settings()
                config['classes']['3'][category] = {'probability': 1.0, 'amount': spec[2]}
                output = notebook.ImageAugmenter('mnist', config, seed=42)(self.images, self.labels)
                self.assertTrue(torch.equal(output[self.labels != 3], self.images[self.labels != 3]))
                self.assertFalse(torch.equal(output[self.labels == 3], self.images[self.labels == 3]))
                self.assertTrue(torch.isfinite(output).all())
                self.assertGreaterEqual(output.min().item(), -1)
                self.assertLessEqual(output.max().item(), 1)
                self.assertTrue(torch.equal(self.images, original))

    def test_seeded_sequence_is_reproducible_and_refreshes_each_call(self):
        config = notebook.augmentation_config('mnist')
        config['enabled'] = True
        first = notebook.ImageAugmenter('mnist', config, seed=42)
        second = notebook.ImageAugmenter('mnist', config, seed=42)
        a = first(self.images, self.labels)
        b = first(self.images, self.labels)
        self.assertTrue(torch.equal(a, second(self.images, self.labels)))
        self.assertTrue(torch.equal(b, second(self.images, self.labels)))
        self.assertFalse(torch.equal(a, b))

    def test_private_generator_preserves_global_rng(self):
        config = notebook.augmentation_config('mnist')
        config['enabled'] = True
        before = torch.random.get_rng_state().clone()
        notebook.ImageAugmenter('mnist', config, seed=42)(self.images, self.labels)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_translation_fills_edges_with_black(self):
        config = settings()
        for digit in config['classes'].values():
            digit['translation'] = {'probability': 1.0, 'amount': 7.0}
        output = notebook.ImageAugmenter('mnist', config, seed=42)(torch.ones_like(self.images), self.labels)
        self.assertEqual(output.min().item(), -1)
        self.assertEqual(output.max().item(), 1)

    def test_erasing_respects_maximum_area(self):
        config = settings()
        for digit in config['classes'].values():
            digit['erasing'] = {'probability': 1.0, 'amount': 0.1}
        output = notebook.ImageAugmenter('mnist', config, seed=42)(torch.ones_like(self.images), self.labels)
        erased = (output == -1).sum(dim=(1, 2, 3))
        self.assertTrue((erased <= 28 * 28 * 0.1).all())
        self.assertTrue((erased > 0).any())

    def test_invalid_settings_are_rejected(self):
        for field, value in [('probability', -0.1), ('probability', 1.1),
                             ('amount', -1), ('amount', float('nan')), ('amount', 31)]:
            with self.subTest(field=field, value=value):
                config = settings()
                config['classes']['0']['rotation'][field] = value
                with self.assertRaises(ValueError):
                    notebook.ImageAugmenter('mnist', config, seed=42)

    def test_training_uses_only_augmented_training_split_and_saves_settings(self):
        config = notebook.augmentation_config('mnist')
        config['enabled'] = True
        images = np.random.default_rng(42).integers(0, 256, (20, 1, 28, 28), dtype=np.uint8)
        labels = np.arange(20) % 10
        original = images.copy()
        calls = []
        seen_by_model = []
        base_augmenter = notebook.ImageAugmenter
        base_model = notebook.ResidualMLP

        class TrackedAugmenter(base_augmenter):
            def __call__(self, batch, targets):
                result = super().__call__(batch, targets)
                calls.append((batch.clone(), result.clone()))
                return result

        class TrackedModel(base_model):
            def forward(self, batch):
                seen_by_model.append((self.training, batch.detach().clone()))
                return super().forward(batch)

        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(notebook.EPOCHS, mnist=2), \
                patch.object(notebook, 'BATCH_SIZE', 7), \
                patch.object(notebook, 'ResidualMLP', partial(TrackedModel, width=8, hidden=16, depth=1)), \
                patch.object(notebook, 'ImageAugmenter', TrackedAugmenter):
            run = Path(folder) / 'run'
            model = notebook.train_with_metrics('mnist', images, labels, seed=42,
                                                 device='cpu', run_dir=run, augmentation=config)
            training_inputs = torch.cat([batch for training, batch in seen_by_model if training])
            validation_inputs = torch.cat([batch for training, batch in seen_by_model if not training])
            augmented_inputs = torch.cat([after for before, after in calls])
            self.assertTrue(torch.equal(training_inputs, augmented_inputs))
            expected_validation = notebook.prepare(images[model.validation_indices.numpy()]).repeat(2, 1, 1, 1)
            self.assertTrue(torch.equal(validation_inputs, expected_validation))
            self.assertEqual(sum(len(before) for before, after in calls), 36)
            self.assertTrue(np.array_equal(images, original))
            self.assertEqual(json.loads((run / 'config.json').read_text())['augmentation'], config)
            checkpoint = torch.load(run / 'best.pt', weights_only=True)
            self.assertEqual(checkpoint['config']['augmentation'], config)
            with (run / 'metrics.csv').open() as file:
                self.assertEqual(len(list(csv.DictReader(file))), 2)

    def test_wrong_dataset_configuration_is_rejected(self):
        config = notebook.augmentation_config('mnist')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            notebook.ImageAugmenter('cifar10', config, seed=42)

    def test_rgb_tasks_apply_flips_to_every_channel_and_accept_last_class(self):
        for task in ('cifar10', 'imagenet32'):
            with self.subTest(task=task):
                config = notebook.augmentation_config(task)
                config['enabled'] = True
                for category in config['default'].values():
                    category['probability'] = 0.
                config['default']['horizontal_flip'] = {'probability': 1., 'amount': 1.}
                images = torch.linspace(-1, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
                original = images.clone()
                labels = torch.tensor([0, notebook.TASKS[task]['num_classes'] - 1])
                augmenter = notebook.ImageAugmenter(task, config, seed=42)
                output = augmenter(images, labels)
                self.assertTrue(torch.allclose(output, images.flip(-1), atol=1e-6))
                self.assertTrue(torch.equal(images, original))
                self.assertEqual(output.shape, images.shape)
                config['enabled'] = False
                self.assertTrue(torch.equal(notebook.ImageAugmenter(task, config, seed=42)(images, labels), images))

    def test_rgb_presets_augment_training_only_and_save_the_selected_config(self):
        for task in ('cifar10', 'imagenet32'):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as folder:
                images = np.random.default_rng(42).integers(0, 256, (12, 3, 32, 32), dtype=np.uint8)
                labels = np.arange(12) % 3
                labels[-1] = notebook.TASKS[task]['num_classes'] - 1
                original = images.copy()
                config = notebook.augmentation_config(task)
                config['enabled'] = True
                for category in config['default'].values():
                    category['probability'] = 0.
                config['default']['horizontal_flip'] = {'probability': 1., 'amount': 1.}
                seen = []
                base_model = notebook.ResidualMLP

                class TrackedModel(base_model):
                    def forward(self, batch):
                        seen.append((self.training, batch.detach().clone()))
                        return super().forward(batch)

                with patch.dict(notebook.EPOCHS, {task: 1}), \
                        patch.object(notebook, 'ResidualMLP', partial(TrackedModel, width=8, hidden=16)):
                    run = Path(folder) / 'run'
                    model = notebook.train_with_metrics(task, images, labels, device='cpu', run_dir=run,
                                                        augmentation=config)
                actual = torch.cat([batch for training, batch in seen if training])
                expected = notebook.prepare(images[model.training_indices.numpy()]).flip(-1)
                # The shuffled training loader may change example order.
                self.assertTrue(torch.allclose(actual.flatten(1).sort(dim=0).values,
                                               expected.flatten(1).sort(dim=0).values, atol=1e-6))
                validation = torch.cat([batch for training, batch in seen if not training])
                self.assertTrue(torch.equal(validation, notebook.prepare(images[model.validation_indices.numpy()])))
                self.assertTrue(np.array_equal(images, original))
                self.assertEqual(json.loads((run / 'config.json').read_text())['augmentation'], config)
                self.assertEqual(torch.load(run / 'best.pt', weights_only=True)['config']['augmentation'], config)

    def test_every_rgb_category_preserves_shape_range_and_source(self):
        for task in ('cifar10', 'imagenet32'):
            for name, spec in notebook.augmentation_categories(task).items():
                with self.subTest(task=task, category=name):
                    config = notebook.augmentation_config(task)
                    config['enabled'] = True
                    for category in config['default'].values():
                        category['probability'] = 0.
                    config['default'][name] = {'probability': 1., 'amount': spec[2]}
                    images = torch.linspace(-1, 1, 4 * 3 * 32 * 32).reshape(4, 3, 32, 32)
                    original = images.clone()
                    output = notebook.ImageAugmenter(task, config, seed=42)(images, torch.arange(4))
                    self.assertEqual(output.shape, images.shape)
                    self.assertFalse(torch.equal(output, images))
                    self.assertTrue(torch.equal(images, original))
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertGreaterEqual(output.min().item(), -1.)
                    self.assertLessEqual(output.max().item(), 1.)

    def test_dashboard_routes_real_controls_to_each_datasets_training_and_saved_config(self):
        source = ast.parse(Path(notebook.__file__).read_text())
        cells = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == '_']
        create = next(node for node in cells if any(
            isinstance(child, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == 'augmentation_controls' for target in child.targets) for child in node.body))
        run = next(node for node in cells if node.args.args and node.args.args[0].arg == 'NUM_CLASSES')
        namespace = dict(vars(notebook))
        create.decorator_list = []
        exec(compile(ast.Module(body=[create], type_ignores=[]), notebook.__file__, 'exec'), namespace)
        controls, = namespace['_']()
        configurations = controls.value
        for task in notebook.TASKS:
            self.assertEqual(notebook.selected_augmentation(task, configurations), notebook.augmentation_config(task))
            configurations[task]['enabled'] = True
        controls._update(configurations)
        namespace['mo'] = SimpleNamespace(stop=lambda *args: None, md=lambda value: value,
                                         vstack=lambda value: None,
                                         status=SimpleNamespace(spinner=lambda **kwargs: nullcontext()))
        namespace['submit_run_to_leaderboard'] = lambda *args: 'Disabled in test.'
        run.decorator_list = []
        exec(compile(ast.Module(body=[run], type_ignores=[]), notebook.__file__, 'exec'), namespace)
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(notebook.EPOCHS, {task: 1 for task in notebook.TASKS}), \
                patch.object(notebook, 'ResidualMLP', partial(notebook.ResidualMLP, width=8, hidden=16)):
            root = Path(folder)
            for task in notebook.TASKS:
                with self.subTest(task=task):
                    shape = (1, 28, 28) if task == 'mnist' else (3, 32, 32)
                    images = np.random.default_rng(42).integers(0, 256, (12, *shape), dtype=np.uint8)
                    labels = np.arange(12) % 3
                    optimizer_name = {'mnist': 'SGD', 'cifar10': 'Adam',
                                      'imagenet32': 'Muon' if 'Muon' in notebook.available_optimizer_names() else 'AdamW'}[task]
                    namespace['_'](
                        NUM_CLASSES={name: spec['num_classes'] for name, spec in notebook.TASKS.items()},
                        RESULTS_FILE=root / 'results.csv',
                        augmentation_controls=controls,
                        optimizer_choice=SimpleNamespace(value=optimizer_name),
                        optimizer_controls=SimpleNamespace(value=notebook.OPTIMIZER_DEFAULTS),
                        device=SimpleNamespace(value='cpu'), load_task=lambda name: (images, labels, images[:4], labels[:4]),
                        run_training=SimpleNamespace(value=True), seed=SimpleNamespace(value=42),
                        task=SimpleNamespace(value=task),
                    )
                    saved = json.loads(next(root.glob(f'runs/{task}-*/config.json')).read_text())
                    self.assertEqual(saved['augmentation'], notebook.selected_augmentation(task, configurations))
                    self.assertEqual(saved['augmentation']['task'], task)
                    self.assertTrue(saved['augmentation']['enabled'])
                    self.assertEqual(saved['optimizer'], optimizer_name)
                    self.assertEqual(saved['optimizer_settings'], notebook.optimizer_config(optimizer_name))

    def test_selected_configuration_is_isolated_and_dataset_specific(self):
        configs = {task: notebook.augmentation_config(task) for task in notebook.TASKS}
        configs['cifar10']['enabled'] = True
        configs['cifar10']['default']['translation']['amount'] = 3.
        selected = notebook.selected_augmentation('cifar10', configs)
        self.assertEqual(selected['task'], 'cifar10')
        self.assertEqual(selected['default']['translation']['amount'], 3.)
        self.assertFalse(notebook.selected_augmentation('mnist', configs)['enabled'])
        self.assertFalse(notebook.selected_augmentation('imagenet32', configs)['enabled'])
        selected['default']['translation']['amount'] = 1.
        self.assertEqual(configs['cifar10']['default']['translation']['amount'], 3.)



if __name__ == '__main__':
    unittest.main()
