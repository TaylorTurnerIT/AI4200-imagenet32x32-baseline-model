"""Checks for label-specific augmentation and its training integration.

Run with: python -m unittest discover -s tests
"""

import csv
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from residual_mlp_baseline import notebook as notebook


def settings(enabled=True):
    config = notebook.digit_augmentation_config()
    config['enabled'] = enabled
    for digit in config['digits'].values():
        for category in digit.values():
            category['probability'] = 0.0
    return config


class DigitAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.images = torch.linspace(-1, 1, 20 * 28 * 28).reshape(20, 1, 28, 28)
        self.labels = torch.arange(20) % 10

    def test_master_off_and_zero_categories_are_exact_noops(self):
        configs = [settings(), notebook.digit_augmentation_config()]
        configs[1]['enabled'] = False
        zero_amounts = notebook.digit_augmentation_config()
        zero_amounts['enabled'] = True
        for digit in zero_amounts['digits'].values():
            for category in digit.values():
                category['amount'] = 0.0
        configs.append(zero_amounts)
        for config in configs:
            with self.subTest(config=config):
                output = notebook.DigitAugmenter(config, seed=42)(self.images, self.labels)
                self.assertTrue(torch.equal(output, self.images))

    def test_every_category_changes_only_its_selected_digit(self):
        original = self.images.clone()
        for category, spec in notebook.AUGMENTATION_CATEGORIES.items():
            with self.subTest(category=category):
                config = settings()
                config['digits']['3'][category] = {'probability': 1.0, 'amount': spec[2]}
                output = notebook.DigitAugmenter(config, seed=42)(self.images, self.labels)
                self.assertTrue(torch.equal(output[self.labels != 3], self.images[self.labels != 3]))
                self.assertFalse(torch.equal(output[self.labels == 3], self.images[self.labels == 3]))
                self.assertTrue(torch.isfinite(output).all())
                self.assertGreaterEqual(output.min().item(), -1)
                self.assertLessEqual(output.max().item(), 1)
                self.assertTrue(torch.equal(self.images, original))

    def test_seeded_sequence_is_reproducible_and_refreshes_each_call(self):
        config = notebook.digit_augmentation_config()
        config['enabled'] = True
        first = notebook.DigitAugmenter(config, seed=42)
        second = notebook.DigitAugmenter(config, seed=42)
        a = first(self.images, self.labels)
        b = first(self.images, self.labels)
        self.assertTrue(torch.equal(a, second(self.images, self.labels)))
        self.assertTrue(torch.equal(b, second(self.images, self.labels)))
        self.assertFalse(torch.equal(a, b))

    def test_private_generator_preserves_global_rng(self):
        config = notebook.digit_augmentation_config()
        config['enabled'] = True
        before = torch.random.get_rng_state().clone()
        notebook.DigitAugmenter(config, seed=42)(self.images, self.labels)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_translation_fills_edges_with_black(self):
        config = settings()
        for digit in config['digits'].values():
            digit['translation'] = {'probability': 1.0, 'amount': 7.0}
        output = notebook.DigitAugmenter(config, seed=42)(torch.ones_like(self.images), self.labels)
        self.assertEqual(output.min().item(), -1)
        self.assertEqual(output.max().item(), 1)

    def test_erasing_respects_maximum_area(self):
        config = settings()
        for digit in config['digits'].values():
            digit['erasing'] = {'probability': 1.0, 'amount': 0.1}
        output = notebook.DigitAugmenter(config, seed=42)(torch.ones_like(self.images), self.labels)
        erased = (output == -1).sum(dim=(1, 2, 3))
        self.assertTrue((erased <= 28 * 28 * 0.1).all())
        self.assertTrue((erased > 0).any())

    def test_invalid_settings_are_rejected(self):
        for field, value in [('probability', -0.1), ('probability', 1.1),
                             ('amount', -1), ('amount', float('nan')), ('amount', 31)]:
            with self.subTest(field=field, value=value):
                config = settings()
                config['digits']['0']['rotation'][field] = value
                with self.assertRaises(ValueError):
                    notebook.DigitAugmenter(config, seed=42)

    def test_training_uses_only_augmented_training_split_and_saves_settings(self):
        config = notebook.digit_augmentation_config()
        config['enabled'] = True
        images = np.random.default_rng(42).integers(0, 256, (20, 1, 28, 28), dtype=np.uint8)
        labels = np.arange(20) % 10
        original = images.copy()
        calls = []
        seen_by_model = []
        base_augmenter = notebook.DigitAugmenter
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
                patch.object(notebook, 'DigitAugmenter', TrackedAugmenter):
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

    def test_other_tasks_ignore_enabled_digit_augmentation(self):
        images = np.random.default_rng(42).integers(0, 256, (12, 3, 32, 32), dtype=np.uint8)
        labels = np.arange(12) % 10
        enabled = notebook.digit_augmentation_config()
        enabled['enabled'] = True
        disabled = notebook.digit_augmentation_config()
        disabled['enabled'] = False
        with patch.dict(notebook.EPOCHS, cifar10=1), \
                patch.object(notebook, 'ResidualMLP', partial(notebook.ResidualMLP, width=8, hidden=16, depth=1)):
            first = notebook.train_with_metrics('cifar10', images, labels, device='cpu', augmentation=enabled)
            second = notebook.train_with_metrics('cifar10', images, labels, device='cpu', augmentation=disabled)
        self.assertEqual(first.training_history[0]['train_loss'], second.training_history[0]['train_loss'])
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]))


if __name__ == '__main__':
    unittest.main()
