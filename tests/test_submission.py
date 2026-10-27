"""Check frozen grading recipes independently of the editable dashboard."""

import importlib.abc
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import residual_mlp_baseline as student
from residual_mlp_baseline import submission, training_core as core


class SubmissionTests(unittest.TestCase):
    def test_public_interface_and_time_allocations(self):
        self.assertEqual(list(inspect.signature(student.train).parameters),
                         ["task", "X_train", "y_train", "seed", "device"])
        self.assertEqual(list(inspect.signature(student.predict).parameters), ["model", "X_test"])
        self.assertEqual(sum(r["max_seconds"] for r in submission.RECIPES.values()), 5400)
        self.assertIs(student.train, submission.train)

    def test_mnist_reconstruction_preserves_historical_seeded_weights_and_logits(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/submission-recipes.json").read_text())
        namespace = {"nn": nn, "torch": torch, "F": F}
        exec(fixture["mnist"]["historical_model_source"], namespace)
        for seed in [0, 42]:
            torch.manual_seed(seed)
            old = namespace["ResidualMLP"](784, 10)
            torch.manual_seed(seed)
            new = submission.MnistResidualMLP()
            for original, restored in zip(old.state_dict().values(), new.state_dict().values()):
                self.assertTrue(torch.equal(original, restored))
            images = torch.randn(3, 1, 28, 28)
            torch.testing.assert_close(old(images), new(images), rtol=0, atol=0)

    def test_all_tasks_train_fresh_without_reading_checkpoint_files(self):
        generator = np.random.default_rng(8)
        for task in core.TASKS:
            shape = (1, 28, 28) if task == "mnist" else (3, 32, 32)
            images = generator.integers(0, 256, size=(8, *shape), dtype=np.uint8)
            labels = np.arange(8) % 2
            recipe = {**submission.RECIPES[task], "epochs": 1, "max_seconds": 120}
            with patch.dict(submission.RECIPES, {task: recipe}), \
                    patch.object(torch, "load", side_effect=AssertionError("checkpoint read")):
                model = student.train(task, images, labels, seed=0, device="cpu")
            probabilities = student.predict(model, images[:3])
            self.assertEqual(probabilities.shape, (3, core.TASKS[task]["num_classes"]))
            self.assertTrue(torch.isfinite(probabilities).all())
            self.assertTrue((probabilities >= 0).all())
            torch.testing.assert_close(probabilities.sum(1), torch.ones(3), atol=1e-5, rtol=0)
            self.assertEqual(model.run_config["seed"], 0)
            self.assertEqual(len(model.training_history), 1)
            self.assertEqual(model.stop_reason, "epochs")
            self.assertEqual(model.run_config["augmentation"], submission.augmentation_for(task))

    def test_budget_discards_partial_epoch_and_restores_complete_checkpoint(self):
        clock = [0.0]

        class TimedModel(core.ResidualMLP):
            def __init__(self, **kwargs):
                super().__init__(**kwargs, width=8, hidden=16, depth=1,
                                 window_hidden=8, window_features=4)
                self.train_calls = 0

            def forward(self, images):
                if self.training:
                    self.train_calls += 1
                    if self.train_calls == 2:
                        clock[0] = 100.0
                elif self.train_calls == 1:
                    self.first_validation_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                return super().forward(images)

        images = np.zeros((8, 3, 32, 32), dtype=np.uint8)
        labels = np.arange(8) % 2
        with patch.object(core.time, "perf_counter", side_effect=lambda: clock[0]):
            model = core.train_with_metrics(
                "cifar10", images, labels, device="cpu", epochs=3,
                model_factory=TimedModel, augmentation=submission.augmentation_for("imagenet32") | {"task": "cifar10"},
                max_seconds=10,
            )
        self.assertEqual(model.stop_reason, "time_budget")
        self.assertEqual(len(model.training_history), 1)
        self.assertEqual(model.best_epoch, 1)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, model.first_validation_state[name]))

    def test_import_does_not_require_marimo(self):
        code = """
import importlib.abc
import sys
class NoDashboard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'marimo' or fullname.startswith('marimo.'):
            raise ModuleNotFoundError('dashboard deliberately unavailable')
sys.meta_path.insert(0, NoDashboard())
import residual_mlp_baseline
assert 'marimo' not in sys.modules
assert callable(residual_mlp_baseline.train)
"""
        subprocess.run([sys.executable, "-c", code], check=True, timeout=60)

    def test_imagenet_normalization_is_bounded_by_batch_size(self):
        images = np.random.default_rng(4).integers(0, 256, (10, 3, 32, 32), dtype=np.uint8)
        labels = np.arange(10) % 2
        original = images.copy()
        sizes = []
        prepare = core.prepare

        def record(batch):
            sizes.append(len(batch))
            return prepare(batch)

        def factory(**kwargs):
            return core.ResidualMLP(**kwargs, width=8, hidden=16, depth=1,
                                    window_hidden=8, window_features=4)

        with patch.object(core, "prepare", side_effect=record):
            model = core.train_with_metrics(
                "imagenet32", images, labels, device="cpu", epochs=1,
                batch_size=3, model_factory=factory,
                augmentation=submission.augmentation_for("imagenet32"),
            )
        self.assertTrue(sizes)
        self.assertLessEqual(max(sizes), 3)
        self.assertTrue(model.run_config["streaming_normalization"])
        self.assertTrue(np.array_equal(images, original))

    def test_optional_diagnostics_do_not_change_training_weights(self):
        from residual_mlp_baseline.analysis import FixedBatchProbe
        images = np.random.default_rng(3).integers(0, 256, (8, 3, 32, 32), dtype=np.uint8)
        labels = np.arange(8) % 2

        def factory(**kwargs):
            return core.ResidualMLP(**kwargs, width=8, hidden=16, depth=1,
                                    window_hidden=8, window_features=4)

        arguments = dict(device="cpu", epochs=2, model_factory=factory,
                         augmentation=submission.augmentation_for("imagenet32"))
        plain = core.train_with_metrics("imagenet32", images, labels, **arguments)
        with tempfile.TemporaryDirectory() as directory:
            probe = FixedBatchProbe(images, labels, 0, Path(directory) / "probe")
            observed = core.train_with_metrics("imagenet32", images, labels, observer=probe, **arguments)
            self.assertEqual([r["stage"] for r in probe.rows], ["initial", "epoch", "selected_checkpoint"])
            self.assertTrue(probe.rows[-1]["branch_to_stream_ratio"])
            self.assertTrue(probe.rows[-1]["rms_gradient"])
            self.assertTrue(probe.rows[-1]["update_to_weight_ratio"])
        for name, tensor in plain.state_dict().items():
            self.assertTrue(torch.equal(tensor, observed.state_dict()[name]))


if __name__ == "__main__":
    unittest.main()
