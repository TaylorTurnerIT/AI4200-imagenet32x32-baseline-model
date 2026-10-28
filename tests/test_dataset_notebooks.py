"""Independent dataset defaults and portable histories through real Marimo graphs."""

import ast
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from residual_mlp_baseline import notebook
from residual_mlp_baseline.saved_history import read_history
from residual_mlp_baseline.notebooks import cifar10, imagenet32, mnist


NOTEBOOKS = {"mnist": mnist, "cifar10": cifar10, "imagenet32": imagenet32}


class DatasetNotebookTests(unittest.TestCase):
    def test_defaults_match_best_completed_validation_run_and_exact_seeded_model(self):
        for task, module in NOTEBOOKS.items():
            with self.subTest(task=task), torch.random.fork_rng(devices=[]):
                archive = read_history(task)
                best = notebook.best_saved_run(archive["entries"])
                self.assertEqual(best["id"], archive["default_run_id"])
                self.assertEqual(module.NOTEBOOK_TASK, task)
                self.assertEqual(module.RESTORED_FROM["run_id"], best["id"])
                self.assertEqual(module.DEFAULT_SEED, best["config"]["seed"])
                self.assertEqual(module.optimizer_config(), best["config"]["optimizer_settings"])
                self.assertEqual(module.augmentation_config(task), best["config"]["augmentation"])
                self.assertEqual(module.MAX_TRAIN_SECONDS, best["config"].get("max_seconds"))
                self.assertFalse(module.LEADERBOARD_ENABLED)
                namespace = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional}
                config = best["config"]
                for source in [config["block_source"], *config.get("support_sources", {}).values(), config["model_source"]]:
                    _, source = notebook.class_snapshot(source)
                    exec(compile(source, "<saved-source>", "exec"), namespace)
                name, _ = notebook.class_snapshot(config["model_source"])
                torch.manual_seed(config["seed"])
                original = namespace[name](**module.TASKS[task]).eval()
                torch.manual_seed(config["seed"])
                current = module.ResidualMLP(**module.TASKS[task]).eval()
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(current.state_dict()[key], value, rtol=0, atol=0)
                shape = (1, 28, 28) if task == "mnist" else (3, 32, 32)
                images = torch.randn(2, *shape)
                torch.testing.assert_close(current(images), original(images), rtol=0, atol=0)

    def test_history_survives_without_local_run_files_and_stays_dataset_specific(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task, module in NOTEBOOKS.items():
                with self.subTest(task=task):
                    archive = read_history(task)
                    entries = module.list_saved_runs(root / "results.csv", task, include_archived=True)
                    self.assertEqual({entry["id"] for entry in entries}, {entry["id"] for entry in archive["entries"]})
                    self.assertTrue(all(entry["config"]["task"] == task for entry in entries))
                    self.assertTrue(any(entry["metrics"] for entry in entries))
                    self.assertTrue(archive["legacy_rows"])
                    local = root / "runs" / f"{task}-new"
                    local.mkdir(parents=True)
                    (local / "config.json").write_text(json.dumps(entries[0]["config"]))
                    updated = module.list_saved_runs(root / "results.csv", task, include_archived=True)
                    self.assertEqual(len(updated), len(entries) + 1)
                    self.assertFalse(next(entry for entry in updated if entry["id"] == local.name)["completed"])

    def test_every_live_graph_opens_with_correct_controls_without_training_or_uploads(self):
        for task, module in NOTEBOOKS.items():
            with self.subTest(task=task), \
                    patch("torch.load", side_effect=AssertionError("No weights on startup")), \
                    patch("urllib.request.urlopen", side_effect=AssertionError("No network on startup")):
                _, definitions = module.app.run()
                self.assertEqual(definitions["task"].value, task)
                self.assertFalse(definitions["run_training"].value)
                self.assertEqual(definitions["epoch_count"].value, module.EPOCHS[task])
                self.assertEqual(definitions["time_limit"].value, (module.MAX_TRAIN_SECONDS or 0) / 60)
                self.assertTrue(all(entry["config"]["task"] == task for entry in definitions["saved_run_catalog"]))

    def test_loading_one_dataset_cannot_overwrite_another_or_its_history(self):
        before = {task: Path(module.__file__).read_bytes() for task, module in NOTEBOOKS.items()}
        archive_before = json.dumps(read_history("mnist"), sort_keys=True)
        entry = next(entry for entry in read_history("mnist")["entries"]
                     if entry["id"] == "mnist-20261024-181654-70e55ef9")
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "mnist.py"
            target.write_bytes(before["mnist"])
            with patch("torch.load", side_effect=AssertionError("Restoration must not load weights")):
                mnist.apply_saved_run(target, entry)
            self.assertEqual(notebook.setup_value(target.read_text(), "RESTORED_FROM")["run_id"], entry["id"])
            name = "temporary_dataset_notebook"
            spec = importlib.util.spec_from_file_location(name, target)
            restored = importlib.util.module_from_spec(spec)
            sys.modules[name] = restored
            try:
                spec.loader.exec_module(restored)
                _, definitions = restored.app.run()
                self.assertEqual(definitions["task"].value, "mnist")
                self.assertEqual(definitions["epoch_count"].value, entry["config"]["epochs"])
                self.assertEqual(restored.ResidualMLP(**restored.TASKS["mnist"]).input_layer.out_features, 128)
            finally:
                sys.modules.pop(name, None)
            other = next(entry for entry in read_history("cifar10")["entries"] if entry["completed"])
            saved = target.read_bytes()
            with self.assertRaisesRegex(ValueError, "own dataset"):
                mnist.apply_saved_run(target, other)
            self.assertEqual(target.read_bytes(), saved)
        self.assertEqual(archive_before, json.dumps(read_history("mnist"), sort_keys=True))
        self.assertEqual(before, {task: Path(module.__file__).read_bytes() for task, module in NOTEBOOKS.items()})

    def test_each_notebook_trains_fresh_and_records_its_own_recipe(self):
        for task, module in NOTEBOOKS.items():
            shape = (1, 28, 28) if task == "mnist" else (3, 32, 32)
            images = np.zeros((8, *shape), dtype=np.uint8)
            with self.subTest(task=task), patch("torch.load", side_effect=AssertionError("No checkpoint reads")):
                model = module.train_with_metrics(task, images, np.arange(8) % 2, device="cpu", epochs=1)
                self.assertEqual(model.run_config["task"], task)
                self.assertEqual(model.run_config["restored_from"]["run_id"], module.RESTORED_FROM["run_id"])
                probabilities = module.predict(model, images[:2])
                self.assertEqual(probabilities.shape, (2, module.TASKS[task]["num_classes"]))
            other = "cifar10" if task == "mnist" else "mnist"
            with self.assertRaisesRegex(ValueError, "dedicated notebook"):
                module.train_with_metrics(other, images, np.arange(8), device="cpu", epochs=1)


if __name__ == "__main__":
    unittest.main()
