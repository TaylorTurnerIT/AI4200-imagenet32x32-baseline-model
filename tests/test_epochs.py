"""Epoch controls must agree with training and saved-run metadata."""

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

from residual_mlp_baseline import notebook


class EpochTests(unittest.TestCase):
    def test_control_uses_dataset_and_restored_defaults(self):
        source = Path(notebook.__file__).read_text()
        tree = ast.parse(source)
        cell = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and any(isinstance(child, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == "epoch_count"
                                    for target in child.targets) for child in node.body))
        self.assertEqual([arg.arg for arg in cell.args.args], ["get_loaded_epochs", "task"])
        cell.name = "make_epoch_control"
        cell.decorator_list = []
        namespace = dict(vars(notebook))
        exec(compile(ast.Module(body=[cell], type_ignores=[]), "<epoch-control>", "exec"), namespace)
        for task in notebook.TASKS:
            control, = namespace["make_epoch_control"](lambda: None, SimpleNamespace(value=task))
            self.assertEqual(control.value, notebook.EPOCHS[task])
        restored = notebook.patch_setup_value(source, "EPOCHS", {**notebook.EPOCHS, "cifar10": 7})
        namespace["EPOCHS"] = notebook.setup_value(restored, "EPOCHS")
        control, = namespace["make_epoch_control"](lambda: None, SimpleNamespace(value="cifar10"))
        self.assertEqual(control.value, 7)

        # Successful loads reset the UI even if the same preset needs no disk write.
        load_cell = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                                 and child.func.id == "apply_saved_run" for child in ast.walk(node)))
        load_cell.name = "load_settings"
        load_cell.decorator_list = []
        exec(compile(ast.Module(body=[load_cell], type_ignores=[]), "<load-settings>", "exec"), namespace)
        get_loaded, set_loaded = notebook.mo.state(None)
        entry = {"id": "same-preset", "config": {"task": "cifar10", "epochs": 3}}
        arguments = {arg.arg: SimpleNamespace(value=None) for arg in load_cell.args.args}
        arguments.update(load_saved=SimpleNamespace(value=True), load_best=SimpleNamespace(value=False),
                         saved_run_catalog=[entry], saved_run_choice=SimpleNamespace(value="same-preset"),
                         set_loaded_epochs=set_loaded)
        with patch.dict(namespace, apply_saved_run=lambda *args: "same-preset"):
            for _ in range(2):
                namespace["load_settings"](**arguments)
                control, = namespace["make_epoch_control"](get_loaded, SimpleNamespace(value="cifar10"))
                self.assertEqual(control.value, 3)

    def test_override_controls_training_history_and_saved_config(self):
        images = np.zeros((8, 3, 32, 32), dtype=np.uint8)
        labels = np.arange(8) % 2
        augmentation = notebook.augmentation_config("cifar10")
        augmentation["enabled"] = False
        model_factory = partial(notebook.ResidualMLP, width=8, depth=1)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(notebook, "ResidualMLP", model_factory), \
                patch.dict(notebook.EPOCHS, {"cifar10": 1}):
            run_dir = Path(directory) / "run"
            model = notebook.train_with_metrics(
                "cifar10", images, labels, device="cpu", run_dir=run_dir, epochs=2,
                augmentation=augmentation, optimizer_settings=notebook.optimizer_config("Adam"))
            self.assertEqual(len(model.training_history), 2)
            self.assertEqual(model.run_config["epochs"], 2)
            self.assertEqual(json.loads((run_dir / "config.json").read_text())["epochs"], 2)
            with (run_dir / "metrics.csv").open() as file:
                self.assertEqual(len(list(csv.DictReader(file))), 2)
            self.assertEqual(notebook.EPOCHS["cifar10"], 1)
            default = notebook.train("cifar10", images, labels, device="cpu")
            self.assertEqual(len(default.training_history), 1)
            self.assertEqual(default.run_config["epochs"], 1)

    def test_invalid_epoch_counts_are_rejected(self):
        for epochs in (0, -1, 1.5, True):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                notebook.train_with_metrics("cifar10", np.zeros((2, 3, 32, 32)), [0, 1],
                                            device="cpu", epochs=epochs)


if __name__ == "__main__":
    unittest.main()
