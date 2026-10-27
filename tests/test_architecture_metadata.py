"""Training metadata must support heads edited through Torch Canvas."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from residual_mlp_baseline import notebook


class ArchitectureMetadataTests(unittest.TestCase):
    def test_single_layer_head_records_actual_block_expansion(self):
        model = notebook.ResidualMLP(**notebook.TASKS["cifar10"], width=8)
        metadata = notebook.model_architecture(model)
        self.assertFalse(hasattr(model, "head_fc1"))
        self.assertEqual(metadata["hidden"], model.blocks[0].value_up.out_features)
        self.assertEqual(metadata["depth"], len(model.blocks))
        self.assertEqual(metadata["width"], 8)

    def test_legacy_head_and_no_blocks_remain_supported(self):
        model = nn.Module()
        model.input_layer = nn.Linear(12, 8)
        model.blocks = nn.ModuleList()
        model.head_fc1 = nn.Linear(8, 16)
        model.head_fc2 = nn.Linear(16, 3)
        self.assertEqual(notebook.model_architecture(model)["hidden"], 16)
        del model.head_fc1
        model.head_fc2 = nn.Linear(8, 3)
        self.assertEqual(notebook.model_architecture(model)["hidden"], 8)

    def test_training_saves_metadata_with_current_single_layer_head(self):
        original = notebook.ResidualMLP
        images = np.zeros((8, 3, 32, 32), dtype=np.uint8)
        labels = np.arange(8) % 2
        augmentation = notebook.augmentation_config("cifar10")
        augmentation["enabled"] = False
        with tempfile.TemporaryDirectory() as directory, torch.random.fork_rng(devices=[]):
            run_dir = Path(directory) / "run"
            with patch.object(notebook, "ResidualMLP", side_effect=lambda **args: original(**args, width=8)), \
                 patch.dict(notebook.EPOCHS, {"cifar10": 1}), patch.object(notebook, "BATCH_SIZE", 8):
                model = notebook.train_with_metrics(
                    "cifar10", images, labels, device="cpu", run_dir=run_dir,
                    augmentation=augmentation, optimizer_settings=notebook.optimizer_config("Adam"))
            config = json.loads((run_dir / "config.json").read_text())
            self.assertEqual(config["architecture"]["hidden"], model.blocks[0].value_up.out_features)
            self.assertEqual(len(model.training_history), 1)
            self.assertTrue(np.isfinite(model.training_history[0]["train_loss"]))


if __name__ == "__main__":
    unittest.main()
