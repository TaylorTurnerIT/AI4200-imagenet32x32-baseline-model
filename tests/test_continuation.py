"""Validate overfitting detection and resumable analysis checkpoints."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from residual_mlp_baseline import notebook
from scripts import continue_experiment as continuation


class ContinuationTests(unittest.TestCase):
    def history(self, validation):
        return [{"epoch": index + 1, "train_loss": 10 - index * .1,
                 "validation_loss": value} for index, value in enumerate(validation)]

    def test_flat_validation_is_not_divergence(self):
        status = continuation.divergence_status(self.history([1.] * 30), 10)
        self.assertFalse(status["confirmed"])
        self.assertEqual(status["candidate_streak"], 0)

    def test_restart_spike_that_recovers_does_not_confirm_divergence(self):
        status = continuation.divergence_status(
            self.history([1.] * 10 + [1.5, 1.4, 1.3, 1.2, 1.1, 1., .95, .9, .85]), 10)
        self.assertFalse(status["confirmed"])

    def test_sustained_overfitting_confirms_after_three_windows(self):
        rows = self.history([1.] * 10 + [1., 1., 1., 1.1, 1.2, 1.3, 1.4, 1.5])
        self.assertFalse(continuation.divergence_status(rows[:-1], 10)["confirmed"])
        status = continuation.divergence_status(rows, 10)
        self.assertTrue(status["confirmed"])
        self.assertEqual(status["confirmation_epoch"], 18)
        self.assertEqual(status["first_candidate_epoch"], 16)
        self.assertEqual(status["minimum_validation_loss_epoch"], 1)

    def test_jointly_worsening_losses_are_not_classed_as_overfitting(self):
        rows = self.history([1 + index * .1 for index in range(30)])
        for row in rows:
            row["train_loss"] = row["validation_loss"]
        self.assertFalse(continuation.divergence_status(rows, 10)["confirmed"])

    def test_nonfinite_loss_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            continuation.divergence_status(self.history([1., float("nan")]), 0)

    def test_continuation_preserves_parent_and_restores_optimizer_on_next_resume(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            folder = root / "data" / "imagenet32"
            folder.mkdir(parents=True)
            images = np.zeros((4, 3, 32, 32), dtype=np.uint8)
            labels = np.array([0, 1, 2, 3])
            for name, data in (("X_train", images), ("y_train", labels),
                               ("X_test", images[:2]), ("y_test", labels[:2])):
                np.save(folder / f"{name}.npy", data)
            parent = root / "runs" / "parent"
            settings = notebook.optimizer_config("AdamW")
            settings["lr"] = 0.
            notebook.train_with_metrics("imagenet32", images, labels, seed=42, device="cpu",
                                        run_dir=parent, epochs=1, optimizer_settings=settings)
            original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.iterdir()}
            original_check_output = continuation.subprocess.check_output
            commit = original_check_output(["git", "rev-parse", "HEAD"], text=True)

            def git_output(command, **kwargs):
                return commit if command == ["git", "rev-parse", "HEAD"] else original_check_output(command, **kwargs)

            def run(source, maximum, checkpoint="best.pt"):
                arguments = ["continue_experiment.py", str(source), "--device", "cpu",
                             "--max-epochs", str(maximum), "--checkpoint", checkpoint]
                before = set((root / "runs").iterdir())
                with patch.object(continuation, "__file__", str(root / "scripts/continue_experiment.py")), \
                     patch.object(sys, "argv", arguments), \
                     patch.object(continuation.subprocess, "check_output", side_effect=git_output), \
                     patch.object(continuation, "finish_run") as finish, \
                     patch.object(notebook, "BATCH_SIZE", notebook.BATCH_SIZE):
                    continuation.main()
                    finish.assert_called_once()
                return (set((root / "runs").iterdir()) - before).pop()

            first = run(parent, 2)
            state = torch.load(first / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(state["epoch"], 2)
            self.assertFalse(state["config"]["continuation"]["optimizer_state_restored"])
            step = next(iter(state["optimizer_states"][0]["state"].values()))["step"]
            self.assertEqual(step.item(), 1)
            second = run(first, 3, "last.pt")
            resumed = torch.load(second / "last.pt", map_location="cpu", weights_only=False)
            self.assertTrue(resumed["config"]["continuation"]["optimizer_state_restored"])
            self.assertTrue(resumed["config"]["continuation"]["rng_state_restored"])
            self.assertEqual(resumed["config"]["continuation"]["origin_epoch"], 1)
            step = next(iter(resumed["optimizer_states"][0]["state"].values()))["step"]
            self.assertEqual(step.item(), 2)
            for name, tensor in state["state_dict"].items():
                torch.testing.assert_close(resumed["state_dict"][name], tensor, rtol=0, atol=0)
            torch.testing.assert_close(resumed["training_indices"], state["training_indices"])
            self.assertEqual(json.loads((second / "divergence.json").read_text())["stop_reason"], "epoch_guard")
            self.assertEqual(original, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.iterdir()})


if __name__ == "__main__":
    unittest.main()
