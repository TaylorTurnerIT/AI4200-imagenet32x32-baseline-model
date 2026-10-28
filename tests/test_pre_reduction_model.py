"""Check the single architectural change before the authorized full-data run."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pre_reduction_model import CandidateProbe, ResidualMLP
from run_pre_reduction_experiment import reserve_attempt
from residual_mlp_baseline import submission, training_core as core


class PreReductionTests(unittest.TestCase):
    def test_attempt_guard_refuses_a_second_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attempt"
            reserve_attempt(output, {"seed": 0})
            original = (output / "attempt.json").read_bytes()
            with self.assertRaises(FileExistsError):
                reserve_attempt(output, {"seed": 42})
            self.assertEqual((output / "attempt.json").read_bytes(), original)

    def test_baseline_weights_and_shuffle_rng_are_preserved(self):
        for seed in (0, 42):
            torch.manual_seed(seed)
            baseline = core.ResidualMLP(**core.TASKS["imagenet32"])
            baseline_rng = torch.get_rng_state()
            torch.manual_seed(seed)
            candidate = ResidualMLP()
            self.assertTrue(torch.equal(torch.get_rng_state(), baseline_rng))
            for name, tensor in baseline.state_dict().items():
                self.assertTrue(torch.equal(tensor, candidate.state_dict()[name]), name)
            difference = sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in baseline.parameters())
            self.assertEqual(difference, 263424)

    def test_zero_branch_recovers_exact_baseline_logits(self):
        torch.manual_seed(0)
        baseline = core.ResidualMLP(**core.TASKS["imagenet32"]).eval()
        torch.manual_seed(0)
        candidate = ResidualMLP().eval()
        torch.nn.init.zeros_(candidate.pre_reduction.fc2.weight)
        torch.nn.init.zeros_(candidate.pre_reduction.fc2.bias)
        images = torch.randn(2, 3, 32, 32)
        torch.testing.assert_close(candidate(images), baseline(images), rtol=0, atol=0)

    def test_new_block_receives_two_by_two_map_and_gradients(self):
        model = ResidualMLP()
        shapes = []
        handle = model.pre_reduction.register_forward_hook(
            lambda module, inputs, output: shapes.append((tuple(inputs[0].shape), tuple(output.shape))))
        try:
            logits = model(torch.randn(2, 3, 32, 32))
            F.cross_entropy(logits, torch.tensor([0, 1])).backward()
        finally:
            handle.remove()
        self.assertEqual(shapes, [((2, 64, 2, 2), (2, 64, 2, 2))])
        for name, parameter in model.pre_reduction.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.pre_reduction.fc1.weight.grad.abs().sum().item(), 0)

    def test_probe_includes_new_branch_without_changing_training(self):
        images = np.random.default_rng(4).integers(0, 256, (8, 3, 32, 32), dtype=np.uint8)
        labels = np.arange(8) % 2
        arguments = dict(seed=0, device="cpu", epochs=1, model_factory=ResidualMLP,
                         augmentation=submission.augmentation_for("imagenet32"),
                         optimizer_settings=submission.RECIPES["imagenet32"]["optimizer"])
        plain = core.train_with_metrics("imagenet32", images, labels, **arguments)
        with tempfile.TemporaryDirectory() as directory:
            probe = CandidateProbe(images, labels, 0, Path(directory) / "probe")
            measured = core.train_with_metrics("imagenet32", images, labels, observer=probe, **arguments)
            self.assertEqual([row["stage"] for row in probe.rows], ["initial", "epoch", "selected_checkpoint"])
            for row in probe.rows:
                self.assertIn("pre_reduction", row["branch_to_stream_ratio"])
                self.assertIn("pre_reduction.residual_stream", row["activation_variance"])
            for name, tensor in plain.state_dict().items():
                self.assertTrue(torch.equal(tensor, measured.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
