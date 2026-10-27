"""Check geometry, pixel layout, and restoration for analysis-only interventions."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ablate_window_stages import controlled_window_output, spatial_control, weight_digest
from residual_mlp_baseline import training_core as core


class WindowStageAblationTests(unittest.TestCase):
    def test_controls_preserve_geometry_and_channel_mapping(self):
        stage = core.RollingWindowBlock(channels=2, window_hidden=4, window_features=3,
                                        stride=2, padding=1)
        images = torch.arange(50, dtype=torch.float32).reshape(1, 2, 5, 5)
        captured = []
        handle = stage[0].register_forward_pre_hook(lambda layer, inputs: captured.append(inputs[0].clone()))
        try:
            single = controlled_window_output(stage, images, "single_pixel")
            self.assertEqual(tuple(single.shape), (1, 3, 3, 3))
            torch.testing.assert_close(captured[-1][0, 0, 0], torch.tensor([0., 0., 0., 0., 25., 25., 25., 25.]))
            torch.testing.assert_close(captured[-1][0, 2, 2], torch.tensor([24., 24., 24., 24., 49., 49., 49., 49.]))
            mean = controlled_window_output(stage, images, "mean_only")
            self.assertEqual(tuple(mean.shape), tuple(single.shape))
            torch.testing.assert_close(captured[-1][0, 0, 0], torch.tensor([3., 3., 3., 3., 28., 28., 28., 28.]))
            torch.testing.assert_close(captured[-1][0, 2, 2], torch.tensor([6., 6., 6., 6., 12.25, 12.25, 12.25, 12.25]))
        finally:
            handle.remove()

    def test_each_stage_restores_logits_and_weights_after_exception(self):
        torch.manual_seed(7)
        model = core.ResidualMLP(**core.TASKS["imagenet32"], width=8, hidden=16,
                                 window_hidden=8, window_features=4).eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.inference_mode():
            original = model(images).clone()
            digest = weight_digest(model)
            for stage in model.window_mlps:
                for mode in ("single_pixel", "mean_only"):
                    with self.assertRaisesRegex(RuntimeError, "test interruption"):
                        with spatial_control(stage, mode):
                            self.assertEqual(tuple(model(images).shape), tuple(original.shape))
                            raise RuntimeError("test interruption")
                    torch.testing.assert_close(model(images), original, rtol=0, atol=0)
            self.assertEqual(weight_digest(model), digest)

    def test_unknown_control_does_not_install_hook(self):
        stage = core.RollingWindowBlock()
        with self.assertRaises(ValueError):
            with spatial_control(stage, "unknown"):
                pass
        self.assertFalse(stage._forward_hooks)


if __name__ == "__main__":
    unittest.main()
