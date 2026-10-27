"""Model edits must preserve the task interface and accept image batches."""

import unittest

import torch

from residual_mlp_baseline.model import ResidualMLP, TASKS


class ModelTests(unittest.TestCase):
    def test_all_tasks_support_training_and_prediction(self):
        for task, config in TASKS.items():
            with self.subTest(task=task):
                shape = (1, 28, 28) if task == "mnist" else (3, 32, 32)
                model = ResidualMLP(**config, width=8, depth=2)
                self.assertEqual(len(model.blocks), 2)
                images = torch.randn(2, *shape)
                logits = model(images)
                self.assertEqual(logits.shape, (2, config["num_classes"]))
                torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
                self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
                self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
                probabilities = model.predict(images)
                torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))


if __name__ == "__main__":
    unittest.main()
