"""Shared per-class metrics across datasets and clean split evaluation."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from residual_mlp_baseline import notebook


class ClassAccuracyTests(unittest.TestCase):
    def test_all_tasks_count_by_true_class_and_include_absent_classes(self):
        for task, count in [('mnist', 10), ('cifar10', 10), ('imagenet32', 1000)]:
            with self.subTest(task=task):
                # Unequal supports distinguish class recall from prediction precision.
                labels = np.array([0, 0, 0, 1, count - 1])
                predictions = np.array([0, 0, 1, 1, 0])
                rows = notebook.per_class_accuracy(task, labels, predictions)
                self.assertEqual(len(rows), count)
                self.assertEqual(rows[0]['correct'], 2)
                self.assertEqual(rows[0]['total'], 3)
                self.assertAlmostEqual(rows[0]['accuracy'], 2 / 3)
                self.assertEqual(rows[1]['accuracy'], 1.)
                self.assertEqual(rows[-1]['accuracy'], 0.)
                self.assertEqual(rows[2]['total'], 0)
                self.assertIsNone(rows[2]['accuracy'])
                self.assertEqual(sum(row['correct'] for row in rows), 3)
                self.assertEqual(sum(row['total'] for row in rows), len(labels))
                expected = {'mnist': '0', 'cifar10': 'airplane', 'imagenet32': 'class 0'}
                self.assertEqual(rows[0]['class_name'], expected[task])
                self.assertEqual(rows[-1]['class_id'], count - 1)

    def test_empty_split_has_no_invented_accuracy(self):
        rows = notebook.per_class_accuracy('mnist', [], [])
        self.assertTrue(all(row['total'] == 0 and row['accuracy'] is None for row in rows))

    def test_invalid_labels_and_shapes_are_rejected(self):
        for labels, predictions in [([0, 1], [0]), ([[0]], [[0]]),
                                    ([-1], [0]), ([10], [0]), ([0], [10]), ([0.5], [0])]:
            with self.subTest(labels=labels, predictions=predictions), self.assertRaises(ValueError):
                notebook.per_class_accuracy('mnist', labels, predictions)

    def test_chunked_evaluation_scores_only_selected_indices(self):
        images = np.arange(7)[:, None]
        labels = np.array([0, 1, 0, 1, 0, 1, 1])
        indices = np.array([6, 1, 4])
        seen = []

        def predict(model, batch):
            seen.extend(batch[:, 0].tolist())
            return torch.nn.functional.one_hot(torch.as_tensor(batch[:, 0] % 2), 10).float()

        with patch.object(notebook, 'predict', predict):
            rows = notebook.evaluate_per_class(None, 'mnist', images, labels, indices, chunk=2)
        self.assertEqual(seen, [6, 1, 4])
        self.assertEqual(rows[0]['total'], 1)
        self.assertEqual(rows[0]['accuracy'], 1.)
        self.assertEqual(rows[1]['total'], 2)
        self.assertEqual(rows[1]['accuracy'], .5)


if __name__ == '__main__':
    unittest.main()
