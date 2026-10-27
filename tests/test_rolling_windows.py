"""Check hand-built window stages against explicit spatial loops."""

import ast
import inspect
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from residual_mlp_baseline import notebook


class RollingWindowTests(unittest.TestCase):
    def test_custom_blocks_preserve_reference_initialization_logits_and_gradients(self):
        snapshot = json.loads((Path(__file__).parent / 'fixtures/rolling-window-reference.json').read_text())
        namespace = {'torch': torch, 'nn': nn, 'F': nn.functional}
        for key in ('block_source', 'model_source'):
            _, source = notebook.class_snapshot(snapshot[key])
            exec(compile(ast.parse(source), '<reference-windows>', 'exec'), namespace)
        for task, settings in notebook.TASKS.items():
            with self.subTest(task=task), torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                reference = namespace['ResidualMLP'](**settings).eval()
                torch.manual_seed(42)
                actual = notebook.ResidualMLP(**settings).eval()
                self.assertEqual(reference.state_dict().keys(), actual.state_dict().keys())
                for name, expected in reference.state_dict().items():
                    torch.testing.assert_close(actual.state_dict()[name], expected, rtol=0, atol=0)
                images = torch.randn(2, *actual.image_shape)
                expected = reference(images)
                result = actual(images)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                expected.sum().backward()
                result.sum().backward()
                for (_, a), (_, b) in zip(actual.named_parameters(), reference.named_parameters()):
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_every_stage_matches_explicit_window_loop(self):
        for task, settings in notebook.TASKS.items():
            with self.subTest(task=task):
                model = notebook.ResidualMLP(**settings, width=8, depth=1,
                                             hidden=16, window_hidden=8, window_features=4).double()
                image = torch.arange(settings['in_features'], dtype=torch.float64) / settings['in_features']
                image = image.reshape(1, *model.image_shape).transpose(2, 3)
                # Exercise non-contiguous inputs and each stage's different channel count.
                for stage, mlp in enumerate(model.window_mlps):
                    _, height, width = model.window_shapes[stage]
                    stride = model.window_strides[stage]
                    actual = model.extract_windows(image, stage)
                    # Explicitly append zeros rather than dropping unmatched edges.
                    if stride == 2 and height % 2:
                        image = torch.cat((image, image.new_zeros(1, image.shape[1], 1, width)), dim=2)
                        height += 1
                    if stride == 2 and width % 2:
                        image = torch.cat((image, image.new_zeros(1, image.shape[1], height, 1)), dim=3)
                        width += 1
                    expected = torch.stack([
                        image[:, :, row:row + 2, column:column + 2].flatten(1)
                        for row in range(0, height - 1, stride)
                        for column in range(0, width - 1, stride)
                    ], dim=1)
                    torch.testing.assert_close(actual, expected)
                    # One shared MLP must agree with applying it separately at every position.
                    expected_features = torch.stack([nn.Sequential.forward(mlp, window) for window in expected.unbind(1)], dim=1)
                    features = nn.Sequential.forward(mlp, actual)
                    torch.testing.assert_close(features, expected_features)
                    image = features.transpose(1, 2).reshape(1, *model.window_shapes[stage + 1])
                self.assertEqual(image.shape, (1, 4, 1, 1))

    def test_both_shrinking_strides_support_backward_without_convolution(self):
        for stride, sizes in ((1, list(range(32, 0, -1))), (2, [32, 31, 16, 8, 4, 2, 1])):
            with self.subTest(stride=stride):
                model = notebook.ResidualMLP(**notebook.TASKS['cifar10'], width=8, depth=1,
                                             hidden=16, window_hidden=8, window_features=4,
                                             shrinking_stride=stride)
                self.assertEqual([shape[1] for shape in model.window_shapes], sizes)
                self.assertTrue(all(isinstance(layer, (notebook.ResidualMLP, notebook.Block19,
                                                      nn.ModuleList, nn.Sequential, nn.Linear,
                                                      nn.GELU, nn.LayerNorm, nn.Dropout))
                                    for layer in model.modules()))
                images = torch.randn(2, 3, 32, 32, requires_grad=True)
                with patch('torch.nn.functional.conv2d', side_effect=AssertionError('No convolution')):
                    logits = model(images)
                    nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
                for parameter in [images, *model.parameters()]:
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                torch.testing.assert_close(model.predict(images).sum(1), torch.ones(2))

    def test_saved_source_restores_window_dimensions_and_stride(self):
        model = notebook.ResidualMLP(**notebook.TASKS['cifar10'], width=8, depth=1,
                                     hidden=16, window_hidden=11, window_features=5,
                                     shrinking_stride=1)
        architecture = notebook.model_architecture(model)
        entry = {'id': 'window-roundtrip', 'config': {
            'task': 'cifar10', 'seed': 0, 'device': 'cpu', 'epochs': 1, 'batch_size': 2,
            'validation_fraction': .1, 'architecture': architecture,
            'optimizer_settings': notebook.optimizer_config('AdamW'),
            'augmentation': notebook.augmentation_config('cifar10'),
            'model_source': inspect.getsource(notebook.ResidualMLP),
            'block_source': inspect.getsource(notebook.Block19),
            'support_sources': {'RollingWindowBlock': inspect.getsource(notebook.RollingWindowBlock)},
        }}
        source = notebook.restore_run_source(inspect.getsource(notebook), entry)
        classes = [node for node in ast.parse(source).body
                   if isinstance(node, ast.ClassDef) and node.name != 'ImageAugmenter']
        for node in classes:
            node.decorator_list = []
        namespace = {'torch': torch, 'nn': nn, 'F': nn.functional}
        exec(compile(ast.Module(body=classes, type_ignores=[]), '<restored-windows>', 'exec'), namespace)
        restored = namespace['ResidualMLP'](**notebook.TASKS['cifar10'])
        self.assertEqual(notebook.model_architecture(restored), architecture)
        self.assertEqual(architecture['in_features'], 3072)
        self.assertEqual(len(restored.window_mlps), 31)

    def test_final_spatial_feature_depends_on_every_original_pixel(self):
        for task, settings in notebook.TASKS.items():
            with self.subTest(task=task):
                model = notebook.ResidualMLP(**settings, width=8, depth=1,
                                             window_hidden=8, window_features=4)
                image = torch.ones(1, *model.image_shape, requires_grad=True)
                features = image
                for stage in range(len(model.window_mlps)):
                    # Positive sums trace structural coverage without activation saturation.
                    windows = model.extract_windows(features, stage)
                    values = windows.sum(dim=-1, keepdim=True).expand(-1, -1, 4)
                    features = values.transpose(1, 2).reshape(1, *model.window_shapes[stage + 1])
                features.sum().backward()
                self.assertTrue((image.grad > 0).all())

    def test_wrong_input_shape_fails_before_window_extraction(self):
        model = notebook.ResidualMLP(**notebook.TASKS['cifar10'], width=8)
        for shape in ((2, 3072), (2, 32, 32, 3), (2, 1, 32, 32)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'NCHW'):
                model(torch.zeros(shape))


if __name__ == '__main__':
    unittest.main()
