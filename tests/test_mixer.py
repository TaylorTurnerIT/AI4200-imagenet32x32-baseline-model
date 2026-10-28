"""Paper equations, patch ordering, transfer initialization, and local training."""

import csv
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from residual_mlp_baseline import notebook
from residual_mlp_baseline.mixer import MIXER_PRESETS, DropPath, MixerBlock, MixerConfig, MlpMixer, mixer_config
from residual_mlp_baseline.mixer_training import (
    ImageTransform, MixerRecipe, learning_rate_factor, predict_mixer, task_config, train_mixer,
)


def tiny_config(task="cifar10"):
    return task_config(task, num_blocks=2, hidden_dim=8, tokens_mlp_dim=5, channels_mlp_dim=13)


def tiny_recipe(**kwargs):
    return MixerRecipe(**(dict(epochs=2, batch_size=4, warmup_steps=1, randaugment_layers=2,
                               mixup_alpha=.5, stochastic_depth=.1) | kwargs))


class MixerArchitectureTests(unittest.TestCase):
    def test_block_matches_equation_one_and_shares_mlp_weights(self):
        config = tiny_config()
        block = MixerBlock(config).double().eval()
        x = torch.randn(2, config.num_tokens, config.hidden_dim, dtype=torch.float64, requires_grad=True)
        # Deliberately compute independent channel/token loops instead of transposes.
        normalized = F.layer_norm(x, (config.hidden_dim,), block.token_norm.weight,
                                  block.token_norm.bias, 1e-6)
        columns = []
        for channel in range(config.hidden_dim):
            value = F.linear(normalized[:, :, channel], block.token_mixing.fc1.weight, block.token_mixing.fc1.bias)
            value = F.gelu(value, approximate="tanh")
            columns.append(F.linear(value, block.token_mixing.fc2.weight, block.token_mixing.fc2.bias))
        u = x + torch.stack(columns, dim=-1)
        normalized = F.layer_norm(u, (config.hidden_dim,), block.channel_norm.weight,
                                  block.channel_norm.bias, 1e-6)
        rows = []
        for token in range(config.num_tokens):
            value = F.linear(normalized[:, token], block.channel_mixing.fc1.weight, block.channel_mixing.fc1.bias)
            rows.append(F.linear(F.gelu(value, approximate="tanh"),
                                 block.channel_mixing.fc2.weight, block.channel_mixing.fc2.bias))
        expected = u + torch.stack(rows, dim=1)
        actual = block(x)
        torch.testing.assert_close(actual, expected)
        grad_actual = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
        grad_expected = torch.autograd.grad(expected.square().sum(), x)[0]
        torch.testing.assert_close(grad_actual, grad_expected)
        self.assertEqual(sum(isinstance(m, nn.Linear) for m in block.modules()), 4)

    def test_patch_projection_matches_reference_strided_convolution(self):
        config = replace(tiny_config(), image_size=8, patch_size=2)
        model = MlpMixer(config).double()
        x = torch.arange(2*3*8*8, dtype=torch.float64).reshape(2, 3, 8, 8)
        # Flax stem kernel layout is [patch_h, patch_w, channels, hidden].
        weight = model.stem.weight.reshape(config.hidden_dim, 2, 2, 3).permute(0, 3, 1, 2)
        expected = F.conv2d(x, weight, model.stem.bias, stride=2).flatten(2).transpose(1, 2)
        torch.testing.assert_close(model.patch_embeddings(x), expected)
        self.assertFalse(any(isinstance(m, (nn.Conv2d, nn.MultiheadAttention)) for m in model.modules()))

    def test_final_norm_mean_and_zero_initialized_head(self):
        model = MlpMixer(tiny_config()).eval()
        x = torch.randn(2, 3, 32, 32)
        torch.testing.assert_close(model(x), torch.zeros(2, 10))
        nn.init.normal_(model.head.weight)
        features = model.patch_embeddings(x)
        for block in model.blocks:
            features = block(features)
        expected = model.head(model.pre_head_layer_norm(features).mean(1))
        torch.testing.assert_close(model(x), expected)
        self.assertFalse(any("position" in name for name, _ in model.named_parameters()))

    def test_all_table_one_presets_without_allocating_large_models(self):
        expected = {"S/32": (8, 512, 256, 2048, 49, 19), "S/16": (8, 512, 256, 2048, 196, 18),
                    "B/32": (12, 768, 384, 3072, 49, 60), "B/16": (12, 768, 384, 3072, 196, 59),
                    "L/32": (24, 1024, 512, 4096, 49, 206), "L/16": (24, 1024, 512, 4096, 196, 207),
                    "H/14": (32, 1280, 640, 5120, 256, 431)}
        self.assertEqual(set(expected), set(MIXER_PRESETS))
        for name, values in expected.items():
            with self.subTest(preset=name), torch.device("meta"):
                config = mixer_config(name)
                model = MlpMixer(config)
                self.assertEqual((config.num_blocks, config.hidden_dim, config.tokens_mlp_dim,
                                  config.channels_mlp_dim, config.num_tokens), values[:5])
                count = sum(p.numel() for key, p in model.named_parameters() if not key.startswith("head."))
                self.assertLess(abs(count/1e6-values[-1]), 1)

    def test_backward_reaches_backbone_after_zero_head_first_update(self):
        model = MlpMixer(tiny_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        x = torch.randn(3, 3, 32, 32)
        for _ in range(2):
            optimizer.zero_grad()
            F.cross_entropy(model(x), torch.tensor([0, 1, 2])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
        self.assertGreater(model.stem.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.blocks[0].token_mixing.fc1.weight.grad.abs().sum().item(), 0)

    def test_stochastic_depth_per_example_ramp_and_eval(self):
        model = MlpMixer(replace(tiny_config(), stochastic_depth=.3))
        rates = [path.probability for block in model.blocks for path in (block.token_drop, block.channel_drop)]
        np.testing.assert_allclose(rates, [0, .1, .2, .3])
        x = torch.ones(128, 3, 4)
        path = DropPath(.5).train()
        y = path(x)
        self.assertTrue(set(y.unique().tolist()) == {0., 2.})
        self.assertTrue(all(row.unique().numel() == 1 for row in y))
        torch.testing.assert_close(path.eval()(x), x)

    def test_appendix_c_matches_independent_spatial_tiles(self):
        model = MlpMixer(replace(tiny_config(), image_size=8, patch_size=2)).double().eval()
        resized = model.resize_for_finetuning(16)
        self.assertEqual(resized.config.tokens_mlp_dim, 4*model.config.tokens_mlp_dim)
        image = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        features = [model.forward_features(image[:, :, y:y+8, x:x+8])
                    for y in (0, 8) for x in (0, 8)]
        torch.testing.assert_close(resized.forward_features(image), torch.stack(features).mean(0))
        weights = resized.blocks[0].token_mixing.fc1.weight
        self.assertEqual(torch.count_nonzero(weights[:5, 16:]).item(), 0)
        self.assertEqual(model.config.image_size, 8)
        self.assertEqual(next(resized.parameters()).dtype, torch.float64)
        with self.assertRaises(ValueError):
            model.resize_for_finetuning(12)

    def test_invalid_dimensions_and_images_fail_clearly(self):
        for kwargs in ({"image_size": 31}, {"patch_size": 0}, {"hidden_dim": True},
                       {"dropout": 1}, {"stochastic_depth": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MixerConfig(**kwargs)
        with self.assertRaisesRegex(ValueError, "NCHW"):
            MlpMixer(tiny_config())(torch.randn(2, 32, 32, 3))


class MixerTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.images = np.random.default_rng(42).integers(0, 256, (12, 3, 32, 32), dtype=np.uint8)
        self.labels = np.arange(12) % 3

    def test_schedule_warmup_and_both_decay_modes(self):
        self.assertEqual([learning_rate_factor(i, 10, 2) for i in range(3)], [.5, 1., 1.])
        self.assertEqual(learning_rate_factor(6, 10, 2), .5)
        self.assertEqual(learning_rate_factor(10, 10, 2), 0)
        self.assertAlmostEqual(learning_rate_factor(4, 10, 2, "finetune"), (1+np.cos(np.pi/4))/2)
        self.assertEqual(learning_rate_factor(0, 1, 0), 1)

    def test_randaugment_every_operation_and_repeatable_stream(self):
        for channels in (1, 3):
            image = self.images[0, :channels]
            a = ImageTransform(32, tiny_recipe(), 7, True)
            b = ImageTransform(32, tiny_recipe(), 7, True)
            torch.testing.assert_close(a(image), b(image))
            from PIL import Image
            pil = Image.fromarray(image[0] if channels == 1 else image.transpose(1, 2, 0))
            for operation in a.operations:
                with self.subTest(channels=channels, operation=operation):
                    augmented = a.augment(pil, operation)
                    self.assertEqual(augmented.size, (32, 32))
                    self.assertEqual(augmented.mode, pil.mode)
        clean = ImageTransform(32, tiny_recipe(), 7, False)(self.images[0])
        torch.testing.assert_close(clean, torch.tensor(self.images[0]).float()/127.5-1)

    def test_training_persistence_determinism_clipping_and_no_network(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch('urllib.request.urlopen', side_effect=AssertionError('No result uploads')) as network, \
                patch('torch.nn.utils.clip_grad_norm_', wraps=nn.utils.clip_grad_norm_) as clip:
            root = Path(folder)
            model = train_mixer("cifar10", self.images, self.labels, seed=7, device="cpu", run_dir=root/"run",
                                config=tiny_config(), recipe=tiny_recipe())
            self.assertEqual(clip.call_count, 6)
            self.assertTrue(all(call.args[1] == 1.0 for call in clip.call_args_list))
            config = json.loads((root/"run/config.json").read_text())
            self.assertFalse(config['leaderboard_enabled'])
            self.assertEqual(config['optimizer'], 'AdamW')
            self.assertEqual(config['warmup_steps'], 1)
            self.assertEqual(config['architecture']['tokens_mlp_dim'], 5)
            self.assertEqual(len(set(model.training_indices.tolist()) & set(model.validation_indices.tolist())), 0)
            self.assertEqual(len(model.training_indices)+len(model.validation_indices), len(self.images))
            checkpoint = torch.load(root/"run/best.pt", weights_only=True)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), checkpoint['state_dict'][name])
            with (root/"run/metrics.csv").open() as file:
                self.assertEqual(len(list(csv.DictReader(file))), 2)
            self.assertGreater(model.head.weight.abs().sum().item(), 0)
            again = train_mixer("cifar10", self.images, self.labels, seed=7, device="cpu",
                                config=tiny_config(), recipe=tiny_recipe())
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, again.state_dict()[name], rtol=0, atol=0)
            probabilities = predict_mixer(model, self.images)
            self.assertEqual(probabilities.shape, (12, 10))
            torch.testing.assert_close(probabilities.sum(1), torch.ones(12))
            notebook.save_run_result(root/"results.csv", root/"run", model, .1, .2, .01, 12)
            self.assertTrue((root/"run/summary.json").exists())
            self.assertEqual(list(root.rglob('leaderboard*')), [])
            network.assert_not_called()

    def test_direct_mixer_api_trains_each_dataset_from_scratch(self):
        for task in notebook.TASKS:
            with self.subTest(task=task), patch('torch.load', side_effect=AssertionError('No saved weights')):
                images = self.images if task != 'mnist' else self.images[:, :1, :28, :28]
                model = train_mixer(task, images, self.labels, device='cpu',
                                    config=tiny_config(task), recipe=tiny_recipe(epochs=1))
                self.assertIsInstance(model, MlpMixer)
                self.assertEqual(model.config.num_classes, notebook.TASKS[task]['num_classes'])
                self.assertEqual(predict_mixer(model, images).shape, (12, model.config.num_classes))
                self.assertFalse(model.run_config['leaderboard_enabled'])

    def test_finetuning_uses_sgd_and_cosine_without_decay(self):
        recipe = MixerRecipe.finetuning(epochs=2, batch_size=4, warmup_steps=1)
        model = train_mixer("cifar10", self.images, self.labels, config=tiny_config(), recipe=recipe, device="cpu")
        self.assertEqual(model.run_config['optimizer'], 'SGD')
        self.assertEqual(model.run_config['learning_rate_schedule'], 'linear_warmup_cosine_decay')
        self.assertEqual(model.run_config['weight_decay'], 0)

    def test_transfer_replaces_head_and_removes_pretraining_regularization(self):
        pretrained = MlpMixer(replace(tiny_config(), dropout=.1, stochastic_depth=.1))
        pretrained.reset_classifier(3)
        self.assertEqual(pretrained.head.out_features, 3)
        self.assertEqual(torch.count_nonzero(pretrained.head.weight).item(), 0)
        recipe = MixerRecipe.finetuning(epochs=1, batch_size=4, warmup_steps=0)
        with patch('torch.load', side_effect=AssertionError('No checkpoint loading')):
            model = train_mixer('cifar10', self.images, self.labels, device='cpu',
                                model=pretrained, recipe=recipe)
        self.assertIs(model, pretrained)
        self.assertEqual(model.config.stochastic_depth, 0)
        self.assertTrue(all(module.p == 0 for module in model.modules() if isinstance(module, nn.Dropout)))
        self.assertTrue(all(module.probability == 0 for module in model.modules() if isinstance(module, DropPath)))
        self.assertFalse(model.run_config['augmentation']['random_resized_crop'])
        self.assertEqual(tuple(predict_mixer(model, self.images).shape), (12, 3))

    def test_bad_training_settings_and_empty_prediction(self):
        for kwargs in ({'epochs': True}, {'batch_size': 0}, {'learning_rate': float('nan')},
                       {'warmup_steps': -1}, {'randaugment_magnitude': 31}, {'phase': 'finetune'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MixerRecipe(**kwargs)
        with self.assertRaises(ValueError):
            train_mixer('cifar10', self.images.astype(float), self.labels, recipe=tiny_recipe(), device='cpu')
        model = MlpMixer(tiny_config())
        self.assertEqual(tuple(predict_mixer(model, self.images[:0]).shape), (0, 10))


if __name__ == '__main__':
    unittest.main()
