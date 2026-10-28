"""Experimental ImageNet model; public grading recipes do not import this file."""

import json

import torch
from torch import nn
from torch.nn import functional as F

from residual_mlp_baseline import training_core as core
from residual_mlp_baseline.analysis import FixedBatchProbe, rms


class PreReductionBlock(nn.Module):
    """Mix all channels and positions while preserving an N×64×2×2 map."""

    def __init__(self, features=256, hidden=512):
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.fc1 = nn.Linear(features, hidden)
        self.fc2 = nn.Linear(hidden, features)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, images):
        stream = images.flatten(1)
        branch = self.fc2(F.gelu(self.fc1(self.norm(stream))))
        return (stream + branch).reshape_as(images)


class ResidualMLP(core.ResidualMLP):
    """Add one 256→512→256 residual MLP immediately before stage six.

    Existing parameters retain their seed-matched baseline initialization. A CPU
    RNG fork initializes the added block without changing the subsequent shuffle
    RNG state. No saved weights are used.
    """

    def __init__(self, in_features=3072, num_classes=1000):
        super().__init__(in_features, num_classes)
        if self.window_shapes[-2] != (64, 2, 2):
            raise ValueError("The experiment expects a 64×2×2 pre-reduction map")
        with torch.random.fork_rng(devices=[]):
            self.pre_reduction = PreReductionBlock()

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != self.image_shape:
            raise ValueError(f"Expected NCHW images with C,H,W={self.image_shape}")
        for index, window_mlp in enumerate(self.window_mlps):
            if index == len(self.window_mlps) - 1:
                x = self.pre_reduction(x)
            x = window_mlp(x)
        x = self.input_layer(x.flatten(1))
        for block in self.blocks:
            x = block(x)
        return self.head_fc2(x)


class CandidateProbe(FixedBatchProbe):
    """Extend the unchanged fixed-batch probe to include the added skip branch."""

    def __call__(self, model, optimizers, epoch, stage):
        measured = {}

        def capture(module, inputs, output):
            stream = inputs[0].detach()
            measured["ratio"] = (rms(output - stream) / rms(stream).clamp_min(1e-12)).item()
            measured["variance"] = output.detach().float().var(unbiased=False).item()

        # deepcopy in FixedBatchProbe carries this read-only hook to its probe
        # copy. The live training model is never evaluated or updated here.
        count = len(self.rows)
        handle = model.pre_reduction.register_forward_hook(capture)
        try:
            super().__call__(model, optimizers, epoch, stage)
        finally:
            handle.remove()
        if len(self.rows) == count:
            return
        row = self.rows[-1]
        row["branch_to_stream_ratio"] = {"pre_reduction": measured["ratio"],
                                          **row["branch_to_stream_ratio"]}
        activation = {}
        for name, value in row["activation_variance"].items():
            activation[name] = value
            if name == "pre_reduction.fc2":
                activation["pre_reduction.residual_stream"] = measured["variance"]
        row["activation_variance"] = activation
        (self.directory / "training-dynamics.json").write_text(
            json.dumps(self.rows, indent=2, allow_nan=False) + "\n")
