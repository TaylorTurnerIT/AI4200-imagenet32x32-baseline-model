"""Optional fixed-batch training diagnostics; disabled in ordinary grading runs."""

import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from . import training_core as core


def rms(tensor):
    return tensor.detach().float().square().mean().sqrt()


class FixedBatchProbe:
    """Measure a copy in eval mode without changing the live model or optimizer.

    Update ratios use one probe-batch optimizer step with copied live optimizer
    state. All observations use the same unaugmented training images. This is
    distinct from measuring a random training minibatch's actual update.
    """

    def __init__(self, images, labels, seed, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        split = torch.randperm(len(images), generator=torch.Generator().manual_seed(seed))
        held_out = min(len(images) - 1, max(1, int(len(images) * core.VALIDATION_FRACTION)))
        indices = split[held_out:held_out + min(32, len(images) - held_out)]
        self.images = core.prepare(images[indices.numpy()])
        self.labels = torch.as_tensor(labels[indices.numpy()]).long()
        self.rows = []
        (self.directory / "fixed-batch.json").write_text(json.dumps({
            "training_indices": indices.tolist(), "seed": seed,
            "mode": "eval", "augmentation": False,
            "update_measurement": "One step on a probe copy with copied live optimizer state",
        }, indent=2) + "\n")

    def __call__(self, model, optimizers, epoch, stage):
        if stage == "epoch" and epoch not in (1, 5, 10, 15, 20, 25):
            return
        clone = copy.deepcopy(model).eval()
        device = next(clone.parameters()).device
        images, labels = self.images.to(device), self.labels.to(device)
        linears = {name: layer for name, layer in clone.named_modules() if isinstance(layer, nn.Linear)}
        row = {"epoch": epoch, "stage": stage, "activation_variance": {},
               "branch_to_stream_ratio": {}, "rms_gradient": {}, "update_to_weight_ratio": {},
               "saturation": {"applicable": False, "reason": "This model uses GELU, not tanh or ReLU."}}
        handles = []

        def activation(name):
            def record(layer, inputs, output):
                row["activation_variance"][name] = output.detach().float().var(unbiased=False).item()
            return record

        def branch(name):
            def record(layer, inputs, output):
                stream = inputs[0].detach()
                row["branch_to_stream_ratio"][name] = (rms(output - stream) / rms(stream).clamp_min(1e-12)).item()
                row["activation_variance"][name + ".residual_stream"] = output.detach().float().var(unbiased=False).item()
            return record

        for name, layer in linears.items():
            handles.append(layer.register_forward_hook(activation(name)))
        for index, layer in enumerate(clone.blocks):
            handles.append(layer.register_forward_hook(branch(f"blocks.{index}")))
        copies, _, _ = core.build_optimizers(clone, model.run_config["optimizer_settings"]
                                            if hasattr(model, "run_config") else self.optimizer_settings)
        for original, duplicate in zip(optimizers, copies):
            duplicate.load_state_dict(copy.deepcopy(original.state_dict()))
            duplicate.zero_grad()
        loss = F.cross_entropy(clone(images), labels)
        loss.backward()
        before = {name: layer.weight.detach().clone() for name, layer in linears.items()}
        for name, layer in linears.items():
            row["rms_gradient"][name] = rms(layer.weight.grad).item()
        for optimizer in copies:
            optimizer.step()
        for name, layer in linears.items():
            row["update_to_weight_ratio"][name] = (
                rms(layer.weight - before[name]) / rms(before[name]).clamp_min(1e-12)).item()
        row["probe_loss"] = loss.item()
        for handle in handles:
            handle.remove()
        self.rows.append(row)
        (self.directory / "training-dynamics.json").write_text(json.dumps(self.rows, indent=2, allow_nan=False) + "\n")
