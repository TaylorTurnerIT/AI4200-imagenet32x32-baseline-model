"""Frozen task-specific grading recipes; every call builds fresh model weights.

The notebook remains the experimental dashboard. The public train() entry point
uses the recipes here so dashboard edits cannot silently change the submission.
"""

import json
import os
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn
from torch.nn import functional as F

from . import training_core as core


class MnistBlock(nn.Module):
    def __init__(self, width=256, hidden=512):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class MnistResidualMLP(nn.Module):
    """Seven-block GELU model recovered from commit 6c19012 and its run metadata.

    Module construction and Xavier traversal preserve the historical seeded
    parameter order even though the seven repeated blocks now share a class.
    """

    def __init__(self, in_features=784, num_classes=10, width=256, hidden=512, depth=7):
        super().__init__()
        self.input_layer = nn.Linear(in_features, width)
        self.blocks = nn.ModuleList([MnistBlock(width, hidden) for _ in range(depth)])
        self.head_norm = nn.LayerNorm(width)
        self.head_fc1 = nn.Linear(width, hidden)
        self.head_fc2 = nn.Linear(hidden, num_classes)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.input_layer(x.flatten(1))
        for block in self.blocks:
            x = block(x)
        return self.head_fc2(F.gelu(self.head_fc1(self.head_norm(x))))

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return self(x).softmax(1)


RECIPES = {
    "mnist": {
        "source_run": "mnist-20261023-105208-872d486e", "epochs": 500,
        "max_seconds": 15 * 60,
        "optimizer": {"name": "SGD", "lr": 0.05, "weight_decay": 0.0001,
                      "momentum": 0.0, "dampening": 0.0, "nesterov": False},
    },
    "cifar10": {
        "source_run": "cifar10-20261026-224057-085c2c3b", "epochs": 60,
        "max_seconds": 15 * 60,
        "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.00001,
                      "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "amsgrad": True},
    },
    "imagenet32": {
        "source_run": "imagenet32-20261027-000850-1c570f4f", "epochs": 25,
        "max_seconds": 60 * 60,
        "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.00001,
                      "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "amsgrad": True},
    },
}


def augmentation_for(task):
    """Freeze effective transforms independently of dashboard controls."""
    default = {
        "rotation": {"probability": 0.5, "amount": 10.0},
        "translation": {"probability": 0.5, "amount": 2.0},
        "scale": {"probability": 0.5, "amount": 0.1},
        "shear": {"probability": 0.25, "amount": 5.0},
        "brightness": {"probability": 0.25, "amount": 0.1},
        "noise": {"probability": 0.25, "amount": 0.05},
        "erasing": {"probability": 0.1, "amount": 0.05},
    }
    if task == "cifar10":
        for name, probability in {"rotation": 0.45, "scale": 0.55, "shear": 0.5,
                                  "brightness": 0.45, "noise": 0.5, "erasing": 0.5}.items():
            default[name]["probability"] = probability
        default["translation"]["amount"] = 4.0
        default["horizontal_flip"] = {"probability": 0.5, "amount": 1.0}
    elif task == "imagenet32":
        default["horizontal_flip"] = {"probability": 0.5, "amount": 1.0}
    return {"task": task, "enabled": task != "imagenet32", "default": default,
            "classes": {str(i): json.loads(json.dumps(default)) for i in range(10)}
            if task == "mnist" else {}}


def train(task, X_train, y_train, seed=0, device="cuda"):
    """Build and train one model from scratch within the task's time allocation.

    Allocations sum to 90 minutes across all three tasks. A small internal
    reserve leaves time to restore the best complete validation checkpoint.
    AI4200_EVIDENCE_DIR optionally saves analysis artifacts; nothing is loaded.
    """
    if task not in RECIPES:
        raise ValueError(f"Unknown task: {task}")
    recipe = RECIPES[task]
    root = os.environ.get("AI4200_EVIDENCE_DIR")
    run_dir = Path(root) / f"{task}-{uuid4().hex[:8]}" if root else None
    observer = None
    analysis_root = os.environ.get("AI4200_ANALYSIS_DIR")
    if task == "imagenet32" and analysis_root:
        from .analysis import FixedBatchProbe
        observer = FixedBatchProbe(X_train, y_train, seed,
                                   Path(analysis_root) / f"{task}-{uuid4().hex[:8]}")
    factory = MnistResidualMLP if task == "mnist" else core.ResidualMLP
    model = core.train_with_metrics(
        task, X_train, y_train, seed=seed, device=device, run_dir=run_dir,
        augmentation=augmentation_for(task), optimizer_settings=recipe["optimizer"],
        epochs=recipe["epochs"], model_factory=factory, batch_size=256,
        max_seconds=recipe["max_seconds"],
        observer=observer,
    )
    model.run_config["submission_recipe"] = recipe
    if run_dir is not None:
        (run_dir / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    return model


def predict(model, X_test):
    """Return one CPU probability row per supplied image."""
    return core.predict(model, X_test)
