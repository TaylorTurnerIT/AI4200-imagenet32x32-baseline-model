"""Frozen, headless model and training core for the assignment harness.

Extracted from the project's validated notebook implementation. No dashboard
library is imported: the instructor installs the package with --no-deps into a
PyTorch/NumPy environment. The dashboard remains editable independently.
"""

import ast
import csv
import copy
import inspect
import json
import textwrap
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

TASKS = {
        "mnist":      {"in_features": 28 * 28,     "num_classes": 10},    # 28x28 grayscale
        "cifar10":    {"in_features": 3 * 32 * 32, "num_classes": 10},    # 32x32 color
        "imagenet32": {"in_features": 3 * 32 * 32, "num_classes": 1000},  # 32x32 color
    }


RESTORED_FROM = None


AUGMENTATION_CONFIG_OVERRIDES = {'cifar10': {'enabled': True,
                 'default': {'rotation': {'probability': 0.45, 'amount': 10.0},
                             'translation': {'probability': 0.5, 'amount': 4.0},
                             'scale': {'probability': 0.55, 'amount': 0.1},
                             'shear': {'probability': 0.5, 'amount': 5.0},
                             'brightness': {'probability': 0.45, 'amount': 0.1},
                             'noise': {'probability': 0.5, 'amount': 0.05},
                             'erasing': {'probability': 0.5, 'amount': 0.05},
                             'horizontal_flip': {'probability': 0.5, 'amount': 1.0}},
                 'classes': {},
                 'task': 'cifar10'}}


LEARNING_RATE = 0.03


BATCH_SIZE = 256


EPOCHS = {'mnist': 500, 'cifar10': 60, 'imagenet32': 10}


WEIGHT_DECAY = 1e-05


MOMENTUM = 0.0


DAMPENING = 0.0


VALIDATION_FRACTION = 0.1


OPTIMIZER_NAME = 'AdamW'


OPTIMIZER_DEFAULTS = {'SGD': {'lr': 0.05, 'weight_decay': 0.0001, 'momentum': 0.0, 'dampening': 0.0, 'nesterov': False},
     'Adam': {'lr': 0.001,
              'weight_decay': 1e-05,
              'beta1': 0.9,
              'beta2': 0.999,
              'eps': 1e-08,
              'amsgrad': False},
     'AdamW': {'lr': 0.001,
               'weight_decay': 1e-05,
               'beta1': 0.9,
               'beta2': 0.999,
               'eps': 1e-08,
               'amsgrad': True},
     'Muon': {'lr': 0.03,
              'weight_decay': 1e-05,
              'momentum': 0.95,
              'nesterov': True,
              'ns_steps': 5,
              'eps': 1e-07,
              'adjust_lr_fn': 'original',
              'auxiliary_lr': 0.001,
              'auxiliary_weight_decay': 1e-05,
              'auxiliary_beta1': 0.9,
              'auxiliary_beta2': 0.999,
              'auxiliary_eps': 1e-08},
     'RMSprop': {'lr': 0.001,
                 'weight_decay': 1e-05,
                 'momentum': 0.0,
                 'alpha': 0.99,
                 'eps': 1e-08,
                 'centered': False},
     'Adagrad': {'lr': 0.01,
                 'weight_decay': 1e-05,
                 'lr_decay': 0.0,
                 'initial_accumulator_value': 0.0,
                 'eps': 1e-10}}


AUGMENTATION_ENABLED = {"mnist": False, "cifar10": False, "imagenet32": False}


AUGMENTATION_CATEGORIES = {
        # name: (label, unit, maximum slider value, step, default probability, default amount)
        "rotation": ("Rotation", "degrees", 30.0, 1.0, 0.5, 10.0),
        "translation": ("Translation", "pixels per axis", 7.0, 0.5, 0.5, 2.0),
        "scale": ("Scale", "fraction around 1", 0.4, 0.01, 0.5, 0.1),
        "shear": ("Horizontal shear", "degrees", 25.0, 1.0, 0.25, 5.0),
        "brightness": ("Brightness", "fraction around 1", 0.5, 0.01, 0.25, 0.1),
        "noise": ("Gaussian noise", "std on 0–1 intensity", 0.3, 0.01, 0.25, 0.05),
        "erasing": ("Random square erasing", "fraction of image area", 0.25, 0.01, 0.1, 0.05),
    }


class Block19(nn.Module):
    def __init__(self, width=512, h1=256, hidden=1024, dropout=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(width)
        self.value_up = nn.Linear(width, hidden)
        self.value_norm = nn.LayerNorm(hidden)
        self.gate_up = nn.Linear(width, hidden)
        self.gate_norm = nn.LayerNorm(hidden)
        self.down = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        normalized = self.input_norm(x)
        h = self.value_up(normalized)
        value = self.value_norm(h)
        a = F.gelu(self.gate_up(normalized))
        gate = self.gate_norm(a)
        gated = value * gate
        return x + self.dropout(self.down(gated))


class RollingWindowBlock(nn.Sequential):
    """Gather padded 2x2 windows, apply a shared MLP, and return an NCHW map.

    Numeric module names preserve existing window_mlps checkpoint keys.
    """

    def __init__(self, channels=3, window_hidden=128, window_features=64, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.add_module("0", nn.Linear(channels * 4, window_hidden))
        self.add_module("1", nn.GELU())
        self.add_module("2", nn.Linear(window_hidden, window_features))
        self.add_module("3", nn.GELU())

    def extract_windows(self, x):
        """Return (batch, output rows, output columns, C*2*2)."""
        x = F.pad(x, (0, self.padding, 0, self.padding))
        x = x.unfold(2, 2, self.stride).unfold(3, 2, self.stride)
        return x.permute(0, 2, 3, 1, 4, 5).flatten(3)

    def forward(self, x):
        x = self.extract_windows(x)
        x = self[0](x)
        x = self[1](x)
        x = self[2](x)
        x = self[3](x)
        return x.permute(0, 3, 1, 2)


class ResidualMLP(nn.Module):
    # torch-canvas input x: e, channels, 32, 32
    # torch-canvas sizes: in_features=3072, num_classes=10
    """Hierarchical shared window MLPs, followed by a residual classifier.

    input -> sliding 2x2 windows, first stride 1, later shrinking_stride
          -> shared Linear -> GELU -> Linear -> GELU per window
          -> restore feature map and repeat until the spatial size is 1x1
          -> flatten final features -> Linear into the backbone
          -> depth x Block                           residual backbone, stays `width` wide
          -> Linear(width -> num_classes)            classification head
          -> logits (num_classes)

    The head has no skip connection because its output size (num_classes)
    differs from its input size (width), so the two cannot be added.
    """

    def __init__(self, in_features, num_classes, width=512, depth=2, hidden=1024,
                 dropout=0.1, window_hidden=128, window_features=64, shrinking_stride=2):
        super().__init__()

        if in_features != 784 and in_features != 3072:
            raise ValueError("Expected MNIST (784) or RGB 32x32 (3072) input features")
        if shrinking_stride != 1 and shrinking_stride != 2:
            raise ValueError("shrinking_stride must be 1 or 2")
        self.in_features = in_features
        channels = 1 if in_features == 784 else 3
        height = 28 if in_features == 784 else 32
        image_width = height
        self.image_shape = (channels, height, image_width)
        self.window_size = 2
        self.shrinking_stride = shrinking_stride
        self.window_shapes = [self.image_shape]
        self.window_strides = []
        self.window_padding = []
        self.window_mlps = nn.ModuleList()
        stages = height - 1 if shrinking_stride == 1 else 6
        for stage in range(stages):
            stride = 1 if stage == 0 else shrinking_stride
            self.window_strides.append(stride)
            # Keep the final row/column when a stride-2 stage has odd size.
            pad_height, pad_width = height % stride, image_width % stride
            self.window_padding.append((pad_height, pad_width))
            self.window_mlps.append(RollingWindowBlock(
                channels, window_hidden, window_features, stride, pad_height))
            height = (height + pad_height - self.window_size) // stride + 1
            image_width = (image_width + pad_width - self.window_size) // stride + 1
            channels = window_features
            self.window_shapes.append((channels, height, image_width))

        # Project the flattened window features down to the backbone width.
        self.input_layer = nn.Linear(channels * height * image_width, width)
        self.blocks = nn.ModuleList([Block19(width, hidden=hidden, dropout=dropout) for _ in range(depth)])
        self.head_fc2 = nn.Linear(width, num_classes)

        # Xavier (Glorot) initialization for every linear layer, zero biases.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def extract_windows(self, x, stage=0):
        """Return (batch, positions, C*2*2), ordered by row, column, channel, dy, dx."""
        if x.ndim != 4 or tuple(x.shape[1:]) != self.window_shapes[stage]:
            raise ValueError(f"Expected NCHW feature map (N, C, H, W) with C,H,W={self.window_shapes[stage]}")
        # Tensor.unfold only creates sliding views; it performs no learned operation.
        return self.window_mlps[stage].extract_windows(x).flatten(1, 2)

    def forward(self, x):
        """Training forward pass. Returns raw logits (no softmax).

        nn.CrossEntropyLoss applies the softmax itself, so training uses logits.
        """
        if x.ndim != 4 or tuple(x.shape[1:]) != self.image_shape:
            raise ValueError(f"Expected NCHW images with C,H,W={self.image_shape}")
        for window_mlp in self.window_mlps:
            x = window_mlp(x)
        x = self.input_layer(x.flatten(1))
        for blk in self.blocks:
            x = blk(x)
        return self.head_fc2(x)

    def predict(self, x):
        """Inference pass, never used during training. Returns class probabilities."""
        self.eval()                           # switch layers to inference behavior
        with torch.no_grad():                 # no gradients needed at inference
            return torch.softmax(self.forward(x), dim=1)


def prepare(X):
    """Convert uint8 images (0-255) to floats in roughly [-1, 1]."""
    return (torch.as_tensor(X).float() / 255.0 - 0.5) / 0.5


def available_optimizer_names():
    """Offer supported presets without requiring optional optimizer packages."""
    return [name for name in OPTIMIZER_DEFAULTS if hasattr(torch.optim, name)]


def optimizer_config(name=None):
    """A fresh preset shared by the dashboard and grading entry point."""
    name = OPTIMIZER_NAME if name is None else name
    if name not in OPTIMIZER_DEFAULTS:
        raise ValueError(f"Unknown optimizer: {name}")
    if name not in available_optimizer_names():
        raise ValueError(f"{name} requires a PyTorch build providing torch.optim.{name}")
    return {"name": name, **json.loads(json.dumps(OPTIMIZER_DEFAULTS[name]))}


def build_optimizers(model, settings=None):
    """Build optimizers and record effective settings plus parameter assignments."""
    settings = {} if settings is None else json.loads(json.dumps(settings))
    config = optimizer_config(settings.get("name"))
    if set(settings) - set(config):
        raise ValueError(f"Unknown settings for {config['name']}: {sorted(set(settings) - set(config))}")
    config.update(settings)
    name = config["name"]
    kwargs = {key: value for key, value in config.items() if key != "name"}
    named = [(key, parameter) for key, parameter in model.named_parameters() if parameter.requires_grad]
    if name == "Muon":
        matrices = [parameter for key, parameter in named if parameter.ndim == 2
                    and not key.startswith(("input_layer.", "head_fc2."))]
        matrix_ids = {id(parameter) for parameter in matrices}
        other = [parameter for _, parameter in named if id(parameter) not in matrix_ids]
        if not matrices:
            raise ValueError("Muon needs at least one hidden 2D weight matrix")
        auxiliary = {key.removeprefix("auxiliary_"): kwargs.pop(key)
                     for key in list(kwargs) if key.startswith("auxiliary_")}
        auxiliary["betas"] = (auxiliary.pop("beta1"), auxiliary.pop("beta2"))
        optimizers = [torch.optim.Muon(matrices, **kwargs)]
        if other:
            optimizers.append(torch.optim.AdamW(other, **auxiliary))
    else:
        if name in ("Adam", "AdamW"):
            kwargs["betas"] = (kwargs.pop("beta1"), kwargs.pop("beta2"))
        optimizers = [getattr(torch.optim, name)([parameter for _, parameter in named], **kwargs)]
    parameter_names = {id(parameter): key for key, parameter in named}
    groups = []
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            groups.append({"optimizer": type(optimizer).__name__,
                           **{key: value for key, value in group.items() if key != "params"},
                           "parameter_names": [parameter_names[id(parameter)] for parameter in group["params"]],
                           "parameter_count": sum(parameter.numel() for parameter in group["params"])})
    # Normalize tuples so in-memory configuration agrees with saved JSON.
    return optimizers, config, json.loads(json.dumps(groups))


def augmentation_categories(task):
    """Slider ranges and distinct defaults for grayscale digits and RGB images."""
    if task not in TASKS:
        raise ValueError(f"Unknown augmentation task: {task}")
    categories = dict(AUGMENTATION_CATEGORIES)
    if task != "mnist":
        # Color datasets share transforms, but keep independent strengths.
        defaults = {
            "rotation": (0.0, 10.0), "translation": (0.5, 4.0 if task == "cifar10" else 2.0),
            "scale": (0.0, 0.1), "shear": (0.0, 5.0),
            "brightness": (0.25, 0.1 if task == "cifar10" else 0.2),
            "noise": (0.0, 0.05), "erasing": (0.0, 0.05),
        }
        categories = {name: spec[:4] + defaults[name] for name, spec in categories.items()}
        categories["horizontal_flip"] = ("Horizontal flip", "1 = enabled", 1.0, 1.0, 0.5, 1.0)
    return categories


def augmentation_config(task):
    """Dataset-tagged defaults shared by dashboard controls and training."""
    if task in AUGMENTATION_CONFIG_OVERRIDES:
        return json.loads(json.dumps(AUGMENTATION_CONFIG_OVERRIDES[task]))
    settings = {name: {"probability": spec[4], "amount": spec[5]}
                for name, spec in augmentation_categories(task).items()}
    return {
        "task": task, "enabled": AUGMENTATION_ENABLED[task], "default": settings,
        "classes": {str(digit): json.loads(json.dumps(settings)) for digit in range(10)}
                   if task == "mnist" else {},
    }


class ImageAugmenter:
    """Fresh dataset-aware transforms on normalized grayscale or RGB CPU batches.

    Uses a private RNG so augmentation does not alter data shuffling or the
    holdout split. Affine transforms share one resampling pass. Black fill is
    -1 in the model's normalized input range. Source images are never mutated.
    """

    def __init__(self, task, config=None, seed=0):
        self.task = task
        self.config = json.loads(json.dumps(augmentation_config(task) if config is None else config))
        if self.config.get("task") != task:
            raise ValueError("Augmentation configuration does not match the selected dataset")
        if not isinstance(self.config.get("enabled"), bool):
            raise ValueError("Augmentation enabled must be a boolean")
        self.num_classes = TASKS[task]["num_classes"]
        self.channels = 1 if task == "mnist" else 3
        categories = augmentation_categories(task)
        self.names = list(categories)
        overrides = self.config["classes"]
        valid_classes = {str(i) for i in range(self.num_classes)}
        if any(class_id not in valid_classes for class_id in overrides):
            raise ValueError("Augmentation override has an invalid class ID")
        values = []
        for class_id in range(self.num_classes):
            row = []
            class_settings = overrides.get(str(class_id), self.config["default"])
            for name, spec in categories.items():
                settings = class_settings[name]
                probability, amount = settings["probability"], settings["amount"]
                if not 0 <= probability <= 1 or not 0 <= amount <= spec[2]:
                    raise ValueError(f"Invalid {name} settings for class {class_id}")
                row.append([probability, amount])
            values.append(row)
        self.parameters = torch.tensor(values, dtype=torch.float32)
        self.generator = torch.Generator().manual_seed(seed)
        self.active = self.config["enabled"] and bool(
            ((self.parameters[:, :, 0] > 0) & (self.parameters[:, :, 1] > 0)).any())

    def __call__(self, images, labels):
        if not self.active:
            return images
        if images.device.type != "cpu" or images.ndim != 4 or images.shape[1] != self.channels:
            raise ValueError(f"{self.task} augmentation expects normalized NCHW CPU images with {self.channels} channels")
        if labels.ndim != 1 or len(labels) != len(images) or bool(((labels < 0) | (labels >= self.num_classes)).any()):
            raise ValueError(f"Augmentation expects one label per image, from 0 to {self.num_classes - 1}")
        count, _, height, width = images.shape
        parameters = self.parameters[labels]
        gates = torch.rand((count, len(self.names)), generator=self.generator) < parameters[:, :, 0]
        strengths = parameters[:, :, 1] * gates
        amounts = {name: strengths[:, i] for i, name in enumerate(self.names)}

        def signed(name):
            return (2 * torch.rand(count, generator=self.generator) - 1) * amounts[name]

        # Invert the sampled forward rotation/shear/scale/translation for affine_grid.
        angle = torch.deg2rad(signed("rotation"))
        shear = torch.tan(torch.deg2rad(signed("shear")))
        scale = 1 + signed("scale")
        cosine, sine = angle.cos(), angle.sin()
        a = (cosine + sine * shear) / scale
        b = (sine - cosine * shear) / scale
        c, d = -sine / scale, cosine / scale
        tx = signed("translation") * (2 / width)
        ty = signed("translation") * (2 / height)
        theta = torch.stack([a, b, -a * tx - b * ty, c, d, -c * tx - d * ty], dim=1).reshape(-1, 2, 3)
        geometric = sum(amounts[name] for name in ("rotation", "translation", "scale", "shear")) > 0
        output = images
        if bool(geometric.any()):
            grid = F.affine_grid(theta, images.shape, align_corners=False)
            warped = F.grid_sample(images + 1, grid, align_corners=False, padding_mode="zeros") - 1
            output = torch.where(geometric[:, None, None, None], warped, images)
        if "horizontal_flip" in amounts:
            flip = amounts["horizontal_flip"] > 0
            output = torch.where(flip[:, None, None, None], output.flip(-1), output)
        brightness = signed("brightness")[:, None, None, None]
        output = (output + 1) * (1 + brightness) - 1
        noise_std = torch.rand(count, generator=self.generator) * amounts["noise"]
        output = output + 2 * noise_std[:, None, None, None] * torch.randn(images.shape, generator=self.generator)
        output = output.clamp(-1, 1)

        area = torch.rand(count, generator=self.generator) * amounts["erasing"]
        side = (area.sqrt() * min(height, width)).floor().long()
        top = (torch.rand(count, generator=self.generator) * (height - side + 1)).floor().long()
        left = (torch.rand(count, generator=self.generator) * (width - side + 1)).floor().long()
        rows = torch.arange(height)[None, :, None]
        cols = torch.arange(width)[None, None, :]
        mask = ((rows >= top[:, None, None]) & (rows < (top + side)[:, None, None]) &
                (cols >= left[:, None, None]) & (cols < (left + side)[:, None, None]))
        output = output.masked_fill(mask[:, None], -1)
        # Preserve exact inputs for classes whose categories are all disabled.
        changed = (strengths > 0).any(dim=1)
        return torch.where(changed[:, None, None, None], output, images)


def model_architecture(model):
    """Describe actual layers, including Canvas models with a single-layer head.

    Preserve the historical meaning of hidden (head expansion) when present.
    Otherwise record the first block's expansion, or backbone width when the
    model has no expanded linear layer.
    """
    head = getattr(model, "head_fc1", None)
    block_linears = [layer for layer in model.blocks[0].modules() if isinstance(layer, nn.Linear)] if len(model.blocks) else []
    hidden = head.out_features if isinstance(head, nn.Linear) else (
        block_linears[0].out_features if block_linears else model.input_layer.out_features)
    architecture = {
        "width": model.input_layer.out_features,
        "hidden": hidden,
        "depth": len(model.blocks),
        "in_features": getattr(model, "in_features", model.input_layer.in_features),
        "num_classes": model.head_fc2.out_features,
        "normalization": sorted({type(layer).__name__ for layer in model.modules()
                                 if "Norm" in type(layer).__name__}),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    if hasattr(model, "window_mlps"):
        architecture.update(
            window_size=model.window_size, shrinking_stride=model.shrinking_stride,
            window_strides=model.window_strides, window_shapes=model.window_shapes,
            window_hidden=model.window_mlps[0][0].out_features,
            window_features=model.window_mlps[0][2].out_features,
            window_order="row, column, channel, dy, dx", window_padding=getattr(model, "window_padding", 0),
            window_weight_sharing="within each stage", window_aggregation="hierarchical",
        )
    return architecture


def train_with_metrics(task, X_train, y_train, seed=0, device="cuda", run_dir=None, augmentation=None,
                       optimizer_settings=None, epochs=None, model_factory=None, batch_size=None,
                       max_seconds=None, observer=None):
    """Train with a seeded holdout; optionally persist each epoch and best weights.

    No test data enters training or checkpoint selection. Tied validation
    accuracies keep the earlier epoch. The grading entry point remains train().
    """
    call_started = time.perf_counter()
    if max_seconds is not None and (not isinstance(max_seconds, (int, float))
                                    or isinstance(max_seconds, bool) or not np.isfinite(max_seconds)
                                    or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive")
    deadline = (call_started + max_seconds - min(30.0, max_seconds * 0.02)
                if max_seconds is not None else None)
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    torch.manual_seed(seed)                   # makes the run reproducible
    if len(X_train) != len(y_train) or len(X_train) < 2:
        raise ValueError("Training needs at least two examples and matching labels")
    epochs = EPOCHS[task] if epochs is None else epochs
    if (not 0 < VALIDATION_FRACTION < 1 or not isinstance(epochs, int)
            or isinstance(epochs, bool) or epochs < 1):
        raise ValueError("Use a validation fraction between 0 and 1 and at least one epoch")

    augmentation = augmentation_config(task) if augmentation is None else augmentation
    augmenter = ImageAugmenter(task, augmentation, seed=seed + 1)

    # 1. Build the model for this task. Only input size and class count differ.
    factory = ResidualMLP if model_factory is None else model_factory
    model = factory(**TASKS[task]).to(device)

    # 2. Wrap the data so we can loop over shuffled mini-batches.
    # Keep ImageNet's memory-mapped uint8 storage; normalize one batch at a time.
    streaming_normalization = task == "imagenet32"
    dataset = TensorDataset(torch.as_tensor(X_train) if streaming_normalization else prepare(X_train),
                            torch.as_tensor(y_train).long())
    split = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    validation_size = min(len(dataset) - 1, max(1, int(len(dataset) * VALIDATION_FRACTION)))
    validation_indices, training_indices = split[:validation_size], split[validation_size:]
    loader = DataLoader(Subset(dataset, training_indices.tolist()), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(Subset(dataset, validation_indices.tolist()), batch_size=batch_size)

    # 3. Choose the loss and the optimizer.
    loss_fn = nn.CrossEntropyLoss()           # expects raw logits
    optimizers, optimizer_settings, optimizer_groups = build_optimizers(model, optimizer_settings)
    functional_operations = set()
    block_cls = type(model.blocks[0]) if len(model.blocks) else None
    model_types = [type(model)]
    if block_cls is not None:
        model_types.append(block_cls)
    support_types = list(dict.fromkeys(type(layer) for layer in model.modules()
                                     if type(layer).__module__ == type(model).__module__
                                     and type(layer) not in model_types))
    model_types.extend(support_types)
    for cls in model_types:
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(cls)))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "F"):
                functional_operations.add(node.func.attr)
    metadata = {
        "task": task, "seed": seed, "device": str(device), "epochs": epochs,
        "restored_from": RESTORED_FROM if RESTORED_FROM and RESTORED_FROM["task"] == task else None,
        "batch_size": batch_size, "learning_rate": optimizer_groups[0]["lr"],
        "max_seconds": max_seconds,
        "streaming_normalization": streaming_normalization,
        "weight_decay": optimizer_groups[0]["weight_decay"],
        "momentum": optimizer_groups[0].get("momentum"),
        "dampening": optimizer_groups[0].get("dampening"),
        "betas": optimizer_groups[0].get("betas"),
        "eps": optimizer_groups[0].get("eps"),
        "optimizer": optimizer_settings["name"], "loss": repr(loss_fn),
        "optimizer_settings": optimizer_settings, "optimizer_groups": optimizer_groups,
        "learning_rate_schedule": "constant", "shuffle": True,
        "architecture": {
            **model_architecture(model),
            "functional_operations": sorted(functional_operations),
        },
        "model_source": inspect.getsource(type(model)),
        "block_source": inspect.getsource(block_cls) if block_cls else None,
        "support_sources": {cls.__name__: inspect.getsource(cls) for cls in support_types},
        "validation_fraction": VALIDATION_FRACTION,
        "training_examples": len(training_indices), "validation_examples": validation_size,
        "selection_metric": "validation_accuracy", "tie_break": "earliest_epoch",
        "model": repr(model),
        "augmentation": augmenter.config,
        "train_accuracy_kind": "training-batch predictions (augmented when enabled), before each batch update",
    }
    fields = ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "elapsed_seconds"]
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with (run_dir / "metrics.csv").open("w", newline="") as file:
            csv.DictWriter(file, fieldnames=fields).writeheader()

    # 4. The training loop.
    history = []
    best_accuracy, best_epoch, best_state = -1.0, 0, None
    best_optimizer_states = None
    if observer is not None:
        observer.optimizer_settings = optimizer_settings
        observer(model, optimizers, 0, "initial")
    started = time.perf_counter()
    stop_reason = "epochs"
    for epoch in range(epochs):
        if deadline is not None and time.perf_counter() >= deadline:
            stop_reason = "time_budget"
            break
        model.train()
        total_loss = torch.zeros((), device=device)
        correct = torch.zeros((), dtype=torch.long, device=device)
        for X_batch, y_batch in loader:
            if deadline is not None and time.perf_counter() >= deadline:
                stop_reason = "time_budget"
                break
            if streaming_normalization:
                X_batch = prepare(X_batch)
            X_batch = augmenter(X_batch, y_batch)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)           # forward pass
            loss = loss_fn(logits, y_batch)   # how wrong are we?

            for optimizer in optimizers:
                optimizer.zero_grad()             # clear old gradients
            loss.backward()                   # backward pass: compute gradients
            for optimizer in optimizers:
                optimizer.step()                  # update the weights

            total_loss += loss.detach() * len(X_batch)
            correct += (logits.detach().argmax(1) == y_batch).sum()

        if stop_reason == "time_budget":
            break

        model.eval()
        validation_loss = torch.zeros((), device=device)
        validation_correct = torch.zeros((), dtype=torch.long, device=device)
        with torch.inference_mode():
            for X_batch, y_batch in validation_loader:
                if deadline is not None and time.perf_counter() >= deadline:
                    stop_reason = "time_budget"
                    break
                if streaming_normalization:
                    X_batch = prepare(X_batch)
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                validation_loss += loss_fn(logits, y_batch) * len(X_batch)
                validation_correct += (logits.argmax(1) == y_batch).sum()
        if stop_reason == "time_budget":
            break
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss.item() / len(training_indices),
            "train_accuracy": correct.item() / len(training_indices),
            "validation_loss": validation_loss.item() / validation_size,
            "validation_accuracy": validation_correct.item() / validation_size,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        if row["validation_accuracy"] > best_accuracy:
            best_accuracy, best_epoch = row["validation_accuracy"], epoch + 1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if observer is not None:
                best_optimizer_states = copy.deepcopy([optimizer.state_dict() for optimizer in optimizers])
            if run_dir is not None:
                torch.save({"state_dict": best_state, "epoch": best_epoch,
                            "validation_accuracy": best_accuracy, "config": metadata,
                            "training_indices": training_indices, "validation_indices": validation_indices},
                           run_dir / "best.pt.partial")
                (run_dir / "best.pt.partial").replace(run_dir / "best.pt")
        if run_dir is not None:
            with (run_dir / "metrics.csv").open("a", newline="") as file:
                csv.DictWriter(file, fieldnames=fields).writerow(row)
        print(f"epoch {epoch + 1}: train loss {row['train_loss']:.4f}, "
              f"train accuracy {row['train_accuracy']:.2%}, "
              f"validation loss {row['validation_loss']:.4f}, "
              f"validation accuracy {row['validation_accuracy']:.2%} (best epoch {best_epoch})",
              flush=True)
        if observer is not None:
            observer(model, optimizers, epoch + 1, "epoch")

    # Restore weights from this run's memory, never from a previous run's saved file.
    if best_state is None:
        raise RuntimeError("Training time budget ended before one complete training/validation epoch")
    model.load_state_dict(best_state)
    model.eval()
    model.run_config = metadata
    model.training_history = history
    model.best_epoch = best_epoch
    model.best_validation_accuracy = best_accuracy
    model.training_indices = training_indices
    model.validation_indices = validation_indices
    model.stop_reason = stop_reason
    model.training_seconds = time.perf_counter() - call_started
    if observer is not None:
        for optimizer, state in zip(optimizers, best_optimizer_states):
            optimizer.load_state_dict(state)
        observer(model, optimizers, best_epoch, "selected_checkpoint")
    if run_dir is not None:
        (run_dir / "training-receipt.json").write_text(json.dumps({
            "task": task, "seed": seed, "stop_reason": stop_reason,
            "epochs_completed": len(history), "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
            "training_seconds": model.training_seconds, "max_seconds": max_seconds,
        }, indent=2) + "\n")
    return model


def predict(model, X):
    """Return class probabilities for X, one row per example, on the CPU."""
    device = next(model.parameters()).device
    X = prepare(X)
    # Predict in chunks so a large test set does not run out of GPU memory.
    chunks = [model.predict(X[i:i + 1024].to(device)).cpu() for i in range(0, len(X), 1024)]
    return torch.cat(chunks)
