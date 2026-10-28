"""Local-only MLP-Mixer training. Paper Section 3 and Appendix B hyperparameters."""

import csv
import inspect
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .mixer import MlpMixer, mixer_config


@dataclass(frozen=True)
class MixerRecipe:
    # Table 4, ImageNet B/L. Weight decay is decoupled (PyTorch AdamW).
    epochs: int = 300
    batch_size: int = 4096
    learning_rate: float = 0.001
    weight_decay: float = 0.1
    warmup_steps: int = 10_000
    gradient_clip: float = 1.0
    randaugment_magnitude: int = 15
    randaugment_layers: int = 2
    mixup_alpha: float = 0.5
    dropout: float = 0.0
    stochastic_depth: float = 0.1
    horizontal_flip: bool = True
    validation_fraction: float = 0.1
    # Fine-tuning uses momentum SGD, no weight decay, cosine decay.
    phase: str = "pretrain"
    momentum: float = 0.9

    @classmethod
    def finetuning(cls, **overrides):
        """Section 3/Appendix A.2: SGD, no weight decay, mixup or RandAugment."""
        return cls(**(dict(phase="finetune", batch_size=512, weight_decay=0.0,
                           mixup_alpha=0.0, randaugment_layers=0, dropout=0.0,
                           stochastic_depth=0.0) | overrides))

    def __post_init__(self):
        for name in ("epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps", "randaugment_layers", "randaugment_magnitude"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        values = (self.learning_rate, self.weight_decay, self.gradient_clip, self.mixup_alpha,
                  self.validation_fraction, self.dropout, self.stochastic_depth, self.momentum)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Recipe values must be finite")
        if (self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0
                or self.mixup_alpha < 0 or not 0 < self.validation_fraction < 1
                or not 0 <= self.dropout < 1 or not 0 <= self.stochastic_depth < 1
                or self.randaugment_magnitude > 30 or not 0 <= self.momentum < 1):
            raise ValueError("Invalid Mixer training settings")
        if self.phase not in ("pretrain", "finetune"):
            raise ValueError("phase must be pretrain or finetune")
        if self.phase == "finetune" and self.weight_decay != 0:
            raise ValueError("Paper fine-tuning uses zero weight decay")


def learning_rate_factor(step, total_steps, warmup_steps, phase="pretrain"):
    """Zero-based optimizer update: linear warmup, then linear or cosine decay."""
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return (1 + math.cos(math.pi * progress)) / 2 if phase == "finetune" else 1 - progress


class ImageTransform:
    """Inception-style crop/flip, two-op RandAugment, then [-1, 1] scaling.

    Uses a standard RandAugment operation pool via Pillow and a 0..30 magnitude
    scale. The paper specifies N=2 and M, but not interpolation, fill, or the
    exact operation pool; those implementation choices are recorded below.
    """

    operations = ("identity", "autocontrast", "equalize", "rotate", "solarize", "posterize",
                  "color", "contrast", "brightness", "sharpness", "shear_x", "shear_y",
                  "translate_x", "translate_y")

    def __init__(self, image_size, recipe, seed, training):
        self.size, self.recipe, self.training = image_size, recipe, training
        self.rng = np.random.default_rng(seed)

    def __call__(self, x):
        array = np.asarray(x)
        array = np.moveaxis(array, 0, -1)
        if array.shape[-1] == 1:
            array = array[..., 0]
        image = Image.fromarray(array)
        r = self.recipe
        if self.training:
            width, height = image.size
            # Fine-tuning uses a fixed resize and random crop, not Inception crop.
            if r.phase == "finetune":
                side = round(self.size * 256 / 224)
                image = image.resize((side, side), Image.Resampling.BILINEAR)
                width, height = image.size
                left = int(self.rng.integers(width-self.size+1))
                top = int(self.rng.integers(height-self.size+1))
                box = (left, top, left+self.size, top+self.size)
            else:
                box = None
            # Inception crop: random area and aspect ratio, then center fallback.
            for _ in range(0 if box is not None else 10):
                area = width * height * self.rng.uniform(0.08, 1.0)
                ratio = math.exp(self.rng.uniform(math.log(3/4), math.log(4/3)))
                w, h = round(math.sqrt(area * ratio)), round(math.sqrt(area / ratio))
                if 0 < w <= width and 0 < h <= height:
                    left = int(self.rng.integers(width - w + 1))
                    top = int(self.rng.integers(height - h + 1))
                    box = (left, top, left + w, top + h)
                    break
            if box is None:
                side = min(width, height)
                left, top = (width-side)//2, (height-side)//2
                box = (left, top, left+side, top+side)
            image = image.crop(box).resize((self.size, self.size), Image.Resampling.BILINEAR)
            if r.horizontal_flip and self.rng.random() < 0.5:
                image = ImageOps.mirror(image)
            for _ in range(r.randaugment_layers):
                image = self.augment(image, str(self.rng.choice(self.operations)))
        elif image.size != (self.size, self.size):
            # Appendix C resize-crop variant (i), scaled for the target resolution.
            side = round(self.size * 256 / 224)
            image = image.resize((side, side), Image.Resampling.BILINEAR)
            start = (side - self.size) // 2
            image = image.crop((start, start, start+self.size, start+self.size))
        array = np.array(image, dtype=np.float32, copy=True)
        if array.ndim == 2:
            array = array[..., None]
        return torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1)

    def augment(self, image, operation):
        magnitude = self.recipe.randaugment_magnitude / 30
        signed = magnitude * (-1 if self.rng.random() < 0.5 else 1)
        if operation == "identity":
            return image
        if operation == "autocontrast":
            return ImageOps.autocontrast(image)
        if operation == "equalize":
            return ImageOps.equalize(image)
        if operation == "solarize":
            return ImageOps.solarize(image, int(255 * (1-magnitude)))
        if operation == "posterize":
            return ImageOps.posterize(image, max(1, 8-round(4*magnitude)))
        if operation == "rotate":
            return image.rotate(30*signed, resample=Image.Resampling.BILINEAR)
        enhancements = {"color": ImageEnhance.Color, "contrast": ImageEnhance.Contrast,
                        "brightness": ImageEnhance.Brightness, "sharpness": ImageEnhance.Sharpness}
        if operation in enhancements:
            return enhancements[operation](image).enhance(1 + 0.9*signed)
        coefficients = {
            "shear_x": (1, 0.3*signed, 0, 0, 1, 0),
            "shear_y": (1, 0, 0, 0.3*signed, 1, 0),
            "translate_x": (1, 0, signed*image.width*150/331, 0, 1, 0),
            "translate_y": (1, 0, 0, 0, 1, signed*image.height*150/331),
        }
        return image.transform(image.size, Image.Transform.AFFINE, coefficients[operation],
                               resample=Image.Resampling.BILINEAR, fillcolor=0)


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transform(self.images[index]), int(self.labels[index])


def task_config(task, preset="S/16", **overrides):
    """Use S widths with 4-pixel patches for local 28/32-pixel datasets."""
    if task not in ("mnist", "cifar10", "imagenet32"):
        raise ValueError(f"Unknown task: {task}")
    defaults = dict(image_size=28 if task == "mnist" else 32, patch_size=4,
                    in_channels=1 if task == "mnist" else 3,
                    num_classes=1000 if task == "imagenet32" else 10)
    return mixer_config(preset, **(defaults | overrides))


def train_mixer(task, images, labels, seed=0, device="cuda", run_dir=None,
                config=None, recipe=None, model=None, restored_from=None):
    """Train locally with a deterministic holdout; never contact a leaderboard.

    Supply a pre-trained model and a finetune recipe for transfer learning.
    Without a model, initialize from scratch. No test data selects weights.
    """
    recipe = recipe or MixerRecipe(batch_size=256, horizontal_flip=task != "mnist")
    config = config or (model.config if model is not None else task_config(task))
    config = replace(config, dropout=recipe.dropout, stochastic_depth=recipe.stochastic_depth)
    shape = tuple(images.shape) if hasattr(images, "shape") else np.asarray(images).shape
    if len(shape) != 4 or shape[1] != config.in_channels:
        raise ValueError("Training images must be NCHW with matching channels")
    if np.asarray(images).dtype != np.uint8:
        raise ValueError("Training images must be uint8 in [0, 255]")
    labels = torch.as_tensor(labels).long()
    if len(images) != len(labels) or len(labels) < 2 or labels.ndim != 1:
        raise ValueError("Training needs at least two examples and matching labels")
    if labels.min() < 0 or labels.max() >= config.num_classes:
        raise ValueError("Labels must be class indices in the configured range")
    if model is not None:
        expected = replace(model.config, dropout=recipe.dropout, stochastic_depth=recipe.stochastic_depth)
        if expected != config:
            raise ValueError("Supplied model architecture must match config; reset its head or resize first")
        # Fine-tuning can remove pre-training regularization without changing weights.
        model.config = config
        for i, block in enumerate(model.blocks):
            block.token_drop.probability = recipe.stochastic_depth*(2*i)/(2*config.num_blocks-1)
            block.channel_drop.probability = recipe.stochastic_depth*(2*i+1)/(2*config.num_blocks-1)
            for module in block.modules():
                if isinstance(module, nn.Dropout):
                    module.p = recipe.dropout
    torch.manual_seed(seed)
    model = (MlpMixer(config) if model is None else model).to(device)
    split = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))
    validation_size = min(len(labels)-1, max(1, int(len(labels)*recipe.validation_fraction)))
    validation_indices, training_indices = split[:validation_size], split[validation_size:]
    train_data = ImageDataset(images, labels, ImageTransform(config.image_size, recipe, seed+1, True))
    validation_data = ImageDataset(images, labels, ImageTransform(config.image_size, recipe, seed+2, False))
    loader = DataLoader(Subset(train_data, training_indices.tolist()), batch_size=recipe.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed+3))
    validation_loader = DataLoader(Subset(validation_data, validation_indices.tolist()), batch_size=recipe.batch_size)
    if recipe.phase == "finetune":
        optimizer = torch.optim.SGD(model.parameters(), lr=recipe.learning_rate, momentum=recipe.momentum)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate,
                                      betas=(0.9, 0.999), weight_decay=recipe.weight_decay)
    total_steps = len(loader)*recipe.epochs
    # Short local runs cannot fit the paper's 10k warmup. Record the adaptation.
    warmup = min(recipe.warmup_steps, max(0, total_steps-1))
    rng = np.random.default_rng(seed+4)
    mixup_generator = torch.Generator().manual_seed(seed+5)
    optimizer_settings = dict(name=type(optimizer).__name__, lr=recipe.learning_rate,
                              weight_decay=recipe.weight_decay)
    metadata = dict(task=task, restored_from=restored_from, seed=seed, device=str(device), epochs=recipe.epochs,
                    batch_size=recipe.batch_size, learning_rate=recipe.learning_rate,
                    weight_decay=recipe.weight_decay, momentum=recipe.momentum if recipe.phase == "finetune" else None,
                    dampening=0, optimizer=type(optimizer).__name__, optimizer_settings=optimizer_settings,
                    betas=[0.9, 0.999] if recipe.phase == "pretrain" else None,
                    loss="CrossEntropyLoss (mixup soft targets)",
                    learning_rate_schedule="linear_warmup_" + ("cosine_decay" if recipe.phase == "finetune" else "linear_decay"),
                    warmup_steps=warmup, total_steps=total_steps, shuffle=True,
                    validation_fraction=recipe.validation_fraction, training_examples=len(training_indices),
                    validation_examples=validation_size, selection_metric="validation_accuracy", tie_break="earliest_epoch",
                    architecture=model.architecture(), mixer_config=asdict(config), mixer_recipe=asdict(recipe),
                    model=repr(model), model_source=inspect.getsource(MlpMixer),
                    implementation_source=Path(__file__).with_name("mixer.py").read_text(),
                    training_source=Path(__file__).read_text(),
                    augmentation=dict(enabled=True, task=task, random_resized_crop=recipe.phase == "pretrain",
                                      crop="random_resized" if recipe.phase == "pretrain" else "resize_random_crop",
                                      horizontal_flip=recipe.horizontal_flip, randaugment_layers=recipe.randaugment_layers,
                                      randaugment_magnitude=recipe.randaugment_magnitude, mixup_alpha=recipe.mixup_alpha,
                                      interpolation="Pillow bilinear", fill=0, magnitude_scale=30,
                                      randaugment_operations=list(ImageTransform.operations)),
                    leaderboard_enabled=False, paper="arXiv:2105.01601v4",
                    train_accuracy_kind="mixup-weighted accuracy on augmented training batches")
    fields = ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "elapsed_seconds", "learning_rate"]
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.json").write_text(json.dumps(metadata, indent=2)+"\n")
        with (run_dir / "metrics.csv").open("w", newline="") as file:
            csv.DictWriter(file, fields).writeheader()
    history, best_state, best_accuracy, best_epoch, step = [], None, -1.0, 0, 0
    started = time.perf_counter()
    for epoch in range(recipe.epochs):
        model.train()
        total_loss, correct = 0.0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            lam = float(rng.beta(recipe.mixup_alpha, recipe.mixup_alpha)) if recipe.mixup_alpha else 1.0
            order = torch.randperm(len(y), generator=mixup_generator).to(device)
            x = lam*x + (1-lam)*x[order]
            targets = lam*nn.functional.one_hot(y, config.num_classes) + (1-lam)*nn.functional.one_hot(y[order], config.num_classes)
            step += 1
            rate = recipe.learning_rate*learning_rate_factor(step-1, total_steps, warmup, recipe.phase)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.detach().item()*len(y)
            prediction = logits.detach().argmax(1)
            correct += (lam*(prediction == y).sum() + (1-lam)*(prediction == y[order]).sum()).item()
        model.eval()
        validation_loss, validation_correct = 0.0, 0
        with torch.inference_mode():
            for x, y in validation_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                validation_loss += nn.functional.cross_entropy(logits, y).item()*len(y)
                validation_correct += (logits.argmax(1) == y).sum().item()
        row = dict(epoch=epoch+1, train_loss=total_loss/len(training_indices), train_accuracy=correct/len(training_indices),
                   validation_loss=validation_loss/validation_size, validation_accuracy=validation_correct/validation_size,
                   elapsed_seconds=time.perf_counter()-started, learning_rate=rate)
        if not all(math.isfinite(value) for value in row.values()):
            raise FloatingPointError("Non-finite training metrics")
        history.append(row)
        if row["validation_accuracy"] > best_accuracy:
            best_accuracy, best_epoch = row["validation_accuracy"], epoch+1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if run_dir is not None:
                torch.save(dict(state_dict=best_state, epoch=best_epoch, validation_accuracy=best_accuracy, config=metadata,
                                training_indices=training_indices, validation_indices=validation_indices), run_dir/"best.pt.partial")
                (run_dir/"best.pt.partial").replace(run_dir/"best.pt")
        if run_dir is not None:
            with (run_dir/"metrics.csv").open("a", newline="") as file:
                csv.DictWriter(file, fields).writerow(row)
        print(f"epoch {epoch+1}: train loss {row['train_loss']:.4f}, "
              f"validation accuracy {row['validation_accuracy']:.2%} (best epoch {best_epoch})", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    model.run_config, model.training_history = metadata, history
    model.best_epoch, model.best_validation_accuracy = best_epoch, best_accuracy
    model.training_indices, model.validation_indices = training_indices, validation_indices
    return model


@torch.no_grad()
def predict_mixer(model, images, chunk=256):
    """Normalize raw uint8 NCHW images and return CPU class probabilities."""
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if len(images) == 0:
        return torch.empty((0, model.config.num_classes))
    device = next(model.parameters()).device
    transform = ImageTransform(model.config.image_size, MixerRecipe(), 0, False)
    model.eval()
    outputs = []
    for start in range(0, len(images), chunk):
        batch = torch.stack([transform(image) for image in images[start:start+chunk]])
        outputs.append(model(batch.to(device=device, dtype=next(model.parameters()).dtype)).softmax(-1).cpu())
    return torch.cat(outputs)
