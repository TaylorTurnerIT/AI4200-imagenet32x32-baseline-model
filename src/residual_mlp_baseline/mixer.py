"""PyTorch implementation of MLP-Mixer, arXiv:2105.01601v4, Section 2/Appendix E."""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace

import torch
from torch import nn


@dataclass(frozen=True)
class MixerConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000
    num_blocks: int = 8
    hidden_dim: int = 512
    tokens_mlp_dim: int = 256
    channels_mlp_dim: int = 2048
    dropout: float = 0.0
    stochastic_depth: float = 0.0
    # Original grid size for Appendix C's spatially grouped token ordering.
    token_group_grid: int | None = None

    def __post_init__(self):
        for name in ("image_size", "patch_size", "in_channels", "num_classes", "num_blocks",
                     "hidden_dim", "tokens_mlp_dim", "channels_mlp_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if not 0 <= self.dropout < 1 or not 0 <= self.stochastic_depth < 1:
            raise ValueError("dropout and stochastic_depth must be in [0, 1)")
        if self.token_group_grid is not None:
            if (isinstance(self.token_group_grid, bool) or not isinstance(self.token_group_grid, int)
                    or self.token_group_grid < 1 or self.grid_size % self.token_group_grid):
                raise ValueError("token_group_grid must divide the patch grid")

    @property
    def grid_size(self):
        return self.image_size // self.patch_size

    @property
    def num_tokens(self):
        return self.grid_size ** 2


# Table 1. Patch sizes and widths are literal paper settings at resolution 224.
MIXER_PRESETS = {
    "S/32": dict(num_blocks=8, patch_size=32, hidden_dim=512, tokens_mlp_dim=256, channels_mlp_dim=2048),
    "S/16": dict(num_blocks=8, patch_size=16, hidden_dim=512, tokens_mlp_dim=256, channels_mlp_dim=2048),
    "B/32": dict(num_blocks=12, patch_size=32, hidden_dim=768, tokens_mlp_dim=384, channels_mlp_dim=3072),
    "B/16": dict(num_blocks=12, patch_size=16, hidden_dim=768, tokens_mlp_dim=384, channels_mlp_dim=3072),
    "L/32": dict(num_blocks=24, patch_size=32, hidden_dim=1024, tokens_mlp_dim=512, channels_mlp_dim=4096),
    "L/16": dict(num_blocks=24, patch_size=16, hidden_dim=1024, tokens_mlp_dim=512, channels_mlp_dim=4096),
    "H/14": dict(num_blocks=32, patch_size=14, hidden_dim=1280, tokens_mlp_dim=640, channels_mlp_dim=5120),
}


def mixer_config(preset="S/16", **overrides):
    """Build a Table 1 configuration, allowing explicit dataset adaptations."""
    if preset not in MIXER_PRESETS:
        raise ValueError(f"Unknown Mixer preset: {preset}")
    return MixerConfig(**(MIXER_PRESETS[preset] | overrides))


class DropPath(nn.Module):
    """Drop a residual MLP branch independently for each example."""

    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class MlpBlock(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        # Flax nn.gelu uses the tanh approximation by default.
        self.gelu = nn.GELU(approximate="tanh")
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.fc2(self.dropout1(self.gelu(self.fc1(x)))))


class MixerBlock(nn.Module):
    def __init__(self, config, token_drop=0.0, channel_drop=0.0):
        super().__init__()
        # Flax LayerNorm default epsilon is 1e-6, over channels only.
        self.token_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.token_mixing = MlpBlock(config.num_tokens, config.tokens_mlp_dim, config.dropout)
        self.channel_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.channel_mixing = MlpBlock(config.hidden_dim, config.channels_mlp_dim, config.dropout)
        self.token_drop = DropPath(token_drop)
        self.channel_drop = DropPath(channel_drop)

    def forward(self, x):
        y = self.token_mixing(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.token_drop(y)
        return x + self.channel_drop(self.channel_mixing(self.channel_norm(x)))


class MlpMixer(nn.Module):
    """Non-overlapping linear patches, Mixer blocks, channel norm, mean, classifier.

    Input is NCHW. The shared Linear stem is equivalent to Appendix E's Conv
    with kernel=stride=patch_size; it introduces no overlapping convolution.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is not None and kwargs:
            raise ValueError("Pass either MixerConfig or keyword settings")
        self.config = config if config is not None else MixerConfig(**kwargs)
        c = self.config
        self.stem = nn.Linear(c.in_channels * c.patch_size ** 2, c.hidden_dim)
        # Appendix B ramps across all 2*num_blocks residual MLP branches.
        rates = [c.stochastic_depth*i/(2*c.num_blocks-1) for i in range(2*c.num_blocks)]
        self.blocks = nn.ModuleList([MixerBlock(c, rates[2*i], rates[2*i+1])
                                     for i in range(c.num_blocks)])
        self.pre_head_layer_norm = nn.LayerNorm(c.hidden_dim, eps=1e-6)
        self.head = nn.Linear(c.hidden_dim, c.num_classes)
        self.apply(self._initialize)
        # Appendix E explicitly zero-initializes the classifier kernel.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            # Flax Dense/Conv defaults: LeCun truncated normal, variance 1/fan_in.
            std = module.in_features ** -0.5 / 0.87962566103423978
            nn.init.trunc_normal_(module.weight, std=std, a=-2*std, b=2*std)
            nn.init.zeros_(module.bias)

    def patch_embeddings(self, x):
        c = self.config
        expected = (c.in_channels, c.image_size, c.image_size)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected NCHW images with shape (N, {expected}), got {tuple(x.shape)}")
        # Match the reference HWIO patch order: patch-row, patch-column, channel.
        patches = x.unfold(2, c.patch_size, c.patch_size).unfold(3, c.patch_size, c.patch_size)
        patches = patches.permute(0, 2, 3, 4, 5, 1).reshape(x.shape[0], c.num_tokens, -1)
        x = self.stem(patches)
        if c.token_group_grid is not None:
            g = c.token_group_grid
            k = c.grid_size // g
            x = x.reshape(x.shape[0], k, g, k, g, c.hidden_dim)
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(x.shape[0], c.num_tokens, c.hidden_dim)
        return x

    def forward_features(self, x):
        x = self.patch_embeddings(x)
        for block in self.blocks:
            x = block(x)
        return self.pre_head_layer_norm(x).mean(dim=1)

    def forward(self, x):
        return self.head(self.forward_features(x))

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return self(x).softmax(dim=-1)

    def architecture(self):
        return {"family": "MLP-Mixer", **asdict(self.config),
                "num_tokens": self.config.num_tokens,
                "parameters": sum(p.numel() for p in self.parameters())}

    def reset_classifier(self, num_classes):
        """Replace the classifier with a zero-initialized downstream task head."""
        config = replace(self.config, num_classes=num_classes)
        head = nn.Linear(config.hidden_dim, config.num_classes).to(
            device=self.head.weight.device, dtype=self.head.weight.dtype)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        head.train(self.training)
        self.head, self.config = head, config

    def resize_for_finetuning(self, image_size):
        """Appendix C: clone to K-times resolution with block-diagonal token MLPs.

        Tokens are ordered by spatial tile, then within tile. This makes each
        block process its original-size region independently at initialization.
        Call on an unresized model before constructing a fine-tuning optimizer.
        """
        c = self.config
        if c.token_group_grid is not None:
            raise ValueError("Resize from the original pre-trained model")
        if (isinstance(image_size, bool) or not isinstance(image_size, int)
                or image_size < c.image_size or image_size % c.image_size):
            raise ValueError("Resolution must be an integer multiple of the original")
        if image_size == c.image_size:
            return deepcopy(self)
        copies = (image_size // c.image_size) ** 2
        new_config = replace(c, image_size=image_size, tokens_mlp_dim=c.tokens_mlp_dim * copies,
                             token_group_grid=c.grid_size)
        device, dtype = self.stem.weight.device, self.stem.weight.dtype
        resized = MlpMixer(new_config).to(device=device, dtype=dtype)
        state = deepcopy(self.state_dict())
        for i in range(c.num_blocks):
            prefix = f"blocks.{i}.token_mixing."
            for layer in ("fc1", "fc2"):
                state[prefix + layer + ".weight"] = torch.block_diag(
                    *[state[prefix + layer + ".weight"]] * copies)
                state[prefix + layer + ".bias"] = state[prefix + layer + ".bias"].repeat(copies)
        resized.load_state_dict(state)
        resized.train(self.training)
        return resized
