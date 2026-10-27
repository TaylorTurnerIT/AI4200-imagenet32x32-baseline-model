"""Compatibility imports for training defined in the consolidated notebook."""

from .training_core import (
    BATCH_SIZE,
    DAMPENING,
    EPOCHS,
    LEARNING_RATE,
    MOMENTUM,
    WEIGHT_DECAY,
    prepare,
)
from .submission import predict, train

__all__ = [
    "BATCH_SIZE",
    "DAMPENING",
    "EPOCHS",
    "LEARNING_RATE",
    "MOMENTUM",
    "WEIGHT_DECAY",
    "predict",
    "prepare",
    "train",
]
