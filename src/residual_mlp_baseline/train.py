"""Compatibility imports for training defined in the consolidated notebook."""

from .notebook import (
    BATCH_SIZE,
    DAMPENING,
    EPOCHS,
    LEARNING_RATE,
    MOMENTUM,
    WEIGHT_DECAY,
    predict,
    prepare,
    train,
)

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
