"""Importable train module and editable marimo notebook."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from residual_mlp_baseline.model import TASKS, ResidualMLP


    # Baseline settings. Deliberately plain so they are easy to beat:
    # plain SGD, constant learning rate, no dropout, no schedule, no augmentation.
    LEARNING_RATE = 0.15
    BATCH_SIZE = 256
    EPOCHS = {"mnist": 50, "cifar10": 30, "imagenet32": 10}


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # Training and prediction

    Edit the settings in the setup cell and the functions below. Keep the
    `train(task, X_train, y_train, seed=0, device="cuda")` and `predict(model, X)`
    interfaces unchanged for grading.

    Save changes, then open `app.py` from the notebook list and click
    **Train and evaluate**. Opening this notebook does not start training.
    """)
    return


@app.function
def prepare(X):
    """Convert uint8 images (0-255) to floats in roughly [-1, 1]."""
    return (torch.as_tensor(X).float() / 255.0 - 0.5) / 0.5


@app.function
def train(task, X_train, y_train, seed=0, device="cuda"):
    """Build a new model and train it from scratch. Returns the trained model."""
    torch.manual_seed(seed)                   # makes the run reproducible

    # 1. Build the model for this task. Only input size and class count differ.
    model = ResidualMLP(**TASKS[task]).to(device)

    # 2. Wrap the data so we can loop over shuffled mini-batches.
    dataset = TensorDataset(prepare(X_train), torch.as_tensor(y_train).long())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Choose the loss and the optimizer.
    loss_fn = nn.CrossEntropyLoss()           # expects raw logits
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # 4. The training loop.
    model.train()
    for epoch in range(EPOCHS[task]):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)           # forward pass
            loss = loss_fn(logits, y_batch)   # how wrong are we?

            optimizer.zero_grad()             # clear old gradients
            loss.backward()                   # backward pass: compute gradients
            optimizer.step()                  # update the weights

            total_loss += loss.item() * len(X_batch)

        print(f"epoch {epoch + 1}: train loss {total_loss / len(dataset):.4f}")

    return model


@app.function
def predict(model, X):
    """Return class probabilities for X, one row per example, on the CPU."""
    device = next(model.parameters()).device
    X = prepare(X)
    # Predict in chunks so a large test set does not run out of GPU memory.
    chunks = [model.predict(X[i:i + 1024].to(device)).cpu() for i in range(0, len(X), 1024)]
    return torch.cat(chunks)


if __name__ == "__main__":
    app.run()
