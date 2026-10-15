# Show available commands.
default:
    @just --list

# Install the locked Python environment.
setup:
    uv sync --locked

# Open the marimo app in the browser; watch for edits from another editor.
app *args:
    ./run --watch {{args}}

# Alias for the marimo app.
alias marimo := app

# Open the network architecture in Zed (or another macOS editor).
model editor="Zed":
    open -a {{quote(editor)}} src/residual_mlp_baseline/model.py

# Open training and prediction in Zed (or another macOS editor).
train editor="Zed":
    open -a {{quote(editor)}} src/residual_mlp_baseline/train.py

# Open the marimo app's Python source in Zed (or another macOS editor).
app-source editor="Zed":
    open -a {{quote(editor)}} app.py
