# Show available commands.
default:
    @just --list

# Install the locked Python environment.
setup:
    uv sync --locked

# Open the three dataset notebooks; watch for external edits.
app *args:
    ./run {{args}}

# Alias for the marimo app.
alias marimo := app

# Open the network architecture in Zed (or another macOS editor).
model task="cifar10" editor="Zed":
    open -a {{quote(editor)}} src/residual_mlp_baseline/notebooks/{{task}}.py

# Open training and prediction in Zed (or another macOS editor).
train task="cifar10" editor="Zed":
    open -a {{quote(editor)}} src/residual_mlp_baseline/notebooks/{{task}}.py

# Open the marimo app's Python source in Zed (or another macOS editor).
app-source task="cifar10" editor="Zed":
    open -a {{quote(editor)}} src/residual_mlp_baseline/notebooks/{{task}}.py
