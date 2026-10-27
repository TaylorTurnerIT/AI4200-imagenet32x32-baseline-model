"""Compatibility imports for the model defined in the consolidated notebook."""

import ast
import inspect
import textwrap

from . import notebook

TASKS = notebook.TASKS
ResidualMLP = notebook.ResidualMLP

# Torch Canvas can rename a block class. Keep this compatibility alias tied to
# the class the current model actually constructs.
constructors = [node.value for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(ResidualMLP))))
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute) and target.attr == "blocks"
                                                       for target in node.targets)]
for node in (child for constructor in constructors for child in ast.walk(constructor)):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        candidate = vars(notebook).get(node.func.id)
        if isinstance(candidate, type) and issubclass(candidate, notebook.nn.Module):
            Block = candidate
            break
else:
    raise ImportError("The model does not reference a residual block class")

__all__ = ["TASKS", "Block", "ResidualMLP"]
