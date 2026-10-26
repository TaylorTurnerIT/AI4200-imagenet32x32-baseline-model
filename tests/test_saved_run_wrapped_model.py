"""Saved-run loading must recover models wrapped in marimo dependency cells."""

import ast
import inspect
import unittest

import torch

from residual_mlp_baseline import notebook


class WrappedModelRestoreTests(unittest.TestCase):
    def test_loading_replaces_broken_cell_and_keeps_one_saved_block_class(self):
        source = inspect.getsource(notebook)
        model = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.ClassDef) and node.name == "ResidualMLP")
        broken = """@app.cell
def _(hidden):
    class ResidualMLP(nn.Module):
        def __init__(self, in_features, num_classes, width=8, hidden1=16):
            super().__init__()
            self.blocks = nn.ModuleList([Block24(width, hidden) for _ in range(6)])
    return (ResidualMLP,)"""
        source = notebook.replace_source_span(source, model, broken, include_decorators=True)
        config = {
            "task": "cifar10", "seed": 42, "device": "cpu", "epochs": 1,
            "batch_size": 8, "validation_fraction": 0.25,
            "optimizer_settings": notebook.optimizer_config("Adam"),
            "augmentation": notebook.augmentation_config("cifar10"),
            "architecture": {"width": 8, "hidden": 16, "depth": 3},
            "model_source": """class ResidualMLP(nn.Module):
    def __init__(self, in_features, num_classes, width=8, hidden1=16, depth=3):
        super().__init__()
        self.input_layer = nn.Linear(in_features, width)
        self.blocks = nn.ModuleList([SavedBlock(width, hidden1) for _ in range(6)])
        self.head_fc2 = nn.Linear(width, num_classes)
    def forward(self, x):
        x = self.input_layer(x.flatten(1))
        for block in self.blocks:
            x = block(x)
        return self.head_fc2(x)
""",
            "block_source": """class SavedBlock(nn.Module):
    def __init__(self, width=8, hidden=16):
        super().__init__()
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        return self.dropout(self.down(F.gelu(self.up(x))))
""",
        }
        entry = {"id": "wrapped-model-regression", "config": config}
        restored = notebook.restore_run_source(source, entry)
        self.assertNotIn("def _(hidden)", restored)
        classes = [node for node in ast.parse(restored).body if isinstance(node, ast.ClassDef)]
        model_classes = [node for node in classes if node.name != "ImageAugmenter"]
        self.assertEqual({node.name for node in model_classes}, {"ResidualMLP", "SavedBlock"})
        namespace = {"nn": torch.nn, "F": torch.nn.functional}
        for node in model_classes:
            node.decorator_list = []
        exec(compile(ast.Module(body=model_classes, type_ignores=[]), "<restored-model>", "exec"), namespace)
        model = namespace["ResidualMLP"](**notebook.TASKS["cifar10"])
        self.assertEqual(len(model.blocks), 3)
        self.assertEqual(model.blocks[0].up.out_features, 16)
        self.assertEqual(model.blocks[0].dropout.p, 0.1)
        self.assertEqual(model(torch.randn(2, 3, 32, 32)).shape, (2, 10))
        self.assertEqual(notebook.restore_run_source(restored, entry), restored)


if __name__ == "__main__":
    unittest.main()
