"""Class thumbnails, selection, and cache refresh for every dataset."""

import ast
import base64
import io
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from residual_mlp_baseline import notebook


def cell_with_arguments(arguments):
    nodes = ast.parse(Path(notebook.__file__).read_text()).body
    cell = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == '_'
                and [argument.arg for argument in node.args.args] == arguments)
    cell.decorator_list = []
    namespace = dict(vars(notebook))
    exec(compile(ast.Module(body=[cell], type_ignores=[]), notebook.__file__, 'exec'), namespace)
    return namespace, namespace['_']


class ClassPreviewTests(unittest.TestCase):
    def test_each_class_has_the_correct_thumbnail_or_a_missing_cache_label(self):
        for task, spec in notebook.TASKS.items():
            with self.subTest(task=task):
                shape = (1, 28, 28) if task == 'mnist' else (3, 32, 32)
                images = np.random.default_rng(42).integers(0, 256, (3, *shape), dtype=np.uint8)
                original = images.copy()
                last = spec['num_classes'] - 1
                rows = notebook.class_preview_rows(task, images, np.array([last, 0, last]))
                self.assertEqual(len(rows), spec['num_classes'])
                self.assertEqual(rows[1]['Preview'], 'Not cached')
                self.assertEqual(rows[-1]['class_id'], last)
                expected_names = {'mnist': '0', 'cifar10': 'airplane', 'imagenet32': 'class 0'}
                self.assertEqual(rows[0]['Class'], expected_names[task])
                for class_id, example in [(0, 1), (last, 0)]:
                    html = rows[class_id]['Preview'].text
                    payload = re.search(r"data:image/png;base64,([^']+)", html).group(1)
                    pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(payload))))
                    expected = images[example, 0] if task == 'mnist' else images[example].transpose(1, 2, 0)
                    self.assertTrue(np.array_equal(pixels, expected))
                self.assertTrue(np.array_equal(images, original))

    def test_selected_row_drives_the_correct_class_and_color_preview(self):
        create_namespace, create = cell_with_arguments(['DATA_DIR', 'preview_refresh', 'task'])
        render_namespace, render = cell_with_arguments([
            'augmentation_controls', 'preview_class', 'preview_images', 'preview_labels', 'preview_seed', 'task'])
        seen = []

        class RecordInputs:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, images, labels):
                seen.append(labels.tolist())
                return images

        render_namespace['ImageAugmenter'] = RecordInputs
        configs = {task: notebook.augmentation_config(task) for task in notebook.TASKS}
        with tempfile.TemporaryDirectory() as folder:
            for task, spec in notebook.TASKS.items():
                with self.subTest(task=task):
                    cache = Path(folder) / task
                    cache.mkdir()
                    shape = (1, 28, 28) if task == 'mnist' else (3, 32, 32)
                    images = np.random.default_rng(42).integers(0, 256, (4, *shape), dtype=np.uint8)
                    last = spec['num_classes'] - 1
                    np.save(cache / 'X_train.npy', images)
                    np.save(cache / 'y_train.npy', np.array([0, last, 0, last]))
                    selected_task = SimpleNamespace(value=task)
                    table, cached_images, cached_labels = create(folder, SimpleNamespace(value=0), selected_task)
                    for class_id in (last, 0):
                        table._update([str(class_id)])
                        self.assertEqual(table.value[0]['class_id'], class_id)
                        render(SimpleNamespace(value=configs), table, cached_images, cached_labels,
                               SimpleNamespace(value=42), selected_task)
                        self.assertEqual(seen[-1], [class_id, class_id])

    def test_preview_controls_work_without_cache_and_refresh_after_download(self):
        namespace, create = cell_with_arguments(['DATA_DIR', 'preview_refresh', 'task'])
        with tempfile.TemporaryDirectory() as folder:
            task = SimpleNamespace(value='cifar10')
            table, images, labels = create(folder, SimpleNamespace(value=0), task)
            self.assertIsNone(images)
            self.assertEqual(len(labels), 0)
            self.assertEqual(table.value[0]['Preview'], 'Not cached')
            cache = Path(folder) / 'cifar10'
            cache.mkdir()
            np.save(cache / 'X_train.npy', np.zeros((1, 3, 32, 32), dtype=np.uint8))
            np.save(cache / 'y_train.npy', np.array([3]))
            table, images, labels = create(folder, SimpleNamespace(value=1), task)
            self.assertEqual(table.value[0]['class_id'], 3)
            self.assertIn('<img', table.value[0]['Preview'].text)
            self.assertEqual(images.shape, (1, 3, 32, 32))


if __name__ == '__main__':
    unittest.main()
