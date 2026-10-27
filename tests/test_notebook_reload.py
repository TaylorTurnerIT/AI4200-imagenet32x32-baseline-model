"""Saved-run source changes must execute in the running notebook."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from marimo._config.config import merge_default_config
from marimo._config.manager import MarimoConfigManager, ProjectConfigManager
from marimo._runtime.commands import SyncGraphCommand
from marimo._session.file_change_handler import EditModeReloadStrategy

from residual_mlp_baseline import notebook


class NotebookReloadTests(unittest.TestCase):
    def test_file_reload_queues_execution_even_when_user_prefers_lazy_reload(self):
        user = Mock()
        user.get_config.return_value = merge_default_config({
            'runtime': {'watcher_on_save': 'lazy'},
        })
        config = MarimoConfigManager(user, ProjectConfigManager(notebook.__file__))
        self.assertEqual(config.get_config()['runtime']['watcher_on_save'], 'autorun')
        session = Mock()
        session.app_file_manager.app.cell_manager = SimpleNamespace(
            cell_ids=lambda: ['setup', 'controls'],
            codes=lambda: ['augmentation_enabled = True',
                           'switch = mo.ui.switch(value=augmentation_enabled)'],
        )
        strategy = EditModeReloadStrategy(config)
        strategy.handle_reload(session, transaction=SimpleNamespace(changes=[]),
                               changed_cell_ids={'setup'})
        command = session.put_control_request.call_args.args[0]
        self.assertIsInstance(command, SyncGraphCommand)
        self.assertEqual(command.run_ids, ['setup'])
        self.assertEqual(command.cells['setup'], 'augmentation_enabled = True')
        self.assertEqual(command.delete_ids, [])

    def test_launcher_always_enables_notebook_file_watching(self):
        project = Path(notebook.__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            uv = Path(folder) / 'uv'
            uv.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@"\n')
            uv.chmod(0o755)
            result = subprocess.run(
                ['/bin/sh', str(project / 'run'), '--headless', '--port', '2718'],
                env={**os.environ, 'PATH': folder + os.pathsep + os.environ.get('PATH', '')},
                capture_output=True, text=True, check=True,
            )
        self.assertEqual(result.stdout.splitlines(), [
            'run', '--locked', 'marimo', 'edit', 'src/residual_mlp_baseline/notebook.py',
            '--watch', '--headless', '--port', '2718',
        ])


if __name__ == '__main__':
    unittest.main()
