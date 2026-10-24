"""Check full-notebook import against the local Torch Canvas desktop parser."""

import os
import shutil
import subprocess
import unittest
from pathlib import Path

from residual_mlp_baseline import notebook


PROJECT = Path(__file__).resolve().parents[1]
CANVAS = Path(os.environ.get('TORCH_CANVAS_ROOT', PROJECT.parent / 'torch-canvas'))


class TorchCanvasCompatibilityTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node') and (CANVAS / 'web' / 'index.html').is_file(),
                         'Requires Node and a sibling Torch Canvas desktop build (or TORCH_CANVAS_ROOT)')
    def test_notebook_imports_and_unchanged_diagram_preserves_source(self):
        script = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
const scripts = [...html.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)]
    .map(match => match[1]).filter(Boolean);
const context = vm.createContext({ console, setTimeout, clearTimeout, performance, window: {} },
    { codeGeneration: { strings: false, wasm: false } });
vm.runInContext(scripts.slice(0, 3).join('\n') + '\nglobalThis.api = NN;', context);
const source = fs.readFileSync(process.argv[2], 'utf8');
const result = context.api.importPython(source);
assert.equal(result.graph.name, 'ResidualMLP');
assert.ok(result.graph.nodes.length > 0);
assert.ok(result.usedClasses.includes('Block'));
assert.equal(context.api.syncSource(source, result.graph), source);
'''
        result = subprocess.run(['node', '-e', script, str(CANVAS / 'web' / 'index.html'), notebook.__file__],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
