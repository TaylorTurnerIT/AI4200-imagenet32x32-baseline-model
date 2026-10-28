"""Prepare matching CUDA and official training data; this script never trains."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

ROOT = Path('/content')
PIN = 'b5ab6e1f75de4cfc257cf85b7e2535cfa217934c'
assert torch.__version__ == '2.11.0+cu130', torch.__version__
assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall', '--no-deps',
                'git+https://github.com/TaylorTurnerIT/AI4200-imagenet32x32-baseline-model.git@' + PIN], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'datasets>=4,<5', 'pillow', 'matplotlib'], check=True)
from datasets import load_dataset
import residual_mlp_baseline
package = Path(residual_mlp_baseline.__file__).parent
reference = json.loads((ROOT/'baseline-qualification.json').read_text())
for name, digest in reference['source_sha256'].items():
    assert hashlib.sha256((package/name).read_bytes()).hexdigest() == digest, name
folder = ROOT/'data/imagenet32'
if not folder.exists():
    partial = folder.with_suffix('.partial')
    partial.mkdir(parents=True, exist_ok=False)
    split = load_dataset('benjamin-paine/imagenet-1k-32x32', split='train')
    size = len(split)
    images = np.lib.format.open_memmap(partial/'X_train.npy', mode='w+', dtype=np.uint8,
                                       shape=(size, 3, 32, 32))
    labels = np.empty(size, dtype=np.int64)
    for start in range(0, size, 10_000):
        batch = split[start:start+10_000]
        rgb = np.stack([np.asarray(image.convert('RGB')) for image in batch['image']])
        stop = start + len(rgb)
        images[start:stop] = rgb.transpose(0, 3, 1, 2)
        labels[start:stop] = batch['label']
        print(f'training cache: {stop:,} / {size:,}', flush=True)
    images.flush()
    np.save(partial/'y_train.npy', labels)
    del images, labels, split
    partial.rename(folder)
expected = json.loads((ROOT/'training-data-reference.json').read_text())
for name, spec in expected.items():
    array = np.load(folder/f'{name}.npy', mmap_mode='r')
    assert list(array.shape) == spec['shape'] and str(array.dtype) == spec['dtype']
    assert hashlib.sha256(memoryview(array)).hexdigest() == spec['array_sha256'], name
origin = json.loads(importlib.metadata.distribution('residual-mlp-baseline').read_text('direct_url.json'))
assert origin['vcs_info']['commit_id'] == PIN
receipt = {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(0), 'ram_gib': os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/2**30,
           'package_origin': origin, 'training_data': expected, 'training_data_matches_reference': True,
           'default_dtype': str(torch.get_default_dtype()), 'torch_threads': torch.get_num_threads(),
           'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
           'prepared_without_training': True}
(ROOT/'pre-reduction-ready.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)
print('PRE_REDUCTION_RUNTIME_READY',flush=True)
