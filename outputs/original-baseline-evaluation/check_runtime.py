import json, os, platform
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
assert 'A100' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
receipt = {
    'python': platform.python_version(), 'torch': torch.__version__,
    'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
    'ram_gib': os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30,
    'default_dtype': str(torch.get_default_dtype()),
    'torch_threads': torch.get_num_threads(),
    'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
    'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
}
assert receipt['ram_gib'] > 64, receipt
Path('/content/original-baseline-runtime.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
