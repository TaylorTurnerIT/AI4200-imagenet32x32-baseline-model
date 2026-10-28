"""Launch the guarded candidate process once; there is no retry loop."""
import subprocess
import sys

subprocess.run([
    sys.executable, '-u', '/content/run_pre_reduction_experiment.py',
    '--data', '/content/data/imagenet32',
    '--output', '/content/ai4200-pre-reduction-run',
    '--protocol', '/content/pre-reduction-protocol.json',
    '--runtime', '/content/pre-reduction-ready.json',
], check=True)
