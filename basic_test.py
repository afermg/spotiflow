"""Standalone smoke test for the Spotiflow wrap.

Loads the pretrained ``general`` model, runs a forward pass on a tiny
synthetic image with two bright spots, and prints the resulting label
mask's unique labels and shape. Should report cuda device.

Run from the repo root:

    nix develop --impure --command python basic_test.py
"""

import os
import sys

import numpy

# PYTHONSAFEPATH=1 (set in flake's runServer/devShell shellHook) keeps
# Python from prepending this script's directory to sys.path — that
# stops the in-tree ``spotiflow/`` source dir from shadowing the
# nix-built compiled package. Append (not prepend) so ``server.py``,
# which is a top-level file next to this one, is still importable.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# server.py reads ``address = sys.argv[1]`` at module top so it works
# under ``nix run -- <ipc-addr>``. We don't open an IPC socket here —
# inject a placeholder so the import doesn't IndexError.
if len(sys.argv) < 2:
    sys.argv.append("ipc:///tmp/spotiflow_basic_test.ipc")

import torch  # noqa: E402

from server import setup  # noqa: E402

assert torch.cuda.is_available(), "CUDA not available — GPU is required"

processor, info = setup(pretrained_name="general", device=0, spot_radius_px=3)
print(f"info: {info}")

# Two clearly-isolated bright spots on a dim background.
rng = numpy.random.default_rng(0)
img = (rng.random((256, 256)) * 50).astype(numpy.float32)
img[100, 100] = 800.0
img[150, 200] = 800.0

# Wrap as NCZYX (N=1, C=1, Z=1, Y=256, X=256).
arr = img[None, None, None, :, :]
labels = processor(arr)
print(f"output shape: {labels.shape}, dtype: {labels.dtype}")
uniq = numpy.unique(labels)
print(f"unique labels: {uniq.tolist()}")
print(f"n_spots detected: {(uniq > 0).sum()}")
