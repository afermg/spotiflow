#!/usr/bin/env python3
"""End-to-end pretrained inference against the Spotiflow OCI container."""

import json
import os

os.environ.setdefault("NAHUAL_IPC_TIMEOUT_MS", "1800000")

import numpy as np
from nahual.process import dispatch_setup_process


def main() -> None:
    address = os.environ.get("NAHUAL_ADDRESS", "tcp://127.0.0.1:5555")
    device = os.environ.get("NAHUAL_DEVICE", "cpu")
    setup, process = dispatch_setup_process("spotiflow")
    info = setup(
        {
            "pretrained_name": "general",
            "device": device,
            "spot_radius_px": 3,
            "min_distance": 1,
        },
        address=address,
    )
    rng = np.random.default_rng(42)
    image = (rng.random((256, 256)) * 50).astype(np.float32)
    image[100, 100] = 800.0
    image[150, 200] = 800.0
    result = process(image[None, None, None, :, :], address=address)
    assert info["device"] == device, info
    assert info["pretrained_name"] == "general", info
    assert result.shape == (1, 256, 256), result.shape
    assert result.dtype == np.int32, result.dtype
    assert result.max() > 0, "No puncta were detected"
    print(
        json.dumps(
            {
                "setup": info,
                "shape": list(result.shape),
                "spots": int(result.max()),
            }
        )
    )


if __name__ == "__main__":
    main()
