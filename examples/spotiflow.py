"""Client demo for the Spotiflow Nahual wrap.

Start the server in a separate shell:

    cd /home/amunoz/projects/nahual_models/spotiflow
    nix run --impure .#default -- ipc:///tmp/spotiflow.ipc

Then run this script. The server returns a per-image int32 label mask
where each detected puncta is rasterised as a disk around its centroid.
"""

import numpy

from nahual.process import dispatch_setup_process

# Spotiflow isn't in nahual's built-in OUTPUT_SIGNATURES registry yet;
# pass the (dict, numpy) signature explicitly.
setup, process = dispatch_setup_process("spotiflow", signature=("dict", "numpy"))
address = "ipc:///tmp/spotiflow.ipc"

# Load the pretrained "general" model on cuda:0 with a 3-px disk per spot.
parameters = {
    "pretrained_name": "general",
    "device": 0,
    "spot_radius_px": 3,
    # Optional: tune the probability threshold (None = model default).
    # "prob_thresh": 0.4,
}
response = setup(parameters, address=address)
print(f"setup info: {response}")

# Synthetic image with two bright spots, formatted as NCZYX.
rng = numpy.random.default_rng(42)
img = (rng.random((256, 256)) * 50).astype(numpy.float32)
img[100, 100] = 800.0
img[150, 200] = 800.0
arr = img[None, None, None, :, :]

label_mask = process(arr, address=address)
print(f"output shape: {label_mask.shape}, dtype: {label_mask.dtype}")
uniq = numpy.unique(label_mask)
print(f"unique labels: {uniq.tolist()}")
print(f"n_spots: {(uniq > 0).sum()}")
