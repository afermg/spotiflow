"""Nahual server for Spotiflow (fluorescence-puncta detector).

Loads a Spotiflow checkpoint (default: the pretrained ``general`` model
from the upstream repo) and serves a label-mask interface compatible
with downstream skimage.measure / cp_measure pipelines:

  * Input  — ``NCZYX`` numpy array (Z and C are squeezed; only the
    first channel is fed to Spotiflow which is a single-channel model).
  * Output — int32 instance label mask, same ``(Y, X)`` as the input,
    with one connected disk per detected spot. Labels run 1..N. Background
    is 0. The disk radius is configurable in ``setup`` (default 3 px).

The label-mask output is the contract that aliby's downstream feature
extractor expects from any segmenter. Spotiflow returns ``(N, 2)``
``(y, x)`` integer coordinates which we rasterise via
``skimage.draw.disk`` here, server-side, so callers don't each need
to roll the same conversion.

Run with:
    nix run . -- ipc:///tmp/spotiflow.ipc
or:
    python server.py ipc:///tmp/spotiflow.ipc
"""

import sys
from functools import partial
from typing import Callable

import numpy
import pynng
import torch
import trio
from nahual.server import responder
from skimage.draw import disk
from spotiflow.model import Spotiflow

address = sys.argv[1]

# Module-level model cache. Re-call ``setup()`` with the same
# (pretrained_name, weights_path, device) tuple as the previous call and
# we reuse the loaded model instead of reinstantiating + re-pulling
# weights. The aliby pipeline calls ``dispatch_segmenter`` → ``setup()``
# once per position, so without this cache a long extraction run can
# reload the model hundreds of times per server — and reload latency
# can push concurrent client requests past the IPC ``recv_timeout``.
_MODEL_CACHE: dict[tuple, tuple] = {}


def setup(
    pretrained_name: str = "general",
    weights_path: str | None = None,
    device: int | None = None,
    prob_thresh: float | None = None,
    spot_radius_px: int = 3,
    min_distance: int = 1,
) -> tuple[Callable, dict]:
    """Load a Spotiflow model and bind per-call inference defaults.

    Parameters
    ----------
    pretrained_name : str
        Name of an upstream pretrained model (``general``, ``hybiss``,
        ``synth_complex`` …). Ignored if ``weights_path`` is set.
    weights_path : str | None
        Local checkpoint directory. Bypasses ``from_pretrained``.
    device : int | None
        CUDA device index. ``None`` → cuda:0 if available, else cpu.
    prob_thresh : float | None
        Spot-probability cut. ``None`` keeps the model's trained default
        (typically ~0.4–0.5 for ``general``).
    spot_radius_px : int
        Disk radius drawn around each spot centroid in the output label
        mask. Tune so the disk roughly covers one puncta footprint at
        your magnification — 3 px is a sensible default for 0.65 µm/px
        confocal at 20×.
    min_distance : int
        Non-max-suppression radius for upstream peak picking. ``1`` keeps
        every local maximum; raise to suppress duplicates on bright spots
        that the heatmap splits.
    """
    if device is None:
        device = 0
    if torch.cuda.is_available():
        torch_device = torch.device(int(device))
        map_location = "cuda"
    else:
        torch_device = torch.device("cpu")
        map_location = "cpu"

    # Reuse the already-loaded model if the same setup args show up
    # again — most aliby workflows hit setup() once per position with
    # identical params, and reload cost (network fetch + state_dict
    # load + .to(device)) is large compared to per-image inference.
    cache_key = (pretrained_name, weights_path, int(device))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        model, _cached_torch_device = cached
    else:
        # Spotiflow's from_pretrained / from_folder accepts only the
        # strings ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"`` for
        # ``map_location`` — not ``"cuda:0"`` or a ``torch.device``.
        # Load to ``"cuda"`` generically, then ``.to()`` the underlying
        # nn.Module onto the specific CUDA index.
        if weights_path is not None:
            model = Spotiflow.from_folder(
                weights_path, map_location=map_location
            )
        else:
            model = Spotiflow.from_pretrained(
                pretrained_name, map_location=map_location
            )

        if torch.cuda.is_available() and int(device) != 0:
            model.model.to(torch_device)

        model.eval()
        _MODEL_CACHE[cache_key] = (model, torch_device)

    info = {
        "device": str(torch_device),
        "pretrained_name": pretrained_name if weights_path is None else None,
        "weights_path": weights_path,
        "prob_thresh": prob_thresh,
        "spot_radius_px": spot_radius_px,
        "min_distance": min_distance,
    }
    processor = partial(
        process,
        model=model,
        device=torch_device,
        prob_thresh=prob_thresh,
        spot_radius_px=spot_radius_px,
        min_distance=min_distance,
    )
    return processor, info


def _coords_to_label_mask(
    coords: numpy.ndarray, shape: tuple[int, int], radius: int
) -> numpy.ndarray:
    """Rasterise ``(N, 2)`` ``(y, x)`` spot centroids into a label mask.

    Each spot gets a unique label 1..N drawn as a disk of ``radius`` px.
    Overlapping disks: later spots overwrite earlier ones — fine for
    sparse puncta but the caller should keep ``radius`` below the
    typical inter-spot distance.
    """
    label_mask = numpy.zeros(shape, dtype=numpy.int32)
    if len(coords) == 0:
        return label_mask
    H, W = shape
    for i, (y, x) in enumerate(coords, start=1):
        rr, cc = disk((int(y), int(x)), radius, shape=shape)
        label_mask[rr, cc] = i
    return label_mask


def process(
    pixels: numpy.ndarray,
    model,
    device: torch.device,
    prob_thresh: float | None,
    spot_radius_px: int,
    min_distance: int,
) -> numpy.ndarray:
    """Detect spots in an ``NCZYX`` array, return a per-image label mask.

    Spotiflow is a single-channel 2D model — we feed the first channel
    of the first Z-plane of each batch entry. Output is ``(N, Y, X)``
    int32; aliby's downstream stack squeezes the batch dim itself.
    """
    if pixels.ndim != 5:
        raise ValueError(f"Expected NCZYX (5D) array, got shape {pixels.shape}")
    N, C, Z, Y, X = pixels.shape

    out = numpy.zeros((N, Y, X), dtype=numpy.int32)
    for n in range(N):
        # Single channel, single Z (Spotiflow is 2D, 1-channel).
        img = pixels[n, 0, 0].astype(numpy.float32)
        predict_kwargs = {}
        if prob_thresh is not None:
            predict_kwargs["prob_thresh"] = prob_thresh
        if min_distance is not None:
            predict_kwargs["min_distance"] = min_distance
        coords, _details = model.predict(img, device=device, **predict_kwargs)
        out[n] = _coords_to_label_mask(
            numpy.asarray(coords), (Y, X), spot_radius_px
        )
    return out


async def main():
    with pynng.Rep0(listen=address, recv_timeout=300_000) as sock:
        print(f"Spotiflow server listening on {address}", flush=True)
        async with trio.open_nursery() as nursery:
            responder_curried = partial(responder, setup=setup)
            nursery.start_soon(responder_curried, sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        pass
