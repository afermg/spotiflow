# Spotiflow Nahual OCI image

Build the reproducible archive and load it into Podman or Docker:

```console
nix build .#oci-image
podman load < result                         # or: docker load < result
```

The image is tagged `nahual/spotiflow:local` and listens on TCP port 5555. The
pretrained model is downloaded on first setup, so persist `/tmp/nahual` as a
model cache:

```console
podman run --rm --device nvidia.com/gpu=all -p 5555:5555 \
  -v nahual-spotiflow-cache:/tmp/nahual nahual/spotiflow:local
```

For Docker, replace the CDI device option with `--gpus all`. CPU operation is
supported. With Nahual and NumPy installed on the host, run pretrained
end-to-end puncta detection with:

```console
NAHUAL_DEVICE=cpu python oci/smoke_test.py
```

The server returns an int32 label mask with a configurable disk around each
predicted spot. Upstream pretrained model names or a mounted custom checkpoint
directory can be selected through setup parameters.
