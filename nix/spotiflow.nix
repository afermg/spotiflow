{
  lib,
  buildPythonPackage,
  setuptools,
  setuptools-scm,
  wheel,
  numpy,
  scipy,
  scikit-image,
  pandas,
  pillow,
  pydash,
  networkx,
  dask,
  lightning,
  configargparse,
  tensorboard,
  tifffile,
  torchvision,
  tqdm,
  typing-extensions,
  zarr,
  psutil,
  torch,
  wandb,
  # Local nix overlay packages
  csbdeep,
}:
buildPythonPackage {
  pname = "spotiflow";
  version = "0.6.0";

  src = ./..;
  pyproject = true;

  # The setuptools build picks up setuptools_scm[toml] from
  # build-system.requires in pyproject.toml. Pin a synthetic version so
  # the nix sandbox (no .git, no tags) doesn't fail at build time.
  env.SETUPTOOLS_SCM_PRETEND_VERSION = "0.6.0";

  # Upstream pyproject pins setuptools <=71.0 as a defence against the
  # license-expression breaking change in setuptools 72+. nixpkgs ships
  # a newer setuptools; relax the constraint. The build still uses the
  # nix-provided setuptools — we're not bypassing dependency resolution,
  # just letting it be satisfied by ≥72.
  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace-fail 'setuptools<=71.0' 'setuptools'
  '';

  build-system = [
    setuptools
    setuptools-scm
    wheel
  ];

  propagatedBuildInputs = [
    torch
    torchvision
    numpy
    scipy
    scikit-image
    pandas
    pillow
    pydash
    networkx
    dask
    lightning
    configargparse
    tensorboard
    tifffile
    tqdm
    typing-extensions
    zarr
    psutil
    csbdeep
    wandb
    # crick is in pyproject install_requires but unused by the inference
    # path (grep -r "crick" spotiflow/ → no hits). It's not in nixpkgs;
    # skip it. dontCheckRuntimeDeps below masks the metadata mismatch.
  ];

  doCheck = false;
  pythonRuntimeDepsCheck = false;
  dontCheckRuntimeDeps = true;

  pythonImportsCheck = [
    # Top-level import pulls torch + the lib.* compiled extensions; the
    # runtime smoke test covers this with PYTHONSAFEPATH=1, where the
    # in-tree source dir doesn't shadow the nix-built package.
  ];

  meta = {
    description = "Accurate and efficient spot detection with stereographic flow regression";
    homepage = "https://github.com/weigertlab/spotiflow";
    license = lib.licenses.bsd3;
  };
}
