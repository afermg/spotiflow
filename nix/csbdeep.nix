{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,
  numpy,
  scipy,
  matplotlib,
  six,
  tifffile,
  tqdm,
  packaging,
}:
buildPythonPackage rec {
  pname = "csbdeep";
  version = "0.8.2";
  format = "setuptools";

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-3gGI051erq/gLTTFD8HNZ9kiuSBihyb73M3z/0ePXoY=";
  };

  build-system = [
    setuptools
  ];

  # Spotiflow only consumes csbdeep.utils.utils.normalize_mi_ma and
  # csbdeep.internals.predict.tile_iterator — none of which import
  # tensorflow. Drop the TF / tf-keras deps here to keep this wrap's
  # closure pure-torch and avoid the 6 h TF rebuild on cudaSupport=true.
  propagatedBuildInputs = [
    numpy
    scipy
    matplotlib
    six
    tifffile
    tqdm
    packaging
  ];

  doCheck = false;
  pythonRuntimeDepsCheck = false;
  dontCheckRuntimeDeps = true;

  pythonImportsCheck = [
    # Importing csbdeep top-level pulls in tensorflow via the models/
    # subpackage — covered by the runtime smoke test, skipped at build.
  ];

  meta = {
    description = "Toolbox for content-aware image restoration (CARE) — torch-only subset for the Spotiflow wrap";
    homepage = "https://github.com/CSBDeep/CSBDeep";
    license = lib.licenses.bsd3;
  };
}
