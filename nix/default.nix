{
  lib,
  pkgs,
  python3Packages,
}: let
  callPackage = lib.callPackageWith (pkgs // packages // python3Packages);
  packages = {
    csbdeep = callPackage ./csbdeep.nix {};
    spotiflow = callPackage ./spotiflow.nix {};
  };
in
  packages
