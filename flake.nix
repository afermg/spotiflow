{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/20075955deac2583bb12f07151c2df830ef346b4";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    nahual-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        modelPackages = pkgs.callPackage ./nix {};
        python_with_pkgs = pkgs.python3.withPackages (pp: [
          inputs.nahual-flake.packages.${system}.nahual
          modelPackages.spotiflow
          pp.scikit-image
        ]);
        runServer = pkgs.writeScriptBin "nahual-spotiflow" ''
          #!${pkgs.bash}/bin/bash
          export PYTHONSAFEPATH=1
          exec ${python_with_pkgs}/bin/python ${self}/server.py \
            "''${1:-tcp://0.0.0.0:5555}"
        '';
        spotiflowApp = {
          type = "app";
          program = "${runServer}/bin/nahual-spotiflow";
        };
      in
        with pkgs; rec {
          formatter = pkgs.alejandra;
          packages =
            modelPackages
            // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
              oci-image = import ./nix/oci-image.nix {
                inherit pkgs;
                name = "spotiflow";
                title = "Nahual Spotiflow";
                description = "Spotiflow puncta detection served through Nahual";
                source = "https://github.com/afermg/spotiflow";
                revision = self.rev or self.dirtyRev or "unknown";
                server = runServer;
                entrypoint = spotiflowApp.program;
              };
            };
          inherit python_with_pkgs;
          scripts.runServer = runServer;
          apps = rec {
            spotiflow = spotiflowApp;
            default = spotiflow;
          };
          devShells.default = mkShell {
            packages = [
              python_with_pkgs
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              python3Packages.tifffile
              python3Packages.pyyaml
            ];
            shellHook = ''
              export PYTHONSAFEPATH=1
              export PYTHONDONTWRITEBYTECODE=1
              export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH
            '';
          };
        }
    );
}
