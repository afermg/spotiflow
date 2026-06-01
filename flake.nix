{
  inputs = {
    # Pin to the exact nixpkgs commit cellpose / ultrack use — it has a
    # cached CUDA torch closure. The bare ``nixos-unstable`` channel
    # rotated past that cache and now triggers a from-source PyTorch +
    # flash-attention build targeting sm_75 → sm_121 (Blackwell), which
    # is hours of compile work on an Ampere host. The pinned sha skips
    # that entirely.
    nixpkgs.url = "github:NixOS/nixpkgs/20075955deac2583bb12f07151c2df830ef346b4";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    nahual-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
      in
      with pkgs;
      rec {
        formatter = pkgs.alejandra;
        packages = pkgs.callPackage ./nix { };

        apps.default =
          let
            python_with_pkgs = python3.withPackages (pp: [
              (inputs.nahual-flake.packages.${system}.nahual)
              packages.spotiflow
              pp.scikit-image
            ]);
            runServer = pkgs.writeScriptBin "runserver.sh" ''
              #!${pkgs.bash}/bin/bash
              # PYTHONSAFEPATH=1 keeps Python from prepending the script
              # directory to sys.path, so the in-tree spotiflow/ source
              # dir cannot shadow the nix-built package (which carries
              # the compiled C-extensions in lib/).
              export PYTHONSAFEPATH=1
              ${python_with_pkgs}/bin/python ${self}/server.py ''${@:-"ipc:///tmp/spotiflow.ipc"}
            '';
          in
          {
            type = "app";
            program = "${runServer}/bin/runserver.sh";
          };

        devShells = {
          default =
            let
              python_with_pkgs = python3.withPackages (pp: [
                (inputs.nahual-flake.packages.${system}.nahual)
                packages.spotiflow
                pp.scikit-image
                pp.tifffile
                pp.pyyaml
              ]);
            in
            mkShell {
              packages = [
                python_with_pkgs
                pkgs.cudaPackages.cudatoolkit
              ];
              shellHook = ''
                # PYTHONSAFEPATH=1 (Python 3.11+) keeps Python from
                # prepending the cwd / script-dir to sys.path. Without
                # this, running ``python -c "import spotiflow"`` from
                # the repo root would pick the in-tree source dir
                # (which has no compiled spotiflow/lib/*.so) over the
                # nix-built package and fail at import.
                export PYTHONSAFEPATH=1
                export PYTHONDONTWRITEBYTECODE=1
                export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
                export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH
              '';
            };
        };
      }
    );
}
