{
  #https://nix.dev/guides/recipes/python-environment.html
  description = "Python development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
  };

  outputs = { self, nixpkgs, nixpkgs-python }:
    let
      system = "x86_64-linux";

      overlays = [
        (final: prev: {
          python3 = prev.python3.override {
            packageOverrides = python-final: python-prev: {
              torch = python-prev.torch.overridePythonAttrs (o: {
                src = pkgs.fetchFromGitHub {
                  owner = "pytorch";
                  repo = "pytorch";
                  rev = "eb158a89bd250620a651f5bf7b691432862db347";
                  fetchSubmodules = true;
                  hash = "sha256-QyfSbcnrQnztnM+mtRcfGzUBVfgA1/+FLpCFhOH6e2w=";
                };
                # nativeBuildInputs = [ pkgs.cudaPackages.nccl ] ++ python-prev.torch.nativeBuildInputs;
              });
            };
          };
        })];
      pkgs = import nixpkgs { inherit system; inherit overlays; config.allowUnfree = true; config.cudaSupport = true; useSystemNccl = true;};

    in
    rec {
        devShells.${system}.default = pkgs.mkShell {
            buildInputs = with pkgs; [
              cudaPackages.nccl
               (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
                    numpy
                    pytorch
                ]))];
        };
    };
}
