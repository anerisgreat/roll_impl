{
  #https://nix.dev/guides/recipes/python-environment.html
  description = "Python development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
  };

  outputs = { self, nixpkgs, nixpkgs-python }:
    let
      system = "x86_64-linux";

      overlays = [

        (final: prev: {
          python3 = prev.python3.override {
            packageOverrides = python-final: python-prev: {
              pytorch = python-prev.pytorch.overridePythonAttrs (o: {
                src = pkgs.fetchFromGitHub {
                  owner = "pytorch";
                  repo = "pytorch";
                  rev = "eb158a89bd250620a651f5bf7b691432862db347";
                  fetchSubmodules = true;
                  hash = "sha256-QyfSbcnrQnztnM+mtRcfGzUBVfgA1/+FLpCFhOH6e2w=";
                };
                patches =
                  (o.patches or []) ++ [./nccl-version-patch.patch];
              });
            };
          };
        })
        ];
      pkgs = import nixpkgs {
            inherit system; inherit overlays;
            config = {
                allowUnfree = true;
                cudaSupport = false;
            };
      };
    in{
        devShells.${system}.default = pkgs.mkShell {
            buildInputs = with pkgs; [
               (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
                    numpy
                    pytorch
                    scikit-learn
                    matplotlib
                    pandas
                    plotly
                    torchvision
                    # ( buildPythonPackage rec {
                    #     pname = "adult_dataset";
                    #     version = "3.0.0";
                    #     src = fetchPypi {
                    #         inherit pname version;
                    #         sha256 = "sha256-/xgipMO3yPqeG+m/kFXzchJ+t51oExq8cWcBeulPMeA=";
                    #     };

                    #     propagatedBuildInputs = [flit torch numpy pandas];
                    #     format="pyproject";
                    # })
                ]))];
        };
    };
}
