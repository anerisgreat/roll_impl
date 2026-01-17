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
          #   packageOverrides = python-final: python-prev: {
          #     pytorch = python-prev.pytorch.overridePythonAttrs (o: {
          #       src = pkgs.fetchFromGitHub {
          #         owner = "pytorch";
          #         repo = "pytorch";
          #         rev = "eb158a89bd250620a651f5bf7b691432862db347";
          #         fetchSubmodules = true;
          #         hash = "sha256-QyfSbcnrQnztnM+mtRcfGzUBVfgA1/+FLpCFhOH6e2w=";
          #       };
          #       patches =
          #         (o.patches or []) ++ [./nccl-version-patch.patch];
          #     });
          #     torchvision = python-prev.torchvision.overridePythonAttrs (o: {
          #       dependencies = [
          #           python-prev.numpy
          #           python-prev.pillow
          #           python-final.pytorch
          #           python-prev.scipy
          #       ];
          #       buildInputs = [
          #           pkgs.libjpeg_turbo
          #           pkgs.libpng
          #           python-final.pytorch.cxxdev
          #       ] ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
          #           # This should match the SDK used by `torch` above
          #           pkgs.apple-sdk_13

          #           # error: unknown type name 'MPSGraphCompilationDescriptor'; did you mean 'MPSGraphExecutionDescriptor'?
          #           # https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphcompilationdescriptor/
          #           (pkgs.darwinMinVersionHook "12.0")];
          #     });
          #   };
          };
        })];
      pkgs = import nixpkgs {
            inherit system; inherit overlays;
            config = {
                allowUnfree = true;
                cudaSupport = false;
            };
      };
      keelDerivation = dsetName: dsetHash: (pkgs.stdenv.mkDerivation rec {
        pname = "${dsetName}-keel-derivation";
        version = "1.0";

        fetchurl = "https://sci2s.ugr.es/keel/dataset/data/imbalanced/${dsetName}.zip";
        src = pkgs.fetchzip rec {
            url = fetchurl;
            hash = dsetHash;
        };
        # phases = [ "unpackPhase" "installPhase" ];
        installPhase = ''
        mkdir -p $out
        cp $src/*.dat $out/
        '';
        # installPhase = "install -D $src $out/" + builtins.baseNameOf fetchurl;
      });
      keelDerivationsToHash = {
        "wisconsin" =
          "sha256-wEDv5f+lk4Ov2+NiIdo+eSV4q6kJheNQCkHpxY+jJGc=";
        "pima" =
          "sha256-HPJdQStgYICIlMnHdW42VsY9RfEjKFfzCh615uv144s=";
        "iris0" =
          "sha256-9zvEfsC2VghQIejWXZsNAYXdKFK7Z/fTnpVBOy6VPuc=";
        "haberman" =
          "sha256-FQ1lJrPmGaPTLaBZZMI6jmYrL3IVB5zFNuZmGEr0LZY=";
        "vehicle2" =
          "sha256-SX/FhGJmawqzCF5pRTqUHTesSoy8eTSUVhdaaT1w/xM=";
        "new-thyroid1" =
          "sha256-umK87ZLCYh9mi+zGZCDpUkJRnSjwFyDbO9UeKWPx0E8=";
        "yeast3" =
          "sha256-1pyi/wd1TB9ipq+4A8tNx7/7Zt2KB7FP3sQtB8xAU04=";
        "vowel0" =
          "sha256-Gdu/fOBegURFnqr5KWMf9Drg2i3aS37OonfBH7h1i9Y=";
        "led7digit-0-2-4-5-6-7-8-9_vs_1" =
          "sha256-3wuW4hAPcFtt4IbeM9AN4BvOaiBl87WaPkvtUrj944k=";
        "ecoli-0-1_vs_5" =
          "sha256-ehLZYXthZMZ+hTitD/NqTBFckd0c+OZ7ulTCyXPHKSA=";
        "cleveland-0_vs_4" =
          "sha256-PANcqEm/bsI/uUPmdVuBoR0ibZiQ5UorRG0uQ/1TevM=";
        "glass4" =
          "sha256-1Pb7aMaW05MNqx2sfJtUmFkuZNSbM2vFI9XlLkhwbjk=";
        "page-blocks-1-3_vs_4" =
          "sha256-OyPo1k6q6JPZiJNKoixc8zWKH/SbVB0IbCvlH9JqN0U=";
        # "dermatology-6" =
        #   "sha256-deDv5f+lk4Ov2+NiIdo+eSV4q6kJheNQCkHpxY+jJGc=";
        # "zoo-3" =
        #   "sha256-zoDv5f+lk4Ov2+NiIdo+eSV4q6kJheNQCkHpxY+jJGc=";
        # "shuttle-6_vs_2-3" =
        #   "sha256-shDv5f+lk4Ov2+NiIdo+eSV4q6kJheNQCkHpxY+jJGc=";
        # "winequality-red-4" =
        #   "sha256-fEDv5f+lk4Ov2+NiIdo+eSV4q6kJheNQCkHpxY+jJGc=";
        # "poker-9_vs_7" =
        #   "sha256-fEDv5f+lk4Ov2+NiIdo+eSV4q6kJheNQCkHpxY+jJGc=";
      };
      keelDerivationsMap = builtins.mapAttrs
        (name: hash: (keelDerivation name hash)) keelDerivationsToHash;
      keelNameToPathMap = builtins.mapAttrs
        (name: deriv: "${deriv}/${name}.dat") keelDerivationsMap;
      keelDerivationsList = builtins.attrValues keelDerivationsMap;
      keelShellHookEntriesMap = builtins.mapAttrs
        (name: path: "export keel_${name}_dir=${path};")
        keelNameToPathMap;
      keelShellHookString = builtins.concatStringsSep "\n"
        (builtins.attrValues keelShellHookEntriesMap);
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
                    ( buildPythonPackage rec {
                        pname = "keel_ds";
                        version = "0.2.5";
                        src = pkgs.fetchFromGitHub {
                            owner = "maicondallg";
                            repo = "KeelDS";
                            rev = "918f000";
                            sha256 = "sha256-zjx2JnlxPPNvnXuSu2Ho+KDZaiqoysVW7HggAeL7Nao=";
                        };

                        propagatedBuildInputs = [pandas poetry-core ];
                        format="pyproject";
                        # patches =
                        #     (o.patches or []) ++ [./adult-dataset-numpy-version.patch];

                    })

                    ( buildPythonPackage rec {
                        pname = "adult_dataset";
                        version = "3.0.0";
                        src = fetchPypi {
                            inherit pname version;
                            sha256 = "sha256-/xgipMO3yPqeG+m/kFXzchJ+t51oExq8cWcBeulPMeA=";
                        };

                        propagatedBuildInputs = [flit pytorch numpy pandas];
                        format="pyproject";
                        patches =
                            (o.patches or []) ++ [./adult-dataset-numpy-version.patch];

                    })
                    ( buildPythonPackage rec {
                        pname = "kdepy";
                        version = "1.1.12";
                        src = fetchPypi {
                            inherit pname version;
                            sha256 = "sha256-6zpiq8WpgvWjDsCjVg4vnL09GbxPchsuu94GlJ1Sr2E=";
                        };

                        propagatedBuildInputs = [numpy scipy cython];
                    })

                ]))];
            shellHook = keelShellHookString;
        };
    };
}
