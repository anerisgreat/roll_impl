{
  #https://nix.dev/guides/recipes/python-environment.html
  description = "Python development environment";
  nixConfig = {
    extra-substituters = [ "https://cuda-maintainers.cachix.org" ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };
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
                cudaSupport = true;
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
        installPhase = ''
        mkdir -p $out
        cp $src/*.dat $out/
        '';
      });

      uciDerivation = dsetName: dsetHash: (pkgs.stdenv.mkDerivation rec {
        pname = "${dsetName}-uci-derivation";
        version = "1.0";

        buildInputs = [pkgs.unzip];

        fetchurl = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip";
        src = pkgs.fetchzip {
            url = fetchurl;
            hash = dsetHash;
            stripRoot = false;
        };
        installPhase = ''
        mkdir -p $out
        mkdir -p /tmp/bank-extract
        cd /tmp/bank-extract
        for f in ${src}/*.zip; do
          unzip -o "$f"
        done
        find . -name "*.csv" -exec cp {} $out/ \;
        '';
      });

      keelDerivationsToHash = {
        # Existing datasets
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
        # New glass datasets
        "glass0" =
          "sha256-T3YAL9g1Wu9paMX51U53xhACmZjLjPuvOQf/dk5+wCk=";
        "glass1" =
          "sha256-L9+XPAfSXH7bdN4g6sD8UiWWgAga8QuNE5/7+wy4hzA=";
        "glass2" =
          "sha256-42Glc6Ujaq6MTfR0EvpBwZGZtCZhYul6TQlnGbW0JN8=";
        "glass5" =
          "sha256-WhqRbiWNykCikh7EB4Zp/oP2A4mgNWmnLa2VrgzW2uw=";
        "glass6" =
          "sha256-c1vib7czReLfmaJci5b+coBSeT37u9qV6eleA9fbGi4=";
      };

      higgsDerivation = pkgs.stdenv.mkDerivation {
        pname = "higgs-uci-derivation";
        version = "1.0";
        src = pkgs.fetchzip {
          url = "https://archive.ics.uci.edu/static/public/280/higgs.zip";
          # Fill this in by running:
          # nix store prefetch-file --hash-type sha256 --unpack https://archive.ics.uci.edu/static/public/280/higgs.zip
          hash = "sha256-PO//JdVhniLRjLyKD1zRfIjGOY6UJFwf1ljxZJKDyHE=";
          stripRoot = false;
        };
        installPhase = ''
          mkdir -p $out
          cp $src/HIGGS.csv.gz $out/
        '';
      };

      uciDerivationsToHash = {
        "bank-additional" =
          "sha256-+myYl/cDZ73d3tTZkdcUa4MoxlZkyvlaPqExha84Vs4=";
      };

      keelDerivationsMap = builtins.mapAttrs
        (name: hash: (keelDerivation name hash)) keelDerivationsToHash;
      uciDerivationsMap = builtins.mapAttrs
        (name: hash: (uciDerivation name hash)) uciDerivationsToHash;

      keelNameToPathMap = builtins.mapAttrs
        (name: deriv: "${deriv}/${name}.dat") keelDerivationsMap;
      uciNameToPathMap = builtins.mapAttrs
        (name: deriv: "${deriv}") uciDerivationsMap;

      keelShellHookEntriesMap = builtins.mapAttrs
        (name: path: "export keel_${builtins.replaceStrings ["-"] ["_"] name}_dir=${path};")
        keelNameToPathMap;
      uciShellHookEntriesMap = builtins.mapAttrs
        (name: path: "export uci_${builtins.replaceStrings ["-"] ["_"] name}_dir=${path};")
        uciNameToPathMap;

      keelShellHookString = builtins.concatStringsSep "\n"
        (builtins.attrValues keelShellHookEntriesMap);
      uciShellHookString = builtins.concatStringsSep "\n"
        (builtins.attrValues uciShellHookEntriesMap);
      manualDatasetsShellHookString = ''
        export uci_higgs_dir=${higgsDerivation};
        export credit_card_fraud_dir=$HOME/.data/creditcard;
        export home_credit_dir=$HOME/.data/homecredit;
        export MLFLOW_TRACKING_URI=https://mlflow.zakobar.com;
        export MLFLOW_TRACKING_USERNAME=admin;
        export MLFLOW_TRACKING_PASSWORD=ioS4YfeIsca8v3K6eQvrY6Kte;
        export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH;
      '';
      fullShellHookString = keelShellHookString + "\n" + uciShellHookString + "\n" + manualDatasetsShellHookString;
    in{
        devShells.${system}.default = pkgs.mkShell {
          buildInputs = with pkgs; [
               sshpass
               kaggle
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
                        patches = [./adult-dataset-numpy-version.patch];

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

                    ( buildPythonPackage rec {
                        pname = "libauc";
                        version = "2.0.1";
                        src = fetchPypi {
                            inherit pname version;
                            sha256 = "sha256-3Tuhk0elQIEgCjnSOlYqxieV+rdvmZRhGAcgpAJswmw=";
                        };

                        propagatedBuildInputs = [pytorch numpy tqdm];
                        doCheck = false;
                    })

                    mlflow

                    imageio
                    scikit-image

                    ( buildPythonPackage rec {
                        pname = "torchxrayvision";
                        version = "1.5.2";
                        src = fetchPypi {
                            inherit pname version;
                            sha256 = "sha256-TS+y9s7GRLWSctQOA9RJ3citKAOknM7h//rS5lK9Oi4=";
                        };

                        propagatedBuildInputs = [pytorch torchvision numpy scipy scikit-learn];
                        preBuild = "touch requirements.txt";
                        doCheck = false;
                    })

                ]))];
            shellHook = fullShellHookString;
        };
    };
}
