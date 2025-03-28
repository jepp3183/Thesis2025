{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let 
          pkgs = import nixpkgs {
            system = system;
            config = { allowUnfree = true; };
          }; 

          pythonPackages = pkgs.python312Packages;

          raiutils = 
            pythonPackages.buildPythonPackage rec {
              pname = "raiutils";
              version = "0.4.2";
              doCheck = false;
              src = pkgs.fetchPypi {
                inherit pname version;
                sha256 = "sha256-0hCk1aBZ5IOI00HuAsuH88krvx8Ly87PBP2TpZnS3KQ=";
              };

              dependencies = [
                pythonPackages.numexpr
              ];
            };

          dice-ml = 
            pythonPackages.buildPythonPackage rec {
              pname = "dice_ml";
              version = "0.11";
              doCheck = false;
              src = pkgs.fetchPypi {
                inherit pname version;
                sha256 = "sha256-Kea+qeTId8qmjs3V2YH5HMjeO3PNrUU/18fgAKM2V20=";
              };
              dependencies = [
                raiutils
              ];
            };

            mmd-critic = 
              pythonPackages.buildPythonPackage rec {
                pname = "mmd_critic";
                version = "0.1.2";
                doCheck = false;
                src = pkgs.fetchPypi {
                  inherit pname version;
                  sha256 = "sha256-IWWkAGGRKT8MHPrEyvIRJIsfQXJFaQEHymHJWlhfs2k=";
                  };
              };

            umap-learn = 
              pythonPackages.buildPythonPackage rec {
                pname = "umap-learn";
                version = "0.5.7";
                doCheck = false;
                src = pkgs.fetchPypi {
                  inherit pname version;
                  sha256 = "sha256-sql5c+TG/86/JBEAqN5YmkyEEmqDKrQPKWxtn8xesZ4=";
                };
                dependencies = [
                  pythonPackages.numba
                  pythonPackages.pynndescent
                ];
              };

        in
        {
          devShells.default = pkgs.mkShell {
            buildInputs = with pythonPackages; [ 
                ipython
                ipykernel
                jupyter
                notebook
                tqdm
                matplotlib
                numpy
                scikit-learn
                pandas
                seaborn
                tensorflow
                keras
                torch
                debugpy
                dice-ml
                mmd-critic
                umap-learn
                # alibi
            ];

            shellHook = ''
            '';
          };
        }
      );
}
