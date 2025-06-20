{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let pkgs = import nixpkgs {
          system = system;
          config = { allowUnfree = true; };
        }; in
        {

          devShells.default = pkgs.mkShell {
            buildInputs = with pkgs; [ 
              typst
            ];

            shellHook = ''
            '';
          };

        }
      );
}
