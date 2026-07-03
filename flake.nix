{
  description = "fishsense-core — dual Cargo + uv workspace for fish-measurement algorithms";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Native libraries the Rust build links against:
        #   * OpenBLAS — ndarray-linalg's `openblas-system` feature locates it via
        #     pkg-config, and the compiled extension dlopen's it at test/run time.
        #   * OpenSSL  — reqwest's compile-time model download (build.rs) on Linux.
        # Note: the OpenCV/libclang toolchain is intentionally absent — the crate
        # dropped the `opencv` dependency in favour of pure-Rust image/imageproc.
        nativeLibs = [
          pkgs.openblas
          pkgs.openssl
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            pkgs.pkg-config
            # Rust toolchain (kept in lockstep by nixpkgs).
            pkgs.cargo
            pkgs.rustc
            pkgs.clippy
            pkgs.rustfmt
            pkgs.rust-analyzer
            # Python side of the workspace (PyO3 wrappers + pytest via uv).
            pkgs.uv
            pkgs.python313
          ];
          buildInputs = nativeLibs;

          # OpenBLAS is loaded dynamically by `cargo test`/the built extension;
          # put it (and OpenSSL) on the loader path for both Linux and macOS.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath nativeLibs;
          shellHook = ''
            export DYLD_FALLBACK_LIBRARY_PATH="${pkgs.lib.makeLibraryPath nativeLibs}''${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
          '';
        };

        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
