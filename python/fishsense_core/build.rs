fn main() {
    // Linux-only: link a small static library that provides glibc 2.38's
    // C23 `__isoc23_strtol`/`__isoc23_strtoll`/etc. as wrappers around the
    // pre-2.38 `strtol`/`strtoll` exports. The pyke-built ONNX Runtime
    // binaries that `ort`'s `download-binaries` pulls in (the default CPU
    // build) reference the C23 versions, so without these stubs the cdylib
    // fails to dlopen on any system with glibc < 2.38 — including Ubuntu
    // 22.04 LTS, which is inside the manylinux_2_34 baseline we claim to
    // support. The C file is guarded on `!__GLIBC_PREREQ(2, 38)`, so on a
    // newer glibc (or the `--features cuda` / load-dynamic build, which links
    // no ORT) it compiles to an empty .o and the static lib is a no-op. See
    // isoc23_compat.c.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rerun-if-changed=src/isoc23_compat.c");
        cc::Build::new()
            .file("src/isoc23_compat.c")
            .flag_if_supported("-Wno-deprecated-declarations")
            .compile("isoc23_compat");
    }
}
