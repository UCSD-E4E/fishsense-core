pub mod errors;
pub mod fish;
pub mod gpu;
pub mod laser;
pub mod spatial;
pub mod world_point_handler;

/// Call once at the top of a test (or `main`) to emit tracing output.
///
/// Respects the `RUST_LOG` environment variable, e.g.
/// `RUST_LOG=fishsense_core=debug cargo test`.
/// Safe to call multiple times — subsequent calls are no-ops.
#[cfg(any(test, feature = "tracing-init"))]
pub fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_test_writer()
        .try_init();
}