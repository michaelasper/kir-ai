//! Public facade for the `llm-engine` server crate.
//!
//! The stable API is intentionally small: route construction, snapshot-backed
//! backend selection, and the configuration types the `llm-engine` binary needs
//! to stay thin. Direct native backend implementations live in
//! `llm-native-runtime` and are not part of the engine routing API.

/// CLI parsing and model-management helpers used by the `llm-engine` binary.
pub mod cli;
#[cfg(feature = "mlx")]
mod mlx;
/// Model CLI command handlers used by the `llm-engine` binary and CLI tests.
pub mod model_cli;
mod server;
mod snapshot_backend;
#[cfg(any(feature = "mlx", feature = "native-qwen", feature = "native-gemma"))]
mod sync_ext;

/// Default public model alias used by the server and contract tests.
pub use llm_util::defaults::DEFAULT_MODEL_ID;

/// HTTP router construction API for embedding the engine server around a
/// `ModelBackend`.
pub mod route {
    #[cfg(feature = "test-utils")]
    pub use super::server::build_router_with_protocol_test_backend;
    pub use super::server::configured_hub_client;
    #[allow(deprecated)]
    pub use super::server::{
        EngineConfigError, EngineOptions, PublicInferenceRateLimit, build_router,
        build_router_with_backend, build_router_with_backend_and_concurrency,
        build_router_with_backend_and_options,
        build_router_with_backend_and_options_allowing_unauthenticated_admin, router_builder,
    };
    pub use llm_server::DEFAULT_INFERENCE_CONCURRENCY_LIMIT;
}

/// Snapshot-backed backend selection API.
///
/// This is the public construction path for serving local model snapshots
/// through `llm-engine`; callers should prefer it over constructing native
/// backend implementations directly.
pub mod snapshot {
    pub use super::snapshot_backend::{SnapshotBackendOptions, open_snapshot_backend};
    pub use llm_native_runtime::{SnapshotBackendLoader, parse_snapshot_model_family};
}

/// MLX sidecar backend construction API.
///
/// The MLX backend is exposed because it is a supported engine route target and
/// contract tests need to inject a loopback sidecar. Internal protocol and
/// metrics helpers remain private to `llm-engine`.
#[cfg(feature = "mlx")]
pub mod mlx_backend {
    pub use super::mlx::{MlxBackend, MlxBackendOptions, MlxTimeouts, MlxToolParserMode};
}

/// Native snapshot configuration accepted by `SnapshotBackendOptions`.
///
/// Direct native backend structs are intentionally not re-exported from
/// `llm-engine`; use `llm-native-runtime` if a test or tool needs those
/// lower-level constructors.
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
pub mod native_backend {
    pub use llm_native_runtime::{
        DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS, DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS,
        DEFAULT_NATIVE_TEXT_PREFIX_CACHE_BYTES, NativeTextDiskCacheConfig, NativeTextLoadOptions,
        NativeTextRuntimeOptions,
    };
}

#[cfg(feature = "mlx")]
pub use mlx_backend::{MlxBackend, MlxBackendOptions, MlxTimeouts, MlxToolParserMode};
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
pub use native_backend::{
    DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS, DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS,
    DEFAULT_NATIVE_TEXT_PREFIX_CACHE_BYTES, NativeTextDiskCacheConfig, NativeTextLoadOptions,
    NativeTextRuntimeOptions,
};
pub use route::EngineConfigError;
#[cfg(feature = "test-utils")]
pub use route::build_router_with_protocol_test_backend;
#[allow(deprecated)]
pub use route::{
    DEFAULT_INFERENCE_CONCURRENCY_LIMIT, EngineOptions, PublicInferenceRateLimit, build_router,
    build_router_with_backend, build_router_with_backend_and_concurrency,
    build_router_with_backend_and_options,
    build_router_with_backend_and_options_allowing_unauthenticated_admin, configured_hub_client,
    router_builder,
};
pub use snapshot::{
    SnapshotBackendLoader, SnapshotBackendOptions, open_snapshot_backend,
    parse_snapshot_model_family,
};
