pub mod cli;
#[cfg(feature = "mlx")]
mod mlx;
pub mod model_cli;
mod server;
mod snapshot_backend;
#[cfg(any(feature = "mlx", feature = "native-qwen", feature = "native-gemma"))]
mod sync_ext;

pub use llm_util::defaults::DEFAULT_MODEL_ID;

pub mod route {
    #[cfg(feature = "test-utils")]
    pub use super::server::build_router_with_protocol_test_backend;
    #[allow(deprecated)]
    pub use super::server::{
        EngineOptions, PublicInferenceRateLimit, build_router, build_router_with_backend,
        build_router_with_backend_and_concurrency, build_router_with_backend_and_options,
        build_router_with_backend_and_options_allowing_unauthenticated_admin, router_builder,
    };
}

pub use llm_native_runtime::parse_snapshot_model_family;
#[cfg(feature = "native-qwen")]
pub use llm_native_runtime::{
    DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS, NativeQwenBackend, NativeQwenLoadOptions,
};
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
pub use llm_native_runtime::{
    DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS, DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS,
    DEFAULT_NATIVE_TEXT_PREFIX_CACHE_BYTES, NativeTextBackend, NativeTextLoadOptions,
    NativeTextRuntimeOptions,
};
#[cfg(feature = "native-gemma")]
pub use llm_native_runtime::{NativeGemmaBackend, NativeGemmaLoadOptions};
#[cfg(feature = "mlx")]
pub use mlx::*;
pub use server::*;
pub use snapshot_backend::*;
