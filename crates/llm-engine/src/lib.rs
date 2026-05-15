mod fs_util;
#[cfg(feature = "mlx")]
mod mlx;
#[cfg(feature = "native-gemma")]
mod native_gemma;
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
mod native_matvec;
#[cfg(feature = "native-qwen")]
mod native_qwen;
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
mod native_text;
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
        EngineOptions, build_router, build_router_with_backend,
        build_router_with_backend_and_concurrency, build_router_with_backend_and_options,
        build_router_with_backend_and_options_allowing_unauthenticated_admin, router_builder,
    };
}

#[cfg(feature = "mlx")]
pub use mlx::*;
#[cfg(feature = "native-gemma")]
pub use native_gemma::*;
#[cfg(feature = "native-qwen")]
pub use native_qwen::*;
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
pub use native_text::*;
pub use server::*;
pub use snapshot_backend::*;
