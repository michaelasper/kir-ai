mod fs_util;
mod mlx;
mod native_gemma;
mod native_matvec;
mod native_qwen;
mod native_text;
mod server;
mod snapshot_backend;
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

pub use mlx::*;
pub use native_gemma::*;
pub use native_qwen::*;
pub use native_text::*;
pub use server::*;
pub use snapshot_backend::*;
