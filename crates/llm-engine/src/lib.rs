mod engine;
mod fs_util;
mod mlx;
mod native_gemma;
mod native_matvec;
mod native_qwen;
mod native_text;
mod snapshot_backend;
mod sync_ext;

pub const DEFAULT_MODEL_ID: &str = "local-qwen36";

pub mod route {
    #[cfg(feature = "test-utils")]
    pub use super::engine::build_router_with_protocol_test_backend;
    pub use super::engine::{
        EngineOptions, build_router, build_router_with_backend,
        build_router_with_backend_and_concurrency, build_router_with_backend_and_options,
    };
}

pub use engine::*;
pub use mlx::*;
pub use native_gemma::*;
pub use native_qwen::*;
pub use native_text::*;
pub use snapshot_backend::*;
