//! Model-family metadata and configuration validation.
//!
//! This crate centralizes the stable family slugs, backend support matrix,
//! prompt/cache adapter metadata, native text model specs, and safetensors index
//! validation used before a model is admitted for local inference.

mod family;
mod gemma;
mod model_spec;
mod native_text;
mod qwen;
mod safetensors;

pub use family::*;
pub use gemma::*;
pub use model_spec::*;
pub use native_text::*;
pub use qwen::*;
pub use safetensors::*;
