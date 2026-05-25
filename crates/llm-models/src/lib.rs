//! Model-family metadata and configuration validation.
//!
//! This crate centralizes the stable family slugs, backend support matrix,
//! prompt/cache adapter metadata, native text model specs, and safetensors index
//! validation used before a model is admitted for local inference.
//!
//! The public facade intentionally re-exports only the model-family metadata,
//! native text configuration specs, and safetensors admission helpers listed
//! below. Implementation modules stay private so adding a helper inside a module
//! does not accidentally expand the crate API.

mod family;
mod gemma;
mod model_spec;
mod native_text;
mod qwen;
mod safetensors;

pub use family::{
    BackendKind, BackendKindParseError, DeepSeekFamilyAdapter, FamilyCapabilityFlags,
    GemmaFamilyAdapter, LlamaFamilyAdapter, ModelFamily, ModelFamilyAdapter, ModelFamilyParseError,
    PromotionStage, QwenFamilyAdapter,
};
pub use gemma::{GemmaAttentionKind, GemmaModelSpec, GemmaWeightLayout};
pub use model_spec::ModelSpec;
pub use native_text::NativeTextModelSpec;
pub use qwen::{AttentionKind, ModelSpecError, QwenModelSpec};
pub use safetensors::SafetensorsIndex;
