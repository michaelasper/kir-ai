use crate::{ModelFamily, ModelSpecError, SafetensorsIndex};

/// Normalized model configuration needed by native text execution.
///
/// Implementations adapt family-specific Hugging Face config JSON into a common
/// contract so loaders can validate tensor presence and dimension assumptions
/// before allocating backend state.
pub trait ModelSpec {
    /// Model family this spec belongs to.
    fn family(&self) -> ModelFamily;
    /// Top-level architecture name from the model config.
    fn architecture(&self) -> &str;
    /// Top-level model type from the model config.
    fn model_type(&self) -> &str;
    /// Text submodel type used by native text execution.
    fn text_model_type(&self) -> &str;
    /// Maximum supported context length.
    fn max_position_embeddings(&self) -> u32;
    /// Number of decoder layers.
    fn num_hidden_layers(&self) -> u32;
    /// Hidden size used by the decoder.
    fn hidden_size(&self) -> u32;
    /// Vocabulary size expected by embedding and output tensors.
    fn vocab_size(&self) -> u32;
    /// Validates that required text inference tensors are present in an index.
    fn validate_text_weights(&self, index: &SafetensorsIndex) -> Result<(), ModelSpecError>;
}
