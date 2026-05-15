use crate::{ModelFamily, ModelSpecError, SafetensorsIndex};

pub trait ModelSpec {
    fn family(&self) -> ModelFamily;
    fn architecture(&self) -> &str;
    fn model_type(&self) -> &str;
    fn text_model_type(&self) -> &str;
    fn max_position_embeddings(&self) -> u32;
    fn num_hidden_layers(&self) -> u32;
    fn hidden_size(&self) -> u32;
    fn vocab_size(&self) -> u32;
    fn validate_text_weights(&self, index: &SafetensorsIndex) -> Result<(), ModelSpecError>;
}
