use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("invalid Metal input shape: {0}")]
    InvalidShape(String),
    #[error("invalid Metal input: {0}")]
    InvalidInput(String),
    #[error("Metal compile error: {0}")]
    Compile(String),
    #[error("Metal pipeline error: {0}")]
    Pipeline(String),
    #[error("Metal execution error: {0}")]
    Execution(String),
}
