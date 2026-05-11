mod core;

#[cfg(feature = "test-utils")]
pub mod protocol_test {
    pub use super::core::ProtocolTestBackend;
}

pub mod traits {
    pub use super::core::{
        BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk, ModelBackend,
        SamplingConfig,
    };
}

pub use core::*;
