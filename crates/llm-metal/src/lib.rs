mod device;

pub mod buffers {
    pub use super::device::Bf16MatrixBuffer;
}

pub mod kernels {
    pub use super::device::{ArgmaxResult, MetalDevice, TopKResult};
}

pub use device::*;
