#[cfg(feature = "native-gemma")]
mod fs_util;
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
mod kv_sync;
#[cfg(feature = "native-gemma")]
mod native_gemma;
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
mod native_matvec;
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
mod native_metrics;
#[cfg(feature = "native-qwen")]
mod native_qwen;
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
mod native_text;
mod snapshot;
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
mod sync_ext;
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
mod warm_order;

pub use llm_util::defaults::DEFAULT_MODEL_ID;

#[cfg(feature = "native-gemma")]
pub use native_gemma::*;
#[cfg(feature = "native-qwen")]
pub use native_qwen::*;
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
pub use native_text::*;
pub use snapshot::{ResolvedSnapshotBackend, SnapshotBackendLoader, parse_snapshot_model_family};
