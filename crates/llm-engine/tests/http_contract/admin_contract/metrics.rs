include!("metrics/support.rs");
#[path = "metrics/active_no_progress.rs"]
mod active_no_progress;
#[path = "metrics/auth_and_cache.rs"]
mod auth_and_cache;
#[path = "metrics/decode_phase.rs"]
mod decode_phase;
#[path = "metrics/endpoint.rs"]
mod endpoint;
#[path = "metrics/inference.rs"]
mod inference;
#[path = "metrics/mlx_sidecar.rs"]
mod mlx_sidecar;
#[path = "metrics/model_store.rs"]
mod model_store;
#[path = "metrics/stream_timing.rs"]
mod stream_timing;
