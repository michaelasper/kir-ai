include!("tests/support.rs");
include!("tests/fixtures.rs");
#[path = "tests/cache.rs"]
mod cache;
#[path = "tests/cancellation.rs"]
mod cancellation;
#[path = "tests/limits.rs"]
mod limits;
#[path = "tests/prefill.rs"]
mod prefill;
#[path = "tests/sampling_streaming.rs"]
mod sampling_streaming;
