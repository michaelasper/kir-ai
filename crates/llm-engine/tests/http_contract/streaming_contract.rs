include!("streaming_contract/support.rs");
#[path = "streaming_contract/disconnects.rs"]
mod disconnects;
#[path = "streaming_contract/heartbeat_stall.rs"]
mod heartbeat_stall;
#[path = "streaming_contract/progress_and_tools.rs"]
mod progress_and_tools;
#[path = "streaming_contract/runtime_errors.rs"]
mod runtime_errors;
#[path = "streaming_contract/validation.rs"]
mod validation;
