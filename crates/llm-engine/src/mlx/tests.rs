include!("tests/support.rs");
#[path = "tests/parser_stop_boundaries.rs"]
mod parser_stop_boundaries;
#[path = "tests/request_usage_metrics.rs"]
mod request_usage_metrics;
#[path = "tests/sse_qwen_parser.rs"]
mod sse_qwen_parser;
#[path = "tests/streaming_tools.rs"]
mod streaming_tools;
#[path = "tests/structured_requests.rs"]
mod structured_requests;
#[path = "tests/transport_boundary.rs"]
mod transport_boundary;
#[path = "tests/validation_health.rs"]
mod validation_health;
