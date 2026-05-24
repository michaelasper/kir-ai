include!("http_contract/support/imports.rs");
include!("http_contract/support/router_support.rs");
include!("http_contract/support/static_backends.rs");
include!("http_contract/support/stream_backends.rs");
include!("http_contract/support/metadata_snapshots.rs");
include!("http_contract/support/hub_server.rs");
include!("http_contract/support/request_helpers.rs");
#[path = "http_contract/admin_contract.rs"]
mod admin_contract;
#[path = "http_contract/chat_contract.rs"]
mod chat_contract;
#[path = "http_contract/completion_contract.rs"]
mod completion_contract;
#[path = "http_contract/core_contract.rs"]
mod core_contract;
#[path = "http_contract/rate_limit_contract.rs"]
mod rate_limit_contract;
#[path = "http_contract/router_builder.rs"]
mod router_builder;
#[path = "http_contract/streaming_contract.rs"]
mod streaming_contract;
