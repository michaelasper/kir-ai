include!("openai_contract/support.rs");
#[path = "openai_contract/message_validation.rs"]
mod message_validation;
#[path = "openai_contract/request_limits.rs"]
mod request_limits;
#[path = "openai_contract/response_shapes.rs"]
mod response_shapes;
#[path = "openai_contract/sampling_controls.rs"]
mod sampling_controls;
#[path = "openai_contract/tool_schema.rs"]
mod tool_schema;
