//! OpenAI-compatible API shapes shared by the server, runtime, and tests.
//!
//! This crate intentionally contains protocol data and validation rules only.
//! Runtime code consumes `Validated<T>` requests from here so unsupported or
//! malformed OpenAI fields fail before prompt rendering or backend scheduling.

pub mod error;
pub mod limits;
pub mod request;
pub mod response;
pub mod tool_schema;
pub mod types;
pub mod validation;

pub use error::*;
pub use limits::*;
pub use request::*;
pub use response::*;
pub use tool_schema::*;
pub use types::*;
pub use validation::*;
