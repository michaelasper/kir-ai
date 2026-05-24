use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};
use serde_json::Value;
use uuid::Uuid;

/// Function tool call emitted by an assistant message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// OpenAI tool call identifier used by later `tool` role messages.
    pub id: String,
    /// Tool call kind; only function calls are supported.
    #[serde(rename = "type")]
    pub call_type: ToolCallType,
    /// Function name and parsed arguments.
    pub function: ToolCallFunction,
}

/// Generates an OpenAI-style opaque tool call identifier.
pub fn generated_tool_call_id() -> String {
    format!("call_{}", Uuid::new_v4().simple())
}

/// Supported OpenAI tool call type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallType {
    /// Function call tool.
    Function,
}

/// Function payload inside an assistant tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallFunction {
    /// Function name selected by the model.
    pub name: String,
    /// Parsed JSON arguments.
    ///
    /// On the wire OpenAI encodes tool call arguments as a JSON string; this
    /// field stores the parsed value so runtime validation can inspect it.
    #[serde(
        serialize_with = "serialize_tool_call_arguments",
        deserialize_with = "deserialize_tool_call_arguments"
    )]
    pub arguments: Value,
}

/// Function tool declaration supplied on a chat completion request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool kind; only function tools are supported.
    #[serde(rename = "type")]
    pub tool_type: ToolCallType,
    /// Function schema and metadata.
    pub function: FunctionDefinition,
}

impl ToolDefinition {
    /// Builds a function tool declaration with a description and JSON schema parameters.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            tool_type: ToolCallType::Function,
            function: FunctionDefinition {
                name: name.into(),
                description: Some(description.into()),
                parameters,
            },
        }
    }
}

/// Returns tool declarations with deterministic JSON schema member ordering.
///
/// The runtime uses this when prompt/cache identity depends on a schema. It
/// does not validate schema support; request validation is responsible for that.
pub fn canonicalize_tool_schemas(tools: &[ToolDefinition]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .cloned()
        .map(|mut tool| {
            tool.function.parameters = canonicalize_json_value(&tool.function.parameters);
            tool
        })
        .collect()
}

/// Serializes tool declarations after canonicalizing nested JSON values.
pub fn canonical_tool_schema_json(tools: &[ToolDefinition]) -> serde_json::Result<String> {
    let value = serde_json::to_value(tools)?;
    serde_json::to_string(&canonicalize_json_value(&value))
}

/// Recursively sorts object members and `required` arrays in a JSON value.
pub fn canonicalize_json_value(value: &Value) -> Value {
    canonicalize_json_member(None, value)
}

fn canonicalize_json_member(key: Option<&str>, value: &Value) -> Value {
    match value {
        Value::Array(items) => {
            let mut canonical_items = items
                .iter()
                .map(|item| canonicalize_json_member(None, item))
                .collect::<Vec<_>>();
            if key == Some("required") && canonical_items.iter().all(Value::is_string) {
                canonical_items.sort_by(|left, right| left.as_str().cmp(&right.as_str()));
            }
            Value::Array(canonical_items)
        }
        Value::Object(object) => {
            let mut keys = object.keys().collect::<Vec<_>>();
            keys.sort();
            let mut canonical = serde_json::Map::new();
            for key in keys {
                canonical.insert(
                    key.clone(),
                    canonicalize_json_member(Some(key), object.get(key).expect("key exists")),
                );
            }
            Value::Object(canonical)
        }
        _ => value.clone(),
    }
}

/// OpenAI-compatible function tool definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Function name exposed to the model.
    pub name: String,
    /// Optional model-facing description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON schema object for function arguments.
    #[serde(default = "empty_object")]
    pub parameters: Value,
}

/// Client policy for whether and which tool the assistant should call.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ToolChoice {
    /// Let the model decide whether to call a tool.
    #[default]
    Auto,
    /// Require the model to answer without a tool call.
    None,
    /// Require at least one declared tool call.
    Required,
    /// Require a specific declared function tool.
    Function {
        /// Required function name.
        name: String,
    },
}

impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(s) if s == "auto" => Ok(Self::Auto),
            Value::String(s) if s == "none" => Ok(Self::None),
            Value::String(s) if s == "required" => Ok(Self::Required),
            Value::Object(mut obj) => {
                let kind = obj
                    .remove("type")
                    .and_then(|v| v.as_str().map(str::to_owned))
                    .ok_or_else(|| serde::de::Error::custom("tool_choice.type is required"))?;
                if kind != "function" {
                    return Err(serde::de::Error::custom(
                        "only function tool_choice is supported",
                    ));
                }
                let function = obj
                    .remove("function")
                    .and_then(|v| v.as_object().cloned())
                    .ok_or_else(|| serde::de::Error::custom("tool_choice.function is required"))?;
                let name = function
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        serde::de::Error::custom("tool_choice.function.name is required")
                    })?;
                Ok(Self::Function {
                    name: name.to_owned(),
                })
            }
            _ => Err(serde::de::Error::custom("invalid tool_choice")),
        }
    }
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::None => serializer.serialize_str("none"),
            Self::Required => serializer.serialize_str("required"),
            Self::Function { name } => {
                serde_json::json!({"type": "function", "function": {"name": name}})
                    .serialize(serializer)
            }
        }
    }
}

fn empty_object() -> Value {
    serde_json::json!({})
}

fn serialize_tool_call_arguments<S>(arguments: &Value, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let encoded = serde_json::to_string(arguments).map_err(serde::ser::Error::custom)?;
    serializer.serialize_str(&encoded)
}

fn deserialize_tool_call_arguments<'de, D>(deserializer: D) -> Result<Value, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::String(arguments) => serde_json::from_str(&arguments).map_err(D::Error::custom),
        arguments => Ok(arguments),
    }
}
