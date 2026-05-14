use super::MlxToolParserMode;
use llm_backend::{BackendModelMetadata, BackendRequest};
use llm_models::ModelFamily;
use serde_json::Value;

pub(super) const MLX_QWEN_CONTROL_STOP_TOKENS: &[&str] = &["<|im_end|>", "<|endoftext|>"];
pub(super) const MLX_DEEPSEEK_CONTROL_STOP_TOKENS: &[&str] =
    &["<｜end▁of▁sentence｜>", "<｜User｜>", "<|endoftext|>"];
const MLX_GEMMA_CONTROL_STOP_TOKENS: &[&str] =
    &["<turn|>", "<|tool_response>", "<eos>", "<|endoftext|>"];
const MLX_LLAMA_CONTROL_STOP_TOKENS: &[&str] = &["<|eot_id|>", "<|end_of_text|>"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MlxUpstreamProtocol {
    Completions,
    ChatCompletions,
}

impl MlxUpstreamProtocol {
    pub(super) fn endpoint_suffix(self) -> &'static str {
        match self {
            Self::Completions => "completions",
            Self::ChatCompletions => "chat/completions",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MlxToolMarkup {
    Json,
    QwenXml,
    DeepSeek,
    Gemma,
}

pub(super) fn mlx_control_stop_tokens_for_metadata(
    metadata: &BackendModelMetadata,
) -> &'static [&'static str] {
    match metadata_family(metadata) {
        Some(ModelFamily::DeepSeek) => MLX_DEEPSEEK_CONTROL_STOP_TOKENS,
        Some(ModelFamily::Gemma) => MLX_GEMMA_CONTROL_STOP_TOKENS,
        Some(ModelFamily::Llama) => MLX_LLAMA_CONTROL_STOP_TOKENS,
        Some(ModelFamily::Qwen) | None => MLX_QWEN_CONTROL_STOP_TOKENS,
    }
}

pub(super) fn mlx_tool_markup_for_metadata(
    metadata: &BackendModelMetadata,
    mode: MlxToolParserMode,
) -> anyhow::Result<MlxToolMarkup> {
    let family = metadata_family(metadata);
    match mode {
        MlxToolParserMode::Json => Ok(match family {
            Some(ModelFamily::DeepSeek) => MlxToolMarkup::DeepSeek,
            Some(ModelFamily::Gemma) => MlxToolMarkup::Gemma,
            Some(ModelFamily::Qwen) | Some(ModelFamily::Llama) | None => MlxToolMarkup::Json,
        }),
        MlxToolParserMode::QwenXml => {
            if !matches!(family, Some(ModelFamily::Qwen) | None) {
                anyhow::bail!(
                    "--mlx-tool-parser qwen-xml is only supported for Qwen or unknown-family MLX metadata"
                );
            }
            Ok(MlxToolMarkup::QwenXml)
        }
        MlxToolParserMode::Auto => Ok(match family {
            Some(ModelFamily::DeepSeek) => MlxToolMarkup::DeepSeek,
            Some(ModelFamily::Gemma) => MlxToolMarkup::Gemma,
            Some(ModelFamily::Llama) => MlxToolMarkup::Json,
            Some(ModelFamily::Qwen) | None => {
                if metadata_looks_like_qwen_xml_model(metadata) {
                    MlxToolMarkup::QwenXml
                } else {
                    MlxToolMarkup::Json
                }
            }
        }),
    }
}

pub(super) fn mlx_chat_template_kwargs_for_metadata(
    metadata: &BackendModelMetadata,
) -> Option<Value> {
    match metadata_family(metadata) {
        Some(ModelFamily::Qwen) => Some(serde_json::json!({"enable_thinking": false})),
        Some(ModelFamily::DeepSeek)
        | Some(ModelFamily::Gemma)
        | Some(ModelFamily::Llama)
        | None => None,
    }
}

pub(super) fn mlx_effective_chat_template_kwargs(
    metadata: &BackendModelMetadata,
    request: &BackendRequest,
) -> Option<Value> {
    request
        .cache_context
        .chat_template_kwargs
        .as_deref()
        .and_then(|kwargs| serde_json::from_str(kwargs).ok())
        .or_else(|| mlx_chat_template_kwargs_for_metadata(metadata))
}

pub(super) fn mlx_upstream_protocol_for_request(
    metadata: &BackendModelMetadata,
    request: &BackendRequest,
) -> MlxUpstreamProtocol {
    if request.conversation_mode {
        if request.chat_context.is_none()
            && matches!(metadata_family(metadata), Some(ModelFamily::Llama))
        {
            return MlxUpstreamProtocol::Completions;
        }
        return MlxUpstreamProtocol::ChatCompletions;
    }
    match metadata_family(metadata) {
        Some(ModelFamily::Gemma) => MlxUpstreamProtocol::ChatCompletions,
        Some(ModelFamily::Qwen) | Some(ModelFamily::DeepSeek) | Some(ModelFamily::Llama) | None => {
            MlxUpstreamProtocol::Completions
        }
    }
}

fn metadata_family(metadata: &BackendModelMetadata) -> Option<ModelFamily> {
    metadata
        .family
        .as_deref()
        .and_then(|family| ModelFamily::parse_slug(family).ok())
}

fn metadata_looks_like_qwen_xml_model(metadata: &BackendModelMetadata) -> bool {
    if metadata
        .repo_id
        .as_deref()
        .is_some_and(looks_like_qwen35_or_qwen36)
        || metadata
            .profile
            .as_deref()
            .is_some_and(looks_like_qwen35_or_qwen36)
    {
        return true;
    }
    if let Some(snapshot_path) = metadata
        .snapshot_path
        .as_ref()
        .map(|path| path.display().to_string())
    {
        return looks_like_qwen35_or_qwen36(&snapshot_path);
    }
    false
}

fn looks_like_qwen35_or_qwen36(value: &str) -> bool {
    let normalized = value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    normalized.contains("qwen35") || normalized.contains("qwen36")
}
