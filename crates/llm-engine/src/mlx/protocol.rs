use super::MlxToolParserMode;
use llm_backend_contracts::{BackendError, BackendModelMetadata, BackendRequest};
use llm_models::ModelFamily;
use serde_json::Value;
use std::path::Path;

const MLX_TOOL_LOGITS_BIAS_KWARG: &str = "enable_tool_logits_bias";
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
) -> anyhow::Result<&'static [&'static str]> {
    match metadata_family(metadata) {
        Ok(Some(ModelFamily::DeepSeek)) => Ok(MLX_DEEPSEEK_CONTROL_STOP_TOKENS),
        Ok(Some(ModelFamily::Gemma)) => Ok(MLX_GEMMA_CONTROL_STOP_TOKENS),
        Ok(Some(ModelFamily::Llama)) => Ok(MLX_LLAMA_CONTROL_STOP_TOKENS),
        Ok(Some(ModelFamily::Qwen)) | Ok(None) => Ok(MLX_QWEN_CONTROL_STOP_TOKENS),
        Ok(Some(family)) => {
            anyhow::bail!(
                "MLX control stop tokens do not support model family `{}`",
                family.canonical_slug()
            );
        }
        Err(err) => Err(err.into()),
    }
}

pub(super) fn mlx_tool_markup_for_metadata(
    metadata: &BackendModelMetadata,
    snapshot_path: Option<&Path>,
    mode: MlxToolParserMode,
) -> anyhow::Result<MlxToolMarkup> {
    let family = metadata_family(metadata)?;
    match mode {
        MlxToolParserMode::Json => Ok(match family {
            Some(ModelFamily::DeepSeek) => MlxToolMarkup::DeepSeek,
            Some(ModelFamily::Gemma) => MlxToolMarkup::Gemma,
            Some(ModelFamily::Qwen) | Some(ModelFamily::Llama) | None => MlxToolMarkup::Json,
            Some(family) => {
                anyhow::bail!(
                    "MLX JSON tool parser does not support model family `{}`",
                    family.canonical_slug()
                );
            }
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
                if metadata_looks_like_qwen_xml_model(metadata, snapshot_path) {
                    MlxToolMarkup::QwenXml
                } else {
                    MlxToolMarkup::Json
                }
            }
            Some(family) => {
                anyhow::bail!(
                    "MLX auto tool parser does not support model family `{}`",
                    family.canonical_slug()
                );
            }
        }),
    }
}

pub(super) fn mlx_chat_template_kwargs_for_metadata(
    metadata: &BackendModelMetadata,
) -> Option<Value> {
    metadata_family(metadata)
        .ok()
        .flatten()
        .and_then(mlx_chat_template_kwargs_for_family)
}

pub(super) fn mlx_effective_chat_template_kwargs(
    metadata: &BackendModelMetadata,
    request: &BackendRequest,
) -> Option<Value> {
    let mut kwargs = mlx_chat_template_kwargs_for_metadata(metadata);
    if mlx_tool_logits_bias_applies(metadata, request) {
        let value = kwargs.get_or_insert_with(|| Value::Object(Default::default()));
        if let Value::Object(map) = value {
            map.insert(MLX_TOOL_LOGITS_BIAS_KWARG.to_owned(), Value::Bool(true));
        }
    }
    kwargs
}

fn mlx_chat_template_kwargs_for_family(family: ModelFamily) -> Option<Value> {
    family
        .adapter()
        .chat_template_kwargs_json()
        .map(|kwargs| serde_json::from_str(kwargs).expect("static chat template kwargs JSON"))
}

fn mlx_tool_logits_bias_applies(metadata: &BackendModelMetadata, request: &BackendRequest) -> bool {
    if metadata_family(metadata).ok().flatten() != Some(ModelFamily::Qwen) {
        return false;
    }
    request.as_chat().is_some_and(|chat| {
        chat.required_tool_choice.is_some() && !chat.chat_context.tools.is_empty()
    })
}

pub(super) fn mlx_upstream_protocol_for_request(
    metadata: &BackendModelMetadata,
    request: &BackendRequest,
) -> Result<MlxUpstreamProtocol, BackendError> {
    match &request.kind {
        llm_backend_contracts::BackendRequestKind::Chat(_) => {
            Ok(MlxUpstreamProtocol::ChatCompletions)
        }
        llm_backend_contracts::BackendRequestKind::RawCompletion(_) => {
            let family = metadata_family(metadata)
                .map_err(|err| BackendError::unsupported_request(err.to_string()))?;
            Ok(match family {
                Some(ModelFamily::Gemma) => MlxUpstreamProtocol::ChatCompletions,
                Some(ModelFamily::Qwen)
                | Some(ModelFamily::DeepSeek)
                | Some(ModelFamily::Llama)
                | None => MlxUpstreamProtocol::Completions,
                Some(family) => {
                    return Err(BackendError::unsupported_request(format!(
                        "MLX raw completion protocol does not support model family `{}`",
                        family.canonical_slug()
                    )));
                }
            })
        }
        _ => Err(BackendError::unsupported_request(
            "unsupported MLX backend request kind",
        )),
    }
}

fn metadata_family(
    metadata: &BackendModelMetadata,
) -> Result<Option<ModelFamily>, llm_models::ModelFamilyParseError> {
    metadata
        .family
        .as_deref()
        .map(ModelFamily::parse_slug)
        .transpose()
}

fn metadata_looks_like_qwen_xml_model(
    metadata: &BackendModelMetadata,
    snapshot_path: Option<&Path>,
) -> bool {
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
    if let Some(snapshot_path) = snapshot_path {
        return looks_like_qwen35_or_qwen36(&snapshot_path.display().to_string());
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
