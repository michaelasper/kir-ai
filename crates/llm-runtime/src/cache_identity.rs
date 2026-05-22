use crate::RuntimeError;
use crate::runtime::Runtime;
use llm_api::{ChatCompletionRequest, ChatMessage, ChatRole, CompletionRequest};
use llm_backend::{BackendCacheContext, ModelBackend};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestCacheIdentity {
    pub prompt_hash: String,
    pub cache_key: String,
    pub cache_template_id: String,
    pub model_family: Option<String>,
    pub tool_schema_hash: Option<String>,
    pub system_prompt_hash: Option<String>,
    pub chat_template_kwargs_hash: Option<String>,
    pub stable_prefix_key: Option<String>,
}

impl RequestCacheIdentity {
    fn chat(
        model_family: Option<String>,
        cache_context: &BackendCacheContext,
        prompt: &str,
        messages: &[ChatMessage],
    ) -> Self {
        let tool_schema_hash = cache_context.tool_schema.as_deref().map(hash_str);
        let system_prompt_hash = hash_system_prompt(messages);
        let chat_template_kwargs_hash = cache_context.chat_template_kwargs.as_deref().map(hash_str);
        let stable_prefix_key = Some(stable_prefix_key([
            ("model-family", model_family.as_deref()),
            (
                "cache-template-id",
                Some(cache_context.cache_template_id.as_str()),
            ),
            ("tool-schema-hash", tool_schema_hash.as_deref()),
            ("system-prompt-hash", system_prompt_hash.as_deref()),
            (
                "chat-template-kwargs-hash",
                chat_template_kwargs_hash.as_deref(),
            ),
        ]));
        Self {
            prompt_hash: hash_str(prompt),
            cache_key: cache_context.key.as_str().to_owned(),
            cache_template_id: cache_context.cache_template_id.clone(),
            model_family,
            tool_schema_hash,
            system_prompt_hash,
            chat_template_kwargs_hash,
            stable_prefix_key,
        }
    }

    fn raw_completion(cache_context: &BackendCacheContext, prompt: &str) -> Self {
        Self {
            prompt_hash: hash_str(prompt),
            cache_key: cache_context.key.as_str().to_owned(),
            cache_template_id: cache_context.cache_template_id.clone(),
            model_family: None,
            tool_schema_hash: None,
            system_prompt_hash: None,
            chat_template_kwargs_hash: None,
            stable_prefix_key: None,
        }
    }
}

impl<B> Runtime<B>
where
    B: ModelBackend,
{
    pub fn chat_request_cache_identity(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<RequestCacheIdentity, RuntimeError> {
        let adapter = self.chat_adapter()?;
        let (cache_context, prompt, _) = self.prepare_chat_backend(adapter, request)?;
        let metadata = self.backend.model_metadata();
        Ok(RequestCacheIdentity::chat(
            metadata.family,
            &cache_context,
            &prompt,
            &request.messages,
        ))
    }

    pub fn completion_request_cache_identity(
        &self,
        request: &CompletionRequest,
    ) -> RequestCacheIdentity {
        RequestCacheIdentity::raw_completion(&BackendCacheContext::raw_prompt(), &request.prompt)
    }
}

fn hash_system_prompt(messages: &[ChatMessage]) -> Option<String> {
    let system_messages = messages
        .iter()
        .filter(|message| message.role == ChatRole::System)
        .map(|message| {
            json!({
                "content": message.content.as_deref(),
                "name": message.name.as_deref(),
            })
        })
        .collect::<Vec<_>>();
    if system_messages.is_empty() {
        return None;
    }
    Some(hash_json(&json!({
        "version": "system-prompt/v1",
        "messages": system_messages,
    })))
}

fn hash_json(value: &Value) -> String {
    hash_str(&value.to_string())
}

fn hash_str(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

fn stable_prefix_key<'a>(
    components: impl IntoIterator<Item = (&'static str, Option<&'a str>)>,
) -> String {
    let mut hasher = Sha256::new();
    update_hash_component(
        &mut hasher,
        "stable-prefix-version",
        Some("request-cache-prefix/v1"),
    );
    for (name, value) in components {
        update_hash_component(&mut hasher, name, value);
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn update_hash_component(hasher: &mut Sha256, name: &str, value: Option<&str>) {
    hasher.update((name.len() as u64).to_le_bytes());
    hasher.update(name.as_bytes());
    match value {
        Some(value) => {
            hasher.update([1]);
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        None => hasher.update([0]),
    }
}
