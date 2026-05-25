use super::{NativeTextDiskCacheError, NativeTextPrefixCacheNamespace};
use sha2::{Digest, Sha256};

const NATIVE_TEXT_DISK_CACHE_ROOT_BLOCK_HASH: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

#[cfg(test)]
pub(super) fn native_text_disk_model_hash_from_namespace(
    namespace: &NativeTextPrefixCacheNamespace,
    snapshot_hash: &str,
) -> String {
    native_text_disk_model_hash(NativeTextDiskModelHashParts {
        model_id: &namespace.model_id,
        backend: &namespace.backend,
        family: namespace.family.as_deref(),
        quantization: namespace.quantization.as_deref(),
        repo_id: namespace.repo_id.as_deref(),
        resolved_commit: namespace.resolved_commit.as_deref(),
        profile: namespace.profile.as_deref(),
        snapshot_hash,
    })
}

#[cfg(test)]
pub(super) fn native_text_disk_snapshot_hash_from_namespace(
    namespace: &NativeTextPrefixCacheNamespace,
) -> String {
    let namespace_hash = native_text_disk_namespace_hash(namespace);
    native_text_disk_snapshot_hash(Some(&namespace_hash))
}

pub(super) fn native_text_disk_snapshot_hash(snapshot_identity: Option<&str>) -> String {
    hash_components(
        "kir-ai-native-text-disk-snapshot/v1",
        [("snapshot_identity", snapshot_identity)],
    )
}

pub(super) struct NativeTextDiskModelHashParts<'a> {
    pub(super) model_id: &'a str,
    pub(super) backend: &'a str,
    pub(super) family: Option<&'a str>,
    pub(super) quantization: Option<&'a str>,
    pub(super) repo_id: Option<&'a str>,
    pub(super) resolved_commit: Option<&'a str>,
    pub(super) profile: Option<&'a str>,
    pub(super) snapshot_hash: &'a str,
}

pub(super) fn native_text_disk_model_hash(parts: NativeTextDiskModelHashParts<'_>) -> String {
    hash_components(
        "kir-ai-native-text-disk-model/v1",
        [
            ("model_id", Some(parts.model_id)),
            ("backend", Some(parts.backend)),
            ("family", parts.family),
            ("quantization", parts.quantization),
            ("repo_id", parts.repo_id),
            ("resolved_commit", parts.resolved_commit),
            ("profile", parts.profile),
            ("snapshot_hash", Some(parts.snapshot_hash)),
        ],
    )
}

pub(super) fn native_text_disk_namespace_hash(
    namespace: &NativeTextPrefixCacheNamespace,
) -> String {
    let cache_layout_version = namespace.cache_layout_version.to_string();
    let cache_tokens = namespace.cache_tokens.to_string();
    let max_prefill_tokens = namespace.max_prefill_tokens.to_string();
    hash_components(
        "kir-ai-native-text-disk-namespace/v1",
        [
            ("model_id", Some(namespace.model_id.as_str())),
            ("backend", Some(namespace.backend.as_str())),
            ("family", namespace.family.as_deref()),
            ("quantization", namespace.quantization.as_deref()),
            ("repo_id", namespace.repo_id.as_deref()),
            ("resolved_commit", namespace.resolved_commit.as_deref()),
            ("profile", namespace.profile.as_deref()),
            ("tokenizer_kind", Some(namespace.tokenizer_kind.as_str())),
            ("tokenizer_hash", Some(namespace.tokenizer_hash.as_str())),
            (
                "tokenizer_normalization",
                Some(namespace.tokenizer_normalization.as_str()),
            ),
            (
                "cache_template_id",
                Some(namespace.cache_template_id.as_str()),
            ),
            (
                "chat_template_kwargs_hash",
                namespace.chat_template_kwargs_hash.as_deref(),
            ),
            (
                "adapter_settings",
                Some(namespace.adapter_settings.as_str()),
            ),
            ("cache_key", Some(namespace.cache_key.as_str())),
            ("tool_schema", namespace.tool_schema.as_deref()),
            ("request_mode", Some(namespace.request_mode.as_str())),
            ("cache_layout_version", Some(cache_layout_version.as_str())),
            ("cache_tokens", Some(cache_tokens.as_str())),
            ("max_prefill_tokens", Some(max_prefill_tokens.as_str())),
        ],
    )
}

pub(super) fn native_text_disk_block_hash(
    model_hash: &str,
    namespace_hash: &str,
    previous_block_hash: &str,
    block_start: usize,
    tokens: &[usize],
) -> String {
    let mut hasher = Sha256::new();
    update_hash_value(&mut hasher, Some("kir-ai-native-text-disk-block/v2"));
    update_hash_value(&mut hasher, Some(model_hash));
    update_hash_value(&mut hasher, Some(namespace_hash));
    update_hash_value(&mut hasher, Some(previous_block_hash));
    hasher.update((block_start as u64).to_le_bytes());
    hasher.update((tokens.len() as u64).to_le_bytes());
    for token in tokens {
        hasher.update((*token as u64).to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub(super) fn native_text_disk_previous_block_hash(
    model_hash: &str,
    namespace_hash: &str,
    block_start: usize,
    block_token_count: usize,
    prefix_tokens: &[usize],
) -> Result<String, NativeTextDiskCacheError> {
    if block_token_count == 0 {
        return Err(NativeTextDiskCacheError::integrity(
            "disk cache block token count is zero",
        ));
    }
    if block_start > prefix_tokens.len() {
        return Err(NativeTextDiskCacheError::integrity(
            "disk cache block start exceeds prefix tokens",
        ));
    }
    if !block_start.is_multiple_of(block_token_count) {
        return Err(NativeTextDiskCacheError::integrity(
            "disk cache block start is not block aligned",
        ));
    }

    let mut previous = NATIVE_TEXT_DISK_CACHE_ROOT_BLOCK_HASH.to_owned();
    let mut current_start = 0_usize;
    while current_start < block_start {
        let current_end = current_start
            .checked_add(block_token_count)
            .ok_or_else(|| {
                NativeTextDiskCacheError::integrity("disk cache block range overflow")
            })?;
        let block_tokens = prefix_tokens
            .get(current_start..current_end)
            .ok_or_else(|| {
                NativeTextDiskCacheError::integrity(
                    "disk cache previous block exceeds prefix tokens",
                )
            })?;
        previous = native_text_disk_block_hash(
            model_hash,
            namespace_hash,
            &previous,
            current_start,
            block_tokens,
        );
        current_start = current_end;
    }
    Ok(previous)
}

fn hash_components<'a>(
    label: &str,
    components: impl IntoIterator<Item = (&'static str, Option<&'a str>)>,
) -> String {
    let mut hasher = Sha256::new();
    update_hash_value(&mut hasher, Some(label));
    for (name, value) in components {
        update_hash_value(&mut hasher, Some(name));
        update_hash_value(&mut hasher, value);
    }
    format!("{:x}", hasher.finalize())
}

fn update_hash_value(hasher: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        None => hasher.update([0]),
    }
}
