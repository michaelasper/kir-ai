use crate::{
    DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS,
    native_matvec::{NativeTextMatvecBackend, native_text_metal_weight_cache_bytes},
    native_text::{
        NativeTextAdapter, NativeTextCandidateDecision, NativeTextDriver,
        NativeTextNextTokenContext, NativeTextPrefixCache, NativeTextPrefixCacheMetrics,
        NativeTextPrefixCacheNamespace, NativeTextPrefixCacheValue,
        NativeTextPrefixNamespaceContext, native_text_prefix_namespace,
    },
};
use async_trait::async_trait;
use futures::stream::BoxStream;
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    GemmaLayerCache, ModelBackend, NativeMatvecBackend, SafeTensorShardStore, SamplingConfig,
    gemma_cache_count_for_spec, gemma_decode_token_with_cache_with_matvec,
    gemma_layer_caches_for_spec, gemma_prefill_sequence_with_cache_with_matvec,
};
use llm_hub::SnapshotManifest;
use llm_models::GemmaModelSpec;
use llm_tokenizer::HuggingFaceTokenizer;
use serde_json::Value;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};
use tokio_util::sync::CancellationToken;

const DEFAULT_NATIVE_GEMMA_PREFIX_CACHE_BYTES: u64 = 512 * 1024 * 1024;
const NATIVE_GEMMA_PREFIX_CACHE_LAYOUT_VERSION: u32 = 1;

#[derive(Clone)]
pub struct NativeGemmaBackend {
    driver: NativeTextDriver<NativeGemmaAdapter>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeGemmaLoadOptions {
    pub eager_materialize_shards: bool,
    pub metal_weight_cache_bytes: Option<u64>,
    pub warm_metal_weight_cache: bool,
}

#[derive(Clone)]
pub(crate) struct NativeGemmaAdapter {
    model_id: String,
    metadata: BackendModelMetadata,
    spec: GemmaModelSpec,
    store: SafeTensorShardStore,
    matvec: NativeTextMatvecBackend,
    max_prefill_tokens: usize,
    top_k: usize,
    chunk_rows: usize,
    prefix_cache: Arc<NativeGemmaPrefixCache>,
}

type NativeGemmaPrefixCache = NativeTextPrefixCache<GemmaLayerCache>;
type NativeGemmaPrefixCacheMetrics = NativeTextPrefixCacheMetrics;

fn native_gemma_prefix_cache_metrics() -> &'static NativeGemmaPrefixCacheMetrics {
    static METRICS: OnceLock<NativeGemmaPrefixCacheMetrics> = OnceLock::new();
    METRICS.get_or_init(NativeGemmaPrefixCacheMetrics::default)
}

pub(crate) fn native_gemma_prefix_cache_metrics_snapshot() -> Value {
    native_gemma_prefix_cache_metrics().snapshot()
}

impl NativeTextPrefixCacheValue for GemmaLayerCache {
    fn prefix_cache_entry_bytes(hidden: &[f32], caches: &[Self]) -> u64 {
        let hidden_bytes = std::mem::size_of_val(hidden) as u64;
        caches.iter().fold(hidden_bytes, |total, cache| {
            total.saturating_add(match cache {
                GemmaLayerCache::Attention(cache) => {
                    ((cache.key_storage().len() + cache.value_storage().len())
                        * std::mem::size_of::<f32>()) as u64
                }
            })
        })
    }
}

impl NativeGemmaBackend {
    pub fn open(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        Self::open_with_options(model_id, snapshot_path, NativeGemmaLoadOptions::default())
    }

    pub fn open_with_options(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
        options: NativeGemmaLoadOptions,
    ) -> anyhow::Result<Self> {
        let model_id = model_id.into();
        let snapshot_path = snapshot_path.as_ref();
        let cache_namespace = snapshot_path.canonicalize()?.to_string_lossy().into_owned();
        let metadata = native_gemma_metadata(&model_id, snapshot_path)?;
        reject_native_gemma_quantized_snapshot(snapshot_path)?;
        let config_json = std::fs::read_to_string(snapshot_path.join("config.json"))?;
        let spec = GemmaModelSpec::from_config_json(&config_json)?;
        let store = SafeTensorShardStore::open(snapshot_path)?;
        store.index().validate_gemma4_text_weights(&spec)?;
        if options.eager_materialize_shards {
            store.materialize_all_shards().map_err(|err| {
                anyhow::anyhow!("native Gemma safetensors materialization failed: {err}")
            })?;
        }
        let matvec = NativeTextMatvecBackend::system_default(
            native_text_metal_weight_cache_bytes(options.metal_weight_cache_bytes),
            &cache_namespace,
        );
        if options.warm_metal_weight_cache {
            let warmup = matvec.warm_bf16_matrix_cache(&store).map_err(|err| {
                anyhow::anyhow!("native Gemma Metal weight cache warm-up failed: {err}")
            })?;
            tracing::info!(
                candidates = warmup.candidates,
                warmed = warmup.warmed,
                already_resident = warmup.already_resident,
                skipped_budget = warmup.skipped_budget,
                skipped_non_metal = warmup.skipped_non_metal,
                "native Gemma Metal BF16 weight cache warm-up complete"
            );
        }
        let tokenizer = HuggingFaceTokenizer::from_file(snapshot_path.join("tokenizer.json"))?;
        let adapter = NativeGemmaAdapter {
            model_id: model_id.clone(),
            metadata: metadata.clone(),
            spec,
            store,
            matvec,
            max_prefill_tokens: 32,
            top_k: 16,
            chunk_rows: 2048,
            prefix_cache: Arc::new(NativeGemmaPrefixCache::new(
                DEFAULT_NATIVE_GEMMA_PREFIX_CACHE_BYTES,
            )),
        };
        Ok(Self {
            driver: NativeTextDriver::new(
                model_id,
                metadata,
                tokenizer,
                adapter,
                DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS,
            ),
        })
    }

    pub fn with_max_new_tokens(mut self, max_new_tokens: u32) -> Self {
        self.driver = self.driver.with_max_new_tokens(max_new_tokens);
        self
    }

    pub fn with_max_prefill_tokens(mut self, max_prefill_tokens: usize) -> Self {
        self.driver = self.driver.with_max_prefill_tokens(max_prefill_tokens);
        self
    }

    pub(crate) fn into_driver(self) -> NativeTextDriver<NativeGemmaAdapter> {
        self.driver
    }

    #[cfg(test)]
    fn start_decode_session(
        &self,
        context_tokens: &[usize],
        max_new_tokens: u32,
        request: &BackendRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativeGemmaDecodeSession, BackendError> {
        self.driver
            .start_decode_session(context_tokens, max_new_tokens, request, cancellation)
    }

    #[cfg(test)]
    fn next_token_from_hidden(
        &self,
        hidden: &[f32],
        sampling: SamplingConfig,
    ) -> Result<usize, BackendError> {
        self.driver.adapter.next_token_from_hidden(hidden, sampling)
    }
}

impl NativeTextAdapter for NativeGemmaAdapter {
    type DecodeSession = NativeGemmaDecodeSession;
    type LayerCache = GemmaLayerCache;

    fn family_display_name(&self) -> &'static str {
        "Gemma"
    }

    fn worker_label(&self) -> &'static str {
        "native Gemma"
    }

    fn set_max_prefill_tokens(&mut self, max_prefill_tokens: usize) {
        self.max_prefill_tokens = max_prefill_tokens.max(1);
    }

    fn encode_prompt(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        request: &BackendRequest,
    ) -> Result<Vec<u32>, BackendError> {
        tokenizer
            .encode(&request.prompt, false)
            .map_err(|err| BackendError::Other(err.to_string()))
    }

    fn decode_output(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        output_ids: &[u32],
    ) -> Result<String, BackendError> {
        tokenizer
            .decode(output_ids, false)
            .map_err(|err| BackendError::Other(err.to_string()))
    }

    fn observe_candidate(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        _emitted_tokens: &[u32],
        token_id: usize,
    ) -> Result<NativeTextCandidateDecision, BackendError> {
        if token_id == 1
            || tokenizer
                .token_to_id("<eos>")
                .is_some_and(|stop_id| token_id == stop_id as usize)
        {
            return Ok(NativeTextCandidateDecision::Stop);
        }
        Ok(NativeTextCandidateDecision::Emit(token_id))
    }

    fn max_position_embeddings(&self) -> u32 {
        self.spec.max_position_embeddings
    }

    fn max_prefill_tokens(&self) -> usize {
        self.max_prefill_tokens
    }

    fn prefix_cache(&self) -> &NativeTextPrefixCache<GemmaLayerCache> {
        &self.prefix_cache
    }

    fn prefix_cache_metrics(&self) -> &NativeTextPrefixCacheMetrics {
        native_gemma_prefix_cache_metrics()
    }

    fn prefix_cache_namespace(
        &self,
        request: &BackendRequest,
        cache_tokens: usize,
    ) -> NativeTextPrefixCacheNamespace {
        native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
            model_id: &self.model_id,
            metadata: &self.metadata,
            request,
            cache_layout_version: NATIVE_GEMMA_PREFIX_CACHE_LAYOUT_VERSION,
            cache_tokens,
            max_prefill_tokens: self.max_prefill_tokens,
        })
    }

    fn layer_count(&self) -> usize {
        gemma_cache_count_for_spec(&self.spec).unwrap_or(self.spec.num_hidden_layers as usize)
    }

    fn allocate_caches(&self, cache_tokens: usize) -> Result<Vec<GemmaLayerCache>, BackendError> {
        gemma_layer_caches_for_spec(&self.spec, cache_tokens)
            .map_err(|err| BackendError::Other(err.to_string()))
    }

    fn prefill_chunk_with_cache(
        &self,
        token_ids: &[usize],
        caches: &mut [GemmaLayerCache],
    ) -> Result<Vec<Vec<f32>>, BackendError> {
        gemma_prefill_sequence_with_cache_with_matvec(
            &self.store,
            &self.spec,
            token_ids,
            caches,
            &self.matvec,
        )
        .map_err(|err| BackendError::Other(err.to_string()))
    }

    fn make_decode_session(
        &self,
        hidden: Vec<f32>,
        caches: Vec<GemmaLayerCache>,
    ) -> NativeGemmaDecodeSession {
        NativeGemmaDecodeSession { hidden, caches }
    }

    fn hidden<'a>(&self, session: &'a NativeGemmaDecodeSession) -> &'a [f32] {
        session.hidden()
    }

    fn step(
        &self,
        session: &mut NativeGemmaDecodeSession,
        token_id: usize,
    ) -> Result<(), BackendError> {
        session.step(&self.store, &self.spec, &self.matvec, token_id)
    }

    fn next_token_from_hidden(
        &self,
        hidden: &[f32],
        sampling: SamplingConfig,
    ) -> Result<usize, BackendError> {
        NativeTextNextTokenContext {
            store: &self.store,
            spec: (&self.spec).into(),
            top_k: self.top_k,
            chunk_rows: self.chunk_rows,
            matvec: &self.matvec,
            family_display_name: "Gemma",
        }
        .select_next_token(hidden, sampling)
    }
}

pub(crate) struct NativeGemmaDecodeSession {
    hidden: Vec<f32>,
    caches: Vec<GemmaLayerCache>,
}

impl NativeGemmaDecodeSession {
    fn hidden(&self) -> &[f32] {
        &self.hidden
    }

    fn step(
        &mut self,
        store: &SafeTensorShardStore,
        spec: &GemmaModelSpec,
        matvec: &impl NativeMatvecBackend,
        token_id: usize,
    ) -> Result<(), BackendError> {
        self.hidden = gemma_decode_token_with_cache_with_matvec(
            store,
            spec,
            token_id,
            &mut self.caches,
            matvec,
        )
        .map_err(|err| BackendError::Other(err.to_string()))?;
        Ok(())
    }
}

fn native_gemma_metadata(
    model_id: &str,
    snapshot_path: &Path,
) -> anyhow::Result<BackendModelMetadata> {
    let manifest_path = snapshot_path.join("llm-engine-manifest.json");
    let mut metadata =
        BackendModelMetadata::new(model_id.to_owned(), "native-gemma").with_family("gemma");
    metadata.loader = Some("native-metal".to_owned());
    metadata.snapshot_path = Some(PathBuf::from(snapshot_path));
    let manifest_bytes = match std::fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(metadata),
        Err(err) => return Err(err.into()),
    };
    let manifest = serde_json::from_slice::<SnapshotManifest>(&manifest_bytes)?;
    if manifest.family != "gemma" {
        anyhow::bail!(
            "native Gemma backend only supports family `gemma`, not `{}`",
            manifest.family
        );
    }
    if manifest.loader != "native-metal" {
        anyhow::bail!(
            "native Gemma backend only supports loader `native-metal`, not `{}`",
            manifest.loader
        );
    }
    metadata.family = Some(manifest.family.clone());
    metadata.loader = Some(manifest.loader.clone());
    metadata.quantization = Some(manifest.quantization.clone());
    metadata.repo_id = Some(manifest.repo_id.clone());
    metadata.resolved_commit = Some(manifest.resolved_commit.clone());
    metadata.profile = Some(manifest.profile.clone());
    metadata.manifest_digest = Some(manifest.digest());
    Ok(metadata)
}

fn reject_native_gemma_quantized_snapshot(snapshot_path: &Path) -> anyhow::Result<()> {
    let config_path = snapshot_path.join("config.json");
    let Ok(config_json) = std::fs::read_to_string(&config_path) else {
        return Ok(());
    };
    let value: Value = serde_json::from_str(&config_json)?;
    let quantization = value
        .get("quantization")
        .or_else(|| value.get("quantization_config"));
    if let Some(quantization) = quantization {
        anyhow::bail!(
            "native Gemma execution currently supports BF16 safetensors, not quantized Gemma weights ({quantization})"
        );
    }
    Ok(())
}

#[async_trait]
impl ModelBackend for NativeGemmaBackend {
    fn model_id(&self) -> &str {
        self.driver.model_id()
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        self.driver.model_metadata()
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        self.generate_with_cancel(request, CancellationToken::new())
            .await
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        self.driver
            .generate_with_cancel(request, cancellation)
            .await
    }

    fn generate_stream<'a>(
        &'a self,
        request: BackendRequest,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        self.generate_stream_with_cancel(request, CancellationToken::new())
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        self.driver
            .generate_stream_with_cancel(request, cancellation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_backend::BackendCacheContext;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn native_gemma_backend_runs_tiny_prefill_and_selects_tied_lm_head_token() {
        let snapshot = temp_snapshot_dir("native-gemma-prefill");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_tiny_gemma4_decoder_snapshot(&snapshot);
        copy_qwen_tokenizer(snapshot.join("tokenizer.json"));

        let backend =
            NativeGemmaBackend::open("local-gemma", &snapshot).expect("backend opens snapshot");
        let decode = backend
            .start_decode_session(
                &[0, 1],
                4,
                &native_gemma_test_request("local-gemma"),
                &CancellationToken::new(),
            )
            .expect("tiny Gemma prefill runs");
        let candidate = backend
            .next_token_from_hidden(decode.hidden(), SamplingConfig::Greedy)
            .expect("tied lm head selects a token");

        assert_eq!(candidate, 1);
        assert_eq!(backend.model_metadata().backend, "native-gemma");
        assert_eq!(backend.model_metadata().family.as_deref(), Some("gemma"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_gemma_backend_rejects_quantized_mlx_snapshot_explicitly() {
        let snapshot = temp_snapshot_dir("native-gemma-quantized");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_tiny_gemma4_decoder_snapshot(&snapshot);
        copy_qwen_tokenizer(snapshot.join("tokenizer.json"));
        let mut config = serde_json::from_str::<serde_json::Value>(
            &std::fs::read_to_string(snapshot.join("config.json")).expect("config"),
        )
        .expect("config json");
        config["quantization"] = json!({"bits": 4, "group_size": 64, "mode": "affine"});
        std::fs::write(snapshot.join("config.json"), config.to_string()).expect("config");

        let err = match NativeGemmaBackend::open("local-gemma", &snapshot) {
            Err(err) => err,
            Ok(_) => panic!("quantized native Gemma fails explicitly"),
        };

        assert!(err.to_string().contains("not quantized Gemma weights"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_gemma_backend_accepts_native_metal_cache_options_with_cpu_fallback() {
        let snapshot = temp_snapshot_dir("native-gemma-metal-options");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_tiny_gemma4_decoder_snapshot(&snapshot);
        copy_qwen_tokenizer(snapshot.join("tokenizer.json"));

        let backend = NativeGemmaBackend::open_with_options(
            "local-gemma",
            &snapshot,
            NativeGemmaLoadOptions {
                metal_weight_cache_bytes: Some(0),
                warm_metal_weight_cache: true,
                ..NativeGemmaLoadOptions::default()
            },
        )
        .expect("backend opens with native-metal cache options");

        assert_eq!(backend.model_metadata().backend, "native-gemma");
        assert_eq!(
            backend.model_metadata().loader.as_deref(),
            Some("native-metal")
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[tokio::test]
    #[ignore = "set KIR_AI_GEMMA_BF16_SNAPSHOT to a local BF16 Gemma 4 text snapshot"]
    async fn native_gemma_real_bf16_snapshot_smoke_generates_one_token() {
        let snapshot = std::env::var_os("KIR_AI_GEMMA_BF16_SNAPSHOT")
            .map(PathBuf::from)
            .expect("KIR_AI_GEMMA_BF16_SNAPSHOT must point at a local BF16 Gemma 4 snapshot");
        let backend = NativeGemmaBackend::open("local-gemma", &snapshot)
            .expect("real BF16 Gemma snapshot opens")
            .with_max_new_tokens(1);
        let output = backend
            .generate(BackendRequest {
                model: "local-gemma".to_owned(),
                prompt: "Hello".to_owned(),
                chat_context: None,
                max_tokens: Some(1),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: false,
                cache_context: BackendCacheContext::default(),
            })
            .await
            .expect("real BF16 Gemma snapshot generates");

        assert!(output.prompt_tokens > 0);
        assert!(output.completion_tokens <= 1);
    }

    fn native_gemma_test_request(model: &str) -> BackendRequest {
        BackendRequest {
            model: model.to_owned(),
            prompt: "hello".to_owned(),
            chat_context: None,
            max_tokens: Some(1),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::default(),
        }
    }

    fn write_tiny_gemma4_decoder_snapshot(root: &Path) {
        std::fs::write(
            root.join("config.json"),
            json!({
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "text_config": {
                    "attention_bias": false,
                    "attention_dropout": 0.0,
                    "attention_k_eq_v": false,
                    "bos_token_id": 2,
                    "dtype": "bfloat16",
                    "enable_moe_block": false,
                    "global_head_dim": null,
                    "head_dim": 2,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2,
                    "hidden_size_per_layer_input": 0,
                    "intermediate_size": 1,
                    "layer_types": ["sliding_attention"],
                    "max_position_embeddings": 8,
                    "model_type": "gemma4_text",
                    "num_attention_heads": 1,
                    "num_global_key_value_heads": null,
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 1,
                    "num_kv_shared_layers": 0,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                        "full_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000.0},
                        "sliding_attention": {"rope_theta": 10000.0}
                    },
                    "sliding_window": 2,
                    "tie_word_embeddings": true,
                    "use_double_wide_mlp": false,
                    "vocab_size": 3,
                    "vocab_size_per_layer_input": 3
                },
                "tie_word_embeddings": true
            })
            .to_string(),
        )
        .expect("config");
        let tensors = [
            (
                "model.language_model.embed_tokens.weight",
                vec![3, 2],
                vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0],
            ),
            ("model.language_model.norm.weight", vec![2], vec![1.0, 1.0]),
            (
                "model.language_model.layers.0.input_layernorm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.0.self_attn.q_proj.weight",
                vec![2, 2],
                vec![0.0, 0.0, 0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.self_attn.k_proj.weight",
                vec![2, 2],
                vec![0.0, 0.0, 0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.self_attn.v_proj.weight",
                vec![2, 2],
                vec![0.0, 0.0, 0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.self_attn.q_norm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.0.self_attn.k_norm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.0.self_attn.o_proj.weight",
                vec![2, 2],
                vec![0.0, 0.0, 0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.post_attention_layernorm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.0.pre_feedforward_layernorm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.0.mlp.gate_proj.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.up_proj.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.down_proj.weight",
                vec![2, 1],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.post_feedforward_layernorm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.0.layer_scalar",
                vec![1],
                vec![1.0],
            ),
        ];
        let safetensors = tiny_owned_multi_safetensors_bf16(&tensors);
        std::fs::write(root.join("model.safetensors"), &safetensors).expect("safetensors");
        let weight_map = tensors
            .iter()
            .map(|(tensor, _, _)| {
                (
                    (*tensor).to_owned(),
                    serde_json::Value::String("model.safetensors".to_owned()),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        std::fs::write(
            root.join("model.safetensors.index.json"),
            json!({
                "metadata": {"total_size": safetensors.len()},
                "weight_map": weight_map
            })
            .to_string(),
        )
        .expect("index");
    }

    fn tiny_owned_multi_safetensors_bf16(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for value in values {
                data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
            let end = data.len();
            header.insert(
                (*name).to_owned(),
                json!({
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [start, end]
                }),
            );
        }
        let header = serde_json::Value::Object(header).to_string();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        bytes
    }

    fn copy_qwen_tokenizer(destination: impl AsRef<Path>) {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/qwen36")
            .join("tokenizer.json");
        std::fs::copy(&source, destination).expect("copy tokenizer");
    }

    fn temp_snapshot_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("kir-ai-{name}-{nanos}"))
    }
}
