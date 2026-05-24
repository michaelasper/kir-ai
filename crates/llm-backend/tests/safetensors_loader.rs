use llm_backend::native::{
    CpuNativeMatvecBackend, InferenceScratchpad, MathError, NativeMatvecBackend, QwenLayerCache,
    QwenMoeDims, QwenMoeRouterProbe, SafeTensorShardStore, TensorLoadError, TopKWeight,
    qwen_decode_token_with_cache, qwen_layer_caches_for_spec, qwen_layer_moe_forward,
    qwen_layer0_moe_router, qwen_prefill_sequence_with_cache,
};
use llm_models::{AttentionKind, ModelFamily, QwenModelSpec};
use std::fmt;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicUsize, Ordering},
};
use tracing::field::{Field, Visit};
use tracing::{Event, Id, Metadata, Subscriber, span};

#[path = "safetensors_loader/qwen_attention.rs"]
mod qwen_attention;
#[path = "safetensors_loader/qwen_moe.rs"]
mod qwen_moe;

#[derive(Default)]
struct RecordingMatvecBackend {
    bf16_row_major_calls: AtomicUsize,
    bf16_range_calls: AtomicUsize,
    conv1d_calls: AtomicUsize,
    dense_f32_calls: AtomicUsize,
    recurrent_cache_update_calls: AtomicUsize,
    softmax_top_k_calls: AtomicUsize,
}

#[derive(Clone, Debug)]
struct RecordedEvent {
    fields: Vec<(String, String)>,
}

impl RecordedEvent {
    fn has_field(&self, name: &str, value: &str) -> bool {
        self.fields
            .iter()
            .any(|(field, recorded)| field == name && recorded == value)
    }
}

static TRACE_EVENTS: OnceLock<Arc<Mutex<Vec<RecordedEvent>>>> = OnceLock::new();

struct TraceCapture {
    events: Arc<Mutex<Vec<RecordedEvent>>>,
}

impl TraceCapture {
    fn start() -> Self {
        let events = Arc::clone(TRACE_EVENTS.get_or_init(|| {
            let events = Arc::new(Mutex::new(Vec::new()));
            let subscriber = RecordingSubscriber {
                events: Arc::clone(&events),
            };
            tracing::subscriber::set_global_default(subscriber)
                .expect("trace test subscriber installs once");
            events
        }));
        events.lock().expect("recorded events lock").clear();
        tracing::callsite::rebuild_interest_cache();
        Self { events }
    }

    fn events(&self) -> Vec<RecordedEvent> {
        self.events.lock().expect("recorded events lock").clone()
    }
}

struct RecordingSubscriber {
    events: Arc<Mutex<Vec<RecordedEvent>>>,
}

impl Subscriber for RecordingSubscriber {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn register_callsite(
        &self,
        _metadata: &'static Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::always()
    }

    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::TRACE)
    }

    fn new_span(&self, _span: &span::Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _span: &Id, _values: &span::Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut visitor = FieldRecorder::default();
        event.record(&mut visitor);
        self.events
            .lock()
            .expect("recorded events lock")
            .push(RecordedEvent {
                fields: visitor.fields,
            });
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

#[derive(Default)]
struct FieldRecorder {
    fields: Vec<(String, String)>,
}

impl FieldRecorder {
    fn record_value(&mut self, field: &Field, value: String) {
        self.fields.push((field.name().to_owned(), value));
    }
}

impl Visit for FieldRecorder {
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.record_value(field, value.to_string());
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.record_value(field, value.to_string());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.record_value(field, value.to_string());
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.record_value(field, value.to_owned());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.record_value(field, format!("{value:?}"));
    }
}

impl RecordingMatvecBackend {
    fn bf16_row_major_calls(&self) -> usize {
        self.bf16_row_major_calls.load(Ordering::Relaxed)
    }

    fn bf16_range_calls(&self) -> usize {
        self.bf16_range_calls.load(Ordering::Relaxed)
    }

    fn conv1d_calls(&self) -> usize {
        self.conv1d_calls.load(Ordering::Relaxed)
    }

    fn dense_f32_calls(&self) -> usize {
        self.dense_f32_calls.load(Ordering::Relaxed)
    }

    fn recurrent_cache_update_calls(&self) -> usize {
        self.recurrent_cache_update_calls.load(Ordering::Relaxed)
    }

    fn softmax_top_k_calls(&self) -> usize {
        self.softmax_top_k_calls.load(Ordering::Relaxed)
    }
}

#[test]
fn safetensors_f32_range_cached_emits_cache_trace_metadata() {
    let root = temp_snapshot_dir("f32-cache-trace");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 12 },
            "weight_map": { "embed.weight": "model-00001-of-00001.safetensors" }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("model-00001-of-00001.safetensors"),
        tiny_safetensors_bf16("embed.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .expect("shard");

    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let capture = TraceCapture::start();
    let first = store
        .bf16_tensor_f32_range_cached("embed.weight", 0, 6)
        .expect("first cached read");
    let second = store
        .bf16_tensor_f32_range_cached("embed.weight", 0, 6)
        .expect("second cached read");
    let events = capture.events();

    assert_eq!(first, second);
    assert!(
        events.iter().any(|event| {
            event.has_field("operation", "safetensors_f32_cache_lookup")
                && event.has_field("cache", "range")
                && event.has_field("cache_hit", "false")
                && event.has_field("tensor", "embed.weight")
        }),
        "first cached read should emit F32 cache miss metadata, got {events:?}"
    );
    assert!(
        events.iter().any(|event| {
            event.has_field("operation", "safetensors_f32_cache_lookup")
                && event.has_field("cache", "range")
                && event.has_field("cache_hit", "true")
                && event.has_field("tensor", "embed.weight")
        }),
        "second cached read should emit F32 cache hit metadata, got {events:?}"
    );
    std::fs::remove_dir_all(root).ok();
}

impl NativeMatvecBackend for RecordingMatvecBackend {
    async fn bf16_matvec_row_major_f32_in_place(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), TensorLoadError> {
        self.bf16_row_major_calls.fetch_add(1, Ordering::Relaxed);
        CpuNativeMatvecBackend
            .bf16_matvec_row_major_f32_in_place(store, tensor, input, output)
            .await
    }

    async fn bf16_matvec_rows_f32_in_place(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        chunk_rows: usize,
        output: &mut [f32],
    ) -> Result<(), TensorLoadError> {
        CpuNativeMatvecBackend
            .bf16_matvec_rows_f32_in_place(store, tensor, input, chunk_rows, output)
            .await
    }

    async fn bf16_matvec_range_row_major_f32_in_place(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        element_offset: usize,
        rows: usize,
        columns: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), TensorLoadError> {
        self.bf16_range_calls.fetch_add(1, Ordering::Relaxed);
        CpuNativeMatvecBackend
            .bf16_matvec_range_row_major_f32_in_place(
                store,
                tensor,
                element_offset,
                rows,
                columns,
                input,
                output,
            )
            .await
    }

    async fn matvec_row_major_f32_in_place(
        &self,
        input: &[f32],
        weights: &[f32],
        rows: usize,
        columns: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        self.dense_f32_calls.fetch_add(1, Ordering::Relaxed);
        CpuNativeMatvecBackend
            .matvec_row_major_f32_in_place(input, weights, rows, columns, output)
            .await
    }

    async fn rms_norm_one_centered_f32_in_place(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        CpuNativeMatvecBackend
            .rms_norm_one_centered_f32_in_place(input, weight, eps, output)
            .await
    }

    async fn softmax_f32_in_place(
        &self,
        scores: &[f32],
        output: &mut [f32],
    ) -> Result<(), MathError> {
        CpuNativeMatvecBackend
            .softmax_f32_in_place(scores, output)
            .await
    }

    async fn linear_attention_conv1d_silu_f32_in_place(
        &self,
        window: &[f32],
        weights: &[f32],
        conv_dim: usize,
        kernel_size: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        self.conv1d_calls.fetch_add(1, Ordering::Relaxed);
        CpuNativeMatvecBackend
            .linear_attention_conv1d_silu_f32_in_place(
                window,
                weights,
                conv_dim,
                kernel_size,
                output,
            )
            .await
    }

    async fn weighted_sum_f32_in_place(
        &self,
        values: &[f32],
        weights: &[f32],
        vector_len: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        CpuNativeMatvecBackend
            .weighted_sum_f32_in_place(values, weights, vector_len, output)
            .await
    }

    async fn linear_attention_recurrent_update_f32_in_place(
        &self,
        state: &[f32],
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        CpuNativeMatvecBackend
            .linear_attention_recurrent_update_f32_in_place(
                state,
                key,
                value,
                memory,
                beta,
                decay,
                key_head_dim,
                value_head_dim,
                output,
            )
            .await
    }

    async fn select_head_rows_f32_in_place(
        &self,
        values: &[f32],
        row_count: usize,
        row_len: usize,
        head_start: usize,
        head_len: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        CpuNativeMatvecBackend
            .select_head_rows_f32_in_place(values, row_count, row_len, head_start, head_len, output)
            .await
    }

    async fn linear_attention_recurrent_cache_update_f32_in_place(
        &self,
        cache: &llm_backend::native::LinearAttentionCache,
        state_start: usize,
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        self.recurrent_cache_update_calls
            .fetch_add(1, Ordering::Relaxed);
        CpuNativeMatvecBackend
            .linear_attention_recurrent_cache_update_f32_in_place(
                cache,
                state_start,
                key,
                value,
                memory,
                beta,
                decay,
                key_head_dim,
                value_head_dim,
                output,
            )
            .await
    }

    async fn softmax_top_k_f32(
        &self,
        logits: &[f32],
        top_k: usize,
    ) -> Result<Vec<TopKWeight>, MathError> {
        self.softmax_top_k_calls.fetch_add(1, Ordering::Relaxed);
        CpuNativeMatvecBackend
            .softmax_top_k_f32(logits, top_k)
            .await
    }
}

fn write_tiny_linear_decoder_snapshot(root: &std::path::Path) {
    std::fs::create_dir_all(root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 256 },
            "weight_map": {
                "model.language_model.embed_tokens.weight": "embed.safetensors",
                "model.language_model.layers.0.input_layernorm.weight": "input_norm.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "qkv.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_z.weight": "z.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_b.weight": "b.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight": "a.safetensors",
                "model.language_model.layers.0.linear_attn.dt_bias": "dt.safetensors",
                "model.language_model.layers.0.linear_attn.A_log": "a_log.safetensors",
                "model.language_model.layers.0.linear_attn.conv1d.weight": "conv.safetensors",
                "model.language_model.layers.0.linear_attn.norm.weight": "attn_norm.safetensors",
                "model.language_model.layers.0.linear_attn.out_proj.weight": "out.safetensors",
                "model.language_model.layers.0.post_attention_layernorm.weight": "post_norm.safetensors",
                "model.language_model.layers.0.mlp.gate.weight": "router.safetensors",
                "model.language_model.layers.0.mlp.experts.gate_up_proj": "experts_gate_up.safetensors",
                "model.language_model.layers.0.mlp.experts.down_proj": "experts_down.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": "shared_gate.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": "shared_up.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": "shared_down.safetensors",
                "model.language_model.layers.0.mlp.shared_expert_gate.weight": "shared_expert_gate.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "embed.safetensors",
            "model.language_model.embed_tokens.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "input_norm.safetensors",
            "model.language_model.layers.0.input_layernorm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "qkv.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0],
        ),
        (
            "z.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "b.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "a.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "dt.safetensors",
            "model.language_model.layers.0.linear_attn.dt_bias",
            vec![1],
            vec![0.0],
        ),
        (
            "a_log.safetensors",
            "model.language_model.layers.0.linear_attn.A_log",
            vec![1],
            vec![0.0],
        ),
        (
            "conv.safetensors",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            vec![4, 1],
            vec![1.0, 1.0, 1.0, 1.0],
        ),
        (
            "attn_norm.safetensors",
            "model.language_model.layers.0.linear_attn.norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "out.safetensors",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "post_norm.safetensors",
            "model.language_model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "router.safetensors",
            "model.language_model.layers.0.mlp.gate.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "experts_gate_up.safetensors",
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        (
            "experts_down.safetensors",
            "model.language_model.layers.0.mlp.experts.down_proj",
            vec![2, 1],
            vec![0.0, 0.0],
        ),
        (
            "shared_gate.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "shared_up.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "shared_down.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            vec![2, 1],
            vec![0.0, 0.0],
        ),
        (
            "shared_expert_gate.safetensors",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
}

fn write_tiny_moe_forward_snapshot(root: &std::path::Path) {
    std::fs::create_dir_all(root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 104 },
            "weight_map": {
                "model.language_model.layers.0.mlp.gate.weight": "router.safetensors",
                "model.language_model.layers.0.mlp.experts.gate_up_proj": "gate_up.safetensors",
                "model.language_model.layers.0.mlp.experts.down_proj": "down.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": "shared_gate.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": "shared_up.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": "shared_down.safetensors",
                "model.language_model.layers.0.mlp.shared_expert_gate.weight": "shared_expert_gate.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "router.safetensors",
            "model.language_model.layers.0.mlp.gate.weight",
            vec![3, 2],
            vec![1.0, 0.0, 0.0, 2.0, 1.0, 1.0],
        ),
        (
            "gate_up.safetensors",
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            vec![2, 2, 2],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        ),
        (
            "down.safetensors",
            "model.language_model.layers.0.mlp.experts.down_proj",
            vec![2, 2, 1],
            vec![1.0, 2.0, 3.0, 4.0],
        ),
        (
            "shared_gate.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "shared_up.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "shared_down.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            vec![2, 1],
            vec![0.0, 0.0],
        ),
        (
            "shared_expert_gate.safetensors",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
}

fn temp_snapshot_dir(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("llm-backend-{label}-{}", std::process::id()))
}

fn tiny_qwen_spec(kind: AttentionKind) -> QwenModelSpec {
    QwenModelSpec {
        family: ModelFamily::Qwen,
        architecture: "Qwen3_5MoeForConditionalGeneration".to_owned(),
        model_type: "qwen3_5_moe".to_owned(),
        text_model_type: "qwen3_5_moe_text".to_owned(),
        hidden_size: 2,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: false,
        rope_theta: 10_000.0,
        partial_rotary_factor: 1.0,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        linear_num_key_heads: 1,
        linear_num_value_heads: 1,
        linear_key_head_dim: 1,
        linear_value_head_dim: 2,
        linear_conv_kernel_dim: 1,
        num_experts: 1,
        num_experts_per_tok: 1,
        moe_intermediate_size: 1,
        shared_expert_intermediate_size: 1,
        max_position_embeddings: 128,
        vocab_size: 2,
        layer_kinds: vec![kind],
    }
}

fn tiny_safetensors_bf16(name: &str, shape: &[usize], values: &[f32]) -> Vec<u8> {
    let mut data = Vec::with_capacity(values.len() * 2);
    for value in values {
        data.extend_from_slice(&f32_to_bf16_bits(*value).to_le_bytes());
    }
    tiny_safetensors(name, "BF16", shape, &data)
}

fn tiny_safetensors(name: &str, dtype: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let header = serde_json::json!({
        name: {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [0, data.len()]
        }
    })
    .to_string();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(data);
    bytes
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} expected {expected}"
        );
    }
}
