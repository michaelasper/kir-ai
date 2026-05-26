use llm_backend::native::{
    CpuNativeMatvecBackend, InferenceScratchpad, MathError, NativeMatvecBackend, QwenLayerCache,
    QwenMoeDims, QwenMoeRouterProbe, SafeTensorArchive, SafeTensorFile, SafeTensorHeader,
    SafeTensorShardStore, TensorLoadError, TopKWeight, qwen_decode_token_with_cache,
    qwen_layer_caches_for_spec, qwen_layer_moe_forward, qwen_layer0_moe_router,
    qwen_prefill_sequence_with_cache,
};
use llm_models::{AttentionKind, ModelFamily, QwenModelSpec};
use llm_test_support::safetensors::{
    TinySafetensorsSnapshot, bf16_bits, tiny_safetensors_bf16, tiny_safetensors_f32,
};
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
    rms_norm_calls: AtomicUsize,
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

    fn rms_norm_calls(&self) -> usize {
        self.rms_norm_calls.load(Ordering::Relaxed)
    }

    fn softmax_top_k_calls(&self) -> usize {
        self.softmax_top_k_calls.load(Ordering::Relaxed)
    }
}

#[test]
fn safetensors_archive_loads_metadata_and_f32_tensor() {
    let bytes = tiny_safetensors_f32("linear.weight", &[2, 2], &[1.0, 2.0, 3.0, 4.0]);

    let archive = SafeTensorArchive::from_bytes(&bytes).expect("archive loads");
    let metadata = archive
        .tensor_metadata("linear.weight")
        .expect("metadata loads");
    let values = archive
        .f32_tensor("linear.weight")
        .expect("f32 tensor loads");

    assert_eq!(metadata.shape, vec![2, 2]);
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn safetensors_header_rejects_non_integer_shape_dimension() {
    let mut bytes = Vec::new();
    let header = serde_json::json!({
        "linear.weight": {
            "dtype": "BF16",
            "shape": [2, "2"],
            "data_offsets": [0, 4]
        }
    })
    .to_string();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0_u8; 4]);

    let err = SafeTensorHeader::from_bytes(&bytes).expect_err("shape validation fails");

    assert_eq!(
        err.message(),
        "tensor `linear.weight` shape must contain integers"
    );
}

#[test]
fn safetensors_f32_range_cached_decodes_without_resident_cache() {
    let root = temp_snapshot_dir("f32-cache-trace");
    TinySafetensorsSnapshot::new()
        .with_bf16_tensor(
            "model-00001-of-00001.safetensors",
            "embed.weight",
            [2, 3],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .write(&root)
        .expect("snapshot");

    let store = SafeTensorShardStore::open(&root).expect("store opens");
    assert_eq!(store.cached_f32_count(), 0);

    let capture = TraceCapture::start();
    let first = store
        .bf16_tensor_f32_range_cached("embed.weight", 0, 6)
        .expect("first cached read");
    let second = store
        .bf16_tensor_f32_range_cached("embed.weight", 0, 6)
        .expect("second cached read");
    let events = capture.events();

    assert_eq!(first, second);
    assert_eq!(store.cached_f32_count(), 0);
    assert_eq!(store.cached_f32_bytes(), 0);
    assert!(
        events.iter().any(|event| {
            event.has_field("operation", "safetensors_f32_cache_bypass")
                && event.has_field("cache", "range")
                && event.has_field("cache_resident", "false")
                && event.has_field("tensor", "embed.weight")
        }),
        "cached range read should emit F32 cache bypass metadata, got {events:?}"
    );
    assert!(
        events
            .iter()
            .filter(|event| {
                event.has_field("operation", "safetensors_f32_cache_bypass")
                    && event.has_field("cache", "range")
                    && event.has_field("tensor", "embed.weight")
            })
            .count()
            >= 2,
        "each cached range read should bypass permanent F32 cache, got {events:?}"
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn safetensors_full_f32_cached_arc_returns_transient_allocations() {
    let root = temp_snapshot_dir("f32-cache-arc-transient");
    TinySafetensorsSnapshot::new()
        .with_bf16_tensor(
            "model-00001-of-00001.safetensors",
            "norm.weight",
            [2],
            [3.0, 4.0],
        )
        .write(&root)
        .expect("snapshot");

    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let first = store
        .bf16_tensor_f32_cached_arc("norm.weight")
        .expect("first full cached arc");
    let second = store
        .bf16_tensor_f32_cached_arc("norm.weight")
        .expect("second full cached arc");

    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(first.as_ref(), &[3.0, 4.0]);
    assert_eq!(second.as_ref(), &[3.0, 4.0]);
    assert_eq!(store.cached_f32_count(), 0);
    assert_eq!(store.cached_f32_bytes(), 0);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn safetensors_bf16_range_into_reuses_caller_decode_buffer() {
    let root = temp_snapshot_dir("bf16-range-into-buffer");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    let shard_path = root.join("model.safetensors");
    std::fs::write(
        &shard_path,
        tiny_safetensors_bf16("embed.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .expect("shard");
    let file = llm_backend::native::SafeTensorFile::open(&shard_path).expect("open shard");
    file.materialize().expect("materialized shard");

    let mut values = Vec::with_capacity(8);
    values.extend_from_slice(&[99.0, 100.0]);
    let original_ptr = values.as_ptr();

    file.bf16_tensor_f32_range_into("embed.weight", 1, 4, &mut values)
        .expect("range decodes into caller buffer");

    assert_eq!(values, vec![2.0, 3.0, 4.0, 5.0]);
    assert_eq!(values.as_ptr(), original_ptr);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn safetensors_with_tensor_bytes_range_borrows_materialized_mmap_range() {
    let root = temp_snapshot_dir("borrow-materialized-range");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    let shard_path = root.join("model.safetensors");
    std::fs::write(
        &shard_path,
        tiny_safetensors_bf16("embed.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .expect("shard");
    let file = SafeTensorFile::open(&shard_path).expect("open shard");
    file.materialize().expect("materialized shard");

    let first_ptr = file
        .with_tensor_bytes_range("embed.weight", 2, 4, |bytes| {
            let expected = tiny_safetensors_bf16_values(&[2.0, 3.0]);
            assert_eq!(bytes, expected.as_slice());
            Ok(bytes.as_ptr() as usize)
        })
        .expect("first borrowed range");
    let second_ptr = file
        .with_tensor_bytes_range("embed.weight", 2, 4, |bytes| Ok(bytes.as_ptr() as usize))
        .expect("second borrowed range");

    assert_eq!(first_ptr, second_ptr);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn safetensors_bf16_row_into_reuses_caller_decode_buffer() {
    let root = temp_snapshot_dir("bf16-row-into-buffer");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    let shard_path = root.join("model.safetensors");
    std::fs::write(
        &shard_path,
        tiny_safetensors_bf16("embed.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .expect("shard");
    let file = llm_backend::native::SafeTensorFile::open(&shard_path).expect("open shard");
    file.materialize().expect("materialized shard");

    let mut values = Vec::with_capacity(4);
    values.push(99.0);
    let original_ptr = values.as_ptr();

    file.bf16_row_f32_into("embed.weight", 1, &mut values)
        .expect("row decodes into caller buffer");

    assert_eq!(values, vec![4.0, 5.0, 6.0]);
    assert_eq!(values.as_ptr(), original_ptr);
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
        self.rms_norm_calls.fetch_add(1, Ordering::Relaxed);
        CpuNativeMatvecBackend
            .rms_norm_one_centered_f32_in_place(input, weight, eps, output)
            .await
    }

    async fn rms_norm_f32_in_place(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        self.rms_norm_calls.fetch_add(1, Ordering::Relaxed);
        CpuNativeMatvecBackend
            .rms_norm_f32_in_place(input, weight, eps, output)
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
    TinySafetensorsSnapshot::new()
        .with_bf16_tensor(
            "embed.safetensors",
            "model.language_model.embed_tokens.weight",
            [2, 2],
            [1.0, 0.0, 0.0, 1.0],
        )
        .with_bf16_tensor(
            "input_norm.safetensors",
            "model.language_model.layers.0.input_layernorm.weight",
            [2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "qkv.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            [4, 2],
            [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0],
        )
        .with_bf16_tensor(
            "z.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            [2, 2],
            [1.0, 0.0, 0.0, 1.0],
        )
        .with_bf16_tensor(
            "b.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "a.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "dt.safetensors",
            "model.language_model.layers.0.linear_attn.dt_bias",
            [1],
            [0.0],
        )
        .with_bf16_tensor(
            "a_log.safetensors",
            "model.language_model.layers.0.linear_attn.A_log",
            [1],
            [0.0],
        )
        .with_bf16_tensor(
            "conv.safetensors",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            [4, 1],
            [1.0, 1.0, 1.0, 1.0],
        )
        .with_bf16_tensor(
            "attn_norm.safetensors",
            "model.language_model.layers.0.linear_attn.norm.weight",
            [2],
            [1.0, 1.0],
        )
        .with_bf16_tensor(
            "out.safetensors",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            [2, 2],
            [1.0, 0.0, 0.0, 1.0],
        )
        .with_bf16_tensor(
            "post_norm.safetensors",
            "model.language_model.layers.0.post_attention_layernorm.weight",
            [2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "router.safetensors",
            "model.language_model.layers.0.mlp.gate.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "experts_gate_up.safetensors",
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            [2, 2],
            [0.0, 0.0, 0.0, 0.0],
        )
        .with_bf16_tensor(
            "experts_down.safetensors",
            "model.language_model.layers.0.mlp.experts.down_proj",
            [2, 1],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "shared_gate.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "shared_up.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "shared_down.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            [2, 1],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "shared_expert_gate.safetensors",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .write(root)
        .expect("snapshot");
}

fn write_tiny_qwen3_dense_decoder_snapshot(root: &std::path::Path) {
    TinySafetensorsSnapshot::new()
        .with_bf16_tensor(
            "embed.safetensors",
            "model.embed_tokens.weight",
            [2, 2],
            [1.0, 0.0, 0.0, 1.0],
        )
        .with_bf16_tensor(
            "input_norm.safetensors",
            "model.layers.0.input_layernorm.weight",
            [2],
            [1.0, 1.0],
        )
        .with_bf16_tensor(
            "q.safetensors",
            "model.layers.0.self_attn.q_proj.weight",
            [2, 2],
            [1.0, 0.0, 0.0, 1.0],
        )
        .with_bf16_tensor(
            "k.safetensors",
            "model.layers.0.self_attn.k_proj.weight",
            [2, 2],
            [1.0, 0.0, 0.0, 1.0],
        )
        .with_bf16_tensor(
            "v.safetensors",
            "model.layers.0.self_attn.v_proj.weight",
            [2, 2],
            [2.0, 0.0, 0.0, 4.0],
        )
        .with_bf16_tensor(
            "q_norm.safetensors",
            "model.layers.0.self_attn.q_norm.weight",
            [2],
            [1.0, 1.0],
        )
        .with_bf16_tensor(
            "k_norm.safetensors",
            "model.layers.0.self_attn.k_norm.weight",
            [2],
            [1.0, 1.0],
        )
        .with_bf16_tensor(
            "o.safetensors",
            "model.layers.0.self_attn.o_proj.weight",
            [2, 2],
            [1.0, 0.0, 0.0, 1.0],
        )
        .with_bf16_tensor(
            "post_norm.safetensors",
            "model.layers.0.post_attention_layernorm.weight",
            [2],
            [1.0, 1.0],
        )
        .with_bf16_tensor(
            "gate.safetensors",
            "model.layers.0.mlp.gate_proj.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "up.safetensors",
            "model.layers.0.mlp.up_proj.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "down.safetensors",
            "model.layers.0.mlp.down_proj.weight",
            [2, 1],
            [0.0, 0.0],
        )
        .write(root)
        .expect("snapshot");
}

fn write_tiny_moe_forward_snapshot(root: &std::path::Path) {
    TinySafetensorsSnapshot::new()
        .with_bf16_tensor(
            "router.safetensors",
            "model.language_model.layers.0.mlp.gate.weight",
            [3, 2],
            [1.0, 0.0, 0.0, 2.0, 1.0, 1.0],
        )
        .with_bf16_tensor(
            "gate_up.safetensors",
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            [2, 2, 2],
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        )
        .with_bf16_tensor(
            "down.safetensors",
            "model.language_model.layers.0.mlp.experts.down_proj",
            [2, 2, 1],
            [1.0, 2.0, 3.0, 4.0],
        )
        .with_bf16_tensor(
            "shared_gate.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "shared_up.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "shared_down.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            [2, 1],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "shared_expert_gate.safetensors",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .write(root)
        .expect("snapshot");
}

fn temp_snapshot_dir(label: &str) -> std::path::PathBuf {
    llm_test_support::safetensors::temp_snapshot_dir("llm-backend", label)
}

fn tiny_safetensors_bf16_values(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&bf16_bits(*value).to_le_bytes());
    }
    bytes
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
        sliding_window: None,
        vocab_size: 2,
        layer_kinds: vec![kind],
    }
}

fn tiny_qwen3_dense_spec() -> QwenModelSpec {
    QwenModelSpec {
        family: ModelFamily::Qwen,
        architecture: "Qwen3ForCausalLM".to_owned(),
        model_type: "qwen3".to_owned(),
        text_model_type: "qwen3".to_owned(),
        hidden_size: 2,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        rope_theta: 10_000.0,
        partial_rotary_factor: 1.0,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        linear_num_key_heads: 0,
        linear_num_value_heads: 0,
        linear_key_head_dim: 0,
        linear_value_head_dim: 0,
        linear_conv_kernel_dim: 0,
        num_experts: 0,
        num_experts_per_tok: 0,
        moe_intermediate_size: 1,
        shared_expert_intermediate_size: 0,
        max_position_embeddings: 128,
        sliding_window: None,
        vocab_size: 2,
        layer_kinds: vec![AttentionKind::FullAttention],
    }
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
