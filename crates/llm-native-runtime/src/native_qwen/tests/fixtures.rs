fn copy_fixture(name: &str, destination: impl AsRef<Path>) {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/qwen36")
        .join(name);
    let destination = destination.as_ref();
    std::fs::copy(&source, destination).expect("copy fixture");
    if name == "model.safetensors.index.json"
        && let Some(root) = destination.parent()
    {
        write_qwen36_static_f32_fixture_shards(root);
    }
}

fn write_qwen36_static_f32_fixture_shards(root: &Path) {
    let config_json = std::fs::read_to_string(root.join("config.json")).expect("Qwen config");
    let spec = QwenModelSpec::from_config_json(&config_json).expect("Qwen spec");
    let index_json =
        std::fs::read_to_string(root.join("model.safetensors.index.json")).expect("Qwen index");
    let index = llm_models::SafetensorsIndex::from_json(&index_json).expect("Qwen index parses");
    let mut shards: TinyBf16ShardMap = std::collections::BTreeMap::new();
    for tensor in qwen_static_f32_tensors_for_spec(&spec) {
        let Some(shard) = index.shard_for(&tensor) else {
            continue;
        };
        let shape = qwen_static_f32_tensor_shape(&spec, &tensor);
        let element_count = shape.iter().product();
        shards
            .entry(shard.to_owned())
            .or_default()
            .push((tensor, shape, vec![0.0; element_count]));
    }
    for (shard, tensors) in shards {
        let path = root.join(shard);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("Qwen shard parent");
        }
        std::fs::write(path, tiny_owned_multi_safetensors_bf16(&tensors))
            .expect("Qwen static f32 fixture shard");
    }
}

fn qwen_static_f32_tensor_shape(spec: &QwenModelSpec, tensor: &str) -> Vec<usize> {
    if tensor == spec.final_norm_weight()
        || tensor.ends_with("input_layernorm.weight")
        || tensor.ends_with("post_attention_layernorm.weight")
    {
        return vec![spec.hidden_size as usize];
    }
    if tensor.ends_with("self_attn.q_norm.weight") || tensor.ends_with("self_attn.k_norm.weight") {
        return vec![spec.head_dim as usize];
    }
    if tensor.ends_with("linear_attn.dt_bias") || tensor.ends_with("linear_attn.A_log") {
        return vec![spec.linear_num_value_heads as usize];
    }
    if tensor.ends_with("linear_attn.norm.weight") {
        return vec![spec.linear_value_head_dim as usize];
    }
    if tensor.ends_with("linear_attn.conv1d.weight") {
        let key_dim = (spec.linear_num_key_heads as usize) * (spec.linear_key_head_dim as usize);
        let value_dim =
            (spec.linear_num_value_heads as usize) * (spec.linear_value_head_dim as usize);
        return vec![
            key_dim * 2 + value_dim,
            spec.linear_conv_kernel_dim as usize,
        ];
    }
    panic!("unknown Qwen static f32 tensor `{tensor}`");
}

fn write_tiny_qwen3_dense_single_file_decoder_snapshot(root: &Path) {
    std::fs::write(
        root.join("config.json"),
        serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "attention_bias": false,
            "hidden_act": "silu",
            "hidden_size": 2,
            "intermediate_size": 1,
            "max_position_embeddings": 16,
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "rms_norm_eps": 1e-6,
            "rope_scaling": null,
            "rope_theta": 1_000_000,
            "sliding_window": null,
            "tie_word_embeddings": true,
            "use_sliding_window": false,
            "vocab_size": 2
        })
        .to_string(),
    )
    .expect("config");
    std::fs::write(
        root.join("model.safetensors"),
        tiny_multi_safetensors_bf16(&[
            ("model.embed_tokens.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
            ("model.norm.weight", &[2], &[1.0, 1.0]),
            ("model.layers.0.input_layernorm.weight", &[2], &[1.0, 1.0]),
            (
                "model.layers.0.self_attn.q_proj.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            ("model.layers.0.self_attn.q_norm.weight", &[2], &[1.0, 1.0]),
            ("model.layers.0.self_attn.k_norm.weight", &[2], &[1.0, 1.0]),
            (
                "model.layers.0.self_attn.o_proj.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                &[2],
                &[1.0, 1.0],
            ),
            ("model.layers.0.mlp.gate_proj.weight", &[1, 2], &[0.0, 0.0]),
            ("model.layers.0.mlp.up_proj.weight", &[1, 2], &[0.0, 0.0]),
            ("model.layers.0.mlp.down_proj.weight", &[2, 1], &[0.0, 0.0]),
        ]),
    )
    .expect("single safetensors");
}

fn write_tiny_qwen3_dense_model_index(root: &Path) {
    let weight_map = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
    .into_iter()
    .map(|tensor| (tensor, "model.safetensors"));
    std::fs::write(
        root.join("model.safetensors.index.json"),
        tiny_safetensors_index_json(1, weight_map),
    )
    .expect("tiny Qwen index");
}

fn write_tiny_linear_decoder_snapshot(root: &Path) {
    TinySafetensorsSnapshot::new()
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.embed_tokens.weight",
            [2, 2],
            [1.0, 0.0, 0.0, 1.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.input_layernorm.weight",
            [2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            [4, 2],
            [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            [2, 2],
            [1.0, 0.0, 0.0, 1.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.linear_attn.dt_bias",
            [1],
            [0.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.linear_attn.A_log",
            [1],
            [0.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            [4, 1],
            [1.0, 1.0, 1.0, 1.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.linear_attn.norm.weight",
            [2],
            [1.0, 1.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            [2, 2],
            [1.0, 0.0, 0.0, 1.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.post_attention_layernorm.weight",
            [2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.mlp.gate.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            [2, 2],
            [0.0, 0.0, 0.0, 0.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.mlp.experts.down_proj",
            [2, 1],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            [2, 1],
            [0.0, 0.0],
        )
        .with_bf16_tensor(
            "model.safetensors",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            [1, 2],
            [0.0, 0.0],
        )
        .write(root)
        .expect("write tiny decoder snapshot");
}

fn zero_layer_qwen_spec(hidden_size: u32, vocab_size: u32) -> QwenModelSpec {
    QwenModelSpec {
        family: llm_models::ModelFamily::Qwen,
        architecture: "Qwen3_5MoeForConditionalGeneration".to_owned(),
        model_type: "qwen3_5_moe".to_owned(),
        text_model_type: "qwen3_5_moe_text".to_owned(),
        hidden_size,
        rms_norm_eps: 0.0,
        tie_word_embeddings: false,
        rope_theta: 1_000_000.0,
        partial_rotary_factor: 1.0,
        num_hidden_layers: 0,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: hidden_size,
        linear_num_key_heads: 1,
        linear_num_value_heads: 1,
        linear_key_head_dim: 1,
        linear_value_head_dim: hidden_size,
        linear_conv_kernel_dim: 1,
        num_experts: 1,
        num_experts_per_tok: 1,
        moe_intermediate_size: 1,
        shared_expert_intermediate_size: 1,
        max_position_embeddings: 1,
        vocab_size,
        layer_kinds: Vec::new(),
    }
}

fn tiny_engine_qwen_spec(kind: llm_models::AttentionKind) -> QwenModelSpec {
    QwenModelSpec {
        family: llm_models::ModelFamily::Qwen,
        architecture: "Qwen3_5MoeForConditionalGeneration".to_owned(),
        model_type: "qwen3_5_moe".to_owned(),
        text_model_type: "qwen3_5_moe_text".to_owned(),
        hidden_size: 2,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: false,
        rope_theta: 1_000_000.0,
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
        max_position_embeddings: 32,
        vocab_size: 2,
        layer_kinds: vec![kind],
    }
}

fn native_qwen_test_backend(
    snapshot: &Path,
    model_id: &str,
    spec: QwenModelSpec,
    max_new_tokens: u32,
    max_prefill_tokens: usize,
    top_k: usize,
    chunk_rows: usize,
) -> NativeQwenBackend {
    let metadata = BackendModelMetadata::new(model_id.to_owned(), "native-qwen");
    let tokenizer =
        HuggingFaceTokenizer::from_file(snapshot.join("tokenizer.json")).expect("tokenizer loads");
    let adapter = NativeQwenAdapter {
        model_id: model_id.to_owned(),
        metadata: metadata.clone(),
        spec,
        store: SafeTensorShardStore::open(snapshot).expect("store opens"),
        matvec: NativeTextMatvecBackend::Cpu,
        max_prefill_tokens,
        top_k,
        chunk_rows,
        prefix_cache: Arc::new(NativeQwenPrefixCache::new(
            DEFAULT_NATIVE_QWEN_PREFIX_CACHE_BYTES,
        )),
        prefix_disk_cache: None,
    };
    NativeQwenBackend {
        driver: NativeTextDriver::new(
            model_id.to_owned(),
            metadata,
            tokenizer,
            adapter,
            max_new_tokens,
        ),
    }
}

fn native_qwen_test_request(model: &str) -> BackendRequest {
    BackendRequest::raw_completion(model, "test", Some(1), SamplingConfig::Greedy)
}
fn native_qwen_test_prefix_namespace(label: &str) -> NativeQwenPrefixCacheNamespace {
    NativeQwenPrefixCacheNamespace {
        model_id: format!("model-{label}"),
        backend: "native-qwen".to_owned(),
        family: Some("qwen".to_owned()),
        quantization: Some("bf16".to_owned()),
        repo_id: Some("local/test".to_owned()),
        resolved_commit: Some("0123456789abcdef0123456789abcdef01234567".to_owned()),
        profile: Some("qwen-test".to_owned()),
        tokenizer_kind: "huggingface-tokenizer-json".to_owned(),
        tokenizer_hash: format!("sha256:qwen-test-tokenizer-{label}"),
        tokenizer_normalization: "llm-tokenizer/hf-json/v1".to_owned(),
        cache_template_id: QwenFamilyAdapter.cache_template_id().to_owned(),
        chat_template_kwargs_hash: Some(
            "sha256:09f707b4df24814500e39b767df141317a1b87a1378d75246164c5a77adce367".to_owned(),
        ),
        adapter_settings: super::NATIVE_QWEN_PREFIX_ADAPTER_SETTINGS.to_owned(),
        cache_key: BackendCacheContext::chat_template_with_kwargs(
            QwenFamilyAdapter.cache_template_id(),
            Some("tool-schema-v1".to_owned()),
            QwenFamilyAdapter
                .chat_template_kwargs_json()
                .map(str::to_owned),
        )
        .key
        .as_str()
        .to_owned(),
        tool_schema: Some("tool-schema-v1".to_owned()),
        request_mode: "chat,json_object=false,required_tool=None".to_owned(),
        cache_layout_version: NATIVE_QWEN_PREFIX_CACHE_LAYOUT_VERSION,
        cache_tokens: 8,
        max_prefill_tokens: 8,
    }
}

fn native_prefix_metric_counter(name: &str) -> u64 {
    native_qwen_prefix_cache_metrics().snapshot()[name]
        .as_u64()
        .unwrap_or_else(|| panic!("prefix metric `{name}` is an unsigned integer"))
}

fn assert_close_vec(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "value {index} differed: actual={actual}, expected={expected}"
        );
    }
}

struct CancelAfterFirstConv {
    cancellation: CancellationToken,
    conv_calls: std::cell::Cell<usize>,
}

impl NativeMatvecBackend for CancelAfterFirstConv {
    async fn bf16_matvec_row_major_f32_in_place(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), TensorLoadError> {
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

    async fn matvec_row_major_f32_in_place(
        &self,
        input: &[f32],
        weights: &[f32],
        rows: usize,
        columns: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
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
        self.conv_calls.set(self.conv_calls.get() + 1);
        if self.conv_calls.get() == 1 {
            self.cancellation.cancel();
        }
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

    #[allow(clippy::too_many_arguments)]
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
}
fn temp_snapshot_dir(label: &str) -> PathBuf {
    llm_test_support::safetensors::temp_snapshot_dir("llm-engine", label)
}
